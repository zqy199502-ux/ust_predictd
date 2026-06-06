"""
利率预测模型模块
"""

import pandas as pd
import numpy as np
from scipy.optimize import minimize
from sklearn.linear_model import Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from .config import CONFIG, ConfigManager

class NelsonSiegelDecomposer:
    def __init__(self, lambda_fixed: float = None):
        self.lambda_fixed = lambda_fixed or CONFIG.yield_curve.ns_fixed_lambda
        self.beta_history = pd.DataFrame()
        self.fitted_curves = pd.DataFrame()
        self.lambda_ = self.lambda_fixed
    
    def _ns_factors(self, tau: np.ndarray, lambda_: float) -> np.ndarray:
        tau = np.asarray(tau)
        tau_safe = np.where(tau < 1e-10, 1e-10, tau)
        f1 = np.ones_like(tau)
        f2 = (1 - np.exp(-lambda_ * tau_safe)) / (lambda_ * tau_safe)
        f3 = f2 - np.exp(-lambda_ * tau_safe)
        return np.column_stack([f1, f2, f3])
    
    def fit(self, yields_df: pd.DataFrame, tenors: list = None) -> pd.DataFrame:
        if tenors is None:
            tenors = [float(c) for c in yields_df.columns]
        
        tau = np.array(tenors)
        betas_list = []
        
        for date, row in yields_df.iterrows():
            y = row.values
            valid_mask = ~np.isnan(y)
            if valid_mask.sum() < 3:
                betas_list.append([np.nan, np.nan, np.nan])
                continue
            
            tau_valid = tau[valid_mask]
            y_valid = y[valid_mask]
            
            X = self._ns_factors(tau_valid, self.lambda_)
            beta = np.linalg.lstsq(X, y_valid, rcond=None)[0]
            betas_list.append(beta)
        
        self.beta_history = pd.DataFrame(betas_list, index=yields_df.index, columns=['level', 'slope', 'curvature'])
        return self.beta_history
    
    def predict(self, beta: np.ndarray, tenors: list) -> np.ndarray:
        X = self._ns_factors(np.array(tenors), self.lambda_)
        return X @ beta
    
    def reconstruct_curve(self, beta_df: pd.DataFrame, tenors: list) -> pd.DataFrame:
        curves = []
        for date, row in beta_df.iterrows():
            if row.isna().any():
                curves.append(np.full(len(tenors), np.nan))
            else:
                curves.append(self.predict(row.values, tenors))
        return pd.DataFrame(curves, index=beta_df.index, columns=tenors)

class MacroYieldPredictor:
    def __init__(self, config: ConfigManager = None):
        self.config = config or CONFIG
        self.models = {}
        self.scalers = {}
    
    def _get_model(self, model_name: str):
        if model_name == 'ridge':
            return Ridge(alpha=1.0, random_state=42)
        elif model_name == 'lasso':
            return Lasso(alpha=0.1, random_state=42)
        elif model_name == 'random_forest':
            return RandomForestRegressor(n_estimators=200, max_depth=8, random_state=42)
        elif model_name == 'gradient_boosting':
            return GradientBoostingRegressor(n_estimators=200, learning_rate=0.05, max_depth=4, random_state=42)
        return Ridge(alpha=1.0)
    
    def fit_single_model(self, X: pd.DataFrame, y: pd.Series, target_name: str = 'target', model_name: str = 'ridge'):
        valid_idx = X.notna().all(axis=1) & y.notna()
        X_clean = X.loc[valid_idx].values
        y_clean = y.loc[valid_idx].values
        
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_clean)
        self.scalers[target_name] = scaler
        
        model = self._get_model(model_name)
        model.fit(X_scaled, y_clean)
        self.models[target_name] = model
        return model
    
    def predict(self, X: pd.DataFrame, target_name: str = 'target') -> pd.Series:
        if target_name in self.scalers:
            X_scaled = self.scalers[target_name].transform(X.values)
        else:
            X_scaled = StandardScaler().fit_transform(X.values)
        
        if target_name in self.models:
            predictions = self.models[target_name].predict(X_scaled)
        else:
            predictions = np.zeros(len(X))
        
        return pd.Series(predictions, index=X.index, name=f'{target_name}_pred')

class CurvePredictor:
    def __init__(self, config: ConfigManager = None):
        self.config = config or CONFIG
        self.ns_decomposer = NelsonSiegelDecomposer()
        self.predictor = MacroYieldPredictor()
        self.predicted_curve = None
    
    def fit(self, macro_features: pd.DataFrame, yields_df: pd.DataFrame, method: str = 'ns', model_name: str = 'ridge'):
        if method == 'ns':
            factors = self.ns_decomposer.fit(yields_df)
        else:
            factors = yields_df[[10.0]] if 10.0 in yields_df.columns else yields_df.iloc[:, -2:-1]
            factors.columns = ['target']
        
        for factor_col in factors.columns:
            self.predictor.fit_single_model(macro_features, factors[factor_col], target_name=factor_col, model_name=model_name)
    
    def predict(self, macro_features: pd.DataFrame, method: str = 'ns', tenors: list = None) -> pd.DataFrame:
        if tenors is None:
            tenors = self.config.yield_curve.tenors
        
        if method == 'ns':
            factor_names = ['level', 'slope', 'curvature']
        else:
            factor_names = ['target']
        
        predicted_factors = {}
        for factor_name in factor_names:
            pred = self.predictor.predict(macro_features, target_name=factor_name)
            predicted_factors[factor_name] = pred
        
        if method == 'ns':
            self.predicted_curve = self.ns_decomposer.reconstruct_curve(pd.DataFrame(predicted_factors), tenors)
        else:
            self.predicted_curve = pd.DataFrame(predicted_factors)
        
        return self.predicted_curve

def predict_yields(macro_features: pd.DataFrame, yields_df: pd.DataFrame, method: str = 'ns', model_name: str = 'ridge') -> CurvePredictor:
    predictor = CurvePredictor()
    predictor.fit(macro_features, yields_df, method=method, model_name=model_name)
    return predictor