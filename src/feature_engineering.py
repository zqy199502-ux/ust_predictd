"""
特征工程模块
"""

from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import mutual_info_regression

from .config import CONFIG, ConfigManager

class FactorPreprocessor:
    def __init__(self, config: Optional[ConfigManager] = None):
        self.config = config or CONFIG
    
    def remove_outliers(self, df: pd.DataFrame, method: str = 'winsorize', threshold: float = 3.0) -> pd.DataFrame:
        df = df.copy()
        for col in df.columns:
            series = df[col].dropna()
            if len(series) < 10:
                continue
            mean, std = series.mean(), series.std()
            upper, lower = mean + threshold * std, mean - threshold * std
            if method == 'winsorize':
                df[col] = df[col].clip(lower, upper)
        return df
    
    def detrend(self, df: pd.DataFrame, method: str = 'first_diff', periods: int = 1) -> pd.DataFrame:
        df = df.copy()
        if method == 'first_diff':
            df = df.diff(periods)
        elif method == 'pct_change':
            df = df.pct_change(periods)
        elif method == 'log_diff':
            df = np.log(df).diff(periods)
        return df
    
    def standardize(self, df: pd.DataFrame, method: str = 'zscore', window: Optional[int] = None) -> pd.DataFrame:
        df = df.copy()
        if window:
            if method == 'zscore':
                roll_mean = df.rolling(window=window, min_periods=window//2).mean()
                roll_std = df.rolling(window=window, min_periods=window//2).std()
                df = (df - roll_mean) / roll_std.replace(0, np.nan)
        else:
            if method == 'zscore':
                for col in df.columns:
                    series = df[col].dropna()
                    if len(series) > 0:
                        mean, std = series.mean(), series.std()
                        if std > 0:
                            df[col] = (df[col] - mean) / std
        return df

class FeatureGenerator:
    def __init__(self, config: Optional[ConfigManager] = None):
        self.config = config or CONFIG
    
    def generate_lag_features(self, df: pd.DataFrame, windows: Optional[List[int]] = None) -> pd.DataFrame:
        if windows is None:
            windows = [1, 3, 6, 12]
        features = {}
        for col in df.columns:
            for lag in windows:
                features[f"{col}_lag{lag}"] = df[col].shift(lag)
        return pd.DataFrame(features, index=df.index)
    
    def generate_momentum_features(self, df: pd.DataFrame, windows: Optional[List[int]] = None) -> pd.DataFrame:
        if windows is None:
            windows = [1, 3, 6, 12]
        features = {}
        for col in df.columns:
            for w in windows:
                features[f"{col}_chg{w}"] = df[col].diff(w)
                features[f"{col}_mom{w}"] = df[col].pct_change(w)
        return pd.DataFrame(features, index=df.index)
    
    def generate_rolling_features(self, df: pd.DataFrame, windows: Optional[List[int]] = None) -> pd.DataFrame:
        if windows is None:
            windows = [3, 6, 12]
        features = {}
        for col in df.columns:
            for window in windows:
                rolling = df[col].rolling(window=window, min_periods=window//2)
                features[f"{col}_rollmean{window}"] = rolling.mean()
                features[f"{col}_rollstd{window}"] = rolling.std()
        return pd.DataFrame(features, index=df.index)
    
    def generate_all_features(self, df: pd.DataFrame) -> pd.DataFrame:
        all_features = [df]
        all_features.append(self.generate_lag_features(df))
        all_features.append(self.generate_momentum_features(df))
        all_features.append(self.generate_rolling_features(df))
        result = pd.concat(all_features, axis=1)
        result = result.replace([np.inf, -np.inf], np.nan).dropna(axis=1, how='all')
        return result

class FeatureSelector:
    def __init__(self):
        self.selected_features = []
    
    def select_by_mutual_info(self, X: pd.DataFrame, y: pd.Series, top_k: int = 50) -> List[str]:
        X_filled = X.fillna(X.median())
        y_aligned = y.loc[X_filled.index].dropna()
        X_valid = X_filled.loc[y_aligned.index]
        
        if len(y_aligned) < 30:
            return X.columns.tolist()[:top_k]
        
        mi_scores = mutual_info_regression(X_valid, y_aligned, random_state=42)
        self.feature_scores = pd.Series(mi_scores, index=X.columns).sort_values(ascending=False)
        return self.feature_scores.head(top_k).index.tolist()
    
    def select_features(self, X: pd.DataFrame, y: pd.Series, method: str = 'mutual_info', top_k: int = 50) -> pd.DataFrame:
        if method == 'mutual_info':
            selected = self.select_by_mutual_info(X, y, top_k)
        else:
            selected = X.columns.tolist()[:top_k]
        self.selected_features = selected
        return X[selected]

class FeaturePipeline:
    def __init__(self, config: Optional[ConfigManager] = None):
        self.config = config or CONFIG
        self.preprocessor = FactorPreprocessor(config)
        self.generator = FeatureGenerator(config)
        self.selector = FeatureSelector()
    
    def transform(self, df: pd.DataFrame, y: Optional[pd.Series] = None, detrend_method: str = 'first_diff',
                  standardize_method: str = 'zscore', select_features: bool = True, top_k: int = 50) -> pd.DataFrame:
        result = df.copy()
        result = self.preprocessor.remove_outliers(result)
        result = self.preprocessor.detrend(result, method=detrend_method)
        result = self.preprocessor.standardize(result, method=standardize_method, window=36)
        result = self.generator.generate_all_features(result)
        
        if select_features and y is not None:
            result = self.selector.select_features(result, y, top_k=top_k)
        
        return result.replace([np.inf, -np.inf], np.nan).dropna(axis=1, how='all')

def build_features(df: pd.DataFrame, y: Optional[pd.Series] = None, **kwargs) -> pd.DataFrame:
    pipeline = FeaturePipeline()
    return pipeline.transform(df, y=y, **kwargs)