"""
Shiny应用专用工具函数
简化版的数据获取、特征工程和预测逻辑
"""
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from fredapi import Fred
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

class ShinyYieldPredictor:
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.fred = Fred(api_key=api_key)
        self.scaler = StandardScaler()
        self.model = Ridge(alpha=1.0, random_state=42)
        self.macro_data = None
        self.yield_data = None
        self.model_trained = False
        self.last_train_date = None
        
        self.key_macro_series = {
            'GDP': 'GDPC1',
            'CPI': 'CPIAUCSL',
            'Unemployment': 'UNRATE',
            'FedFunds': 'DFF',
            'InflationExp': 'T10YIE',
            'VIX': 'VIXCLS',
            'DollarIndex': 'DTWEXBGS',
            'IndProduction': 'INDPRO'
        }
        
        self.yield_series = {
            '3M': 'DGS3MO',
            '2Y': 'DGS2',
            '5Y': 'DGS5',
            '10Y': 'DGS10',
            '30Y': 'DGS30'
        }
    
    def test_api_key(self) -> bool:
        try:
            self.fred.get_series('DGS10', limit=1)
            return True
        except Exception:
            return False
    
    def fetch_data(self, start_date: str = '2010-01-01'):
        end_date = datetime.now().strftime('%Y-%m-%d')
        
        macro_data = {}
        for name, series_id in self.key_macro_series.items():
            try:
                data = self.fred.get_series(series_id, start_date, end_date)
                macro_data[name] = data
            except Exception as e:
                print(f"Warning: Could not fetch {series_id}: {e}")
        
        self.macro_data = pd.DataFrame(macro_data)
        
        yield_data = {}
        for name, series_id in self.yield_series.items():
            try:
                data = self.fred.get_series(series_id, start_date, end_date)
                yield_data[name] = data
            except Exception as e:
                print(f"Warning: Could not fetch {series_id}: {e}")
        
        self.yield_data = pd.DataFrame(yield_data)
        
        self.macro_data = self.macro_data.fillna(method='ffill').fillna(method='bfill')
        self.yield_data = self.yield_data.fillna(method='ffill').fillna(method='bfill')
        
        return self.macro_data, self.yield_data
    
    def create_features(self, macro_data: pd.DataFrame) -> pd.DataFrame:
        features = macro_data.copy()
        
        for col in features.columns:
            features[f'{col}_lag1'] = features[col].shift(21)
            features[f'{col}_lag3'] = features[col].shift(63)
            features[f'{col}_lag6'] = features[col].shift(126)
        
        for col in macro_data.columns:
            features[f'{col}_mom'] = features[col].pct_change(21)
            features[f'{col}_yoy'] = features[col].pct_change(252)
        
        features = features.dropna()
        
        return features
    
    def train_model(self, macro_data: pd.DataFrame, yield_data: pd.DataFrame):
        if macro_data is None or yield_data is None or len(macro_data) < 100:
            raise ValueError("Insufficient data to train model")
        
        common_idx = macro_data.index.intersection(yield_data.index)
        macro_aligned = macro_data.loc[common_idx]
        yield_aligned = yield_data.loc[common_idx]
        
        features = self.create_features(macro_aligned)
        
        target = yield_aligned['10Y'].shift(-30) - yield_aligned['10Y']
        target = target.loc[features.index].dropna()
        
        features = features.loc[target.index]
        
        X_scaled = self.scaler.fit_transform(features)
        
        self.model.fit(X_scaled, target.values)
        
        self.model_trained = True
        self.last_train_date = datetime.now()
        
        self.last_features = features.iloc[-1:].copy()
        self.last_yield = yield_aligned['10Y'].iloc[-1]
        
        return {
            'last_yield': self.last_yield,
            'last_date': yield_aligned.index[-1],
            'n_samples': len(features)
        }
    
    def predict(self, horizon_days: int = 30) -> dict:
        if not self.model_trained:
            raise ValueError("Model not trained yet")
        
        X_scaled = self.scaler.transform(self.last_features)
        
        pred_change_30 = self.model.predict(X_scaled)[0]
        pred_change = pred_change_30 * (horizon_days / 30.0)
        
        pred_yield = self.last_yield + pred_change
        
        return {
            'current_yield': self.last_yield,
            'predicted_yield': pred_yield,
            'predicted_change': pred_change,
            'horizon_days': horizon_days,
            'last_date': self.last_features.index[0]
        }
    
    def plot_yield_curve(self, yield_data: pd.DataFrame) -> plt.Figure:
        tenors = [0.25, 2, 5, 10, 30]
        yields = [
            yield_data['3M'].iloc[-1],
            yield_data['2Y'].iloc[-1],
            yield_data['5Y'].iloc[-1],
            yield_data['10Y'].iloc[-1],
            yield_data['30Y'].iloc[-1]
        ]
        
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(tenors, yields, marker='o', linewidth=2, color='#1f77b4', label='Current Yield Curve')
        ax.set_xlabel('Maturity (Years)', fontsize=12)
        ax.set_ylabel('Yield (%)', fontsize=12)
        ax.set_title('U.S. Treasury Yield Curve', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend()
        plt.tight_layout()
        
        return fig
    
    def plot_history_and_prediction(self, yield_data: pd.DataFrame, prediction: dict) -> plt.Figure:
        fig, ax = plt.subplots(figsize=(12, 6))
        
        recent = yield_data['10Y'].last('730D')
        ax.plot(recent.index, recent.values, label='10Y Treasury Yield', color='#1f77b4', linewidth=2)
        
        last_date = recent.index[-1]
        pred_date = last_date + timedelta(days=prediction['horizon_days'])
        
        ax.scatter(pred_date, prediction['predicted_yield'], 
                   color='#ff7f0e', s=100, zorder=5, label=f'Prediction (T+{prediction["horizon_days"]})')
        
        ax.plot([last_date, pred_date], 
                [prediction['current_yield'], prediction['predicted_yield']],
                color='#ff7f0e', linestyle='--', alpha=0.7)
        
        ax.set_xlabel('Date', fontsize=12)
        ax.set_ylabel('Yield (%)', fontsize=12)
        ax.set_title('10-Year Treasury Yield History and Prediction', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend()
        plt.tight_layout()
        
        return fig
    
    def plot_macro_heatmap(self, macro_data: pd.DataFrame) -> plt.Figure:
        recent = macro_data.last('365D').copy()
        normalized = (recent - recent.mean()) / recent.std()
        corr = normalized.corr()
        
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(corr, annot=True, cmap='coolwarm', center=0, 
                    square=True, linewidths=1, cbar_kws={"shrink": 0.8}, ax=ax)
        ax.set_title('Macro Factor Correlation Heatmap', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        return fig