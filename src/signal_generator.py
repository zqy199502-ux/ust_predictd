"""
信号生成与久期择时模块
"""

import pandas as pd
import numpy as np
from scipy import stats

from .config import CONFIG, ConfigManager

class SingleFactorSignal:
    def __init__(self, config: ConfigManager = None):
        self.config = config or CONFIG
    
    def trend_signal(self, series: pd.Series, short_window: int = 3, long_window: int = 12) -> pd.Series:
        short_ma = series.ewm(span=short_window, adjust=False).mean()
        long_ma = series.ewm(span=long_window, adjust=False).mean()
        deviation = (short_ma - long_ma) / long_ma.abs().replace(0, np.nan)
        signal = np.tanh(deviation * 10)
        signal.name = f"{series.name}_trend"
        return signal
    
    def momentum_signal(self, series: pd.Series, windows: list = [1, 3, 6, 12]) -> pd.Series:
        momentum_score = pd.Series(0, index=series.index)
        weights = [0.4, 0.3, 0.15, 0.15]
        for i, window in enumerate(windows):
            if i < len(weights):
                change = series.pct_change(window)
                change_std = change.rolling(36, min_periods=12).std().replace(0, np.nan)
                change_z = change / change_std
                momentum_score += np.tanh(change_z) * weights[i]
        momentum_score.name = f"{series.name}_momentum"
        return momentum_score
    
    def zscore_signal(self, series: pd.Series, window: int = 36) -> pd.Series:
        rolling_mean = series.rolling(window=window, min_periods=window//2).mean()
        rolling_std = series.rolling(window=window, min_periods=window//2).std().replace(0, np.nan)
        zscore = (series - rolling_mean) / rolling_std
        signal = np.tanh(zscore * 0.5)
        signal.name = f"{series.name}_zscore"
        return signal
    
    def generate_all_signals(self, df: pd.DataFrame) -> dict:
        signals = {'trend': pd.DataFrame(index=df.index), 'momentum': pd.DataFrame(index=df.index), 'zscore': pd.DataFrame(index=df.index)}
        for col in df.columns:
            signals['trend'][f"{col}_trend"] = self.trend_signal(df[col])
            signals['momentum'][f"{col}_mom"] = self.momentum_signal(df[col])
            signals['zscore'][f"{col}_zsc"] = self.zscore_signal(df[col])
        return signals

class SignalSynthesizer:
    def __init__(self, config: ConfigManager = None):
        self.config = config or CONFIG
    
    def simple_average(self, signals_df: pd.DataFrame) -> pd.Series:
        return signals_df.mean(axis=1)
    
    def synthesize(self, signals_dict: dict, method: str = 'simple', forward_returns: pd.Series = None) -> pd.Series:
        if method == 'simple':
            all_signals = pd.concat(signals_dict.values(), axis=1)
            synthesized = self.simple_average(all_signals)
        else:
            all_signals = pd.concat(signals_dict.values(), axis=1)
            synthesized = self.simple_average(all_signals)
        
        smoothing = self.config.signal.signal_smoothing
        if smoothing > 1:
            synthesized = synthesized.ewm(span=smoothing, adjust=False).mean()
        
        synthesized = np.tanh(synthesized)
        synthesized.name = 'composite_signal'
        return synthesized

class DurationTimer:
    def __init__(self, config: ConfigManager = None):
        self.config = config or CONFIG
    
    def signal_to_duration(self, signal: pd.Series, method: str = 'linear') -> pd.Series:
        min_dur = self.config.backtest.duration_target_min
        max_dur = self.config.backtest.duration_target_max
        neutral = self.config.backtest.duration_neutral
        
        if method == 'linear':
            duration = neutral + signal * (max_dur - neutral)
            duration = duration.clip(min_dur, max_dur)
        else:
            duration = pd.Series(neutral, index=signal.index)
        
        duration.name = 'target_duration'
        return duration
    
    def run_duration_timing(self, composite_signal: pd.Series, yield_data: pd.Series, method: str = 'linear') -> pd.DataFrame:
        result = pd.DataFrame(index=composite_signal.index)
        result['signal'] = composite_signal
        result['target_duration'] = self.signal_to_duration(composite_signal, method)
        result['duration'] = result['target_duration']
        result['turnover'] = result['duration'].diff().abs()
        return result

class BacktestEngine:
    def __init__(self, config: ConfigManager = None):
        self.config = config or CONFIG
    
    def calculate_strategy_returns(self, duration_df: pd.DataFrame, yield_changes: pd.Series) -> pd.DataFrame:
        result = duration_df.copy()
        bond_returns = -result['duration'].shift(1) * yield_changes
        tc = result['turnover'] * self.config.backtest.transaction_cost
        result['bond_return'] = bond_returns
        result['transaction_cost'] = tc
        result['net_return'] = bond_returns - tc
        result['cumulative_return'] = (1 + result['net_return']).cumprod()
        return result
    
    def calculate_metrics(self, strategy_returns: pd.Series, benchmark_returns: pd.Series = None) -> dict:
        returns = strategy_returns.dropna()
        if len(returns) < 12:
            return {}
        
        annual_return = returns.mean() * 12
        annual_vol = returns.std() * np.sqrt(12)
        sharpe = (annual_return - 0.02) / annual_vol if annual_vol > 0 else 0
        
        cumulative = (1 + returns).cumprod()
        rolling_max = cumulative.expanding().max()
        drawdown = (cumulative - rolling_max) / rolling_max
        max_drawdown = drawdown.min()
        
        metrics = {
            'annual_return': annual_return,
            'annual_volatility': annual_vol,
            'sharpe_ratio': sharpe,
            'max_drawdown': max_drawdown,
            'calmar_ratio': annual_return / abs(max_drawdown) if max_drawdown != 0 else 0,
            'win_rate': (returns > 0).mean()
        }
        return metrics
    
    def run_full_backtest(self, duration_df: pd.DataFrame, yield_data: pd.Series, benchmark_duration: float = 5.0) -> dict:
        yield_changes = yield_data.pct_change().dropna()
        strategy = self.calculate_strategy_returns(duration_df, yield_changes)
        benchmark = -benchmark_duration * yield_changes
        metrics = self.calculate_metrics(strategy['net_return'], benchmark)
        return {'returns': strategy, 'benchmark': benchmark, 'metrics': metrics}

def generate_duration_signal(macro_data: pd.DataFrame, yield_changes: pd.Series, method: str = 'simple') -> pd.Series:
    signal_gen = SingleFactorSignal()
    signals = signal_gen.generate_all_signals(macro_data)
    synthesizer = SignalSynthesizer()
    composite = synthesizer.synthesize(signals, method=method, forward_returns=yield_changes)
    return composite