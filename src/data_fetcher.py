"""
高频宏观因子数据获取模块
"""

import os
import time
import pickle
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import hashlib

import pandas as pd
import numpy as np
from fredapi import Fred
import requests

from .config import CONFIG, ConfigManager

class FREDDataFetcher:
    def __init__(self, config: Optional[ConfigManager] = None, cache_dir: str = "./data/cache"):
        self.config = config or CONFIG
        self.fred_config = self.config.fred
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._init_fred_client()
        self._last_request_time = 0
    
    def _init_fred_client(self):
        try:
            self.fred = Fred(api_key=self.fred_config.api_key)
            self.fred.get_series('DGS10', limit=1)
        except Exception as e:
            self.fred = None
    
    def _get_cache_path(self, series_id: str, start_date: str, end_date: str, frequency: Optional[str] = None) -> Path:
        freq_str = f"_{frequency}" if frequency else ""
        cache_key = f"{series_id}_{start_date}_{end_date}{freq_str}"
        cache_hash = hashlib.md5(cache_key.encode()).hexdigest()[:12]
        return self.cache_dir / f"{series_id}_{cache_hash}.pkl"
    
    def _load_from_cache(self, cache_path: Path, max_age_days: int = 7) -> Optional[pd.Series]:
        if not cache_path.exists():
            return None
        cache_age = datetime.now() - datetime.fromtimestamp(cache_path.stat().st_mtime)
        if cache_age.days > max_age_days:
            return None
        try:
            with open(cache_path, 'rb') as f:
                return pickle.load(f)
        except:
            return None
    
    def _save_to_cache(self, cache_path: Path, data: pd.Series):
        try:
            with open(cache_path, 'wb') as f:
                pickle.dump(data, f)
        except:
            pass
    
    def fetch_series(self, series_id: str, start_date: Optional[str] = None, end_date: Optional[str] = None,
                     frequency: Optional[str] = None, force_refresh: bool = False) -> Optional[pd.Series]:
        start_date = start_date or self.config.backtest.start_date
        end_date = end_date or self.config.backtest.end_date
        cache_path = self._get_cache_path(series_id, start_date, end_date, frequency)
        
        if not force_refresh:
            cached_data = self._load_from_cache(cache_path)
            if cached_data is not None:
                return cached_data
        
        for attempt in range(self.fred_config.max_retry):
            try:
                if self.fred:
                    data = self.fred.get_series(series_id, observation_start=start_date,
                                                observation_end=end_date, frequency=frequency)
                else:
                    params = {'series_id': series_id, 'api_key': self.fred_config.api_key,
                              'file_type': 'json', 'observation_start': start_date,
                              'observation_end': end_date}
                    if frequency:
                        params['frequency'] = frequency
                    url = f"{self.fred_config.base_url}/series/observations?{requests.compat.urlencode(params)}"
                    response = requests.get(url, timeout=self.fred_config.request_timeout)
                    response.raise_for_status()
                    data = response.json()['observations']
                    dates = [pd.to_datetime(obs['date']) for obs in data]
                    values = [float(obs['value']) if obs['value'] != '.' else np.nan for obs in data]
                    data = pd.Series(values, index=dates, name=series_id).dropna()
                
                if isinstance(data, pd.DataFrame):
                    data = data.iloc[:, 0]
                data = data.dropna()
                data.name = series_id
                self._save_to_cache(cache_path, data)
                return data
            except Exception as e:
                if attempt < self.fred_config.max_retry - 1:
                    time.sleep(self.fred_config.retry_delay)
        return None
    
    def fetch_multiple(self, series_dict: Dict[str, str], start_date: Optional[str] = None,
                       end_date: Optional[str] = None, target_frequency: str = 'm',
                       resample_method: str = 'last', force_refresh: bool = False) -> pd.DataFrame:
        all_data = {}
        for series_id, desc in series_dict.items():
            series = self.fetch_series(series_id, start_date, end_date, frequency=None, force_refresh=force_refresh)
            if series is not None:
                all_data[series_id] = series
        
        if not all_data:
            return pd.DataFrame()
        
        df = pd.DataFrame(all_data)
        df = self._resample_to_frequency(df, target_frequency, method=resample_method)
        df = self._fill_missing(df, method='ffill', limit=5)
        return df
    
    def _resample_to_frequency(self, df: pd.DataFrame, frequency: str, method: str = 'last') -> pd.DataFrame:
        freq_map = {'d': 'D', 'w': 'W-FRI', 'm': 'ME', 'q': 'QE', 'a': 'YE'}
        pandas_freq = freq_map.get(frequency, 'ME')
        
        if method == 'last':
            return df.resample(pandas_freq).last()
        elif method == 'mean':
            return df.resample(pandas_freq).mean()
        return df.resample(pandas_freq).last()
    
    def _fill_missing(self, df: pd.DataFrame, method: str = 'ffill', limit: int = 5) -> pd.DataFrame:
        if method == 'ffill':
            return df.fillna(method='ffill', limit=limit)
        return df.fillna(method='bfill', limit=limit)
    
    def get_yield_curve_data(self, start_date: Optional[str] = None, end_date: Optional[str] = None,
                             frequency: str = 'd', force_refresh: bool = False) -> pd.DataFrame:
        tenor_series = self.config.yield_curve.tenor_series
        df = self.fetch_multiple(series_dict=tenor_series, start_date=start_date,
                                 end_date=end_date, target_frequency=frequency,
                                 resample_method='last', force_refresh=force_refresh)
        
        rename_map = {v: float(k) for k, v in tenor_series.items() if v in df.columns}
        df = df.rename(columns=rename_map)
        return df.reindex(sorted(df.columns), axis=1)
    
    def get_macro_factors(self, categories: Optional[List[str]] = None, start_date: Optional[str] = None,
                          end_date: Optional[str] = None, target_frequency: str = 'm',
                          force_refresh: bool = False) -> Dict[str, pd.DataFrame]:
        category_map = self.config.factors.get_category_map()
        if categories is None:
            categories = list(category_map.keys())
        
        result = {}
        all_series = self.config.factors.get_all_series()
        
        for category in categories:
            if category not in category_map:
                continue
            series_ids = category_map[category]
            category_dict = {sid: all_series[sid] for sid in series_ids if sid in all_series}
            df = self.fetch_multiple(series_dict=category_dict, start_date=start_date,
                                     end_date=end_date, target_frequency=target_frequency,
                                     resample_method='last', force_refresh=force_refresh)
            result[category] = df
        
        return result
    
    def get_all_data(self, start_date: Optional[str] = None, end_date: Optional[str] = None,
                     target_frequency: str = 'm', force_refresh: bool = False) -> Tuple[pd.DataFrame, pd.DataFrame]:
        macro_dict = self.get_macro_factors(categories=None, start_date=start_date,
                                            end_date=end_date, target_frequency=target_frequency,
                                            force_refresh=force_refresh)
        macro_data = pd.concat(macro_dict.values(), axis=1)
        macro_data = macro_data.loc[:, ~macro_data.columns.duplicated()]
        
        yield_data = self.get_yield_curve_data(start_date=start_date, end_date=end_date,
                                               frequency='d', force_refresh=force_refresh)
        
        if target_frequency != 'd':
            yield_data = self._resample_to_frequency(yield_data, target_frequency, method='last')
            yield_data = self._fill_missing(yield_data, method='ffill', limit=5)
        
        common_start = max(macro_data.index.min(), yield_data.index.min())
        common_end = min(macro_data.index.max(), yield_data.index.max())
        
        macro_data = macro_data.loc[common_start:common_end]
        yield_data = yield_data.loc[common_start:common_end]
        
        return macro_data, yield_data

def fetch_all_data(api_key: Optional[str] = None, start_date: str = "2000-01-01",
                   end_date: str = "2024-12-31", target_frequency: str = 'm',
                   cache_dir: str = "./data/cache") -> Tuple[pd.DataFrame, pd.DataFrame]:
    if api_key:
        os.environ["FRED_API_KEY"] = api_key
    config = ConfigManager()
    fetcher = FREDDataFetcher(config=config, cache_dir=cache_dir)
    return fetcher.get_all_data(start_date=start_date, end_date=end_date, target_frequency=target_frequency)