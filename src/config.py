"""
配置管理模块
"""

import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional
import yaml

@dataclass
class FREDConfig:
    api_key: str = os.getenv("FRED_API_KEY", "YOUR_FRED_API_KEY_HERE")
    base_url: str = "https://api.stlouisfed.org/fred"
    frequency_map: Dict[str, str] = field(default_factory=lambda: {
        "daily": "d", "weekly": "w", "monthly": "m", "quarterly": "q", "annual": "a"
    })
    max_retry: int = 3
    retry_delay: float = 2.0
    request_timeout: int = 30
    rate_limit_per_second: int = 10

@dataclass
class FactorConfig:
    growth_factors: Dict[str, str] = field(default_factory=lambda: {
        "GDPC1": "实际GDP", "PCEC1": "实际个人消费支出", "GPDIC1": "实际国内私人投资",
        "INDPRO": "工业生产指数", "CMRMTSPL": "实际制造业贸易销售", "RETAILSPC": "实际零售额",
        "USSLIND": "美国领先指标指数", "BAMLH0A0HYM2EY": "高收益债利差",
        "BAMLC0A0CMEY": "投资级债利差", "WEI": "每周经济指数", "STLFSI2": "金融压力指数",
        "NFCI": "全国金融条件指数", "ANFCI": "调整后的全国金融条件指数"
    })
    
    inflation_factors: Dict[str, str] = field(default_factory=lambda: {
        "CPIAUCSL": "CPI", "CPILFESL": "核心CPI", "PCEPI": "PCE价格指数",
        "PCEPILFE": "核心PCE", "PPIFIS": "PPI", "MICH": "密歇根通胀预期",
        "T5YIE": "5年通胀预期", "T10YIE": "10年通胀预期", "T5YIFR": "5年远期通胀率",
        "T10Y2Y": "10Y-2Y利差", "DCOILWTICO": "WTI原油价格", "CSUSHPISA": "Case-Shiller房价指数"
    })
    
    labor_factors: Dict[str, str] = field(default_factory=lambda: {
        "PAYEMS": "非农就业人数", "UNRATE": "失业率", "U6RATE": "U6不充分就业率",
        "ICSA": "初次申请失业金人数", "IC4WSA": "4周平均初请", "CCSA": "持续申请失业金人数",
        "AWHNONAG": "平均每周工时", "CES0500000003": "平均时薪", "JTSJOL": "职位空缺数"
    })
    
    monetary_factors: Dict[str, str] = field(default_factory=lambda: {
        "DFF": "联邦基金利率", "DTB3": "3月期国库券利率", "DGS2": "2年期国债收益率",
        "DGS5": "5年期国债收益率", "DGS10": "10年期国债收益率", "DGS30": "30年期国债收益率",
        "WALCL": "联储资产负债表总规模"
    })
    
    global_factors: Dict[str, str] = field(default_factory=lambda: {
        "DEXUSEU": "美元/欧元汇率", "DEXJPUS": "美元/日元汇率", "DTWEXBGS": "美元指数",
        "VIXCLS": "VIX波动率指数"
    })
    
    fiscal_factors: Dict[str, str] = field(default_factory=lambda: {
        "FYFSD": "联邦财政赤字", "GFDEBTN": "联邦债务总额"
    })
    
    def get_all_series(self) -> Dict[str, str]:
        all_series = {}
        for attr in dir(self):
            if attr.endswith('_factors') and isinstance(getattr(self, attr), dict):
                all_series.update(getattr(self, attr))
        return all_series
    
    def get_category_map(self) -> Dict[str, List[str]]:
        return {
            "growth": list(self.growth_factors.keys()),
            "inflation": list(self.inflation_factors.keys()),
            "labor": list(self.labor_factors.keys()),
            "monetary": list(self.monetary_factors.keys()),
            "global": list(self.global_factors.keys()),
            "fiscal": list(self.fiscal_factors.keys()),
        }

@dataclass
class YieldCurveConfig:
    tenors: List[float] = field(default_factory=lambda: [0.25, 0.5, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0, 20.0, 30.0])
    tenor_series: Dict[str, str] = field(default_factory=lambda: {
        "0.25": "DTB3", "0.5": "DTB6", "1.0": "DGS1", "2.0": "DGS2", "3.0": "DGS3",
        "5.0": "DGS5", "7.0": "DGS7", "10.0": "DGS10", "20.0": "DGS20", "30.0": "DGS30"
    })
    ns_fixed_lambda: Optional[float] = 0.0609
    n_pcs: int = 3

@dataclass
class ModelConfig:
    lookback_windows: List[int] = field(default_factory=lambda: [1, 3, 6, 12])
    zscore_window: int = 12
    ema_spans: List[int] = field(default_factory=lambda: [3, 6, 12])
    ensemble_models: List[str] = field(default_factory=lambda: ["ridge", "lasso", "random_forest", "gradient_boosting"])
    ridge_alphas: List[float] = field(default_factory=lambda: [0.001, 0.01, 0.1, 1.0, 10.0])
    rf_n_estimators: int = 200
    rf_max_depth: Optional[int] = 8
    rf_min_samples_leaf: int = 5
    gb_n_estimators: int = 200
    gb_learning_rate: float = 0.05
    gb_max_depth: int = 4
    cv_folds: int = 5

@dataclass
class BacktestConfig:
    start_date: str = "2000-01-01"
    end_date: str = "2024-12-31"
    train_window: int = 60
    test_window: int = 1
    step_size: int = 1
    transaction_cost: float = 0.001
    duration_target_min: float = 1.0
    duration_target_max: float = 10.0
    duration_neutral: float = 5.0
    benchmark: str = "DGS10"

@dataclass
class SignalConfig:
    factor_weights: Dict[str, float] = field(default_factory=lambda: {
        "growth": 0.25, "inflation": 0.25, "labor": 0.20, "monetary": 0.20, "global": 0.10
    })
    signal_smoothing: int = 3
    max_position_change: float = 2.0
    volatility_target: float = 0.05

class ConfigManager:
    def __init__(self, config_path: Optional[str] = None):
        self.fred = FREDConfig()
        self.factors = FactorConfig()
        self.yield_curve = YieldCurveConfig()
        self.model = ModelConfig()
        self.backtest = BacktestConfig()
        self.signal = SignalConfig()
        if config_path and os.path.exists(config_path):
            self.load_from_yaml(config_path)
    
    def load_from_yaml(self, path: str):
        with open(path, 'r') as f:
            config = yaml.safe_load(f)
        for section, values in config.items():
            if hasattr(self, section):
                current = getattr(self, section)
                for key, val in values.items():
                    if hasattr(current, key):
                        setattr(current, key, val)
    
    def save_to_yaml(self, path: str):
        config_dict = {
            "fred": self.fred.__dict__, "factors": self.factors.__dict__,
            "yield_curve": self.yield_curve.__dict__, "model": self.model.__dict__,
            "backtest": self.backtest.__dict__, "signal": self.signal.__dict__
        }
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False)

CONFIG = ConfigManager()