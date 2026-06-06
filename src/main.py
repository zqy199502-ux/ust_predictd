"""
主程序入口
"""

import argparse
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np

from .config import ConfigManager
from .data_fetcher import fetch_all_data
from .feature_engineering import build_features
from .yield_predictor import predict_yields
from .signal_generator import generate_duration_signal, DurationTimer, BacktestEngine
from .visualization import YieldCurveVisualizer, PredictionVisualizer

def main(api_key: str = None, horizon: int = 12, backtest: bool = False):
    config = ConfigManager()
    if api_key:
        import os
        os.environ['FRED_API_KEY'] = api_key
    
    print("Fetching data...")
    macro_data, yield_data = fetch_all_data(api_key=api_key,
                                            start_date=config.backtest.start_date,
                                            end_date=config.backtest.end_date,
                                            target_frequency='m')
    
    if macro_data.empty or yield_data.empty:
        print("Failed to fetch data. Exiting.")
        return
    
    print("Building features...")
    target_tenor = 10.0
    if target_tenor in yield_data.columns:
        y = yield_data[target_tenor]
    else:
        y = yield_data.iloc[:, -1]
    
    features = build_features(macro_data, y=y, detrend_method='first_diff',
                             standardize_method='zscore', select_features=True, top_k=50)
    
    print("Training prediction model...")
    predictor = predict_yields(features, yield_data, method='ns', model_name='ridge')
    
    last_features = features.iloc[-horizon:]
    if len(last_features) < horizon:
        last_features = pd.concat([features.iloc[-1:]] * horizon, ignore_index=True)
    
    print(f"Predicting yields for T+{horizon} months...")
    predictions = predictor.predict(last_features, method='ns')
    predicted_curve = predictions.iloc[-1]
    current_curve = yield_data.iloc[-1]
    
    print("\n=== Current vs Predicted Yields ===")
    comparison = pd.DataFrame({
        'Current': current_curve.round(2),
        'Predicted': predicted_curve.round(2),
        'Change (bps)': ((predicted_curve - current_curve) * 100).round(0)
    })
    print(comparison)
    
    if backtest:
        print("\nRunning backtest...")
        composite_signal = generate_duration_signal(macro_data, y.pct_change(), method='simple')
        timer = DurationTimer()
        duration_df = timer.run_duration_timing(composite_signal, y)
        
        engine = BacktestEngine()
        backtest_result = engine.run_full_backtest(duration_df, y, benchmark_duration=5.0)
        
        print("\n=== Backtest Metrics ===")
        for metric, value in backtest_result['metrics'].items():
            if 'return' in metric.lower() or 'ratio' in metric.lower() or 'volatility' in metric.lower():
                print(f"{metric}: {value:.4f}")
            elif 'drawdown' in metric.lower():
                print(f"{metric}: {value:.2%}")
        
        viz = PredictionVisualizer()
        fig = viz.visualize_backtest(backtest_result['returns'], backtest_result['benchmark'])
        fig.show()
    
    viz = YieldCurveVisualizer()
    fig = viz.plot_curve_comparison(current_curve, predicted_curve, yield_data, 
                                    title=f'Yield Curve Prediction (T+{horizon} months)')
    fig.show()
    
    return {
        'current': current_curve,
        'predicted': predicted_curve,
        'comparison': comparison
    }

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='UST Yield Prediction Tool')
    parser.add_argument('--api-key', type=str, help='FRED API key')
    parser.add_argument('--horizon', type=int, default=12, help='Prediction horizon in months')
    parser.add_argument('--backtest', action='store_true', help='Run backtest')
    
    args = parser.parse_args()
    main(api_key=args.api_key, horizon=args.horizon, backtest=args.backtest)