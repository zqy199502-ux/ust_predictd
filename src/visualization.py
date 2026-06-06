"""
可视化模块
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px

from .config import CONFIG, ConfigManager

class YieldCurveVisualizer:
    def __init__(self, config: ConfigManager = None):
        self.config = config or CONFIG
        self.colors = {
            'current': '#1f77b4',
            'predicted': '#ff7f0e',
            'historical': '#7f7f7f',
            'mean': '#2ca02c',
            'shaded': 'rgba(31, 119, 180, 0.1)'
        }
    
    def plot_curve_comparison(self, current_curve: pd.Series, predicted_curve: pd.Series,
                             historical_data: pd.DataFrame = None, title: str = 'Yield Curve Comparison'):
        fig = go.Figure()
        tenors = sorted(current_curve.index.tolist())
        
        fig.add_trace(go.Scatter(
            x=tenors,
            y=current_curve[tenors].values,
            mode='lines+markers',
            name='Current Yield',
            line=dict(width=3, color=self.colors['current']),
            marker=dict(size=8, color=self.colors['current'])
        ))
        
        fig.add_trace(go.Scatter(
            x=tenors,
            y=predicted_curve[tenors].values,
            mode='lines+markers',
            name='Predicted Yield',
            line=dict(width=3, color=self.colors['predicted'], dash='dash'),
            marker=dict(size=8, color=self.colors['predicted'])
        ))
        
        if historical_data is not None and not historical_data.empty:
            historical_mean = historical_data.mean(axis=0)
            historical_std = historical_data.std(axis=0)
            fig.add_trace(go.Scatter(
                x=tenors + tenors[::-1],
                y=list(historical_mean[tenors] + historical_std[tenors]) + 
                  list(historical_mean[tenors] - historical_std[tenors])[::-1],
                fill='toself',
                fillcolor=self.colors['shaded'],
                line=dict(width=0),
                name='Historical Range',
                showlegend=True
            ))
            fig.add_trace(go.Scatter(
                x=tenors,
                y=historical_mean[tenors].values,
                mode='lines',
                name='Historical Mean',
                line=dict(width=2, color=self.colors['mean'], dash='dot'),
                opacity=0.7
            ))
        
        fig.update_layout(
            title=title,
            xaxis_title='Tenor (Years)',
            yaxis_title='Yield (%)',
            legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
            hovermode='x unified',
            template='plotly_white'
        )
        
        return fig
    
    def plot_time_series(self, data: pd.DataFrame, title: str = 'Time Series',
                         ylabel: str = 'Value', highlight_points: dict = None):
        fig = go.Figure()
        for col in data.columns:
            fig.add_trace(go.Scatter(
                x=data.index,
                y=data[col],
                mode='lines',
                name=col,
                connectgaps=True
            ))
        
        if highlight_points:
            for label, point in highlight_points.items():
                fig.add_trace(go.Scatter(
                    x=[point['x']],
                    y=[point['y']],
                    mode='markers',
                    name=label,
                    marker=dict(size=12, symbol='star')
                ))
        
        fig.update_layout(
            title=title,
            xaxis_title='Date',
            yaxis_title=ylabel,
            legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
            template='plotly_white'
        )
        
        return fig
    
    def plot_factor_decomposition(self, factors: pd.DataFrame, title: str = 'NS Factor Decomposition'):
        fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.1,
                           subplot_titles=('Level (Long-term rate)', 'Slope (Term spread)', 'Curvature'))
        
        fig.add_trace(go.Scatter(x=factors.index, y=factors['level'], mode='lines', name='Level'), row=1, col=1)
        fig.add_trace(go.Scatter(x=factors.index, y=factors['slope'], mode='lines', name='Slope'), row=2, col=1)
        fig.add_trace(go.Scatter(x=factors.index, y=factors['curvature'], mode='lines', name='Curvature'), row=3, col=1)
        
        fig.update_layout(height=600, title=title, template='plotly_white', showlegend=False)
        fig.update_yaxes(title_text='Factor Value', row=1, col=1)
        fig.update_yaxes(title_text='Factor Value', row=2, col=1)
        fig.update_yaxes(title_text='Factor Value', row=3, col=1)
        
        return fig

class PredictionVisualizer:
    def __init__(self, config: ConfigManager = None):
        self.config = config or CONFIG
        self.curve_visualizer = YieldCurveVisualizer(config)
    
    def visualize_prediction(self, current_curve: pd.Series, predicted_curve: pd.Series,
                            historical_data: pd.DataFrame = None, horizon: int = 1):
        title = f'Yield Curve Prediction (T+{horizon} months)'
        return self.curve_visualizer.plot_curve_comparison(current_curve, predicted_curve,
                                                           historical_data, title)
    
    def visualize_backtest(self, returns_df: pd.DataFrame, benchmark_returns: pd.Series = None):
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=returns_df.index,
            y=returns_df['cumulative_return'],
            mode='lines',
            name='Strategy',
            line=dict(width=2)
        ))
        
        if benchmark_returns is not None:
            benchmark_cumulative = (1 + benchmark_returns).cumprod()
            benchmark_cumulative = benchmark_cumulative.loc[returns_df.index]
            fig.add_trace(go.Scatter(
                x=benchmark_cumulative.index,
                y=benchmark_cumulative.values,
                mode='lines',
                name='Benchmark',
                line=dict(width=2, dash='dash')
            ))
        
        fig.update_layout(
            title='Backtest Results',
            xaxis_title='Date',
            yaxis_title='Cumulative Return',
            legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
            template='plotly_white'
        )
        
        return fig
    
    def visualize_metrics(self, metrics: dict):
        labels = ['Annual Return', 'Annual Volatility', 'Sharpe Ratio', 'Max Drawdown', 'Calmar Ratio', 'Win Rate']
        values = [
            f"{metrics.get('annual_return', 0):.2%}",
            f"{metrics.get('annual_volatility', 0):.2%}",
            f"{metrics.get('sharpe_ratio', 0):.2f}",
            f"{metrics.get('max_drawdown', 0):.2%}",
            f"{metrics.get('calmar_ratio', 0):.2f}",
            f"{metrics.get('win_rate', 0):.2%}"
        ]
        
        fig = go.Figure(data=[go.Table(
            header=dict(values=['Metric', 'Value'], fill_color='lightblue', align='left'),
            cells=dict(values=[labels, values], fill_color='white', align='left'))
        ])
        fig.update_layout(title='Backtest Metrics', template='plotly_white')
        
        return fig

class MacroDashboard:
    def __init__(self, config: ConfigManager = None):
        self.config = config or CONFIG
    
    def create_summary_table(self, current_yields: pd.Series, predicted_yields: pd.Series):
        comparison_df = pd.DataFrame({
            'Current': current_yields.round(2),
            'Predicted': predicted_yields.round(2),
            'Change (bps)': ((predicted_yields - current_yields) * 100).round(0)
        })
        return comparison_df
    
    def format_yield_summary(self, comparison_df: pd.DataFrame) -> str:
        summary_lines = []
        for tenor, row in comparison_df.iterrows():
            change_dir = '+' if row['Change (bps)'] >= 0 else ''
            summary_lines.append(
                f"{tenor}Y: {row['Current']:.2f}% -> {row['Predicted']:.2f}% ({change_dir}{row['Change (bps)']:.0f}bps)"
            )
        return '<br>'.join(summary_lines)

def create_yield_curve_figure(current_curve: pd.Series, predicted_curve: pd.Series, 
                              historical_data: pd.DataFrame = None, horizon: int = 1):
    viz = PredictionVisualizer()
    return viz.visualize_prediction(current_curve, predicted_curve, historical_data, horizon)

def create_summary_table(current_yields: pd.Series, predicted_yields: pd.Series) -> pd.DataFrame:
    dashboard = MacroDashboard()
    return dashboard.create_summary_table(current_yields, predicted_yields)