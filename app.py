"""
美国国债利率预测Shiny应用
提供交互式界面来预测利率走势
"""
from shiny import App, render, ui, reactive
from shinywidgets import output_widget, render_plot
import matplotlib.pyplot as plt
from datetime import datetime
from shiny_utils import ShinyYieldPredictor

app_state = {
    'predictor': None,
    'macro_data': None,
    'yield_data': None,
    'prediction': None,
    'model_info': None
}

app_ui = ui.page_sidebar(
    ui.sidebar(
        ui.h3("控制面板"),
        ui.input_password("api_key", "FRED API 密钥", placeholder="输入您的FRED API密钥"),
        ui.input_action_button("validate_key", "验证密钥", class_="btn-primary"),
        ui.output_text("key_status"),
        ui.hr(),
        ui.input_date("start_date", "数据开始日期", value="2015-01-01"),
        ui.input_action_button("update_model", "更新模型", class_="btn-success"),
        ui.hr(),
        ui.input_slider("horizon", "预测期限（天）", min=1, max=30, value=10),
        ui.hr(),
        ui.p("获取FRED API密钥: https://fred.stlouisfed.org/docs/api/api_key.html"),
        width=300
    ),
    ui.layout_columns(
        ui.card(
            ui.card_header("当前10年期国债收益率"),
            ui.h2(output_text("current_yield", inline=True)),
            ui.p(output_text("last_update", inline=True)),
            style="background-color: #e3f2fd;"
        ),
        ui.card(
            ui.card_header("预测收益率"),
            ui.h2(output_text("predicted_yield", inline=True)),
            ui.p(output_text("prediction_change", inline=True)),
            style="background-color: #fff3e0;"
        ),
        col_widths=[6, 6]
    ),
    ui.hr(),
    ui.layout_columns(
        ui.card(
            ui.card_header("收益率曲线"),
            output_plot("yield_curve_plot")
        ),
        ui.card(
            ui.card_header("历史与预测"),
            output_plot("history_pred_plot")
        ),
        col_widths=[6, 6]
    ),
    ui.card(
        ui.card_header("宏观因子相关性热力图"),
        output_plot("macro_heatmap")
    ),
    ui.card(
        ui.card_header("模型信息"),
        ui.output_table("model_info_table")
    ),
    title="美国国债利率预测 (UST PredictD)",
    fillable=True
)

def server(input, output, session):
    @reactive.Effect
    @reactive.event(input.validate_key)
    def _():
        api_key = input.api_key()
        if not api_key:
            ui.notification_show("请输入API密钥", type="error")
            return
        try:
            predictor = ShinyYieldPredictor(api_key)
            if predictor.test_api_key():
                app_state['predictor'] = predictor
                ui.notification_show("API密钥验证成功！", type="message")
            else:
                ui.notification_show("API密钥无效", type="error")
        except Exception as e:
            ui.notification_show(f"验证失败: {str(e)}", type="error")
    
    @output
    @render.text
    def key_status():
        if app_state['predictor'] is not None:
            return "✓ API密钥已验证"
        return "请先验证API密钥"
    
    @reactive.Effect
    @reactive.event(input.update_model)
    def _():
        if app_state['predictor'] is None:
            ui.notification_show("请先验证API密钥", type="error")
            return
        with ui.Progress(min=0, max=100) as p:
            p.set(message="正在获取数据...", value=20)
            try:
                start_date = input.start_date().strftime("%Y-%m-%d")
                macro_data, yield_data = app_state['predictor'].fetch_data(start_date)
                app_state['macro_data'] = macro_data
                app_state['yield_data'] = yield_data
                p.set(message="正在训练模型...", value=60)
                model_info = app_state['predictor'].train_model(macro_data, yield_data)
                app_state['model_info'] = model_info
                p.set(message="生成预测...", value=90)
                prediction = app_state['predictor'].predict(input.horizon())
                app_state['prediction'] = prediction
                p.set(message="完成！", value=100)
                ui.notification_show("模型更新成功！", type="message")
            except Exception as e:
                ui.notification_show(f"更新失败: {str(e)}", type="error")
    
    @reactive.Effect
    @reactive.event(input.horizon)
    def _():
        if app_state['predictor'] is not None and app_state['predictor'].model_trained:
            app_state['prediction'] = app_state['predictor'].predict(input.horizon())
    
    @output
    @render.text
    def current_yield():
        if app_state['prediction'] is None:
            return "-- %"
        return f"{app_state['prediction']['current_yield']:.2f}%"
    
    @output
    @render.text
    def last_update():
        if app_state['prediction'] is None:
            return ""
        return f"最后更新: {app_state['prediction']['last_date'].strftime('%Y-%m-%d')}"
    
    @output
    @render.text
    def predicted_yield():
        if app_state['prediction'] is None:
            return "-- %"
        return f"{app_state['prediction']['predicted_yield']:.2f}%"
    
    @output
    @render.text
    def prediction_change():
        if app_state['prediction'] is None:
            return ""
        change = app_state['prediction']['predicted_change']
        horizon = app_state['prediction']['horizon_days']
        arrow = "↑" if change > 0 else "↓" if change < 0 else "→"
        return f"{arrow} {abs(change):.2f}% (T+{horizon}天)"
    
    @output
    @render.plot
    def yield_curve_plot():
        if app_state['yield_data'] is None or app_state['predictor'] is None:
            fig, ax = plt.subplots()
            ax.text(0.5, 0.5, "请先更新模型", ha='center', va='center', fontsize=16)
            ax.axis('off')
            return fig
        return app_state['predictor'].plot_yield_curve(app_state['yield_data'])
    
    @output
    @render.plot
    def history_pred_plot():
        if app_state['yield_data'] is None or app_state['prediction'] is None:
            fig, ax = plt.subplots()
            ax.text(0.5, 0.5, "请先更新模型", ha='center', va='center', fontsize=16)
            ax.axis('off')
            return fig
        return app_state['predictor'].plot_history_and_prediction(app_state['yield_data'], app_state['prediction'])
    
    @output
    @render.plot
    def macro_heatmap():
        if app_state['macro_data'] is None or app_state['predictor'] is None:
            fig, ax = plt.subplots()
            ax.text(0.5, 0.5, "请先更新模型", ha='center', va='center', fontsize=16)
            ax.axis('off')
            return fig
        return app_state['predictor'].plot_macro_heatmap(app_state['macro_data'])
    
    @output
    @render.table
    def model_info_table():
        if app_state['model_info'] is None:
            return None
        import pandas as pd
        df = pd.DataFrame([{
            "最后数据日期": app_state['model_info']['last_date'].strftime('%Y-%m-%d'),
            "训练样本数": app_state['model_info']['n_samples'],
            "模型类型": "Ridge Regression"
        }])
        return df

app = App(app_ui, server)

if __name__ == "__main__":
    app.run()