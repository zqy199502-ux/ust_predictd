# U.S. Treasury Duration Timing & Yield Prediction Framework (UST PredictD)

A comprehensive quantitative framework for U.S. Treasury duration timing and yield curve prediction using high-frequency macroeconomic factors from FRED API. Includes an interactive Shiny web application for easy use.

## 🚀 Shiny Web Application (Recommended)

The easiest way to use this framework is through the interactive Shiny web application.

### Features
- **FRED API Integration** - Input your FRED API key to fetch real-time data
- **One-Click Model Update** - Train the prediction model with latest data
- **Flexible Prediction Horizon** - Drag the slider to choose prediction timeframe (1-30 days)
- **Rich Visualizations** - View yield curve, historical trends, and macro factor correlations

### Quick Start for Shiny App

```bash
# Install dependencies
pip install -r requirements_shiny.txt

# Run the application
python app.py
```

Then open your browser and navigate to the URL shown in the terminal (typically http://127.0.0.1:8000).

## License

MIT License