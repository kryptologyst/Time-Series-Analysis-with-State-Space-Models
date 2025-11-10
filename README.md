# Time Series Analysis with State Space Models

A comprehensive Python project for time series analysis featuring state space models, advanced forecasting methods, and interactive visualizations.

## Features

### Core Models
- **State Space Models**: Kalman filters and structural time series models
- **ARIMA**: Automatic parameter selection with pmdarima
- **Prophet**: Facebook's forecasting tool for trend and seasonality
- **LSTM**: Deep learning approach for complex patterns
- **Ensemble Methods**: Combine multiple models for robust forecasting

### Analysis Capabilities
- **Anomaly Detection**: Isolation Forest for outlier identification
- **Component Decomposition**: Trend, seasonal, and noise separation
- **Probabilistic Forecasting**: Confidence intervals and uncertainty quantification
- **Model Comparison**: Side-by-side evaluation of different approaches

### Visualization
- **Interactive Dashboards**: Streamlit web interface
- **Static Plots**: Matplotlib and Seaborn for publication-quality figures
- **Interactive Charts**: Plotly for exploration and analysis
- **Component Analysis**: Detailed breakdown of time series components

## Installation

### Prerequisites
- Python 3.10 or higher
- pip package manager

### Setup
1. Clone the repository:
```bash
git clone https://github.com/kryptologyst/Time-Series-Analysis-with-State-Space-Models.git
cd Time-Series-Analysis-with-State-Space-Models
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Quick Start

### Command Line Usage

1. **Generate synthetic data and run basic analysis**:
```bash
python src/data_generator.py
```

2. **Run the complete analysis pipeline**:
```bash
python src/models.py
```

3. **Launch the interactive dashboard**:
```bash
streamlit run app.py
```

### Python API Usage

```python
from src.data_generator import TimeSeriesGenerator, TimeSeriesConfig
from src.models import StateSpaceModel, ARIMAModel, ModelEnsemble
from src.visualization import TimeSeriesVisualizer

# Generate synthetic time series data
config = TimeSeriesConfig(n_points=200, trend_strength=0.05)
generator = TimeSeriesGenerator(config)
time, observed, components = generator.generate_simple_series()

# Fit state space model
ssm = StateSpaceModel()
ssm.fit(observed)
forecast, ci = ssm.predict(steps=12)

# Fit multiple models
ensemble = ModelEnsemble(['state_space', 'arima', 'prophet'])
ensemble.fit(observed)
ensemble_forecast, ensemble_ci = ensemble.predict(steps=12)

# Create visualizations
visualizer = TimeSeriesVisualizer()
fig = visualizer.plot_forecast(time, observed, forecast, ci)
```

## Project Structure

```
time-series-analysis/
├── src/                    # Source code
│   ├── data_generator.py   # Synthetic data generation
│   ├── models.py          # Forecasting models
│   └── visualization.py   # Plotting functions
├── tests/                 # Unit tests
│   └── test_models.py    # Comprehensive test suite
├── config/               # Configuration files
│   └── config.yaml      # Main configuration
├── data/                # Data directories
│   ├── raw/            # Raw data files
│   ├── processed/      # Processed data
│   └── external/      # External datasets
├── models/             # Saved model files
├── notebooks/          # Jupyter notebooks
├── app.py             # Streamlit dashboard
├── requirements.txt   # Python dependencies
├── .gitignore        # Git ignore rules
└── README.md         # This file
```

## Configuration

The project uses YAML configuration files for easy customization. Key settings include:

### Data Generation
```yaml
data:
  synthetic:
    n_points: 200
    trend_strength: 0.05
    seasonal_amplitude: 2.0
    seasonal_period: 12
    noise_scale: 0.5
    random_seed: 42
```

### Model Parameters
```yaml
models:
  state_space:
    level: "local linear trend"
    seasonal: 12
    trend: true
  
  arima:
    auto_arima: true
    max_p: 5
    max_q: 5
  
  prophet:
    yearly_seasonality: true
    seasonality_mode: "additive"
```

## Web Dashboard

Launch the interactive Streamlit dashboard:

```bash
streamlit run app.py
```

### Dashboard Features
- **Data Generation**: Interactive controls for synthetic data parameters
- **Model Selection**: Choose which models to fit and compare
- **Real-time Visualization**: Interactive plots with Plotly
- **Anomaly Detection**: Identify outliers in your time series
- **Model Comparison**: Side-by-side evaluation of forecasting performance

## Advanced Usage

### Custom Data Sources

Replace synthetic data with your own time series:

```python
import pandas as pd
import numpy as np

# Load your data
df = pd.read_csv('your_data.csv')
time = df['timestamp'].values
observed = df['value'].values

# Use with existing models
ssm = StateSpaceModel()
ssm.fit(observed)
forecast, ci = ssm.predict(steps=24)
```

### Model Customization

Create custom model configurations:

```python
from src.models import StateSpaceModel, ModelConfig

# Custom state space model
config = ModelConfig(
    level="local level",
    seasonal=6,
    trend=False
)
ssm = StateSpaceModel(config)
ssm.fit(observed)
```

### Batch Processing

Process multiple time series:

```python
from src.data_generator import TimeSeriesGenerator

# Generate multiple series
generator = TimeSeriesGenerator()
multiple_series = generator.generate_multiple_series()

# Process each series
results = {}
for name, (time, observed, components) in multiple_series.items():
    ssm = StateSpaceModel()
    ssm.fit(observed)
    forecast, ci = ssm.predict(12)
    results[name] = (forecast, ci)
```

## Testing

Run the comprehensive test suite:

```bash
# Run all tests
pytest tests/

# Run with coverage
pytest tests/ --cov=src

# Run specific test file
pytest tests/test_models.py

# Run with verbose output
pytest tests/ -v
```

### Test Coverage
- Data generation and validation
- Model fitting and prediction
- Visualization functionality
- Error handling and edge cases
- Integration tests for complete pipelines

## Performance Optimization

### Large Datasets
For datasets with >10,000 points:
- Use chunked processing
- Enable parallel model fitting
- Consider dimensionality reduction

### Memory Management
- Use data generators for large datasets
- Clear model objects when not needed
- Monitor memory usage with large ensembles

## Troubleshooting

### Common Issues

1. **Import Errors**: Ensure all dependencies are installed
```bash
pip install -r requirements.txt
```

2. **Model Fitting Failures**: Check data quality and parameters
```python
# Validate data
assert not np.any(np.isnan(observed))
assert len(observed) > 50  # Minimum for reliable fitting
```

3. **Memory Issues**: Reduce batch size or use data sampling
```python
# Sample data for large datasets
sample_indices = np.random.choice(len(observed), size=1000, replace=False)
sampled_data = observed[sample_indices]
```

### Debug Mode
Enable detailed logging:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature-name`
3. Make your changes and add tests
4. Run the test suite: `pytest tests/`
5. Commit your changes: `git commit -m "Add feature"`
6. Push to the branch: `git push origin feature-name`
7. Submit a pull request

### Development Setup
```bash
# Install development dependencies
pip install -r requirements.txt
pip install pytest pytest-cov black flake8 mypy

# Run code formatting
black src/ tests/

# Run linting
flake8 src/ tests/

# Run type checking
mypy src/
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- **statsmodels**: State space models and statistical analysis
- **pmdarima**: Automatic ARIMA model selection
- **Prophet**: Facebook's forecasting library
- **Streamlit**: Web application framework
- **Plotly**: Interactive visualization library

## References

1. Durbin, J., & Koopman, S. J. (2012). Time series analysis by state space methods.
2. Hyndman, R. J., & Athanasopoulos, G. (2018). Forecasting: principles and practice.
3. Taylor, S. J., & Letham, B. (2018). Forecasting at scale.
4. Goodfellow, I., Bengio, Y., & Courville, A. (2016). Deep learning.

## Support

For questions, issues, or contributions:
- Create an issue on GitHub
- Check the documentation
- Review the test cases for usage examples

---

**Note**: This project is designed for educational and research purposes. Always validate results and consider domain-specific requirements when applying to real-world problems.
# Time-Series-Analysis-with-State-Space-Models
