"""
Unit tests for time series analysis project.

This module contains comprehensive tests for data generation, models, and visualization.
"""

import pytest
import numpy as np
import pandas as pd
from unittest.mock import patch, MagicMock
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from data_generator import TimeSeriesGenerator, TimeSeriesConfig, load_config, save_data
from models import StateSpaceModel, ARIMAModel, ProphetModel, AnomalyDetector, ModelEnsemble
from visualization import TimeSeriesVisualizer, InteractiveVisualizer


class TestTimeSeriesConfig:
    """Test TimeSeriesConfig class."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = TimeSeriesConfig()
        assert config.n_points == 200
        assert config.trend_strength == 0.05
        assert config.seasonal_amplitude == 2.0
        assert config.seasonal_period == 12
        assert config.noise_scale == 0.5
        assert config.random_seed == 42
    
    def test_custom_config(self):
        """Test custom configuration values."""
        config = TimeSeriesConfig(
            n_points=100,
            trend_strength=0.1,
            seasonal_amplitude=3.0,
            seasonal_period=6,
            noise_scale=1.0,
            random_seed=123
        )
        assert config.n_points == 100
        assert config.trend_strength == 0.1
        assert config.seasonal_amplitude == 3.0
        assert config.seasonal_period == 6
        assert config.noise_scale == 1.0
        assert config.random_seed == 123


class TestTimeSeriesGenerator:
    """Test TimeSeriesGenerator class."""
    
    def test_init_default(self):
        """Test generator initialization with default config."""
        generator = TimeSeriesGenerator()
        assert generator.config.n_points == 200
        assert generator.config.random_seed == 42
    
    def test_init_custom(self):
        """Test generator initialization with custom config."""
        config = TimeSeriesConfig(n_points=100, random_seed=123)
        generator = TimeSeriesGenerator(config)
        assert generator.config.n_points == 100
        assert generator.config.random_seed == 123
    
    def test_generate_trend(self):
        """Test trend generation."""
        generator = TimeSeriesGenerator()
        time = np.arange(10)
        trend = generator.generate_trend(time)
        
        expected = 0.05 * time
        np.testing.assert_array_equal(trend, expected)
    
    def test_generate_seasonal(self):
        """Test seasonal generation."""
        generator = TimeSeriesGenerator()
        time = np.arange(12)
        seasonal = generator.generate_seasonal(time)
        
        # Check that seasonal component has correct period
        assert len(seasonal) == 12
        assert np.isclose(seasonal[0], seasonal[12], atol=1e-10)  # Should be periodic
    
    def test_generate_noise(self):
        """Test noise generation."""
        generator = TimeSeriesGenerator()
        noise = generator.generate_noise(100)
        
        assert len(noise) == 100
        assert np.isclose(np.mean(noise), 0, atol=0.5)  # Mean should be close to 0
        assert np.isclose(np.std(noise), 0.5, atol=0.2)  # Std should be close to 0.5
    
    def test_generate_simple_series(self):
        """Test simple series generation."""
        generator = TimeSeriesGenerator()
        time, observed, components = generator.generate_simple_series()
        
        assert len(time) == 200
        assert len(observed) == 200
        assert 'trend' in components
        assert 'seasonal' in components
        assert 'noise' in components
        assert 'observed' in components
        
        # Check that observed is sum of components
        expected_observed = components['trend'] + components['seasonal'] + components['noise']
        np.testing.assert_array_equal(observed, expected_observed)
    
    def test_generate_multiple_series(self):
        """Test multiple series generation."""
        generator = TimeSeriesGenerator()
        series_data = generator.generate_multiple_series()
        
        assert len(series_data) == 3
        assert 'strong_trend' in series_data
        assert 'strong_seasonal' in series_data
        assert 'multi_seasonal' in series_data
        
        # Check each series has correct structure
        for name, (time, observed, components) in series_data.items():
            assert len(time) == len(observed)
            assert isinstance(components, dict)
    
    def test_generate_anomaly_series(self):
        """Test anomaly series generation."""
        generator = TimeSeriesGenerator()
        time, observed, components = generator.generate_anomaly_series(anomaly_prob=0.1)
        
        assert len(time) == 200
        assert len(observed) == 200
        assert 'anomalies' in components
        
        # Check that anomalies are added
        anomaly_count = np.sum(components['anomalies'] != 0)
        assert anomaly_count > 0


class TestStateSpaceModel:
    """Test StateSpaceModel class."""
    
    def test_init(self):
        """Test model initialization."""
        model = StateSpaceModel()
        assert not model.fitted
        assert model.model is None
        assert model.result is None
    
    def test_fit(self):
        """Test model fitting."""
        # Generate test data
        generator = TimeSeriesGenerator(TimeSeriesConfig(n_points=50))
        time, observed, components = generator.generate_simple_series()
        
        model = StateSpaceModel()
        model.fit(observed)
        
        assert model.fitted
        assert model.model is not None
        assert model.result is not None
    
    def test_predict_before_fit(self):
        """Test prediction before fitting raises error."""
        model = StateSpaceModel()
        with pytest.raises(ValueError, match="Model must be fitted"):
            model.predict(12)
    
    def test_predict_after_fit(self):
        """Test prediction after fitting."""
        # Generate test data
        generator = TimeSeriesGenerator(TimeSeriesConfig(n_points=50))
        time, observed, components = generator.generate_simple_series()
        
        model = StateSpaceModel()
        model.fit(observed)
        forecast, ci = model.predict(12)
        
        assert len(forecast) == 12
        assert ci.shape == (12, 2)  # Lower and upper bounds
    
    def test_get_smoothed_state_before_fit(self):
        """Test getting smoothed state before fitting raises error."""
        model = StateSpaceModel()
        with pytest.raises(ValueError, match="Model must be fitted"):
            model.get_smoothed_state()
    
    def test_get_smoothed_state_after_fit(self):
        """Test getting smoothed state after fitting."""
        # Generate test data
        generator = TimeSeriesGenerator(TimeSeriesConfig(n_points=50))
        time, observed, components = generator.generate_simple_series()
        
        model = StateSpaceModel()
        model.fit(observed)
        smoothed = model.get_smoothed_state()
        
        assert len(smoothed) == 50
    
    def test_get_components(self):
        """Test getting components."""
        # Generate test data
        generator = TimeSeriesGenerator(TimeSeriesConfig(n_points=50))
        time, observed, components = generator.generate_simple_series()
        
        model = StateSpaceModel()
        model.fit(observed)
        components = model.get_components()
        
        assert isinstance(components, dict)
        assert 'level' in components


class TestARIMAModel:
    """Test ARIMAModel class."""
    
    def test_init(self):
        """Test model initialization."""
        model = ARIMAModel()
        assert not model.fitted
        assert model.model is None
    
    def test_fit(self):
        """Test model fitting."""
        # Generate test data
        generator = TimeSeriesGenerator(TimeSeriesConfig(n_points=50))
        time, observed, components = generator.generate_simple_series()
        
        model = ARIMAModel()
        model.fit(observed)
        
        assert model.fitted
        assert model.model is not None
    
    def test_predict_before_fit(self):
        """Test prediction before fitting raises error."""
        model = ARIMAModel()
        with pytest.raises(ValueError, match="Model must be fitted"):
            model.predict(12)
    
    def test_predict_after_fit(self):
        """Test prediction after fitting."""
        # Generate test data
        generator = TimeSeriesGenerator(TimeSeriesConfig(n_points=50))
        time, observed, components = generator.generate_simple_series()
        
        model = ARIMAModel()
        model.fit(observed)
        forecast, ci = model.predict(12)
        
        assert len(forecast) == 12
        assert ci.shape == (12, 2)


class TestProphetModel:
    """Test ProphetModel class."""
    
    def test_init(self):
        """Test model initialization."""
        model = ProphetModel()
        assert not model.fitted
        assert model.model is None
    
    def test_fit(self):
        """Test model fitting."""
        # Generate test data
        generator = TimeSeriesGenerator(TimeSeriesConfig(n_points=50))
        time, observed, components = generator.generate_simple_series()
        
        model = ProphetModel()
        model.fit(observed)
        
        assert model.fitted
        assert model.model is not None
    
    def test_predict_before_fit(self):
        """Test prediction before fitting raises error."""
        model = ProphetModel()
        with pytest.raises(ValueError, match="Model must be fitted"):
            model.predict(12)
    
    def test_predict_after_fit(self):
        """Test prediction after fitting."""
        # Generate test data
        generator = TimeSeriesGenerator(TimeSeriesConfig(n_points=50))
        time, observed, components = generator.generate_simple_series()
        
        model = ProphetModel()
        model.fit(observed)
        forecast, ci = model.predict(12)
        
        assert len(forecast) == 12
        assert ci.shape == (12, 2)


class TestAnomalyDetector:
    """Test AnomalyDetector class."""
    
    def test_init(self):
        """Test detector initialization."""
        detector = AnomalyDetector(contamination=0.1)
        assert detector.contamination == 0.1
        assert not detector.fitted
    
    def test_fit(self):
        """Test detector fitting."""
        # Generate test data
        generator = TimeSeriesGenerator(TimeSeriesConfig(n_points=50))
        time, observed, components = generator.generate_simple_series()
        
        detector = AnomalyDetector()
        detector.fit(observed)
        
        assert detector.fitted
    
    def test_predict_before_fit(self):
        """Test prediction before fitting raises error."""
        detector = AnomalyDetector()
        with pytest.raises(ValueError, match="Model must be fitted"):
            detector.predict(np.array([1, 2, 3]))
    
    def test_predict_after_fit(self):
        """Test prediction after fitting."""
        # Generate test data
        generator = TimeSeriesGenerator(TimeSeriesConfig(n_points=50))
        time, observed, components = generator.generate_simple_series()
        
        detector = AnomalyDetector()
        detector.fit(observed)
        predictions = detector.predict(observed)
        
        assert len(predictions) == len(observed)
        assert all(pred in [-1, 1] for pred in predictions)  # Should be -1 or 1
    
    def test_get_anomaly_indices(self):
        """Test getting anomaly indices."""
        # Generate test data
        generator = TimeSeriesGenerator(TimeSeriesConfig(n_points=50))
        time, observed, components = generator.generate_simple_series()
        
        detector = AnomalyDetector()
        detector.fit(observed)
        anomaly_indices = detector.get_anomaly_indices(observed)
        
        assert isinstance(anomaly_indices, np.ndarray)
        assert len(anomaly_indices) <= len(observed)


class TestModelEnsemble:
    """Test ModelEnsemble class."""
    
    def test_init(self):
        """Test ensemble initialization."""
        ensemble = ModelEnsemble(['state_space', 'arima'])
        assert ensemble.models == ['state_space', 'arima']
        assert len(ensemble.fitted_models) == 0
        assert ensemble.weights is None
    
    def test_fit(self):
        """Test ensemble fitting."""
        # Generate test data
        generator = TimeSeriesGenerator(TimeSeriesConfig(n_points=50))
        time, observed, components = generator.generate_simple_series()
        
        ensemble = ModelEnsemble(['state_space', 'arima'])
        ensemble.fit(observed)
        
        assert len(ensemble.fitted_models) > 0
        assert ensemble.weights is not None
    
    def test_predict_before_fit(self):
        """Test prediction before fitting raises error."""
        ensemble = ModelEnsemble()
        with pytest.raises(ValueError, match="No models fitted"):
            ensemble.predict(12)
    
    def test_predict_after_fit(self):
        """Test prediction after fitting."""
        # Generate test data
        generator = TimeSeriesGenerator(TimeSeriesConfig(n_points=50))
        time, observed, components = generator.generate_simple_series()
        
        ensemble = ModelEnsemble(['state_space'])
        ensemble.fit(observed)
        forecast, ci = ensemble.predict(12)
        
        assert len(forecast) == 12
        assert ci.shape == (12, 2)


class TestTimeSeriesVisualizer:
    """Test TimeSeriesVisualizer class."""
    
    def test_init(self):
        """Test visualizer initialization."""
        visualizer = TimeSeriesVisualizer()
        assert visualizer.figsize == (12, 8)
        assert visualizer.style == 'seaborn-v0_8'
    
    def test_plot_time_series(self):
        """Test time series plotting."""
        visualizer = TimeSeriesVisualizer()
        time = np.arange(10)
        data = np.random.randn(10)
        
        fig = visualizer.plot_time_series(time, data, title="Test Plot")
        
        assert fig is not None
        assert len(fig.axes) == 1
    
    def test_plot_components(self):
        """Test components plotting."""
        visualizer = TimeSeriesVisualizer()
        time = np.arange(10)
        components = {
            'trend': np.arange(10),
            'seasonal': np.sin(np.arange(10)),
            'noise': np.random.randn(10)
        }
        
        fig = visualizer.plot_components(time, components, title="Test Components")
        
        assert fig is not None
        assert len(fig.axes) == 3
    
    def test_plot_forecast(self):
        """Test forecast plotting."""
        visualizer = TimeSeriesVisualizer()
        time = np.arange(10)
        observed = np.random.randn(10)
        forecast = np.random.randn(5)
        ci = np.random.randn(5, 2)
        
        fig = visualizer.plot_forecast(time, observed, forecast, ci, title="Test Forecast")
        
        assert fig is not None
        assert len(fig.axes) == 1
    
    def test_plot_anomalies(self):
        """Test anomaly plotting."""
        visualizer = TimeSeriesVisualizer()
        time = np.arange(10)
        data = np.random.randn(10)
        anomaly_indices = np.array([2, 5])
        
        fig = visualizer.plot_anomalies(time, data, anomaly_indices, title="Test Anomalies")
        
        assert fig is not None
        assert len(fig.axes) == 1
    
    def test_plot_model_comparison(self):
        """Test model comparison plotting."""
        visualizer = TimeSeriesVisualizer()
        time = np.arange(10)
        observed = np.random.randn(10)
        forecasts = {
            'model1': np.random.randn(5),
            'model2': np.random.randn(5)
        }
        
        fig = visualizer.plot_model_comparison(time, observed, forecasts, title="Test Comparison")
        
        assert fig is not None
        assert len(fig.axes) == 1
    
    def test_plot_residuals(self):
        """Test residuals plotting."""
        visualizer = TimeSeriesVisualizer()
        residuals = np.random.randn(50)
        
        fig = visualizer.plot_residuals(residuals, title="Test Residuals")
        
        assert fig is not None
        assert len(fig.axes) == 4  # 2x2 subplot


class TestInteractiveVisualizer:
    """Test InteractiveVisualizer class."""
    
    def test_init(self):
        """Test interactive visualizer initialization."""
        viz = InteractiveVisualizer()
        assert viz is not None
    
    def test_create_interactive_forecast(self):
        """Test interactive forecast creation."""
        viz = InteractiveVisualizer()
        time = np.arange(10)
        observed = np.random.randn(10)
        forecast = np.random.randn(5)
        ci = np.random.randn(5, 2)
        
        fig = viz.create_interactive_forecast(time, observed, forecast, ci)
        
        assert fig is not None
        assert len(fig.data) >= 2  # At least observed and forecast
    
    def test_create_model_comparison(self):
        """Test interactive model comparison creation."""
        viz = InteractiveVisualizer()
        time = np.arange(10)
        observed = np.random.randn(10)
        forecasts = {
            'model1': np.random.randn(5),
            'model2': np.random.randn(5)
        }
        
        fig = viz.create_model_comparison(time, observed, forecasts)
        
        assert fig is not None
        assert len(fig.data) >= 3  # At least observed and two forecasts
    
    def test_create_components_plot(self):
        """Test interactive components plot creation."""
        viz = InteractiveVisualizer()
        time = np.arange(10)
        components = {
            'trend': np.arange(10),
            'seasonal': np.sin(np.arange(10)),
            'noise': np.random.randn(10)
        }
        
        fig = viz.create_components_plot(time, components)
        
        assert fig is not None
        assert len(fig.data) == 3  # Three components


class TestUtilityFunctions:
    """Test utility functions."""
    
    def test_save_data(self, tmp_path):
        """Test data saving functionality."""
        # Create test data
        time = np.arange(10)
        observed = np.random.randn(10)
        components = {
            'trend': np.arange(10),
            'seasonal': np.sin(np.arange(10))
        }
        
        data = {'test_series': (time, observed, components)}
        
        # Save data
        save_data(data, str(tmp_path))
        
        # Check files were created
        assert (tmp_path / "test_series_series.csv").exists()
        assert (tmp_path / "test_series_metadata.yaml").exists()
        
        # Check CSV content
        df = pd.read_csv(tmp_path / "test_series_series.csv")
        assert len(df) == 10
        assert 'time' in df.columns
        assert 'observed' in df.columns
        assert 'trend' in df.columns
        assert 'seasonal' in df.columns


# Integration tests
class TestIntegration:
    """Integration tests for the complete pipeline."""
    
    def test_full_pipeline(self):
        """Test complete pipeline from data generation to visualization."""
        # Generate data
        generator = TimeSeriesGenerator(TimeSeriesConfig(n_points=50))
        time, observed, components = generator.generate_simple_series()
        
        # Fit models
        ssm = StateSpaceModel()
        ssm.fit(observed)
        
        arima = ARIMAModel()
        arima.fit(observed)
        
        # Generate forecasts
        ssm_forecast, ssm_ci = ssm.predict(12)
        arima_forecast, arima_ci = arima.predict(12)
        
        # Create visualizations
        visualizer = TimeSeriesVisualizer()
        fig1 = visualizer.plot_time_series(time, observed)
        fig2 = visualizer.plot_forecast(time, observed, ssm_forecast, ssm_ci)
        
        # Test anomaly detection
        detector = AnomalyDetector()
        detector.fit(observed)
        anomaly_indices = detector.get_anomaly_indices(observed)
        
        # All operations should complete without errors
        assert len(ssm_forecast) == 12
        assert len(arima_forecast) == 12
        assert fig1 is not None
        assert fig2 is not None
        assert isinstance(anomaly_indices, np.ndarray)


if __name__ == "__main__":
    pytest.main([__file__])
