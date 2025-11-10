"""
State Space Models implementation for time series analysis.

This module provides modern implementations of state space models including
Kalman filters, structural time series models, and advanced forecasting methods.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
import warnings

# State space models
from statsmodels.tsa.statespace.kalman_filter import KalmanFilter
from statsmodels.tsa.statespace.structural import UnobservedComponents
from statsmodels.tsa.statespace.sarimax import SARIMAX

# Additional forecasting models
from statsmodels.tsa.arima.model import ARIMA
# from pmdarima import auto_arima  # Optional dependency
# from prophet import Prophet  # Optional dependency

# Machine learning models
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn

# Utilities
from loguru import logger


@dataclass
class ModelConfig:
    """Configuration for state space models."""
    level: str = "local linear trend"
    seasonal: int = 12
    trend: bool = True
    auto_arima: bool = True
    max_p: int = 5
    max_q: int = 5
    max_P: int = 2
    max_Q: int = 2


class StateSpaceModel:
    """Modern implementation of state space models."""
    
    def __init__(self, config: Optional[ModelConfig] = None):
        """
        Initialize the state space model.
        
        Args:
            config: Model configuration
        """
        self.config = config or ModelConfig()
        self.model = None
        self.result = None
        self.fitted = False
        
    def fit(self, data: np.ndarray, **kwargs) -> 'StateSpaceModel':
        """
        Fit the state space model to data.
        
        Args:
            data: Time series data
            **kwargs: Additional arguments for model fitting
            
        Returns:
            Self for method chaining
        """
        try:
            self.model = UnobservedComponents(
                data, 
                level=self.config.level,
                seasonal=self.config.seasonal if self.config.seasonal > 0 else None,
                trend=self.config.trend
            )
            
            self.result = self.model.fit(disp=False, **kwargs)
            self.fitted = True
            
            logger.info(f"State space model fitted successfully. AIC: {self.result.aic:.2f}")
            
        except Exception as e:
            logger.error(f"Error fitting state space model: {e}")
            raise
            
        return self
    
    def predict(self, steps: int = 12, **kwargs) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate predictions.
        
        Args:
            steps: Number of steps to forecast
            **kwargs: Additional arguments for prediction
            
        Returns:
            Tuple of (forecast, confidence_intervals)
        """
        if not self.fitted:
            raise ValueError("Model must be fitted before making predictions")
        
        forecast = self.result.forecast(steps=steps, **kwargs)
        
        # Get confidence intervals
        ci = self.result.get_forecast(steps=steps, **kwargs).conf_int()
        
        return forecast, ci
    
    def get_smoothed_state(self) -> np.ndarray:
        """
        Get smoothed state estimates.
        
        Returns:
            Smoothed state estimates
        """
        if not self.fitted:
            raise ValueError("Model must be fitted before getting smoothed state")
        
        return self.result.smoothed_state[0]  # Level component
    
    def get_components(self) -> Dict[str, np.ndarray]:
        """
        Get decomposed components.
        
        Returns:
            Dictionary of components
        """
        if not self.fitted:
            raise ValueError("Model must be fitted before getting components")
        
        components = {}
        
        # Level component
        components['level'] = self.result.smoothed_state[0]
        
        # Trend component (if available)
        if self.config.trend and self.result.smoothed_state.shape[0] > 1:
            components['trend'] = self.result.smoothed_state[1]
        
        # Seasonal component (if available)
        if self.config.seasonal > 0:
            seasonal_start_idx = 2 if self.config.trend else 1
            if self.result.smoothed_state.shape[0] > seasonal_start_idx:
                components['seasonal'] = self.result.smoothed_state[seasonal_start_idx]
        
        return components


class ARIMAModel:
    """ARIMA model implementation with auto-selection."""
    
    def __init__(self, config: Optional[ModelConfig] = None):
        """
        Initialize ARIMA model.
        
        Args:
            config: Model configuration
        """
        self.config = config or ModelConfig()
        self.model = None
        self.fitted = False
        
    def fit(self, data: np.ndarray, **kwargs) -> 'ARIMAModel':
        """
        Fit ARIMA model to data.
        
        Args:
            data: Time series data
            **kwargs: Additional arguments
            
        Returns:
            Self for method chaining
        """
        try:
            # Use manual ARIMA with simple parameter selection
            self.model = ARIMA(data, order=(1, 1, 1))
            self.model = self.model.fit()
            
            self.fitted = True
            logger.info(f"ARIMA model fitted successfully. AIC: {self.model.aic:.2f}")
            
        except Exception as e:
            logger.error(f"Error fitting ARIMA model: {e}")
            raise
            
        return self
    
    def predict(self, steps: int = 12, **kwargs) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate predictions.
        
        Args:
            steps: Number of steps to forecast
            **kwargs: Additional arguments
            
        Returns:
            Tuple of (forecast, confidence_intervals)
        """
        if not self.fitted:
            raise ValueError("Model must be fitted before making predictions")
        
        forecast = self.model.forecast(steps=steps, **kwargs)
        ci = self.model.get_forecast(steps=steps, **kwargs).conf_int()
        return forecast, ci


# class ProphetModel:
#     """Prophet model implementation."""
#     
#     def __init__(self, config: Optional[Dict] = None):
#         """
#         Initialize Prophet model.
#         
#         Args:
#             config: Prophet configuration
#         """
#         self.config = config or {
#             'yearly_seasonality': True,
#             'weekly_seasonality': False,
#             'daily_seasonality': False,
#             'seasonality_mode': 'additive'
#         }
#         self.model = None
#         self.fitted = False
#         
#     def fit(self, data: np.ndarray, **kwargs) -> 'ProphetModel':
#         """
#         Fit Prophet model to data.
#         
#         Args:
#             data: Time series data
#             **kwargs: Additional arguments
#             
#         Returns:
#             Self for method chaining
#         """
#         try:
#             # Prepare data for Prophet
#             df = pd.DataFrame({
#                 'ds': pd.date_range(start='2020-01-01', periods=len(data), freq='M'),
#                 'y': data
#             })
#             
#             self.model = Prophet(**self.config)
#             self.model.fit(df, **kwargs)
#             self.fitted = True
#             
#             logger.info("Prophet model fitted successfully")
#             
#         except Exception as e:
#             logger.error(f"Error fitting Prophet model: {e}")
#             raise
#             
#         return self
#     
#     def predict(self, steps: int = 12, **kwargs) -> Tuple[np.ndarray, np.ndarray]:
#         """
#         Generate predictions.
#         
#         Args:
#             steps: Number of steps to forecast
#             **kwargs: Additional arguments
#             
#         Returns:
#             Tuple of (forecast, confidence_intervals)
#         """
#         if not self.fitted:
#             raise ValueError("Model must be fitted before making predictions")
#         
#         # Create future dataframe
#         future = self.model.make_future_dataframe(periods=steps, freq='M')
#         forecast = self.model.predict(future)
#         
#         # Extract forecast values
#         forecast_values = forecast['yhat'].iloc[-steps:].values
#         ci_lower = forecast['yhat_lower'].iloc[-steps:].values
#         ci_upper = forecast['yhat_upper'].iloc[-steps:].values
#         
#         ci = np.column_stack([ci_lower, ci_upper])
#         
#         return forecast_values, ci


class LSTMModel(nn.Module):
    """LSTM model for time series forecasting."""
    
    def __init__(self, input_size: int = 1, hidden_size: int = 64, 
                 num_layers: int = 2, dropout: float = 0.2):
        """
        Initialize LSTM model.
        
        Args:
            input_size: Input feature size
            hidden_size: Hidden layer size
            num_layers: Number of LSTM layers
            dropout: Dropout rate
        """
        super(LSTMModel, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )
        
        self.fc = nn.Linear(hidden_size, 1)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor
            
        Returns:
            Output tensor
        """
        # Initialize hidden state
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        
        # LSTM forward pass
        out, _ = self.lstm(x, (h0, c0))
        
        # Take the last output
        out = self.dropout(out[:, -1, :])
        out = self.fc(out)
        
        return out


class AnomalyDetector:
    """Anomaly detection using Isolation Forest."""
    
    def __init__(self, contamination: float = 0.1):
        """
        Initialize anomaly detector.
        
        Args:
            contamination: Expected proportion of anomalies
        """
        self.contamination = contamination
        self.model = IsolationForest(contamination=contamination, random_state=42)
        self.scaler = StandardScaler()
        self.fitted = False
        
    def fit(self, data: np.ndarray) -> 'AnomalyDetector':
        """
        Fit the anomaly detector.
        
        Args:
            data: Time series data
            
        Returns:
            Self for method chaining
        """
        # Reshape data for sklearn
        X = data.reshape(-1, 1)
        X_scaled = self.scaler.fit_transform(X)
        
        self.model.fit(X_scaled)
        self.fitted = True
        
        logger.info("Anomaly detector fitted successfully")
        
        return self
    
    def predict(self, data: np.ndarray) -> np.ndarray:
        """
        Predict anomalies.
        
        Args:
            data: Time series data
            
        Returns:
            Anomaly scores (-1 for anomalies, 1 for normal)
        """
        if not self.fitted:
            raise ValueError("Model must be fitted before making predictions")
        
        X = data.reshape(-1, 1)
        X_scaled = self.scaler.transform(X)
        
        return self.model.predict(X_scaled)
    
    def get_anomaly_indices(self, data: np.ndarray) -> np.ndarray:
        """
        Get indices of anomalous points.
        
        Args:
            data: Time series data
            
        Returns:
            Array of anomaly indices
        """
        predictions = self.predict(data)
        return np.where(predictions == -1)[0]


class ModelEnsemble:
    """Ensemble of multiple forecasting models."""
    
    def __init__(self, models: List[str] = None):
        """
        Initialize model ensemble.
        
        Args:
            models: List of model names to include
        """
        self.models = models or ['state_space', 'arima']  # Removed prophet for now
        self.fitted_models = {}
        self.weights = None
        
    def fit(self, data: np.ndarray) -> 'ModelEnsemble':
        """
        Fit all models in the ensemble.
        
        Args:
            data: Time series data
            
        Returns:
            Self for method chaining
        """
        for model_name in self.models:
            try:
                if model_name == 'state_space':
                    model = StateSpaceModel()
                elif model_name == 'arima':
                    model = ARIMAModel()
                # elif model_name == 'prophet':
                #     model = ProphetModel()
                else:
                    continue
                
                model.fit(data)
                self.fitted_models[model_name] = model
                
            except Exception as e:
                logger.warning(f"Failed to fit {model_name}: {e}")
        
        # Calculate weights based on AIC (lower is better)
        self._calculate_weights()
        
        logger.info(f"Ensemble fitted with {len(self.fitted_models)} models")
        
        return self
    
    def _calculate_weights(self) -> None:
        """Calculate ensemble weights based on model performance."""
        if len(self.fitted_models) == 0:
            return
        
        # Simple equal weights for now
        n_models = len(self.fitted_models)
        self.weights = {name: 1.0 / n_models for name in self.fitted_models.keys()}
    
    def predict(self, steps: int = 12) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate ensemble predictions.
        
        Args:
            steps: Number of steps to forecast
            
        Returns:
            Tuple of (forecast, confidence_intervals)
        """
        if not self.fitted_models:
            raise ValueError("No models fitted")
        
        predictions = []
        conf_intervals = []
        
        for name, model in self.fitted_models.items():
            try:
                pred, ci = model.predict(steps)
                predictions.append(pred)
                conf_intervals.append(ci)
            except Exception as e:
                logger.warning(f"Failed to predict with {name}: {e}")
        
        if not predictions:
            raise ValueError("No successful predictions")
        
        # Weighted average
        weights_array = np.array([self.weights[name] for name in self.fitted_models.keys()])
        weights_array = weights_array / weights_array.sum()
        
        ensemble_pred = np.average(predictions, axis=0, weights=weights_array)
        
        # Average confidence intervals
        ensemble_ci = np.average(conf_intervals, axis=0, weights=weights_array)
        
        return ensemble_pred, ensemble_ci


if __name__ == "__main__":
    # Example usage
    from src.data_generator import TimeSeriesGenerator, TimeSeriesConfig
    
    # Generate test data
    config = TimeSeriesConfig()
    generator = TimeSeriesGenerator(config)
    time, observed, components = generator.generate_simple_series()
    
    # Test state space model
    ssm = StateSpaceModel()
    ssm.fit(observed)
    forecast, ci = ssm.predict(12)
    
    print(f"State space model forecast shape: {forecast.shape}")
    print(f"Confidence interval shape: {ci.shape}")
    
    # Test ensemble
    ensemble = ModelEnsemble(['state_space', 'arima'])
    ensemble.fit(observed)
    ensemble_forecast, ensemble_ci = ensemble.predict(12)
    
    print(f"Ensemble forecast shape: {ensemble_forecast.shape}")
