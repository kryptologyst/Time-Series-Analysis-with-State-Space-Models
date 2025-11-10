"""
Data generation module for time series analysis.

This module provides functions to generate synthetic time series data
with various components including trends, seasonality, and noise.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import yaml
from pathlib import Path


@dataclass
class TimeSeriesConfig:
    """Configuration class for time series generation."""
    n_points: int = 200
    trend_strength: float = 0.05
    seasonal_amplitude: float = 2.0
    seasonal_period: int = 12
    noise_scale: float = 0.5
    random_seed: int = 42


class TimeSeriesGenerator:
    """Generator for synthetic time series data."""
    
    def __init__(self, config: Optional[TimeSeriesConfig] = None):
        """
        Initialize the time series generator.
        
        Args:
            config: Configuration object. If None, uses default values.
        """
        self.config = config or TimeSeriesConfig()
        np.random.seed(self.config.random_seed)
    
    def generate_trend(self, time: np.ndarray) -> np.ndarray:
        """
        Generate linear trend component.
        
        Args:
            time: Time array
            
        Returns:
            Trend component
        """
        return self.config.trend_strength * time
    
    def generate_seasonal(self, time: np.ndarray) -> np.ndarray:
        """
        Generate seasonal component.
        
        Args:
            time: Time array
            
        Returns:
            Seasonal component
        """
        return self.config.seasonal_amplitude * np.sin(2 * np.pi * time / self.config.seasonal_period)
    
    def generate_noise(self, n_points: int) -> np.ndarray:
        """
        Generate noise component.
        
        Args:
            n_points: Number of points
            
        Returns:
            Noise component
        """
        return np.random.normal(scale=self.config.noise_scale, size=n_points)
    
    def generate_simple_series(self) -> Tuple[np.ndarray, np.ndarray, Dict[str, np.ndarray]]:
        """
        Generate a simple time series with trend, seasonality, and noise.
        
        Returns:
            Tuple of (time, observed_values, components_dict)
        """
        time = np.arange(self.config.n_points)
        
        trend = self.generate_trend(time)
        seasonal = self.generate_seasonal(time)
        noise = self.generate_noise(self.config.n_points)
        
        observed = trend + seasonal + noise
        
        components = {
            'trend': trend,
            'seasonal': seasonal,
            'noise': noise,
            'observed': observed
        }
        
        return time, observed, components
    
    def generate_multiple_series(self, n_series: int = 3) -> Dict[str, Tuple[np.ndarray, np.ndarray, Dict[str, np.ndarray]]]:
        """
        Generate multiple time series with different characteristics.
        
        Args:
            n_series: Number of series to generate
            
        Returns:
            Dictionary of series with different configurations
        """
        series_dict = {}
        
        # Series 1: Strong trend, weak seasonality
        config1 = TimeSeriesConfig(
            n_points=self.config.n_points,
            trend_strength=0.1,
            seasonal_amplitude=1.0,
            seasonal_period=12,
            noise_scale=0.3,
            random_seed=42
        )
        gen1 = TimeSeriesGenerator(config1)
        series_dict['strong_trend'] = gen1.generate_simple_series()
        
        # Series 2: Weak trend, strong seasonality
        config2 = TimeSeriesConfig(
            n_points=self.config.n_points,
            trend_strength=0.02,
            seasonal_amplitude=3.0,
            seasonal_period=12,
            noise_scale=0.4,
            random_seed=43
        )
        gen2 = TimeSeriesGenerator(config2)
        series_dict['strong_seasonal'] = gen2.generate_simple_series()
        
        # Series 3: No trend, multiple seasonalities
        config3 = TimeSeriesConfig(
            n_points=self.config.n_points,
            trend_strength=0.0,
            seasonal_amplitude=2.0,
            seasonal_period=12,
            noise_scale=0.6,
            random_seed=44
        )
        gen3 = TimeSeriesGenerator(config3)
        time, observed, components = gen3.generate_simple_series()
        
        # Add additional seasonal component
        additional_seasonal = 1.5 * np.sin(2 * np.pi * time / 6)
        components['seasonal_2'] = additional_seasonal
        components['observed'] = components['observed'] + additional_seasonal
        
        series_dict['multi_seasonal'] = (time, components['observed'], components)
        
        return series_dict
    
    def generate_anomaly_series(self, anomaly_prob: float = 0.05) -> Tuple[np.ndarray, np.ndarray, Dict[str, np.ndarray]]:
        """
        Generate time series with anomalies.
        
        Args:
            anomaly_prob: Probability of anomaly at each point
            
        Returns:
            Tuple of (time, observed_values, components_dict)
        """
        time, observed, components = self.generate_simple_series()
        
        # Add anomalies
        anomalies = np.random.random(self.config.n_points) < anomaly_prob
        anomaly_magnitude = np.random.normal(0, 3 * self.config.noise_scale, self.config.n_points)
        anomaly_values = anomalies * anomaly_magnitude
        
        components['anomalies'] = anomaly_values
        components['observed'] = components['observed'] + anomaly_values
        
        return time, components['observed'], components


def load_config(config_path: str) -> TimeSeriesConfig:
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        TimeSeriesConfig object
    """
    with open(config_path, 'r') as f:
        config_dict = yaml.safe_load(f)
    
    synthetic_config = config_dict['data']['synthetic']
    return TimeSeriesConfig(**synthetic_config)


def save_data(data: Dict[str, Tuple[np.ndarray, np.ndarray, Dict[str, np.ndarray]]], 
              output_dir: str) -> None:
    """
    Save generated data to files.
    
    Args:
        data: Dictionary of generated time series
        output_dir: Output directory path
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    for name, (time, observed, components) in data.items():
        # Create DataFrame
        df = pd.DataFrame({
            'time': time,
            'observed': observed,
            **components
        })
        
        # Save to CSV
        df.to_csv(output_path / f"{name}_series.csv", index=False)
        
        # Save metadata
        metadata = {
            'n_points': len(time),
            'components': list(components.keys()),
            'description': f"Generated time series: {name}"
        }
        
        with open(output_path / f"{name}_metadata.yaml", 'w') as f:
            yaml.dump(metadata, f)


if __name__ == "__main__":
    # Example usage
    config = TimeSeriesConfig()
    generator = TimeSeriesGenerator(config)
    
    # Generate simple series
    time, observed, components = generator.generate_simple_series()
    print(f"Generated series with {len(time)} points")
    print(f"Components: {list(components.keys())}")
    
    # Generate multiple series
    multiple_series = generator.generate_multiple_series()
    print(f"Generated {len(multiple_series)} different series")
    
    # Save data
    save_data(multiple_series, "data/processed")
