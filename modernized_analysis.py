"""
Modernized State Space Models implementation.

This script demonstrates the original functionality with modern Python practices,
type hints, error handling, and comprehensive documentation.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Tuple, Dict, Optional
import warnings
from pathlib import Path

# State space models
from statsmodels.tsa.statespace.structural import UnobservedComponents

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')


def generate_synthetic_data(n_points: int = 200, 
                          trend_strength: float = 0.05,
                          seasonal_amplitude: float = 2.0,
                          seasonal_period: int = 12,
                          noise_scale: float = 0.5,
                          random_seed: int = 42) -> Tuple[np.ndarray, np.ndarray, Dict[str, np.ndarray]]:
    """
    Generate synthetic time series data with trend, seasonality, and noise.
    
    Args:
        n_points: Number of data points to generate
        trend_strength: Strength of the linear trend
        seasonal_amplitude: Amplitude of seasonal component
        seasonal_period: Period of seasonal component
        noise_scale: Standard deviation of noise
        random_seed: Random seed for reproducibility
        
    Returns:
        Tuple of (time_array, observed_values, components_dict)
    """
    np.random.seed(random_seed)
    
    # Generate time array
    time = np.arange(n_points)
    
    # Generate components
    trend = trend_strength * time
    seasonal = seasonal_amplitude * np.sin(2 * np.pi * time / seasonal_period)
    noise = np.random.normal(scale=noise_scale, size=n_points)
    
    # Combine components
    observed = trend + seasonal + noise
    
    # Store components for analysis
    components = {
        'trend': trend,
        'seasonal': seasonal,
        'noise': noise,
        'observed': observed
    }
    
    return time, observed, components


def fit_state_space_model(data: np.ndarray, 
                         level: str = 'local linear trend',
                         seasonal: int = 12) -> UnobservedComponents:
    """
    Fit a state space model to the time series data.
    
    Args:
        data: Time series data
        level: Level component type
        seasonal: Seasonal period (0 for no seasonality)
        
    Returns:
        Fitted UnobservedComponents model
    """
    # Define State Space Model
    model = UnobservedComponents(
        data, 
        level=level, 
        seasonal=seasonal if seasonal > 0 else None
    )
    
    # Fit the model
    result = model.fit(disp=False)
    
    return result


def extract_components(result: UnobservedComponents) -> Dict[str, np.ndarray]:
    """
    Extract decomposed components from fitted state space model.
    
    Args:
        result: Fitted UnobservedComponents model
        
    Returns:
        Dictionary of extracted components
    """
    components = {}
    
    # Extract smoothed state components
    smoothed_state = result.smoothed_state
    
    # Level component (always present)
    components['level'] = smoothed_state[0]
    
    # Trend component (if available)
    if smoothed_state.shape[0] > 1:
        components['trend'] = smoothed_state[1]
    
    # Seasonal component (if available)
    if smoothed_state.shape[0] > 2:
        components['seasonal'] = smoothed_state[2]
    
    return components


def create_comprehensive_plot(time: np.ndarray,
                             observed: np.ndarray,
                             components: Dict[str, np.ndarray],
                             forecast: np.ndarray,
                             confidence_interval: Optional[np.ndarray] = None,
                             title: str = "State Space Model Analysis") -> plt.Figure:
    """
    Create a comprehensive plot showing original data, components, and forecast.
    
    Args:
        time: Time array
        observed: Observed data
        components: Dictionary of decomposed components
        forecast: Forecast values
        confidence_interval: Confidence interval array
        title: Plot title
        
    Returns:
        Matplotlib figure
    """
    # Create subplots
    n_components = len(components)
    fig, axes = plt.subplots(n_components + 2, 1, figsize=(12, 3 * (n_components + 2)))
    
    # Plot 1: Original time series
    axes[0].plot(time, observed, label="Observed", color='blue', alpha=0.7)
    axes[0].set_title("Original Time Series", fontsize=14, fontweight='bold')
    axes[0].set_ylabel("Value")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()
    
    # Plot components
    component_names = ['trend', 'seasonal', 'level']
    colors = ['green', 'orange', 'purple']
    
    for i, (comp_name, color) in enumerate(zip(component_names, colors)):
        if comp_name in components:
            axes[i + 1].plot(time, components[comp_name], 
                           label=f"{comp_name.title()} Component", 
                           color=color, linewidth=2)
            axes[i + 1].set_title(f"{comp_name.title()} Component", fontsize=14, fontweight='bold')
            axes[i + 1].set_ylabel("Value")
            axes[i + 1].grid(True, alpha=0.3)
            axes[i + 1].legend()
    
    # Plot forecast
    forecast_time = np.arange(len(time), len(time) + len(forecast))
    axes[-1].plot(time, observed, label="Observed", color='blue', alpha=0.7)
    axes[-1].plot(forecast_time, forecast, label="Forecast", color='red', linestyle='--', linewidth=2)
    
    # Add confidence interval if available
    if confidence_interval is not None:
        ci_lower = confidence_interval[:, 0]
        ci_upper = confidence_interval[:, 1]
        axes[-1].fill_between(forecast_time, ci_lower, ci_upper, 
                            alpha=0.3, color='red', label='95% Confidence Interval')
    
    axes[-1].set_title("Forecast", fontsize=14, fontweight='bold')
    axes[-1].set_xlabel("Time")
    axes[-1].set_ylabel("Value")
    axes[-1].grid(True, alpha=0.3)
    axes[-1].legend()
    
    # Overall title
    fig.suptitle(title, fontsize=16, fontweight='bold')
    
    plt.tight_layout()
    return fig


def main():
    """
    Main function demonstrating state space model analysis.
    """
    print("State Space Models - Modern Implementation")
    print("=" * 50)
    
    # 1. Generate synthetic data
    print("1. Generating synthetic time series data...")
    time, observed, components = generate_synthetic_data(
        n_points=200,
        trend_strength=0.05,
        seasonal_amplitude=2.0,
        seasonal_period=12,
        noise_scale=0.5,
        random_seed=42
    )
    
    print(f"   Generated {len(time)} data points")
    print(f"   Components: {list(components.keys())}")
    print(f"   Data range: [{observed.min():.2f}, {observed.max():.2f}]")
    
    # 2. Fit State Space Model
    print("\n2. Fitting State Space Model...")
    result = fit_state_space_model(observed, level='local linear trend', seasonal=12)
    
    print(f"   Model fitted successfully")
    print(f"   AIC: {result.aic:.2f}")
    print(f"   BIC: {result.bic:.2f}")
    
    # 3. Extract components
    print("\n3. Extracting decomposed components...")
    extracted_components = extract_components(result)
    
    print(f"   Extracted components: {list(extracted_components.keys())}")
    
    # 4. Generate forecast
    print("\n4. Generating forecast...")
    forecast_steps = 12
    forecast = result.forecast(steps=forecast_steps)
    confidence_interval = result.get_forecast(steps=forecast_steps).conf_int()
    
    print(f"   Generated {forecast_steps}-step forecast")
    print(f"   Forecast range: [{forecast.min():.2f}, {forecast.max():.2f}]")
    
    # 5. Create comprehensive visualization
    print("\n5. Creating comprehensive visualization...")
    fig = create_comprehensive_plot(
        time, observed, extracted_components, forecast, confidence_interval,
        title="State Space Model - Trend + Seasonal Smoothing"
    )
    
    # Save the plot
    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)
    fig.savefig(output_dir / "state_space_analysis.png", dpi=300, bbox_inches='tight')
    
    print(f"   Plot saved to: {output_dir / 'state_space_analysis.png'}")
    
    # 6. Display results
    print("\n6. Analysis Summary:")
    print(f"   - Original data variance: {np.var(observed):.2f}")
    print(f"   - Trend component variance: {np.var(extracted_components.get('trend', [0])):.2f}")
    print(f"   - Seasonal component variance: {np.var(extracted_components.get('seasonal', [0])):.2f}")
    print(f"   - Model log-likelihood: {result.llf:.2f}")
    
    # Show the plot
    plt.show()
    
    print("\nAnalysis complete!")


if __name__ == "__main__":
    main()
