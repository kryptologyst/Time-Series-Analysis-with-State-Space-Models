"""
Visualization module for time series analysis.

This module provides comprehensive plotting functions for time series data,
forecasts, components, and anomaly detection results.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from typing import Dict, List, Optional, Tuple, Union
import warnings

# Set style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")


class TimeSeriesVisualizer:
    """Comprehensive time series visualization class."""
    
    def __init__(self, figsize: Tuple[int, int] = (12, 8), style: str = 'seaborn-v0_8'):
        """
        Initialize the visualizer.
        
        Args:
            figsize: Default figure size
            style: Matplotlib style
        """
        self.figsize = figsize
        self.style = style
        plt.style.use(style)
        
    def plot_time_series(self, 
                        time: np.ndarray, 
                        data: np.ndarray, 
                        title: str = "Time Series",
                        xlabel: str = "Time",
                        ylabel: str = "Value",
                        **kwargs) -> plt.Figure:
        """
        Plot a simple time series.
        
        Args:
            time: Time array
            data: Data array
            title: Plot title
            xlabel: X-axis label
            ylabel: Y-axis label
            **kwargs: Additional plotting arguments
            
        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=self.figsize)
        
        ax.plot(time, data, **kwargs)
        ax.set_title(title, fontsize=16, fontweight='bold')
        ax.set_xlabel(xlabel, fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def plot_components(self, 
                      time: np.ndarray, 
                      components: Dict[str, np.ndarray],
                      title: str = "Time Series Components",
                      **kwargs) -> plt.Figure:
        """
        Plot time series components separately.
        
        Args:
            time: Time array
            components: Dictionary of component arrays
            title: Plot title
            **kwargs: Additional plotting arguments
            
        Returns:
            Matplotlib figure
        """
        n_components = len(components)
        fig, axes = plt.subplots(n_components, 1, figsize=(self.figsize[0], self.figsize[1] * n_components))
        
        if n_components == 1:
            axes = [axes]
        
        for i, (name, data) in enumerate(components.items()):
            axes[i].plot(time, data, **kwargs)
            axes[i].set_title(f"{name.title()} Component", fontsize=14)
            axes[i].set_ylabel("Value", fontsize=12)
            axes[i].grid(True, alpha=0.3)
            
            if i == n_components - 1:
                axes[i].set_xlabel("Time", fontsize=12)
        
        fig.suptitle(title, fontsize=16, fontweight='bold')
        plt.tight_layout()
        return fig
    
    def plot_forecast(self, 
                     time: np.ndarray, 
                     observed: np.ndarray,
                     forecast: np.ndarray,
                     confidence_interval: Optional[np.ndarray] = None,
                     title: str = "Time Series Forecast",
                     **kwargs) -> plt.Figure:
        """
        Plot observed data and forecast.
        
        Args:
            time: Time array for observed data
            observed: Observed data
            forecast: Forecast values
            confidence_interval: Confidence interval array (2 columns: lower, upper)
            title: Plot title
            **kwargs: Additional plotting arguments
            
        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=self.figsize)
        
        # Plot observed data
        ax.plot(time, observed, label='Observed', color='blue', alpha=0.7)
        
        # Plot forecast
        forecast_time = np.arange(len(time), len(time) + len(forecast))
        ax.plot(forecast_time, forecast, label='Forecast', color='red', linestyle='--', linewidth=2)
        
        # Plot confidence interval
        if confidence_interval is not None:
            ci_lower = confidence_interval[:, 0]
            ci_upper = confidence_interval[:, 1]
            ax.fill_between(forecast_time, ci_lower, ci_upper, 
                          alpha=0.3, color='red', label='95% Confidence Interval')
        
        ax.set_title(title, fontsize=16, fontweight='bold')
        ax.set_xlabel("Time", fontsize=12)
        ax.set_ylabel("Value", fontsize=12)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def plot_anomalies(self, 
                      time: np.ndarray, 
                      data: np.ndarray,
                      anomaly_indices: np.ndarray,
                      title: str = "Anomaly Detection",
                      **kwargs) -> plt.Figure:
        """
        Plot time series with highlighted anomalies.
        
        Args:
            time: Time array
            data: Data array
            anomaly_indices: Indices of anomalous points
            title: Plot title
            **kwargs: Additional plotting arguments
            
        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=self.figsize)
        
        # Plot normal data
        normal_mask = np.ones(len(data), dtype=bool)
        normal_mask[anomaly_indices] = False
        
        ax.plot(time[normal_mask], data[normal_mask], 
               label='Normal', color='blue', alpha=0.7, **kwargs)
        
        # Plot anomalies
        if len(anomaly_indices) > 0:
            ax.scatter(time[anomaly_indices], data[anomaly_indices], 
                      label='Anomalies', color='red', s=50, zorder=5)
        
        ax.set_title(title, fontsize=16, fontweight='bold')
        ax.set_xlabel("Time", fontsize=12)
        ax.set_ylabel("Value", fontsize=12)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def plot_model_comparison(self, 
                            time: np.ndarray,
                            observed: np.ndarray,
                            forecasts: Dict[str, np.ndarray],
                            title: str = "Model Comparison",
                            **kwargs) -> plt.Figure:
        """
        Compare multiple model forecasts.
        
        Args:
            time: Time array for observed data
            observed: Observed data
            forecasts: Dictionary of model forecasts
            title: Plot title
            **kwargs: Additional plotting arguments
            
        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=self.figsize)
        
        # Plot observed data
        ax.plot(time, observed, label='Observed', color='black', linewidth=2)
        
        # Plot forecasts
        colors = plt.cm.Set1(np.linspace(0, 1, len(forecasts)))
        for i, (model_name, forecast) in enumerate(forecasts.items()):
            forecast_time = np.arange(len(time), len(time) + len(forecast))
            ax.plot(forecast_time, forecast, 
                   label=f'{model_name.title()} Forecast', 
                   color=colors[i], linestyle='--', linewidth=2)
        
        ax.set_title(title, fontsize=16, fontweight='bold')
        ax.set_xlabel("Time", fontsize=12)
        ax.set_ylabel("Value", fontsize=12)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def plot_residuals(self, 
                      residuals: np.ndarray,
                      title: str = "Residuals Analysis",
                      **kwargs) -> plt.Figure:
        """
        Plot residual analysis.
        
        Args:
            residuals: Residual values
            title: Plot title
            **kwargs: Additional plotting arguments
            
        Returns:
            Matplotlib figure
        """
        fig, axes = plt.subplots(2, 2, figsize=(self.figsize[0], self.figsize[1]))
        
        # Time series of residuals
        axes[0, 0].plot(residuals, **kwargs)
        axes[0, 0].set_title("Residuals Over Time")
        axes[0, 0].set_xlabel("Time")
        axes[0, 0].set_ylabel("Residuals")
        axes[0, 0].grid(True, alpha=0.3)
        
        # Histogram of residuals
        axes[0, 1].hist(residuals, bins=30, alpha=0.7, **kwargs)
        axes[0, 1].set_title("Residuals Distribution")
        axes[0, 1].set_xlabel("Residuals")
        axes[0, 1].set_ylabel("Frequency")
        axes[0, 1].grid(True, alpha=0.3)
        
        # Q-Q plot
        from scipy import stats
        stats.probplot(residuals, dist="norm", plot=axes[1, 0])
        axes[1, 0].set_title("Q-Q Plot")
        axes[1, 0].grid(True, alpha=0.3)
        
        # Autocorrelation
        from statsmodels.tsa.stattools import acf
        lags = min(40, len(residuals) // 4)
        autocorr = acf(residuals, nlags=lags)
        axes[1, 1].bar(range(len(autocorr)), autocorr, **kwargs)
        axes[1, 1].set_title("Autocorrelation Function")
        axes[1, 1].set_xlabel("Lag")
        axes[1, 1].set_ylabel("Autocorrelation")
        axes[1, 1].grid(True, alpha=0.3)
        
        fig.suptitle(title, fontsize=16, fontweight='bold')
        plt.tight_layout()
        return fig


class InteractiveVisualizer:
    """Interactive visualization using Plotly."""
    
    def __init__(self):
        """Initialize interactive visualizer."""
        pass
    
    def create_interactive_forecast(self, 
                                  time: np.ndarray,
                                  observed: np.ndarray,
                                  forecast: np.ndarray,
                                  confidence_interval: Optional[np.ndarray] = None,
                                  title: str = "Interactive Time Series Forecast") -> go.Figure:
        """
        Create interactive forecast plot.
        
        Args:
            time: Time array for observed data
            observed: Observed data
            forecast: Forecast values
            confidence_interval: Confidence interval array
            title: Plot title
            
        Returns:
            Plotly figure
        """
        fig = go.Figure()
        
        # Add observed data
        fig.add_trace(go.Scatter(
            x=time,
            y=observed,
            mode='lines',
            name='Observed',
            line=dict(color='blue', width=2)
        ))
        
        # Add forecast
        forecast_time = np.arange(len(time), len(time) + len(forecast))
        fig.add_trace(go.Scatter(
            x=forecast_time,
            y=forecast,
            mode='lines',
            name='Forecast',
            line=dict(color='red', width=2, dash='dash')
        ))
        
        # Add confidence interval
        if confidence_interval is not None:
            ci_lower = confidence_interval[:, 0]
            ci_upper = confidence_interval[:, 1]
            
            fig.add_trace(go.Scatter(
                x=forecast_time,
                y=ci_upper,
                mode='lines',
                line=dict(width=0),
                showlegend=False,
                hoverinfo='skip'
            ))
            
            fig.add_trace(go.Scatter(
                x=forecast_time,
                y=ci_lower,
                mode='lines',
                line=dict(width=0),
                fill='tonexty',
                fillcolor='rgba(255,0,0,0.2)',
                name='95% Confidence Interval',
                hoverinfo='skip'
            ))
        
        fig.update_layout(
            title=title,
            xaxis_title="Time",
            yaxis_title="Value",
            hovermode='x unified',
            template='plotly_white'
        )
        
        return fig
    
    def create_model_comparison(self, 
                              time: np.ndarray,
                              observed: np.ndarray,
                              forecasts: Dict[str, np.ndarray],
                              title: str = "Interactive Model Comparison") -> go.Figure:
        """
        Create interactive model comparison plot.
        
        Args:
            time: Time array for observed data
            observed: Observed data
            forecasts: Dictionary of model forecasts
            title: Plot title
            
        Returns:
            Plotly figure
        """
        fig = go.Figure()
        
        # Add observed data
        fig.add_trace(go.Scatter(
            x=time,
            y=observed,
            mode='lines',
            name='Observed',
            line=dict(color='black', width=3)
        ))
        
        # Add forecasts
        colors = px.colors.qualitative.Set1
        for i, (model_name, forecast) in enumerate(forecasts.items()):
            forecast_time = np.arange(len(time), len(time) + len(forecast))
            fig.add_trace(go.Scatter(
                x=forecast_time,
                y=forecast,
                mode='lines',
                name=f'{model_name.title()} Forecast',
                line=dict(color=colors[i % len(colors)], width=2, dash='dash')
            ))
        
        fig.update_layout(
            title=title,
            xaxis_title="Time",
            yaxis_title="Value",
            hovermode='x unified',
            template='plotly_white'
        )
        
        return fig
    
    def create_components_plot(self, 
                              time: np.ndarray,
                              components: Dict[str, np.ndarray],
                              title: str = "Interactive Components") -> go.Figure:
        """
        Create interactive components plot.
        
        Args:
            time: Time array
            components: Dictionary of component arrays
            title: Plot title
            
        Returns:
            Plotly figure
        """
        n_components = len(components)
        
        fig = make_subplots(
            rows=n_components,
            cols=1,
            subplot_titles=list(components.keys()),
            vertical_spacing=0.1
        )
        
        colors = px.colors.qualitative.Set1
        
        for i, (name, data) in enumerate(components.items()):
            fig.add_trace(
                go.Scatter(
                    x=time,
                    y=data,
                    mode='lines',
                    name=name,
                    line=dict(color=colors[i % len(colors)], width=2)
                ),
                row=i+1,
                col=1
            )
        
        fig.update_layout(
            title=title,
            height=200 * n_components,
            template='plotly_white'
        )
        
        return fig


def save_plot(fig: Union[plt.Figure, go.Figure], 
              filename: str, 
              format: str = 'png',
              dpi: int = 300) -> None:
    """
    Save plot to file.
    
    Args:
        fig: Matplotlib or Plotly figure
        filename: Output filename
        format: File format
        dpi: DPI for raster formats
    """
    if isinstance(fig, plt.Figure):
        fig.savefig(filename, format=format, dpi=dpi, bbox_inches='tight')
    elif isinstance(fig, go.Figure):
        fig.write_html(filename.replace('.png', '.html'))
        fig.write_image(filename, format=format, width=1200, height=800)


if __name__ == "__main__":
    # Example usage
    from src.data_generator import TimeSeriesGenerator, TimeSeriesConfig
    
    # Generate test data
    config = TimeSeriesConfig()
    generator = TimeSeriesGenerator(config)
    time, observed, components = generator.generate_simple_series()
    
    # Create visualizer
    visualizer = TimeSeriesVisualizer()
    
    # Plot time series
    fig1 = visualizer.plot_time_series(time, observed, title="Generated Time Series")
    
    # Plot components
    fig2 = visualizer.plot_components(time, components, title="Time Series Components")
    
    # Create interactive plot
    interactive_viz = InteractiveVisualizer()
    fig3 = interactive_viz.create_components_plot(time, components)
    
    print("Visualization examples created successfully!")
