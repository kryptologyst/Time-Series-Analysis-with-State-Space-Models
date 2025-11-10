"""
Streamlit dashboard for time series analysis.

This module provides an interactive web interface for exploring time series data,
fitting models, and visualizing results.
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import yaml
from pathlib import Path
import sys

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent / "src"))

from data_generator import TimeSeriesGenerator, TimeSeriesConfig, load_config
from models import StateSpaceModel, ARIMAModel, ProphetModel, ModelEnsemble, AnomalyDetector
from visualization import InteractiveVisualizer


def load_app_config() -> dict:
    """Load application configuration."""
    config_path = Path(__file__).parent / "config" / "config.yaml"
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def initialize_session_state():
    """Initialize Streamlit session state."""
    if 'data_generated' not in st.session_state:
        st.session_state.data_generated = False
    if 'models_fitted' not in st.session_state:
        st.session_state.models_fitted = False
    if 'current_data' not in st.session_state:
        st.session_state.current_data = None
    if 'fitted_models' not in st.session_state:
        st.session_state.fitted_models = {}


def generate_data(config: dict) -> tuple:
    """Generate synthetic time series data."""
    ts_config = TimeSeriesConfig(**config['data']['synthetic'])
    generator = TimeSeriesGenerator(ts_config)
    
    # Generate multiple series
    series_data = generator.generate_multiple_series()
    
    return series_data, generator


def fit_models(data: np.ndarray, config: dict) -> dict:
    """Fit multiple models to the data."""
    models = {}
    
    # State Space Model
    try:
        ssm_config = config['models']['state_space']
        ssm = StateSpaceModel()
        ssm.fit(data)
        models['State Space'] = ssm
    except Exception as e:
        st.error(f"Failed to fit State Space Model: {e}")
    
    # ARIMA Model
    try:
        arima_config = config['models']['arima']
        arima = ARIMAModel()
        arima.fit(data)
        models['ARIMA'] = arima
    except Exception as e:
        st.error(f"Failed to fit ARIMA Model: {e}")
    
    # Prophet Model
    try:
        prophet_config = config['models']['prophet']
        prophet = ProphetModel(prophet_config)
        prophet.fit(data)
        models['Prophet'] = prophet
    except Exception as e:
        st.error(f"Failed to fit Prophet Model: {e}")
    
    return models


def create_forecast_plot(time: np.ndarray, observed: np.ndarray, 
                        forecasts: dict, title: str) -> go.Figure:
    """Create interactive forecast plot."""
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
    colors = ['red', 'blue', 'green', 'orange', 'purple']
    for i, (model_name, (forecast, ci)) in enumerate(forecasts.items()):
        forecast_time = np.arange(len(time), len(time) + len(forecast))
        
        # Add forecast line
        fig.add_trace(go.Scatter(
            x=forecast_time,
            y=forecast,
            mode='lines',
            name=f'{model_name} Forecast',
            line=dict(color=colors[i % len(colors)], width=2, dash='dash')
        ))
        
        # Add confidence interval
        if ci is not None:
            ci_lower = ci[:, 0]
            ci_upper = ci[:, 1]
            
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
                fillcolor=f'rgba({colors[i % len(colors)].replace("red", "255,0,0").replace("blue", "0,0,255").replace("green", "0,255,0").replace("orange", "255,165,0").replace("purple", "128,0,128")},0.2)',
                name=f'{model_name} 95% CI',
                hoverinfo='skip'
            ))
    
    fig.update_layout(
        title=title,
        xaxis_title="Time",
        yaxis_title="Value",
        hovermode='x unified',
        template='plotly_white',
        height=600
    )
    
    return fig


def create_components_plot(time: np.ndarray, components: dict, title: str) -> go.Figure:
    """Create components plot."""
    n_components = len(components)
    
    fig = make_subplots(
        rows=n_components,
        cols=1,
        subplot_titles=list(components.keys()),
        vertical_spacing=0.1
    )
    
    colors = ['blue', 'red', 'green', 'orange', 'purple']
    
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


def main():
    """Main Streamlit application."""
    st.set_page_config(
        page_title="Time Series Analysis Dashboard",
        page_icon="📈",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Load configuration
    config = load_app_config()
    
    # Initialize session state
    initialize_session_state()
    
    # Title and description
    st.title("📈 Time Series Analysis Dashboard")
    st.markdown("""
    This dashboard provides interactive tools for time series analysis using state space models,
    ARIMA, Prophet, and other advanced forecasting methods.
    """)
    
    # Sidebar
    st.sidebar.header("Configuration")
    
    # Data generation parameters
    st.sidebar.subheader("Data Generation")
    n_points = st.sidebar.slider("Number of Points", 50, 500, config['data']['synthetic']['n_points'])
    trend_strength = st.sidebar.slider("Trend Strength", 0.0, 0.2, config['data']['synthetic']['trend_strength'])
    seasonal_amplitude = st.sidebar.slider("Seasonal Amplitude", 0.0, 5.0, config['data']['synthetic']['seasonal_amplitude'])
    noise_scale = st.sidebar.slider("Noise Scale", 0.0, 2.0, config['data']['synthetic']['noise_scale'])
    
    # Model selection
    st.sidebar.subheader("Model Selection")
    use_state_space = st.sidebar.checkbox("State Space Model", value=True)
    use_arima = st.sidebar.checkbox("ARIMA", value=True)
    use_prophet = st.sidebar.checkbox("Prophet", value=True)
    
    # Forecast parameters
    st.sidebar.subheader("Forecasting")
    forecast_steps = st.sidebar.slider("Forecast Steps", 5, 50, 12)
    
    # Generate data button
    if st.sidebar.button("Generate New Data", type="primary"):
        with st.spinner("Generating time series data..."):
            # Update config with sidebar values
            config['data']['synthetic']['n_points'] = n_points
            config['data']['synthetic']['trend_strength'] = trend_strength
            config['data']['synthetic']['seasonal_amplitude'] = seasonal_amplitude
            config['data']['synthetic']['noise_scale'] = noise_scale
            
            series_data, generator = generate_data(config)
            st.session_state.current_data = series_data
            st.session_state.data_generated = True
            st.session_state.models_fitted = False
            st.session_state.fitted_models = {}
        
        st.success("Data generated successfully!")
    
    # Main content
    if st.session_state.data_generated:
        # Data selection
        st.subheader("📊 Data Selection")
        series_names = list(st.session_state.current_data.keys())
        selected_series = st.selectbox("Select Time Series", series_names)
        
        if selected_series:
            time, observed, components = st.session_state.current_data[selected_series]
            
            # Display data info
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Data Points", len(time))
            with col2:
                st.metric("Mean Value", f"{np.mean(observed):.2f}")
            with col3:
                st.metric("Std Deviation", f"{np.std(observed):.2f}")
            
            # Plot original data
            st.subheader("📈 Original Time Series")
            fig_data = go.Figure()
            fig_data.add_trace(go.Scatter(
                x=time,
                y=observed,
                mode='lines',
                name='Observed',
                line=dict(color='blue', width=2)
            ))
            fig_data.update_layout(
                title=f"{selected_series.replace('_', ' ').title()} Time Series",
                xaxis_title="Time",
                yaxis_title="Value",
                template='plotly_white',
                height=400
            )
            st.plotly_chart(fig_data, use_container_width=True)
            
            # Plot components
            st.subheader("🔍 Time Series Components")
            fig_components = create_components_plot(time, components, "Decomposed Components")
            st.plotly_chart(fig_components, use_container_width=True)
            
            # Model fitting
            st.subheader("🤖 Model Fitting")
            
            if st.button("Fit Models", type="primary"):
                with st.spinner("Fitting models..."):
                    models = fit_models(observed, config)
                    st.session_state.fitted_models = models
                    st.session_state.models_fitted = True
                
                st.success(f"Successfully fitted {len(models)} models!")
            
            # Display model results
            if st.session_state.models_fitted and st.session_state.fitted_models:
                st.subheader("📊 Model Results")
                
                # Model comparison
                forecasts = {}
                for model_name, model in st.session_state.fitted_models.items():
                    try:
                        forecast, ci = model.predict(forecast_steps)
                        forecasts[model_name] = (forecast, ci)
                    except Exception as e:
                        st.error(f"Error predicting with {model_name}: {e}")
                
                if forecasts:
                    st.subheader("🔮 Forecast Comparison")
                    fig_forecast = create_forecast_plot(time, observed, forecasts, "Model Forecasts")
                    st.plotly_chart(fig_forecast, use_container_width=True)
                    
                    # Model metrics
                    st.subheader("📈 Model Metrics")
                    metrics_data = []
                    for model_name, model in st.session_state.fitted_models.items():
                        try:
                            if hasattr(model, 'result') and hasattr(model.result, 'aic'):
                                aic = model.result.aic
                            elif hasattr(model, 'model') and hasattr(model.model, 'aic'):
                                aic = model.model.aic()
                            else:
                                aic = "N/A"
                            
                            metrics_data.append({
                                'Model': model_name,
                                'AIC': aic,
                                'Status': 'Fitted'
                            })
                        except:
                            metrics_data.append({
                                'Model': model_name,
                                'AIC': 'N/A',
                                'Status': 'Error'
                            })
                    
                    metrics_df = pd.DataFrame(metrics_data)
                    st.dataframe(metrics_df, use_container_width=True)
                
                # Anomaly detection
                st.subheader("🚨 Anomaly Detection")
                
                if st.button("Detect Anomalies"):
                    with st.spinner("Detecting anomalies..."):
                        detector = AnomalyDetector(contamination=0.1)
                        detector.fit(observed)
                        anomaly_indices = detector.get_anomaly_indices(observed)
                        
                        # Plot anomalies
                        fig_anomalies = go.Figure()
                        
                        # Normal points
                        normal_mask = np.ones(len(observed), dtype=bool)
                        normal_mask[anomaly_indices] = False
                        
                        fig_anomalies.add_trace(go.Scatter(
                            x=time[normal_mask],
                            y=observed[normal_mask],
                            mode='lines',
                            name='Normal',
                            line=dict(color='blue', width=2)
                        ))
                        
                        # Anomalies
                        if len(anomaly_indices) > 0:
                            fig_anomalies.add_trace(go.Scatter(
                                x=time[anomaly_indices],
                                y=observed[anomaly_indices],
                                mode='markers',
                                name='Anomalies',
                                marker=dict(color='red', size=8)
                            ))
                        
                        fig_anomalies.update_layout(
                            title="Anomaly Detection Results",
                            xaxis_title="Time",
                            yaxis_title="Value",
                            template='plotly_white',
                            height=400
                        )
                        
                        st.plotly_chart(fig_anomalies, use_container_width=True)
                        
                        st.info(f"Detected {len(anomaly_indices)} anomalies out of {len(observed)} points")
    
    else:
        st.info("👈 Please generate data using the sidebar controls to begin analysis.")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    **Time Series Analysis Dashboard** - Built with Streamlit, Plotly, and modern Python libraries.
    
    Features:
    - State Space Models (Kalman Filter)
    - ARIMA with automatic parameter selection
    - Prophet forecasting
    - Anomaly detection
    - Interactive visualizations
    """)


if __name__ == "__main__":
    main()
