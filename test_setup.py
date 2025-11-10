#!/usr/bin/env python3
"""
Test script to verify the time series analysis project setup.

This script tests the basic functionality and ensures all components work correctly.
"""

import sys
from pathlib import Path
import numpy as np
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

def test_imports():
    """Test that all required modules can be imported."""
    print("Testing imports...")
    
    try:
        # Core libraries
        import numpy as np
        import pandas as pd
        import matplotlib.pyplot as plt
        import seaborn as sns
        import plotly.graph_objects as go
        
        # Time series libraries
        import statsmodels
        from statsmodels.tsa.statespace.structural import UnobservedComponents
        from statsmodels.tsa.arima.model import ARIMA
        
        # Additional libraries
        # import pmdarima  # Optional dependency
        # import prophet  # Optional dependency
        from sklearn.ensemble import IsolationForest
        import torch
        
        # Project modules
        sys.path.append(str(Path(__file__).parent / "src"))
        from data_generator import TimeSeriesGenerator, TimeSeriesConfig
        from models import StateSpaceModel, ARIMAModel, AnomalyDetector
        from visualization import TimeSeriesVisualizer, InteractiveVisualizer
        
        print("✓ All imports successful!")
        return True
        
    except ImportError as e:
        print(f"✗ Import error: {e}")
        return False


def test_data_generation():
    """Test data generation functionality."""
    print("\nTesting data generation...")
    
    try:
        from data_generator import TimeSeriesGenerator, TimeSeriesConfig
        
        # Test basic generation
        config = TimeSeriesConfig(n_points=50, random_seed=42)
        generator = TimeSeriesGenerator(config)
        time, observed, components = generator.generate_simple_series()
        
        assert len(time) == 50
        assert len(observed) == 50
        assert 'trend' in components
        assert 'seasonal' in components
        assert 'noise' in components
        
        print("✓ Data generation successful!")
        return True
        
    except Exception as e:
        print(f"✗ Data generation error: {e}")
        return False


def test_models():
    """Test model fitting functionality."""
    print("\nTesting model fitting...")
    
    try:
        from data_generator import TimeSeriesGenerator, TimeSeriesConfig
        from models import StateSpaceModel, ARIMAModel
        
        # Generate test data
        config = TimeSeriesConfig(n_points=50, random_seed=42)
        generator = TimeSeriesGenerator(config)
        time, observed, components = generator.generate_simple_series()
        
        # Test State Space Model
        ssm = StateSpaceModel()
        ssm.fit(observed)
        forecast, ci = ssm.predict(12)
        
        assert len(forecast) == 12
        assert ci.shape == (12, 2)
        
        # Test ARIMA Model
        arima = ARIMAModel()
        arima.fit(observed)
        forecast, ci = arima.predict(12)
        
        assert len(forecast) == 12
        assert ci.shape == (12, 2)
        
        print("✓ Model fitting successful!")
        return True
        
    except Exception as e:
        print(f"✗ Model fitting error: {e}")
        return False


def test_visualization():
    """Test visualization functionality."""
    print("\nTesting visualization...")
    
    try:
        from data_generator import TimeSeriesGenerator, TimeSeriesConfig
        from visualization import TimeSeriesVisualizer
        
        # Generate test data
        config = TimeSeriesConfig(n_points=50, random_seed=42)
        generator = TimeSeriesGenerator(config)
        time, observed, components = generator.generate_simple_series()
        
        # Test visualization
        visualizer = TimeSeriesVisualizer()
        fig = visualizer.plot_time_series(time, observed, title="Test Plot")
        
        assert fig is not None
        
        print("✓ Visualization successful!")
        return True
        
    except Exception as e:
        print(f"✗ Visualization error: {e}")
        return False


def run_basic_example():
    """Run a basic example to demonstrate functionality."""
    print("\nRunning basic example...")
    
    try:
        from data_generator import TimeSeriesGenerator, TimeSeriesConfig
        from models import StateSpaceModel
        from visualization import TimeSeriesVisualizer
        
        # Generate data
        config = TimeSeriesConfig(n_points=100, random_seed=42)
        generator = TimeSeriesGenerator(config)
        time, observed, components = generator.generate_simple_series()
        
        # Fit model
        ssm = StateSpaceModel()
        ssm.fit(observed)
        forecast, ci = ssm.predict(12)
        
        # Create visualization
        visualizer = TimeSeriesVisualizer()
        fig = visualizer.plot_forecast(time, observed, forecast, ci, 
                                     title="Basic Example - State Space Model")
        
        # Save plot
        output_dir = Path("output")
        output_dir.mkdir(exist_ok=True)
        fig.savefig(output_dir / "test_plot.png", dpi=150, bbox_inches='tight')
        
        print("✓ Basic example completed successfully!")
        print(f"  - Generated {len(time)} data points")
        print(f"  - Fitted state space model (AIC: {ssm.result.aic:.2f})")
        print(f"  - Generated {len(forecast)}-step forecast")
        print(f"  - Saved plot to: {output_dir / 'test_plot.png'}")
        
        return True
        
    except Exception as e:
        print(f"✗ Basic example error: {e}")
        return False


def main():
    """Main test function."""
    print("Time Series Analysis Project - Test Suite")
    print("=" * 50)
    
    # Run tests
    tests = [
        test_imports,
        test_data_generation,
        test_models,
        test_visualization,
        run_basic_example
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
    
    # Summary
    print("\n" + "=" * 50)
    print(f"Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! The project is ready to use.")
        print("\nNext steps:")
        print("1. Run the Streamlit dashboard: streamlit run app.py")
        print("2. Explore the example notebook: notebooks/example_analysis.ipynb")
        print("3. Run the modernized analysis: python modernized_analysis.py")
    else:
        print("❌ Some tests failed. Please check the error messages above.")
        print("Make sure all dependencies are installed: pip install -r requirements.txt")
    
    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
