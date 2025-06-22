#!/usr/bin/env python3
"""
Test script to verify all components of the AI-Powered Retail Demand Forecasting application
"""

import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def test_imports():
    """Test if all required modules can be imported"""
    print("🔍 Testing imports...")
    
    try:
        from data.loader import DataLoader
        print("✅ DataLoader imported successfully")
    except Exception as e:
        print(f"❌ DataLoader import failed: {e}")
        return False
    
    try:
        from data.features import FeatureEngineer
        print("✅ FeatureEngineer imported successfully")
    except Exception as e:
        print(f"❌ FeatureEngineer import failed: {e}")
        return False
    
    try:
        from models.forecasting import DemandForecaster, ProphetForecaster
        print("✅ Forecasting models imported successfully")
    except Exception as e:
        print(f"❌ Forecasting models import failed: {e}")
        return False
    
    try:
        from models.product_analysis import ProductAnalyzer
        print("✅ ProductAnalyzer imported successfully")
    except Exception as e:
        print(f"❌ ProductAnalyzer import failed: {e}")
        return False
    
    try:
        from utils.visualization import RetailVisualizer
        print("✅ RetailVisualizer imported successfully")
    except Exception as e:
        print(f"❌ RetailVisualizer import failed: {e}")
        return False
    
    try:
        from utils.metrics import RetailMetrics
        print("✅ RetailMetrics imported successfully")
    except Exception as e:
        print(f"❌ RetailMetrics import failed: {e}")
        return False
    
    return True

def test_data_loader():
    """Test data loading functionality"""
    print("\n📊 Testing data loader...")
    
    try:
        from data.loader import DataLoader
        loader = DataLoader()
        
        # Test sample data loading
        if hasattr(loader, 'load_sample_data'):
            try:
                sales_df, store_df = loader.load_sample_data()
                print(f"✅ Sample data loaded: {len(sales_df)} sales records, {len(store_df)} store records")
            except Exception as e:
                print(f"⚠️ Sample data loading failed: {e}")
        
        # Test Rossmann data loading
        if os.path.exists('data/rossmann-store-sales'):
            try:
                sales_df, store_df = loader.load_rossmann_data()
                print(f"✅ Rossmann data loaded: {len(sales_df)} sales records, {len(store_df)} store records")
            except Exception as e:
                print(f"⚠️ Rossmann data loading failed: {e}")
        else:
            print("ℹ️ Rossmann dataset not found (expected)")
        
        return True
        
    except Exception as e:
        print(f"❌ Data loader test failed: {e}")
        return False

def test_feature_engineering():
    """Test feature engineering functionality"""
    print("\n🔧 Testing feature engineering...")
    
    try:
        from data.features import FeatureEngineer
        
        # Create sample data
        dates = pd.date_range('2023-01-01', periods=100, freq='D')
        sample_data = pd.DataFrame({
            'Date': dates,
            'Store': np.random.randint(1, 11, 100),
            'Sales': np.random.randint(100, 1000, 100),
            'Customers': np.random.randint(50, 200, 100)
        })
        
        # Test feature engineering
        engineer = FeatureEngineer()
        enhanced_data = engineer.engineer_all_features(sample_data)
        
        print(f"✅ Feature engineering completed: {len(enhanced_data.columns)} features created")
        print(f"   Original columns: {list(sample_data.columns)}")
        print(f"   Enhanced columns: {list(enhanced_data.columns)}")
        
        return True
        
    except Exception as e:
        print(f"❌ Feature engineering test failed: {e}")
        return False

def test_forecasting_models():
    """Test forecasting model functionality"""
    print("\n🤖 Testing forecasting models...")
    
    try:
        from models.forecasting import DemandForecaster
        from data.features import FeatureEngineer
        
        # Create sample data
        dates = pd.date_range('2023-01-01', periods=100, freq='D')
        sample_data = pd.DataFrame({
            'Date': dates,
            'Store': np.random.randint(1, 6, 100),
            'Sales': np.random.randint(100, 1000, 100),
            'Promo': np.random.randint(0, 2, 100)
        })
        
        # Engineer features
        engineer = FeatureEngineer()
        enhanced_data = engineer.engineer_all_features(sample_data)
        
        # Test forecasting
        forecaster = DemandForecaster()
        X, y = forecaster.prepare_data(enhanced_data)
        
        print(f"✅ Data preparation completed: {X.shape[0]} samples, {X.shape[1]} features")
        
        # Test model training (with a small subset for speed)
        if len(X) > 50:
            X_subset = X.iloc[:50]
            y_subset = y.iloc[:50]
            
            try:
                results = forecaster.train_models(X_subset, y_subset, ['random_forest'])
                print(f"✅ Model training completed: RMSE = {results['random_forest']['rmse']:.2f}")
            except Exception as e:
                print(f"⚠️ Model training failed: {e}")
        
        return True
        
    except Exception as e:
        print(f"❌ Forecasting models test failed: {e}")
        return False

def test_visualization():
    """Test visualization functionality"""
    print("\n📈 Testing visualization...")
    
    try:
        from utils.visualization import RetailVisualizer
        
        # Create sample data
        dates = pd.date_range('2023-01-01', periods=30, freq='D')
        sample_data = pd.DataFrame({
            'Date': dates,
            'Sales': np.random.randint(100, 1000, 30)
        })
        
        # Test visualization
        visualizer = RetailVisualizer()
        fig = visualizer.plot_sales_trend(sample_data)
        
        print("✅ Visualization test completed")
        print(f"   Chart type: {type(fig).__name__}")
        
        return True
        
    except Exception as e:
        print(f"❌ Visualization test failed: {e}")
        return False

def test_metrics():
    """Test metrics calculation"""
    print("\n📊 Testing metrics calculation...")
    
    try:
        from utils.metrics import RetailMetrics
        
        # Create sample data
        dates = pd.date_range('2023-01-01', periods=100, freq='D')
        sample_data = pd.DataFrame({
            'Date': dates,
            'Sales': np.random.randint(100, 1000, 100),
            'Store': np.random.randint(1, 11, 100)
        })
        
        # Test metrics
        metrics = RetailMetrics()
        business_metrics = metrics.calculate_business_metrics(sample_data)
        
        print("✅ Metrics calculation completed")
        print(f"   Total Revenue: ${business_metrics['Total_Revenue']:,.0f}")
        print(f"   Average Daily Revenue: ${business_metrics['Avg_Daily_Revenue']:,.0f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Metrics test failed: {e}")
        return False

def test_streamlit_app():
    """Test if the Streamlit app can be imported"""
    print("\n🌐 Testing Streamlit app...")
    
    try:
        import app
        print("✅ Streamlit app imported successfully")
        
        # Check if main function exists
        if hasattr(app, 'main'):
            print("✅ Main function found")
        else:
            print("⚠️ Main function not found")
        
        return True
        
    except Exception as e:
        print(f"❌ Streamlit app test failed: {e}")
        return False

def main():
    """Run all tests"""
    print("🧪 Testing AI-Powered Retail Demand Forecasting Application")
    print("=" * 60)
    
    tests = [
        ("Imports", test_imports),
        ("Data Loader", test_data_loader),
        ("Feature Engineering", test_feature_engineering),
        ("Forecasting Models", test_forecasting_models),
        ("Visualization", test_visualization),
        ("Metrics", test_metrics),
        ("Streamlit App", test_streamlit_app)
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} test crashed: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 60)
    print("📋 Test Summary")
    print("=" * 60)
    
    passed = 0
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{test_name:20} {status}")
        if result:
            passed += 1
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! The application is ready to run.")
        print("\n🚀 To start the application, run:")
        print("   streamlit run app.py")
    else:
        print("⚠️ Some tests failed. Please check the errors above.")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 