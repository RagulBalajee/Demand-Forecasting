#!/usr/bin/env python3
"""
Test script to verify installation of all dependencies
"""

def test_imports():
    """Test all required imports"""
    print("🧪 Testing imports...")
    
    try:
        import pandas as pd
        print("✅ pandas imported successfully")
    except ImportError as e:
        print(f"❌ pandas import failed: {e}")
        return False
    
    try:
        import numpy as np
        print("✅ numpy imported successfully")
    except ImportError as e:
        print(f"❌ numpy import failed: {e}")
        return False
    
    try:
        import sklearn
        print("✅ scikit-learn imported successfully")
    except ImportError as e:
        print(f"❌ scikit-learn import failed: {e}")
        return False
    
    try:
        import xgboost as xgb
        print("✅ xgboost imported successfully")
    except ImportError as e:
        print(f"❌ xgboost import failed: {e}")
        return False
    
    try:
        import lightgbm as lgb
        print("✅ lightgbm imported successfully")
    except ImportError as e:
        print(f"❌ lightgbm import failed: {e}")
        return False
    
    try:
        import plotly
        print("✅ plotly imported successfully")
    except ImportError as e:
        print(f"❌ plotly import failed: {e}")
        return False
    
    try:
        import streamlit as st
        print("✅ streamlit imported successfully")
    except ImportError as e:
        print(f"❌ streamlit import failed: {e}")
        return False
    
    try:
        from prophet import Prophet
        print("✅ prophet imported successfully")
    except ImportError as e:
        print(f"❌ prophet import failed: {e}")
        return False
    
    return True

def test_custom_modules():
    """Test our custom modules"""
    print("\n🔧 Testing custom modules...")
    
    try:
        from data.loader import DataLoader
        print("✅ DataLoader imported successfully")
    except ImportError as e:
        print(f"❌ DataLoader import failed: {e}")
        return False
    
    try:
        from data.features import FeatureEngineer
        print("✅ FeatureEngineer imported successfully")
    except ImportError as e:
        print(f"❌ FeatureEngineer import failed: {e}")
        return False
    
    try:
        from models.forecasting import DemandForecaster
        print("✅ DemandForecaster imported successfully")
    except ImportError as e:
        print(f"❌ DemandForecaster import failed: {e}")
        return False
    
    try:
        from models.product_analysis import ProductAnalyzer
        print("✅ ProductAnalyzer imported successfully")
    except ImportError as e:
        print(f"❌ ProductAnalyzer import failed: {e}")
        return False
    
    try:
        from utils.visualization import RetailVisualizer
        print("✅ RetailVisualizer imported successfully")
    except ImportError as e:
        print(f"❌ RetailVisualizer import failed: {e}")
        return False
    
    try:
        from utils.metrics import RetailMetrics
        print("✅ RetailMetrics imported successfully")
    except ImportError as e:
        print(f"❌ RetailMetrics import failed: {e}")
        return False
    
    return True

def test_basic_functionality():
    """Test basic functionality"""
    print("\n🚀 Testing basic functionality...")
    
    try:
        from data.loader import DataLoader
        from data.features import FeatureEngineer
        from models.forecasting import DemandForecaster
        from models.product_analysis import ProductAnalyzer
        
        # Initialize components
        data_loader = DataLoader()
        feature_engineer = FeatureEngineer()
        forecaster = DemandForecaster()
        product_analyzer = ProductAnalyzer()
        
        print("✅ All components initialized successfully")
        
        # Load sample data
        sales_df, store_df = data_loader.load_sample_data()
        print(f"✅ Sample data loaded: {len(sales_df)} records")
        
        # Test preprocessing
        processed_df = data_loader.preprocess_data(sales_df, store_df)
        print(f"✅ Data preprocessing completed: {processed_df.shape}")
        
        # Test feature engineering
        feature_df = feature_engineer.engineer_all_features(processed_df)
        print(f"✅ Feature engineering completed: {len(feature_engineer.feature_columns)} features")
        
        # Test product analysis
        product_metrics = product_analyzer.calculate_product_metrics(sales_df)
        print(f"✅ Product analysis completed: {len(product_metrics)} products")
        
        print("✅ All basic functionality tests passed!")
        return True
        
    except Exception as e:
        print(f"❌ Basic functionality test failed: {e}")
        return False

def main():
    print("🛍️ AI-Powered Retail Demand Forecasting - Installation Test")
    print("=" * 70)
    
    # Test imports
    imports_ok = test_imports()
    
    # Test custom modules
    modules_ok = test_custom_modules()
    
    # Test basic functionality
    functionality_ok = test_basic_functionality()
    
    print("\n" + "=" * 70)
    print("📋 TEST RESULTS SUMMARY")
    print("=" * 70)
    
    if imports_ok and modules_ok and functionality_ok:
        print("🎉 ALL TESTS PASSED!")
        print("✅ The system is ready to use.")
        print("\n🚀 Next steps:")
        print("   1. Run demo: python demo.py")
        print("   2. Start web app: streamlit run app.py")
    else:
        print("❌ SOME TESTS FAILED!")
        print("🔧 Please check the error messages above and install missing dependencies.")
        print("\n💡 Installation help:")
        print("   pip install -r requirements.txt")
    
    print("=" * 70)

if __name__ == "__main__":
    main() 