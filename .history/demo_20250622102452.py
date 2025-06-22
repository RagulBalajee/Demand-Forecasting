#!/usr/bin/env python3
"""
Demo script for AI-Powered Retail Demand Forecasting System
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Import our custom modules
from data.loader import DataLoader
from data.features import FeatureEngineer
from models.forecasting import DemandForecaster
from models.product_analysis import ProductAnalyzer
from utils.visualization import RetailVisualizer
from utils.metrics import RetailMetrics

def main():
    print("🛍️ AI-Powered Retail Demand Forecasting System")
    print("=" * 60)
    
    # Initialize components
    print("\n📦 Initializing components...")
    data_loader = DataLoader()
    feature_engineer = FeatureEngineer()
    forecaster = DemandForecaster()
    product_analyzer = ProductAnalyzer()
    visualizer = RetailVisualizer()
    metrics = RetailMetrics()
    
    # Load sample data
    print("\n📊 Loading sample data...")
    sales_df, store_df = data_loader.load_sample_data()
    print(f"✅ Loaded {len(sales_df)} sales records")
    
    # Preprocess data
    print("\n🔧 Preprocessing data...")
    processed_df = data_loader.preprocess_data(sales_df, store_df)
    
    # Engineer features
    print("\n⚙️ Engineering features...")
    feature_df = feature_engineer.engineer_all_features(processed_df)
    
    # Train forecasting models
    print("\n🤖 Training forecasting models...")
    X, y = forecaster.prepare_data(feature_df)
    results = forecaster.train_models(X, y, ['random_forest', 'xgboost'])
    print(f"🏆 Best model: {forecaster.best_model_name}")
    
    # Generate forecast
    print("\n🔮 Generating forecast...")
    future_dates = pd.date_range(start=sales_df['Date'].max() + timedelta(days=1), periods=30, freq='D')
    forecast_df = forecaster.predict_future(feature_df, future_dates, feature_engineer)
    
    # Product analysis
    print("\n📈 Analyzing products...")
    product_metrics = product_analyzer.calculate_product_metrics(sales_df)
    classified_products = product_analyzer.classify_products(product_metrics)
    
    # Stock recommendations
    print("\n📦 Generating stock recommendations...")
    stock_recommendations = product_analyzer.calculate_stock_recommendations(classified_products)
    
    # Calculate metrics
    print("\n📊 Calculating performance metrics...")
    business_metrics = metrics.calculate_business_metrics(sales_df)
    
    # Display results
    print("\n" + "=" * 60)
    print("📋 SYSTEM DEMO RESULTS")
    print("=" * 60)
    
    print(f"\n💰 Business Performance:")
    print(f"   Total Revenue: ${business_metrics['Total_Revenue']:,.0f}")
    print(f"   Average Daily Revenue: ${business_metrics['Avg_Daily_Revenue']:,.0f}")
    
    print(f"\n🤖 Model Performance:")
    for model_name, metrics_dict in results.items():
        print(f"   {model_name.upper()}: R² = {metrics_dict['r2']:.3f}")
    
    print(f"\n🏷️ Product Categories:")
    category_counts = classified_products['Product_Category'].value_counts()
    for category, count in category_counts.items():
        print(f"   {category}: {count} products")
    
    print(f"\n🔮 Forecast Summary:")
    total_forecast_sales = forecast_df['Sales'].sum()
    print(f"   Total Forecasted Sales: {total_forecast_sales:,.0f}")
    
    print("\n" + "=" * 60)
    print("✅ Demo completed successfully!")
    print("🚀 To run the full web application, use: streamlit run app.py")
    print("=" * 60)

if __name__ == "__main__":
    main() 