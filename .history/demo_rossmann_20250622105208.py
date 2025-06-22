#!/usr/bin/env python3
"""
Demo script for Rossmann Store Sales dataset
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
    print("🛍️ AI-Powered Retail Demand Forecasting - Rossmann Dataset")
    print("=" * 70)
    
    # Initialize components
    print("\n📦 Initializing components...")
    data_loader = DataLoader()
    feature_engineer = FeatureEngineer()
    forecaster = DemandForecaster()
    product_analyzer = ProductAnalyzer()
    visualizer = RetailVisualizer()
    metrics = RetailMetrics()
    
    # Load Rossmann dataset
    print("\n📊 Loading Rossmann dataset...")
    try:
        sales_df, store_df = data_loader.load_rossmann_data()
        print(f"✅ Loaded {len(sales_df):,} sales records")
        print(f"   🏪 Stores: {sales_df['Store'].nunique()}")
        print(f"   📅 Date range: {sales_df['Date'].min()} to {sales_df['Date'].max()}")
        print(f"   💰 Total sales: {sales_df['Sales'].sum():,}")
        
    except Exception as e:
        print(f"❌ Error loading Rossmann dataset: {e}")
        print("💡 Please run 'python setup_dataset.py' first to organize your dataset.")
        return
    
    # Preprocess data
    print("\n🔧 Preprocessing data...")
    processed_df = data_loader.preprocess_data(sales_df, store_df)
    print(f"✅ Preprocessed data shape: {processed_df.shape}")
    
    # Engineer features
    print("\n⚙️ Engineering features...")
    feature_df = feature_engineer.engineer_all_features(processed_df)
    print(f"✅ Engineered {len(feature_engineer.feature_columns)} features")
    
    # Train forecasting models
    print("\n🤖 Training forecasting models...")
    X, y = forecaster.prepare_data(feature_df)
    results = forecaster.train_models(X, y, ['random_forest', 'xgboost'])
    print(f"🏆 Best model: {forecaster.best_model_name}")
    
    # Generate forecast
    print("\n🔮 Generating forecast...")
    # Use test data dates for forecasting
    test_df = data_loader.load_rossmann_test_data()
    future_dates = test_df['Date'].unique()
    forecast_df = forecaster.predict_future(feature_df, future_dates, feature_engineer)
    print(f"✅ Generated forecast for {len(future_dates)} dates")
    
    # Product analysis (using stores as products for Rossmann)
    print("\n📈 Analyzing store performance...")
    # Aggregate by store and date for store-level analysis
    store_sales = sales_df.groupby(['Store', 'Date'])['Sales'].sum().reset_index()
    store_metrics = product_analyzer.calculate_product_metrics(store_sales, product_col='Store')
    classified_stores = product_analyzer.classify_products(store_metrics)
    
    # Stock recommendations (store-level)
    print("\n📦 Generating store-level recommendations...")
    stock_recommendations = product_analyzer.calculate_stock_recommendations(classified_stores)
    
    # Calculate metrics
    print("\n📊 Calculating performance metrics...")
    business_metrics = metrics.calculate_business_metrics(sales_df)
    store_portfolio_metrics = metrics.calculate_product_metrics(store_sales, product_col='Store')
    seasonal_metrics = metrics.calculate_seasonal_metrics(sales_df)
    inventory_metrics = metrics.calculate_inventory_metrics(stock_recommendations)
    
    # Display results
    print("\n" + "=" * 70)
    print("📋 ROSSMANN DATASET ANALYSIS RESULTS")
    print("=" * 70)
    
    # Business metrics
    print(f"\n💰 Business Performance:")
    print(f"   Total Revenue: ${business_metrics['Total_Revenue']:,.0f}")
    print(f"   Average Daily Revenue: ${business_metrics['Avg_Daily_Revenue']:,.0f}")
    print(f"   Revenue Growth Rate: {business_metrics['Revenue_Growth_Rate']:.1f}%")
    if business_metrics['Total_Customers'] > 0:
        print(f"   Total Customers: {business_metrics['Total_Customers']:,.0f}")
        print(f"   Average Transaction Value: ${business_metrics['Avg_Transaction_Value']:.2f}")
    
    # Store metrics
    print(f"\n🏪 Store Portfolio:")
    print(f"   Total Stores: {store_portfolio_metrics['Number_of_Products']}")
    print(f"   Sales Concentration (Top 20%): {store_portfolio_metrics['Sales_Concentration_Top20']:.1%}")
    print(f"   Store Sales CV: {store_portfolio_metrics['Product_Sales_CV']:.2f}")
    
    # Seasonal metrics
    print(f"\n🌤️ Seasonal Analysis:")
    print(f"   Peak Season: {seasonal_metrics['Peak_Season']}")
    print(f"   Peak/Off-Peak Ratio: {seasonal_metrics['Peak_Off_Peak_Ratio']:.2f}")
    print(f"   Seasonal Concentration: {seasonal_metrics['Seasonal_Concentration']:.1%}")
    
    # Inventory metrics
    print(f"\n📦 Store Management:")
    print(f"   Total Inventory Value: ${inventory_metrics['Total_Inventory_Value']:,.0f}")
    print(f"   Average Stock Turnover: {inventory_metrics['Avg_Stock_Turnover']:.1f}")
    print(f"   Stock-out Risk: {inventory_metrics['Stock_Out_Risk']:.1%}")
    print(f"   Overstock Risk: {inventory_metrics['Overstock_Risk']:.1%}")
    
    # Model performance
    print(f"\n🤖 Model Performance:")
    for model_name, metrics_dict in results.items():
        print(f"   {model_name.upper()}:")
        print(f"     MAE: {metrics_dict['mae']:.2f}")
        print(f"     R²: {metrics_dict['r2']:.3f}")
        print(f"     CV MAE: {metrics_dict['cv_mae']:.2f}")
    
    # Store categories
    category_counts = classified_stores['Product_Category'].value_counts()
    print(f"\n🏷️ Store Categories:")
    for category, count in category_counts.items():
        print(f"   {category}: {count} stores ({count/len(classified_stores)*100:.1f}%)")
    
    # Top stores
    print(f"\n🏆 Top 5 Stores by Sales:")
    top_stores = store_metrics.nlargest(5, 'Total_Sales')
    for _, row in top_stores.iterrows():
        print(f"   Store {int(row['Store'])}: {row['Total_Sales']:,.0f} sales")
    
    # Forecast summary
    print(f"\n🔮 Forecast Summary:")
    total_forecast_sales = forecast_df['Sales'].sum()
    avg_daily_forecast = forecast_df.groupby('Date')['Sales'].sum().mean()
    print(f"   Total Forecasted Sales: {total_forecast_sales:,.0f}")
    print(f"   Average Daily Forecast: {avg_daily_forecast:,.0f}")
    
    # Promotional effectiveness
    if business_metrics['Promo_Effectiveness'] > 0:
        print(f"\n🎯 Promotional Analysis:")
        print(f"   Promo Frequency: {business_metrics['Promo_Frequency']:.1%}")
        print(f"   Promo Effectiveness: {business_metrics['Promo_Effectiveness']:.2f}x")
    
    print("\n" + "=" * 70)
    print("✅ Rossmann dataset analysis completed successfully!")
    print("🚀 To run the full web application with this dataset:")
    print("   streamlit run app.py")
    print("=" * 70)

if __name__ == "__main__":
    main() 