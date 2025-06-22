#!/usr/bin/env python3
"""
Quick demo script for Rossmann Store Sales dataset - Key Analysis
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Import our custom modules
from data.loader import DataLoader
from data.features import FeatureEngineer
from models.product_analysis import ProductAnalyzer
from utils.metrics import RetailMetrics

def main():
    print("🛍️ Quick Rossmann Dataset Analysis")
    print("=" * 60)
    
    # Initialize components
    print("\n📦 Initializing components...")
    data_loader = DataLoader()
    feature_engineer = FeatureEngineer()
    product_analyzer = ProductAnalyzer()
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
        return
    
    # Show dataset structure
    print(f"\n📋 Dataset Structure:")
    print(f"   Sales columns: {list(sales_df.columns)}")
    print(f"   Store columns: {list(store_df.columns)}")
    
    # Basic statistics
    print(f"\n📊 Basic Statistics:")
    print(f"   Average daily sales: {sales_df['Sales'].mean():.0f}")
    print(f"   Sales std dev: {sales_df['Sales'].std():.0f}")
    print(f"   Max daily sales: {sales_df['Sales'].max():,}")
    print(f"   Min daily sales: {sales_df['Sales'].min():,}")
    
    # Store analysis
    print(f"\n🏪 Store Analysis:")
    store_sales = sales_df.groupby('Store')['Sales'].sum().sort_values(ascending=False)
    print(f"   Top 5 stores by total sales:")
    for i, (store, sales) in enumerate(store_sales.head().items(), 1):
        print(f"     {i}. Store {store}: {sales:,.0f} sales")
    
    print(f"   Bottom 5 stores by total sales:")
    for i, (store, sales) in enumerate(store_sales.tail().items(), 1):
        print(f"     {i}. Store {store}: {sales:,.0f} sales")
    
    # Time analysis
    print(f"\n📅 Time Analysis:")
    daily_sales = sales_df.groupby('Date')['Sales'].sum()
    print(f"   Average daily revenue: ${daily_sales.mean():,.0f}")
    print(f"   Best day: {daily_sales.idxmax()} (${daily_sales.max():,.0f})")
    print(f"   Worst day: {daily_sales.idxmin()} (${daily_sales.min():,.0f})")
    
    # Monthly trends
    monthly_sales = sales_df.groupby(sales_df['Date'].dt.to_period('M'))['Sales'].sum()
    print(f"\n📈 Monthly Trends:")
    print(f"   Best month: {monthly_sales.idxmax()} (${monthly_sales.max():,.0f})")
    print(f"   Worst month: {monthly_sales.idxmin()} (${monthly_sales.min():,.0f})")
    
    # Promotional analysis
    if 'Promo' in sales_df.columns:
        print(f"\n🎯 Promotional Analysis:")
        promo_sales = sales_df.groupby('Promo')['Sales'].agg(['mean', 'count'])
        print(f"   Average sales with promo: ${promo_sales.loc[1, 'mean']:,.0f}")
        print(f"   Average sales without promo: ${promo_sales.loc[0, 'mean']:,.0f}")
        promo_effectiveness = promo_sales.loc[1, 'mean'] / promo_sales.loc[0, 'mean']
        print(f"   Promo effectiveness: {promo_effectiveness:.2f}x")
    
    # Holiday analysis
    if 'StateHoliday' in sales_df.columns:
        print(f"\n🎄 Holiday Analysis:")
        holiday_sales = sales_df.groupby('StateHoliday')['Sales'].mean()
        print(f"   Average sales on holidays: ${holiday_sales.mean():,.0f}")
        print(f"   Average sales on regular days: ${sales_df[sales_df['StateHoliday'] == 0]['Sales'].mean():,.0f}")
    
    # Store type analysis
    if 'StoreType' in store_df.columns:
        print(f"\n🏷️ Store Type Analysis:")
        store_type_sales = sales_df.merge(store_df[['Store', 'StoreType']], on='Store')
        type_analysis = store_type_sales.groupby('StoreType')['Sales'].agg(['mean', 'sum', 'count'])
        for store_type in type_analysis.index:
            print(f"   Type {store_type}: ${type_analysis.loc[store_type, 'mean']:,.0f} avg, "
                  f"${type_analysis.loc[store_type, 'sum']:,.0f} total")
    
    # Competition analysis
    if 'CompetitionDistance' in store_df.columns:
        print(f"\n🏁 Competition Analysis:")
        comp_analysis = store_df.groupby('StoreType')['CompetitionDistance'].agg(['mean', 'min', 'max'])
        for store_type in comp_analysis.index:
            print(f"   Type {store_type}: {comp_analysis.loc[store_type, 'mean']:.0f}m avg distance")
    
    # Calculate business metrics
    print(f"\n📊 Business Metrics:")
    business_metrics = metrics.calculate_business_metrics(sales_df)
    print(f"   Total Revenue: ${business_metrics['Total_Revenue']:,.0f}")
    print(f"   Average Daily Revenue: ${business_metrics['Avg_Daily_Revenue']:,.0f}")
    if business_metrics['Total_Customers'] > 0:
        print(f"   Total Customers: {business_metrics['Total_Customers']:,.0f}")
        print(f"   Average Transaction Value: ${business_metrics['Avg_Transaction_Value']:.2f}")
    
    # Seasonal analysis
    print(f"\n🌤️ Seasonal Analysis:")
    seasonal_metrics = metrics.calculate_seasonal_metrics(sales_df)
    print(f"   Peak Season: {seasonal_metrics['Peak_Season']}")
    print(f"   Peak/Off-Peak Ratio: {seasonal_metrics['Peak_Off_Peak_Ratio']:.2f}")
    print(f"   Seasonal Concentration: {seasonal_metrics['Seasonal_Concentration']:.1%}")
    
    # Store portfolio analysis
    print(f"\n🏪 Store Portfolio:")
    store_portfolio = metrics.calculate_product_metrics(sales_df.groupby(['Store', 'Date'])['Sales'].sum().reset_index(), product_col='Store')
    print(f"   Total Stores: {store_portfolio['Number_of_Products']}")
    print(f"   Sales Concentration (Top 20%): {store_portfolio['Sales_Concentration_Top20']:.1%}")
    print(f"   Store Sales CV: {store_portfolio['Product_Sales_CV']:.2f}")
    
    print("\n" + "=" * 60)
    print("✅ Quick Rossmann analysis completed!")
    print("🚀 For full forecasting and web app:")
    print("   streamlit run app.py")
    print("=" * 60)

if __name__ == "__main__":
    main() 