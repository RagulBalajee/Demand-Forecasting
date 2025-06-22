#!/usr/bin/env python3
"""
Demo script for AI-Powered Retail Demand Forecasting
Shows key features and capabilities of the application
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

def create_demo_data():
    """Create sample retail data for demonstration"""
    print("📊 Creating demo retail data...")
    
    # Generate dates
    start_date = datetime(2023, 1, 1)
    end_date = datetime(2023, 12, 31)
    dates = pd.date_range(start_date, end_date, freq='D')
    
    # Create sample data
    data = []
    stores = [1, 2, 3, 4, 5]
    products = ['Product A', 'Product B', 'Product C', 'Product D', 'Product E']
    
    for date in dates:
        for store in stores:
            for product in products:
                # Base sales with some seasonality
                base_sales = 100 + 50 * np.sin(2 * np.pi * date.dayofyear / 365)
                
                # Add some randomness
                sales = max(0, int(base_sales + np.random.normal(0, 20)))
                
                # Add promotional effect
                promo = np.random.choice([0, 1], p=[0.8, 0.2])
                if promo:
                    sales = int(sales * 1.3)  # 30% boost during promotions
                
                # Add weekend effect
                if date.weekday() >= 5:  # Weekend
                    sales = int(sales * 1.2)
                
                # Add holiday effect (simplified)
                if date.month == 12:  # December holiday season
                    sales = int(sales * 1.4)
                
                data.append({
                    'Date': date,
                    'Store': store,
                    'Product': product,
                    'Sales': sales,
                    'Customers': max(1, int(sales / 20)),
                    'Promo': promo
                })
    
    sales_df = pd.DataFrame(data)
    
    # Create store information
    store_df = pd.DataFrame({
        'Store': stores,
        'StoreType': ['A', 'B', 'A', 'C', 'B'],
        'Assortment': ['a', 'b', 'a', 'c', 'b'],
        'CompetitionDistance': [1000, 2000, 1500, 3000, 1200]
    })
    
    print(f"✅ Created {len(sales_df):,} sales records for {len(stores)} stores")
    print(f"   Date range: {sales_df['Date'].min()} to {sales_df['Date'].max()}")
    print(f"   Total sales: ${sales_df['Sales'].sum():,}")
    
    return sales_df, store_df

def demo_analysis():
    """Demonstrate key analysis capabilities"""
    print("\n🔍 Running demo analysis...")
    
    # Create demo data
    sales_df, store_df = create_demo_data()
    
    # Basic statistics
    print("\n📊 Basic Statistics:")
    print(f"   Average daily sales: ${sales_df['Sales'].mean():.0f}")
    print(f"   Total revenue: ${sales_df['Sales'].sum():,}")
    print(f"   Number of stores: {sales_df['Store'].nunique()}")
    print(f"   Number of products: {sales_df['Product'].nunique()}")
    
    # Store performance
    print("\n🏪 Store Performance:")
    store_sales = sales_df.groupby('Store')['Sales'].sum().sort_values(ascending=False)
    for store, sales in store_sales.items():
        print(f"   Store {store}: ${sales:,.0f}")
    
    # Product performance
    print("\n📦 Product Performance:")
    product_sales = sales_df.groupby('Product')['Sales'].sum().sort_values(ascending=False)
    for product, sales in product_sales.items():
        print(f"   {product}: ${sales:,.0f}")
    
    # Promotional analysis
    print("\n🎯 Promotional Analysis:")
    promo_sales = sales_df.groupby('Promo')['Sales'].agg(['mean', 'count'])
    print(f"   Average sales with promo: ${promo_sales.loc[1, 'mean']:,.0f}")
    print(f"   Average sales without promo: ${promo_sales.loc[0, 'mean']:,.0f}")
    promo_effectiveness = promo_sales.loc[1, 'mean'] / promo_sales.loc[0, 'mean']
    print(f"   Promo effectiveness: {promo_effectiveness:.2f}x")
    
    # Seasonal analysis
    print("\n🌤️ Seasonal Analysis:")
    monthly_sales = sales_df.groupby(sales_df['Date'].dt.month)['Sales'].sum()
    best_month = monthly_sales.idxmax()
    worst_month = monthly_sales.idxmin()
    print(f"   Best month: {best_month} (${monthly_sales[best_month]:,.0f})")
    print(f"   Worst month: {worst_month} (${monthly_sales[worst_month]:,.0f})")
    
    # Weekly pattern
    print("\n📅 Weekly Pattern:")
    weekly_sales = sales_df.groupby(sales_df['Date'].dt.dayofweek)['Sales'].mean()
    day_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    for i, day in enumerate(day_names):
        print(f"   {day}: ${weekly_sales[i]:.0f}")
    
    return sales_df, store_df

def demo_forecasting():
    """Demonstrate forecasting capabilities"""
    print("\n🔮 Forecasting Demo:")
    
    # Import forecasting components
    try:
        from data.features import FeatureEngineer
        from models.forecasting import DemandForecaster
        from utils.metrics import RetailMetrics
        
        # Get demo data
        sales_df, store_df = demo_analysis()
        
        # Feature engineering
        print("\n🔧 Feature Engineering:")
        engineer = FeatureEngineer()
        enhanced_df = engineer.engineer_all_features(sales_df)
        print(f"   Original features: {len(sales_df.columns)}")
        print(f"   Enhanced features: {len(enhanced_df.columns)}")
        print(f"   New features: {[col for col in enhanced_df.columns if col not in sales_df.columns]}")
        
        # Forecasting
        print("\n🤖 Model Training:")
        forecaster = DemandForecaster()
        X, y = forecaster.prepare_data(enhanced_df)
        print(f"   Training data: {X.shape[0]} samples, {X.shape[1]} features")
        
        # Train a simple model
        if len(X) > 100:
            X_subset = X.iloc[:100]
            y_subset = y.iloc[:100]
            
            results = forecaster.train_models(X_subset, y_subset, ['random_forest'])
            print(f"   Model trained successfully!")
            print(f"   RMSE: {results['random_forest']['rmse']:.2f}")
            print(f"   R² Score: {results['random_forest']['r2']:.3f}")
        
        # Business metrics
        print("\n📊 Business Metrics:")
        metrics = RetailMetrics()
        business_metrics = metrics.calculate_business_metrics(sales_df)
        print(f"   Total Revenue: ${business_metrics['Total_Revenue']:,.0f}")
        print(f"   Average Daily Revenue: ${business_metrics['Avg_Daily_Revenue']:,.0f}")
        
        if 'Total_Customers' in business_metrics:
            print(f"   Total Customers: {business_metrics['Total_Customers']:,.0f}")
            print(f"   Average Transaction Value: ${business_metrics['Avg_Transaction_Value']:.2f}")
        
        print("\n✅ Forecasting demo completed successfully!")
        
    except Exception as e:
        print(f"❌ Forecasting demo failed: {e}")
        print("   This might be due to missing dependencies or data issues.")

def main():
    """Main demo function"""
    print("🎬 AI-Powered Retail Demand Forecasting Demo")
    print("=" * 60)
    
    print("\nThis demo showcases the key features of our retail forecasting application.")
    print("It will create sample data and demonstrate various analysis capabilities.")
    
    # Run demos
    try:
        demo_analysis()
        demo_forecasting()
        
        print("\n" + "=" * 60)
        print("🎉 Demo completed successfully!")
        print("\n🚀 To run the full web application:")
        print("   streamlit run app.py")
        print("\n📖 For more information, see README.md")
        
    except Exception as e:
        print(f"\n❌ Demo failed: {e}")
        print("Please check your installation and dependencies.")

if __name__ == "__main__":
    main() 