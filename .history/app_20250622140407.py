import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import warnings
import os
import io
import base64
from typing import Dict, List, Optional, Tuple
import calendar
import seaborn as sns
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')

# Import our custom modules
from data.loader import DataLoader
from data.features import FeatureEngineer
from models.forecasting import DemandForecaster, ProphetForecaster
from models.product_analysis import ProductAnalyzer
from utils.visualization import RetailVisualizer
from utils.metrics import RetailMetrics

# Page configuration
st.set_page_config(
    page_title="AI-Powered Retail Demand Forecasting",
    page_icon="🛍️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern look
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 2rem;
        padding: 1rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 15px;
        box-shadow: 0 8px 32px rgba(0,0,0,0.1);
        text-align: center;
        margin: 0.5rem 0;
    }
    .metric-value {
        font-size: 2rem;
        font-weight: bold;
        margin-bottom: 0.5rem;
    }
    .metric-label {
        font-size: 0.9rem;
        opacity: 0.9;
    }
    .stButton>button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 25px;
        border: none;
        padding: 0.75rem 2rem;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(0,0,0,0.2);
    }
    .upload-section {
        background: #f8f9fa;
        padding: 2rem;
        border-radius: 15px;
        border: 2px dashed #dee2e6;
        text-align: center;
        margin: 1rem 0;
    }
    .success-message {
        background: linear-gradient(135deg, #28a745 0%, #20c997 100%);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    .warning-message {
        background: linear-gradient(135deg, #ffc107 0%, #fd7e14 100%);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    .info-box {
        background: linear-gradient(135deg, #17a2b8 0%, #6f42c1 100%);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    .chart-container {
        background: white;
        padding: 1rem;
        border-radius: 15px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
def init_session_state():
    if 'data_loaded' not in st.session_state:
        st.session_state.data_loaded = False
    if 'models_trained' not in st.session_state:
        st.session_state.models_trained = False
    if 'forecast_generated' not in st.session_state:
        st.session_state.forecast_generated = False
    if 'sales_data' not in st.session_state:
        st.session_state.sales_data = None
    if 'store_data' not in st.session_state:
        st.session_state.store_data = None
    if 'dataset_name' not in st.session_state:
        st.session_state.dataset_name = "None"
    if 'forecast_results' not in st.session_state:
        st.session_state.forecast_results = None
    if 'model_performance' not in st.session_state:
        st.session_state.model_performance = None
    if 'feature_importance' not in st.session_state:
        st.session_state.feature_importance = None

# Initialize components
@st.cache_resource
def initialize_components():
    return {
        'data_loader': DataLoader(),
        'feature_engineer': FeatureEngineer(),
        'forecaster': DemandForecaster(),
        'prophet_forecaster': ProphetForecaster(),
        'product_analyzer': ProductAnalyzer(),
        'visualizer': RetailVisualizer(),
        'metrics': RetailMetrics()
    }

def home_page():
    """🏠 Home Page - Dashboard Overview"""
    st.markdown('<h1 class="main-header">🛍️ AI-Powered Retail Demand Forecasting</h1>', unsafe_allow_html=True)
    
    st.markdown("""
    <div style="text-align: center; margin-bottom: 2rem;">
        <p style="font-size: 1.2rem; color: #666;">
            Advanced machine learning-powered demand forecasting for retail businesses.<br>
            Predict sales, optimize inventory, and drive business growth with AI insights.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    if not st.session_state.data_loaded:
        st.markdown("""
        <div class="info-box">
            <h3>🚀 Get Started</h3>
            <p>Upload your retail data or use our sample Rossmann dataset to begin forecasting!</p>
        </div>
        """, unsafe_allow_html=True)
        return
    
    # KPIs Section
    st.subheader("📊 Key Performance Indicators")
    
    sales_df = st.session_state.sales_data
    store_df = st.session_state.store_data
    
    # Calculate KPIs
    total_sales = sales_df['Sales'].sum()
    avg_daily_sales = sales_df.groupby('Date')['Sales'].sum().mean()
    
    # Calculate growth (comparing first and last month)
    sales_df['Month'] = sales_df['Date'].dt.to_period('M')
    monthly_sales = sales_df.groupby('Month')['Sales'].sum()
    if len(monthly_sales) >= 2:
        growth_rate = ((monthly_sales.iloc[-1] - monthly_sales.iloc[0]) / monthly_sales.iloc[0]) * 100
    else:
        growth_rate = 0
    
    # Top selling product/store
    if 'Product' in sales_df.columns:
        top_product = sales_df.groupby('Product')['Sales'].sum().idxmax()
        top_product_sales = sales_df.groupby('Product')['Sales'].sum().max()
    else:
        top_product = "Store-based"
        top_product_sales = total_sales
    
    # Display KPIs in cards
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">${total_sales:,.0f}</div>
            <div class="metric-label">Total Sales</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">${avg_daily_sales:,.0f}</div>
            <div class="metric-label">Avg Daily Sales</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{growth_rate:+.1f}%</div>
            <div class="metric-label">Sales Growth</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{top_product}</div>
            <div class="metric-label">Top Performer</div>
        </div>
        """, unsafe_allow_html=True)
    
    # Charts Section
    st.subheader("📈 Sales Analytics")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
        st.write("**Sales Over Time**")
        
        # Sales trend chart
        daily_sales = sales_df.groupby('Date')['Sales'].sum().reset_index()
        fig = px.line(daily_sales, x='Date', y='Sales', 
                     title="Daily Sales Trend",
                     template='plotly_white')
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
        st.write("**Sales by Store Type**")
        
        # Store type analysis
        if 'StoreType' in store_df.columns and 'Store' in sales_df.columns:
            store_sales = sales_df.merge(store_df[['Store', 'StoreType']], on='Store')
            store_type_sales = store_sales.groupby('StoreType')['Sales'].sum()
            
            fig = px.pie(values=store_type_sales.values, names=store_type_sales.index,
                        title="Sales Distribution by Store Type",
                        template='plotly_white')
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Store type data not available")
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Product-wise sales chart
    if 'Product' in sales_df.columns:
        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
        st.write("**Product-wise Sales**")
        
        product_sales = sales_df.groupby('Product')['Sales'].sum().sort_values(ascending=False).head(10)
        fig = px.bar(x=product_sales.index, y=product_sales.values,
                    title="Top 10 Products by Sales",
                    template='plotly_white')
        fig.update_layout(height=400, xaxis_title="Product", yaxis_title="Sales")
        st.plotly_chart(fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

def upload_data_page():
    """📤 Upload Data Page"""
    st.subheader("📤 Upload Your Data")
    
    st.markdown("""
    <div class="upload-section">
        <h3>📁 Data Upload</h3>
        <p>Upload your retail sales data in CSV format or use our sample datasets</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Data source selection
    data_source = st.radio(
        "Choose your data source:",
        ["📊 Upload Custom CSV", "🗂️ Use Rossmann Dataset", "📋 Use Sample Data"]
    )
    
    if data_source == "📊 Upload Custom CSV":
        uploaded_file = st.file_uploader(
            "Choose a CSV file",
            type=['csv'],
            help="Upload your retail sales data. Expected columns: Date, Store, Sales, (optional: Product, Customers, Promo, etc.)"
        )
        
        if uploaded_file is not None:
            try:
                # Read the uploaded file
                df = pd.read_csv(uploaded_file)
                
                # Basic validation
                required_cols = ['Date', 'Sales']
                missing_cols = [col for col in required_cols if col not in df.columns]
                
                if missing_cols:
                    st.error(f"❌ Missing required columns: {missing_cols}")
                    return
                
                # Convert Date column
                df['Date'] = pd.to_datetime(df['Date'])
                
                # Show preview
                st.success("✅ File uploaded successfully!")
                st.write("**Data Preview:**")
                st.dataframe(df.head())
                st.write(f"**Shape:** {df.shape}")
                st.write(f"**Columns:** {list(df.columns)}")
                
                # Store data
                if st.button("📊 Load This Data"):
                    st.session_state.sales_data = df
                    st.session_state.store_data = pd.DataFrame()  # Empty store data
                    st.session_state.data_loaded = True
                    st.session_state.dataset_name = "Custom Upload"
                    st.success("✅ Data loaded successfully! You can now proceed to forecasting.")
                
            except Exception as e:
                st.error(f"❌ Error reading file: {str(e)}")
    
    elif data_source == "🗂️ Use Rossmann Dataset":
        if os.path.exists('data/rossmann-store-sales'):
            st.success("✅ Rossmann dataset found!")
            
            if st.button("📊 Load Rossmann Dataset"):
                with st.spinner("Loading Rossmann dataset..."):
                    try:
                        components = initialize_components()
                        sales_df, store_df = components['data_loader'].load_rossmann_data()
                        
                        st.session_state.sales_data = sales_df
                        st.session_state.store_data = store_df
                        st.session_state.data_loaded = True
                        st.session_state.dataset_name = "Rossmann Store Sales"
                        
                        st.success("✅ Rossmann dataset loaded successfully!")
                        
                        # Show preview
                        col1, col2 = st.columns(2)
                        with col1:
                            st.write("**Sales Data Preview:**")
                            st.dataframe(sales_df.head())
                        with col2:
                            st.write("**Store Data Preview:**")
                            st.dataframe(store_df.head())
                            
                    except Exception as e:
                        st.error(f"❌ Error loading Rossmann dataset: {str(e)}")
        else:
            st.error("❌ Rossmann dataset not found. Please ensure the dataset is in the correct location.")
    
    elif data_source == "📋 Use Sample Data":
        if st.button("📊 Load Sample Data"):
            with st.spinner("Loading sample data..."):
                try:
                    components = initialize_components()
                    sales_df, store_df = components['data_loader'].load_sample_data()
                    
                    st.session_state.sales_data = sales_df
                    st.session_state.store_data = store_df
                    st.session_state.data_loaded = True
                    st.session_state.dataset_name = "Sample Data"
                    
                    st.success("✅ Sample data loaded successfully!")
                    
                    # Show preview
                    col1, col2 = st.columns(2)
                    with col1:
                        st.write("**Sales Data Preview:**")
                        st.dataframe(sales_df.head())
                    with col2:
                        st.write("**Store Data Preview:**")
                        st.dataframe(store_df.head())
                        
                except Exception as e:
                    st.error(f"❌ Error loading sample data: {str(e)}")
    
    # Data validation and success message
    if st.session_state.data_loaded:
        st.markdown("""
        <div class="success-message">
            <h3>✅ Data Successfully Loaded!</h3>
            <p>Your data is ready for analysis and forecasting. Navigate to the Sales Forecast page to generate predictions.</p>
        </div>
        """, unsafe_allow_html=True)

def sales_forecast_page():
    """🔮 Sales Forecast Page"""
    st.subheader("🔮 Sales Forecasting")
    
    if not st.session_state.data_loaded:
        st.warning("⚠️ Please load data first in the Upload Data section.")
        return
    
    sales_df = st.session_state.sales_data
    store_df = st.session_state.store_data
    
    # Selection controls
    col1, col2 = st.columns(2)
    
    with col1:
        # Store/Product selection
        if 'Store' in sales_df.columns:
            stores = sorted(sales_df['Store'].unique())
            selected_store = st.selectbox("Select Store:", stores)
        else:
            selected_store = None
        
        if 'Product' in sales_df.columns:
            products = sorted(sales_df['Product'].unique())
            selected_product = st.selectbox("Select Product:", products)
        else:
            selected_product = None
    
    with col2:
        # Date range selection
        min_date = sales_df['Date'].min()
        max_date = sales_df['Date'].max()
        
        forecast_start = st.date_input(
            "Forecast Start Date:",
            value=max_date + timedelta(days=1),
            min_value=max_date + timedelta(days=1)
        )
        
        forecast_days = st.slider("Forecast Period (days):", 7, 90, 30)
    
    # Model selection
    st.write("**Model Configuration:**")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        model_type = st.selectbox(
            "Forecasting Model:",
            ["Random Forest", "XGBoost", "LightGBM", "Prophet", "Ensemble"]
        )
    
    with col2:
        confidence_level = st.slider("Confidence Level:", 0.8, 0.99, 0.95, 0.01)
    
    with col3:
        include_features = st.multiselect(
            "Include Features:",
            ["Promo", "Holidays", "Seasonality", "Store Type", "Competition"],
            default=["Promo", "Seasonality"]
        )
    
    # Run forecast button
    if st.button("🚀 Generate Forecast", type="primary"):
        with st.spinner("Training models and generating forecast..."):
            try:
                components = initialize_components()
                
                # Prepare data with better filtering logic
                filtered_df = sales_df.copy()
                
                # Apply store filter if selected
                if selected_store and 'Store' in filtered_df.columns:
                    filtered_df = filtered_df[filtered_df['Store'] == selected_store].copy()
                    st.info(f"📊 Filtered to Store {selected_store}: {len(filtered_df)} records")
                
                # Apply product filter if selected
                if selected_product and 'Product' in filtered_df.columns:
                    filtered_df = filtered_df[filtered_df['Product'] == selected_product].copy()
                    st.info(f"📦 Filtered to Product {selected_product}: {len(filtered_df)} records")
                
                # Check if we have enough data
                if len(filtered_df) < 10:
                    st.error(f"❌ Insufficient data for forecasting. Only {len(filtered_df)} records found after filtering.")
                    st.info("💡 Try selecting different store/product combinations or use the full dataset.")
                    return
                
                # Engineer features
                feature_engineer = components['feature_engineer']
                
                # Use simplified feature engineering for small datasets
                if len(filtered_df) < 50:
                    st.info(f"📊 Using simplified feature engineering for small dataset ({len(filtered_df)} records)")
                    
                    # Create basic features only
                    enhanced_df = filtered_df.copy()
                    
                    # Time features (no NaN issues)
                    enhanced_df['Year'] = enhanced_df['Date'].dt.year
                    enhanced_df['Month'] = enhanced_df['Date'].dt.month
                    enhanced_df['Day'] = enhanced_df['Date'].dt.day
                    enhanced_df['DayOfWeek'] = enhanced_df['Date'].dt.dayofweek
                    enhanced_df['Quarter'] = enhanced_df['Date'].dt.quarter
                    enhanced_df['DayOfYear'] = enhanced_df['Date'].dt.dayofyear
                    
                    # Cyclical encoding
                    enhanced_df['Month_Sin'] = np.sin(2 * np.pi * enhanced_df['Month'] / 12)
                    enhanced_df['Month_Cos'] = np.cos(2 * np.pi * enhanced_df['Month'] / 12)
                    enhanced_df['DayOfWeek_Sin'] = np.sin(2 * np.pi * enhanced_df['DayOfWeek'] / 7)
                    enhanced_df['DayOfWeek_Cos'] = np.cos(2 * np.pi * enhanced_df['DayOfWeek'] / 7)
                    
                    # Weekend indicator
                    enhanced_df['IsWeekend'] = (enhanced_df['DayOfWeek'] >= 5).astype(int)
                    
                    # Month end/beginning indicators
                    enhanced_df['IsMonthEnd'] = enhanced_df['Date'].dt.is_month_end.astype(int)
                    enhanced_df['IsMonthStart'] = enhanced_df['Date'].dt.is_month_start.astype(int)
                    
                    # Seasonal features
                    enhanced_df['IsChristmas'] = ((enhanced_df['Date'].dt.month == 12) & (enhanced_df['Date'].dt.day == 25)).astype(int)
                    enhanced_df['IsNewYear'] = ((enhanced_df['Date'].dt.month == 1) & (enhanced_df['Date'].dt.day == 1)).astype(int)
                    enhanced_df['IsValentines'] = ((enhanced_df['Date'].dt.month == 2) & (enhanced_df['Date'].dt.day == 14)).astype(int)
                    enhanced_df['IsSummerHoliday'] = enhanced_df['Date'].dt.month.isin([7, 8]).astype(int)
                    enhanced_df['IsWinterHoliday'] = ((enhanced_df['Date'].dt.month == 12) & (enhanced_df['Date'].dt.day >= 20)).astype(int)
                    enhanced_df['IsHolidaySeason'] = enhanced_df['Date'].dt.month.isin([11, 12]).astype(int)
                    
                    # Simple lag features (fill NaN with 0)
                    enhanced_df['Sales_Lag1'] = enhanced_df['Sales'].shift(1).fillna(enhanced_df['Sales'].mean())
                    enhanced_df['Sales_Lag7'] = enhanced_df['Sales'].shift(7).fillna(enhanced_df['Sales'].mean())
                    
                    # Simple rolling features (fill NaN with mean)
                    enhanced_df['Sales_MA7'] = enhanced_df['Sales'].rolling(7, min_periods=1).mean()
                    enhanced_df['Sales_STD7'] = enhanced_df['Sales'].rolling(7, min_periods=1).std().fillna(0)
                    
                    # Interaction features
                    if 'Promo' in enhanced_df.columns:
                        enhanced_df['Promo_Weekend'] = enhanced_df['Promo'] * enhanced_df['IsWeekend']
                        enhanced_df['Promo_Sales_Lag1'] = enhanced_df['Promo'] * enhanced_df['Sales_Lag1']
                    
                    # Encode categorical variables
                    for col in enhanced_df.select_dtypes(include=['object', 'category']).columns:
                        if col != 'Date':
                            enhanced_df[col + '_Encoded'] = pd.Categorical(enhanced_df[col]).codes
                    
                    filtered_df = enhanced_df
                    
                else:
                    # Use full feature engineering for larger datasets
                    filtered_df = feature_engineer.engineer_all_features(filtered_df)
                
                # Train model
                forecaster = components['forecaster']
                X, y = forecaster.prepare_data(filtered_df)
                
                # Check if we have enough samples for training
                if len(X) < 10:
                    st.error(f"❌ Insufficient samples for training. Only {len(X)} samples available after feature engineering.")
                    st.info("💡 Try using more data or different filtering options.")
                    return
                
                # Map model names
                model_mapping = {
                    "Random Forest": "random_forest",
                    "XGBoost": "xgboost", 
                    "LightGBM": "lightgbm",
                    "Prophet": "prophet"
                }
                
                if model_type != "Prophet":
                    model_name = model_mapping[model_type]
                    
                    # Use fewer CV folds if we have limited data
                    if len(X) < 50:
                        st.info(f"📊 Limited data detected ({len(X)} samples). Using simplified training.")
                        # For small datasets, we'll use a simpler approach
                        from sklearn.model_selection import train_test_split
                        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                        
                        # Train model
                        if model_name == "random_forest":
                            from sklearn.ensemble import RandomForestRegressor
                            model = RandomForestRegressor(n_estimators=50, random_state=42)
                        elif model_name == "xgboost":
                            import xgboost as xgb
                            model = xgb.XGBRegressor(n_estimators=50, random_state=42)
                        elif model_name == "lightgbm":
                            import lightgbm as lgb
                            model = lgb.LGBMRegressor(n_estimators=50, random_state=42)
                        else:
                            from sklearn.ensemble import GradientBoostingRegressor
                            model = GradientBoostingRegressor(n_estimators=50, random_state=42)
                        
                        model.fit(X_train, y_train)
                        y_pred = model.predict(X_test)
                        
                        # Calculate metrics
                        from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
                        mae = mean_absolute_error(y_test, y_pred)
                        rmse = mean_squared_error(y_test, y_pred, squared=False)
                        r2 = r2_score(y_test, y_pred)
                        
                        performance = {
                            model_name: {
                                'mae': mae,
                                'rmse': rmse,
                                'r2': r2
                            }
                        }
                        
                        # Store model for prediction
                        forecaster.models[model_name] = model
                        forecaster.best_model = model
                        forecaster.best_model_name = model_name
                        
                    else:
                        performance = forecaster.train_models(X, y, [model_name])
                    
                    # Generate future dates
                    future_dates = pd.date_range(
                        start=forecast_start,
                        periods=forecast_days,
                        freq='D'
                    )
                    
                    # Make predictions
                    forecast_df = forecaster.predict_future(
                        filtered_df, future_dates, feature_engineer, model_name
                    )
                    
                    # Store results
                    st.session_state.forecast_results = {
                        'historical': filtered_df,
                        'forecast': forecast_df,
                        'performance': performance[model_name],
                        'model_name': model_name
                    }
                    
                else:
                    # Prophet forecasting
                    prophet_forecaster = components['prophet_forecaster']
                    prophet_data = prophet_forecaster.prepare_prophet_data(filtered_df)
                    prophet_models = prophet_forecaster.train_prophet_models(prophet_data)
                    forecast_df = prophet_forecaster.predict_future(forecast_days)
                    
                    st.session_state.forecast_results = {
                        'historical': filtered_df,
                        'forecast': forecast_df,
                        'performance': {'rmse': 0, 'mae': 0, 'r2': 0},
                        'model_name': 'prophet'
                    }
                
                st.session_state.forecast_generated = True
                st.success("✅ Forecast generated successfully!")
                
            except Exception as e:
                st.error(f"❌ Error generating forecast: {str(e)}")
                st.info("💡 Try using different filtering options or the full dataset.")
    
    # Display results
    if st.session_state.forecast_generated and st.session_state.forecast_results:
        results = st.session_state.forecast_results
        
        st.subheader("📊 Forecast Results")
        
        # Performance metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("RMSE", f"{results['performance']['rmse']:.2f}")
        with col2:
            st.metric("MAE", f"{results['performance']['mae']:.2f}")
        with col3:
            st.metric("R² Score", f"{results['performance']['r2']:.3f}")
        with col4:
            st.metric("Model", results['model_name'].title())
        
        # Forecast visualization
        st.write("**Actual vs Predicted Sales**")
        
        # Prepare data for plotting
        hist_agg = results['historical'].groupby('Date')['Sales'].sum().reset_index()
        hist_agg['Type'] = 'Historical'
        
        forecast_agg = results['forecast'].groupby('Date')['Sales'].sum().reset_index()
        forecast_agg['Type'] = 'Forecast'
        
        combined_df = pd.concat([hist_agg, forecast_agg], ignore_index=True)
        
        fig = px.line(combined_df, x='Date', y='Sales', color='Type',
                     title="Historical vs Forecasted Sales",
                     template='plotly_white')
        fig.update_layout(height=500)
        st.plotly_chart(fig, use_container_width=True)
        
        # Download forecast
        st.write("**Download Forecast Results**")
        
        # Prepare download data
        download_df = results['forecast'][['Date', 'Sales']].copy()
        download_df.columns = ['Date', 'Predicted_Sales']
        
        csv = download_df.to_csv(index=False)
        st.download_button(
            label="📥 Download Forecast as CSV",
            data=csv,
            file_name=f"sales_forecast_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv"
        )

def inventory_suggestions_page():
    """📦 Inventory Suggestions Page"""
    st.subheader("📦 Inventory Management Suggestions")
    
    if not st.session_state.forecast_generated:
        st.warning("⚠️ Please generate a forecast first in the Sales Forecast section.")
        return
    
    results = st.session_state.forecast_results
    forecast_df = results['forecast']
    
    # Inventory parameters
    st.write("**Inventory Parameters:**")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        safety_stock_days = st.slider("Safety Stock (days):", 3, 14, 7)
    
    with col2:
        reorder_point_days = st.slider("Reorder Point (days):", 1, 7, 3)
    
    with col3:
        lead_time_days = st.slider("Lead Time (days):", 1, 14, 5)
    
    # Calculate inventory suggestions
    if st.button("📊 Generate Inventory Suggestions"):
        with st.spinner("Calculating inventory recommendations..."):
            try:
                # Calculate daily average demand
                daily_demand = forecast_df['Sales'].mean()
                
                # Calculate safety stock
                safety_stock = daily_demand * safety_stock_days
                
                # Calculate reorder point
                reorder_point = daily_demand * reorder_point_days
                
                # Calculate economic order quantity (simplified)
                eoq = np.sqrt(2 * daily_demand * 365 * 10 / 0.2)  # Assuming $10 order cost, 20% holding cost
                
                # Create inventory recommendations
                inventory_recs = pd.DataFrame({
                    'Metric': ['Daily Average Demand', 'Safety Stock', 'Reorder Point', 'Economic Order Quantity', 'Max Stock Level'],
                    'Value': [daily_demand, safety_stock, reorder_point, eoq, safety_stock + eoq],
                    'Unit': ['units/day', 'units', 'units', 'units', 'units']
                })
                
                # Store recommendations
                st.session_state.inventory_recommendations = inventory_recs
                
                st.success("✅ Inventory recommendations generated!")
                
            except Exception as e:
                st.error(f"❌ Error generating inventory suggestions: {str(e)}")
    
    # Display inventory recommendations
    if 'inventory_recommendations' in st.session_state:
        st.subheader("📋 Inventory Recommendations")
        
        recs = st.session_state.inventory_recommendations
        
        # Display as metrics
        col1, col2, col3, col4, col5 = st.columns(5)
        
        for i, (_, row) in enumerate(recs.iterrows()):
            with [col1, col2, col3, col4, col5][i]:
                st.metric(
                    row['Metric'],
                    f"{row['Value']:.1f}",
                    row['Unit']
                )
        
        # Inventory chart
        st.write("**Stock Level Timeline**")
        
        # Create stock level simulation
        dates = forecast_df['Date'].tolist()
        stock_levels = []
        current_stock = recs[recs['Metric'] == 'Max Stock Level']['Value'].iloc[0]
        reorder_level = recs[recs['Metric'] == 'Reorder Point']['Value'].iloc[0]
        
        for demand in forecast_df['Sales']:
            current_stock -= demand
            if current_stock <= reorder_level:
                current_stock += recs[recs['Metric'] == 'Economic Order Quantity']['Value'].iloc[0]
            stock_levels.append(max(0, current_stock))
        
        stock_df = pd.DataFrame({
            'Date': dates,
            'Stock_Level': stock_levels,
            'Reorder_Point': [reorder_level] * len(dates)
        })
        
        fig = px.line(stock_df, x='Date', y=['Stock_Level', 'Reorder_Point'],
                     title="Projected Stock Levels",
                     template='plotly_white')
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
        
        # Low stock warnings
        low_stock_dates = stock_df[stock_df['Stock_Level'] <= reorder_level]
        if not low_stock_dates.empty:
            st.warning(f"⚠️ Low stock warnings: {len(low_stock_dates)} days with stock below reorder point")
        
        # Download inventory plan
        st.write("**Download Inventory Plan**")
        
        inventory_plan = pd.DataFrame({
            'Date': dates,
            'Predicted_Demand': forecast_df['Sales'],
            'Projected_Stock': stock_levels,
            'Reorder_Needed': [stock <= reorder_level for stock in stock_levels]
        })
        
        csv = inventory_plan.to_csv(index=False)
        st.download_button(
            label="📥 Download Inventory Plan as CSV",
            data=csv,
            file_name=f"inventory_plan_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv"
        )

def model_insights_page():
    """🧠 Model Insights Page"""
    st.subheader("🧠 Model Insights & Analytics")
    
    if not st.session_state.forecast_generated:
        st.warning("⚠️ Please generate a forecast first to see model insights.")
        return
    
    results = st.session_state.forecast_results
    
    # Feature importance
    st.write("**Feature Importance Analysis**")
    
    if results['model_name'] != 'prophet':
        try:
            components = initialize_components()
            forecaster = components['forecaster']
            
            # Get feature importance
            importance_df = forecaster.get_feature_importance(results['model_name'])
            
            if importance_df is not None:
                # Plot feature importance
                fig = px.bar(importance_df.head(10), x='importance', y='feature',
                           title="Top 10 Most Important Features",
                           orientation='h',
                           template='plotly_white')
                fig.update_layout(height=500)
                st.plotly_chart(fig, use_container_width=True)
                
                # Store for download
                st.session_state.feature_importance = importance_df
                
                # Download feature importance
                csv = importance_df.to_csv(index=False)
                st.download_button(
                    label="📥 Download Feature Importance as CSV",
                    data=csv,
                    file_name="feature_importance.csv",
                    mime="text/csv"
                )
            else:
                st.info("Feature importance not available for this model type.")
                
        except Exception as e:
            st.error(f"❌ Error getting feature importance: {str(e)}")
    else:
        st.info("Feature importance analysis is not available for Prophet models.")
    
    # Correlation analysis
    st.write("**Feature Correlation Analysis**")
    
    try:
        # Prepare data for correlation
        hist_df = results['historical'].copy()
        
        # Select numeric columns
        numeric_cols = hist_df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 1:
            corr_matrix = hist_df[numeric_cols].corr()
            
            # Create correlation heatmap
            fig = px.imshow(corr_matrix,
                           title="Feature Correlation Heatmap",
                           template='plotly_white',
                           color_continuous_scale='RdBu')
            fig.update_layout(height=500)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Insufficient numeric features for correlation analysis.")
            
    except Exception as e:
        st.error(f"❌ Error creating correlation heatmap: {str(e)}")
    
    # Model performance insights
    st.write("**Model Performance Insights**")
    
    performance = results['performance']
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric("Root Mean Square Error (RMSE)", f"{performance['rmse']:.2f}")
        st.metric("Mean Absolute Error (MAE)", f"{performance['mae']:.2f}")
    
    with col2:
        st.metric("R² Score", f"{performance['r2']:.3f}")
        st.metric("Model Type", results['model_name'].title())
    
    # Performance interpretation
    st.write("**Performance Interpretation:**")
    
    if performance['r2'] > 0.8:
        st.success("🎯 Excellent model performance! The model explains most of the variance in sales.")
    elif performance['r2'] > 0.6:
        st.info("📈 Good model performance. The model provides reliable predictions.")
    elif performance['r2'] > 0.4:
        st.warning("⚠️ Moderate model performance. Consider feature engineering or different models.")
    else:
        st.error("❌ Poor model performance. Review data quality and model selection.")
    
    # Retrain model option
    st.write("**Model Retraining**")
    
    if st.button("🔄 Retrain Model with Current Data"):
        st.info("Model retraining feature will be implemented in future versions.")

def sales_calendar_page():
    """📅 Sales Calendar & Event Insights"""
    st.subheader("📅 Sales Calendar & Event Insights")
    
    if not st.session_state.data_loaded:
        st.warning("⚠️ Please load data first to view calendar insights.")
        return
    
    sales_df = st.session_state.sales_data
    
    # Calendar view
    st.write("**Interactive Sales Calendar**")
    
    # Year and month selection
    col1, col2 = st.columns(2)
    
    with col1:
        year = st.selectbox("Select Year:", sorted(sales_df['Date'].dt.year.unique()))
    
    with col2:
        month = st.selectbox("Select Month:", range(1, 13))
    
    # Filter data for selected year/month
    filtered_df = sales_df[
        (sales_df['Date'].dt.year == year) & 
        (sales_df['Date'].dt.month == month)
    ].copy()
    
    if filtered_df.empty:
        st.warning(f"No data available for {calendar.month_name[month]} {year}")
        return
    
    # Daily sales aggregation
    daily_sales = filtered_df.groupby('Date')['Sales'].sum().reset_index()
    daily_sales['Day'] = daily_sales['Date'].dt.day
    daily_sales['DayOfWeek'] = daily_sales['Date'].dt.day_name()
    
    # Create calendar heatmap
    st.write(f"**Sales Heatmap for {calendar.month_name[month]} {year}**")
    
    # Create calendar grid
    cal_days = []
    for day in range(1, calendar.monthrange(year, month)[1] + 1):
        date = datetime(year, month, day)
        day_sales = daily_sales[daily_sales['Day'] == day]['Sales'].sum()
        cal_days.append({
            'Day': day,
            'Date': date,
            'Sales': day_sales,
            'DayOfWeek': date.strftime('%A')
        })
    
    cal_df = pd.DataFrame(cal_days)
    
    # Create heatmap
    fig = px.imshow(
        cal_df['Sales'].values.reshape(-1, 7),
        title=f"Sales Calendar - {calendar.month_name[month]} {year}",
        template='plotly_white',
        color_continuous_scale='Blues'
    )
    fig.update_layout(height=400)
    st.plotly_chart(fig, use_container_width=True)
    
    # Event impact analysis
    st.write("**Event Impact Analysis**")
    
    # Check for promotional events
    if 'Promo' in sales_df.columns:
        promo_analysis = sales_df.groupby(['Date', 'Promo'])['Sales'].sum().reset_index()
        promo_analysis['Event_Type'] = promo_analysis['Promo'].map({1: 'Promotion', 0: 'Regular Day'})
        
        fig = px.box(promo_analysis, x='Event_Type', y='Sales',
                    title="Sales Impact of Promotions",
                    template='plotly_white')
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    # Holiday analysis
    if 'StateHoliday' in sales_df.columns:
        holiday_analysis = sales_df.groupby(['Date', 'StateHoliday'])['Sales'].sum().reset_index()
        holiday_analysis['Event_Type'] = holiday_analysis['StateHoliday'].map({
            'a': 'Public Holiday', 'b': 'Easter', 'c': 'Christmas', 0: 'Regular Day'
        })
        
        fig = px.box(holiday_analysis, x='Event_Type', y='Sales',
                    title="Sales Impact of Holidays",
                    template='plotly_white')
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    # Weekly pattern analysis
    st.write("**Weekly Sales Pattern**")
    
    weekly_pattern = sales_df.groupby(sales_df['Date'].dt.day_name())['Sales'].mean().reindex([
        'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'
    ])
    
    fig = px.bar(x=weekly_pattern.index, y=weekly_pattern.values,
                title="Average Sales by Day of Week",
                template='plotly_white')
    fig.update_layout(height=400, xaxis_title="Day of Week", yaxis_title="Average Sales")
    st.plotly_chart(fig, use_container_width=True)
    
    # Monthly comparison
    if st.session_state.forecast_generated:
        st.write("**Monthly Forecast Comparison**")
        
        results = st.session_state.forecast_results
        forecast_df = results['forecast']
        
        # Compare actual vs forecast for the same month
        actual_monthly = sales_df[
            (sales_df['Date'].dt.year == year) & 
            (sales_df['Date'].dt.month == month)
        ]['Sales'].sum()
        
        forecast_monthly = forecast_df[
            (forecast_df['Date'].dt.year == year) & 
            (forecast_df['Date'].dt.month == month)
        ]['Sales'].sum()
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Actual Sales", f"${actual_monthly:,.0f}")
        with col2:
            st.metric("Forecasted Sales", f"${forecast_monthly:,.0f}")

def about_page():
    """📞 About / Contact Page"""
    st.subheader("📞 About This Tool")
    
    st.markdown("""
    <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 2rem; border-radius: 15px; margin: 2rem 0;">
        <h2>🛍️ AI-Powered Retail Demand Forecasting</h2>
        <p style="font-size: 1.1rem;">
            This advanced forecasting tool leverages machine learning to predict retail sales demand, 
            helping businesses optimize inventory, reduce costs, and increase profitability.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Features
    st.write("**✨ Key Features:**")
    
    features = [
        "🔮 **Advanced Forecasting**: Multiple ML models (Random Forest, XGBoost, LightGBM, Prophet)",
        "📊 **Interactive Dashboard**: Real-time KPIs and visualizations",
        "📦 **Inventory Optimization**: Smart stock level recommendations",
        "📅 **Event Analysis**: Holiday and promotional impact insights",
        "🧠 **Model Insights**: Feature importance and performance analytics",
        "📤 **Data Flexibility**: Support for custom CSV uploads and sample datasets"
    ]
    
    for feature in features:
        st.write(f"• {feature}")
    
    # Dataset information
    st.write("**📊 Dataset Information:**")
    
    st.markdown("""
    This tool is designed to work with the **Rossmann Store Sales** dataset from Kaggle, 
    which contains historical sales data for Rossmann drug stores across Germany.
    
    **Dataset Features:**
    - Store information (type, assortment, competition distance)
    - Sales data (daily sales, customers, promotions)
    - Holiday and state information
    - Competition data
    
    **Source**: [Rossmann Store Sales on Kaggle](https://www.kaggle.com/c/rossmann-store-sales)
    """)
    
    # Technical details
    st.write("**🔧 Technical Details:**")
    
    tech_details = [
        "**Framework**: Streamlit for web interface",
        "**ML Libraries**: Scikit-learn, XGBoost, LightGBM, Prophet",
        "**Visualization**: Plotly for interactive charts",
        "**Data Processing**: Pandas, NumPy for data manipulation",
        "**Deployment**: Ready for cloud deployment"
    ]
    
    for detail in tech_details:
        st.write(f"• {detail}")
    
    # Contact information
    st.write("**📞 Contact Information:**")
    
    contact_info = {
        "📧 Email": "developer@retailforecasting.com",
        "💼 LinkedIn": "linkedin.com/in/retail-forecasting",
        "🐙 GitHub": "github.com/retail-forecasting",
        "🌐 Website": "retailforecasting.ai"
    }
    
    for platform, link in contact_info.items():
        st.write(f"• **{platform}**: {link}")
    
    # Version and updates
    st.write("**📋 Version Information:**")
    st.write("• **Current Version**: 1.0.0")
    st.write("• **Last Updated**: December 2024")
    st.write("• **License**: MIT License")

def main():
    """Main function to run the Streamlit app"""
    init_session_state()
    components = initialize_components()
    
    # Sidebar
    st.sidebar.markdown("""
    <div style="text-align: center; padding: 1rem;">
        <h2>🛍️ Retail Forecasting</h2>
        <p>AI-Powered Demand Prediction</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Auto-load Rossmann data if available
    if not st.session_state.data_loaded and os.path.exists('data/rossmann-store-sales'):
        try:
            sales_df, store_df = components['data_loader'].load_rossmann_data()
            st.session_state.sales_data = sales_df
            st.session_state.store_data = store_df
            st.session_state.data_loaded = True
            st.session_state.dataset_name = "Rossmann Store Sales (Auto-loaded)"
        except Exception as e:
            st.sidebar.error(f"Auto-load failed: {e}")
    
    # Navigation
    st.sidebar.title("Navigation")
    
    page_options = {
        "🏠 Home": home_page,
        "📤 Upload Data": upload_data_page,
        "🔮 Sales Forecast": sales_forecast_page,
        "📦 Inventory Suggestions": inventory_suggestions_page,
        "🧠 Model Insights": model_insights_page,
        "📅 Sales Calendar": sales_calendar_page,
        "📞 About": about_page
    }
    
    selection = st.sidebar.radio("Go to", list(page_options.keys()))
    
    # Display current dataset info
    if st.session_state.data_loaded:
        st.sidebar.success(f"✅ {st.session_state.dataset_name}")
        st.sidebar.write(f"📊 {len(st.session_state.sales_data):,} records loaded")
    
    # Exit button
    st.sidebar.markdown("---")
    if st.sidebar.button("🚪 Exit Application"):
        st.stop()
    
    # Run selected page
    page_options[selection]()

if __name__ == "__main__":
    main() 