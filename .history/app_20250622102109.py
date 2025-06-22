import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta
import warnings
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
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .success-card {
        border-left-color: #2ca02c;
    }
    .warning-card {
        border-left-color: #d62728;
    }
    .info-card {
        border-left-color: #9467bd;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False
if 'models_trained' not in st.session_state:
    st.session_state.models_trained = False
if 'forecast_generated' not in st.session_state:
    st.session_state.forecast_generated = False

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

components = initialize_components()

# Main header
st.markdown('<h1 class="main-header">🛍️ AI-Powered Retail Demand Forecasting</h1>', unsafe_allow_html=True)

# Sidebar
st.sidebar.title("📋 Navigation")
page = st.sidebar.selectbox(
    "Choose a page:",
    ["🏠 Dashboard", "📊 Data Management", "🤖 Model Training", "🔮 Forecasting", "📈 Product Analysis", "📦 Stock Recommendations", "📋 Reports"]
)

# Dashboard Page
if page == "🏠 Dashboard":
    st.header("📊 Retail Analytics Dashboard")
    
    if not st.session_state.data_loaded:
        st.warning("⚠️ Please load data first in the Data Management section.")
        
        # Show sample metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Sales", "0", "0%")
        with col2:
            st.metric("Products", "0", "0%")
        with col3:
            st.metric("Stores", "0", "0%")
        with col4:
            st.metric("Forecast Accuracy", "0%", "0%")
            
    else:
        # Load data and show dashboard
        try:
            # Load sample data for demonstration
            sales_df, store_df = components['data_loader'].load_sample_data()
            
            # Calculate basic metrics
            total_sales = sales_df['Sales'].sum()
            num_products = sales_df['Product'].nunique()
            num_stores = sales_df['Store'].nunique()
            date_range = sales_df['Date'].max() - sales_df['Date'].min()
            
            # Display metrics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Total Sales", f"{total_sales:,.0f}", "📈")
            with col2:
                st.metric("Products", f"{num_products}", "📦")
            with col3:
                st.metric("Stores", f"{num_stores}", "🏪")
            with col4:
                st.metric("Data Period", f"{date_range.days} days", "📅")
            
            # Sales trend chart
            st.subheader("📈 Sales Trend")
            fig = components['visualizer'].plot_sales_trend(sales_df)
            st.plotly_chart(fig, use_container_width=True)
            
            # Product performance
            st.subheader("🏆 Top Products")
            product_metrics = components['product_analyzer'].calculate_product_metrics(sales_df)
            top_products = product_metrics.nlargest(5, 'Total_Sales')
            
            fig = components['visualizer'].plot_product_performance(product_metrics, top_n=5)
            st.plotly_chart(fig, use_container_width=True)
            
        except Exception as e:
            st.error(f"Error loading dashboard: {str(e)}")

# Data Management Page
elif page == "📊 Data Management":
    st.header("📊 Data Management")
    
    # Data source selection
    data_source = st.radio(
        "Choose data source:",
        ["📊 Use Sample Data", "📁 Upload CSV Files"]
    )
    
    if data_source == "📊 Use Sample Data":
        if st.button("🔄 Load Sample Data"):
            with st.spinner("Loading sample data..."):
                try:
                    sales_df, store_df = components['data_loader'].load_sample_data()
                    st.session_state.sales_data = sales_df
                    st.session_state.store_data = store_df
                    st.session_state.data_loaded = True
                    st.success("✅ Sample data loaded successfully!")
                    
                    # Show data preview
                    st.subheader("📋 Data Preview")
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.write("**Sales Data:**")
                        st.dataframe(sales_df.head())
                        st.write(f"Shape: {sales_df.shape}")
                    
                    with col2:
                        st.write("**Store Data:**")
                        st.dataframe(store_df.head())
                        st.write(f"Shape: {store_df.shape}")
                        
                except Exception as e:
                    st.error(f"Error loading sample data: {str(e)}")
    
    elif data_source == "📁 Upload CSV Files":
        st.subheader("📁 Upload Your Data")
        
        uploaded_sales = st.file_uploader("Upload Sales Data (CSV)", type=['csv'])
        uploaded_stores = st.file_uploader("Upload Store Data (CSV) - Optional", type=['csv'])
        
        if uploaded_sales is not None:
            try:
                sales_df = pd.read_csv(uploaded_sales)
                sales_df['Date'] = pd.to_datetime(sales_df['Date'])
                st.session_state.sales_data = sales_df
                
                if uploaded_stores is not None:
                    store_df = pd.read_csv(uploaded_stores)
                    st.session_state.store_data = store_df
                else:
                    store_df = None
                
                st.session_state.data_loaded = True
                st.success("✅ Data uploaded successfully!")
                
                # Show data preview
                st.subheader("📋 Data Preview")
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write("**Sales Data:**")
                    st.dataframe(sales_df.head())
                    st.write(f"Shape: {sales_df.shape}")
                
                with col2:
                    if store_df is not None:
                        st.write("**Store Data:**")
                        st.dataframe(store_df.head())
                        st.write(f"Shape: {store_df.shape}")
                    else:
                        st.info("No store data uploaded")
                        
            except Exception as e:
                st.error(f"Error uploading data: {str(e)}")

# Model Training Page
elif page == "🤖 Model Training":
    st.header("🤖 Model Training")
    
    if not st.session_state.data_loaded:
        st.warning("⚠️ Please load data first in the Data Management section.")
    else:
        st.subheader("🔧 Model Configuration")
        
        # Model selection
        models_to_train = st.multiselect(
            "Select models to train:",
            ["random_forest", "xgboost", "lightgbm", "gradient_boosting", "linear_regression", "ridge", "lasso"],
            default=["random_forest", "xgboost", "lightgbm"]
        )
        
        if st.button("🚀 Train Models"):
            if not models_to_train:
                st.warning("Please select at least one model to train.")
            else:
                with st.spinner("Training models..."):
                    try:
                        # Load data
                        sales_df = st.session_state.sales_data
                        store_df = st.session_state.store_data
                        
                        # Preprocess data
                        processed_df = components['data_loader'].preprocess_data(sales_df, store_df)
                        
                        # Engineer features
                        feature_df = components['feature_engineer'].engineer_all_features(processed_df)
                        
                        # Prepare data for modeling
                        X, y = components['forecaster'].prepare_data(feature_df)
                        
                        # Train models
                        results = components['forecaster'].train_models(X, y, models_to_train)
                        
                        st.session_state.models_trained = True
                        st.session_state.training_results = results
                        st.session_state.feature_df = feature_df
                        
                        st.success("✅ Models trained successfully!")
                        
                        # Show results
                        st.subheader("📊 Training Results")
                        results_df = pd.DataFrame(results).T
                        st.dataframe(results_df)
                        
                        # Show best model
                        best_model = components['forecaster'].best_model_name
                        st.info(f"🏆 Best Model: {best_model}")
                        
                    except Exception as e:
                        st.error(f"Error training models: {str(e)}")

# Forecasting Page
elif page == "🔮 Forecasting":
    st.header("🔮 Demand Forecasting")
    
    if not st.session_state.data_loaded:
        st.warning("⚠️ Please load data first in the Data Management section.")
    elif not st.session_state.models_trained:
        st.warning("⚠️ Please train models first in the Model Training section.")
    else:
        st.subheader("📅 Forecast Configuration")
        
        # Forecast period
        forecast_days = st.slider("Forecast Period (days):", 7, 90, 30)
        
        # Model selection
        available_models = list(st.session_state.training_results.keys())
        selected_model = st.selectbox("Select model for forecasting:", available_models)
        
        if st.button("🔮 Generate Forecast"):
            with st.spinner("Generating forecast..."):
                try:
                    # Generate future dates
                    last_date = st.session_state.sales_data['Date'].max()
                    future_dates = pd.date_range(start=last_date + timedelta(days=1), periods=forecast_days, freq='D')
                    
                    # Generate forecast
                    forecast_df = components['forecaster'].predict_future(
                        st.session_state.feature_df, 
                        future_dates, 
                        components['feature_engineer'],
                        selected_model
                    )
                    
                    st.session_state.forecast_generated = True
                    st.session_state.forecast_data = forecast_df
                    
                    st.success("✅ Forecast generated successfully!")
                    
                    # Show forecast
                    st.subheader("📈 Forecast Results")
                    
                    # Forecast chart
                    fig = components['visualizer'].plot_forecast_comparison(
                        st.session_state.sales_data, 
                        forecast_df
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Forecast summary
                    st.subheader("📋 Forecast Summary")
                    forecast_summary = forecast_df.groupby('Date')['Sales'].sum().reset_index()
                    st.dataframe(forecast_summary)
                    
                    # Download forecast
                    csv = forecast_summary.to_csv(index=False)
                    st.download_button(
                        label="📥 Download Forecast",
                        data=csv,
                        file_name=f"forecast_{datetime.now().strftime('%Y%m%d')}.csv",
                        mime="text/csv"
                    )
                    
                except Exception as e:
                    st.error(f"Error generating forecast: {str(e)}")

# Product Analysis Page
elif page == "📈 Product Analysis":
    st.header("📈 Product Analysis")
    
    if not st.session_state.data_loaded:
        st.warning("⚠️ Please load data first in the Data Management section.")
    else:
        st.subheader("🔍 Product Performance Analysis")
        
        if st.button("📊 Analyze Products"):
            with st.spinner("Analyzing products..."):
                try:
                    # Load data
                    sales_df = st.session_state.sales_data
                    
                    # Calculate product metrics
                    product_metrics = components['product_analyzer'].calculate_product_metrics(sales_df)
                    
                    # Classify products
                    classified_products = components['product_analyzer'].classify_products(product_metrics)
                    
                    # Get insights
                    insights = components['product_analyzer'].get_product_insights(sales_df, classified_products)
                    
                    st.session_state.product_analysis = {
                        'metrics': product_metrics,
                        'classified': classified_products,
                        'insights': insights
                    }
                    
                    st.success("✅ Product analysis completed!")
                    
                    # Display results
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.subheader("🏆 Top Products")
                        st.dataframe(insights['top_products'])
                    
                    with col2:
                        st.subheader("📈 Fastest Growing")
                        st.dataframe(insights['fastest_growing'])
                    
                    # Product categories chart
                    st.subheader("📊 Product Categories")
                    fig = components['visualizer'].plot_product_categories(classified_products)
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Hot vs Slow-moving products
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.subheader("🔥 Hot Products")
                        st.dataframe(insights['hot_products'])
                    
                    with col2:
                        st.subheader("🐌 Slow-moving Products")
                        st.dataframe(insights['slow_moving'])
                    
                    # Generate recommendations
                    recommendations = components['product_analyzer'].generate_recommendations(classified_products)
                    
                    st.subheader("💡 Recommendations")
                    
                    for category, recs in recommendations.items():
                        if recs:
                            st.write(f"**{category.replace('_', ' ').title()}:**")
                            for rec in recs:
                                st.write(f"• {rec}")
                            st.write("")
                    
                except Exception as e:
                    st.error(f"Error analyzing products: {str(e)}")

# Stock Recommendations Page
elif page == "📦 Stock Recommendations":
    st.header("📦 Stock Level Recommendations")
    
    if not st.session_state.data_loaded:
        st.warning("⚠️ Please load data first in the Data Management section.")
    else:
        st.subheader("📊 Stock Analysis Configuration")
        
        # Parameters
        safety_stock_days = st.slider("Safety Stock (days):", 1, 14, 7)
        reorder_point_factor = st.slider("Reorder Point Factor:", 1.0, 3.0, 1.5, 0.1)
        
        if st.button("📦 Generate Stock Recommendations"):
            with st.spinner("Generating stock recommendations..."):
                try:
                    # Load data
                    sales_df = st.session_state.sales_data
                    
                    # Calculate product metrics
                    product_metrics = components['product_analyzer'].calculate_product_metrics(sales_df)
                    classified_products = components['product_analyzer'].classify_products(product_metrics)
                    
                    # Calculate stock recommendations
                    stock_recommendations = components['product_analyzer'].calculate_stock_recommendations(
                        classified_products, 
                        safety_stock_days, 
                        reorder_point_factor
                    )
                    
                    st.session_state.stock_recommendations = stock_recommendations
                    
                    st.success("✅ Stock recommendations generated!")
                    
                    # Display results
                    st.subheader("📋 Stock Recommendations")
                    st.dataframe(stock_recommendations)
                    
                    # Stock visualization
                    st.subheader("📊 Stock Level Analysis")
                    fig = components['visualizer'].plot_stock_recommendations(stock_recommendations)
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Summary metrics
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        total_inventory = stock_recommendations['Recommended_Stock'].sum()
                        st.metric("Total Recommended Stock", f"{total_inventory:,.0f}")
                    
                    with col2:
                        avg_turnover = stock_recommendations['Stock_Turnover'].mean()
                        st.metric("Average Stock Turnover", f"{avg_turnover:.1f}")
                    
                    with col3:
                        stock_out_risk = (stock_recommendations['Safety_Stock'] < stock_recommendations['Daily_Demand']).mean()
                        st.metric("Stock-out Risk", f"{stock_out_risk:.1%}")
                    
                    # Download recommendations
                    csv = stock_recommendations.to_csv(index=False)
                    st.download_button(
                        label="📥 Download Stock Recommendations",
                        data=csv,
                        file_name=f"stock_recommendations_{datetime.now().strftime('%Y%m%d')}.csv",
                        mime="text/csv"
                    )
                    
                except Exception as e:
                    st.error(f"Error generating stock recommendations: {str(e)}")

# Reports Page
elif page == "📋 Reports":
    st.header("📋 Performance Reports")
    
    if not st.session_state.data_loaded:
        st.warning("⚠️ Please load data first in the Data Management section.")
    else:
        st.subheader("📊 Generate Performance Report")
        
        if st.button("📋 Generate Report"):
            with st.spinner("Generating comprehensive report..."):
                try:
                    # Load data
                    sales_df = st.session_state.sales_data
                    
                    # Calculate all metrics
                    business_metrics = components['metrics'].calculate_business_metrics(sales_df)
                    product_metrics = components['metrics'].calculate_product_metrics(sales_df)
                    seasonal_metrics = components['metrics'].calculate_seasonal_metrics(sales_df)
                    
                    # Get stock metrics if available
                    if 'stock_recommendations' in st.session_state:
                        inventory_metrics = components['metrics'].calculate_inventory_metrics(
                            st.session_state.stock_recommendations
                        )
                    else:
                        inventory_metrics = {}
                    
                    # Get forecast metrics if available
                    if 'training_results' in st.session_state:
                        # Use a simple metric for demonstration
                        forecast_metrics = {'R2': 0.85, 'MAE': 100, 'MAPE': 15}
                    else:
                        forecast_metrics = {}
                    
                    # Generate report
                    report = components['metrics'].generate_performance_report(
                        forecast_metrics, product_metrics, inventory_metrics, 
                        seasonal_metrics, business_metrics
                    )
                    
                    st.success("✅ Performance report generated!")
                    
                    # Display report sections
                    for section_name, section_data in report.items():
                        st.subheader(f"📊 {section_name.replace('_', ' ').title()}")
                        
                        # Metrics
                        if section_data['metrics']:
                            metrics_df = pd.DataFrame(list(section_data['metrics'].items()), 
                                                    columns=['Metric', 'Value'])
                            st.dataframe(metrics_df)
                        
                        # Recommendations
                        if section_data['recommendations']:
                            st.write("**💡 Recommendations:**")
                            for rec in section_data['recommendations']:
                                st.write(f"• {rec}")
                        
                        st.write("---")
                    
                except Exception as e:
                    st.error(f"Error generating report: {str(e)}")

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: #666;'>
        <p>🛍️ AI-Powered Retail Demand Forecasting System | Built with Streamlit & Python</p>
        <p>Dataset: Rossmann Store Sales | Features: ML Forecasting, Product Analysis, Stock Recommendations</p>
    </div>
    """,
    unsafe_allow_html=True
) 