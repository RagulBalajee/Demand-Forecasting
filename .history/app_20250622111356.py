import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta
import warnings
import os

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
    page_title="AI Retail Forecasting",
    page_icon="🛍️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for a more polished look
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #2E86C1;
        text-align: center;
        margin-bottom: 1rem;
    }
    .stMetric {
        background-color: #FFFFFF;
        border: 1px solid #E0E0E0;
        border-radius: 0.5rem;
        padding: 1rem;
        box-shadow: 0 4px 6px rgba(0,0,0,0.04);
    }
    .stButton>button {
        background-color: #2E86C1;
        color: white;
        border-radius: 0.5rem;
        border: none;
        padding: 0.5rem 1rem;
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

def display_dashboard():
    st.header("📊 Retail Analytics Dashboard")
    
    if not st.session_state.data_loaded:
        st.warning("⚠️ Please load data first in the Data Management section.")
        return

    sales_df = st.session_state.sales_data
    store_df = st.session_state.store_data
    dataset_name = st.session_state.dataset_name

    st.info(f"**Displaying Dashboard for: `{dataset_name}`**")

    # Metrics
    total_sales = sales_df['Sales'].sum()
    num_stores = sales_df['Store'].nunique()
    
    if 'Product' in sales_df.columns:
        num_products = sales_df['Product'].nunique()
    else:
        num_products = "N/A"
        
    if 'Customers' in sales_df.columns:
        total_customers = sales_df['Customers'].sum()
    else:
        total_customers = "N/A"

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Revenue", f"${total_sales:,.0f}", "📈")
    col2.metric("Total Stores", f"{num_stores}", "🏪")
    col3.metric("Total Products", f"{num_products}", "📦")
    col4.metric("Total Customers", f"{total_customers:,.0f}" if isinstance(total_customers, (int, float)) else "N/A", "👥")

    # Visualizations
    st.subheader("Sales Trends")
    agg_sales = sales_df.groupby('Date')['Sales'].sum().reset_index()
    fig = components['visualizer'].plot_sales_trend(agg_sales, date_col='Date', sales_col='Sales')
    st.plotly_chart(fig, use_container_width=True)
    
    # More visualizations can be added here
    
def data_management_page():
    st.header("📊 Data Management")
    
    data_source_options = ["📊 Use Sample Data", "🗂️ Use Rossmann Dataset"]
    if os.path.exists('data/rossmann-store-sales'):
        st.success("✅ Rossmann dataset found!")
    else:
        st.warning("Rossmann dataset not found at `data/rossmann-store-sales`.")
        data_source_options.pop(1)

    data_source = st.radio("Choose data source:", data_source_options)

    if st.button(f"Load {data_source.split(' ')[2]} Data"):
        with st.spinner("Loading data..."):
            try:
                if "Sample" in data_source:
                    sales_df, store_df = components['data_loader'].load_sample_data()
                    st.session_state.dataset_name = "Sample Data"
                elif "Rossmann" in data_source:
                    sales_df, store_df = components['data_loader'].load_rossmann_data()
                    st.session_state.dataset_name = "Rossmann Store Sales"

                st.session_state.sales_data = sales_df
                st.session_state.store_data = store_df
                st.session_state.data_loaded = True
                st.success("✅ Data loaded successfully!")
            except Exception as e:
                st.error(f"Error loading data: {str(e)}")

    if st.session_state.data_loaded:
        st.subheader("📋 Data Preview")
        sales_df = st.session_state.sales_data
        store_df = st.session_state.store_data
        
        col1, col2 = st.columns(2)
        with col1:
            st.write("**Sales Data:**")
            st.dataframe(sales_df.head())
            st.write(f"Shape: {sales_df.shape}")
        with col2:
            st.write("**Store Data:**")
            st.dataframe(store_df.head())
            st.write(f"Shape: {store_df.shape}")

def main():
    """Main function to run the Streamlit app"""
    init_session_state()
    components = initialize_components()
    
    st.sidebar.title("🛍️ Retail Forecasting")
    
    # Automatically load Rossmann data if available and no other data is loaded
    if not st.session_state.data_loaded and os.path.exists('data/rossmann-store-sales'):
        try:
            sales_df, store_df = components['data_loader'].load_rossmann_data()
            st.session_state.sales_data = sales_df
            st.session_state.store_data = store_df
            st.session_state.data_loaded = True
            st.session_state.dataset_name = "Rossmann Store Sales (Auto-loaded)"
        except Exception as e:
            st.sidebar.error(f"Auto-load failed: {e}")

    # Sidebar Navigation
    page_options = {
        "🏠 Dashboard": display_dashboard,
        "📊 Data Management": data_management_page,
        # "🤖 Model Training": model_training_page,
        # "🔮 Forecasting": forecasting_page,
        # "📈 Product Analysis": product_analysis_page,
        # "📦 Stock Recommendations": stock_recommendations_page,
        # "📋 Reports": reports_page
    }
    
    st.sidebar.title("Navigation")
    selection = st.sidebar.radio("Go to", list(page_options.keys()))
    
    # Page Header
    st.markdown(f'<h1 class="main-header">{selection}</h1>', unsafe_allow_html=True)

    # Display selected page
    page = page_options[selection]
    page()

if __name__ == "__main__":
    main() 