import pandas as pd
import numpy as np
from typing import Tuple, Optional
import requests
import io
import zipfile
import os

class DataLoader:
    """Data loader for retail sales forecasting"""
    
    def __init__(self):
        self.sales_data = None
        self.store_data = None
        
    def load_sample_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Load sample Rossmann store sales data
        Returns tuple of (sales_data, store_data)
        """
        # Create sample sales data similar to Rossmann dataset
        np.random.seed(42)
        
        # Generate sample sales data
        dates = pd.date_range('2022-01-01', '2023-12-31', freq='D')
        stores = range(1, 11)  # 10 stores
        products = range(1, 21)  # 20 products
        
        sales_records = []
        for date in dates:
            for store in stores:
                for product in products:
                    # Base sales with seasonality and trends
                    base_sales = np.random.poisson(50)
                    
                    # Weekend effect
                    weekend_boost = 1.3 if date.weekday() >= 5 else 1.0
                    
                    # Monthly seasonality
                    monthly_effect = 1.2 if date.month in [11, 12] else 0.8 if date.month in [1, 2] else 1.0
                    
                    # Store-specific effects
                    store_effect = np.random.normal(1.0, 0.2)
                    
                    # Product-specific effects
                    product_effect = np.random.normal(1.0, 0.3)
                    
                    # Calculate final sales
                    final_sales = int(base_sales * weekend_boost * monthly_effect * store_effect * product_effect)
                    
                    sales_records.append({
                        'Date': date,
                        'Store': store,
                        'Product': product,
                        'Sales': max(0, final_sales),
                        'Customers': max(0, int(final_sales * np.random.uniform(0.8, 1.2))),
                        'Promo': np.random.choice([0, 1], p=[0.7, 0.3]),
                        'SchoolHoliday': 1 if date.month in [7, 8] or date.weekday() >= 5 else 0,
                        'StateHoliday': 1 if date.month == 12 and date.day == 25 else 0
                    })
        
        sales_df = pd.DataFrame(sales_records)
        
        # Create store information
        store_records = []
        for store in stores:
            store_records.append({
                'Store': store,
                'StoreType': np.random.choice(['a', 'b', 'c', 'd']),
                'Assortment': np.random.choice(['a', 'b', 'c']),
                'CompetitionDistance': np.random.uniform(100, 5000),
                'CompetitionOpenSinceMonth': np.random.randint(1, 13),
                'CompetitionOpenSinceYear': np.random.randint(2010, 2023),
                'Promo2': np.random.choice([0, 1]),
                'Promo2SinceWeek': np.random.randint(1, 53) if np.random.choice([0, 1]) else 0,
                'Promo2SinceYear': np.random.randint(2010, 2023) if np.random.choice([0, 1]) else 0,
                'PromoInterval': np.random.choice(['Jan,Apr,Jul,Oct', 'Feb,May,Aug,Nov', 'Mar,Jun,Sept,Dec', ''])
            })
        
        store_df = pd.DataFrame(store_records)
        
        self.sales_data = sales_df
        self.store_data = store_df
        
        return sales_df, store_df
    
    def load_from_csv(self, sales_file: str, store_file: Optional[str] = None) -> Tuple[pd.DataFrame, Optional[pd.DataFrame]]:
        """
        Load data from CSV files
        Args:
            sales_file: Path to sales CSV file
            store_file: Path to store CSV file (optional)
        Returns:
            Tuple of (sales_data, store_data)
        """
        try:
            sales_df = pd.read_csv(sales_file)
            sales_df['Date'] = pd.to_datetime(sales_df['Date'])
            
            store_df = None
            if store_file:
                store_df = pd.read_csv(store_file)
            
            self.sales_data = sales_df
            self.store_data = store_df
            
            return sales_df, store_df
            
        except Exception as e:
            raise Exception(f"Error loading data: {str(e)}")
    
    def preprocess_rossmann_data(self, sales_df: pd.DataFrame, store_df: pd.DataFrame) -> pd.DataFrame:
        """
        Preprocess Rossmann store sales data for modeling
        Args:
            sales_df: Sales dataframe
            store_df: Store dataframe
        Returns:
            Preprocessed dataframe
        """
        df = sales_df.copy()
        
        # Extract date features
        df['Year'] = df['Date'].dt.year
        df['Month'] = df['Date'].dt.month
        df['Day'] = df['Date'].dt.day
        df['DayOfWeek'] = df['Date'].dt.dayofweek
        df['WeekOfYear'] = df['Date'].dt.isocalendar().week
        df['Quarter'] = df['Date'].dt.quarter
        
        # Create lag features (store-level, no product)
        df = df.sort_values(['Store', 'Date'])
        df['Sales_Lag1'] = df.groupby('Store')['Sales'].shift(1)
        df['Sales_Lag7'] = df.groupby('Store')['Sales'].shift(7)
        df['Sales_Lag30'] = df.groupby('Store')['Sales'].shift(30)
        
        # Rolling averages
        df['Sales_MA7'] = df.groupby('Store')['Sales'].rolling(7).mean().values
        df['Sales_MA30'] = df.groupby('Store')['Sales'].rolling(30).mean().values
        
        # Fill NaN values
        df = df.fillna(0)
        
        # Merge with store data
        df = df.merge(store_df, on='Store', how='left')
        
        # Encode categorical variables
        if 'StoreType' in df.columns:
            df['StoreType_Encoded'] = pd.Categorical(df['StoreType']).codes
        if 'Assortment' in df.columns:
            df['Assortment_Encoded'] = pd.Categorical(df['Assortment']).codes
        if 'PromoInterval' in df.columns:
            df['PromoInterval_Encoded'] = pd.Categorical(df['PromoInterval']).codes
        
        # Handle StateHoliday (can be string or numeric)
        if 'StateHoliday' in df.columns:
            if df['StateHoliday'].dtype == 'object':
                df['StateHoliday_Encoded'] = pd.Categorical(df['StateHoliday']).codes
            else:
                df['StateHoliday_Encoded'] = df['StateHoliday']
        
        return df

    def preprocess_data(self, sales_df: pd.DataFrame, store_df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """
        Preprocess the sales data for modeling
        Args:
            sales_df: Sales dataframe
            store_df: Store dataframe (optional)
        Returns:
            Preprocessed dataframe
        """
        # Check if this is Rossmann data (has Store column but no Product column)
        if 'Store' in sales_df.columns and 'Product' not in sales_df.columns:
            if store_df is not None:
                return self.preprocess_rossmann_data(sales_df, store_df)
            else:
                # Create dummy store data for Rossmann format
                stores = sales_df['Store'].unique()
                dummy_store_df = pd.DataFrame({
                    'Store': stores,
                    'StoreType': 'a',
                    'Assortment': 'a',
                    'CompetitionDistance': 1000,
                    'CompetitionOpenSinceMonth': 1,
                    'CompetitionOpenSinceYear': 2010,
                    'Promo2': 0,
                    'Promo2SinceWeek': 0,
                    'Promo2SinceYear': 0,
                    'PromoInterval': ''
                })
                return self.preprocess_rossmann_data(sales_df, dummy_store_df)
        
        # Original preprocessing for sample data with Product column
        df = sales_df.copy()
        
        # Extract date features
        df['Year'] = df['Date'].dt.year
        df['Month'] = df['Date'].dt.month
        df['Day'] = df['Date'].dt.day
        df['DayOfWeek'] = df['Date'].dt.dayofweek
        df['WeekOfYear'] = df['Date'].dt.isocalendar().week
        df['Quarter'] = df['Date'].dt.quarter
        
        # Create lag features
        df = df.sort_values(['Store', 'Product', 'Date'])
        df['Sales_Lag1'] = df.groupby(['Store', 'Product'])['Sales'].shift(1)
        df['Sales_Lag7'] = df.groupby(['Store', 'Product'])['Sales'].shift(7)
        df['Sales_Lag30'] = df.groupby(['Store', 'Product'])['Sales'].shift(30)
        
        # Rolling averages
        df['Sales_MA7'] = df.groupby(['Store', 'Product'])['Sales'].rolling(7).mean().values
        df['Sales_MA30'] = df.groupby(['Store', 'Product'])['Sales'].rolling(30).mean().values
        
        # Fill NaN values
        df = df.fillna(0)
        
        # Merge with store data if available
        if store_df is not None:
            df = df.merge(store_df, on='Store', how='left')
            
            # Encode categorical variables
            if 'StoreType' in df.columns:
                df['StoreType_Encoded'] = pd.Categorical(df['StoreType']).codes
            if 'Assortment' in df.columns:
                df['Assortment_Encoded'] = pd.Categorical(df['Assortment']).codes
        
        return df
    
    def get_aggregated_data(self, df: pd.DataFrame, group_by: str = 'Date') -> pd.DataFrame:
        """
        Aggregate data by date or other grouping
        Args:
            df: Input dataframe
            group_by: Column to group by (default: 'Date')
        Returns:
            Aggregated dataframe
        """
        if group_by == 'Date':
            agg_df = df.groupby('Date').agg({
                'Sales': 'sum',
                'Customers': 'sum',
                'Promo': 'mean',
                'SchoolHoliday': 'mean',
                'StateHoliday': 'mean'
            }).reset_index()
        elif group_by == 'Product':
            agg_df = df.groupby('Product').agg({
                'Sales': 'sum',
                'Customers': 'sum',
                'Promo': 'mean'
            }).reset_index()
        elif group_by == 'Store':
            agg_df = df.groupby('Store').agg({
                'Sales': 'sum',
                'Customers': 'sum',
                'Promo': 'mean'
            }).reset_index()
        else:
            raise ValueError(f"Unsupported group_by: {group_by}")
        
        return agg_df
    
    def load_rossmann_data(self, data_dir: str = 'data/rossmann-store-sales') -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Load actual Rossmann Store Sales dataset
        Args:
            data_dir: Directory containing the Rossmann dataset files
        Returns:
            Tuple of (sales_data, store_data)
        """
        try:
            # Load sales data
            sales_file = f"{data_dir}/train.csv"
            sales_df = pd.read_csv(sales_file)
            sales_df['Date'] = pd.to_datetime(sales_df['Date'])
            
            # Load store data
            store_file = f"{data_dir}/store.csv"
            store_df = pd.read_csv(store_file)
            
            # Load test data for future predictions
            test_file = f"{data_dir}/test.csv"
            if os.path.exists(test_file):
                test_df = pd.read_csv(test_file)
                test_df['Date'] = pd.to_datetime(test_df['Date'])
                # Add dummy Sales column for test data
                test_df['Sales'] = 0
                test_df['Customers'] = 0
                test_df['Open'] = 1
                test_df['Promo'] = 0
                test_df['StateHoliday'] = 0
                test_df['SchoolHoliday'] = 0
                
                # Combine train and test data
                sales_df = pd.concat([sales_df, test_df], ignore_index=True)
            
            self.sales_data = sales_df
            self.store_data = store_df
            
            return sales_df, store_df
            
        except FileNotFoundError as e:
            raise FileNotFoundError(f"Rossmann dataset not found in {data_dir}. Please ensure the dataset files are in the correct location.")
        except Exception as e:
            raise Exception(f"Error loading Rossmann data: {str(e)}")
    
    def load_rossmann_test_data(self, data_dir: str = 'data/rossmann-store-sales') -> pd.DataFrame:
        """
        Load Rossmann test data for predictions
        Args:
            data_dir: Directory containing the Rossmann dataset files
        Returns:
            Test dataframe
        """
        try:
            test_file = f"{data_dir}/test.csv"
            test_df = pd.read_csv(test_file)
            test_df['Date'] = pd.to_datetime(test_df['Date'])
            
            # Add required columns for feature engineering
            test_df['Sales'] = 0  # Will be predicted
            test_df['Customers'] = 0
            test_df['Open'] = 1
            test_df['Promo'] = 0
            test_df['StateHoliday'] = 0
            test_df['SchoolHoliday'] = 0
            
            return test_df
            
        except FileNotFoundError:
            raise FileNotFoundError(f"Test file not found in {data_dir}")
        except Exception as e:
            raise Exception(f"Error loading test data: {str(e)}") 