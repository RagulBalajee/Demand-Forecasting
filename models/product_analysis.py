import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class ProductAnalyzer:
    """Product performance analysis and stock recommendations"""
    
    def __init__(self):
        self.product_metrics = {}
        self.stock_recommendations = {}
        
    def calculate_product_metrics(self, df: pd.DataFrame, 
                                date_col: str = 'Date',
                                product_col: str = 'Product',
                                sales_col: str = 'Sales',
                                store_col: str = 'Store') -> pd.DataFrame:
        """
        Calculate comprehensive product performance metrics
        Args:
            df: Sales dataframe
            date_col: Date column name
            product_col: Product column name
            sales_col: Sales column name
            store_col: Store column name
        Returns:
            DataFrame with product metrics
        """
        # Group by product and calculate metrics
        product_metrics = df.groupby(product_col).agg({
            sales_col: ['sum', 'mean', 'std', 'count'],
            store_col: 'nunique'
        }).reset_index()
        
        # Flatten column names
        product_metrics.columns = [
            f"{col[0]}_{col[1]}" if col[1] else col[0] 
            for col in product_metrics.columns
        ]
        
        # Rename columns for clarity
        product_metrics = product_metrics.rename(columns={
            f'{sales_col}_sum': 'Total_Sales',
            f'{sales_col}_mean': 'Avg_Daily_Sales',
            f'{sales_col}_std': 'Sales_Std',
            f'{sales_col}_count': 'Days_With_Sales',
            f'{store_col}_nunique': 'Stores_Selling'
        })
        
        # Calculate additional metrics
        total_days = df[date_col].nunique()
        product_metrics['Sales_Frequency'] = product_metrics['Days_With_Sales'] / total_days
        product_metrics['Sales_Variability'] = product_metrics['Sales_Std'] / product_metrics['Avg_Daily_Sales']
        product_metrics['Sales_Variability'] = product_metrics['Sales_Variability'].fillna(0)
        
        # Calculate recent performance (last 30 days)
        recent_date = df[date_col].max()
        recent_start = recent_date - timedelta(days=30)
        recent_data = df[df[date_col] >= recent_start]
        
        recent_metrics = recent_data.groupby(product_col)[sales_col].agg(['sum', 'mean']).reset_index()
        recent_metrics.columns = [product_col, 'Recent_30d_Sales', 'Recent_30d_Avg']
        
        product_metrics = product_metrics.merge(recent_metrics, on=product_col, how='left')
        
        # Calculate growth rate
        product_metrics['Growth_Rate'] = (
            product_metrics['Recent_30d_Avg'] / product_metrics['Avg_Daily_Sales']
        ).fillna(1)
        
        # Calculate trend (comparing last 30 days vs previous 30 days)
        if len(df) > 60:  # Need at least 60 days of data
            prev_start = recent_start - timedelta(days=30)
            prev_data = df[(df[date_col] >= prev_start) & (df[date_col] < recent_start)]
            
            prev_metrics = prev_data.groupby(product_col)[sales_col].mean().reset_index()
            prev_metrics.columns = [product_col, 'Prev_30d_Avg']
            
            product_metrics = product_metrics.merge(prev_metrics, on=product_col, how='left')
            product_metrics['Trend_Ratio'] = (
                product_metrics['Recent_30d_Avg'] / product_metrics['Prev_30d_Avg']
            ).fillna(1)
        else:
            product_metrics['Trend_Ratio'] = 1.0
        
        self.product_metrics = product_metrics
        return product_metrics
    
    def classify_products(self, product_metrics: pd.DataFrame,
                         sales_threshold: float = None,
                         frequency_threshold: float = 0.5,
                         growth_threshold: float = 1.1) -> pd.DataFrame:
        """
        Classify products into categories (Hot, Normal, Slow-moving)
        Args:
            product_metrics: Product metrics dataframe
            sales_threshold: Sales threshold for classification (if None, uses median)
            frequency_threshold: Frequency threshold for slow-moving products
            growth_threshold: Growth threshold for hot products
        Returns:
            DataFrame with product classifications
        """
        df = product_metrics.copy()
        
        # Set thresholds if not provided
        if sales_threshold is None:
            sales_threshold = df['Total_Sales'].median()
        
        # Classify products
        conditions = [
            # Hot products: High sales, high growth, good frequency
            (df['Total_Sales'] > sales_threshold) & 
            (df['Sales_Frequency'] > frequency_threshold) & 
            (df['Growth_Rate'] > growth_threshold),
            
            # Slow-moving products: Low sales or low frequency
            (df['Total_Sales'] < sales_threshold * 0.5) | 
            (df['Sales_Frequency'] < frequency_threshold * 0.5)
        ]
        
        choices = ['Hot', 'Slow-moving']
        df['Product_Category'] = np.select(conditions, choices, default='Normal')
        
        # Add subcategories
        df['Subcategory'] = 'Standard'
        
        # Hot products subcategories
        hot_mask = df['Product_Category'] == 'Hot'
        df.loc[hot_mask & (df['Growth_Rate'] > 1.5), 'Subcategory'] = 'Trending'
        df.loc[hot_mask & (df['Sales_Frequency'] > 0.8), 'Subcategory'] = 'Stable_High'
        
        # Slow-moving products subcategories
        slow_mask = df['Product_Category'] == 'Slow-moving'
        df.loc[slow_mask & (df['Sales_Frequency'] < 0.2), 'Subcategory'] = 'Very_Slow'
        df.loc[slow_mask & (df['Total_Sales'] < sales_threshold * 0.2), 'Subcategory'] = 'Low_Volume'
        
        return df
    
    def calculate_stock_recommendations(self, product_metrics: pd.DataFrame,
                                      safety_stock_days: int = 7,
                                      reorder_point_factor: float = 1.5) -> pd.DataFrame:
        """
        Calculate stock level recommendations
        Args:
            product_metrics: Product metrics dataframe
            safety_stock_days: Number of days for safety stock
            reorder_point_factor: Factor for reorder point calculation
        Returns:
            DataFrame with stock recommendations
        """
        df = product_metrics.copy()
        
        # Calculate daily demand (average daily sales)
        df['Daily_Demand'] = df['Avg_Daily_Sales']
        
        # Calculate safety stock
        df['Safety_Stock'] = df['Daily_Demand'] * safety_stock_days * df['Sales_Variability']
        
        # Calculate reorder point
        df['Reorder_Point'] = (df['Daily_Demand'] * reorder_point_factor + df['Safety_Stock']).round()
        
        # Calculate economic order quantity (simplified)
        annual_demand = df['Daily_Demand'] * 365
        order_cost = 50  # Fixed order cost
        holding_cost_rate = 0.2  # 20% of unit cost
        unit_cost = 10  # Assumed unit cost
        
        df['EOQ'] = np.sqrt((2 * annual_demand * order_cost) / (unit_cost * holding_cost_rate)).round()
        
        # Calculate maximum stock level
        df['Max_Stock'] = (df['Reorder_Point'] + df['EOQ']).round()
        
        # Calculate recommended stock levels based on product category
        df['Recommended_Stock'] = df['Max_Stock']
        
        # Adjust for product categories
        hot_mask = df['Product_Category'] == 'Hot'
        slow_mask = df['Product_Category'] == 'Slow-moving'
        
        # Hot products: increase stock levels
        df.loc[hot_mask, 'Recommended_Stock'] = (df.loc[hot_mask, 'Max_Stock'] * 1.2).round()
        
        # Slow-moving products: reduce stock levels
        df.loc[slow_mask, 'Recommended_Stock'] = (df.loc[slow_mask, 'Max_Stock'] * 0.7).round()
        
        # Calculate stock turnover
        df['Stock_Turnover'] = (annual_demand / df['Recommended_Stock']).round(2)
        
        # Add stock status
        df['Stock_Status'] = 'Normal'
        df.loc[df['Stock_Turnover'] < 12, 'Stock_Status'] = 'Slow_Turnover'
        df.loc[df['Stock_Turnover'] > 24, 'Stock_Status'] = 'Fast_Turnover'
        
        return df
    
    def get_product_insights(self, df: pd.DataFrame, 
                           product_metrics: pd.DataFrame,
                           top_n: int = 10) -> Dict[str, pd.DataFrame]:
        """
        Get key product insights
        Args:
            df: Original sales dataframe
            product_metrics: Product metrics dataframe
            top_n: Number of top products to return
        Returns:
            Dictionary with various product insights
        """
        insights = {}
        
        # Top performing products
        insights['top_products'] = product_metrics.nlargest(top_n, 'Total_Sales')[
            ['Product', 'Total_Sales', 'Avg_Daily_Sales', 'Product_Category', 'Growth_Rate']
        ]
        
        # Fastest growing products
        insights['fastest_growing'] = product_metrics.nlargest(top_n, 'Growth_Rate')[
            ['Product', 'Growth_Rate', 'Recent_30d_Avg', 'Avg_Daily_Sales', 'Product_Category']
        ]
        
        # Most consistent products (low variability)
        insights['most_consistent'] = product_metrics.nsmallest(top_n, 'Sales_Variability')[
            ['Product', 'Sales_Variability', 'Sales_Frequency', 'Avg_Daily_Sales', 'Product_Category']
        ]
        
        # Slow-moving products
        slow_products = product_metrics[product_metrics['Product_Category'] == 'Slow-moving']
        insights['slow_moving'] = slow_products.nlargest(top_n, 'Total_Sales')[
            ['Product', 'Total_Sales', 'Sales_Frequency', 'Growth_Rate', 'Subcategory']
        ]
        
        # Hot products
        hot_products = product_metrics[product_metrics['Product_Category'] == 'Hot']
        insights['hot_products'] = hot_products.nlargest(top_n, 'Total_Sales')[
            ['Product', 'Total_Sales', 'Growth_Rate', 'Sales_Frequency', 'Subcategory']
        ]
        
        return insights
    
    def generate_recommendations(self, product_metrics: pd.DataFrame) -> Dict[str, List[str]]:
        """
        Generate actionable recommendations based on product analysis
        Args:
            product_metrics: Product metrics dataframe
        Returns:
            Dictionary with recommendations by category
        """
        recommendations = {
            'hot_products': [],
            'slow_moving': [],
            'general': []
        }
        
        # Hot products recommendations
        hot_products = product_metrics[product_metrics['Product_Category'] == 'Hot']
        if len(hot_products) > 0:
            recommendations['hot_products'].append(
                f"Focus on {len(hot_products)} hot products with high growth potential"
            )
            recommendations['hot_products'].append(
                "Consider increasing stock levels and marketing efforts for trending products"
            )
        
        # Slow-moving products recommendations
        slow_products = product_metrics[product_metrics['Product_Category'] == 'Slow-moving']
        if len(slow_products) > 0:
            recommendations['slow_moving'].append(
                f"Review {len(slow_products)} slow-moving products for potential discontinuation"
            )
            recommendations['slow_moving'].append(
                "Consider promotional campaigns to clear slow-moving inventory"
            )
        
        # General recommendations
        total_products = len(product_metrics)
        hot_ratio = len(hot_products) / total_products
        slow_ratio = len(slow_products) / total_products
        
        recommendations['general'].append(
            f"Product portfolio: {hot_ratio:.1%} hot, {slow_ratio:.1%} slow-moving, {1-hot_ratio-slow_ratio:.1%} normal"
        )
        
        return recommendations 