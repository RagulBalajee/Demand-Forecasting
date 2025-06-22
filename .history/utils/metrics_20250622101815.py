import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

class RetailMetrics:
    """Performance metrics for retail demand forecasting"""
    
    def __init__(self):
        self.metrics = {}
    
    def calculate_forecast_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """
        Calculate comprehensive forecasting metrics
        Args:
            y_true: True values
            y_pred: Predicted values
        Returns:
            Dictionary of metrics
        """
        # Basic metrics
        mae = mean_absolute_error(y_true, y_pred)
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_true, y_pred)
        
        # Additional metrics
        mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
        smape = 100 * np.mean(2 * np.abs(y_pred - y_true) / (np.abs(y_true) + np.abs(y_pred)))
        
        # Bias metrics
        bias = np.mean(y_pred - y_true)
        bias_percentage = (bias / np.mean(y_true)) * 100
        
        # Directional accuracy
        directional_accuracy = np.mean(np.sign(np.diff(y_true)) == np.sign(np.diff(y_pred)))
        
        metrics = {
            'MAE': mae,
            'MSE': mse,
            'RMSE': rmse,
            'R2': r2,
            'MAPE': mape,
            'SMAPE': smape,
            'Bias': bias,
            'Bias_Percentage': bias_percentage,
            'Directional_Accuracy': directional_accuracy
        }
        
        return metrics
    
    def calculate_product_metrics(self, df: pd.DataFrame,
                                product_col: str = 'Product',
                                sales_col: str = 'Sales',
                                date_col: str = 'Date') -> Dict[str, float]:
        """
        Calculate product portfolio metrics
        Args:
            df: Sales dataframe
            product_col: Product column name
            sales_col: Sales column name
            date_col: Date column name
        Returns:
            Dictionary of product metrics
        """
        # Total sales
        total_sales = df[sales_col].sum()
        
        # Number of products
        num_products = df[product_col].nunique()
        
        # Average daily sales per product
        avg_daily_sales_per_product = df.groupby([date_col, product_col])[sales_col].sum().groupby(product_col).mean().mean()
        
        # Sales concentration (top 20% of products)
        product_sales = df.groupby(product_col)[sales_col].sum().sort_values(ascending=False)
        top_20_percent = int(len(product_sales) * 0.2)
        concentration = product_sales.head(top_20_percent).sum() / total_sales
        
        # Product turnover rate
        total_days = df[date_col].nunique()
        product_turnover = num_products / total_days
        
        # Sales variability across products
        product_std = df.groupby(product_col)[sales_col].sum().std()
        product_cv = product_std / product_sales.mean()
        
        metrics = {
            'Total_Sales': total_sales,
            'Number_of_Products': num_products,
            'Avg_Daily_Sales_Per_Product': avg_daily_sales_per_product,
            'Sales_Concentration_Top20': concentration,
            'Product_Turnover_Rate': product_turnover,
            'Product_Sales_CV': product_cv
        }
        
        return metrics
    
    def calculate_inventory_metrics(self, stock_df: pd.DataFrame) -> Dict[str, float]:
        """
        Calculate inventory performance metrics
        Args:
            stock_df: Stock recommendations dataframe
        Returns:
            Dictionary of inventory metrics
        """
        # Total inventory value
        total_inventory_value = (stock_df['Recommended_Stock'] * 10).sum()  # Assuming $10 per unit
        
        # Average stock turnover
        avg_stock_turnover = stock_df['Stock_Turnover'].mean()
        
        # Inventory efficiency
        inventory_efficiency = stock_df['Daily_Demand'].sum() / stock_df['Recommended_Stock'].sum()
        
        # Stock-out risk (products with low safety stock)
        low_safety_stock = (stock_df['Safety_Stock'] < stock_df['Daily_Demand']).sum()
        stock_out_risk = low_safety_stock / len(stock_df)
        
        # Overstock risk (products with high stock levels)
        overstock_threshold = stock_df['Daily_Demand'] * 30  # 30 days of demand
        overstock_products = (stock_df['Recommended_Stock'] > overstock_threshold).sum()
        overstock_risk = overstock_products / len(stock_df)
        
        # Inventory distribution by category
        hot_inventory = stock_df[stock_df['Product_Category'] == 'Hot']['Recommended_Stock'].sum()
        slow_inventory = stock_df[stock_df['Product_Category'] == 'Slow-moving']['Recommended_Stock'].sum()
        normal_inventory = stock_df[stock_df['Product_Category'] == 'Normal']['Recommended_Stock'].sum()
        
        total_inventory = stock_df['Recommended_Stock'].sum()
        
        metrics = {
            'Total_Inventory_Value': total_inventory_value,
            'Avg_Stock_Turnover': avg_stock_turnover,
            'Inventory_Efficiency': inventory_efficiency,
            'Stock_Out_Risk': stock_out_risk,
            'Overstock_Risk': overstock_risk,
            'Hot_Inventory_Ratio': hot_inventory / total_inventory,
            'Slow_Inventory_Ratio': slow_inventory / total_inventory,
            'Normal_Inventory_Ratio': normal_inventory / total_inventory
        }
        
        return metrics
    
    def calculate_seasonal_metrics(self, df: pd.DataFrame,
                                 date_col: str = 'Date',
                                 sales_col: str = 'Sales') -> Dict[str, float]:
        """
        Calculate seasonal performance metrics
        Args:
            df: Sales dataframe
            date_col: Date column name
            sales_col: Sales column name
        Returns:
            Dictionary of seasonal metrics
        """
        # Add seasonal columns
        df_analysis = df.copy()
        df_analysis['Month'] = df_analysis[date_col].dt.month
        df_analysis['Season'] = df_analysis[date_col].dt.month.map({
            12: 'Winter', 1: 'Winter', 2: 'Winter',
            3: 'Spring', 4: 'Spring', 5: 'Spring',
            6: 'Summer', 7: 'Summer', 8: 'Summer',
            9: 'Fall', 10: 'Fall', 11: 'Fall'
        })
        
        # Seasonal sales
        seasonal_sales = df_analysis.groupby('Season')[sales_col].sum()
        total_sales = seasonal_sales.sum()
        
        # Seasonal concentration
        seasonal_concentration = seasonal_sales.max() / total_sales
        
        # Seasonal variability
        seasonal_cv = seasonal_sales.std() / seasonal_sales.mean()
        
        # Peak season performance
        peak_season = seasonal_sales.idxmax()
        peak_season_sales = seasonal_sales.max()
        
        # Off-peak season performance
        off_peak_season = seasonal_sales.idxmin()
        off_peak_season_sales = seasonal_sales.min()
        
        # Peak to off-peak ratio
        peak_off_peak_ratio = peak_season_sales / off_peak_season_sales
        
        metrics = {
            'Seasonal_Concentration': seasonal_concentration,
            'Seasonal_Variability': seasonal_cv,
            'Peak_Season': peak_season,
            'Peak_Season_Sales': peak_season_sales,
            'Off_Peak_Season': off_peak_season,
            'Off_Peak_Season_Sales': off_peak_season_sales,
            'Peak_Off_Peak_Ratio': peak_off_peak_ratio
        }
        
        return metrics
    
    def calculate_business_metrics(self, df: pd.DataFrame,
                                 sales_col: str = 'Sales',
                                 date_col: str = 'Date') -> Dict[str, float]:
        """
        Calculate business performance metrics
        Args:
            df: Sales dataframe
            sales_col: Sales column name
            date_col: Date column name
        Returns:
            Dictionary of business metrics
        """
        # Revenue metrics
        total_revenue = df[sales_col].sum()
        avg_daily_revenue = df.groupby(date_col)[sales_col].sum().mean()
        revenue_growth = self._calculate_growth_rate(df, sales_col, date_col)
        
        # Customer metrics (if available)
        if 'Customers' in df.columns:
            total_customers = df['Customers'].sum()
            avg_customers_per_day = df.groupby(date_col)['Customers'].sum().mean()
            avg_transaction_value = total_revenue / total_customers
        else:
            total_customers = 0
            avg_customers_per_day = 0
            avg_transaction_value = 0
        
        # Promotional effectiveness (if available)
        if 'Promo' in df.columns:
            promo_days = df['Promo'].sum()
            total_days = len(df[date_col].unique())
            promo_frequency = promo_days / total_days
            
            promo_sales = df[df['Promo'] == 1][sales_col].sum()
            non_promo_sales = df[df['Promo'] == 0][sales_col].sum()
            promo_effectiveness = (promo_sales / promo_days) / (non_promo_sales / (total_days - promo_days))
        else:
            promo_frequency = 0
            promo_effectiveness = 0
        
        # Time-based metrics
        date_range = df[date_col].max() - df[date_col].min()
        days_in_period = date_range.days
        
        metrics = {
            'Total_Revenue': total_revenue,
            'Avg_Daily_Revenue': avg_daily_revenue,
            'Revenue_Growth_Rate': revenue_growth,
            'Total_Customers': total_customers,
            'Avg_Customers_Per_Day': avg_customers_per_day,
            'Avg_Transaction_Value': avg_transaction_value,
            'Promo_Frequency': promo_frequency,
            'Promo_Effectiveness': promo_effectiveness,
            'Days_in_Period': days_in_period
        }
        
        return metrics
    
    def _calculate_growth_rate(self, df: pd.DataFrame, sales_col: str, date_col: str) -> float:
        """
        Calculate growth rate between first and last period
        Args:
            df: Sales dataframe
            sales_col: Sales column name
            date_col: Date column name
        Returns:
            Growth rate as percentage
        """
        # Get first and last month of data
        monthly_sales = df.groupby(df[date_col].dt.to_period('M'))[sales_col].sum()
        
        if len(monthly_sales) < 2:
            return 0.0
        
        first_month_sales = monthly_sales.iloc[0]
        last_month_sales = monthly_sales.iloc[-1]
        
        if first_month_sales == 0:
            return 0.0
        
        growth_rate = ((last_month_sales - first_month_sales) / first_month_sales) * 100
        return growth_rate
    
    def generate_performance_report(self, forecast_metrics: Dict,
                                  product_metrics: Dict,
                                  inventory_metrics: Dict,
                                  seasonal_metrics: Dict,
                                  business_metrics: Dict) -> Dict[str, Dict]:
        """
        Generate comprehensive performance report
        Args:
            forecast_metrics: Forecasting performance metrics
            product_metrics: Product portfolio metrics
            inventory_metrics: Inventory performance metrics
            seasonal_metrics: Seasonal performance metrics
            business_metrics: Business performance metrics
        Returns:
            Dictionary with categorized performance report
        """
        report = {
            'forecasting_performance': {
                'summary': 'Forecasting model performance metrics',
                'metrics': forecast_metrics,
                'recommendations': self._get_forecast_recommendations(forecast_metrics)
            },
            'product_portfolio': {
                'summary': 'Product portfolio analysis',
                'metrics': product_metrics,
                'recommendations': self._get_product_recommendations(product_metrics)
            },
            'inventory_management': {
                'summary': 'Inventory performance analysis',
                'metrics': inventory_metrics,
                'recommendations': self._get_inventory_recommendations(inventory_metrics)
            },
            'seasonal_analysis': {
                'summary': 'Seasonal performance patterns',
                'metrics': seasonal_metrics,
                'recommendations': self._get_seasonal_recommendations(seasonal_metrics)
            },
            'business_performance': {
                'summary': 'Overall business performance',
                'metrics': business_metrics,
                'recommendations': self._get_business_recommendations(business_metrics)
            }
        }
        
        return report
    
    def _get_forecast_recommendations(self, metrics: Dict) -> List[str]:
        """Get recommendations based on forecasting metrics"""
        recommendations = []
        
        if metrics.get('MAPE', 0) > 20:
            recommendations.append("High MAPE indicates poor forecast accuracy. Consider feature engineering or model tuning.")
        
        if metrics.get('Bias_Percentage', 0) > 10:
            recommendations.append("Significant forecast bias detected. Model may be systematically over/under-predicting.")
        
        if metrics.get('Directional_Accuracy', 0) < 0.6:
            recommendations.append("Low directional accuracy. Model struggles to predict trend changes.")
        
        if metrics.get('R2', 0) < 0.5:
            recommendations.append("Low R² score. Consider adding more relevant features or using different models.")
        
        return recommendations
    
    def _get_product_recommendations(self, metrics: Dict) -> List[str]:
        """Get recommendations based on product metrics"""
        recommendations = []
        
        if metrics.get('Sales_Concentration_Top20', 0) > 0.8:
            recommendations.append("High sales concentration in top 20% of products. Consider diversifying product portfolio.")
        
        if metrics.get('Product_Sales_CV', 0) > 2:
            recommendations.append("High product sales variability. Consider standardizing product offerings.")
        
        return recommendations
    
    def _get_inventory_recommendations(self, metrics: Dict) -> List[str]:
        """Get recommendations based on inventory metrics"""
        recommendations = []
        
        if metrics.get('Stock_Out_Risk', 0) > 0.2:
            recommendations.append("High stock-out risk. Consider increasing safety stock levels.")
        
        if metrics.get('Overstock_Risk', 0) > 0.3:
            recommendations.append("High overstock risk. Consider reducing stock levels for slow-moving products.")
        
        if metrics.get('Avg_Stock_Turnover', 0) < 12:
            recommendations.append("Low stock turnover. Consider reducing inventory levels or improving demand forecasting.")
        
        return recommendations
    
    def _get_seasonal_recommendations(self, metrics: Dict) -> List[str]:
        """Get recommendations based on seasonal metrics"""
        recommendations = []
        
        if metrics.get('Peak_Off_Peak_Ratio', 0) > 3:
            recommendations.append("High seasonal variation. Consider seasonal inventory planning and staffing.")
        
        if metrics.get('Seasonal_Concentration', 0) > 0.4:
            recommendations.append("High seasonal concentration. Consider diversifying seasonal offerings.")
        
        return recommendations
    
    def _get_business_recommendations(self, metrics: Dict) -> List[str]:
        """Get recommendations based on business metrics"""
        recommendations = []
        
        if metrics.get('Revenue_Growth_Rate', 0) < 5:
            recommendations.append("Low revenue growth. Consider marketing initiatives or product expansion.")
        
        if metrics.get('Promo_Effectiveness', 0) > 1.5:
            recommendations.append("High promotional effectiveness. Consider increasing promotional activities.")
        
        return recommendations 