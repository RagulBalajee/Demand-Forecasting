import plotly.graph_objects as go
import plotly.express as px
import plotly.subplots as sp
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import matplotlib.pyplot as plt
import seaborn as sns

class RetailVisualizer:
    """Visualization utilities for retail demand forecasting"""
    
    def __init__(self):
        self.color_palette = {
            'primary': '#1f77b4',
            'secondary': '#ff7f0e',
            'success': '#2ca02c',
            'warning': '#d62728',
            'info': '#9467bd',
            'light': '#8c564b',
            'dark': '#e377c2'
        }
    
    def plot_sales_trend(self, df: pd.DataFrame, 
                        date_col: str = 'Date',
                        sales_col: str = 'Sales',
                        group_col: str = None,
                        title: str = "Sales Trend Over Time") -> go.Figure:
        """
        Create interactive sales trend plot
        Args:
            df: Sales dataframe
            date_col: Date column name
            sales_col: Sales column name
            group_col: Column to group by (optional)
            title: Plot title
        Returns:
            Plotly figure object
        """
        if group_col:
            # Group by date and category
            agg_df = df.groupby([date_col, group_col])[sales_col].sum().reset_index()
            
            fig = px.line(agg_df, x=date_col, y=sales_col, color=group_col,
                         title=title, template='plotly_white')
        else:
            # Aggregate by date only
            agg_df = df.groupby(date_col)[sales_col].sum().reset_index()
            
            fig = px.line(agg_df, x=date_col, y=sales_col,
                         title=title, template='plotly_white')
        
        fig.update_layout(
            xaxis_title="Date",
            yaxis_title="Sales",
            hovermode='x unified',
            showlegend=True
        )
        
        return fig
    
    def plot_forecast_comparison(self, historical_df: pd.DataFrame,
                               forecast_df: pd.DataFrame,
                               date_col: str = 'Date',
                               sales_col: str = 'Sales',
                               title: str = "Historical vs Forecasted Sales") -> go.Figure:
        """
        Create comparison plot between historical and forecasted data
        Args:
            historical_df: Historical sales data
            forecast_df: Forecasted sales data
            date_col: Date column name
            sales_col: Sales column name
            title: Plot title
        Returns:
            Plotly figure object
        """
        # Prepare historical data
        hist_agg = historical_df.groupby(date_col)[sales_col].sum().reset_index()
        hist_agg['Type'] = 'Historical'
        
        # Prepare forecast data
        forecast_agg = forecast_df.groupby(date_col)[sales_col].sum().reset_index()
        forecast_agg['Type'] = 'Forecast'
        
        # Combine data
        combined_df = pd.concat([hist_agg, forecast_agg], ignore_index=True)
        
        # Create plot
        fig = px.line(combined_df, x=date_col, y=sales_col, color='Type',
                     title=title, template='plotly_white',
                     color_discrete_map={'Historical': self.color_palette['primary'],
                                       'Forecast': self.color_palette['secondary']})
        
        fig.update_layout(
            xaxis_title="Date",
            yaxis_title="Sales",
            hovermode='x unified',
            showlegend=True
        )
        
        return fig
    
    def plot_product_performance(self, product_metrics: pd.DataFrame,
                               metric: str = 'Total_Sales',
                               top_n: int = 10,
                               title: str = "Top Products by Sales") -> go.Figure:
        """
        Create bar chart for product performance
        Args:
            product_metrics: Product metrics dataframe
            metric: Metric to plot
            top_n: Number of top products to show
            title: Plot title
        Returns:
            Plotly figure object
        """
        # Get top products
        top_products = product_metrics.nlargest(top_n, metric)
        
        # Create color mapping based on product category
        color_map = {
            'Hot': self.color_palette['success'],
            'Normal': self.color_palette['primary'],
            'Slow-moving': self.color_palette['warning']
        }
        
        fig = px.bar(top_products, x='Product', y=metric,
                    color='Product_Category',
                    title=title,
                    template='plotly_white',
                    color_discrete_map=color_map)
        
        fig.update_layout(
            xaxis_title="Product",
            yaxis_title=metric.replace('_', ' '),
            showlegend=True,
            xaxis={'categoryorder': 'total descending'}
        )
        
        return fig
    
    def plot_product_categories(self, product_metrics: pd.DataFrame,
                              title: str = "Product Category Distribution") -> go.Figure:
        """
        Create pie chart for product category distribution
        Args:
            product_metrics: Product metrics dataframe
            title: Plot title
        Returns:
            Plotly figure object
        """
        category_counts = product_metrics['Product_Category'].value_counts()
        
        colors = [self.color_palette['success'], self.color_palette['primary'], self.color_palette['warning']]
        
        fig = px.pie(values=category_counts.values, names=category_counts.index,
                    title=title, template='plotly_white',
                    color_discrete_sequence=colors)
        
        fig.update_traces(textposition='inside', textinfo='percent+label')
        
        return fig
    
    def plot_seasonal_patterns(self, df: pd.DataFrame,
                             date_col: str = 'Date',
                             sales_col: str = 'Sales',
                             group_col: str = None,
                             title: str = "Seasonal Sales Patterns") -> go.Figure:
        """
        Create seasonal pattern visualization
        Args:
            df: Sales dataframe
            date_col: Date column name
            sales_col: Sales column name
            group_col: Column to group by (optional)
            title: Plot title
        Returns:
            Plotly figure object
        """
        # Add month and day of week
        df_analysis = df.copy()
        df_analysis['Month'] = df_analysis[date_col].dt.month
        df_analysis['DayOfWeek'] = df_analysis[date_col].dt.dayofweek
        df_analysis['DayName'] = df_analysis[date_col].dt.day_name()
        
        # Create subplots
        fig = sp.make_subplots(
            rows=2, cols=1,
            subplot_titles=('Monthly Sales Pattern', 'Weekly Sales Pattern'),
            vertical_spacing=0.1
        )
        
        # Monthly pattern
        if group_col:
            monthly_data = df_analysis.groupby(['Month', group_col])[sales_col].mean().reset_index()
            for category in monthly_data[group_col].unique():
                cat_data = monthly_data[monthly_data[group_col] == category]
                fig.add_trace(
                    go.Scatter(x=cat_data['Month'], y=cat_data[sales_col],
                              mode='lines+markers', name=f'{category}',
                              showlegend=True),
                    row=1, col=1
                )
        else:
            monthly_data = df_analysis.groupby('Month')[sales_col].mean().reset_index()
            fig.add_trace(
                go.Scatter(x=monthly_data['Month'], y=monthly_data[sales_col],
                          mode='lines+markers', name='Average Sales',
                          line=dict(color=self.color_palette['primary'])),
                row=1, col=1
            )
        
        # Weekly pattern
        if group_col:
            weekly_data = df_analysis.groupby(['DayOfWeek', group_col])[sales_col].mean().reset_index()
            for category in weekly_data[group_col].unique():
                cat_data = weekly_data[weekly_data[group_col] == category]
                fig.add_trace(
                    go.Scatter(x=cat_data['DayOfWeek'], y=cat_data[sales_col],
                              mode='lines+markers', name=f'{category}',
                              showlegend=False),
                    row=2, col=1
                )
        else:
            weekly_data = df_analysis.groupby('DayOfWeek')[sales_col].mean().reset_index()
            fig.add_trace(
                go.Scatter(x=weekly_data['DayOfWeek'], y=weekly_data[sales_col],
                          mode='lines+markers', name='Average Sales',
                          line=dict(color=self.color_palette['secondary'])),
                row=2, col=1
            )
        
        # Update layout
        fig.update_layout(
            title=title,
            template='plotly_white',
            height=600
        )
        
        # Update axes
        fig.update_xaxes(title_text="Month", row=1, col=1)
        fig.update_yaxes(title_text="Average Sales", row=1, col=1)
        fig.update_xaxes(title_text="Day of Week (0=Monday)", row=2, col=1)
        fig.update_yaxes(title_text="Average Sales", row=2, col=1)
        
        return fig
    
    def plot_stock_recommendations(self, stock_df: pd.DataFrame,
                                 title: str = "Stock Level Recommendations") -> go.Figure:
        """
        Create stock recommendations visualization
        Args:
            stock_df: Stock recommendations dataframe
            title: Plot title
        Returns:
            Plotly figure object
        """
        # Create scatter plot
        fig = px.scatter(stock_df, x='Daily_Demand', y='Recommended_Stock',
                        color='Product_Category',
                        size='Total_Sales',
                        hover_data=['Product', 'Stock_Turnover', 'Safety_Stock'],
                        title=title,
                        template='plotly_white',
                        color_discrete_map={
                            'Hot': self.color_palette['success'],
                            'Normal': self.color_palette['primary'],
                            'Slow-moving': self.color_palette['warning']
                        })
        
        # Add diagonal line for 1:1 ratio
        max_val = max(stock_df['Daily_Demand'].max(), stock_df['Recommended_Stock'].max())
        fig.add_trace(
            go.Scatter(x=[0, max_val], y=[0, max_val],
                      mode='lines', name='1:1 Ratio',
                      line=dict(color='gray', dash='dash'))
        )
        
        fig.update_layout(
            xaxis_title="Daily Demand",
            yaxis_title="Recommended Stock Level",
            showlegend=True
        )
        
        return fig
    
    def plot_feature_importance(self, importance_df: pd.DataFrame,
                              top_n: int = 15,
                              title: str = "Feature Importance") -> go.Figure:
        """
        Create feature importance visualization
        Args:
            importance_df: Feature importance dataframe
            top_n: Number of top features to show
            title: Plot title
        Returns:
            Plotly figure object
        """
        # Get top features
        top_features = importance_df.head(top_n)
        
        fig = px.bar(top_features, x='importance', y='feature',
                    orientation='h',
                    title=title,
                    template='plotly_white',
                    color='importance',
                    color_continuous_scale='Blues')
        
        fig.update_layout(
            xaxis_title="Importance Score",
            yaxis_title="Feature",
            showlegend=False,
            yaxis={'categoryorder': 'total ascending'}
        )
        
        return fig
    
    def create_dashboard_summary(self, metrics: Dict) -> go.Figure:
        """
        Create summary dashboard with key metrics
        Args:
            metrics: Dictionary of key metrics
        Returns:
            Plotly figure object
        """
        # Create subplots for different metrics
        fig = sp.make_subplots(
            rows=2, cols=2,
            subplot_titles=('Total Sales', 'Product Categories', 'Growth Rate', 'Stock Status'),
            specs=[[{"type": "indicator"}, {"type": "pie"}],
                   [{"type": "bar"}, {"type": "bar"}]]
        )
        
        # Total Sales indicator
        if 'total_sales' in metrics:
            fig.add_trace(
                go.Indicator(
                    mode="number+delta",
                    value=metrics['total_sales'],
                    delta={'reference': metrics.get('prev_sales', 0)},
                    title={'text': "Total Sales"},
                    domain={'row': 0, 'column': 0}
                ),
                row=1, col=1
            )
        
        # Product Categories pie chart
        if 'category_distribution' in metrics:
            categories = list(metrics['category_distribution'].keys())
            values = list(metrics['category_distribution'].values())
            
            fig.add_trace(
                go.Pie(labels=categories, values=values, name="Categories"),
                row=1, col=2
            )
        
        # Growth Rate bar chart
        if 'growth_rates' in metrics:
            products = list(metrics['growth_rates'].keys())[:5]  # Top 5
            rates = list(metrics['growth_rates'].values())[:5]
            
            fig.add_trace(
                go.Bar(x=products, y=rates, name="Growth Rate"),
                row=2, col=1
            )
        
        # Stock Status bar chart
        if 'stock_status' in metrics:
            statuses = list(metrics['stock_status'].keys())
            counts = list(metrics['stock_status'].values())
            
            fig.add_trace(
                go.Bar(x=statuses, y=counts, name="Stock Status"),
                row=2, col=2
            )
        
        fig.update_layout(
            title="Retail Dashboard Summary",
            template='plotly_white',
            height=600,
            showlegend=False
        )
        
        return fig
    
    def plot_forecast_confidence(self, forecast_df: pd.DataFrame,
                               date_col: str = 'Date',
                               sales_col: str = 'Sales',
                               lower_col: str = 'yhat_lower',
                               upper_col: str = 'yhat_upper',
                               title: str = "Forecast with Confidence Intervals") -> go.Figure:
        """
        Create forecast plot with confidence intervals
        Args:
            forecast_df: Forecast dataframe with confidence intervals
            date_col: Date column name
            sales_col: Sales column name
            lower_col: Lower bound column name
            upper_col: Upper bound column name
            title: Plot title
        Returns:
            Plotly figure object
        """
        fig = go.Figure()
        
        # Add confidence interval
        fig.add_trace(go.Scatter(
            x=forecast_df[date_col],
            y=forecast_df[upper_col],
            mode='lines',
            line=dict(width=0),
            showlegend=False,
            name='Upper Bound'
        ))
        
        fig.add_trace(go.Scatter(
            x=forecast_df[date_col],
            y=forecast_df[lower_col],
            mode='lines',
            line=dict(width=0),
            fillcolor='rgba(68, 68, 68, 0.3)',
            fill='tonexty',
            showlegend=False,
            name='Lower Bound'
        ))
        
        # Add forecast line
        fig.add_trace(go.Scatter(
            x=forecast_df[date_col],
            y=forecast_df[sales_col],
            mode='lines',
            line=dict(color=self.color_palette['primary']),
            name='Forecast'
        ))
        
        fig.update_layout(
            title=title,
            xaxis_title="Date",
            yaxis_title="Sales",
            template='plotly_white',
            showlegend=True
        )
        
        return fig 