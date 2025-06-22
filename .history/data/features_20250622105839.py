import pandas as pd
import numpy as np
from typing import List, Dict, Any
from datetime import datetime, timedelta

class FeatureEngineer:
    """Feature engineering for retail sales forecasting"""
    
    def __init__(self):
        self.feature_columns = []
        
    def create_time_features(self, df: pd.DataFrame, date_col: str = 'Date') -> pd.DataFrame:
        """
        Create time-based features
        Args:
            df: Input dataframe
            date_col: Name of date column
        Returns:
            Dataframe with time features
        """
        df = df.copy()
        
        # Basic time features
        df['Year'] = df[date_col].dt.year
        df['Month'] = df[date_col].dt.month
        df['Day'] = df[date_col].dt.day
        df['DayOfWeek'] = df[date_col].dt.dayofweek
        df['WeekOfYear'] = df[date_col].dt.isocalendar().week
        df['Quarter'] = df[date_col].dt.quarter
        df['DayOfYear'] = df[date_col].dt.dayofyear
        
        # Cyclical encoding for periodic features
        df['Month_Sin'] = np.sin(2 * np.pi * df['Month'] / 12)
        df['Month_Cos'] = np.cos(2 * np.pi * df['Month'] / 12)
        df['DayOfWeek_Sin'] = np.sin(2 * np.pi * df['DayOfWeek'] / 7)
        df['DayOfWeek_Cos'] = np.cos(2 * np.pi * df['DayOfWeek'] / 7)
        df['DayOfYear_Sin'] = np.sin(2 * np.pi * df['DayOfYear'] / 365)
        df['DayOfYear_Cos'] = np.cos(2 * np.pi * df['DayOfYear'] / 365)
        
        # Weekend indicator
        df['IsWeekend'] = (df['DayOfWeek'] >= 5).astype(int)
        
        # Month end/beginning indicators
        df['IsMonthEnd'] = df[date_col].dt.is_month_end.astype(int)
        df['IsMonthStart'] = df[date_col].dt.is_month_start.astype(int)
        
        # Quarter end/beginning indicators
        df['IsQuarterEnd'] = df[date_col].dt.is_quarter_end.astype(int)
        df['IsQuarterStart'] = df[date_col].dt.is_quarter_start.astype(int)
        
        return df
    
    def _get_valid_group_cols(self, df: pd.DataFrame, group_cols: list) -> list:
        """Return only the group columns that exist in the DataFrame."""
        return [col for col in group_cols if col in df.columns]

    def create_lag_features(self, df: pd.DataFrame, target_col: str = 'Sales', 
                          group_cols: List[str] = None, lags: List[int] = None) -> pd.DataFrame:
        """
        Create lag features for time series
        Args:
            df: Input dataframe
            target_col: Target column for lagging
            group_cols: Columns to group by (e.g., ['Store', 'Product'])
            lags: List of lag periods
        Returns:
            Dataframe with lag features
        """
        df = df.copy()
        
        if lags is None:
            lags = [1, 7, 14, 30]
        
        if group_cols is None:
            group_cols = ['Store', 'Product']
        
        group_cols = self._get_valid_group_cols(df, group_cols)
        
        # Sort by group columns and date
        sort_cols = group_cols + ['Date']
        df = df.sort_values(sort_cols)
        
        # Create lag features
        for lag in lags:
            lag_col = f'{target_col}_Lag{lag}'
            if group_cols:
                df[lag_col] = df.groupby(group_cols)[target_col].shift(lag)
            else:
                df[lag_col] = df[target_col].shift(lag)
        
        return df
    
    def create_rolling_features(self, df: pd.DataFrame, target_col: str = 'Sales',
                              group_cols: List[str] = None, windows: List[int] = None) -> pd.DataFrame:
        """
        Create rolling window features
        Args:
            df: Input dataframe
            target_col: Target column for rolling calculations
            group_cols: Columns to group by
            windows: List of window sizes
        Returns:
            Dataframe with rolling features
        """
        df = df.copy()
        
        if windows is None:
            windows = [7, 14, 30]
        
        if group_cols is None:
            group_cols = ['Store', 'Product']
        
        group_cols = self._get_valid_group_cols(df, group_cols)
        
        # Sort by group columns and date
        sort_cols = group_cols + ['Date']
        df = df.sort_values(sort_cols)
        
        # Create rolling features
        for window in windows:
            # Rolling mean
            mean_col = f'{target_col}_MA{window}'
            std_col = f'{target_col}_STD{window}'
            min_col = f'{target_col}_Min{window}'
            max_col = f'{target_col}_Max{window}'
            if group_cols:
                df[mean_col] = df.groupby(group_cols)[target_col].rolling(window).mean().values
                df[std_col] = df.groupby(group_cols)[target_col].rolling(window).std().values
                df[min_col] = df.groupby(group_cols)[target_col].rolling(window).min().values
                df[max_col] = df.groupby(group_cols)[target_col].rolling(window).max().values
            else:
                df[mean_col] = df[target_col].rolling(window).mean().values
                df[std_col] = df[target_col].rolling(window).std().values
                df[min_col] = df[target_col].rolling(window).min().values
                df[max_col] = df[target_col].rolling(window).max().values
        
        return df
    
    def create_interaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create interaction features between variables
        Args:
            df: Input dataframe
        Returns:
            Dataframe with interaction features
        """
        df = df.copy()
        
        # Promo interactions
        if 'Promo' in df.columns and 'Sales_Lag1' in df.columns:
            df['Promo_Sales_Lag1'] = df['Promo'] * df['Sales_Lag1']
        
        if 'Promo' in df.columns and 'IsWeekend' in df.columns:
            df['Promo_Weekend'] = df['Promo'] * df['IsWeekend']
        
        # Holiday interactions
        if 'SchoolHoliday' in df.columns and 'Sales_Lag7' in df.columns:
            df['SchoolHoliday_Sales_Lag7'] = df['SchoolHoliday'] * df['Sales_Lag7']
        
        if 'StateHoliday' in df.columns and 'Sales_Lag7' in df.columns:
            df['StateHoliday_Sales_Lag7'] = df['StateHoliday'] * df['Sales_Lag7']
        
        # Store type interactions
        if 'StoreType_Encoded' in df.columns and 'Sales_MA7' in df.columns:
            df['StoreType_Sales_MA7'] = df['StoreType_Encoded'] * df['Sales_MA7']
        
        return df
    
    def create_seasonal_features(self, df: pd.DataFrame, date_col: str = 'Date') -> pd.DataFrame:
        """
        Create seasonal features
        Args:
            df: Input dataframe
            date_col: Name of date column
        Returns:
            Dataframe with seasonal features
        """
        df = df.copy()
        
        # Holiday features
        df['IsChristmas'] = ((df[date_col].dt.month == 12) & (df[date_col].dt.day == 25)).astype(int)
        df['IsNewYear'] = ((df[date_col].dt.month == 1) & (df[date_col].dt.day == 1)).astype(int)
        df['IsValentines'] = ((df[date_col].dt.month == 2) & (df[date_col].dt.day == 14)).astype(int)
        df['IsEaster'] = ((df[date_col].dt.month == 4) & (df[date_col].dt.day.isin([9, 10, 11, 12, 13, 14, 15, 16]))).astype(int)
        
        # School holiday periods
        df['IsSummerHoliday'] = ((df[date_col].dt.month.isin([7, 8]))).astype(int)
        df['IsWinterHoliday'] = ((df[date_col].dt.month == 12) & (df[date_col].dt.day >= 20)).astype(int)
        
        # Month categories
        df['IsHolidaySeason'] = df[date_col].dt.month.isin([11, 12]).astype(int)
        df['IsBackToSchool'] = df[date_col].dt.month.isin([8, 9]).astype(int)
        
        return df
    
    def create_competition_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create competition-related features
        Args:
            df: Input dataframe
        Returns:
            Dataframe with competition features
        """
        df = df.copy()
        
        if 'CompetitionDistance' in df.columns:
            # Log transform competition distance
            df['CompetitionDistance_Log'] = np.log1p(df['CompetitionDistance'])
            
            # Competition distance categories
            df['CompetitionDistance_Cat'] = pd.cut(
                df['CompetitionDistance'], 
                bins=[0, 1000, 2000, 3000, 5000, float('inf')], 
                labels=[1, 2, 3, 4, 5]
            ).astype(float)
        
        if 'CompetitionOpenSinceMonth' in df.columns and 'CompetitionOpenSinceYear' in df.columns:
            # Calculate competition duration
            df['CompetitionDuration'] = (
                (df['Date'].dt.year - df['CompetitionOpenSinceYear']) * 12 + 
                (df['Date'].dt.month - df['CompetitionOpenSinceMonth'])
            )
            df['CompetitionDuration'] = df['CompetitionDuration'].clip(lower=0)
        
        return df
    
    def engineer_all_features(self, df: pd.DataFrame, target_col: str = 'Sales',
                            group_cols: List[str] = None) -> pd.DataFrame:
        """
        Apply all feature engineering steps
        Args:
            df: Input dataframe
            target_col: Target column name
            group_cols: Columns to group by for lag/rolling features
        Returns:
            Dataframe with all engineered features
        """
        if group_cols is None:
            group_cols = ['Store', 'Product']
        
        group_cols = self._get_valid_group_cols(df, group_cols)
        
        # Apply all feature engineering steps
        df = self.create_time_features(df)
        df = self.create_lag_features(df, target_col, group_cols)
        df = self.create_rolling_features(df, target_col, group_cols)
        df = self.create_seasonal_features(df)
        df = self.create_competition_features(df)
        df = self.create_interaction_features(df)
        
        # Store feature column names
        self.feature_columns = [col for col in df.columns if col not in ['Date', 'Store', 'Product', target_col]]
        
        return df
    
    def get_feature_importance_ranking(self, model, feature_names: List[str] = None) -> Dict[str, float]:
        """
        Get feature importance ranking from a trained model
        Args:
            model: Trained model with feature_importances_ attribute
            feature_names: List of feature names
        Returns:
            Dictionary of feature names and their importance scores
        """
        if not hasattr(model, 'feature_importances_'):
            raise ValueError("Model does not have feature_importances_ attribute")
        
        if feature_names is None:
            feature_names = self.feature_columns
        
        importance_dict = dict(zip(feature_names, model.feature_importances_))
        return dict(sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)) 