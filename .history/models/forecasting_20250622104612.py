import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import lightgbm as lgb
from prophet import Prophet
import joblib
import warnings
warnings.filterwarnings('ignore')

class DemandForecaster:
    """Demand forecasting using multiple ML models"""
    
    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.feature_columns = []
        self.target_column = 'Sales'
        self.best_model = None
        self.best_model_name = None
        
    def prepare_data(self, df: pd.DataFrame, target_col: str = 'Sales',
                    feature_cols: List[str] = None) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Prepare data for modeling
        Args:
            df: Input dataframe
            target_col: Target column name
            feature_cols: List of feature columns
        Returns:
            Tuple of (X, y) for modeling
        """
        if feature_cols is None:
            # Exclude non-feature columns
            exclude_cols = ['Date', 'Store', 'Product', target_col]
            feature_cols = [col for col in df.columns if col not in exclude_cols]
        
        # Encode any non-numeric columns
        X = df[feature_cols].copy()
        for col in X.select_dtypes(include=['object', 'category']).columns:
            X[col], _ = pd.factorize(X[col])
        y = df[target_col]
        
        # Remove rows with NaN values
        valid_idx = X.notnull().all(axis=1) & y.notnull()
        X = X[valid_idx]
        y = y[valid_idx]
        
        self.feature_columns = list(X.columns)
        self.target_column = target_col
        
        return X, y
    
    def train_models(self, X: pd.DataFrame, y: pd.Series, 
                    models_to_train: List[str] = None) -> Dict[str, Any]:
        """
        Train multiple forecasting models
        Args:
            X: Feature matrix
            y: Target variable
            models_to_train: List of model names to train
        Returns:
            Dictionary with model performance metrics
        """
        if models_to_train is None:
            models_to_train = ['random_forest', 'xgboost', 'lightgbm', 'gradient_boosting']
        
        # Initialize models
        model_configs = {
            'random_forest': RandomForestRegressor(n_estimators=100, random_state=42),
            'xgboost': xgb.XGBRegressor(n_estimators=100, random_state=42),
            'lightgbm': lgb.LGBMRegressor(n_estimators=100, random_state=42),
            'gradient_boosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
            'linear_regression': LinearRegression(),
            'ridge': Ridge(alpha=1.0),
            'lasso': Lasso(alpha=0.1)
        }
        
        # Time series cross-validation
        tscv = TimeSeriesSplit(n_splits=5)
        
        results = {}
        best_score = float('inf')
        
        for model_name in models_to_train:
            if model_name not in model_configs:
                continue
                
            print(f"Training {model_name}...")
            model = model_configs[model_name]
            
            # Cross-validation
            cv_scores = cross_val_score(model, X, y, cv=tscv, scoring='neg_mean_absolute_error')
            cv_mae = -cv_scores.mean()
            cv_std = cv_scores.std()
            
            # Train on full dataset
            model.fit(X, y)
            y_pred = model.predict(X)
            
            # Calculate metrics
            mae = mean_absolute_error(y, y_pred)
            mse = mean_squared_error(y, y_pred)
            rmse = np.sqrt(mse)
            r2 = r2_score(y, y_pred)
            
            # Store model and results
            self.models[model_name] = model
            results[model_name] = {
                'cv_mae': cv_mae,
                'cv_std': cv_std,
                'mae': mae,
                'mse': mse,
                'rmse': rmse,
                'r2': r2
            }
            
            # Track best model
            if cv_mae < best_score:
                best_score = cv_mae
                self.best_model = model
                self.best_model_name = model_name
        
        return results
    
    def predict(self, X: pd.DataFrame, model_name: str = None) -> np.ndarray:
        """
        Make predictions using a trained model
        Args:
            X: Feature matrix
            model_name: Name of model to use (if None, uses best model)
        Returns:
            Predictions array
        """
        if model_name is None:
            if self.best_model is None:
                raise ValueError("No model trained yet. Call train_models() first.")
            model = self.best_model
        else:
            if model_name not in self.models:
                raise ValueError(f"Model '{model_name}' not found. Available models: {list(self.models.keys())}")
            model = self.models[model_name]
        
        return model.predict(X)
    
    def predict_future(self, df: pd.DataFrame, future_dates: pd.DatetimeIndex,
                      feature_engineer, model_name: str = None) -> pd.DataFrame:
        """
        Predict future sales for given dates
        Args:
            df: Historical data
            future_dates: Future dates to predict
            feature_engineer: FeatureEngineer instance
            model_name: Model to use for prediction
        Returns:
            DataFrame with predictions
        """
        # Create future data frame
        future_df = pd.DataFrame({'Date': future_dates})
        
        # Merge with existing data structure
        if 'Store' in df.columns and 'Product' in df.columns:
            # Get unique store-product combinations
            store_products = df[['Store', 'Product']].drop_duplicates()
            
            # Create cartesian product of future dates and store-products
            future_data = []
            for _, row in store_products.iterrows():
                for date in future_dates:
                    future_data.append({
                        'Date': date,
                        'Store': row['Store'],
                        'Product': row['Product']
                    })
            future_df = pd.DataFrame(future_data)
        
        # Engineer features for future data
        future_df = feature_engineer.engineer_all_features(future_df)
        
        # Prepare features for prediction
        X_future = future_df[self.feature_columns]
        
        # Make predictions
        predictions = self.predict(X_future, model_name)
        
        # Add predictions to future dataframe
        future_df[self.target_column] = predictions
        
        return future_df
    
    def get_feature_importance(self, model_name: str = None) -> pd.DataFrame:
        """
        Get feature importance from trained model
        Args:
            model_name: Name of model (if None, uses best model)
        Returns:
            DataFrame with feature importance
        """
        if model_name is None:
            model_name = self.best_model_name
        
        if model_name not in self.models:
            raise ValueError(f"Model '{model_name}' not found")
        
        model = self.models[model_name]
        
        if not hasattr(model, 'feature_importances_'):
            raise ValueError(f"Model '{model_name}' does not support feature importance")
        
        importance_df = pd.DataFrame({
            'feature': self.feature_columns,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        return importance_df
    
    def save_model(self, filepath: str, model_name: str = None):
        """
        Save trained model to file
        Args:
            filepath: Path to save model
            model_name: Name of model to save (if None, saves best model)
        """
        if model_name is None:
            model_name = self.best_model_name
        
        if model_name not in self.models:
            raise ValueError(f"Model '{model_name}' not found")
        
        model_data = {
            'model': self.models[model_name],
            'feature_columns': self.feature_columns,
            'target_column': self.target_column,
            'model_name': model_name
        }
        
        joblib.dump(model_data, filepath)
    
    def load_model(self, filepath: str):
        """
        Load trained model from file
        Args:
            filepath: Path to saved model
        """
        model_data = joblib.load(filepath)
        
        self.models[model_data['model_name']] = model_data['model']
        self.feature_columns = model_data['feature_columns']
        self.target_column = model_data['target_column']
        self.best_model = model_data['model']
        self.best_model_name = model_data['model_name']

class ProphetForecaster:
    """Time series forecasting using Facebook Prophet"""
    
    def __init__(self):
        self.models = {}
        self.forecasts = {}
        
    def prepare_prophet_data(self, df: pd.DataFrame, group_cols: List[str] = None) -> Dict[str, pd.DataFrame]:
        """
        Prepare data for Prophet models
        Args:
            df: Input dataframe
            group_cols: Columns to group by (e.g., ['Store', 'Product'])
        Returns:
            Dictionary of Prophet-formatted dataframes
        """
        if group_cols is None:
            group_cols = ['Store', 'Product']
        
        prophet_data = {}
        
        # Group by specified columns and aggregate sales
        grouped = df.groupby(group_cols + ['Date'])['Sales'].sum().reset_index()
        
        for name, group in grouped.groupby(group_cols):
            group_key = '_'.join([str(x) for x in name]) if len(group_cols) > 1 else str(name)
            
            # Prophet requires 'ds' (date) and 'y' (target) columns
            prophet_df = group[['Date', 'Sales']].copy()
            prophet_df.columns = ['ds', 'y']
            prophet_df = prophet_df.sort_values('ds')
            
            prophet_data[group_key] = prophet_df
        
        return prophet_data
    
    def train_prophet_models(self, prophet_data: Dict[str, pd.DataFrame],
                           changepoint_prior_scale: float = 0.05,
                           seasonality_prior_scale: float = 10.0) -> Dict[str, Prophet]:
        """
        Train Prophet models for each group
        Args:
            prophet_data: Dictionary of Prophet-formatted dataframes
            changepoint_prior_scale: Prophet parameter for trend flexibility
            seasonality_prior_scale: Prophet parameter for seasonality strength
        Returns:
            Dictionary of trained Prophet models
        """
        models = {}
        
        for group_key, data in prophet_data.items():
            print(f"Training Prophet model for {group_key}...")
            
            model = Prophet(
                changepoint_prior_scale=changepoint_prior_scale,
                seasonality_prior_scale=seasonality_prior_scale,
                yearly_seasonality=True,
                weekly_seasonality=True,
                daily_seasonality=False
            )
            
            # Add custom seasonalities
            model.add_seasonality(name='monthly', period=30.5, fourier_order=5)
            model.add_seasonality(name='quarterly', period=91.25, fourier_order=8)
            
            model.fit(data)
            models[group_key] = model
        
        self.models = models
        return models
    
    def predict_future(self, periods: int = 30, freq: str = 'D') -> Dict[str, pd.DataFrame]:
        """
        Make future predictions using Prophet models
        Args:
            periods: Number of periods to predict
            freq: Frequency of predictions ('D' for daily, 'W' for weekly, etc.)
        Returns:
            Dictionary of forecast dataframes
        """
        forecasts = {}
        
        for group_key, model in self.models.items():
            # Create future dataframe
            future = model.make_future_dataframe(periods=periods, freq=freq)
            
            # Make prediction
            forecast = model.predict(future)
            
            # Keep only future predictions
            future_forecast = forecast[forecast['ds'] > forecast['ds'].max() - pd.Timedelta(days=periods)]
            
            forecasts[group_key] = future_forecast
        
        self.forecasts = forecasts
        return forecasts
    
    def get_forecast_summary(self) -> pd.DataFrame:
        """
        Get summary of all forecasts
        Returns:
            DataFrame with forecast summary
        """
        summary_data = []
        
        for group_key, forecast in self.forecasts.items():
            # Get the last prediction
            last_pred = forecast.iloc[-1]
            
            summary_data.append({
                'Group': group_key,
                'Date': last_pred['ds'],
                'Predicted_Sales': last_pred['yhat'],
                'Lower_Bound': last_pred['yhat_lower'],
                'Upper_Bound': last_pred['yhat_upper']
            })
        
        return pd.DataFrame(summary_data) 