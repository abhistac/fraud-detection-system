import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
import logging

class FeatureEngineer:
    def __init__(self, config):
        self.config = config
        
    def create_time_features(self, df):
        """Create time-based features"""
        if not self.config['features']['time_features']:
            return df
        
        df_featured = df.copy()
        
        # Convert time to hours
        df_featured['Time_hours'] = df_featured['Time'] / 3600
        
        # Hour of day (assuming Time is seconds from start)
        df_featured['Hour_of_day'] = (df_featured['Time'] / 3600) % 24
        
        # Day of transaction (0 or 1 for this 2-day dataset)
        df_featured['Day'] = (df_featured['Time'] // (24 * 3600)).astype(int)
        
        # Time-based risk periods
        df_featured['Night_transaction'] = ((df_featured['Hour_of_day'] >= 22) | 
                                          (df_featured['Hour_of_day'] <= 6)).astype(int)
        
        # Weekend/weekday (assuming starts on Monday)
        df_featured['Is_weekend'] = (df_featured['Day'] % 7 >= 5).astype(int)
        
        logging.info("Time features created")
        return df_featured
    
    def create_amount_features(self, df):
        """Create amount-based features"""
        if not self.config['features']['amount_features']:
            return df
        
        df_featured = df.copy()
        
        # Log transformation of amount (handle zeros)
        df_featured['Amount_log'] = np.log1p(df_featured['Amount'])
        
        # Amount percentiles
        amount_percentiles = np.percentile(df_featured['Amount'], [25, 50, 75, 90, 95, 99])
        df_featured['Amount_percentile'] = pd.cut(df_featured['Amount'], 
                                                bins=[-np.inf] + list(amount_percentiles) + [np.inf],
                                                labels=range(7))
        
        # Amount categories
        df_featured['Amount_category'] = pd.cut(df_featured['Amount'],
                                              bins=[0, 10, 50, 100, 500, 1000, np.inf],
                                              labels=['micro', 'small', 'medium', 'large', 'very_large', 'extreme'])
        
        # High amount flag
        df_featured['High_amount'] = (df_featured['Amount'] > df_featured['Amount'].quantile(0.95)).astype(int)
        
        # Zero amount flag
        df_featured['Zero_amount'] = (df_featured['Amount'] == 0).astype(int)
        
        logging.info("Amount features created")
        return df_featured
    
    def create_pca_combinations(self, df):
        """Create new features from PCA component combinations"""
        if not self.config['features']['pca_combinations']:
            return df
        
        df_featured = df.copy()
        
        # Get PCA columns
        pca_cols = [col for col in df.columns if col.startswith('V')]
        
        # Create combination features
        df_featured['V1_V2_interaction'] = df_featured['V1'] * df_featured['V2']
        df_featured['V1_V3_interaction'] = df_featured['V1'] * df_featured['V3']
        df_featured['V2_V3_interaction'] = df_featured['V2'] * df_featured['V3']
        
        # Sum of first few components
        df_featured['V1_V5_sum'] = df_featured[['V1', 'V2', 'V3', 'V4', 'V5']].sum(axis=1)
        df_featured['V6_V10_sum'] = df_featured[['V6', 'V7', 'V8', 'V9', 'V10']].sum(axis=1)
        
        # Statistical combinations
        df_featured['PCA_mean'] = df_featured[pca_cols].mean(axis=1)
        df_featured['PCA_std'] = df_featured[pca_cols].std(axis=1)
        df_featured['PCA_skew'] = df_featured[pca_cols].skew(axis=1)
        
        # Negative value counts (common in PCA)
        df_featured['Negative_V_count'] = (df_featured[pca_cols] < 0).sum(axis=1)
        
        logging.info("PCA combination features created")
        return df_featured
    
    def create_rolling_features(self, df):
        """Create rolling window statistics"""
        if not self.config['features']['rolling_statistics']:
            return df
        
        df_featured = df.copy()
        
        # Sort by time for rolling calculations
        df_featured = df_featured.sort_values('Time').reset_index(drop=True)
        
        # Rolling statistics for amount
        windows = [10, 50, 100]
        for window in windows:
            df_featured[f'Amount_rolling_mean_{window}'] = (
                df_featured['Amount'].rolling(window=window, min_periods=1).mean()
            )
            df_featured[f'Amount_rolling_std_{window}'] = (
                df_featured['Amount'].rolling(window=window, min_periods=1).std()
            )
            
        # Transaction frequency features
        df_featured['Trans_frequency_1h'] = (
            df_featured.groupby(df_featured['Time'] // 3600)['Time'].transform('count')
        )
        
        logging.info("Rolling features created")
        return df_featured
    
    def engineer_features(self, df):
        """Main feature engineering pipeline"""
        logging.info("Starting feature engineering...")
        
        df_engineered = df.copy()
        
        # Apply all feature engineering steps
        df_engineered = self.create_time_features(df_engineered)
        df_engineered = self.create_amount_features(df_engineered)
        df_engineered = self.create_pca_combinations(df_engineered)
        df_engineered = self.create_rolling_features(df_engineered)
        
        # Handle categorical features
        categorical_features = ['Amount_percentile', 'Amount_category']
        for feature in categorical_features:
            if feature in df_engineered.columns:
                df_engineered = pd.get_dummies(df_engineered, columns=[feature], prefix=feature)
        
        # Handle any NaN values created during feature engineering
        df_engineered = df_engineered.fillna(0)
        
        logging.info(f"Feature engineering complete. Final shape: {df_engineered.shape}")
        return df_engineered