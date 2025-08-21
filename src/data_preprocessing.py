import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, RobustScaler
import logging

class DataPreprocessor:
    def __init__(self, config):
        self.config = config
        self.scaler = RobustScaler()
        
    def load_data(self, filepath):
        """Load and perform initial data validation"""
        try:
            df = pd.read_csv(filepath)
            logging.info(f"Data loaded successfully. Shape: {df.shape}")
            return df
        except Exception as e:
            logging.error(f"Error loading data: {e}")
            raise
    
    def explore_data(self, df):
        """Generate data exploration summary"""
        exploration_summary = {
            'shape': df.shape,
            'fraud_rate': df['Class'].mean(),
            'missing_values': df.isnull().sum().sum(),
            'feature_types': df.dtypes.value_counts().to_dict(),
            'fraud_distribution': df['Class'].value_counts().to_dict()
        }
        
        logging.info(f"Dataset exploration: {exploration_summary}")
        return exploration_summary
    
    def handle_missing_values(self, df):
        """Handle missing values if any"""
        if df.isnull().sum().sum() > 0:
            # For this dataset, we don't expect missing values
            # But good practice to check
            df_clean = df.dropna()
            logging.info(f"Removed {len(df) - len(df_clean)} rows with missing values")
            return df_clean
        return df
    
    def detect_outliers(self, df, columns=None):
        """Detect outliers using IQR method"""
        if columns is None:
            columns = df.select_dtypes(include=[np.number]).columns
            columns = [col for col in columns if col != 'Class']
        
        outliers_info = {}
        for col in columns:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
            outliers_info[col] = len(outliers)
        
        return outliers_info
    
    def scale_features(self, X_train, X_test, features_to_scale=None):
        """Scale numerical features"""
        if features_to_scale is None:
            features_to_scale = ['Time', 'Amount']
        
        X_train_scaled = X_train.copy()
        X_test_scaled = X_test.copy()
        
        if features_to_scale:
            X_train_scaled[features_to_scale] = self.scaler.fit_transform(X_train[features_to_scale])
            X_test_scaled[features_to_scale] = self.scaler.transform(X_test[features_to_scale])

        # Handle any NaN values created during scaling
        X_train_scaled = X_train_scaled.fillna(0)
        X_test_scaled = X_test_scaled.fillna(0)
        
        return X_train_scaled, X_test_scaled
    
    def split_data(self, df, target_column='Class'):
        """Split data into train and test sets"""
        X = df.drop(target_column, axis=1)
        y = df[target_column]
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, 
            test_size=self.config['model']['test_size'],
            random_state=self.config['model']['random_state'],
            stratify=y
        )
        
        logging.info(f"Data split - Train: {X_train.shape}, Test: {X_test.shape}")
        logging.info(f"Fraud rate - Train: {y_train.mean():.4f}, Test: {y_test.mean():.4f}")
        
        return X_train, X_test, y_train, y_test