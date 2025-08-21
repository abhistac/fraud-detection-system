import unittest
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from models import FraudDetectionModels
from data_preprocessing import DataPreprocessor
from feature_engineering import FeatureEngineer

class TestFraudDetectionModels(unittest.TestCase):
    
    def setUp(self):
        """Set up test data and configuration"""
        self.config = {
            'model': {
                'test_size': 0.2,
                'random_state': 42,
                'cv_folds': 3
            },
            'models': {
                'logistic_regression': {
                    'C': 1.0,
                    'class_weight': 'balanced',
                    'max_iter': 1000
                },
                'random_forest': {
                    'n_estimators': 10,  # Small for testing
                    'max_depth': 3,
                    'class_weight': 'balanced'
                },
                'xgboost': {
                    'n_estimators': 10,
                    'max_depth': 3,
                    'learning_rate': 0.1,
                    'scale_pos_weight': 10
                },
                'lightgbm': {
                    'n_estimators': 10,
                    'max_depth': 3,
                    'learning_rate': 0.1,
                    'class_weight': 'balanced'
                }
            },
            'features': {
                'time_features': False,
                'amount_features': False,
                'pca_combinations': False,
                'rolling_statistics': False
            }
        }
        
        # Create sample data
        X, y = make_classification(
            n_samples=1000,
            n_features=10,
            n_informative=8,
            n_redundant=2,
            n_classes=2,
            weights=[0.95, 0.05],  # Imbalanced
            random_state=42
        )
        
        self.df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(10)])
        self.df['Class'] = y
        
        # Initialize models
        self.models = FraudDetectionModels(self.config)
    
    def test_model_initialization(self):
        """Test model initialization"""
        self.models.initialize_models()
        
        expected_models = ['logistic', 'random_forest', 'xgboost', 'lightgbm']
        for model_name in expected_models:
            self.assertIn(model_name, self.models.models)
    
    def test_class_imbalance_handling(self):
        """Test class imbalance handling methods"""
        X = self.df.drop('Class', axis=1)
        y = self.df['Class']
        
        # Test SMOTE
        X_resampled, y_resampled = self.models.handle_class_imbalance(X, y, method='smote')
        
        # Check if minority class is oversampled
        original_minority_count = np.sum(y == 1)
        resampled_minority_count = np.sum(y_resampled == 1)
        
        self.assertGreater(resampled_minority_count, original_minority_count)
    
    def test_model_training(self):
        """Test model training"""
        X = self.df.drop('Class', axis=1)
        y = self.df['Class']
        
        self.models.initialize_models()
        
        # Train a single model
        model = self.models.train_single_model('logistic', X, y, use_smote=False)
        
        # Check if model is trained
        self.assertIn('logistic', self.models.trained_models)
        
        # Test predictions
        predictions = model.predict(X)
        probabilities = model.predict_proba(X)
        
        self.assertEqual(len(predictions), len(X))
        self.assertEqual(probabilities.shape, (len(X), 2))
    
    def test_predictions(self):
        """Test model predictions"""
        X = self.df.drop('Class', axis=1)
        y = self.df['Class']
        
        self.models.initialize_models()
        self.models.train_single_model('logistic', X, y, use_smote=False)
        
        # Test predict_proba method
        probabilities = self.models.predict_proba('logistic', X)
        self.assertEqual(len(probabilities), len(X))
        self.assertTrue(all(0 <= p <= 1 for p in probabilities))
        
        # Test predict method
        predictions = self.models.predict('logistic', X)
        self.assertEqual(len(predictions), len(X))
        self.assertTrue(all(p in [0, 1] for p in predictions))
    
    def test_cross_validation(self):
        """Test cross-validation functionality"""
        X = self.df.drop('Class', axis=1)
        y = self.df['Class']
        
        self.models.initialize_models()
        self.models.train_single_model('logistic', X, y, use_smote=False)
        
        # Test cross-validation
        cv_results = self.models.cross_validate_models(X, y)
        
        # Check if results are returned
        self.assertIn('logistic', cv_results)
        self.assertIn('mean', cv_results['logistic'])
        self.assertIn('std', cv_results['logistic'])
    
    def test_ensemble_creation(self):
        """Test ensemble model creation"""
        X = self.df.drop('Class', axis=1)
        y = self.df['Class']
        
        self.models.initialize_models()
        
        # Train individual models first
        for model_name in ['xgboost', 'lightgbm', 'random_forest']:
            self.models.train_single_model(model_name, X, y, use_smote=False)
        
        # Create ensemble
        self.models.create_ensemble_model(X, y)
        
        # Check if ensemble is created
        self.assertIn('ensemble', self.models.trained_models)
        
        # Test ensemble predictions
        predictions = self.models.predict('ensemble', X)
        probabilities = self.models.predict_proba('ensemble', X)
        
        self.assertEqual(len(predictions), len(X))
        self.assertEqual(len(probabilities), len(X))

class TestDataPreprocessor(unittest.TestCase):
    
    def setUp(self):
        """Set up test data"""
        self.config = {
            'model': {
                'test_size': 0.2,
                'random_state': 42
            }
        }
        
        # Create sample DataFrame
        np.random.seed(42)
        self.df = pd.DataFrame({
            'Time': np.random.uniform(0, 172800, 1000),
            'V1': np.random.normal(0, 1, 1000),
            'V2': np.random.normal(0, 1, 1000),
            'Amount': np.random.lognormal(2, 1, 1000),
            'Class': np.random.choice([0, 1], 1000, p=[0.99, 0.01])
        })
        
        self.preprocessor = DataPreprocessor(self.config)
    
    def test_data_exploration(self):
        """Test data exploration functionality"""
        summary = self.preprocessor.explore_data(self.df)
        
        # Check if summary contains expected keys
        expected_keys = ['shape', 'fraud_rate', 'missing_values', 'feature_types', 'fraud_distribution']
        for key in expected_keys:
            self.assertIn(key, summary)
        
        # Check if shape is correct
        self.assertEqual(summary['shape'], self.df.shape)
    
    def test_data_splitting(self):
        """Test data splitting functionality"""
        X_train, X_test, y_train, y_test = self.preprocessor.split_data(self.df)
        
        # Check if shapes are correct
        expected_train_size = int(len(self.df) * 0.8)
        expected_test_size = len(self.df) - expected_train_size
        
        self.assertEqual(len(X_train), expected_train_size)
        self.assertEqual(len(X_test), expected_test_size)
        self.assertEqual(len(y_train), expected_train_size)
        self.assertEqual(len(y_test), expected_test_size)
    
    def test_outlier_detection(self):
        """Test outlier detection"""
        outliers_info = self.preprocessor.detect_outliers(self.df)
        
        # Check if outliers are detected for numerical columns
        numerical_columns = ['Time', 'V1', 'V2', 'Amount']
        for col in numerical_columns:
            self.assertIn(col, outliers_info)
            self.assertIsInstance(outliers_info[col], int)

class TestFeatureEngineer(unittest.TestCase):
    
    def setUp(self):
        """Set up test data"""
        self.config = {
            'features': {
                'time_features': True,
                'amount_features': True,
                'pca_combinations': True,
                'rolling_statistics': False  # Set to False for testing speed
            }
        }
        
        # Create sample DataFrame that mimics credit card data
        np.random.seed(42)
        self.df = pd.DataFrame({
            'Time': np.random.uniform(0, 172800, 1000),
            'V1': np.random.normal(0, 1, 1000),
            'V2': np.random.normal(0, 1, 1000),
            'V3': np.random.normal(0, 1, 1000),
            'Amount': np.random.lognormal(2, 1, 1000),
            'Class': np.random.choice([0, 1], 1000, p=[0.99, 0.01])
        })
        
        self.feature_engineer = FeatureEngineer(self.config)
    
    def test_time_features(self):
        """Test time feature creation"""
        df_featured = self.feature_engineer.create_time_features(self.df)
        
        # Check if time features are created
        expected_features = ['Time_hours', 'Hour_of_day', 'Day', 'Night_transaction', 'Is_weekend']
        for feature in expected_features:
            self.assertIn(feature, df_featured.columns)
    
    def test_amount_features(self):
        """Test amount feature creation"""
        df_featured = self.feature_engineer.create_amount_features(self.df)
        
        # Check if amount features are created
        expected_features = ['Amount_log', 'High_amount', 'Zero_amount']
        for feature in expected_features:
            self.assertIn(feature, df_featured.columns)
    
    def test_pca_combinations(self):
        """Test PCA combination features"""
        df_featured = self.feature_engineer.create_pca_combinations(self.df)
        
        # Check if PCA combination features are created
        expected_features = ['V1_V2_interaction', 'V1_V3_interaction', 'PCA_mean', 'PCA_std']
        for feature in expected_features:
            self.assertIn(feature, df_featured.columns)
    
    def test_feature_engineering_pipeline(self):
        """Test complete feature engineering pipeline"""
        original_shape = self.df.shape
        df_featured = self.feature_engineer.engineer_features(self.df)
        
        # Check if new features are added
        self.assertGreater(df_featured.shape[1], original_shape[1])
        
        # Check if original features are preserved
        original_features = ['Time', 'V1', 'V2', 'V3', 'Amount', 'Class']
        for feature in original_features:
            self.assertIn(feature, df_featured.columns)

if __name__ == '__main__':
    # Run all tests
    unittest.main(verbosity=2)