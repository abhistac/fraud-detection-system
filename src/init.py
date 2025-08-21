"""
Credit Card Fraud Detection System

A comprehensive machine learning pipeline for detecting fraudulent credit card transactions.
This package includes data preprocessing, feature engineering, model training, and evaluation.
"""

__version__ = "1.0.0"
__author__ = "Your Name"
__email__ = "your.email@example.com"

# Import main classes for easy access
from .data_preprocessing import DataPreprocessor
from .feature_engineering import FeatureEngineer
from .models import FraudDetectionModels
from .utils import ModelEvaluator, BusinessMetricsCalculator

__all__ = [
    'DataPreprocessor',
    'FeatureEngineer', 
    'FraudDetectionModels',
    'ModelEvaluator',
    'BusinessMetricsCalculator'
]