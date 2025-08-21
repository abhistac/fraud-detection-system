import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.metrics import classification_report, roc_auc_score, precision_recall_curve
from sklearn.model_selection import cross_val_score, StratifiedKFold
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline as ImbPipeline
import xgboost as xgb
import lightgbm as lgb
import joblib
import logging

class FraudDetectionModels:
    def __init__(self, config):
        self.config = config
        self.models = {}
        self.trained_models = {}
        self.feature_importance = {}
        
    def initialize_models(self):
        """Initialize all models with configurations"""
        
        # Logistic Regression
        self.models['logistic'] = LogisticRegression(
            **self.config['models']['logistic_regression'],
            random_state=self.config['model']['random_state']
        )
        
        # Random Forest
        self.models['random_forest'] = RandomForestClassifier(
            **self.config['models']['random_forest'],
            random_state=self.config['model']['random_state']
        )
        
        # XGBoost
        self.models['xgboost'] = xgb.XGBClassifier(
            **self.config['models']['xgboost'],
            random_state=self.config['model']['random_state'],
            eval_metric='logloss'
        )
        
        # LightGBM
        self.models['lightgbm'] = lgb.LGBMClassifier(
            **self.config['models']['lightgbm'],
            random_state=self.config['model']['random_state'],
            verbose=-1
        )
        
        logging.info("Models initialized")
    
    def handle_class_imbalance(self, X_train, y_train, method='smote'):
        """Handle class imbalance using various techniques"""
        
        if method == 'smote':
            smote = SMOTE(random_state=self.config['model']['random_state'])
            X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
            
        elif method == 'undersample':
            undersampler = RandomUnderSampler(random_state=self.config['model']['random_state'])
            X_resampled, y_resampled = undersampler.fit_resample(X_train, y_train)
            
        elif method == 'combined':
            # First oversample minority class, then undersample majority
            smote = SMOTE(sampling_strategy=0.5, random_state=self.config['model']['random_state'])
            under = RandomUnderSampler(sampling_strategy=0.8, random_state=self.config['model']['random_state'])
            
            pipeline = ImbPipeline([
                ('smote', smote),
                ('under', under)
            ])
            X_resampled, y_resampled = pipeline.fit_resample(X_train, y_train)
        
        else:
            X_resampled, y_resampled = X_train, y_train
        
        logging.info(f"Class imbalance handled using {method}. New shape: {X_resampled.shape}")
        logging.info(f"New class distribution: {np.bincount(y_resampled)}")
        
        return X_resampled, y_resampled
    
    def train_single_model(self, model_name, X_train, y_train, use_smote=True):
        """Train a single model"""
        
        model = self.models[model_name]
        
        # Handle class imbalance for tree-based models differently
        if use_smote and model_name in ['logistic', 'random_forest']:
            X_train_resampled, y_train_resampled = self.handle_class_imbalance(X_train, y_train, 'smote')
        else:
            X_train_resampled, y_train_resampled = X_train, y_train
        
        # Train model
        model.fit(X_train_resampled, y_train_resampled)
        
        # Store trained model
        self.trained_models[model_name] = model
        
        # Store feature importance if available
        if hasattr(model, 'feature_importances_'):
            self.feature_importance[model_name] = model.feature_importances_
        elif hasattr(model, 'coef_'):
            self.feature_importance[model_name] = np.abs(model.coef_[0])
        
        logging.info(f"{model_name} model trained successfully")
        return model
    
    def train_all_models(self, X_train, y_train):
        """Train all models"""
        
        self.initialize_models()
        
        for model_name in self.models.keys():
            self.train_single_model(model_name, X_train, y_train)
        
        # Create ensemble model
        self.create_ensemble_model(X_train, y_train)
        
        logging.info("All models trained successfully")
    
    def create_ensemble_model(self, X_train, y_train):
        """Create ensemble model from trained models"""
        
        # Use the best performing models for ensemble
        ensemble_models = [
            ('xgboost', self.trained_models['xgboost']),
            ('lightgbm', self.trained_models['lightgbm']),
            ('random_forest', self.trained_models['random_forest'])
        ]
        
        ensemble = VotingClassifier(
            estimators=ensemble_models,
            voting='soft'  # Use probability averaging
        )
        
        # Train ensemble (it will use the already trained models)
        ensemble.fit(X_train, y_train)
        
        self.trained_models['ensemble'] = ensemble
        logging.info("Ensemble model created")
    
    def cross_validate_models(self, X_train, y_train, scoring='roc_auc'):
        """Perform cross-validation for all models"""
        
        cv_results = {}
        cv = StratifiedKFold(n_splits=self.config['model']['cv_folds'], 
                           shuffle=True, 
                           random_state=self.config['model']['random_state'])
        
        for model_name, model in self.trained_models.items():
            if model_name == 'ensemble':
                continue  # Skip ensemble for CV as it's complex
                
            scores = cross_val_score(model, X_train, y_train, cv=cv, scoring=scoring)
            cv_results[model_name] = {
                'mean': scores.mean(),
                'std': scores.std(),
                'scores': scores
            }
            
            logging.info(f"{model_name} CV {scoring}: {scores.mean():.4f} (+/- {scores.std() * 2:.4f})")
        
        return cv_results
    
    def predict_proba(self, model_name, X):
        """Get prediction probabilities"""
        model = self.trained_models[model_name]
        return model.predict_proba(X)[:, 1]
    
    def predict(self, model_name, X):
        """Get predictions"""
        model = self.trained_models[model_name]
        return model.predict(X)
    
    def save_models(self, filepath_prefix):
        """Save all trained models"""
        for model_name, model in self.trained_models.items():
            filepath = f"{filepath_prefix}_{model_name}.joblib"
            joblib.dump(model, filepath)
            logging.info(f"Model {model_name} saved to {filepath}")
    
    def load_models(self, filepath_prefix):
        """Load trained models"""
        model_names = ['logistic', 'random_forest', 'xgboost', 'lightgbm', 'ensemble']
        
        for model_name in model_names:
            try:
                filepath = f"{filepath_prefix}_{model_name}.joblib"
                model = joblib.load(filepath)
                self.trained_models[model_name] = model
                logging.info(f"Model {model_name} loaded from {filepath}")
            except FileNotFoundError:
                logging.warning(f"Model file not found: {filepath}")