import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
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
        rs = self.config['model']['random_state']
        self.models['logistic'] = LogisticRegression(
            **self.config['models']['logistic_regression'], random_state=rs
        )
        self.models['random_forest'] = RandomForestClassifier(
            **self.config['models']['random_forest'], random_state=rs
        )
        self.models['xgboost'] = xgb.XGBClassifier(
            **self.config['models']['xgboost'], random_state=rs, eval_metric='logloss'
        )
        self.models['lightgbm'] = lgb.LGBMClassifier(
            **self.config['models']['lightgbm'], random_state=rs, verbose=-1
        )

    def handle_class_imbalance(self, X_train, y_train, method='smote'):
        rs = self.config['model']['random_state']
        if method == 'smote':
            X_resampled, y_resampled = SMOTE(random_state=rs).fit_resample(X_train, y_train)
        elif method == 'undersample':
            X_resampled, y_resampled = RandomUnderSampler(random_state=rs).fit_resample(X_train, y_train)
        elif method == 'combined':
            pipeline = ImbPipeline([
                ('smote', SMOTE(sampling_strategy=0.5, random_state=rs)),
                ('under', RandomUnderSampler(sampling_strategy=0.8, random_state=rs))
            ])
            X_resampled, y_resampled = pipeline.fit_resample(X_train, y_train)
        else:
            X_resampled, y_resampled = X_train, y_train

        logging.info(f"Resampled with {method}: {X_resampled.shape}, class dist: {np.bincount(y_resampled)}")
        return X_resampled, y_resampled

    def train_single_model(self, model_name, X_train, y_train, use_smote=True):
        model = self.models[model_name]

        if use_smote and model_name in ['logistic', 'random_forest']:
            X_train, y_train = self.handle_class_imbalance(X_train, y_train, 'smote')

        model.fit(X_train, y_train)
        self.trained_models[model_name] = model

        if hasattr(model, 'feature_importances_'):
            self.feature_importance[model_name] = model.feature_importances_
        elif hasattr(model, 'coef_'):
            self.feature_importance[model_name] = np.abs(model.coef_[0])

        logging.info(f"{model_name} trained")
        return model

    def train_all_models(self, X_train, y_train):
        self.initialize_models()
        for model_name in self.models:
            self.train_single_model(model_name, X_train, y_train)
        self.create_ensemble_model(X_train, y_train)

    def create_ensemble_model(self, X_train, y_train):
        estimators = [
            ('xgboost', self.trained_models['xgboost']),
            ('lightgbm', self.trained_models['lightgbm']),
            ('random_forest', self.trained_models['random_forest'])
        ]
        ensemble = VotingClassifier(estimators=estimators, voting='soft')
        ensemble.fit(X_train, y_train)
        self.trained_models['ensemble'] = ensemble
        logging.info("Ensemble model created")

    def cross_validate_models(self, X_train, y_train, scoring='roc_auc'):
        cv = StratifiedKFold(
            n_splits=self.config['model']['cv_folds'],
            shuffle=True,
            random_state=self.config['model']['random_state']
        )
        cv_results = {}
        for model_name, model in self.trained_models.items():
            if model_name == 'ensemble':
                continue
            scores = cross_val_score(model, X_train, y_train, cv=cv, scoring=scoring)
            cv_results[model_name] = {'mean': scores.mean(), 'std': scores.std(), 'scores': scores}
            logging.info(f"{model_name} CV {scoring}: {scores.mean():.4f} ± {scores.std() * 2:.4f}")
        return cv_results

    def predict_proba(self, model_name, X):
        return self.trained_models[model_name].predict_proba(X)[:, 1]

    def predict(self, model_name, X):
        return self.trained_models[model_name].predict(X)

    def save_models(self, filepath_prefix):
        for model_name, model in self.trained_models.items():
            path = f"{filepath_prefix}_{model_name}.joblib"
            joblib.dump(model, path)
            logging.info(f"Saved {model_name} to {path}")

    def load_models(self, filepath_prefix):
        for model_name in ['logistic', 'random_forest', 'xgboost', 'lightgbm', 'ensemble']:
            path = f"{filepath_prefix}_{model_name}.joblib"
            try:
                self.trained_models[model_name] = joblib.load(path)
            except FileNotFoundError:
                logging.warning(f"Model file not found: {path}")
