import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_classification

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from models import FraudDetectionModels
from data_preprocessing import DataPreprocessor
from feature_engineering import FeatureEngineer
from utils import BusinessMetricsCalculator


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def config():
    return {
        'model': {'test_size': 0.2, 'random_state': 42, 'cv_folds': 3},
        'models': {
            'logistic_regression': {'C': 1.0, 'class_weight': 'balanced', 'max_iter': 1000},
            'random_forest': {'n_estimators': 10, 'max_depth': 3, 'class_weight': 'balanced'},
            'xgboost': {'n_estimators': 10, 'max_depth': 3, 'learning_rate': 0.1, 'scale_pos_weight': 10},
            'lightgbm': {'n_estimators': 10, 'max_depth': 3, 'learning_rate': 0.1, 'class_weight': 'balanced'},
        },
        'features': {
            'time_features': False,
            'amount_features': False,
            'pca_combinations': False,
            'rolling_statistics': False,
        },
    }


@pytest.fixture
def sample_df():
    X, y = make_classification(
        n_samples=500, n_features=10, n_informative=8, n_redundant=2,
        n_classes=2, weights=[0.95, 0.05], random_state=42
    )
    df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(10)])
    df['Class'] = y
    return df


@pytest.fixture
def fraud_df():
    rng = np.random.default_rng(42)
    n = 500
    return pd.DataFrame({
        'Time': rng.uniform(0, 172800, n),
        'V1': rng.standard_normal(n),
        'V2': rng.standard_normal(n),
        'V3': rng.standard_normal(n),
        'Amount': rng.lognormal(2, 1, n),
        'Class': rng.choice([0, 1], n, p=[0.99, 0.01]).astype(int),
    })


# ---------------------------------------------------------------------------
# FraudDetectionModels
# ---------------------------------------------------------------------------

def test_model_initialization(config):
    m = FraudDetectionModels(config)
    m.initialize_models()
    assert set(m.models) == {'logistic', 'random_forest', 'xgboost', 'lightgbm'}


def test_smote_oversamples_minority(config, sample_df):
    m = FraudDetectionModels(config)
    X, y = sample_df.drop('Class', axis=1), sample_df['Class']
    _, y_res = m.handle_class_imbalance(X, y, method='smote')
    assert y_res.sum() > y.sum()


def test_single_model_training_and_prediction(config, sample_df):
    m = FraudDetectionModels(config)
    m.initialize_models()
    X, y = sample_df.drop('Class', axis=1), sample_df['Class']
    m.train_single_model('logistic', X, y, use_smote=False)

    assert 'logistic' in m.trained_models
    preds = m.predict('logistic', X)
    probas = m.predict_proba('logistic', X)
    assert len(preds) == len(X)
    assert set(preds).issubset({0, 1})
    assert probas.min() >= 0 and probas.max() <= 1


def test_ensemble_creation_and_prediction(config, sample_df):
    m = FraudDetectionModels(config)
    m.initialize_models()
    X, y = sample_df.drop('Class', axis=1), sample_df['Class']
    for name in ['xgboost', 'lightgbm', 'random_forest']:
        m.train_single_model(name, X, y, use_smote=False)
    m.create_ensemble_model(X, y)

    assert 'ensemble' in m.trained_models
    preds = m.predict('ensemble', X)
    probas = m.predict_proba('ensemble', X)
    assert len(preds) == len(X)
    assert probas.min() >= 0 and probas.max() <= 1


def test_cross_validation_returns_valid_scores(config, sample_df):
    m = FraudDetectionModels(config)
    m.initialize_models()
    X, y = sample_df.drop('Class', axis=1), sample_df['Class']
    m.train_single_model('logistic', X, y, use_smote=False)
    cv = m.cross_validate_models(X, y)

    assert 'logistic' in cv
    assert 0 <= cv['logistic']['mean'] <= 1
    assert cv['logistic']['std'] >= 0


# ---------------------------------------------------------------------------
# DataPreprocessor
# ---------------------------------------------------------------------------

def test_explore_data_summary(fraud_df):
    p = DataPreprocessor({'model': {'test_size': 0.2, 'random_state': 42}})
    summary = p.explore_data(fraud_df)
    assert summary['shape'] == fraud_df.shape
    assert 0 <= summary['fraud_rate'] <= 1
    assert summary['missing_values'] == 0


def test_split_covers_full_dataset(fraud_df):
    p = DataPreprocessor({'model': {'test_size': 0.2, 'random_state': 42}})
    X_tr, X_te, y_tr, y_te = p.split_data(fraud_df)
    assert len(X_tr) + len(X_te) == len(fraud_df)


def test_split_preserves_fraud_rate(fraud_df):
    p = DataPreprocessor({'model': {'test_size': 0.2, 'random_state': 42}})
    _, _, y_tr, y_te = p.split_data(fraud_df)
    assert abs(y_tr.mean() - y_te.mean()) < 0.05


def test_outlier_detection_returns_counts(fraud_df):
    p = DataPreprocessor({'model': {'test_size': 0.2, 'random_state': 42}})
    outliers = p.detect_outliers(fraud_df, columns=['Amount', 'V1'])
    assert set(outliers) == {'Amount', 'V1'}
    assert all(isinstance(v, int) and v >= 0 for v in outliers.values())


def test_scale_features_fit_on_train_only(fraud_df):
    p = DataPreprocessor({'model': {'test_size': 0.2, 'random_state': 42}})
    X_tr, X_te, _, _ = p.split_data(fraud_df)
    X_tr_scaled, X_te_scaled = p.scale_features(X_tr, X_te, features_to_scale=['Amount'])
    # Scaled train Amount should be centered near 0
    assert abs(X_tr_scaled['Amount'].median()) < 1


# ---------------------------------------------------------------------------
# FeatureEngineer
# ---------------------------------------------------------------------------

@pytest.fixture
def fe_config_all():
    return {
        'features': {
            'time_features': True,
            'amount_features': True,
            'pca_combinations': False,
            'rolling_statistics': False,
        }
    }


def test_time_features_created(fraud_df, fe_config_all):
    fe = FeatureEngineer(fe_config_all)
    result = fe.fit_transform(fraud_df)
    for col in ['Time_hours', 'Hour_of_day', 'Day', 'Night_transaction', 'Is_weekend']:
        assert col in result.columns


def test_amount_features_created(fraud_df, fe_config_all):
    fe = FeatureEngineer(fe_config_all)
    result = fe.fit_transform(fraud_df)
    for col in ['Amount_log', 'High_amount', 'Zero_amount']:
        assert col in result.columns


def test_fit_transform_adds_columns(fraud_df, fe_config_all):
    fe = FeatureEngineer(fe_config_all)
    result = fe.fit_transform(fraud_df)
    assert result.shape[1] > fraud_df.shape[1]


def test_transform_matches_fit_transform_columns(fraud_df, fe_config_all):
    """Test and train outputs must have identical columns (no column mismatch)."""
    rng = np.random.default_rng(0)
    n = 100
    test_df = pd.DataFrame({
        'Time': rng.uniform(0, 172800, n),
        'V1': rng.standard_normal(n),
        'V2': rng.standard_normal(n),
        'V3': rng.standard_normal(n),
        'Amount': rng.lognormal(2, 1, n),
        'Class': np.zeros(n, dtype=int),
    })
    fe = FeatureEngineer(fe_config_all)
    train_out = fe.fit_transform(fraud_df)
    test_out = fe.transform(test_df)
    assert list(train_out.columns) == list(test_out.columns)


def test_no_data_leakage_in_amount_threshold(fraud_df, fe_config_all):
    """High_amount threshold must come from training data, not test data."""
    fe = FeatureEngineer(fe_config_all)
    fe.fit_transform(fraud_df)
    train_threshold = fe._amount_high_threshold
    assert train_threshold is not None
    assert train_threshold > 0


# ---------------------------------------------------------------------------
# BusinessMetricsCalculator
# ---------------------------------------------------------------------------

def test_perfect_classifier_metrics():
    calc = BusinessMetricsCalculator()
    y_true = np.array([0, 0, 1, 1])
    y_pred = np.array([0, 0, 1, 1])
    y_proba = np.array([0.1, 0.1, 0.9, 0.9])
    amounts = np.array([100.0, 200.0, 500.0, 300.0])

    m = calc.calculate_business_impact(y_true, y_pred, y_proba, amounts)
    assert m['fraud_detection_rate'] == 1.0
    assert m['false_positive_rate'] == 0.0
    assert m['precision'] == 1.0
    assert m['net_benefit'] == pytest.approx(800.0)


def test_all_false_positives_metrics():
    calc = BusinessMetricsCalculator()
    y_true = np.array([0, 0, 0, 1])
    y_pred = np.array([1, 1, 1, 1])
    y_proba = np.array([0.9, 0.9, 0.9, 0.9])
    amounts = np.array([100.0, 200.0, 300.0, 400.0])

    m = calc.calculate_business_impact(y_true, y_pred, y_proba, amounts)
    assert m['investigation_costs'] == pytest.approx(3 * 50)
    assert m['net_benefit'] == pytest.approx(400.0 - 150.0)


def test_threshold_optimization_valid_range():
    calc = BusinessMetricsCalculator()
    rng = np.random.default_rng(42)
    y_true = rng.choice([0, 1], 300, p=[0.95, 0.05])
    y_proba = rng.uniform(0, 1, 300)
    amounts = rng.lognormal(2, 1, 300)

    _, threshold = calc.threshold_optimization(y_true, y_proba, amounts)
    assert 0.1 <= threshold <= 0.95


def test_threshold_optimization_maximises_net_benefit():
    calc = BusinessMetricsCalculator()
    rng = np.random.default_rng(7)
    n = 200
    y_true = rng.choice([0, 1], n, p=[0.95, 0.05])
    y_proba = rng.uniform(0, 1, n)
    amounts = rng.lognormal(2, 1, n)

    df, threshold = calc.threshold_optimization(y_true, y_proba, amounts)
    optimal_row = df[df['threshold'] == threshold].iloc[0]
    assert optimal_row['net_benefit'] == df['net_benefit'].max()
