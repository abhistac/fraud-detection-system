import os
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import roc_auc_score

from data_preprocessing import DataPreprocessor
from feature_engineering import FeatureEngineer
from models import FraudDetectionModels
from utils import ModelEvaluator, BusinessMetricsCalculator

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

CONFIG_PATH = Path(__file__).parent.parent / 'config' / 'config.yaml'


def load_config():
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


def download_data():
    print("Download the Credit Card Fraud Detection dataset from Kaggle:")
    print("https://www.kaggle.com/mlg-ulb/creditcardfraud")
    print("Save as: data/raw/creditcard.csv")

    data_path = Path(__file__).parent.parent / 'data' / 'raw' / 'creditcard.csv'
    if not data_path.exists():
        logging.warning("Dataset not found — generating sample data for demonstration")
        create_sample_data(data_path)

    return str(data_path)


def create_sample_data(data_path):
    data_path.parent.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(42)
    n_samples, n_fraud = 10000, 20

    time = rng.uniform(0, 172800, n_samples)
    v_features = rng.standard_normal((n_samples, 28))
    amount = rng.lognormal(2, 1, n_samples)
    labels = np.zeros(n_samples, dtype=int)

    v_features[:n_fraud, :5] = rng.normal(2, 1, (n_fraud, 5))
    amount[:n_fraud] = rng.lognormal(3, 1.5, n_fraud)
    labels[:n_fraud] = 1

    data = np.column_stack([time, v_features, amount, labels])
    columns = ['Time'] + [f'V{i}' for i in range(1, 29)] + ['Amount', 'Class']
    df = pd.DataFrame(data, columns=columns)
    df['Class'] = df['Class'].astype(int)
    df.to_csv(data_path, index=False)
    logging.info(f"Sample data: {len(df)} rows, {df['Class'].sum()} fraud cases")


def main():
    config = load_config()

    results_dir = Path(__file__).parent.parent / 'results'
    results_dir.mkdir(exist_ok=True)

    data_path = download_data()

    preprocessor = DataPreprocessor(config)
    feature_engineer = FeatureEngineer(config)
    models = FraudDetectionModels(config)
    evaluator = ModelEvaluator()
    business_calculator = BusinessMetricsCalculator()

    logging.info("Loading data...")
    df = preprocessor.load_data(data_path)
    preprocessor.explore_data(df)
    df = preprocessor.handle_missing_values(df)

    # Split before feature engineering to prevent data leakage
    X_train_raw, X_test_raw, y_train_raw, y_test_raw = preprocessor.split_data(df)

    # Preserve original test amounts for business impact (before scaling)
    test_amounts = X_test_raw['Amount'].values

    logging.info("Engineering features...")
    df_train = X_train_raw.assign(Class=y_train_raw)
    df_test = X_test_raw.assign(Class=y_test_raw)

    df_train_featured = feature_engineer.fit_transform(df_train)
    df_test_featured = feature_engineer.transform(df_test)

    X_train = df_train_featured.drop('Class', axis=1)
    y_train = df_train_featured['Class'].astype(int)
    X_test = df_test_featured.drop('Class', axis=1)
    y_test = df_test_featured['Class'].astype(int)

    X_train_scaled, X_test_scaled = preprocessor.scale_features(X_train, X_test)

    logging.info("Training models...")
    models.train_all_models(X_train_scaled, y_train)

    logging.info("Cross-validating...")
    cv_results = models.cross_validate_models(X_train_scaled, y_train)

    logging.info("Evaluating models...")
    model_results = {}
    for model_name in models.trained_models:
        y_pred = models.predict(model_name, X_test_scaled)
        y_proba = models.predict_proba(model_name, X_test_scaled)
        model_results[model_name] = {'y_true': y_test, 'y_pred': y_pred, 'y_proba': y_proba}
        metrics = evaluator.calculate_metrics(y_test, y_pred, y_proba)
        logging.info(f"{model_name} - ROC-AUC: {metrics['roc_auc']:.4f}")

    logging.info("Calculating business impact...")
    ensemble = model_results['ensemble']
    business_metrics = business_calculator.calculate_business_impact(
        ensemble['y_true'],
        ensemble['y_pred'],
        ensemble['y_proba'],
        test_amounts
    )

    _, optimal_threshold = business_calculator.threshold_optimization(
        ensemble['y_true'],
        ensemble['y_proba'],
        test_amounts
    )

    logging.info(f"Optimal threshold: {optimal_threshold:.3f}")
    logging.info(f"Net benefit: ${business_metrics['net_benefit']:,.0f}")

    logging.info("Generating visualizations...")
    evaluator.plot_roc_curves(model_results, str(results_dir / 'roc_curves.png'))
    evaluator.plot_precision_recall_curves(model_results, str(results_dir / 'pr_curves.png'))
    evaluator.plot_feature_importance(
        X_train_scaled.columns.tolist(),
        models.feature_importance,
        save_path=str(results_dir / 'feature_importance.png')
    )
    evaluator.plot_confusion_matrices(model_results, str(results_dir / 'confusion_matrices.png'))
    evaluator.create_interactive_dashboard(model_results, str(results_dir / 'dashboard.html'))

    logging.info("Saving models...")
    models.save_models(str(results_dir / 'model'))

    summary = {
        'optimal_threshold': float(optimal_threshold),
        'business_metrics': {k: float(v) for k, v in business_metrics.items()},
        'cv_results': {k: {'mean': float(v['mean']), 'std': float(v['std'])} for k, v in cv_results.items()},
        'model_performance': {
            name: {'roc_auc': float(roc_auc_score(r['y_true'], r['y_proba']))}
            for name, r in model_results.items()
        }
    }

    with open(results_dir / 'summary.json', 'w') as f:
        json.dump(summary, f, indent=2)

    logging.info("Pipeline complete. Results in results/")
    return summary


if __name__ == "__main__":
    main()
