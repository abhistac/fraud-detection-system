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


def load_config():
    with open('config/config.yaml', 'r') as f:
        return yaml.safe_load(f)


def download_data():
    print("Download the Credit Card Fraud Detection dataset from Kaggle:")
    print("https://www.kaggle.com/mlg-ulb/creditcardfraud")
    print("Save as: data/raw/creditcard.csv")

    data_path = Path("data/raw/creditcard.csv")
    if not data_path.exists():
        logging.warning("Dataset not found — generating sample data for demonstration")
        create_sample_data()

    return str(data_path)


def create_sample_data():
    os.makedirs("data/raw", exist_ok=True)

    np.random.seed(42)
    n_samples = 10000
    n_fraud = int(n_samples * 0.002)

    rows = []
    for i in range(n_samples):
        time = np.random.uniform(0, 172800)
        v_features = np.random.normal(0, 1, 28)
        if i < n_fraud:
            amount = np.random.lognormal(3, 1.5)
            v_features[:5] = np.random.normal(2, 1, 5)
            class_label = 1
        else:
            amount = np.random.lognormal(2, 1)
            class_label = 0
        rows.append([time] + list(v_features) + [amount, class_label])

    columns = ['Time'] + [f'V{i}' for i in range(1, 29)] + ['Amount', 'Class']
    df = pd.DataFrame(rows, columns=columns)
    df.to_csv("data/raw/creditcard.csv", index=False)
    logging.info(f"Sample data: {len(df)} rows, {df['Class'].sum()} fraud cases")


def main():
    config = load_config()

    os.makedirs("results", exist_ok=True)
    os.makedirs("data/processed", exist_ok=True)

    data_path = download_data()

    preprocessor = DataPreprocessor(config)
    feature_engineer = FeatureEngineer(config)
    models = FraudDetectionModels(config)
    evaluator = ModelEvaluator()
    business_calculator = BusinessMetricsCalculator()

    logging.info("Loading data...")
    df = preprocessor.load_data(data_path)
    preprocessor.explore_data(df)

    logging.info("Preprocessing...")
    df_clean = preprocessor.handle_missing_values(df)

    logging.info("Engineering features...")
    df_featured = feature_engineer.engineer_features(df_clean)

    X_train, X_test, y_train, y_test = preprocessor.split_data(df_featured)
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
        X_test['Amount'].values
    )

    _, optimal_threshold = business_calculator.threshold_optimization(
        ensemble['y_true'],
        ensemble['y_proba'],
        X_test['Amount'].values
    )

    logging.info(f"Optimal threshold: {optimal_threshold:.3f}")
    logging.info(f"Net benefit: ${business_metrics['net_benefit']:,.0f}")

    logging.info("Generating visualizations...")
    evaluator.plot_roc_curves(model_results, 'results/roc_curves.png')
    evaluator.plot_precision_recall_curves(model_results, 'results/pr_curves.png')
    evaluator.plot_feature_importance(
        X_train_scaled.columns.tolist(),
        models.feature_importance,
        save_path='results/feature_importance.png'
    )
    evaluator.plot_confusion_matrices(model_results, 'results/confusion_matrices.png')
    evaluator.create_interactive_dashboard(model_results, 'results/dashboard.html')

    logging.info("Saving models...")
    models.save_models('results/model')

    summary = {
        'optimal_threshold': float(optimal_threshold),
        'business_metrics': {k: float(v) for k, v in business_metrics.items()},
        'cv_results': {k: {'mean': float(v['mean']), 'std': float(v['std'])} for k, v in cv_results.items()},
        'model_performance': {
            name: {'roc_auc': float(roc_auc_score(r['y_true'], r['y_proba']))}
            for name, r in model_results.items()
        }
    }

    with open('results/summary.json', 'w') as f:
        json.dump(summary, f, indent=2)

    logging.info("Pipeline complete. Results in results/")
    return summary


if __name__ == "__main__":
    main()
