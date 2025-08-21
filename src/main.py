import pandas as pd
import numpy as np
import yaml
import logging
import os
from pathlib import Path

from data_preprocessing import DataPreprocessor
from feature_engineering import FeatureEngineer
from models import FraudDetectionModels
from utils import ModelEvaluator, BusinessMetricsCalculator

# Setup logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(levelname)s - %(message)s')

def load_config():
    """Load configuration file"""
    with open('config/config.yaml', 'r') as file:
        config = yaml.safe_load(file)
    return config

def download_data():
    """Download and prepare the dataset"""
    # For this example, we'll simulate the dataset
    # In practice, you would download from Kaggle using their API
    
    print("Note: Please download the Credit Card Fraud Detection dataset from Kaggle")
    print("URL: https://www.kaggle.com/mlg-ulb/creditcardfraud")
    print("Save as: data/raw/creditcard.csv")
    
    # Check if data exists
    data_path = Path("data/raw/creditcard.csv")
    if not data_path.exists():
        # Create sample data for demonstration
        logging.warning("Creating sample data for demonstration")
        create_sample_data()
    
    return str(data_path)

def create_sample_data():
    """Create sample data for demonstration purposes"""
    
    # Create directories
    os.makedirs("data/raw", exist_ok=True)
    
    # Generate sample data that mimics the real dataset structure
    np.random.seed(42)
    n_samples = 10000
    n_fraud = int(n_samples * 0.002)  # 0.2% fraud rate
    
    # Create sample data
    data = []
    
    for i in range(n_samples):
        # Time feature (0 to 172800 seconds = 48 hours)
        time = np.random.uniform(0, 172800)
        
        # Generate PCA features V1-V28
        v_features = np.random.normal(0, 1, 28)
        
        # Amount feature
        if i < n_fraud:
            # Fraudulent transactions - different distribution
            amount = np.random.lognormal(3, 1.5)
            class_label = 1
            # Modify some V features for fraud
            v_features[:5] = np.random.normal(2, 1, 5)  # Different pattern
        else:
            # Normal transactions
            amount = np.random.lognormal(2, 1)
            class_label = 0
        
        # Create row
        row = [time] + list(v_features) + [amount, class_label]
        data.append(row)
    
    # Create DataFrame
    columns = ['Time'] + [f'V{i}' for i in range(1, 29)] + ['Amount', 'Class']
    df = pd.DataFrame(data, columns=columns)
    
    # Save sample data
    df.to_csv("data/raw/creditcard.csv", index=False)
    logging.info(f"Sample data created with {len(df)} rows, {df['Class'].sum()} fraud cases")

def main():
    """Main execution pipeline"""
    
    # Load configuration
    config = load_config()
    
    # Setup paths
    os.makedirs("results", exist_ok=True)
    os.makedirs("data/processed", exist_ok=True)
    
    # Download/prepare data
    data_path = download_data()
    
    # Initialize components
    preprocessor = DataPreprocessor(config)
    feature_engineer = FeatureEngineer(config)
    models = FraudDetectionModels(config)
    evaluator = ModelEvaluator()
    business_calculator = BusinessMetricsCalculator()
    
    # Load and explore data
    logging.info("Loading and exploring data...")
    df = preprocessor.load_data(data_path)
    exploration_summary = preprocessor.explore_data(df)
    
    # Data preprocessing
    logging.info("Preprocessing data...")
    df_clean = preprocessor.handle_missing_values(df)
    outliers_info = preprocessor.detect_outliers(df_clean)
    
    # Feature engineering
    logging.info("Engineering features...")
    df_featured = feature_engineer.engineer_features(df_clean)
    
    # Split data
    X_train, X_test, y_train, y_test = preprocessor.split_data(df_featured)
    
    # Scale features
    X_train_scaled, X_test_scaled = preprocessor.scale_features(X_train, X_test)
    
    # Train models
    logging.info("Training models...")
    models.train_all_models(X_train_scaled, y_train)
    
    # Cross-validation
    logging.info("Performing cross-validation...")
    cv_results = models.cross_validate_models(X_train_scaled, y_train)
    
    # Evaluate models
    logging.info("Evaluating models...")
    model_results = {}
    
    for model_name in models.trained_models.keys():
        y_pred = models.predict(model_name, X_test_scaled)
        y_proba = models.predict_proba(model_name, X_test_scaled)
        
        model_results[model_name] = {
            'y_true': y_test,
            'y_pred': y_pred,
            'y_proba': y_proba
        }
        
        # Calculate metrics
        metrics = evaluator.calculate_metrics(y_test, y_pred, y_proba)
        
        logging.info(f"{model_name} - ROC-AUC: {metrics['roc_auc']:.4f}")
    
    # Business impact analysis
    logging.info("Calculating business impact...")
    
    # Use ensemble model for business analysis
    ensemble_results = model_results['ensemble']
    business_metrics = business_calculator.calculate_business_impact(
        ensemble_results['y_true'],
        ensemble_results['y_pred'],
        ensemble_results['y_proba'],
        X_test['Amount'].values
    )
    
    # Threshold optimization
    threshold_df, optimal_threshold = business_calculator.threshold_optimization(
        ensemble_results['y_true'],
        ensemble_results['y_proba'],
        X_test['Amount'].values
    )
    
    logging.info(f"Optimal threshold: {optimal_threshold:.3f}")
    logging.info(f"Net benefit at optimal threshold: ${business_metrics['net_benefit']:,.0f}")
    
    # Generate visualizations
    logging.info("Generating visualizations...")
    
    # ROC curves
    evaluator.plot_roc_curves(model_results, 'results/roc_curves.png')
    
    # Precision-recall curves
    evaluator.plot_precision_recall_curves(model_results, 'results/pr_curves.png')
    
    # Feature importance
    feature_names = X_train_scaled.columns.tolist()
    evaluator.plot_feature_importance(
        feature_names, 
        models.feature_importance, 
        save_path='results/feature_importance.png'
    )
    
    # Confusion matrices
    evaluator.plot_confusion_matrices(model_results, 'results/confusion_matrices.png')
    
    # Interactive dashboard
    dashboard = evaluator.create_interactive_dashboard(
        model_results, 
        'results/dashboard.html'
    )
    
    # Save models
    logging.info("Saving models...")
    models.save_models('results/model')
    
    # Save results summary
    results_summary = {
        'exploration_summary': exploration_summary,
        'cv_results': cv_results,
        'business_metrics': business_metrics,
        'optimal_threshold': optimal_threshold,
        'model_performance': {
            name: {
                'roc_auc': roc_auc_score(results['y_true'], results['y_proba']),
                'precision': evaluator.calculate_metrics(
                    results['y_true'], results['y_pred'], results['y_proba']
                )['classification_report']['1']['precision'],
                'recall': evaluator.calculate_metrics(
                    results['y_true'], results['y_pred'], results['y_proba']
                )['classification_report']['1']['recall']
            }
            for name, results in model_results.items()
        }
    }
    
    # Save as JSON
    import json
    with open('results/summary.json', 'w') as f:
        json.dump(results_summary, f, indent=2, default=str)
    
    logging.info("Pipeline completed successfully!")
    logging.info("Check the 'results/' directory for outputs")
    
    return results_summary

if __name__ == "__main__":
    results = main()