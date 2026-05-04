import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_auc_score, roc_curve, precision_recall_curve,
    average_precision_score
)
import logging


class ModelEvaluator:
    def calculate_metrics(self, y_true, y_pred, y_proba):
        return {
            'roc_auc': roc_auc_score(y_true, y_proba),
            'average_precision': average_precision_score(y_true, y_proba),
            'classification_report': classification_report(y_true, y_pred, output_dict=True)
        }

    def plot_roc_curves(self, models_results, save_path=None):
        plt.figure(figsize=(10, 8))
        for model_name, results in models_results.items():
            fpr, tpr, _ = roc_curve(results['y_true'], results['y_proba'])
            auc = roc_auc_score(results['y_true'], results['y_proba'])
            plt.plot(fpr, tpr, label=f'{model_name} (AUC = {auc:.3f})')
        plt.plot([0, 1], [0, 1], 'k--', label='Random')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curves — Model Comparison')
        plt.legend()
        plt.grid(True)
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

    def plot_precision_recall_curves(self, models_results, save_path=None):
        plt.figure(figsize=(10, 8))
        for model_name, results in models_results.items():
            precision, recall, _ = precision_recall_curve(results['y_true'], results['y_proba'])
            avg_precision = average_precision_score(results['y_true'], results['y_proba'])
            plt.plot(recall, precision, label=f'{model_name} (AP = {avg_precision:.3f})')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curves — Model Comparison')
        plt.legend()
        plt.grid(True)
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

    def plot_feature_importance(self, feature_names, importance_dict, top_n=20, save_path=None):
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        axes = axes.ravel()

        for idx, (model_name, importance) in enumerate(importance_dict.items()):
            if idx >= 4:
                break
            top_indices = np.argsort(importance)[-top_n:]
            top_features = [feature_names[i] for i in top_indices]
            top_importance = importance[top_indices]

            axes[idx].barh(range(len(top_features)), top_importance)
            axes[idx].set_yticks(range(len(top_features)))
            axes[idx].set_yticklabels(top_features)
            axes[idx].set_title(f'{model_name} — Top {top_n} Features')
            axes[idx].set_xlabel('Importance')

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

    def plot_confusion_matrices(self, models_results, save_path=None):
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        axes = axes.ravel()

        for idx, (model_name, results) in enumerate(models_results.items()):
            if idx >= 4:
                break
            cm = confusion_matrix(results['y_true'], results['y_pred'])
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[idx])
            axes[idx].set_title(f'{model_name} — Confusion Matrix')
            axes[idx].set_xlabel('Predicted')
            axes[idx].set_ylabel('Actual')

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

    def create_interactive_dashboard(self, models_results, save_path=None):
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('ROC Curves', 'Precision-Recall Curves', 'Model ROC-AUC', ''),
            specs=[[{"type": "scatter"}, {"type": "scatter"}],
                   [{"type": "bar"}, {"type": "scatter"}]]
        )

        for model_name, results in models_results.items():
            fpr, tpr, _ = roc_curve(results['y_true'], results['y_proba'])
            auc = roc_auc_score(results['y_true'], results['y_proba'])
            fig.add_trace(
                go.Scatter(x=fpr, y=tpr, mode='lines', name=f'{model_name} (AUC={auc:.3f})'),
                row=1, col=1
            )

        fig.add_trace(
            go.Scatter(x=[0, 1], y=[0, 1], mode='lines', line=dict(dash='dash'), name='Random'),
            row=1, col=1
        )

        for model_name, results in models_results.items():
            precision, recall, _ = precision_recall_curve(results['y_true'], results['y_proba'])
            avg_precision = average_precision_score(results['y_true'], results['y_proba'])
            fig.add_trace(
                go.Scatter(x=recall, y=precision, mode='lines',
                           name=f'{model_name} (AP={avg_precision:.3f})', showlegend=False),
                row=1, col=2
            )

        model_names = list(models_results.keys())
        roc_aucs = [roc_auc_score(r['y_true'], r['y_proba']) for r in models_results.values()]
        fig.add_trace(go.Bar(x=model_names, y=roc_aucs, name='ROC-AUC', showlegend=False), row=2, col=1)

        fig.update_layout(height=800, title_text="Fraud Detection Model Dashboard")

        if save_path:
            fig.write_html(save_path)

        return fig


class BusinessMetricsCalculator:
    def __init__(self):
        self.investigation_cost_per_case = 50

    def calculate_business_impact(self, y_true, y_pred, y_proba, transaction_amounts):
        tp = np.sum((y_true == 1) & (y_pred == 1))
        fp = np.sum((y_true == 0) & (y_pred == 1))
        fn = np.sum((y_true == 1) & (y_pred == 0))
        tn = np.sum((y_true == 0) & (y_pred == 0))

        fraud_detected_amount = np.sum(transaction_amounts[(y_true == 1) & (y_pred == 1)])
        fraud_missed_amount = np.sum(transaction_amounts[(y_true == 1) & (y_pred == 0)])
        legitimate_blocked_amount = np.sum(transaction_amounts[(y_true == 0) & (y_pred == 1)])

        return {
            'fraud_detection_rate': tp / (tp + fn) if (tp + fn) > 0 else 0,
            'false_positive_rate': fp / (fp + tn) if (fp + tn) > 0 else 0,
            'precision': tp / (tp + fp) if (tp + fp) > 0 else 0,
            'recall': tp / (tp + fn) if (tp + fn) > 0 else 0,
            'fraud_prevented_amount': fraud_detected_amount,
            'fraud_missed_amount': fraud_missed_amount,
            'legitimate_blocked_amount': legitimate_blocked_amount,
            'investigation_costs': fp * self.investigation_cost_per_case,
            'net_benefit': fraud_detected_amount - (fp * self.investigation_cost_per_case),
            'total_investigations_required': tp + fp,
            'investigation_efficiency': tp / (tp + fp) if (tp + fp) > 0 else 0
        }

    def threshold_optimization(self, y_true, y_proba, transaction_amounts):
        thresholds = np.arange(0.1, 0.95, 0.05)
        results = []
        for threshold in thresholds:
            y_pred = (y_proba >= threshold).astype(int)
            metrics = self.calculate_business_impact(y_true, y_pred, y_proba, transaction_amounts)
            metrics['threshold'] = threshold
            results.append(metrics)

        threshold_df = pd.DataFrame(results)
        optimal_idx = threshold_df['net_benefit'].idxmax()
        optimal_threshold = threshold_df.iloc[optimal_idx]['threshold']
        return threshold_df, optimal_threshold
