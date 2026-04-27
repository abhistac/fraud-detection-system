# Credit Card Fraud Detection System

Ensemble ML system for detecting credit card fraud on a highly imbalanced dataset — 284,807 transactions with only 492 fraudulent cases (0.17%). Compares five algorithms, handles class imbalance with SMOTE, optimizes the probability threshold for business impact, and delivers results through an interactive Tableau dashboard.

**98.0% ROC-AUC. $5,976 net benefit demonstrated on test data.**

---

## Live dashboard

🔗 [**View on Tableau Public →**](https://public.tableau.com/views/Book1_17558033052330/Dashboard1)

Includes: transaction distribution (log scale), model performance comparison, business impact analysis, and key metrics across all 284K transactions.

---

## Results

| Model | ROC-AUC | Notes |
|-------|---------|-------|
| **Ensemble** | **98.0%** | Best overall — combines RF + LightGBM + XGBoost |
| Random Forest | 97.9% | Best for feature importance interpretation |
| LightGBM | 97.8% | Fastest training |
| Logistic Regression | 97.6% | Most interpretable |
| XGBoost | 97.3% | Most robust to overfitting |

**Business impact on test set:**
- Optimal threshold: 45% probability (tuned for net benefit, not just accuracy)
- Net benefit: $5,976 — accounts for fraud prevented minus cost of false positives
- Fraud detection rate: 97–98% across models

---

## Architecture

```
Raw data (284K transactions)
         │
         ▼
┌─────────────────────┐
│  Data Preprocessing │  Scaling, validation, train/val/test split
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│ Feature Engineering │  Charlson index, time features, interaction terms
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│   SMOTE Balancing   │  Synthetic minority oversampling (0.17% → balanced)
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│  Model Training ×5  │  LR · RF · XGBoost · LightGBM · Ensemble
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│ Threshold Tuning    │  Optimize for net business benefit at 45%
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│ Tableau Dashboard   │  ROC curves, confusion matrix, business impact
└─────────────────────┘
```

---

## Dataset

[Credit Card Fraud Detection](https://www.kaggle.com/mlg-ulb/creditcardfraud) — Kaggle / ULB Machine Learning Group

- 284,807 transactions over 2 days
- 492 fraud cases (0.173% fraud rate)
- 28 anonymized PCA features + Time + Amount

---

## Quick start

```bash
git clone https://github.com/abhistac/fraud-detection-system.git
cd fraud-detection-system
pip install -r requirements.txt

# Download dataset from Kaggle and place in data/raw/
# Then run the full pipeline:
python src/main.py
```

---

## Key technical decisions

**Why SMOTE over class weights?** SMOTE generates synthetic minority samples in feature space rather than just reweighting the loss. On this dataset it produced better calibrated probability estimates, which matters when tuning the threshold for business impact.

**Why 45% threshold instead of 50%?** At 50%, the model misses high-confidence fraud cases that fall just below the cutoff. Lowering to 45% recovers these with minimal increase in false positives — net benefit analysis confirmed $5,976 improvement on test data.

**Why ensemble?** Individual models each had blind spots. Combining RF (strong on feature interactions) + LightGBM (fast, handles sparse features) + XGBoost (robust regularization) with soft voting reduced variance without sacrificing any single model's strengths.

---

## Stack

Python · Scikit-learn · XGBoost · LightGBM · SMOTE (imbalanced-learn) · Plotly · Tableau Public · pytest

---

## Author

**Abhista Atchutuni** — AI & Data Engineer  
[linkedin.com/in/abhistac](https://linkedin.com/in/abhistac) · [abhistaca@gmail.com](mailto:abhistaca@gmail.com) · [abhistac.github.io](https://abhistac.github.io)
