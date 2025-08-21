# Credit Card Fraud Detection System

## Business Problem
Credit card fraud costs the financial industry billions annually. This project builds a machine learning system to detect fraudulent transactions in real-time, balancing fraud detection with customer experience.

## Key Achievements
- **98.0% ROC-AUC** with ensemble model 🏆
- **97-98% fraud detection rate** across multiple algorithms
- **$5,976 net benefit** demonstrated on test data
- **Real-time scoring** capability with production-ready pipeline
- **Interactive Tableau dashboard** with logarithmic scaling for imbalanced data visualization

## Technical Highlights
- Advanced feature engineering with domain expertise
- Comprehensive model comparison (5 algorithms)
- Class imbalance handling with SMOTE and ensemble methods
- Production-ready pipeline with comprehensive testing
- Professional BI dashboard solving data visualization challenges

## Dataset
Using the [Credit Card Fraud Detection Dataset](https://www.kaggle.com/mlg-ulb/creditcardfraud) from Kaggle:
- **284,807 transactions** over 2 days
- **492 fraudulent cases** (0.173% fraud rate)
- **Features**: Time, Amount, and 28 anonymized PCA features

## 📊 Interactive Dashboard

### Tableau Public Dashboard
🔗 **[View Live Dashboard](https://public.tableau.com/views/Book1_17558033052330/Dashboard1?:language=en-US&publish=yes&:sid=&:redirect=auth&:display_count=n&:origin=viz_share_link)**

![Dashboard Overview](tableau-dashboard/dashboard-overview.png)

### Key Visualizations
- **Transaction Distribution (Log Scale)**: Clearly shows fraud vs legitimate transactions
- **Model Performance**: 98% ROC-AUC across 5 different algorithms  
- **Business Impact**: $5,976 net benefit demonstration
- **Key Metrics**: 284,807 transactions analyzed with 0.173% fraud rate

### Technical Achievements
- ✅ Solved data visualization challenge with logarithmic scaling
- ✅ Professional BI dashboard using Tableau Public
- ✅ Interactive filtering and drill-down capabilities
- ✅ Industry-standard data visualization practices

## Quick Start
```bash
# Clone repository
git clone https://github.com/yourusername/fraud-detection-system.git
cd fraud-detection-system

# Install dependencies
pip install -r requirements.txt

# Run the complete pipeline
python src/main.py

# View interactive dashboard
open results/fraud_detection_dashboard.html
```

## Model Performance

| Model | ROC-AUC | Key Strength |
|-------|---------|--------------|
| **Ensemble** | **98.0%** | **Best overall performance** |
| Random Forest | 97.9% | Feature importance insights |
| LightGBM | 97.8% | Fast training speed |
| Logistic Regression | 97.6% | Interpretability |
| XGBoost | 97.3% | Robust to overfitting |

## Business Impact

- **Fraud Prevention**: $5,976 demonstrated on test data
- **Optimal Threshold**: 45% probability cutoff
- **Investigation Efficiency**: Focused review on high-risk transactions
- **Customer Experience**: Minimal false positives with high detection rate

## Repository Structure
```
fraud-detection-system/
├── README.md
├── requirements.txt
├── data/
│   ├── raw/
│   └── processed/
├── src/
│   ├── data_preprocessing.py
│   ├── feature_engineering.py
│   ├── models.py
│   ├── utils.py
│   └── main.py
├── config/
│   └── model_configs.py
├── tableau-dashboard/
│   ├── fraud-detection-dashboard.twbx
│   ├── dashboard-overview.png
│   ├── transaction-distribution.png
│   ├── model-performance.png
│   └── business-impact.png
├── results/
│   ├── fraud_detection_dashboard.html
│   ├── roc_curves.png
│   └── project_overview.png
└── tests/
    └── test_models.py
```

## Technologies Used

- **Machine Learning**: Python, scikit-learn, XGBoost, LightGBM, ensemble methods
- **Data Processing**: pandas, numpy, SMOTE for imbalance handling
- **Visualization**: Plotly, matplotlib, Tableau Public
- **Development**: pytest, modular design, configuration management
- **Business Intelligence**: Interactive dashboards, logarithmic scaling

## Key Features

- **Data Preprocessing**: Automated cleaning, scaling, and validation
- **Feature Engineering**: Advanced techniques for time-series financial data
- **Model Training**: Multiple algorithms with ensemble optimization
- **Evaluation**: Comprehensive metrics including business impact analysis
- **Visualization**: Interactive dashboards solving imbalanced data visualization
- **Production Ready**: Modular code, testing, and deployment configuration

## Usage Examples
```python
# Load trained model
from src.models import FraudDetectionModels
import joblib

model = joblib.load('results/model_ensemble.joblib')

# Score new transaction
fraud_probability = model.predict_proba(new_transaction)[0][1]

# Apply optimized threshold
is_fraud = fraud_probability > 0.45  # 45% threshold optimized for business impact
```

## Results
The final ensemble model achieves:

- **98.0% ROC-AUC** on test data
- **97-98% fraud detection rate** across models
- **$5,976 net benefit** on test dataset
- **Optimal 45% threshold** for business impact

## Future Improvements

- Real-time streaming pipeline integration
- Deep learning model exploration (neural networks)
- Advanced anomaly detection techniques
- A/B testing framework for threshold optimization
- Integration with cloud platforms (AWS, GCP)

## Author
**Abhista Atchutuni**
- LinkedIn: https://linkedin.com/in/abhistac
- Email: abhistaca@gmail.com