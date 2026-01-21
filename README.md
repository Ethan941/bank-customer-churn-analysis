# bank-customer-churn-analysis
# Bank Customer Churn Prediction (Finance / Banking Analytics)

## Business context
Banks want to reduce customer churn by identifying clients at risk of leaving and triggering retention actions (offers, advisor outreach, tailored products).

## Project goals
- Explore and understand churn drivers (EDA)
- Build a churn prediction model (baseline + improved)
- Explain predictions with interpretable ML (SHAP)
- Provide actionable business recommendations

## Dataset
Public dataset (e.g., Kaggle: Bank Customer Churn).  
> Note: dataset is not included in this repository. Please download it from the source and place it in `data/raw/`.

## Tech stack
- Python (Pandas, NumPy)
- Data visualization (Matplotlib/Seaborn)
- Machine Learning (Scikit-learn)
- Explainability (SHAP)

## Repository structure
- `notebooks/`: exploration and EDA
- `src/`: reusable code (preprocessing, training, explainability)
- `reports/`: results, figures, business insights
- `models/`: saved trained models (optional)

## How to run
1. Create an environment
```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
