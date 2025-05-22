# Project: King County Housing Price Prediction

## Introduction
This project aims to predict house prices in King County (USA) using various regression models. We perform exploratory data analysis (EDA), compare different algorithms (Linear Regression, Random Forest, XGBoost, SVR, CatBoost), and tune the best-performing model to achieve accurate predictions.

## Dataset Source
- **File:** `kc_house_data.csv`
- **Source:** [King County House Sales Data on Kaggle](https://www.kaggle.com/harlfoxem/housesalesprediction) (downloaded and saved in the project directory)

## Libraries Used
- **Data manipulation:** `pandas`, `numpy`
- **Visualization:** `matplotlib`, `seaborn`, `yellowbrick`
- **Modeling:**
  - `scikit-learn` (LinearRegression, RandomForestRegressor, SVR, metrics, model_selection)
  - `xgboost` (XGBRegressor)
  - `catboost` (CatBoostRegressor)
  - `shap` (SHAP explainability)
- **Hyperparameter Tuning:** `GridSearchCV`, `RandomizedSearchCV` from `scikit-learn`

## Evaluation Metrics
- **Root Mean Squared Error (RMSE):** Measures average prediction error in the target units (USD).
- **R² Score:** Proportion of variance in the target explained by the model. Values closer to 1.0 indicate better fit.

## Trained Model
- The best-performing model (CatBoostRegressor) is tuned via `RandomizedSearchCV` and saved to disk.

### Saving the Model
```python
import joblib
# After training `best_cat_model`:
joblib.dump(best_cat_model, 'catboost_model.pkl')
```

### Loading the Model
```python
import joblib
model = joblib.load('catboost_model.pkl')
```

## Usage
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Run the notebook or script:
   ```bash
   python train_model.py
   ```
3. Evaluate or make predictions:
   ```python
   model.predict(new_data)
   ```

## File Structure
```
├── data/
│   └── kc_house_data.csv
├── notebooks/
│   └── ML_To_Present.ipynb
├── models/
│   └── catboost_model.pkl
├── train_model.py
├── requirements.txt
└── README.md
```

---
*This README was generated to document the King County Housing Price Prediction project.*

