# ML_GHANA

📁 Dataset
Files:

ghana_rentals.csv
Cleaned_Ghana_Rentals.csv

Source:
Tonaton.com - Ghana Rental Listings

Description:
Contains 17,890 rental listings from Ghana with features like location, bedrooms, bathrooms, furnishing status, and rental price.

🧠 Models Implemented
Linear Regression
Random Forest Regressor
XGBoost Regressor
CatBoost Regressor
Support Vector Regression (SVR)

🛠️ Libraries Used
pandas, numpy, scikit-learn
xgboost, catboost
matplotlib, seaborn, joblib

📂 Python Files Structure
🔹 Models
linear_regression.py, random_forest.py, xgboost_standard.py, catboost_standard.py, svr_standard.py
_log.py versions apply log transformation on the target variable
_tuned.py versions include hyperparameter tuning

🔹 Data
ghana_rentals.csv – Raw scraped data
Cleaned_Ghana_Rentals.csv – Preprocessed dataset

🚀 Key Steps in Implementation
Collected rental data from Tonaton.com
Cleaned and preprocessed data (outliers, encoding, geocoding)
Engineered features like location and room counts
Trained & tuned multiple ML models
Evaluated using R², MAE, RMSE – CatBoost performed best (R² = 0.876)
Analyzed features – Location, bedrooms, and furnishing were most impactful

📊 Evaluation Metrics
MAE, MSE, RMSE, R² Score

💾 Model Saving
Best model saved as catboost_model.cbm

Load with:
from catboost import CatBoostRegressor
model = CatBoostRegressor() 
model.load_model("catboost_model.cbm")