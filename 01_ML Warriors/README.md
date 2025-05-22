# 🏠 Ames Housing Price Prediction - XGBoost Model

## 📌 Project Overview

This notebook implements the **XGBoost Regressor** to predict housing prices using the **Ames Housing Dataset**. The dataset includes a comprehensive set of residential property features such as lot size, overall quality, number of rooms, garage details, and neighborhood — all of which influence real estate values.

The notebook follows a full machine learning pipeline with preprocessing, feature engineering, modeling, hyperparameter tuning, and evaluation, focused solely on XGBoost.

---

## 📁 Dataset

- **File Used**:
  - `AmesHousing.csv`

- **Source**:  
  [Kaggle - Ames Housing Dataset](https://www.kaggle.com/datasets/prevek18/ames-housing-dataset)

- **Description**:  
  The dataset contains **79 explanatory variables** describing (almost) every aspect of residential homes in Ames, Iowa. It is a widely-used alternative to the Boston Housing dataset for regression tasks.

---

## 📂 Notebook Overview

**File**: `xgboost.ipynb`

The notebook covers the following stages:

1. **Data Loading & Cleaning**  
   - Reads the CSV file  
   - Handles missing values  
   - Drops irrelevant columns  
   - Encodes categorical variables

2. **Feature Engineering**  
   - Creates or modifies features for better model performance  
   - Selects important features

3. **Preprocessing**  
   - Applies **StandardScaler**  
   - Optionally uses **log transformation** on the target (`SalePrice`)

4. **Model Training**  
   - Uses `XGBRegressor` from the XGBoost library  
   - Optional **hyperparameter tuning** using `GridSearchCV` or manual tuning

5. **Evaluation**  
   - Metrics used:
     - **MAE** (Mean Absolute Error)  
     - **RMSE** (Root Mean Squared Error)  
     - **R² Score**  
     - **Adjusted R² Score**  
   - Visualization of:
     - Actual vs Predicted Prices  
     - Residuals

6. **Saving the Model**  
   - Saves the trained model using `joblib` for future inference

---

## 🧠 Model: XGBoost Regressor

XGBoost is a high-performance gradient boosting framework optimized for speed and accuracy. It efficiently handles structured data and supports parallel processing, regularization, and advanced tree pruning techniques.

---

## 🛠️ Libraries Used

```python
pandas
numpy
scikit-learn
xgboost
matplotlib
seaborn                                        
joblib
```
---

## To install all dependencies in another environment or system
pip install -r requirements.txt
