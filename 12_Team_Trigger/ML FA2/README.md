# 🏠 Delhi Housing Price Prediction - ML FA2 Project

## 📌 Project Overview

### [🚀 View the Project on GitHub](https://github.com/ParthGhadole/XGBoost_Delhi_ML_FA2)

This project focuses on comparing multiple regression models to predict **housing prices in Delhi, India**. The dataset includes features such as location, area, number of bedrooms (BHK), and furnishing status, which influence real estate prices. Various preprocessing techniques including **standard scaling**, **log transformation**, and **hyperparameter tuning** are applied to improve model performance. The objective is to evaluate and compare different regression algorithms on the same dataset to identify the best-performing model for Delhi's housing market.

## 👥 Contributors

- Rishita Awasthi
- Jiya Bargir
- Visesh Chauhan
- Parth Ghadole

---

## 📁 Dataset

- **Files**:
  - `Cleaned_Dataset.csv`
  - `dataset.csv`

- **Source**:  
  [Kaggle: Housing Price Dataset of Delhi, India](https://www.kaggle.com/datasets/goelyash/housing-price-dataset-of-delhiindia?resource=download)

- **Description**: The dataset contains real estate data from Delhi with multiple attributes including location, area, BHK, and price. It serves as a base for building and evaluating various regression models.

---

## 🧠 Models Implemented

The following regression algorithms are explored and evaluated:

1. **Linear Regression**
2. **Random Forest Regressor**
3. **XGBoost Regressor**
4. **CatBoost Regressor**
5. **Support Vector Regression (SVR)**
6. **K-Nearest Neighbors (KNN)**
7. **ElasticNet**
8. **Gradient Boosting Regressor**

---
## 🛠️ Libraries Used

- `pandas`
- `numpy`
- `scikit-learn`
- `xgboost`
- `catboost`
- `matplotlib`
- `seaborn`
- `joblib`

## 📂 Python Files Structure

### 🔹 1. XGBoost
- `1.understand.py`: Data exploration and understanding  
- `2.preprocess.py`: Data preprocessing steps  
- `3.standard.py`: Basic XGBoost model training  
- `4.Log_Transformed_Standard.py`: XGBoost with log-transformed target  
- `5.Standard_hypertuning.py`: Hyperparameter tuning on standard data  
- `6.Log_Transformed_hypertuning.py`: Tuning with log-transformed target  
- `7.predict.py`: Final prediction and testing  
- `Cleaned_Dataset.csv`, `dataset.csv`: Dataset used for this model

### 🔹 2. Linear Regression
- `Linear_Regression.py`: Basic linear regression
- `Linear_Regression_with_log_target.py`: Linear regression with log target

### 🔹 3. Random Forest Regressor
- `RFR.py`: Standard Random Forest model
- `RFR_with_log_tareget.py`: Random Forest with log target

### 🔹 4. CatBoost
- `CB.py`: Standard CatBoost model
- `CB_with_log_target.py`: CatBoost with log-transformed target

### 🔹 5. SVR
- `SVR.py`: Standard SVR model
- `SVR_with_log_target.py`: SVR with log-transformed target

### 🔹 6. KNN
- `KNN.py`: Standard K-Nearest Neighbors
- `KNN_log.py`: KNN with log-transformed target

### 🔹 7. ElasticNet
- `EN.py`: Basic ElasticNet regression
- `EN_with_Log.py`: ElasticNet with log-transformed target

### 🔹 8. Gradient Boosting Regressor
- `GBR.py`: Standard Gradient Boosting model
- `GBR_with_log.py`: Gradient Boosting with log-transformed target

---

## 📊 Evaluation Metrics

Models are evaluated using the following metrics:

- **MAE** (Mean Absolute Error)  
- **RMSE** (Root Mean Squared Error)  
- **R² Score**
- **Adjusted R² Score**

These metrics help assess the performance and accuracy of each model in predicting house prices.

---

## 💾 Model Saving (Optional)

Models are saved using `joblib`
- best_xgboost_model.pkl

