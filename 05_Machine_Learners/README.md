# 🏡 House Price Prediction using Machine Learning

This repository contains two end-to-end machine learning projects focused on predicting residential property prices using various regression models. Each notebook represents a different geographic market:

- **Chennai, India** (`ML_Miniproject_1.ipynb`)
- **Ames, Iowa, USA** (`ML_Miniproject_2.ipynb`)

These projects include data preprocessing, feature engineering, model training, evaluation, and comparison using linear and ensemble machine learning models.

---

## 📄 Files in this Repository

| File Name               | Description |
|------------------------|-------------|
| `ML_Miniproject_1.ipynb` | Chennai Housing Dataset Project |
| `ML_Miniproject_2.ipynb` | Ames Housing Dataset Project |

---

## 🔍 Project Overviews

### 🏙️ Chennai House Price Prediction (`ML_Miniproject_1.ipynb`)

- **Dataset**: [Chennai Housing Sales Price – Kaggle](https://www.kaggle.com/datasets/kunwarakash/chennai-housing-sales-price)  
  (7,170 entries)
- **Preprocessing**:
  - Renamed columns for consistency
  - Imputed missing values (median for numerical, mode for categorical)
  - Extracted numeric BHK values from "Size" column
  - One-hot encoded categorical features like "Area"
- **Models**:
  - Linear Regression
  - Lasso & Ridge Regression
  - Support Vector Regression (SVR)
  - Random Forest Regressor
  - XGBoost Regressor
- **Best Model**: **XGBoost**, with R² ≈ 0.92

---

### 🏘️ Ames Housing Price Prediction (`ML_Miniproject_2.ipynb`)

- **Dataset**: [Ames Housing Dataset – Kaggle](https://www.kaggle.com/datasets/marcopale/housing)  
  (2,930 entries, 80+ features)
- **Preprocessing**:
  - Imputed missing numerical and categorical values
  - Handled "None" for features like Basement/Garage
  - Used ordinal encoding for quality features
  - Applied one-hot encoding for nominal features
  - Scaled numerical data where appropriate
- **Models**:
  - Linear, Lasso, Ridge Regression
  - SVR, Random Forest, XGBoost
- **Best Model**: **XGBoost**, with R² ≈ 0.93

---

