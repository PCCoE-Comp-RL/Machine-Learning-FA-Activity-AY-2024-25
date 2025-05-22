import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

df = pd.read_csv("C:\\Users\\Lenovo\\Documents\\ML FA2\\1.XGBoost\\Cleaned_Dataset.csv")

target_col = 'price'
X = df.drop(columns=[target_col])
y = np.log1p(df[target_col])  

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = XGBRegressor(objective='reg:squarederror', random_state=42)
model.fit(X_train, y_train)

y_pred_log = model.predict(X_test)

# Inverse transform predictions
# y_pred = np.expm1(y_pred_log)
# y_actual = np.expm1(y_test)

y_pred = y_pred_log
y_actual = y_test

rmse = np.sqrt(mean_squared_error(y_actual, y_pred))
mae = mean_absolute_error(y_actual, y_pred)
r2 = r2_score(y_actual, y_pred)
n = len(y_actual)
p = X_test.shape[1]
adj_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1)
rmse_normalized = rmse / np.mean(y_actual)
mae_normalized = mae / np.mean(y_actual)
model_score = model.score(X_test, y_test)  # on log scale

print("Log-Transformed XGBoost Evaluation:")
print(f"RMSE: {rmse}")
print(f"MAE: {mae}")
print(f"Normalized RMSE: {rmse_normalized}")
print(f"Normalized MAE: {mae_normalized}")
print(f"R²: {r2}")
print(f"Adjusted R²: {adj_r2}")
print(f"Model Score : {model_score}")
