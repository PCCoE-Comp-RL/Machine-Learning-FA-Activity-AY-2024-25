import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

df = pd.read_csv("C:\\Users\\Lenovo\\Documents\\ML FA2\\1.XGBoost\\Cleaned_Dataset.csv")

target_col = 'price'
X = df.drop(columns=[target_col])
y = np.log1p(df[target_col])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

rf_model = RandomForestRegressor(random_state=42)
rf_model.fit(X_train, y_train)

y_pred = rf_model.predict(X_test)

rmse_log = np.sqrt(mean_squared_error(y_test, y_pred))
mae_log = mean_absolute_error(y_test, y_pred)
r2_log = r2_score(y_test, y_pred)
n_log = len(y_test)
p_log = X_test.shape[1]
adj_r2_log = 1 - (1 - r2_log) * (n_log - 1) / (n_log - p_log - 1)
rmse_normalized_log = rmse_log / np.mean(y_test)
mae_normalized_log = mae_log / np.mean(y_test)
model_score_log = rf_model.score(X_test, y_test)

print("\nRandom Forest Regressor Evaluation (With Log Transformation):")
print(f"RMSE: {rmse_log}")
print(f"MAE: {mae_log}")
print(f"Normalized RMSE: {rmse_normalized_log}")
print(f"Normalized MAE: {mae_normalized_log}")
print(f"R²: {r2_log}")
print(f"Adjusted R²: {adj_r2_log}")
# print(f"Model Score (Log R²): {model_score_log}")
print(f"Model Score: {model_score_log}")