import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

df = pd.read_csv("C:\\Users\\Lenovo\\Documents\\ML FA2\\1.XGBoost\\Cleaned_Dataset.csv")

target_col = 'price'
X = df.drop(columns=[target_col])
y = df[target_col]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

svr_model = SVR(kernel='rbf')
svr_model.fit(X_train, y_train)

y_pred = svr_model.predict(X_test)

rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
n = len(y_test)
p = X_test.shape[1]
adj_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1)
rmse_normalized = rmse / np.mean(y_test)
mae_normalized = mae / np.mean(y_test)
model_score = svr_model.score(X_test, y_test)

print("SVR Evaluation :")
print(f"RMSE: {rmse}")
print(f"MAE: {mae}")
print(f"Normalized RMSE: {rmse_normalized}")
print(f"Normalized MAE: {mae_normalized}")
print(f"R²: {r2}")
print(f"Adjusted R²: {adj_r2}")
print(f"Model Score (R²): {model_score}")
