import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

df = pd.read_csv("1.XGBoost//Cleaned_Dataset.csv")

target_col = 'price'
X = df.drop(columns=[target_col])
y = np.log1p(df[target_col])  # log-transform target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

xgb = XGBRegressor(objective='reg:squarederror', random_state=42)

# param_grid = {
#     'n_estimators': [100, 200, 300, 400],                     # Number of boosting rounds
#     'learning_rate': [0.01, 0.05, 0.1, 0.2],                  # Step size for each boosting round
#     'max_depth': [3, 5, 7, 9],                                  # Maximum depth of each tree
#     'min_child_weight': [1, 3, 5, 7],                           # Minimum sum of instance weight needed in a child
#     'subsample': [0.6, 0.7, 0.8, 0.9, 1.0],                    # Fraction of samples used per boosting round
#     'gamma': [0, 0.1, 0.2, 0.3, 0.4],                         # Minimum loss reduction required to make a partition
#     'scale_pos_weight': [1, 2, 3],                             # Controls the balance of positive and negative weights
#     'reg_alpha': [0, 0.1, 0.5, 1],                             # L1 regularization term on weights
#     'reg_lambda': [0, 0.1, 0.5, 1],                            # L2 regularization term on weights
#     'booster': ['gbtree', 'gblinear', 'dart'],                 # The type of boosting model
#     'max_delta_step': [0, 1, 5], 
#     'objective': ['reg:squarederror'],                         # Objective function
# }

param_grid = {
    'n_estimators': [550],                     # Number of boosting rounds
    'learning_rate': [0.05],                  # Step size for each boosting round
    'max_depth': [7],                                  # Maximum depth of each tree
    'min_child_weight': [1],                           # Minimum sum of instance weight needed in a child
    'subsample': [0.6],                    # Fraction of samples used per boosting round
    'gamma': [0],                         # Minimum loss reduction required to make a partition
    'scale_pos_weight': [1],                             # Controls the balance of positive and negative weights
    'reg_alpha': [0.1],                             # L1 regularization term on weights
    'max_delta_step': [0],                               # Maximum step size in each boosting round
    'reg_lambda': [1],                            # L2 regularization term on weights
    'booster': ['gbtree'],                 # The type of boosting model
    'objective': ['reg:squarederror'],                         # Objective function
}


grid_search = GridSearchCV(
    estimator=xgb,
    param_grid=param_grid,
    cv=3,
    scoring='neg_mean_squared_error',
    n_jobs=-1,
    verbose=1,
    return_train_score=True
)

grid_search.fit(X_train, y_train)

print("Best parameters found:")
print(grid_search.best_params_)

best_model = grid_search.best_estimator_

y_pred_best = best_model.predict(X_test)

rmse_best = np.sqrt(mean_squared_error(y_test, y_pred_best))
mae_best = mean_absolute_error(y_test, y_pred_best)
r2_best = r2_score(y_test, y_pred_best)
n = len(y_test)
p = X_test.shape[1]
adj_r2_best = 1 - (1 - r2_best) * (n - 1) / (n - p - 1)
rmse_normalized_best = rmse_best / np.mean(y_test)
mae_normalized_best = mae_best / np.mean(y_test)
model_score_best = best_model.score(X_test, y_test)

print("\nXGBoost Evaluation with Hyperparameter Tuning:")
print(f"RMSE: {rmse_best}")
print(f"MAE: {mae_best}")
print(f"Normalized RMSE: {rmse_normalized_best}")
print(f"Normalized MAE: {mae_normalized_best}")
print(f"R²: {r2_best}")
print(f"Adjusted R²: {adj_r2_best}")


import joblib

joblib.dump(best_model, 'best_xgboost_model.pkl')
print("\nModel saved as 'best_xgboost_model.pkl'")