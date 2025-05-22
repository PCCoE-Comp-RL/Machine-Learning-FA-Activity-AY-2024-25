import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

class MumbaiHousePriceModel:
    def __init__(self, data_path):
        self.data_path = data_path
        self.df = None
        self.X = None
        self.y = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.preprocessor = None
        self.models = {}
        self.results = {}
        self.categorical_cols = ['type', 'status', 'age', 'locality', 'region']
        self.numerical_cols = ['area', 'bhk', 'price_per_sqft']
        
    def load_and_prepare_data(self):
        """Load and prepare the dataset"""
        # Load the dataset
        self.df = pd.read_csv(self.data_path)
        
        # Display basic information
        print("Dataset Information:")
        print("-" * 50)
        print(f"Number of rows: {self.df.shape[0]}")
        print(f"Number of columns: {self.df.shape[1]}")
        print("\nColumns:", self.df.columns.tolist())
        print("\nMissing values per column:")
        print(self.df.isnull().sum())
        
        # Prepare features and target
        self.X = self.df.drop(['price_cr'], axis=1)
        self.y = self.df['price_cr']
        
        # Split the data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=0.2, random_state=42
        )
        
        # Create preprocessing pipeline
        self.preprocessor = ColumnTransformer(
            transformers=[
                ('num', StandardScaler(), self.numerical_cols),
                ('cat', OneHotEncoder(handle_unknown='ignore'), self.categorical_cols)
            ])
    
    def train_and_evaluate_model(self, model, model_name, is_tuned=False):
        """Train and evaluate a model"""
        print(f"\nTraining {model_name}...")
        
        # Create pipeline with preprocessing and model
        pipeline = Pipeline(steps=[
            ('preprocessor', self.preprocessor),
            ('model', model)
        ])
        
        # Train the model
        pipeline.fit(self.X_train, self.y_train)
        
        # Make predictions
        y_pred = pipeline.predict(self.X_test)
        
        # Calculate metrics
        mse = mean_squared_error(self.y_test, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(self.y_test, y_pred)
        r2 = r2_score(self.y_test, y_pred)
        
        print(f"\nPerformance metrics for {model_name}:")
        print(f"RMSE: {rmse:.4f}")
        print(f"MAE: {mae:.4f}")
        print(f"R² Score: {r2:.4f}")
        
        # Plot actual vs predicted values
        plt.figure(figsize=(10, 6))
        plt.scatter(self.y_test, y_pred, alpha=0.5)
        plt.plot([self.y_test.min(), self.y_test.max()], 
                [self.y_test.min(), self.y_test.max()], 'r--', lw=2)
        plt.xlabel('Actual Price (Crores)')
        plt.ylabel('Predicted Price (Crores)')
        plt.title(f'{model_name} - Actual vs Predicted Prices')
        plt.savefig(f'MA-fa-2 code/{model_name.lower().replace(" ", "_")}_predictions.png')
        plt.close()
        
        # Plot feature importance for tree-based models
        if hasattr(model, 'feature_importances_'):
            feature_names = (self.numerical_cols + 
                           list(self.preprocessor.named_transformers_['cat']
                               .get_feature_names_out(self.categorical_cols)))
            
            feature_importance = pd.DataFrame({
                'feature': feature_names,
                'importance': model.feature_importances_
            })
            feature_importance = feature_importance.sort_values('importance', ascending=False)
            
            plt.figure(figsize=(12, 6))
            sns.barplot(x='importance', y='feature', data=feature_importance.head(10))
            plt.title(f'Top 10 Feature Importance - {model_name}')
            plt.tight_layout()
            plt.savefig(f'MA-fa-2 code/{model_name.lower().replace(" ", "_")}_feature_importance.png')
            plt.close()
        
        return {
            'model': pipeline,
            'metrics': {
                'rmse': rmse,
                'mae': mae,
                'r2': r2
            }
        }
    
    def train_default_models(self):
        """Train default models"""
        # Initialize default models
        self.models = {
            'Linear Regression': LinearRegression(),
            'Random Forest': RandomForestRegressor(
                n_estimators=100,
                max_depth=10,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42
            ),
            'SVR': SVR(),
            'MLP': MLPRegressor(
                hidden_layer_sizes=(100, 50),
                max_iter=1000,
                random_state=42
            ),
            'XGBoost': XGBRegressor(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                random_state=42
            )
        }
        
        # Train and evaluate each model
        for model_name, model in self.models.items():
            self.results[model_name] = self.train_and_evaluate_model(model, model_name)
    
    def hyperparameter_tuning(self):
        """Perform hyperparameter tuning for models"""
        # Define parameter grids for each model
        param_grids = {
            'Random Forest': {
                'model__n_estimators': [50, 100, 200],
                'model__max_depth': [5, 10, 15],
                'model__min_samples_split': [2, 5, 10],
                'model__min_samples_leaf': [1, 2, 4]
            },
            'SVR': {
                'model__kernel': ['linear', 'rbf'],
                'model__C': [0.1, 1, 10],
                'model__epsilon': [0.01, 0.1, 1]
            },
            'MLP': {
                'model__hidden_layer_sizes': [(50,), (100,), (100, 50)],
                'model__activation': ['relu', 'tanh'],
                'model__alpha': [0.0001, 0.001, 0.01]
            },
            'XGBoost': {
                'model__n_estimators': [50, 100, 200],
                'model__max_depth': [3, 6, 9],
                'model__learning_rate': [0.01, 0.1, 0.2],
                'model__subsample': [0.6, 0.8, 1.0],
                'model__colsample_bytree': [0.6, 0.8, 1.0]
            }
        }
        
        # Perform tuning for each model
        tuned_models = {}
        for model_name, param_grid in param_grids.items():
            print(f"\nTuning {model_name}...")
            
            # Create pipeline
            pipeline = Pipeline(steps=[
                ('preprocessor', self.preprocessor),
                ('model', self.models[model_name])
            ])
            
            # Perform grid search
            grid_search = GridSearchCV(
                pipeline,
                param_grid,
                cv=5,
                scoring='neg_mean_squared_error',
                n_jobs=-1
            )
            
            grid_search.fit(self.X_train, self.y_train)
            
            # Store best model
            tuned_models[model_name] = grid_search.best_estimator_
            
            print(f"Best parameters for {model_name}:")
            print(grid_search.best_params_)
            print(f"Best score: {np.sqrt(-grid_search.best_score_):.4f}")
        
        return tuned_models
    
    def train_tuned_models(self, tuned_models):
        """Train and evaluate tuned models"""
        print("\nTraining and evaluating tuned models...")
        
        # Store tuned model results
        self.tuned_results = {}
        
        for model_name, tuned_model in tuned_models.items():
            print(f"\nEvaluating tuned {model_name}...")
            
            # Train and evaluate the tuned model
            self.tuned_results[model_name] = self.train_and_evaluate_model(
                tuned_model.named_steps['model'],
                f"Tuned {model_name}",
                is_tuned=True
            )
            
            # Compare with default model
            default_metrics = self.results[model_name]['metrics']
            tuned_metrics = self.tuned_results[model_name]['metrics']
            
            print("\nImprovement over default model:")
            print(f"RMSE: {((default_metrics['rmse'] - tuned_metrics['rmse']) / default_metrics['rmse'] * 100):.2f}%")
            print(f"MAE: {((default_metrics['mae'] - tuned_metrics['mae']) / default_metrics['mae'] * 100):.2f}%")
            print(f"R²: {((tuned_metrics['r2'] - default_metrics['r2']) / default_metrics['r2'] * 100):.2f}%")
    
    def save_results(self):
        """Save model results to files"""
        # Save metrics to a file
        with open('MA-fa-2 code/model_results.txt', 'w') as f:
            f.write("Model Results\n")
            f.write("=" * 50 + "\n\n")
            
            # Save default model results
            f.write("Default Models:\n")
            f.write("-" * 30 + "\n")
            for model_name, result in self.results.items():
                f.write(f"Model: {model_name}\n")
                f.write("-" * 20 + "\n")
                f.write(f"Performance Metrics:\n")
                f.write(f"RMSE: {result['metrics']['rmse']:.4f}\n")
                f.write(f"MAE: {result['metrics']['mae']:.4f}\n")
                f.write(f"R² Score: {result['metrics']['r2']:.4f}\n\n")
            
            # Save tuned model results if available
            if hasattr(self, 'tuned_results'):
                f.write("\nTuned Models:\n")
                f.write("-" * 30 + "\n")
                for model_name, result in self.tuned_results.items():
                    f.write(f"Model: {model_name}\n")
                    f.write("-" * 20 + "\n")
                    f.write(f"Performance Metrics:\n")
                    f.write(f"RMSE: {result['metrics']['rmse']:.4f}\n")
                    f.write(f"MAE: {result['metrics']['mae']:.4f}\n")
                    f.write(f"R² Score: {result['metrics']['r2']:.4f}\n\n")
        
        # Save models
        for model_name, result in self.results.items():
            joblib.dump(
                result['model'],
                f'MA-fa-2 code/{model_name.lower().replace(" ", "_")}_model.joblib'
            )
        
        # Save tuned models if available
        if hasattr(self, 'tuned_results'):
            for model_name, result in self.tuned_results.items():
                joblib.dump(
                    result['model'],
                    f'MA-fa-2 code/tuned_{model_name.lower().replace(" ", "_")}_model.joblib'
                )

# Example usage:
if __name__ == "__main__":
    # Initialize the model class
    model = MumbaiHousePriceModel('Dataset cleaning/cleaned_mumbai_house_prices_30k.csv')
    
    # Load and prepare data
    model.load_and_prepare_data()
    
    # Train default models
    model.train_default_models()
    
    # Perform hyperparameter tuning and train tuned models
    tuned_models = model.hyperparameter_tuning()
    model.train_tuned_models(tuned_models)
    
    # Save results
    model.save_results() 