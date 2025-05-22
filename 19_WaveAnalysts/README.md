# Airbnb Price Prediction Using Machine Learning and Sentiment Analysis

**Authors**:  
 Yashraj Haridas(yashraj.haridas22@pccoepune.org)
Hrushikesh Halge(hrushikesh.halge22@pccoepune.org)
Onkar Karande(onkar.karande22@pccoepune.org)
**Citation**:  
This project is based on the paper: [https://arxiv.org/abs/1907.12665](https://arxiv.org/abs/1907.12665)

---

## Overview

This repository contains code for predicting Airbnb listing prices in New York City using machine learning techniques and sentiment analysis of user reviews. The project involves data preprocessing, feature selection, and training various models to evaluate their performance.

---

## Prerequisites

Ensure the following Python packages are installed:

- `TextBlob`
- `scikit-learn`
- `numpy`
- `pandas`

You can install the required packages using:

```bash
pip install -r requirements.txt
```

---

## Dataset

1. Create a directory named `Data` in the project root.
2. Download the datasets from the following Google Drive link and place them into the `Data` directory:  
   [Dataset Link](https://drive.google.com/drive/folders/1xk5RyR-UgF6M-ddhn11SXHEWJeB0fQo5?usp=sharing)

---

## Data Preprocessing

1. **Sentiment Analysis**:  
   Analyze the sentiment of reviews and generate a file with sentiment scores:

   ```bash
   python sentiment_analysis.py
   ```

2. **Data Cleaning**:  
   Clean the dataset to handle missing values and inconsistencies:

   ```bash
   python data_cleanup.py
   ```

3. **Normalization and Splitting**:  
   Normalize the data and split it into training, validation, and test sets:

   ```bash
   python data_preprocessing_reviews.py
   ```

---

## Feature Selection

1. **P-Value Feature Selection**:  
   Select features based on p-values:

   ```bash
   python feature_selection.py
   ```

2. **Lasso Cross-Validation**:  
   Perform Lasso regression with cross-validation for feature selection:

   ```bash
   python cv.py
   ```

---

## Model Training and Evaluation

Train and evaluate the models using the prepared data:

```bash
python run_models.py
```

**Configuration Options**:

- **Lasso CV Feature Selection**:

  - Comment out line 240:
    ```python
    # coeffs = np.load('../Data/selected_coefs_pvals.npy')
    ```
  - Uncomment line 241:
    ```python
    coeffs = np.load('../Data/selected_coefs.npy')
    ```

- **P-Value Feature Selection**:

  - Uncomment line 240:
    ```python
    coeffs = np.load('../Data/selected_coefs_pvals.npy')
    ```
  - Comment out line 241:
    ```python
    # coeffs = np.load('../Data/selected_coefs.npy')
    ```

- **Baseline Linear Regression**:
  - Uncomment lines 277 and 278:
    ```python
    print("--------------------Linear Regression--------------------")
    LinearModel(X_concat, y_concat, X_test, y_test)
    ```
  - Comment out all lines below these.
  - Also, comment out lines 268, 269, and 270:
    ```python
    # X_train = X_train[list(col_set)]
    # X_val = X_val[list(col_set)]
    # X_test = X_test[list(col_set)]
    ```

**Note**: Some models may take a considerable amount of time to train and evaluate.

---

## License

This project is licensed under the MIT License.

---

For more details, please refer to the [original repository](https://github.com/PouyaREZ/AirBnbPricePrediction).
