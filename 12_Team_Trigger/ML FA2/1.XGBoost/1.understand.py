import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

file_path = "C:\\Users\\Lenovo\\Documents\\ML FA2\\1.XGBoost\\dataset.csv"
df = pd.read_csv(file_path)

print("Dataset Shape:", df.shape)
print("\nFirst 5 rows:")
print(df.head())

print("\nData Info:")
print(df.info())

print("\nSummary Statistics:")
print(df.describe())

print("\nMissing Values:")
print(df.isnull().sum())

print("\nPercentage of Missing Values:")
print((df.isnull().sum() / len(df)) * 100)

print("\nNumber of Duplicate Rows:", df.duplicated().sum())

categorical_cols = df.select_dtypes(include=['object']).columns
print("\nUnique values in categorical columns:")
for col in categorical_cols:
    print(f"- {col}: {df[col].nunique()} unique values")

important_cats = ['Status', 'neworold', 'Furnished_status', 'type_of_building']
print("\nValue counts of key categorical variables:")
for col in important_cats:
    if col in df.columns:
        print(f"\n{col}:\n{df[col].value_counts(dropna=False)}")

df_cleaned = df.drop(columns=['Unnamed: 0', 'desc'], errors='ignore')
for col in ['Balcony', 'parking', 'Lift']:
    df_cleaned[col] = df_cleaned[col].fillna(df_cleaned[col].median())
for col in ['Furnished_status', 'Landmarks', 'Status']:
    df_cleaned[col] = df_cleaned[col].fillna(df_cleaned[col].mode()[0])

for col in df_cleaned.select_dtypes(include='object').columns:
    df_cleaned[col] = df_cleaned[col].astype('category').cat.codes

df_cleaned = df_cleaned.dropna()

plt.figure(figsize=(12, 8))
corr = df_cleaned.corr()
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f')
plt.title("Correlation Matrix")
plt.tight_layout()
plt.show()