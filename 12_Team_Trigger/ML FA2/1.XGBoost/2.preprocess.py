import pandas as pd

file_path = "dataset.csv"
df = pd.read_csv(file_path)

df = df.drop(columns=['Unnamed: 0', 'desc', 'Price_sqft', 'Landmarks' ,'Address'], errors='ignore')

for col in ['Balcony', 'parking', 'Lift']:
    if col in df.columns:
        df[col] = df[col].fillna(df[col].median())

if 'Furnished_status' in df.columns:
    df['Furnished_status'] = df['Furnished_status'].fillna(df['Furnished_status'].mode()[0])

# if 'Furnished_status' in df.columns:
#     df['Furnished_status'] = df['Furnished_status'].fillna('Unk')

if 'Status' in df.columns:
    df['Status'] = df['Status'].fillna(df['Status'].mode()[0])

for col in df.select_dtypes(include='object').columns:
    df[col] = df[col].astype('category').cat.codes

df = df.apply(pd.to_numeric, errors='ignore')

print(df.info())
print(df.head())

df = df.dropna()

df = df.drop_duplicates()

df.to_csv("Cleaned_Dataset.csv", index=False)

print("Preprocessing complete. Cleaned data saved as 'Cleaned_Housing_Data.csv'")
print(df.info())
print(df.head())

#Encoding to Actual
# Status = {Under_Construction: 1 , Ready to Move: 0}
# neworold:{Resale:1 , New PRop:0}
# Furnished Status: {Semi Furnished:1 , unfurnished: 2}
# Type of building: { Flat: 0 ,Indivisula: 1}
