import joblib
import numpy as np
import pandas as pd

model = joblib.load('best_xgboost_model.pkl')

feature_names = [
    'area',
    'latitude',
    'longitude',
    'Bedrooms',
    'Bathrooms',
    'Balcony',
    'Status',
    'neworold',
    'parking',
    'Furnished_status',
    'Lift',
    'type_of_building'
]

print("Enter property details:\n")
input_values = []

input_values.append(float(input("Area (in sq ft): ")))
input_values.append(float(input("Latitude: ")))
input_values.append(float(input("Longitude: ")))
input_values.append(int(input("Number of Bedrooms: ")))
input_values.append(int(input("Number of Bathrooms: ")))
input_values.append(int(input("Number of Balconies: ")))
print("Status → 1 = Under Construction, 0 = Ready to Move")
input_values.append(int(input("Enter Status: ")))
print("Property Age → 1 = Resale, 0 = New Property")
input_values.append(int(input("Enter New or Old: ")))
input_values.append(int(input("Number of Parking Spaces: ")))
print("Furnishing Status → 1 = Semi Furnished, 2 = Unfurnished")
input_values.append(int(input("Enter Furnished Status: ")))
input_values.append(int(input("Lift Available? (1 = Yes, 0 = No): ")))
print("Type of Building → 0 = Flat, 1 = Individual")
input_values.append(int(input("Enter Type of Building: ")))

input_array = np.array(input_values).reshape(1, -1)
input_df = pd.DataFrame(input_array, columns=feature_names)

log_price_pred = model.predict(input_df)[0]
price_pred = np.expm1(log_price_pred)

print(f"\nPredicted Property Price: ₹{price_pred:,.2f}")
