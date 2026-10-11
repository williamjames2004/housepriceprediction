
import pandas as pd
import joblib

# Load trained model
model = joblib.load("house_price_model.pkl")

# Get house details
area = float(input("Area (sq. ft.): "))
bedrooms = int(input("Bedrooms: "))
bathrooms = int(input("Bathrooms: "))
stories = int(input("Stories: "))

mainroad = input("Main Road (yes/no): ").strip().lower()
guestroom = input("Guest Room (yes/no): ").strip().lower()
basement = input("Basement (yes/no): ").strip().lower()
hotwaterheating = input("Hot Water Heating (yes/no): ").strip().lower()
airconditioning = input("Air Conditioning (yes/no): ").strip().lower()

parking = int(input("Parking Spaces: "))
prefarea = input("Preferred Area (yes/no): ").strip().lower()

furnishingstatus = input(
    "Furnishing Status (furnished/semi-furnished/unfurnished): "
).strip().lower()

# Prepare input data
house = pd.DataFrame([{
    "area": area,
    "bedrooms": bedrooms,
    "bathrooms": bathrooms,
    "stories": stories,
    "mainroad": mainroad,
    "guestroom": guestroom,
    "basement": basement,
    "hotwaterheating": hotwaterheating,
    "airconditioning": airconditioning,
    "parking": parking,
    "prefarea": prefarea,
    "furnishingstatus": furnishingstatus
}])

# Predict price
predicted_price = model.predict(house)[0]

print("\nPredicted House Price:", round(predicted_price, 2))