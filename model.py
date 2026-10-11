
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

# Load dataset
data = pd.read_csv("data.csv")

# Input features
features = [
    "area", "bedrooms", "bathrooms", "stories", "mainroad",
    "guestroom", "basement", "hotwaterheating",
    "airconditioning", "parking", "prefarea", "furnishingstatus"
]

# Output
target = "price"

X = data[features]
y = data[target]

# Separate numerical and categorical columns
numerical_features = [
    "area", "bedrooms", "bathrooms", "stories", "parking"
]

categorical_features = [
    "mainroad", "guestroom", "basement", "hotwaterheating",
    "airconditioning", "prefarea", "furnishingstatus"
]

# Preprocessing
preprocessor = ColumnTransformer([
    ("categorical", OneHotEncoder(handle_unknown="ignore"),
     categorical_features),
    ("numerical", "passthrough", numerical_features)
])

# Create model
model = Pipeline([
    ("preprocessor", preprocessor),
    ("regressor", RandomForestRegressor(
        n_estimators=200,
        random_state=42
    ))
])

# Evaluate when enough rows are available
if len(data) >= 5:
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    print("Model Evaluation")
    print("MAE:", mean_absolute_error(y_test, predictions))

    if y_test.nunique() > 1:
        print("R2 Score:", r2_score(y_test, predictions))
else:
    print("Not enough rows for reliable evaluation.")

# Train final model on all available data
model.fit(X, y)

# Save trained model
joblib.dump(model, "house_price_model.pkl")

print("House price model trained and saved successfully!")