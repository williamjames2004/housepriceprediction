from pathlib import Path

import joblib
import pandas as pd
from flask import Flask, render_template, request

app = Flask(__name__)

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "house_price_model.pkl"

FEATURES = [
    "area", "bedrooms", "bathrooms", "stories", "mainroad",
    "guestroom", "basement", "hotwaterheating",
    "airconditioning", "parking", "prefarea", "furnishingstatus"
]

CATEGORICAL_OPTIONS = {
    "mainroad": ["yes", "no"],
    "guestroom": ["yes", "no"],
    "basement": ["yes", "no"],
    "hotwaterheating": ["yes", "no"],
    "airconditioning": ["yes", "no"],
    "prefarea": ["yes", "no"],
    "furnishingstatus": [
        "furnished", "semi-furnished", "unfurnished"
    ]
}

model = None

if MODEL_PATH.exists():
    model = joblib.load(MODEL_PATH)


@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    error = None

    values = {
        "area": "7420",
        "bedrooms": "4",
        "bathrooms": "2",
        "stories": "3",
        "mainroad": "yes",
        "guestroom": "no",
        "basement": "no",
        "hotwaterheating": "no",
        "airconditioning": "yes",
        "parking": "2",
        "prefarea": "yes",
        "furnishingstatus": "furnished"
    }

    if request.method == "POST":
        values = {
            feature: request.form.get(feature, "").strip().lower()
            for feature in FEATURES
        }

        try:
            if model is None:
                raise FileNotFoundError(
                    "Model file not found. Run python model.py first."
                )

            area = float(values["area"])
            bedrooms = int(values["bedrooms"])
            bathrooms = int(values["bathrooms"])
            stories = int(values["stories"])
            parking = int(values["parking"])

            if area <= 0:
                raise ValueError("Area must be greater than zero.")

            if not 1 <= bedrooms <= 20:
                raise ValueError("Bedrooms must be between 1 and 20.")

            if not 1 <= bathrooms <= 20:
                raise ValueError("Bathrooms must be between 1 and 20.")

            if not 1 <= stories <= 20:
                raise ValueError("Stories must be between 1 and 20.")

            if not 0 <= parking <= 20:
                raise ValueError("Parking spaces must be between 0 and 20.")

            for field, options in CATEGORICAL_OPTIONS.items():
                if values[field] not in options:
                    raise ValueError(
                        f"Please select a valid option for {field}."
                    )

            house = pd.DataFrame([{
                "area": area,
                "bedrooms": bedrooms,
                "bathrooms": bathrooms,
                "stories": stories,
                "mainroad": values["mainroad"],
                "guestroom": values["guestroom"],
                "basement": values["basement"],
                "hotwaterheating": values["hotwaterheating"],
                "airconditioning": values["airconditioning"],
                "parking": parking,
                "prefarea": values["prefarea"],
                "furnishingstatus": values["furnishingstatus"]
            }], columns=FEATURES)

            prediction = float(model.predict(house)[0])

            if prediction < 0:
                raise ValueError(
                    "The model returned an invalid price. Try different inputs."
                )

        except FileNotFoundError as exc:
            error = str(exc)

        except (ValueError, TypeError, OverflowError) as exc:
            error = str(exc) or "Please enter valid property details."

        except Exception:
            app.logger.exception("House price prediction failed.")
            error = (
                "Prediction could not be completed. "
                "Check your model and dataset."
            )

    return render_template(
        "index.html",
        prediction=prediction,
        error=error,
        values=values
    )


if __name__ == "__main__":
    app.run(debug=True)