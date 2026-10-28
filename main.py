import joblib
import pandas as pd

from pathlib import Path
from fastapi import FastAPI
from pydantic import BaseModel, Field

# Load the trained model
file_path = Path(__file__).with_name("house_price_model.joblib")
saved_data = joblib.load(file_path)

model = saved_data["model"]
features = saved_data["features"]

# Create the API
app = FastAPI(title="House Price Prediction API")


# Input data
class HouseInput(BaseModel):
    area: float = Field(gt=0)
    bedrooms: int = Field(gt=0)
    bathrooms: int = Field(gt=0)
    stories: int = Field(gt=0)
    parking: int = Field(ge=0)


# Home page
@app.get("/")
def home():
    return {
        "message": "House Price Prediction API is running!"
    }


# Health check
@app.get("/health")
def health():
    return {"status": "ok"}


# Predict house price
@app.post("/predict")
def predict(data: HouseInput):
    house = pd.DataFrame(
        [[
            data.area,
            data.bedrooms,
            data.bathrooms,
            data.stories,
            data.parking
        ]],
        columns=features
    )

    price = float(model.predict(house)[0])

    return {
        "predicted_price": round(price, 2)
    }
