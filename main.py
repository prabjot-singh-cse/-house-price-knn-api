
import joblib
import pandas as pd

from pathlib import Path
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

# Load the trained model
file_path = Path(__file__).with_name("house_price_model.joblib")
saved_data = joblib.load(file_path)

model = saved_data["model"]
features = saved_data["features"]

app = FastAPI(title="House Price Prediction API")


class HouseInput(BaseModel):
    area: float = Field(gt=0)
    bedrooms: int = Field(gt=0)
    bathrooms: int = Field(gt=0)
    stories: int = Field(gt=0)
    parking: int = Field(ge=0)


# Website frontend
@app.get("/", response_class=HTMLResponse)
def home():
    return """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>House Price Predictor</title>

        <style>
            * {
                box-sizing: border-box;
            }

            body {
                margin: 0;
                font-family: Arial, sans-serif;
                background: #f1f5f9;
                color: #1e293b;
            }

            .container {
                max-width: 520px;
                margin: 50px auto;
                padding: 30px;
                background: white;
                border-radius: 16px;
                box-shadow: 0 8px 30px #00000012;
            }

            h1 {
                text-align: center;
                margin-bottom: 10px;
            }

            .subtitle {
                text-align: center;
                color: #64748b;
                margin-bottom: 25px;
            }

            label {
                display: block;
                margin-top: 16px;
                margin-bottom: 6px;
                font-weight: bold;
            }

            input {
                width: 100%;
                padding: 12px;
                border: 1px solid #cbd5e1;
                border-radius: 8px;
                font-size: 16px;
            }

            button {
                width: 100%;
                padding: 14px;
                margin-top: 24px;
                border: none;
                border-radius: 8px;
                background: #2563eb;
                color: white;
                font-size: 16px;
                font-weight: bold;
                cursor: pointer;
            }

            button:hover {
                background: #1d4ed8;
            }

            #result {
                margin-top: 22px;
                padding: 16px;
                border-radius: 8px;
                background: #eff6ff;
                text-align: center;
                overflow-wrap: anywhere;
            }

            .hidden {
                display: none;
            }

            .note {
                font-size: 12px;
                color: #64748b;
                margin-top: 18px;
                line-height: 1.5;
            }
        </style>
    </head>

    <body>
        <div class="container">
            <h1>House Price Predictor</h1>

            <p class="subtitle">
                Enter your house details to estimate its price using KNN.
            </p>

            <form id="predictionForm">
                <label for="area">Area (sq. ft.)</label>
                <input id="area" type="number" min="0.01"
                       step="any" required placeholder="e.g. 5000">

                <label for="bedrooms">Bedrooms</label>
                <input id="bedrooms" type="number" min="1"
                       step="1" required placeholder="e.g. 3">

                <label for="bathrooms">Bathrooms</label>
                <input id="bathrooms" type="number" min="1"
                       step="1" required placeholder="e.g. 2">

                <label for="stories">Stories</label>
                <input id="stories" type="number" min="1"
                       step="1" required placeholder="e.g. 2">

                <label for="parking">Parking spaces</label>
                <input id="parking" type="number" min="0"
                       step="1" required placeholder="e.g. 1">

                <button id="submitButton" type="submit">
                    Predict House Price
                </button>
            </form>

            <div id="result" class="hidden" aria-live="polite"></div>

            <p class="note">
                This is an educational ML project using a Kaggle dataset.
                Predictions are estimates in the dataset's original price
                units and are not professional property valuations.
            </p>
        </div>

        <script>
            const form = document.getElementById("predictionForm");
            const result = document.getElementById("result");
            const button = document.getElementById("submitButton");

            form.addEventListener("submit", async function(event) {
                event.preventDefault();

                result.classList.remove("hidden");
                result.textContent = "Predicting...";
                button.disabled = true;

                const house = {
                    area: Number(document.getElementById("area").value),
                    bedrooms: Number(document.getElementById("bedrooms").value),
                    bathrooms: Number(document.getElementById("bathrooms").value),
                    stories: Number(document.getElementById("stories").value),
                    parking: Number(document.getElementById("parking").value)
                };

                try {
                    const response = await fetch("/predict", {
                        method: "POST",
                        headers: {
                            "Content-Type": "application/json"
                        },
                        body: JSON.stringify(house)
                    });

                    const data = await response.json();

                    if (!response.ok) {
                        throw new Error("Please check the values and try again.");
                    }

                    const price = new Intl.NumberFormat(
                        "en-US",
                        { maximumFractionDigits: 2 }
                    ).format(data.predicted_price);

                    result.textContent = "Estimated price: " + price;

                } catch (error) {
                    result.textContent = "Error: " + error.message;
                } finally {
                    button.disabled = false;
                }
            });
        </script>
    </body>
    </html>
    """


@app.get("/health")
def health():
    return {"status": "ok"}


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

    return {"predicted_price": round(price, 2)}
