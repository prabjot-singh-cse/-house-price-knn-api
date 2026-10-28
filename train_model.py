import pandas as pd
import joblib

from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error

# Load Kaggle dataset
file_path = Path(__file__).with_name("Housing.csv")
data = pd.read_csv(file_path)

# Select features and target
features = ["area", "bedrooms", "bathrooms", "stories", "parking"]

X = data[features]
y = data["price"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Create KNN model
model = Pipeline([
    ("scaler", StandardScaler()),
    ("knn", KNeighborsRegressor(n_neighbors=5))
])

# Train model
model.fit(X_train, y_train)

# Evaluate model
predictions = model.predict(X_test)
mae = mean_absolute_error(y_test, predictions)

print("Dataset rows:", len(data))
print("Mean Absolute Error:", round(mae, 2))

# Save model and feature names
model_data = {
    "model": model,
    "features": features
}

model_path = Path(__file__).with_name("house_price_model.joblib")
joblib.dump(model_data, model_path)

print("Model trained and saved successfully!")

# Test one prediction
new_house = pd.DataFrame(
    [[5000, 3, 2, 2, 1]],
    columns=features
)

price = model.predict(new_house)[0]

print("Predicted house price:", round(price, 2))
