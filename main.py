from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import numpy as np
import datetime

# Load the trained model (deserialized)
model_path = "notebooks\sales_model_09-23-2024-13-49-43.pkl" 
# Update this path with your saved model file
model = joblib.load(model_path)

# Define the input data schema for the API using Pydantic
class SalesPredictionInput(BaseModel):
    Store: int
    CompetitionDistance: float
    Promo: int
    DayOfWeek: int
    DaysToHoliday: int
    DaysAfterHoliday: int
    Year: int
    Month: int
    Day: int
    WeekOfYear: int
    IsWeekend: int
    IsBeginningOfMonth: int
    IsMidMonth: int
    IsEndOfMonth: int
    Quarter: int
    StoreType: str

# Create the FastAPI app
app = FastAPI()

# API root - a welcome message
@app.get("/")
def read_root():
    return {"message": "Welcome to the Sales Prediction API!"}

# Prediction endpoint
@app.post("/predict/")
def predict_sales(data: SalesPredictionInput):
    try:
        # Extract data as a dictionary and prepare it for prediction
        input_data = [
            data.CompetitionDistance, data.DaysToHoliday, data.DaysAfterHoliday, data.Year, 
            data.Month, data.Day, data.WeekOfYear, data.IsWeekend, data.IsBeginningOfMonth, 
            data.IsMidMonth, data.IsEndOfMonth, data.Quarter, data.StoreType, data.Promo, data.DayOfWeek
        ]

        # Prepare input as a 2D numpy array for model prediction
        input_data = np.array(input_data).reshape(1, -1)

        # Make prediction using the loaded model
        prediction = model.predict(input_data)

        # Return the prediction as JSON
        return {"predicted_sales": prediction[0]}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Run the application using Uvicorn server
# In terminal: `uvicorn filename:app --reload`
