from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pickle
import pandas as pd
from sklearn.linear_model import LinearRegression

# Initialize FastAPI app
app = FastAPI()

# Load the trained Linear Regression model from the pickle file
model_path = "/Users/rairamones/Desktop/sales_property/random_forest_model.pkl"
with open(model_path, "rb") as file:
    model = pickle.load(file)

# Define input data class
class InputData(BaseModel):
    LotArea: float
    YearBuilt: int
    FirstFlrSF: int
    SecondFlrSF: int
    FullBath: int
    BedroomAbvGr: int
    TotRmsAbvGrd: int

# Define endpoint for making predictions
@app.post("/predict")
async def predict(data: InputData):
    try:
        # Prepare data for prediction
        input_data = [[data.LotArea, data.YearBuilt, data.FirstFlrSF, data.SecondFlrSF,
                       data.FullBath, data.BedroomAbvGr, data.TotRmsAbvGrd]]
        # Make prediction
        prediction = model.predict(input_data)[0]
        return {"prediction": prediction}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
