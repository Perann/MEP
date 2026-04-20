from fastapi import FastAPI
import pandas as pd
import os
from dotenv import load_dotenv
import logging
import mlflow.pyfunc
from .ModelLoader import load_model


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


app = FastAPI(title="Flight Price API")
model = None


@app.get("/")
def home():
    return {"status": "API running", "model_ready": model is not None}


@app.post("/predict")
def predict(data: dict):
    load_model()
    if model is None:
        return {"error": "Model not loaded"}

    logger.info(f"Received request: {data}")
    df = pd.DataFrame([data])

    prediction = model.predict(df)[0]
    logger.info(f"Prediction: {prediction}")
    return {"price": float(prediction)}


if __name__ == "__main__":
    ex_data = {
        "airline": "SpiceJet",
        "source_city": "Delhi",
        "departure_time": "Evening",
        "stops": "zero",
        "arrival_time": "Night",
        "destination_city": "Mumbai",
        "class": "Economy",
        "duration": 2.17,
        "days_left": 1,
    }
    predict(ex_data)
