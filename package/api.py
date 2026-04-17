from fastapi import FastAPI
import pandas as pd
import os
from dotenv import load_dotenv
import logging
import mlflow.pyfunc


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


app = FastAPI(title="Flight Price API")
model = None


def load_model():
    global model
    load_dotenv()

    required_vars = [
        "S3_ACCESS_KEY",
        "S3_SECRET_KEY",
        "S3_ENDPOINT",
        "MLFLOW_TRACKING_URI",
    ]
    if any(not os.getenv(var) for var in required_vars):
        raise RuntimeError("Missing environment variables")

    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))

    os.environ["MLFLOW_TRACKING_USERNAME"] = os.getenv("MLFLOW_TRACKING_USERNAME", "")
    os.environ["MLFLOW_TRACKING_PASSWORD"] = os.getenv("MLFLOW_TRACKING_PASSWORD", "")
    os.environ["AWS_ACCESS_KEY_ID"] = os.getenv("S3_ACCESS_KEY")
    os.environ["AWS_SECRET_ACCESS_KEY"] = os.getenv("S3_SECRET_KEY")
    os.environ["MLFLOW_S3_ENDPOINT_URL"] = os.getenv("S3_ENDPOINT")

    session_token = os.getenv("S3_SESSION_TOKEN")
    if session_token:
        os.environ["AWS_SESSION_TOKEN"] = session_token

    if model is None:
        try:
            model_uri = "models:/price_predictor/latest"
            model = mlflow.pyfunc.load_model(model_uri)
            logger.info("Model downloaded successfuly")
        except Exception as e:
            logger.error(f"Erreur : {e}")
            raise e


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
