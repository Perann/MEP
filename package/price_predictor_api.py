from fastapi import FastAPI, HTTPException
import pandas as pd
import logging
import os
from dotenv import load_dotenv

from .ModelLoader import load_model

from .FlightDataCollector import FlightDataCollector

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Flight Price API")
model = None


def _is_full_input(data: dict) -> bool:
    required_fields = [
        "airline",
        "source_city",
        "departure_time",
        "stops",
        "arrival_time",
        "destination_city",
        "class",
        "duration",
        "days_left",
    ]
    return all(data.get(field) is not None for field in required_fields)


def build_features(data: dict) -> dict:
    """
    Build the final feature payload for the model.
    If input is partial, enrich it using AviationStack.
    """
    if _is_full_input(data):
        return {
            "airline": data["airline"],
            "source_city": data["source_city"],
            "departure_time": data["departure_time"],
            "stops": data["stops"],
            "arrival_time": data["arrival_time"],
            "destination_city": data["destination_city"],
            "class": data["class"],
            "duration": data["duration"],
            "days_left": data["days_left"],
        }

    minimal_fields = ["source_city", "destination_city", "class"]
    minimal_input_ok = all(data.get(field) is not None for field in minimal_fields)

    if not minimal_input_ok:
        raise HTTPException(
            status_code=422,
            detail=(
                "You must provide either all model features or only "
                "'source_city', 'destination_city', and 'class'."
            ),
        )

    load_dotenv()
    api_key = os.getenv("AVIATION_STACK_API_KEY")
    if not api_key:
        raise HTTPException(
            status_code=500,
            detail="AVIATION_STACK_API_KEY is missing from environment variables",
        )

    collector = FlightDataCollector(api_key)
    enriched_data = collector.fetch_and_format(
        source_city=data["source_city"],
        destination_city=data["destination_city"],
        ticket_class=data["class"],
    )

    if "error" in enriched_data:
        raise HTTPException(status_code=400, detail=enriched_data["error"])

    return enriched_data


@app.on_event("startup")
def startup_event():
    global model
    try:
        model = load_model()
        logger.info("Model loaded successfully")
    except Exception as e:
        logger.error(f"Model loading failed: {e}")
        model = None


@app.get("/")
def home():
    return {"status": "API running", "model_ready": model is not None}


@app.post("/predict")
def predict(data: dict):
    if model is None:  # ← on utilise le global, sans recharger
        raise HTTPException(status_code=503, detail="Model not available")

    logger.info(f"Received request: {data}")
    try:
        features = build_features(data)
        logger.info(f"Features: {features}")
        prediction = model.predict(pd.DataFrame([features]))[0]
        logger.info(f"Prediction: {prediction}")
        return {"price": float(prediction)}
    except HTTPException:
        raise
    except Exception as exc:
        logger.error(f"Prediction failed: {exc}")
        raise HTTPException(status_code=400, detail="Invalid input for prediction")


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

    minimal_data = {
        "source_city": "Delhi",
        "destination_city": "Mumbai",
        "class": "Economy",
    }
    predict(minimal_data)
