from fastapi import FastAPI, HTTPException
import pandas as pd
import logging
import os
from contextlib import asynccontextmanager
from fastapi import Header

from .ModelLoader import load_model
from .FlightDataCollector import FlightDataCollector

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

model = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan function to handle startup and shutdown events.
    On startup, it loads the ML model and makes it available globally.
    On shutdown, it performs any necessary cleanup.
    """
    global model
    try:
        model = load_model()
        logger.info("Model loaded successfully")
    except Exception as e:
        logger.error(f"Model loading failed: {e}")
        raise RuntimeError(f"Cannot start API without model: {e}")

    yield
    logger.info("Shutting down API")


app = FastAPI(title="Flight Price API", lifespan=lifespan)


def build_features(data: dict, api_key: str | None) -> dict:
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

    if not api_key:
        raise HTTPException(
            status_code=400,
            detail="Missing Aviationstack API key. Provide it in 'x-aviation-key' header.",
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


@app.get("/")
def home():
    return {"status": "API running", "model_ready": model is not None}


@app.post("/predict")
def predict(data: dict, x_aviation_key: str | None = Header(default=None)):
    """
    Predict flight price based on input data.
    Accepts either full feature set or minimal input (source_city, destination_city, class).
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not available")

    logger.info(f"Received request: {data}")
    try:
        features = build_features(data, x_aviation_key)
        prediction = model.predict(pd.DataFrame([features]))[0]
        return {"price": float(prediction)}

    except HTTPException:
        raise
    except Exception as exc:
        logger.error(f"Prediction failed: {exc}")
        raise HTTPException(status_code=400, detail="Invalid input for prediction")


def _is_full_input(data: dict) -> bool:
    """
    Check if the input data contains all required features for direct prediction.
    """
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


if __name__ == "__main__":
    model = load_model()
    test_api_key = os.getenv("AVIATION_STACK_API_KEY")

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

    features_full = build_features(ex_data, api_key=None)
    pred_full = model.predict(pd.DataFrame([features_full]))[0]

    print("FULL INPUT PREDICTION:", float(pred_full))

    minimal_data = {
        "source_city": "Delhi",
        "destination_city": "Mumbai",
        "class": "Economy",
    }

    try:
        features_min = build_features(minimal_data, api_key=test_api_key)
        pred_min = model.predict(pd.DataFrame([features_min]))[0]

        print("MINIMAL INPUT PREDICTION:", float(pred_min))

    except Exception as e:
        print("MINIMAL INPUT FAILED:", e)
