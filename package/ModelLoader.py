import os
import logging

from dotenv import load_dotenv
import mlflow
import mlflow.pyfunc


logger = logging.getLogger(__name__)

_model = None


def load_model():
    """
    Load the MLflow model only once and cache it.
    """
    global _model

    if _model is not None:
        return _model

    load_dotenv()

    required_vars = [
        "S3_ACCESS_KEY",
        "S3_SECRET_KEY",
        "S3_ENDPOINT",
        "MLFLOW_TRACKING_URI",
    ]

    missing_vars = [var for var in required_vars if not os.getenv(var)]
    if missing_vars:
        raise RuntimeError(f"Missing environment variables: {', '.join(missing_vars)}")

    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))

    os.environ["MLFLOW_TRACKING_USERNAME"] = os.getenv("MLFLOW_TRACKING_USERNAME", "")
    os.environ["MLFLOW_TRACKING_PASSWORD"] = os.getenv("MLFLOW_TRACKING_PASSWORD", "")
    os.environ["AWS_ACCESS_KEY_ID"] = os.getenv("S3_ACCESS_KEY")
    os.environ["AWS_SECRET_ACCESS_KEY"] = os.getenv("S3_SECRET_KEY")
    os.environ["MLFLOW_S3_ENDPOINT_URL"] = os.getenv("S3_ENDPOINT")

    session_token = os.getenv("S3_SESSION_TOKEN")
    if session_token:
        os.environ["AWS_SESSION_TOKEN"] = session_token

    try:
        model_uri = "models:/price_predictor/latest"
        _model = mlflow.pyfunc.load_model(model_uri)
        logger.info("Model loaded successfully from MLflow")
        return _model
    except Exception as exc:
        logger.exception("Failed to load model")
        raise RuntimeError("Unable to load ML model") from exc
