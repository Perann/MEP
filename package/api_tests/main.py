from fastapi import FastAPI
import pandas as pd
import os
from dotenv import load_dotenv
import boto3
import io
import joblib
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv(override=True)

BUCKET = "pnedjar"
KEY = "diffusion/models/price_predictor.joblib"


AWS_ACCESS_KEY = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
AWS_TOKEN = os.getenv("AWS_SESSION_TOKEN")
S3_ENDPOINT = os.getenv("S3_ENDPOINT") or os.getenv("AWS_S3_ENDPOINT")


app = FastAPI(title="Flight Price API")

model = None  # lazy loading



def load_model():
    global model

    if model is not None:
        logger.info("Model already loaded")
        return

    logger.info("Loading model from S3")

    if AWS_ACCESS_KEY is None or AWS_SECRET_KEY is None:
        raise Exception(
            "Missing AWS credentials. Check AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY"
        )
    logger.info(f"Connecting to S3 endpoint: {S3_ENDPOINT}")
    s3 = boto3.client(
        "s3",
        endpoint_url=S3_ENDPOINT,
        aws_access_key_id=AWS_ACCESS_KEY,
        aws_secret_access_key=AWS_SECRET_KEY,
        aws_session_token=AWS_TOKEN,
    )
    logger.info(f"⬇downloading model: s3://{BUCKET}/{KEY}")
    response = s3.get_object(Bucket=BUCKET, Key=KEY)
    model_file = io.BytesIO(response["Body"].read())
    logger.info("Model downloaded, loading with joblib")
    model = joblib.load(model_file)
    
    logger.info("Model loaded")

print(type(model))

@app.get("/")
def home():
    return {"status": "API running "}


@app.post("/predict")
def predict(data: dict):
    logger.info(f"Received request: {data}")
    load_model()

    df = pd.DataFrame([data])

    prediction = model.predict(df)[0]
    logger.info(f"Prediction: {prediction}")

    return {
        "price": float(prediction)
    }






