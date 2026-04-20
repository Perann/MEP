✈️ Plane Ticket Price Prediction

📌 Project Description

The goal of this project is to build a machine learning model that predicts the price of airplane tickets based on multiple flight characteristics.

The API offers two ways to use the model:

🧾 1. Full data input

You can directly provide all flight details (airline, cities, times, duration, etc.).
The model will return an estimated ticket price.

⚡ 2. Minimal input (API enrichment)

This mode requires an AviationStack API key.

You only provide:

source city
destination city
travel class (Economy / Premium)

Optionally, a date (depending on your AviationStack plan).

The API will then:

Fetch flight information from AviationStack
Enrich the input features
Predict the ticket price
📡 API Endpoint
POST /predict
🧾 Request Format

You can provide either a full feature set or a minimal input.

✔️ Full input example
{
  "airline": "SpiceJet",
  "source_city": "Delhi",
  "departure_time": "Evening",
  "stops": "zero",
  "arrival_time": "Night",
  "destination_city": "Mumbai",
  "class": "Economy",
  "duration": 2.17,
  "days_left": 1
}
✔️ Minimal input example
{
  "source_city": "Delhi",
  "destination_city": "Mumbai",
  "class": "Economy"
}

In this case, the API automatically enriches the data using external flight information (requires an AviationStack API key).

🔐 Header (required for minimal input)
x-aviation-key: YOUR_API_KEY
📤 Response
{
  "price": 12345.67
}
📊 Dataset

The model was trained using the following public dataset:

Flight Price Prediction Dataset (Kaggle)

🗂 Project Structure

The main scripts are located in the package folder:

FlightDataCollector.py → Handles requests to AviationStack API when needed
ModelLoader.py → Loads the trained model from MLflow
price_predictor_api.py → FastAPI application
trainer.py → Model training and MLflow tracking

📁 notebooks/ contains exploratory analysis and research experiments.

🛠 How to Contribute

This project uses Pixi
, a modern package and workflow manager. It ensures a reproducible environment across Linux, macOS, and Windows.

1. Prerequisites

Install Pixi CLI:

curl -fsSL https://pixi.sh/install.sh | bash
2. Environment setup
git clone <your-repo-url>
cd <your-project-name>
pixi install
3. Run the project

You can run commands inside the managed environment:

pixi run python src/train.py
4. Code quality (pre-commit)

We use Black for formatting.

Install pre-commit hooks:

pixi run pre-commit install