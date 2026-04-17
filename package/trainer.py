import os
import json
import tempfile
from pathlib import Path

import pandas as pd
import mlflow
import mlflow.sklearn

from dotenv import load_dotenv

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import root_mean_squared_error, mean_absolute_error
from mlflow.models import infer_signature
import warnings

warnings.filterwarnings("ignore", message=".*Inferred schema contains integer column.*")


SEED = 42
DATA_PATH = (
    "https://minio.lab.sspcloud.fr/pnedjar/diffusion/data/airlines_flights_data.parquet"
)
CUSTOM_S3_PATH = "s3://pnedjar/diffusion/models/"
EXPERIMENT_NAME = "flight_ticket_predictor"
TARGET = "price"


def configure_tracking() -> str:
    """
    Configure environment variables for S3 artifact storage and MLflow tracking.
    Returns the experiment_id.
    """

    load_dotenv()

    os.environ["AWS_ACCESS_KEY_ID"] = os.getenv("S3_ACCESS_KEY", "")
    os.environ["AWS_SECRET_ACCESS_KEY"] = os.getenv("S3_SECRET_KEY", "")
    os.environ["MLFLOW_S3_ENDPOINT_URL"] = os.getenv("S3_ENDPOINT", "")

    session_token = os.getenv("S3_SESSION_TOKEN")
    if session_token:
        os.environ["AWS_SESSION_TOKEN"] = session_token

    tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
    if not tracking_uri:
        raise ValueError("MLFLOW_TRACKING_URI is missing in environment variables.")

    mlflow.set_tracking_uri(tracking_uri)

    exp = mlflow.get_experiment_by_name(EXPERIMENT_NAME)
    if exp is None:
        experiment_id = mlflow.create_experiment(
            name=EXPERIMENT_NAME,
            artifact_location=CUSTOM_S3_PATH,
        )
    else:
        experiment_id = exp.experiment_id

    return experiment_id


def load_data(path: str) -> pd.DataFrame:
    """
    Load raw dataset and apply basic type cleaning.
    """
    df = pd.read_parquet(path).set_index("index")

    df = df.drop(columns="flight").astype(
        {
            "class": "category",
            "stops": "category",
            "airline": "category",
        }
    )
    return df


def build_pipeline() -> Pipeline:
    """
    Build preprocessing + model pipeline.
    """
    categorical_features = [
        "airline",
        "source_city",
        "departure_time",
        "stops",
        "arrival_time",
        "destination_city",
        "class",
    ]
    numerical_features = ["duration", "days_left"]

    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
            ("num", MinMaxScaler(), numerical_features),
        ],
        remainder="passthrough",
    )

    model = GradientBoostingRegressor(random_state=SEED)

    pipeline = Pipeline(
        [
            ("preprocessor", preprocessor),
            ("model", model),
        ]
    )
    return pipeline


def build_search(pipeline: Pipeline) -> RandomizedSearchCV:
    """
    Build randomized hyperparameter search on the full pipeline.
    """
    param_distributions = {
        "model__n_estimators": [100, 200, 300, 500],
        "model__learning_rate": [0.01, 0.03, 0.05, 0.1, 0.2],
        "model__max_depth": [2, 3, 5, 7],
        "model__min_samples_split": [2, 5, 10],
        "model__min_samples_leaf": [1, 2, 4],
        "model__subsample": [0.8, 0.9, 1.0],
    }

    search = RandomizedSearchCV(
        estimator=pipeline,
        param_distributions=param_distributions,
        n_iter=20,
        scoring="neg_root_mean_squared_error",
        cv=5,
        refit=True,
        return_train_score=True,
        random_state=SEED,
        n_jobs=-1,
        verbose=2,
    )
    return search


def log_search_space(param_distributions: dict) -> str:
    """
    Save search space locally and return file path.
    """
    tmp_dir = tempfile.mkdtemp()
    file_path = Path(tmp_dir) / "search_space.json"

    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(param_distributions, f, indent=2)

    return str(file_path)


def train() -> None:
    """
    End-to-end training with MLflow tracking.
    """
    experiment_id = configure_tracking()
    print("Loading Data...")
    df = load_data(DATA_PATH)

    X = df.drop(columns=TARGET)
    y = df[TARGET]

    print("Splitting data...")
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=SEED,
    )

    print("Building pipeline and search...")
    pipeline = build_pipeline()
    search = build_search(pipeline)

    print("Logging search space...")
    with mlflow.start_run(
        experiment_id=experiment_id,
        run_name="gbr_randomized_search_prod",
    ):
        mlflow.sklearn.autolog(log_models=False)

        mlflow.set_tags(
            {
                "project": "flight_ticket_predictor",
                "model_family": "GradientBoostingRegressor",
                "training_type": "randomized_search_cv",
                "dataset": "airlines_flights_data",
                "target": TARGET,
                "framework": "scikit-learn",
            }
        )

        search.fit(X_train, y_train)
        best_model = search.best_estimator_
        y_pred = best_model.predict(X_test)

        test_rmse = root_mean_squared_error(y_test, y_pred)
        test_mae = mean_absolute_error(y_test, y_pred)

        mlflow.log_param("test_size", 0.2)
        mlflow.log_param("random_seed", SEED)
        mlflow.log_param("target_column", TARGET)
        mlflow.log_param("search_n_iter", search.n_iter)
        mlflow.log_param("search_cv", search.cv)
        mlflow.log_param("search_scoring", search.scoring)

        mlflow.log_metric("best_cv_rmse", -search.best_score_)
        mlflow.log_metric("test_rmse", test_rmse)
        mlflow.log_metric("test_mae", test_mae)

        mlflow.log_params(search.best_params_)

        cv_results = pd.DataFrame(search.cv_results_)
        with tempfile.TemporaryDirectory() as tmp_dir:
            cv_results_path = Path(tmp_dir) / "cv_results.csv"
            cv_results.to_csv(cv_results_path, index=False)
            mlflow.log_artifact(str(cv_results_path), artifact_path="reports")

            search_space_path = Path(tmp_dir) / "search_space.json"
            with open(search_space_path, "w", encoding="utf-8") as f:
                json.dump(search.param_distributions, f, indent=2)
            mlflow.log_artifact(str(search_space_path), artifact_path="reports")

        input_example = X_train.head(5)
        signature = infer_signature(X_train, best_model.predict(X_train))

        mlflow.sklearn.log_model(
            sk_model=best_model,
            artifact_path="model",
            signature=signature,
            input_example=input_example,
        )

        print("Best params:", search.best_params_)
        print("Best CV RMSE:", -search.best_score_)
        print("Test RMSE:", test_rmse)
        print("Test MAE:", test_mae)


if __name__ == "__main__":
    train()
