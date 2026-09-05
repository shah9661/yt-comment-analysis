import os
import pickle
import pytest
import pandas as pd
import mlflow
from dotenv import load_dotenv
from mlflow.tracking import MlflowClient
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score

load_dotenv()
DAGSHUB_TOKEN = os.getenv("DAGSHUB_TOKEN")
if not DAGSHUB_TOKEN:
    raise EnvironmentError(
        "DAGSHUB_TOKEN environment variable is not set"
    )

REPO_OWNER = "shanshad0999"
REPO_NAME = "yt-comment-analysis"
MODEL_NAME = "my_model"
MODEL_ALIAS = "staging"

HOLDOUT_DATA_PATH = "data/interim/test_processed.csv"
VECTORIZER_ARTIFACT_PATH = "tfidf_vectorizer.pkl"

os.environ["MLFLOW_TRACKING_USERNAME"] = REPO_OWNER
os.environ["MLFLOW_TRACKING_PASSWORD"] = DAGSHUB_TOKEN

MLFLOW_TRACKING_URI = (f"https://dagshub.com/{REPO_OWNER}/{REPO_NAME}.mlflow")
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

def get_model_info(model_name: str,alias: str = MODEL_ALIAS) -> dict:
    try:
        client = MlflowClient()
        model_version = client.get_model_version_by_alias(model_name,alias)
        return {
            "model_name": model_name,
            "model_version": model_version.version,
            "run_id": model_version.run_id,
            "model_uri": f"models:/{model_name}@{alias}",
        }
    except Exception as e:
        raise RuntimeError(
            f"Failed to retrieve model info "
            f"for '{model_name}' with alias '{alias}': {e}"
        ) from e
def load_model(model_uri: str):
    try:
        return mlflow.pyfunc.load_model(model_uri)
    except Exception as e:
        raise RuntimeError(
            f"Failed to load MLflow model "
            f"from '{model_uri}': {e}"
        ) from e

def load_vectorizer(model_info: dict):
    try:
        run_id = model_info["run_id"]
        vectorizer_path = mlflow.artifacts.download_artifacts(
            run_id=run_id,
            artifact_path=VECTORIZER_ARTIFACT_PATH
        )
        with open(vectorizer_path, "rb") as file:
            vectorizer = pickle.load(file)
        return vectorizer
    except Exception as e:
        raise RuntimeError(
            f"Failed to load TF-IDF vectorizer "
            f"for run '{model_info.get('run_id')}': {e}"
        ) from e

def load_production_resources():
    model_info = get_model_info(model_name=MODEL_NAME,alias=MODEL_ALIAS)
    model = load_model(model_uri=model_info["model_uri"])
    vectorizer = load_vectorizer(model_info=model_info)
    return model, vectorizer, model_info

def test_model_performance():
    model, vectorizer, model_info = (load_production_resources())
    holdout_data = pd.read_csv(HOLDOUT_DATA_PATH)
    assert not holdout_data.empty, ("Holdout test dataset is empty")
    X_holdout_raw = (holdout_data.iloc[:, 0].fillna(""))
    y_holdout = holdout_data.iloc[:, -1]

    X_holdout_tfidf = vectorizer.transform(X_holdout_raw)

    y_pred = model.predict(X_holdout_tfidf)
    accuracy = accuracy_score(y_holdout,y_pred)
    precision = precision_score(
        y_holdout,
        y_pred,
        average="weighted",
        zero_division=1
    )
    recall = recall_score(
        y_holdout,
        y_pred,
        average="weighted",
        zero_division=1
    )

    f1 = f1_score(
        y_holdout,
        y_pred,
        average="weighted",
        zero_division=1
    )
    print("\n======================================")
    print("       MODEL PERFORMANCE")
    print("======================================")
    print(f"Model Name : {model_info['model_name']}")
    print(f"Version    : {model_info['model_version']}")
    print(f"Alias      : {MODEL_ALIAS}")
    print(f"Accuracy   : {accuracy:.4f}")
    print(f"Precision  : {precision:.4f}")
    print(f"Recall     : {recall:.4f}")
    print(f"F1 Score   : {f1:.4f}")
    print("======================================\n")

    expected_accuracy = 0.40
    expected_precision = 0.40
    expected_recall = 0.40
    expected_f1 = 0.40

    assert accuracy >= expected_accuracy, (
        f"Accuracy should be at least "
        f"{expected_accuracy}, got {accuracy:.4f}"
    )

    assert precision >= expected_precision, (
        f"Precision should be at least "
        f"{expected_precision}, got {precision:.4f}"
    )

    assert recall >= expected_recall, (
        f"Recall should be at least "
        f"{expected_recall}, got {recall:.4f}"
    )

    assert f1 >= expected_f1, (
        f"F1 score should be at least "
        f"{expected_f1}, got {f1:.4f}"
    )

    print(
        f"Performance test PASSED for "
        f"model '{MODEL_NAME}' "
        f"version {model_info['model_version']}."
    )