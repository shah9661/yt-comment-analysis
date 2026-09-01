import os
import pickle
import mlflow
from dotenv import load_dotenv

load_dotenv()

DAGSHUB_TOKEN = os.getenv("DAGSHUB_TOKEN")

if not DAGSHUB_TOKEN:
    raise EnvironmentError(
        "DAGSHUB_TOKEN environment variable is not set"
    )

REPO_OWNER = "shanshad0999"
REPO_NAME = "yt-comment-analysis"

MODEL_NAME = "my_model"
MODEL_ALIAS = "champion"

os.environ["MLFLOW_TRACKING_USERNAME"] = REPO_OWNER
os.environ["MLFLOW_TRACKING_PASSWORD"] = DAGSHUB_TOKEN

MLFLOW_TRACKING_URI = (
    f"https://dagshub.com/{REPO_OWNER}/{REPO_NAME}.mlflow"
)

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)


def get_model_info(model_name: str,alias: str = MODEL_ALIAS) -> dict:
    try:
        client = mlflow.MlflowClient()
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
            artifact_path="tfidf_vectorizer.pkl"
            )
        with open(vectorizer_path, "rb") as f:
            vectorizer = pickle.load(f)
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