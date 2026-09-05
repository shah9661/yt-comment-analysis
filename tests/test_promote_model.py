import os
import mlflow
import pytest
from dotenv import load_dotenv
from mlflow.tracking import MlflowClient

load_dotenv()
DAGSHUB_TOKEN = os.getenv("DAGSHUB_TOKEN")
if not DAGSHUB_TOKEN:
    raise EnvironmentError(
        "DAGSHUB_TOKEN environment variable is not set"
    )

REPO_OWNER = "shanshad0999"
REPO_NAME = "yt-comment-analysis"

MODEL_NAME = "my_model"

STAGING_ALIAS = "staging"
CHAMPION_ALIAS = "champion"

os.environ["MLFLOW_TRACKING_USERNAME"] = REPO_OWNER
os.environ["MLFLOW_TRACKING_PASSWORD"] = DAGSHUB_TOKEN

MLFLOW_TRACKING_URI = (f"https://dagshub.com/{REPO_OWNER}/{REPO_NAME}.mlflow")

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

def promote_staging_to_champion():
    try:
        client = MlflowClient()
        staging_model = client.get_model_version_by_alias(MODEL_NAME,STAGING_ALIAS)
        staging_version = staging_model.version
        client.set_registered_model_alias(name=MODEL_NAME,
            alias=CHAMPION_ALIAS,version=staging_version)
        return staging_version
    except Exception as e:
        raise RuntimeError(
            f"Failed to promote model "
            f"from '{STAGING_ALIAS}' to '{CHAMPION_ALIAS}': {e}"
        ) from e

def test_promote_staging_to_champion():
    client = MlflowClient()
    staging_model = client.get_model_version_by_alias(MODEL_NAME,STAGING_ALIAS)
    assert staging_model is not None, (
        f"No model found with '{STAGING_ALIAS}' alias"
    )
    staging_version = staging_model.version
    print(f"\nStaging model found:"
          f" {MODEL_NAME} version {staging_version}")

    promoted_version = promote_staging_to_champion()
    assert promoted_version == staging_version

    champion_model = client.get_model_version_by_alias(MODEL_NAME,CHAMPION_ALIAS)
    assert champion_model is not None, ("Champion alias was not created")
    assert champion_model.version == staging_version, (
        f"Expected champion version "
        f"{staging_version}, "
        f"but got {champion_model.version}"
    )
    print(
        f"Model '{MODEL_NAME}' version "
        f"{staging_version} successfully promoted "
        f"from '{STAGING_ALIAS}' to '{CHAMPION_ALIAS}'."
    )