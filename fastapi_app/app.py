from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

from fastapi_app.model_loader import load_production_resources
from fastapi_app.clean_comment import preprocess_comment

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
class PredictionRequest(BaseModel):
    comment: str

model, vectorizer, model_info = load_production_resources()



@app.get("/")
def home():
    return {
        "message": "YouTube Comment Sentiment API",
        "model_version": model_info["model_version"]
    }
@app.post("/predict")
def predict(request: PredictionRequest):
    try:
        comment = preprocess_comment(request.comment)
        features = vectorizer.transform([comment])
        prediction = model.predict(features)

        return {
            "comment": comment,
            "prediction": prediction.tolist()
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(e)}"
        )