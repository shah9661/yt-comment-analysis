from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
import io
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from fastapi_app.model_loader import load_production_resources
from fastapi_app.clean_comment import preprocess_comment
from wordcloud import WordCloud
from nltk.corpus import stopwords
import pandas as pd
import matplotlib.dates as mdates



app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class PredictionRequest(BaseModel):
    comment: list[str]
class SentimentCounts(BaseModel):
    sentiment_counts: dict[str, int]

class SentimentData(BaseModel):
    timestamp: str
    sentiment: int

class TrendGraphRequest(BaseModel):
    sentiment_data: list[SentimentData]

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
        cleaned_comments = [preprocess_comment(comment) for comment in request.comment]
        features = vectorizer.transform(cleaned_comments)
        prediction = model.predict(features)
        return {
            "comment": cleaned_comments,
            "prediction": prediction.tolist()
        }
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(e)}"
        )

@app.post("/generate_chart")
def generate_chart(request: SentimentCounts):
    try:
        sentiment_counts = request.sentiment_counts
        if not sentiment_counts:
            raise HTTPException(
                status_code=400,
                detail="No sentiment counts provided"
            )
        labels = ["Positive", "Neutral", "Negative"]
        sizes = [
            int(sentiment_counts.get("1", 0)),
            int(sentiment_counts.get("0", 0)),
            int(sentiment_counts.get("-1", 0))
        ]
        if sum(sizes) == 0:
            raise HTTPException(
                status_code=400,
                detail="Sentiment counts sum to zero"
            )
        colors = ["#36A2EB", "#C9CBCF", "#FF6384"]
        plt.figure(figsize=(6, 6))
        plt.pie(
            sizes,
            labels=labels,
            colors=colors,
            autopct="%1.1f%%",
            startangle=140,
            textprops={"color": "white"}
        )
        plt.axis("equal")
        img_io = io.BytesIO()
        plt.savefig(img_io,format="PNG",transparent=True,bbox_inches="tight")
        img_io.seek(0)
        plt.close()
        return StreamingResponse(img_io,media_type="image/png")
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Chart generation failed: {str(e)}"
        )


@app.post("/generate_wordcloud")
def generate_wordcloud(request:PredictionRequest):
    try:
        cleaned_comments =[preprocess_comment(comment) for comment in request.comment]
        if not cleaned_comments:
            raise HTTPException(
                status_code=400,
                detail="No comments provided in word cloud"
            )
        text = " ".join(cleaned_comments)
        wordcloud = WordCloud(
            width=800,
            height=400,
            background_color='black',
            colormap='Blues',
            stopwords=set(stopwords.words('english')),
            collocations=False
        ).generate(text)

        img_io = io.BytesIO()
        wordcloud.to_image().save(img_io, format='PNG')
        img_io.seek(0)
        return StreamingResponse(img_io, media_type="image/png")
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Word cloud generation failed: {str(e)}"
        )



@app.post("/generate_trend_graph")
def generate_trend_graph(request: TrendGraphRequest):
    try:
        if not request.sentiment_data:
            raise HTTPException(
                status_code=400,
                detail="No sentiment data provided"
            )
        df = pd.DataFrame([
            {
                "timestamp": item.timestamp,
                "sentiment": item.sentiment
            }
            for item in request.sentiment_data
        ])
        df["timestamp"] = pd.to_datetime(df["timestamp"])

        df.set_index("timestamp", inplace=True)

        df["sentiment"] = df["sentiment"].astype(int)

        monthly_counts = (df.resample("ME")["sentiment"].value_counts().unstack(fill_value=0))

        monthly_totals = monthly_counts.sum(axis=1)

        monthly_percentages = (monthly_counts.T / monthly_totals).T * 100

        for sentiment_value in [-1, 0, 1]:
            if sentiment_value not in monthly_percentages.columns:
                monthly_percentages[sentiment_value] = 0

        monthly_percentages = monthly_percentages[[-1, 0, 1]]
        sentiment_labels = {
            -1: "Negative",
            0: "Neutral",
            1: "Positive"
        }
        plt.figure(figsize=(12, 6))
        colors = {
            -1: "red",
            0: "gray",
            1: "green"
        }
        for sentiment_value in [-1, 0, 1]:
            plt.plot(
                monthly_percentages.index,
                monthly_percentages[sentiment_value],
                marker="o",
                linestyle="-",
                label=sentiment_labels[sentiment_value],
                color=colors[sentiment_value]
            )
        plt.title("Monthly Sentiment Percentage Over Time")
        plt.xlabel("Month")
        plt.ylabel("Percentage of Comments (%)")

        plt.grid(True)
        plt.xticks(rotation=45)

        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
        plt.gca().xaxis.set_major_locator(mdates.AutoDateLocator(maxticks=12))
        plt.legend()
        plt.tight_layout()
        img_io = io.BytesIO()
        plt.savefig(img_io,format="PNG",bbox_inches="tight")
        img_io.seek(0)
        plt.close()
        return StreamingResponse(img_io,media_type="image/png")
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Trend graph generation failed: {str(e)}"
        )