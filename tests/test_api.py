import pytest
import requests
BASE_URL = "http://127.0.0.1:8000"

def test_home():
    response = requests.get(f"{BASE_URL}/")
    assert response.status_code == 200
    data = response.json()
    assert data["message"] == ("YouTube Comment Sentiment API")
    assert "model_version" in data
    assert data["model_version"] is not None

def test_predict():
    payload = {
        "comment": [
            "This video is amazing!",
            "I really enjoyed this video.",
            "This is a terrible video."
        ]
    }
    response = requests.post(
        f"{BASE_URL}/predict",
        json=payload
    )
    assert response.status_code == 200
    data = response.json()
    assert "comment" in data
    assert "prediction" in data
    assert len(data["comment"]) == 3
    assert len(data["prediction"]) == 3

def test_predict_empty_comments():
    payload = {"comment": []}
    response = requests.post(f"{BASE_URL}/predict",json=payload)
    assert response.status_code in [200, 500]

def test_generate_chart():
    payload = {
        "sentiment_counts": {
            "1": 50,
            "0": 30,
            "-1": 20
        }
    }
    response = requests.post(
        f"{BASE_URL}/generate_chart",
        json=payload
    )
    assert response.status_code == 200
    assert response.headers["content-type"].startswith(
        "image/png"
    )
    assert len(response.content) > 0

def test_generate_chart_empty_counts():
    payload = {"sentiment_counts": {}}
    response = requests.post(
        f"{BASE_URL}/generate_chart",
        json=payload
    )
    assert response.status_code == 400
    data = response.json()
    assert data["detail"] == (
        "No sentiment counts provided"
    )

def test_generate_chart_zero_counts():
    payload = {
        "sentiment_counts": {
            "1": 0,
            "0": 0,
            "-1": 0
        }
    }
    response = requests.post(
        f"{BASE_URL}/generate_chart",
        json=payload
    )
    assert response.status_code == 400
    data = response.json()
    assert data["detail"] == (
        "Sentiment counts sum to zero"
    )

def test_generate_wordcloud():
    payload = {
        "comment": [
            "This video is amazing",
            "Amazing content",
            "I really enjoyed this video",
            "Great video"
        ]
    }

    response = requests.post(
        f"{BASE_URL}/generate_wordcloud",
        json=payload
    )

    assert response.status_code == 200
    assert response.headers["content-type"].startswith(
        "image/png"
    )
    assert len(response.content) > 0

def test_generate_wordcloud_empty_comments():
    payload = {
        "comment": []
    }
    response = requests.post(
        f"{BASE_URL}/generate_wordcloud",
        json=payload
    )
    assert response.status_code == 400
    data = response.json()
    assert data["detail"] == (
        "No comments provided in word cloud"
    )

def test_generate_trend_graph():
    payload = {
        "sentiment_data": [
            {
                "timestamp": "2025-01-15T10:00:00",
                "sentiment": 1
            },
            {
                "timestamp": "2025-01-20T12:00:00",
                "sentiment": 0
            },
            {
                "timestamp": "2025-02-10T14:00:00",
                "sentiment": -1
            },
            {
                "timestamp": "2025-02-15T16:00:00",
                "sentiment": 1
            },
            {
                "timestamp": "2025-03-05T11:00:00",
                "sentiment": 1
            }
        ]
    }
    response = requests.post(
        f"{BASE_URL}/generate_trend_graph",
        json=payload
    )
    assert response.status_code == 200
    assert response.headers["content-type"].startswith(
        "image/png"
    )
    assert len(response.content) > 0

def test_generate_trend_graph_empty_data():
    payload = {
        "sentiment_data": []
    }
    response = requests.post(
        f"{BASE_URL}/generate_trend_graph",
        json=payload
    )
    assert response.status_code == 400
    data = response.json()
    assert data["detail"] == (
        "No sentiment data provided"
    )