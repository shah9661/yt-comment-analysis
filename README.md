<a target="_blank" href="https://cookiecutter-data-science.drivendata.org/">
    <img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" />
</a>

# Influencer Insights — YouTube Comment Sentiment Analysis

An end-to-end **NLP and Machine Learning project** that analyzes YouTube comments and converts unstructured audience feedback into actionable insights.

The project covers the ML lifecycle, including **data preprocessing, feature engineering, model development, hyperparameter tuning, experiment tracking, data versioning, API development, frontend integration, containerization, CI/CD, and AWS deployment**.

---

## Overview

Large influencers can receive thousands of comments on their YouTube videos, making manual analysis time-consuming and difficult to scale.

**Influencer Insights** automates comment analysis through:

- Multiclass sentiment classification
- Sentiment distribution analysis
- Comment-level sentiment insights
- Word-cloud analysis
- Sentiment trend analysis
- Average comment-length analysis
- Exportable analysis reports

---

## Problem Statement

The objective is to automatically extract meaningful information from large volumes of unstructured YouTube comments.

The dataset presents real-world NLP challenges including:

- Noisy and informal text
- Slang and emojis
- Multilingual comments
- Spam and bot comments
- Sarcasm
- Evolving language and concept drift
- Class imbalance

These challenges require appropriate preprocessing, feature engineering, model selection, and evaluation.

## End-to-End Workflow

```text
Data Collection
      ↓
Data Cleaning & Preprocessing
      ↓
EDA & Feature Engineering
      ↓
Model Training
      ↓
Hyperparameter Tuning
      ↓
Model Evaluation
      ↓
MLflow Experiment Tracking
      ↓
DVC Pipeline & Model Registry
      ↓
   FastAPI
      ↓
Testing & CI/CD
      ↓
Docker + AWS Deployment
```

---

## Machine Learning

### Model: LightGBM

The final model uses **LightGBM** for multiclass sentiment classification.

Class imbalance is handled using `class_weight="balanced"` and `is_unbalance=True`. L1 and L2 regularization are also applied.

```python
best_model = lgb.LGBMClassifier(
    objective="multiclass",
    num_class=3,
    metric="multi_logloss",
    is_unbalance=True,
    class_weight="balanced",
    reg_alpha=0.1,
    reg_lambda=0.1,
    learning_rate=learning_rate,
    max_depth=max_depth,
    n_estimators=n_estimators
)
```

### Best Configuration

| Parameter | Value |
|---|---:|
| N-gram range | `(1, 3)` |
| Max features | `10,000` |
| Learning rate | `0.09` |
| Max depth | `20` |
| N estimators | `367` |
| Test size | `0.20` |

The `(1, 3)` range represents **unigrams, bigrams, and trigrams**.

---

## Model Evaluation

### Classification Report

| Class | Precision | Recall | F1-Score |
|---|---:|---:|---:|
| 0 | 86.04% | 96.02% | 90.76% |
| 1 | 91.01% | 83.62% | 87.16% |
| -1 | 80.59% | 78.28% | 79.42% |
| **Macro Avg** | **85.88%** | **85.97%** | **85.78%** |
| **Weighted Avg** | **86.90%** | **86.76%** | **86.67%** | |

Because the dataset is imbalanced, **Macro F1-score** is reported alongside weighted metrics to give equal importance to each class.

### Confusion Matrix

| Actual \ Predicted | 0 | 1 | 2 |
|---|---:|---:|---:|
| **0** | **1308** | 149 | 214 |
| **1** | 61 | **2484** | 42 |
| **2** | 254 | 254 | **2593** |

The diagonal represents correctly classified samples.

> **Label note:** The recorded MLflow metrics use labels `0`, `1`, and `-1`, while the project requirements describe the sentiment categories as Positive, Neutral, and Negative. The numeric-to-sentiment mapping is therefore not assumed in this README.

---

## MLOps

### MLflow

Used for:

- Experiment tracking
- Parameter and metric logging
- Artifact tracking
- Experiment comparison
- Model version management
- Model Registry

### DVC

Used for:

- Dataset versioning
- ML pipeline versioning
- Reproducibility
- Collaboration

### AWS S3

Used to store datasets, processed data, and model artifacts associated with the DVC workflow.

---

## Application Architecture

```text
 Chrome Extension
       |
       v
Fetch Comment
       |
       v
    FastAPI
       |
       v
   ML Model
       |
       v
Sentiment Predictions
       |
       v
Insights & Visualizations
```

### Backend — FastAPI

The trained model is exposed through a **FastAPI REST API** that handles prediction requests from the frontend.

### Frontend — Chrome Extension 
https://github.com/shah9661/youtube_comments_analysis_plugin

The Chrome Extension is part of the frontend layer and is built using:

- JavaScript
- HTML
- CSS

It interacts with YouTube, sends comment data to the FastAPI backend, and displays sentiment insights and visualizations.

---

## Deployment

### CI/CD

GitHub Actions automates testing, building, and deployment workflows.

```text
     Git Push 
          |
          v
       Testing
          |
          v
        Build
          |
          v
   Docker Image Build
          |
          v
      Deployment
```

### Docker

The application is containerized using Docker. The container image is built and pushed to a container registry before deployment.

### AWS

The deployment architecture uses:

- **Amazon ECR** — container image storage
- **Amazon EC2** — application and model hosting
- **Amazon S3** — data and model artifact storage
- **AWS IAM** — access and permission management

---

## Technology Stack

| Category | Technologies |
|---|---|
| Programming | Python, JavaScript |
| Machine Learning | LightGBM, scikit-learn |
| NLP | NLTK, spaCy |
| Data Processing | Pandas, NumPy |
| Hyperparameter Tuning | Optuna |
| Experiment Tracking | MLflow |
| Data Versioning | DVC |
| Backend API | FastAPI |
| Frontend | HTML, CSS, JavaScript, Chrome Extension APIs |
| Testing | Pytest |
| CI/CD | GitHub Actions |
| Containerization | Docker |
| Cloud | AWS EC2, ECR, S3, IAM |

---

## Key Technical Highlights

- Built a **multiclass NLP classification pipeline** for YouTube comments.
- Addressed **class imbalance** using class-weighting techniques.
- Used **LightGBM** with L1/L2 regularization.
- Tuned hyperparameters using **Optuna**.
- Achieved **86.67% weighted F1-score** and **85.78% macro F1-score** on the test set.
- Implemented **MLflow** for experiment tracking and model management.
- Implemented **DVC** for data and pipeline versioning.
- Built a **FastAPI REST API** for model inference.
- Integrated the model with a **Chrome Extension frontend**.
- Containerized the application using **Docker**.
- Automated testing and deployment using **GitHub Actions**.
- Deployed the application using **AWS infrastructure**.

---

## Future Improvements

- Improve multilingual sentiment analysis
- Improve sarcasm detection
- Add dedicated spam and bot detection
- Monitor concept drift
- Improve real-time comment processing
- Implement automated model monitoring and retraining

---

## Author

**Shanshad Alam**

Machine Learning / AI — Influencer Insights

