# Ad Click Prediction – End-to-End Machine Learning Application

Python • Angular • FastAPI • Docker • MLflow

University: Tek-Up — Guided ML Project (Python for Data Science 2)

Author: Mohamed Haroun Mezned

## Project Overview

This project is a complete End-to-End Machine Learning System that predicts the probability of a user clicking on an online advertisement.
It includes:

• Data scraping & enrichment

• Full ML pipeline with SMOTE & MLflow

• FastAPI backend for prediction

• Angular interactive dashboard

• Docker containerization for deployment

                ┌──────────────────────────┐
                │Phase 1: Data Engineering │
                │Scraping • Cleaning • EDA │
                └───────────────┬──────────┘
                                ▼
                  ┌─────────────────────────┐
                  │ Phase 2: ML Pipeline    │
                  │ Preprocessing • SMOTE   │
                  │ XGBoost/RF • MLflow     │
                  └───────────────┬─────────┘
                                  ▼
                ┌────────────────────────────────┐
                │ Phase 3: Backend + Frontend    │
                │ FastAPI • Angular • Docker     │
                └────────────────────────────────┘

## Dataset
Dataset used: ad_click_dataset.csv(from Kaggle)

Link → https://www.kaggle.com/datasets/abhishekmungoli/ad-click-dataset

Contains demographic, device, browsing, and ad interaction data.

## Data Dictionary
| Column           | Type    | Description                        | Notes                    |
| ---------------- | ------- | ---------------------------------- | ------------------------ |
| id               | Integer | Unique User Identifier             | Dropped (not predictive) |
| full_name        | String  | User Name                          | Dropped                  |
| age              | Float   | User age                           | Missing values           |
| gender           | String  | Male/Female/Non-Binary             | Missing values           |
| device_type      | String  | Desktop/Mobile/Tablet              | Missing values           |
| ad_position      | String  | Top/Side/Bottom                    | Categorical              |
| browsing_history | String  | Content category (Shopping, News…) | Used for scraping trends |
| time_of_day      | String  | Morning/Afternoon/Night            | Categorical              |
| click            | Integer | Target variable (0/1)              | 1 = Clicked              |

## Data Quality
The dataset contains real-world noise:

• Missing values → handled with SimpleImputer

• Imbalanced classes → solved using SMOTE

• Text fields requiring preprocessing

## Key Features
1. External Data Enrichment (Web Scraping)

    Source: CNBC Technology Section

    → Creates a feature is_trending based on tech-related titles.

2. Feature Engineering

  | Feature              | Description                   |
  | -------------------- | ----------------------------- |
  | `tech_savvy_segment` | Based on age + device type    |
  | `is_holiday_today`   | Tunisian holidays API/Scraper |
  | `is_trending`        | Trending tech news matching   |

## Machine Learning Pipeline
### Models used
• RandomForestClassifier

• XGBoostClassifier

### Pipeline Stages
• SimpleImputer

• OneHotEncoder

• SMOTE

• GridSearchCV

• MLflow experiment tracking

### Artifacts
• best_model.pkl

• MLflow logs
## Backend (FastAPI)
### Endpoints
| Method | Route      | Description                    |
| ------ | ---------- | ------------------------------ |
| GET    | `/`        | Health check                   |
| POST   | `/predict` | Returns prediction probability |

### Run FastAPI
cmd :
    
    uvicorn app:app --reload --host 0.0.0.0 --port 8000

Access Swagger Docs → http://localhost:8000/docs

## Frontend (Angular)
Built with Angular 17, using:

• Angular Material

• Reactive Forms

• HttpClient for API calls

Features:

• Prediction input form

• Probability gauge

• Responsive dashboard

Run : 

    ng serve

URL → http://localhost:4200

## Installation & Setup
1.Clone Projet:

    git clone https://github.com/your-username/ad-click-project.git
    cd ad-click_project

2.Backend Setup:

    python -m venv venv
    source venv/bin/activate       # Windows: venv\Scripts\activate
    pip install -r requirements.txt

3.Frontend Setup:

    cd frontend
    npm install

4.Run in Dev Mode:

  • Backend :

    cd frontend
    ng serve

• Frontend :
  
    cd code
    uvicorn app:app --reload

## Docker Deployment

From project root:

    docker-compose up --build
Access app → http://localhost:4200

## Project Structure

    ad_click_project/
    │── code/
    │   ├── app.py
    │   ├── model.pkl
    │   ├── scraping.py
    │   └── eda.py
    │
    │── frontend/
    │   └── src/app/
    │       ├── components/
    │       └── services/
    │
    │── data/
    │   ├── ad_click_dataset.csv
    │   └── ad_data_enriched.csv
    │
    │── tutos/
    │── Dockerfile
    │── Dockerfile.frontend
    │── docker-compose.yml
    └── README.md
    
## Author

Mohamed Haroun Mezned 

Tek-Up University 2026

























