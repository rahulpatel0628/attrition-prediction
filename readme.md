# 🧠 AI-Powered Employee Attrition & Retention Risk System

A production-grade Machine Learning project that predicts employee attrition probability and classifies risk level (Low / Medium / High) using the IBM HR Analytics dataset.

---

## 📁 Project Structure

```
attrition-project/
│
├── data/
│   └── WA_Fn-UseC_-HR-Employee-Attrition.csv   ← Place dataset here
│
├── notebooks/
│   └── 01_eda_and_modeling.ipynb                ← Full EDA + Model Training Notebook
│
├── src/
│   ├── data_cleaning.py                         ← Step 2: Data Cleaning
│   ├── preprocessing.py                         ← Step 3: Preprocessing Pipeline
│   ├── eda.py                                   ← Step 4: EDA Plots
│   ├── feature_engineering.py                   ← Step 5: Feature Engineering
│   ├── train.py                                 ← Step 6,7,8: Train, Tune, Select Best Model
│   └── predict.py                               ← Inference logic
│
├── models/
│   ├── best_model.pkl                           ← Saved best model (joblib)
│   ├── scaler.pkl                               ← Saved scaler
│   ├── encoder.pkl                              ← Saved encoders
│   └── feature_list.json                        ← Feature names used in training
│
├── frontend/
│   ├── index.html                               ← Main UI
│   ├── style.css                                ← Styles
│   └── app.js                                   ← API calls + Charts
│
├── main.py                                      ← FastAPI App
├── Dockerfile                                   ← Docker config
├── docker-compose.yml                           ← Docker Compose
├── requirements.txt                             ← All dependencies
└── README.md
```

---

## 🚀 Quick Start

### 1. Clone & Setup Environment
```bash
# Create virtual environment
python -m venv venv
venv\Scripts\activate          # Windows

# Install dependencies
pip install -r requirements.txt
```

### 2. Download Dataset
Download from Kaggle:
👉 https://www.kaggle.com/datasets/pavansubhasht/ibm-hr-analytics-attrition-dataset

Place `WA_Fn-UseC_-HR-Employee-Attrition.csv` inside the `data/` folder.

### 3. Train the Model
```bash
python src/train.py
```

### 4. Run FastAPI Server
```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

### 5. Open Frontend
Open `frontend/index.html` in your browser.

### 6. Run with Docker
```bash
docker-compose up --build
```

---

## 📊 ML Pipeline

| Step | Description |
|------|-------------|
| Data Cleaning | Remove nulls, drop constants, fix types |
| Preprocessing | Encode categoricals, scale numerics |
| EDA | Visualize attrition patterns |
| Feature Engineering | RFM-style HR features |
| Model Training | XGBoost, Random Forest, LightGBM |
| Hyperparameter Tuning | Optuna |
| Feature Selection | SHAP values |
| Best Model | ROC-AUC, F1-Score comparison |
| Save | joblib .pkl files |
| API | FastAPI + Docker |
| Frontend | HTML/CSS/JS Dashboard |

---

## 🎯 API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Health check |
| `/predict` | POST | Predict attrition risk |
| `/docs` | GET | Swagger UI |

---

## 📦 Tech Stack
- **ML**: Scikit-learn, XGBoost, LightGBM, Ensemble technic,GreadSearchCV
- **Backend**: FastAPI, Uvicorn, Pydantic
- **Deployment**: Docker, Docker Compose
- **Frontend**: HTML5, CSS3, Vanilla JS, Chart.js
