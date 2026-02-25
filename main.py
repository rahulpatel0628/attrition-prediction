
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional, List
import uvicorn



app = FastAPI(
    title="Employee Attrition Risk API",
    description="Predicts the probability an employee will leave, with risk level and retention recommendations.",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)




class EmployeeInput(BaseModel):
   
    Age: int = Field(..., ge=18, le=65, example=32)
    BusinessTravel: str = Field(..., example="Travel_Frequently")
    DailyRate: int = Field(..., ge=100, le=1500, example=800)
    Department: str = Field(..., example="Sales")
    DistanceFromHome: int = Field(..., ge=1, le=30, example=15)
    Education: int = Field(..., ge=1, le=5, example=3)
    EducationField: str = Field(..., example="Life Sciences")
    EnvironmentSatisfaction: int = Field(..., ge=1, le=4, example=2)
    Gender: str = Field(..., example="Male")
    HourlyRate: int = Field(..., ge=30, le=100, example=65)
    JobInvolvement: int = Field(..., ge=1, le=4, example=2)
    JobLevel: int = Field(..., ge=1, le=5, example=2)
    JobRole: str = Field(..., example="Sales Representative")
    JobSatisfaction: int = Field(..., ge=1, le=4, example=2)
    MaritalStatus: str = Field(..., example="Single")
    MonthlyIncome: int = Field(..., ge=1000, le=20000, example=3500)
    MonthlyRate: int = Field(..., ge=2000, le=27000, example=14000)
    NumCompaniesWorked: int = Field(..., ge=0, le=10, example=3)
    OverTime: int = Field(..., ge=0, le=1, example=1)
    PercentSalaryHike: int = Field(..., ge=10, le=25, example=11)
    PerformanceRating: int = Field(..., ge=3, le=4, example=3)
    RelationshipSatisfaction: int = Field(..., ge=1, le=4, example=2)
    StockOptionLevel: int = Field(..., ge=0, le=3, example=0)
    TotalWorkingYears: int = Field(..., ge=0, le=40, example=8)
    TrainingTimesLastYear: int = Field(..., ge=0, le=6, example=2)
    WorkLifeBalance: int = Field(..., ge=1, le=4, example=2)
    YearsAtCompany: int = Field(..., ge=0, le=40, example=3)
    YearsInCurrentRole: int = Field(..., ge=0, le=18, example=2)
    YearsSinceLastPromotion: int = Field(..., ge=0, le=15, example=2)
    YearsWithCurrManager: int = Field(..., ge=0, le=17, example=1)

    class Config:
        json_schema_extra = {
            "example": {
                "Age": 32,
                "BusinessTravel": "Travel_Frequently",
                "DailyRate": 800,
                "Department": "Sales",
                "DistanceFromHome": 15,
                "Education": 3,
                "EducationField": "Life Sciences",
                "EnvironmentSatisfaction": 2,
                "Gender": "Male",
                "HourlyRate": 65,
                "JobInvolvement": 2,
                "JobLevel": 2,
                "JobRole": "Sales Representative",
                "JobSatisfaction": 2,
                "MaritalStatus": "Single",
                "MonthlyIncome": 3500,
                "MonthlyRate": 14000,
                "NumCompaniesWorked": 3,
                "OverTime": 1,
                "PercentSalaryHike": 11,
                "PerformanceRating": 3,
                "RelationshipSatisfaction": 2,
                "StockOptionLevel": 0,
                "TotalWorkingYears": 8,
                "TrainingTimesLastYear": 2,
                "WorkLifeBalance": 2,
                "YearsAtCompany": 3,
                "YearsInCurrentRole": 2,
                "YearsSinceLastPromotion": 2,
                "YearsWithCurrManager": 1
            }
        }


class PredictionResponse(BaseModel):
  
    will_attrite: bool
    attrition_probability: float
    risk_level: str
    recommended_actions: List[str]
    status: str = "success"




@app.get("/", tags=["Health"])
def health_check():
    
    return {
        "status": "Online",
        "service": "Employee Attrition Risk API",
        "version": "1.0.0",
        "docs": "/docs"
    }


@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
def predict_attrition(employee: EmployeeInput):
    
    try:
        from src.predict import predict_attrition as run_prediction
        employee_dict = employee.model_dump()
        result = run_prediction(employee_dict)
        result["status"] = "success"
        return result
    except FileNotFoundError:
        raise HTTPException(
            status_code=503,
            detail="Model not found. Please run 'python src/train.py' first."
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/model-info", tags=["Info"])
def model_info():
    
    import json
    from pathlib import Path
    metadata_path = Path("models/model_metadata.json")
    if not metadata_path.exists():
        raise HTTPException(status_code=404, detail="Model metadata not found. Train the model first.")
    with open(metadata_path) as f:
        return json.load(f)


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)

