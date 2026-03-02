from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List
import joblib
import pandas as pd
import uvicorn
import os
import sys

# Insert src into path so we can import feature_engineering
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
from feature_engineering import create_derived_features

app = FastAPI(
    title="Placement Probability Predictor API",
    description="API to predict student placement chances",
    version="1.0.0"
)

# Load Models
base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
model_path = os.path.join(base_dir, 'models', 'placement_model.pkl')
preprocessor_path = os.path.join(base_dir, 'models', 'preprocessor.pkl')
engineered_data_path = os.path.join(base_dir, 'data', 'placement_data_engineered.csv')

try:
    model = joblib.load(model_path)
    preprocessor = joblib.load(preprocessor_path)
    # Get expected features from engineered dataset
    df_engineered = pd.read_csv(engineered_data_path)
    expected_features = df_engineered.drop('Placed', axis=1).columns.tolist()
except Exception as e:
    model = None
    preprocessor = None
    expected_features = None
    print(f"Warning: Models or data not found. Ensure pipeline is run first. Details: {e}")


# Input Schema
class SkillItem(BaseModel):
    skill: str = Field(..., description="Name of the skill (e.g., Python, AWS, Communication)")
    level: str = Field(..., description="Proficiency level (e.g., Beginner, Intermediate, Advanced, Expert)")

class StudentFeatures(BaseModel):
    CGPA: float = Field(..., ge=0.0, le=10.0, description="Cumulative Grade Point Average")
    Backlogs: int = Field(..., ge=0, description="Number of active backlogs")
    Branch: str = Field(..., description="Engineering Branch (e.g., CSE, IT, ECE, CYBER, AIDS, AIML)")
    tenth_Percentage: float = Field(..., alias="10th_Percentage", ge=0.0, le=100.0)
    twelfth_Percentage: float = Field(..., alias="12th_Percentage", ge=0.0, le=100.0)
    skills: List[SkillItem] = Field(..., description="List of skills the user possesses")
    DSA_Score: float = Field(..., ge=0.0, le=100.0)
    Projects_Count: int = Field(..., ge=0)
    Certifications: int = Field(..., ge=0)
    Internships: int = Field(..., ge=0)
    Hackathons_Participated: int = Field(..., ge=0)
    Coding_Contest_Rating: int = Field(..., ge=0)
    Clubs: int = Field(..., ge=0)
    Leadership_Roles: int = Field(..., ge=0)

    class Config:
        populate_by_name = True

class PredictionResponse(BaseModel):
    placement_probability: float
    risk_level: str

@app.get("/health")
def health_check():
    return {"status": "healthy", "model_loaded": model is not None}

@app.post("/predict", response_model=PredictionResponse)
def predict_placement(student: StudentFeatures):
    if model is None or preprocessor is None:
        raise HTTPException(status_code=503, detail="Model is currently unavailable. Please check server logs.")
        
    try:
        # 1. Convert to DataFrame
        # Fast API handles the alias correctly, but dumping uses fields
        # Note: 'skills' contains a list of Pydantic models. We dump dict format.
        student_dict = student.dict(by_alias=True)
        # Convert List of dicts explicitly for pandas
        student_dict['skills'] = [s for s in student_dict['skills']]
        
        df = pd.DataFrame([student_dict])
        
        # 2. Add derived features via module
        df_engineered = create_derived_features(df)
        
        # 3. Ensure we have exactly expected columns to avoid mismatch errors
        missing_cols = set(expected_features) - set(df_engineered.columns)
        if missing_cols:
             raise HTTPException(status_code=500, detail=f"Feature engineering failed, missing: {missing_cols}")
        
        X = df_engineered[expected_features]
        
        # 4. Preprocess
        X_processed = preprocessor.transform(X)
        
        # 5. Predict
        probability = float(model.predict_proba(X_processed)[0][1])
        
        # 6. Determine Risk Level
        if probability > 0.75:
            risk = "High Chance"
        elif probability >= 0.50:
            risk = "Moderate Chance"
        else:
            risk = "Low Chance"
            
        return PredictionResponse(
            placement_probability=round(probability * 100, 2),
            risk_level=risk
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

# Test function starter
if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
