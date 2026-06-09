import os
import sys
import json
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException, UploadFile, File, Depends
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import text
from pypdf import PdfReader
import io

# Setup path so it can import from parent directory correctly
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.abspath(os.path.join(current_dir, '..')))

from api.schemas import (
    StudentProfile, PredictionResponse, ParsedResumeResponse,
    MiningTipsResponse, WarehouseStats, ModelInfoResponse
)
from database.connection import engine, check_connection
from ml.resume_parser import ResumeParser
from ml.recommender import CandidateRecommender
from ml.mining_engine import get_placement_rules

# ==========================================
# 1. SETUP & APP CONFIGURATION
# ==========================================
app = FastAPI(
    title="Placement Prediction & Career Analytics API",
    description="REST API providing explainable machine learning predictions, TF-IDF resume analytics, and Apriori career strategy mining.",
    version="2.0"
)

# Enable CORS for Streamlit and React integrations
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize Core Services
resume_parser = ResumeParser()
recommender = CandidateRecommender()

# Load trained Random Forest Pipeline
model_path = os.path.join(current_dir, '..', 'ml', 'models', 'placement_pipeline.pkl')
feature_cols_path = os.path.join(current_dir, '..', 'ml', 'models', 'feature_columns.pkl')

pipeline = None
feature_columns = []

if os.path.exists(model_path):
    try:
        pipeline = joblib.load(model_path)
        feature_columns = joblib.load(feature_cols_path)
        print("[+] API Success: ML Pipeline loaded successfully.")
    except Exception as e:
        print(f"[-] API Error: Failed to deserialize ML model: {e}")
else:
    print("[-] API Warning: Model pipeline file not found. Predict endpoint running in degraded mode.")


# ==========================================
# 2. ENDPOINTS
# ==========================================

@app.get("/")
def read_root():
    return {
        "status": "Online",
        "service": "Placement Prediction & Candidate Evaluation API",
        "version": "2.0",
        "database_connected": check_connection()
    }

@app.post("/predict", response_model=PredictionResponse)
def predict_placement(profile: StudentProfile):
    """
    Computes candidate placement probability using the Random Forest classifier
    and runs simulations to generate a dynamic, mathematically-backed career action plan.
    """
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Machine Learning model pipeline is not trained or loaded.")
        
    try:
        # 1. Map input profile to feature dataframe format
        input_data = {col: 0 for col in feature_columns}
        input_data['cgpa'] = profile.cgpa
        input_data['coding_score'] = profile.coding_score
        input_data['internships'] = profile.internships
        input_data['projects_count'] = profile.projects_count
        input_data['certifications_count'] = profile.certifications_count
        input_data['work_ex_bin'] = 1 if profile.work_experience == 'Yes' else 0
        
        # Skill mappings
        skill_mapping = {
            'Python': 'skill_python',
            'SQL': 'skill_sql',
            'Java': 'skill_java',
            'Data Analysis': 'skill_data_analysis',
            'Web Dev': 'skill_web_dev',
            'Machine Learning': 'skill_ml',
            'Cloud': 'skill_cloud',
            'React': 'skill_react'
        }
        for skill in profile.skills:
            if skill in skill_mapping:
                input_data[skill_mapping[skill]] = 1
                
        df_input = pd.DataFrame([input_data])[feature_columns]
        
        # 2. Perform inference
        prediction = pipeline.predict(df_input)[0]
        probability = pipeline.predict_proba(df_input)[0][1]
        
        # 3. Dynamic recommendation yields
        profile_dict = profile.model_dump()
        recommendations = recommender.generate_recommendations(profile_dict)
        
        # 4. Formulate contextual advice based on model probability
        if probability >= 0.85:
            advice = "Outstanding profile! You have a high probability of securing top-tier placements. Polish situational interview responses and system design concepts."
        elif probability >= 0.60:
            advice = "Strong foundation! To establish absolute competitive advantage, act on the high-yield recommendations below (specifically adding targeted skills or an internship)."
        else:
            advice = "Placement prospects currently require strategic development. Completing a technical internship or building additional projects will dramatically improve your employability metrics."
            
        return {
            "placement_status": "Placed" if prediction == 1 else "Not Placed",
            "probability": round(probability * 100, 1),
            "advice": advice,
            "recommendations": recommendations
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference pipeline crashed: {str(e)}")


@app.post("/parse-resume", response_model=ParsedResumeResponse)
def parse_resume(file: UploadFile = File(...)):
    """
    Ingests a candidate resume PDF, extracts text, categorizes the professional domain
    using a TF-IDF Cosine Similarity engine, and pulls structural credentials using pattern matching.
    """
    if not file.filename.lower().endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Only PDF formatted resumes are supported.")
        
    try:
        contents = file.file.read()
        pdf_file = io.BytesIO(contents)
        reader = PdfReader(pdf_file)
        
        raw_text = ""
        for page in reader.pages:
            raw_text += page.extract_text() or ""
            
        if not raw_text.strip():
            raise HTTPException(status_code=422, detail="Unable to extract indexable text from uploaded PDF resume.")
            
        profile = resume_parser.extract_profile(raw_text)
        return profile
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Resume parser execution failure: {str(e)}")


@app.get("/tips", response_model=MiningTipsResponse)
def get_strategic_tips():
    """
    Executes the Apriori Association Rule algorithm on the data warehouse
    to discover trending technical skill combinations and practical career boosts.
    """
    try:
        rules = get_placement_rules()
        return {"rules": rules}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Pattern mining crashed: {str(e)}")


@app.get("/stats", response_model=WarehouseStats)
def get_warehouse_statistics():
    """
    Queries the PostgreSQL Data Warehouse facts table to compute overall placement distribution,
    total cohort counts, and baseline average salary.
    """
    try:
        query = """
        SELECT 
            COUNT(student_id) as total,
            SUM(CASE WHEN placement_status = 'Placed' THEN 1 ELSE 0 END) as placed,
            SUM(CASE WHEN placement_status = 'Not Placed' THEN 1 ELSE 0 END) as not_placed,
            AVG(CASE WHEN placement_status = 'Placed' THEN salary ELSE NULL END) as avg_sal
        FROM fact_placements;
        """
        
        with engine.connect() as conn:
            result = conn.execute(text(query)).fetchone()
            
        total = int(result[0] or 0)
        placed = int(result[1] or 0)
        not_placed = int(result[2] or 0)
        avg_salary = float(result[3] or 0.0)
        
        placement_rate = (placed / total * 100) if total > 0 else 0.0
        
        return {
            "placed_count": placed,
            "not_placed_count": not_placed,
            "total_records": total,
            "average_salary": round(avg_salary, 2),
            "placement_rate": round(placement_rate, 2)
        }
        
    except Exception as db_err:
        print(f"[-] Database query failed in stats route: {db_err}")
        # Fallback count logic using processed CSV
        try:
            csv_path = os.path.join(current_dir, '..', 'data', 'processed', 'master_student_dataset.csv')
            df = pd.read_csv(csv_path)
            total = len(df)
            placed = int((df['Placement_Status'] == 'Placed').sum())
            not_placed = total - placed
            avg_salary = float(df[df['Placement_Status'] == 'Placed']['Salary'].mean() or 0.0)
            placement_rate = (placed / total * 100) if total > 0 else 0.0
            
            return {
                "placed_count": placed,
                "not_placed_count": not_placed,
                "total_records": total,
                "average_salary": round(avg_salary, 2),
                "placement_rate": round(placement_rate, 2)
            }
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to fetch warehouse stats: {str(e)}")


@app.get("/model-info", response_model=ModelInfoResponse)
def get_model_validation_info():
    """
    Exposes direct metrics metadata (Precision, Recall, F1-Score, Cross-Validation, Feature Importances)
    to serve transparent explainability insights.
    """
    metrics_path = os.path.join(current_dir, '..', 'ml', 'models', 'evaluation_metrics.json')
    
    if not os.path.exists(metrics_path):
        raise HTTPException(status_code=404, detail="Model evaluation metrics JSON not found. Please run training pipeline first.")
        
    try:
        with open(metrics_path, 'r') as f:
            metrics_payload = json.load(f)
        return metrics_payload
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to read model validation logs: {str(e)}")