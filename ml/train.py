import os
import json
import joblib
import pandas as pd
import numpy as np
from sqlalchemy import text
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

from database.connection import engine
from ml.pipeline import create_production_pipeline, ALL_FEATURES

def load_training_data():
    """Extracts the fully integrated analytical view from the PostgreSQL warehouse, with fallback to CSV"""
    print("[*] Connecting to PostgreSQL Warehouse to extract student records...")
    
    query = """
    SELECT 
        s.student_id,
        s.cgpa,
        s.coding_score,
        s.work_experience,
        e.internships,
        e.projects_count,
        e.certifications_count,
        t.skill_python,
        t.skill_sql,
        t.skill_java,
        t.skill_data_analysis,
        t.skill_web_dev,
        t.skill_ml,
        t.skill_cloud,
        t.skill_react,
        f.placement_status
    FROM fact_placements f
    JOIN dim_student s ON f.student_id = s.student_id
    JOIN dim_experience e ON f.student_id = e.student_id
    JOIN dim_skills t ON f.student_id = t.student_id;
    """
    
    try:
        with engine.connect() as conn:
            df = pd.read_sql(text(query), conn)
        if len(df) > 0:
            print(f"[+] Loaded {len(df)} student profiles from the PostgreSQL warehouse.")
            return df
        print("[-] PostgreSQL tables are empty. Falling back to local processed dataset.")
    except Exception as e:
        print(f"[-] Database query failed ({e}). Falling back to processed CSV dataset.")
        
    current_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(current_dir, '..', 'data', 'processed', 'master_student_dataset.csv')
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Neither PostgreSQL nor local CSV dataset found at: {csv_path}")
        
    print(f"[i] Loading training data from: {csv_path}")
    raw_csv = pd.read_csv(csv_path)
    
    df = pd.DataFrame({
        'student_id': raw_csv['Student_ID'],
        'cgpa': raw_csv['CGPA'],
        'coding_score': raw_csv['Coding_Score'],
        'work_experience': raw_csv['Work_Exp'],
        'internships': raw_csv['Internships'],
        'projects_count': raw_csv['Projects_Count'],
        'certifications_count': raw_csv['Certifications_Count'],
        'skill_python': raw_csv['Skills'].apply(lambda x: 1 if 'Python' in str(x) else 0),
        'skill_sql': raw_csv['Skills'].apply(lambda x: 1 if 'SQL' in str(x) else 0),
        'skill_java': raw_csv['Skills'].apply(lambda x: 1 if 'Java' in str(x) else 0),
        'skill_data_analysis': raw_csv['Skills'].apply(lambda x: 1 if 'Data Analysis' in str(x) else 0),
        'skill_web_dev': raw_csv['Skills'].apply(lambda x: 1 if 'Web Dev' in str(x) else 0),
        'skill_ml': raw_csv['Skills'].apply(lambda x: 1 if 'Machine Learning' in str(x) else 0),
        'skill_cloud': raw_csv['Skills'].apply(lambda x: 1 if 'Cloud' in str(x) else 0),
        'skill_react': raw_csv['Skills'].apply(lambda x: 1 if 'React' in str(x) else 0),
        'placement_status': raw_csv['Placement_Status']
    })
    print(f"[+] Loaded {len(df)} student profiles from the CSV dataset.")
    return df

def preprocess_df(df):
    """Encodes categorical fields and maps skill flags to integer matrices"""
    print("[*] Preprocessing raw SQL columns for machine learning...")
    
    # 1. Map Work Experience (Yes/No -> 1/0)
    df['work_ex_bin'] = df['work_experience'].map({'Yes': 1, 'No': 0}).fillna(0).astype(int)
    
    # 2. Map target Placement_Status (Placed/Not Placed -> 1/0)
    df['target'] = df['placement_status'].map({'Placed': 1, 'Not Placed': 0}).fillna(0).astype(int)
    
    # 3. Ensure boolean skill columns are cast to integers
    skill_cols = [c for c in df.columns if c.startswith('skill_')]
    for col in skill_cols:
        df[col] = df[col].astype(int)
        
    return df

def train_and_evaluate():
    # 1. Load and Clean
    raw_df = load_training_data()
    df = preprocess_df(raw_df)
    
    X = df[ALL_FEATURES]
    y = df['target']
    
    # 2. Train/Test Split (80/20)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    print(f"[i] Splitting dataset: Train shape={X_train.shape}, Test shape={X_test.shape}")
    
    # 3. Initialize Sklearn Pipeline
    pipeline = create_production_pipeline()
    
    # 4. 5-Fold Cross Validation on Training Set
    print("[*] Executing 5-Fold Cross Validation to verify model stability...")
    cv_scores = cross_val_score(pipeline, X_train, y_train, cv=5, scoring='accuracy')
    cv_mean = float(np.mean(cv_scores))
    cv_std = float(np.std(cv_scores))
    print(f"[+] Cross Validation Accuracy: {cv_mean*100:.2f}% (+/- {cv_std*100:.2f}%)")
    
    # 5. Fit model pipeline
    print("[*] Training final Random Forest Classifier...")
    pipeline.fit(X_train, y_train)
    print("[+] Model pipeline fitted successfully.")
    
    # 6. Evaluate on Test Set
    print("[*] Measuring model performance on unseen test data...")
    y_pred = pipeline.predict(X_test)
    y_prob = pipeline.predict_proba(X_test)[:, 1]
    
    accuracy = float(accuracy_score(y_test, y_pred))
    precision = float(precision_score(y_test, y_pred))
    recall = float(recall_score(y_test, y_pred))
    f1 = float(f1_score(y_test, y_pred))
    
    # Extract confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = map(int, cm.ravel())
    print(f"[+] Test Accuracy: {accuracy*100:.2f}%")
    print(f"[+] F1-Score: {f1:.4f}")
    
    # 7. Extract Feature Importances (Explainable AI)
    classifier = pipeline.named_steps['classifier']
    importances = classifier.feature_importances_
    
    # Map feature importances to feature list in order
    feature_importances = {}
    for feature, importance in zip(ALL_FEATURES, importances):
        feature_importances[feature] = float(importance)
        
    # Sort feature importances descending
    sorted_importances = dict(sorted(feature_importances.items(), key=lambda item: item[1], reverse=True))
    
    # 8. Save artifacts
    current_dir = os.path.dirname(os.path.abspath(__file__))
    model_dir = os.path.join(current_dir, 'models')
    os.makedirs(model_dir, exist_ok=True)
    
    model_path = os.path.join(model_dir, 'placement_pipeline.pkl')
    cols_path = os.path.join(model_dir, 'feature_columns.pkl')
    metrics_path = os.path.join(model_dir, 'evaluation_metrics.json')
    
    joblib.dump(pipeline, model_path)
    joblib.dump(ALL_FEATURES, cols_path)
    
    # Structure full metrics payload for front-end consumption
    metrics_payload = {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "cv_accuracy_mean": cv_mean,
        "cv_accuracy_std": cv_std,
        "confusion_matrix": {
            "true_negatives": tn,
            "false_positives": fp,
            "false_negatives": fn,
            "true_positives": tp
        },
        "feature_importances": sorted_importances
    }
    
    with open(metrics_path, 'w') as f:
        json.dump(metrics_payload, f, indent=4)
        
    print(f"[+] Serialized model pipeline to: {model_path}")
    print(f"[+] Serialized feature lists to: {cols_path}")
    print(f"[+] Serialized metrics JSON to: {metrics_path}")
    print("[+] Model training pipeline executed successfully!")

if __name__ == "__main__":
    train_and_evaluate()
