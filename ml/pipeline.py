from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier

# Define explicit feature categories for explainable modeling
NUMERIC_FEATURES = ['cgpa', 'coding_score', 'internships', 'projects_count', 'certifications_count']
CATEGORICAL_BINARY_FEATURES = [
    'work_ex_bin',
    'skill_python',
    'skill_sql',
    'skill_java',
    'skill_data_analysis',
    'skill_web_dev',
    'skill_ml',
    'skill_cloud',
    'skill_react'
]

ALL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_BINARY_FEATURES

def create_production_pipeline():
    """
    Creates the scikit-learn classification pipeline.
    Standardizes continuous numeric features while keeping categorical binary columns as-is.
    Uses an optimized Random Forest Classifier.
    """
    # 1. Preprocessor configuration
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), NUMERIC_FEATURES)
        ],
        remainder='passthrough' # Mappings like categorical binaries are left untouched
    )
    
    # 2. Complete Model Pipeline
    pipeline = Pipeline(
        steps=[
            ('preprocessor', preprocessor),
            ('classifier', RandomForestClassifier(
                n_estimators=120,
                max_depth=12,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42,
                n_jobs=-1
            ))
        ]
    )
    
    return pipeline
