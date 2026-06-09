from pydantic import BaseModel, Field
from typing import List, Dict, Optional

class StudentProfile(BaseModel):
    cgpa: float = Field(..., ge=0.0, le=10.0, description="Cumulative Grade Point Average (0-10)")
    coding_score: int = Field(..., ge=0, le=100, description="Technical coding assessment score (0-100)")
    internships: int = Field(..., ge=0, le=5, description="Number of technical internships completed")
    work_experience: str = Field(..., description="Formal full-time work experience ('Yes' or 'No')")
    skills: List[str] = Field(default=[], description="List of technical skills candidate possesses")
    projects_count: int = Field(default=0, ge=0, le=10, description="Number of major technical projects completed")
    certifications_count: int = Field(default=0, ge=0, le=10, description="Number of professional certifications completed")

class RecommendationItem(BaseModel):
    action: str
    yield_boost: float
    type: str

class PredictionResponse(BaseModel):
    placement_status: str
    probability: float
    advice: str
    recommendations: List[RecommendationItem]

class DomainInsights(BaseModel):
    primary_domain: str
    confidence_score: float
    domain_affinities: Dict[str, float]

class ParsedResumeResponse(BaseModel):
    domain_insights: DomainInsights
    skills: List[str]
    cgpa: float
    internships: int
    work_experience: str
    projects_count: int
    certifications_count: int

class AprioriRule(BaseModel):
    antecedents: List[str]
    support: float
    confidence: float
    lift: float
    interpretation: str

class MiningTipsResponse(BaseModel):
    rules: List[AprioriRule]

class WarehouseStats(BaseModel):
    placed_count: int
    not_placed_count: int
    total_records: int
    average_salary: float
    placement_rate: float

class ConfusionMatrix(BaseModel):
    true_negatives: int
    false_positives: int
    false_negatives: int
    true_positives: int

class ModelInfoResponse(BaseModel):
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    cv_accuracy_mean: float
    cv_accuracy_std: float
    confusion_matrix: ConfusionMatrix
    feature_importances: Dict[str, float]
