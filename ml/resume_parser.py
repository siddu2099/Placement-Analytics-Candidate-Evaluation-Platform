import re
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Standard English Stop Words to keep cleaning light and dependency-free
STOP_WORDS = {
    'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 'your', 'yours', 
    'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', 'her', 'hers', 'herself', 
    'it', 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which', 
    'who', 'whom', 'this', 'that', 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 
    'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 
    'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 
    'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during', 'before', 
    'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 
    'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 
    'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 
    'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 
    'will', 'just', 'don', 'should', 'now'
}

# 1. Idealized Domain Corpora mapping
DOMAIN_CORPORA = {
    "Data Science / Analytics": (
        "python sql pandas numpy machine learning statistics scikit learn models analysis "
        "predictive modeling datasets regression classification clustering deep learning tableau "
        "matplotlib seaborn visualization analytics r programming data mining neural networks"
    ),
    "Frontend Development": (
        "javascript react html css frontend nextjs tailwind bootstrap typescript web development "
        "ui ux responsive interactive layout component design css3 visual styling DOM animations"
    ),
    "Backend Development": (
        "node express backend java spring boot sql postgresql mongodb server RESTful API "
        "databases system architecture graphql microservices caching validation docker routing security"
    ),
    "Cloud & DevOps": (
        "aws cloud docker kubernetes devops terraform CI CD pipeline automation linux azure "
        "sysadmin shell scripting deployment serverless monitoring configuration management ansible"
    )
}

class ResumeParser:
    def __init__(self):
        # Setup the TF-IDF Vectorizer fitted on our core domain documents
        self.domains = list(DOMAIN_CORPORA.keys())
        self.documents = list(DOMAIN_CORPORA.values())
        
        self.vectorizer = TfidfVectorizer()
        self.tfidf_matrix = self.vectorizer.fit_transform(self.documents)
        
    def _clean_text(self, text: str) -> str:
        """Lowercases, strips punctuation, and removes common stop words"""
        text = text.lower()
        # Strip special chars & keep alphanumeric spaces
        text = re.sub(r'[^a-z0-9\s#\+\-\.]', ' ', text)
        # Tokenize and filter stop words
        words = [w for w in text.split() if w not in STOP_WORDS]
        return " ".join(words)
        
    def classify_domain(self, cleaned_resume_text: str) -> dict:
        """
        Classifies the resume into one of the core professional domains
        using TF-IDF Cosine Similarity.
        """
        if not cleaned_resume_text.strip():
            return {
                "primary_domain": "General Software Engineering",
                "confidence_score": 0.0,
                "domain_affinities": {d: 0.0 for d in self.domains}
            }
            
        resume_vec = self.vectorizer.transform([cleaned_resume_text])
        similarities = cosine_similarity(resume_vec, self.tfidf_matrix)[0]
        
        affinities = {}
        for domain, score in zip(self.domains, similarities):
            affinities[domain] = float(np.round(score * 100, 1))
            
        max_idx = np.argmax(similarities)
        primary_domain = self.domains[max_idx]
        confidence = affinities[primary_domain]
        
        # If similarity scores are extremely low, default gracefully
        if confidence < 5.0:
            return {
                "primary_domain": "General Software Engineering",
                "confidence_score": 0.0,
                "domain_affinities": affinities
            }
            
        return {
            "primary_domain": primary_domain,
            "confidence_score": confidence,
            "domain_affinities": affinities
        }

    def extract_profile(self, raw_text: str) -> dict:
        """
        Extracts structural profile features from raw resume text
        using highly accurate boundary-checked regex patterns.
        """
        cleaned_text = self._clean_text(raw_text)
        
        # 1. Domain Classification
        domain_insights = self.classify_domain(cleaned_text)
        
        # 2. Skill Extraction (Boundary-checked match against vocabulary)
        skills_vocab = {
            "Python": r'\bpython\b',
            "SQL": r'\bsql\b|\bpostgresql\b|\bmysql\b',
            "Java": r'\bjava\b', # avoids JavaScript matching
            "Data Analysis": r'\bdata\s+analysis\b|\banalytics\b|\bpandas\b',
            "Web Dev": r'\bweb\s+dev\b|\bweb\s+development\b|\bhtml\b|\bcss\b',
            "Machine Learning": r'\bmachine\s+learning\b|\bml\b|\bscikit\b',
            "Cloud": r'\bcloud\b|\baws\b|\bazure\b|\bgcp\b',
            "React": r'\breact\b|\breactjs\b'
        }
        
        extracted_skills = []
        for skill_name, pattern in skills_vocab.items():
            if re.search(pattern, raw_text.lower()):
                extracted_skills.append(skill_name)
                
        # 3. CGPA Extraction
        # Patterns like: "CGPA: 8.5", "CGPA 9.2", "GPA 3.8/4", "8.7 CGPA"
        cgpa_val = 7.5 # Default benchmark if missing
        cgpa_match = re.search(r'\b(cgpa|gpa|pointer)\b\s*(?:of|is|:)?\s*([0-9]\.[0-9]{1,2})', raw_text.lower())
        if cgpa_match:
            try:
                cgpa_val = float(cgpa_match.group(2))
                # Normalize 4.0 scale to 10.0 scale if detected
                if cgpa_val <= 4.0:
                    cgpa_val = round((cgpa_val / 4.0) * 10.0, 2)
            except ValueError:
                pass
        else:
            # Alternate pattern: just digits like "8.5 CGPA"
            alt_match = re.search(r'\b([0-9]\.[0-9]{1,2})\s*(cgpa|gpa)\b', raw_text.lower())
            if alt_match:
                try:
                    cgpa_val = float(alt_match.group(1))
                    if cgpa_val <= 4.0:
                        cgpa_val = round((cgpa_val / 4.0) * 10.0, 2)
                except ValueError:
                    pass
                    
        # 4. Internships count extraction
        # Searches for occurrences of "intern" or "internship"
        internship_count = 0
        intern_matches = re.findall(r'\bintern(?:ship)?\b', raw_text.lower())
        # Logical mapping: each pair of mentions roughly correlates with 1 internship project
        if intern_matches:
            internship_count = min(3, len(intern_matches) // 2 + 1)
            
        # 5. Work Experience (Binary check)
        # Matches typical patterns of formal corporate jobs: "software engineer", "analyst", "developer", "experience"
        work_exp = "No"
        experience_keywords = [
            r'software\s+engineer', r'developer', r'system\s+engineer', 
            r'data\s+analyst', r'years\s+experience', r'full-time', r'work\s+experience'
        ]
        # Make sure it's not matching internship listings
        if "intern" not in raw_text.lower():
            for kw in experience_keywords:
                if re.search(kw, raw_text.lower()):
                    work_exp = "Yes"
                    break
                    
        # 6. Projects Count
        # Matches mentions of "projects" or lists of distinct project sections
        projects_count = 1
        project_sections = re.findall(r'\bproject\b', raw_text.lower())
        if project_sections:
            projects_count = min(5, len(project_sections))
            
        # 7. Certifications Count
        # Looks for "certified", "certification", "aws certified", etc.
        certifications_count = 0
        cert_matches = re.findall(r'\bcertificat(?:e|ion)\b|\bcertified\b', raw_text.lower())
        if cert_matches:
            certifications_count = min(4, len(cert_matches))
            
        return {
            "domain_insights": domain_insights,
            "skills": extracted_skills,
            "cgpa": cgpa_val,
            "internships": internship_count,
            "work_experience": work_exp,
            "projects_count": projects_count,
            "certifications_count": certifications_count
        }

if __name__ == "__main__":
    # Test resume classification
    test_ds_resume = """
    Siddu Ramachandran
    Email: siddu@gmail.com
    OBJECTIVE: Passonate Data Science student seeking ML internship.
    EDUCATION: B.Tech in CSE - CGPA: 8.78
    EXPERIENCE: 
    * Data Analyst Intern at Zen Corp. Cleaned data using Python and SQL. Built predictive models.
    SKILLS: Pandas, NumPy, Python, SQL, Machine Learning, Scikit-Learn, Tableau.
    PROJECTS:
    - Customer Churn classification model using Random Forest.
    - Financial sentiment analysis using Python NLP.
    """
    
    parser = ResumeParser()
    profile = parser.extract_profile(test_ds_resume)
    print("Classified Domain:", profile['domain_insights']['primary_domain'])
    print("Confidence:", profile['domain_insights']['confidence_score'], "%")
    print("Extracted Skills:", profile['skills'])
    print("Parsed CGPA:", profile['cgpa'])
    print("Parsed Internships:", profile['internships'])
    print("Work Experience?:", profile['work_experience'])
