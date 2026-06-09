import os
import joblib
import pandas as pd
from ml.pipeline import ALL_FEATURES

class CandidateRecommender:
    def __init__(self):
        current_dir = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(current_dir, 'models', 'placement_pipeline.pkl')
        
        if os.path.exists(model_path):
            self.pipeline = joblib.load(model_path)
            self.model_loaded = True
        else:
            self.pipeline = None
            self.model_loaded = False
            print("[-] Warning: Placement pipeline model not found in models/. Recommender running in degraded mode.")

    def _get_probability(self, profile_dict: dict) -> float:
        """Helper to map a structural dictionary to the trained scikit-learn Pipeline format"""
        if not self.model_loaded:
            # Fallback heuristic if model isn't trained yet
            return 0.50
            
        # Structure the input dictionary to match ML pipeline feature columns
        # Map inputs back to feature names (lowercase columns)
        input_data = {col: 0 for col in ALL_FEATURES}
        
        input_data['cgpa'] = float(profile_dict.get('cgpa', 7.0))
        input_data['coding_score'] = int(profile_dict.get('coding_score', 60))
        input_data['internships'] = int(profile_dict.get('internships', 0))
        input_data['projects_count'] = int(profile_dict.get('projects_count', 0))
        input_data['certifications_count'] = int(profile_dict.get('certifications_count', 0))
        
        work_ex = profile_dict.get('work_experience', 'No')
        input_data['work_ex_bin'] = 1 if work_ex == 'Yes' else 0
        
        # Skill mappings
        skills_str = str(profile_dict.get('skills', []))
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
        
        for clean_name, db_col in skill_mapping.items():
            if clean_name in skills_str:
                input_data[db_col] = 1
                
        # Query Model
        df_input = pd.DataFrame([input_data])[ALL_FEATURES]
        prob = self.pipeline.predict_proba(df_input)[0][1]
        return float(prob)

    def generate_recommendations(self, profile: dict) -> list:
        """
        Runs mathematical 'what-if' simulations on the candidate profile
        to discover and rank precise recommendations with predicted yield deltas.
        """
        if not self.model_loaded:
            return [{"action": "Activate database and train model pipeline to receive AI insights.", "yield_boost": 0.0}]
            
        base_prob = self._get_probability(profile)
        
        # If candidate already has nearly 95%+ probability, advise on behavioral polish
        if base_prob >= 0.95:
            return [{
                "action": "Exemplary Profile! You are highly competitive. Focus on mock behavioral interviews, core DSA speed, and company culture research.",
                "yield_boost": 0.0,
                "type": "polish"
            }]
            
        simulations = []
        skills_pool = ['Python', 'SQL', 'Java', 'Data Analysis', 'Web Dev', 'Machine Learning', 'Cloud', 'React']
        current_skills = profile.get('skills', [])
        if isinstance(current_skills, str):
            current_skills = [s.strip() for s in current_skills.split(",") if s.strip() and s.strip() != "None"]
            
        # 1. Simulate Skill Additions
        for skill in skills_pool:
            if skill not in current_skills:
                # Clone profile and add skill
                sim_profile = profile.copy()
                sim_skills = list(current_skills) + [skill]
                sim_profile['skills'] = sim_skills
                
                sim_prob = self._get_probability(sim_profile)
                delta = sim_prob - base_prob
                
                if delta > 0.005:
                    simulations.append({
                        "action": f"Acquire practical proficiency in '{skill}'",
                        "yield_boost": round(delta * 100, 1),
                        "type": "skill"
                    })
                    
        # 2. Simulate Internship Addition
        current_internships = int(profile.get('internships', 0))
        if current_internships < 3:
            sim_profile = profile.copy()
            sim_profile['internships'] = current_internships + 1
            
            sim_prob = self._get_probability(sim_profile)
            delta = sim_prob - base_prob
            
            if delta > 0.005:
                simulations.append({
                    "action": f"Secure and complete one additional technical internship (Current: {current_internships} -> Target: {current_internships + 1})",
                    "yield_boost": round(delta * 100, 1),
                    "type": "experience"
                })
                
        # 3. Simulate Project Addition
        current_projects = int(profile.get('projects_count', 0))
        if current_projects < 5:
            sim_profile = profile.copy()
            sim_profile['projects_count'] = current_projects + 1
            
            sim_prob = self._get_probability(sim_profile)
            delta = sim_prob - base_prob
            
            if delta > 0.005:
                simulations.append({
                    "action": f"Build one additional portfolio technical project (Current: {current_projects} -> Target: {current_projects + 1})",
                    "yield_boost": round(delta * 100, 1),
                    "type": "projects"
                })
                
        # 4. Simulate Coding Score Improvement
        current_coding = int(profile.get('coding_score', 50))
        if current_coding < 85:
            sim_profile = profile.copy()
            sim_profile['coding_score'] = min(100, current_coding + 15)
            
            sim_prob = self._get_probability(sim_profile)
            delta = sim_prob - base_prob
            
            if delta > 0.005:
                simulations.append({
                    "action": f"Practice HackerRank/LeetCode problems to increase coding index (Current: {current_coding} -> Target: {min(100, current_coding + 15)})",
                    "yield_boost": round(delta * 100, 1),
                    "type": "skills"
                })
                
        # Sort recommendations by yield boost descending
        sorted_simulations = sorted(simulations, key=lambda x: x['yield_boost'], reverse=True)
        
        # Return top 4 most impactful recommendations
        return sorted_simulations[:4]

if __name__ == "__main__":
    recommender = CandidateRecommender()
    
    # Test profile with low probability
    test_student = {
        "cgpa": 6.5,
        "coding_score": 52,
        "internships": 0,
        "work_experience": "No",
        "projects_count": 1,
        "certifications_count": 0,
        "skills": ["Java"]
    }
    
    recs = recommender.generate_recommendations(test_student)
    print("Baseline simulated recommendations:")
    for r in recs:
        print(f"- {r['action']} (Boost: +{r['yield_boost']}% Probability)")
