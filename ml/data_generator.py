import pandas as pd
import numpy as np
import random
import os

# Set seed for reproducibility
np.random.seed(42)
random.seed(42)

def generate_placement_dataset(num_students=10000):
    print(f"[*] Generating statistically realistic dataset for {num_students} candidates...")
    
    # 1. CGPA: Normal distribution centered around 7.2 with std dev of 1.1, clipped to realistic bounds [5.0, 10.0]
    cgpa = np.random.normal(7.2, 1.1, num_students)
    cgpa = np.clip(cgpa, 5.0, 10.0)
    cgpa = np.round(cgpa, 2)
    
    # 2. Coding Score: HackerRank/LeetCode score [0-100], slightly correlated with CGPA
    # Base score is normal centered at 55 + correlation with CGPA
    coding_base = np.random.normal(55, 12, num_students)
    cgpa_booster = (cgpa - 7.2) * 8
    coding_score = coding_base + cgpa_booster
    coding_score = np.clip(coding_score, 0, 100)
    coding_score = np.round(coding_score, 0).astype(int)
    
    # 3. Internships: 0 to 3, heavily skewed towards 0 and 1
    internships = np.random.choice([0, 1, 2, 3], size=num_students, p=[0.55, 0.30, 0.11, 0.04])
    
    # 4. Work Experience: 'Yes' (15% base probability) or 'No' (85%)
    work_exp = np.random.choice(['Yes', 'No'], size=num_students, p=[0.15, 0.85])
    
    # 5. Technical Projects: 0 to 5 major projects
    projects_count = np.random.choice([0, 1, 2, 3, 4, 5], size=num_students, p=[0.15, 0.35, 0.25, 0.15, 0.07, 0.03])
    
    # 6. Professional Certifications: 0 to 4
    certifications_count = np.random.choice([0, 1, 2, 3, 4], size=num_students, p=[0.40, 0.35, 0.15, 0.08, 0.02])
    
    # 7. Skillset Generation: Pool of 8 standard developer/analyst skills
    skills_pool = ['Python', 'SQL', 'Java', 'Data Analysis', 'Web Dev', 'Machine Learning', 'Cloud', 'React']
    
    student_skills = []
    for i in range(num_students):
        # A student's capacity to have technical skills correlates with CGPA and coding score
        score_factor = (cgpa[i] / 10.0) * 0.4 + (coding_score[i] / 100.0) * 0.6
        
        # Decide how many skills they have based on their capacity factor (0 to 4)
        max_skills = int(np.clip(np.random.normal(score_factor * 5, 1.2), 0, 5))
        
        if max_skills == 0:
            student_skills.append("None")
        else:
            # High coding scores increase chance of Machine Learning, React, and Cloud
            skill_weights = [0.18, 0.18, 0.15, 0.12, 0.15, 0.08, 0.07, 0.07]
            if coding_score[i] > 80:
                skill_weights = [0.12, 0.12, 0.10, 0.10, 0.10, 0.18, 0.13, 0.15]
            
            # Pick non-repeating skills
            chosen = np.random.choice(skills_pool, size=min(max_skills, len(skills_pool)), replace=False, p=skill_weights)
            student_skills.append(", ".join(chosen))
            
    # 8. Placement Logic (Mathematically Defensible Logit Formula)
    placement_status = []
    salary = []
    
    for i in range(num_students):
        skills_str = student_skills[i]
        skills_list = [s.strip() for s in skills_str.split(",") if s.strip() != "None"]
        skills_count = len(skills_list)
        
        # Build composite weighted score representing employability
        # CGPA (Max 18 pts), Coding Score (Max 5 pts), Internships (Max 4.5 pts), WorkEx (2.0 pts), Projects (Max 6.0 pts)
        score = (
            1.8 * cgpa[i] + 
            0.05 * coding_score[i] + 
            1.5 * internships[i] + 
            2.0 * (1.0 if work_exp[i] == 'Yes' else 0.0) + 
            1.2 * projects_count[i] + 
            0.8 * certifications_count[i] + 
            0.5 * skills_count - 
            18.5
        )
        
        # Sigmoid probability mapping
        probability = 1.0 / (1.0 + np.exp(-score))
        
        # Add realistic interview performance random noise (Standard dev = 0.06)
        noise = np.random.normal(0, 0.06)
        final_probability = np.clip(probability + noise, 0.0, 1.0)
        
        if final_probability >= 0.50:
            placement_status.append('Placed')
            
            # Base salary: 350k. Add increments based on specific skill weights and stats
            base_sal = 350000
            cgpa_premium = max(0, (cgpa[i] - 6.0) * 45000)
            coding_premium = coding_score[i] * 3500
            internship_premium = internships[i] * 60000
            
            # Combo bonuses
            skill_premium = 0
            if "Machine Learning" in skills_str and "Python" in skills_str:
                skill_premium += 40000
            if "React" in skills_str and "Web Dev" in skills_str:
                skill_premium += 30000
            skill_premium += skills_count * 15000
            
            total_salary = int(base_sal + cgpa_premium + coding_premium + internship_premium + skill_premium)
            # Add market variance
            total_salary = int(total_salary * np.random.uniform(0.92, 1.08))
            salary.append(total_salary)
        else:
            placement_status.append('Not Placed')
            salary.append(0)

    # 9. Create DataFrame
    df = pd.DataFrame({
        'Student_ID': [f"STU_{10000 + i}" for i in range(num_students)],
        'CGPA': cgpa,
        'Coding_Score': coding_score,
        'Skills': student_skills,
        'Internships': internships,
        'Work_Exp': work_exp,
        'Projects_Count': projects_count,
        'Certifications_Count': certifications_count,
        'Placement_Status': placement_status,
        'Salary': salary
    })

    # Save to c:\Users\siddu\Pictures\placement-prediction-system\data\processed
    current_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(current_dir, '..', 'data', 'processed')
    os.makedirs(output_dir, exist_ok=True)
    
    file_path = os.path.join(output_dir, 'master_student_dataset.csv')
    df.to_csv(file_path, index=False)
    
    print(f"[+] Statistical Synthesis complete! Generated and saved {num_students} student profiles to:\n   {file_path}")
    print(f"[i] Placement Rate: {df['Placement_Status'].value_counts(normalize=True).get('Placed', 0) * 100:.2f}%")
    print(f"[i] Avg Salary (Placed): INR {df[df['Salary'] > 0]['Salary'].mean():,.2f}")
    return df

if __name__ == "__main__":
    generate_placement_dataset()
