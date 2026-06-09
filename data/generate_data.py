import pandas as pd
import numpy as np
import random
import os

def generate_student_data(num_students=10000): # <--- INCREASED TO 10,000
    print(f"Generating data for {num_students} students...")
    
    # 1. ACADEMIC SCORES (Normal Distribution)
    # Mean=7.0, StdDev=1.2 (Realistic spread)
    cgpa = np.random.normal(7.0, 1.2, num_students)
    cgpa = np.clip(cgpa, 0, 10) # Keep between 0-10
    cgpa = np.round(cgpa, 2)

    # Degree Percentage usually correlates with CGPA
    degree_p = (cgpa * 9.5) + np.random.normal(0, 5, num_students)
    degree_p = np.clip(degree_p, 40, 100)
    degree_p = np.round(degree_p, 1)

    # 2. EXPERIENCE
    # 40% of students have internships
    internships = np.random.choice([0, 1, 2, 3], size=num_students, p=[0.6, 0.25, 0.1, 0.05])
    
    # 10% have work experience
    work_exp = np.random.choice(['Yes', 'No'], size=num_students, p=[0.10, 0.90])

    # 3. SKILLS (Random Subsets)
    all_skills = ['Python', 'SQL', 'Java', 'Data Analysis', 'Web Dev', 'Machine Learning', 'Cloud', 'React']
    student_skills = []
    
    for _ in range(num_students):
        # Pick 0 to 4 skills randomly
        num_skills = np.random.randint(0, 5)
        chosen = random.sample(all_skills, k=num_skills)
        student_skills.append(", ".join(chosen) if chosen else "None")

    # 4. PLACEMENT LOGIC (The "Ground Truth")
    # We define the rules of the universe here, so the ML model has something to learn.
    placement_status = []
    
    for i in range(num_students):
        score = 0
        
        # Academic Base Score
        if cgpa[i] > 8.0: score += 40
        elif cgpa[i] > 6.5: score += 20
        
        # Skill Bonus
        skills = student_skills[i]
        if "Python" in skills: score += 10
        if "SQL" in skills: score += 5
        if "Data Analysis" in skills and "Python" in skills: score += 15 # Combo Bonus
        if "Web Dev" in skills and "React" in skills: score += 15 # Combo Bonus
        if "Cloud" in skills: score += 10
        if "Machine Learning" in skills: score += 10
        
        # Experience Bonus (Huge Factor)
        score += (internships[i] * 15)
        if work_exp[i] == 'Yes': score += 20
        
        # Random Luck Factor (Interview Performance)
        luck = random.randint(-10, 10)
        final_score = score + luck
        
        # Threshold to get placed
        if final_score >= 65:
            placement_status.append('Placed')
        else:
            placement_status.append('Not Placed')

    # 5. CREATE DATAFRAME
    df = pd.DataFrame({
        'Student_ID': range(1, num_students + 1),
        'CGPA': cgpa,
        'Degree_P': degree_p,
        'Skills': student_skills,
        'Internships': internships,
        'Work_Exp': work_exp,
        'Placement_Status': placement_status
    })

    # 6. SAVE
    # Create folder if not exists
    output_dir = os.path.join(os.path.dirname(__file__), 'processed')
    os.makedirs(output_dir, exist_ok=True)
    
    file_path = os.path.join(output_dir, 'master_student_dataset.csv')
    df.to_csv(file_path, index=False)
    print(f"✅ Successfully saved {num_students} records to: {file_path}")

if __name__ == "__main__":
    generate_student_data()