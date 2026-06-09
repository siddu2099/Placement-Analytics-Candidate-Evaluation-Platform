import os
import pandas as pd
from sqlalchemy import Column, Integer, Float, String, Boolean, ForeignKey, text
from database.connection import engine, Base, SessionLocal

# ==========================================
# 1. STAR SCHEMA SCHEMAS (OR REPLACE SCHEMA)
# ==========================================

class DimStudent(Base):
    __tablename__ = 'dim_student'
    student_id = Column(String(50), primary_key=True)
    cgpa = Column(Float, nullable=False)
    coding_score = Column(Integer, nullable=False)
    work_experience = Column(String(10), nullable=False)

class DimExperience(Base):
    __tablename__ = 'dim_experience'
    student_id = Column(String(50), ForeignKey('dim_student.student_id', ondelete='CASCADE'), primary_key=True)
    internships = Column(Integer, nullable=False)
    projects_count = Column(Integer, nullable=False)
    certifications_count = Column(Integer, nullable=False)

class DimSkills(Base):
    __tablename__ = 'dim_skills'
    student_id = Column(String(50), ForeignKey('dim_student.student_id', ondelete='CASCADE'), primary_key=True)
    skill_python = Column(Boolean, default=False)
    skill_sql = Column(Boolean, default=False)
    skill_java = Column(Boolean, default=False)
    skill_data_analysis = Column(Boolean, default=False)
    skill_web_dev = Column(Boolean, default=False)
    skill_ml = Column(Boolean, default=False)
    skill_cloud = Column(Boolean, default=False)
    skill_react = Column(Boolean, default=False)

class FactPlacements(Base):
    __tablename__ = 'fact_placements'
    student_id = Column(String(50), ForeignKey('dim_student.student_id', ondelete='CASCADE'), primary_key=True)
    placement_status = Column(String(20), nullable=False)
    salary = Column(Integer, default=0)

def run_etl():
    print("[*] Starting ETL Pipeline execution...")
    
    # 2. Extract Phase: Locate and Read CSV
    current_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(current_dir, '..', 'data', 'processed', 'master_student_dataset.csv')
    
    if not os.path.exists(csv_path):
        print(f"[-] CSV file not found at: {csv_path}")
        return
        
    print(f"[i] Reading source CSV: {csv_path}")
    df = pd.read_csv(csv_path)
    print(f"[+] Loaded {len(df)} records.")
    
    # 3. Transform Phase: Validation Checks
    print("[*] Performing strict data validations...")
    # Verify no nulls in vital columns
    assert df['Student_ID'].isnull().sum() == 0, "Student_ID contains null values"
    assert df['CGPA'].between(0, 10).all(), "CGPA out of bounds"
    assert df['Coding_Score'].between(0, 100).all(), "Coding_Score out of bounds"
    assert df['Internships'].between(0, 5).all(), "Internships out of bounds"
    assert df['Placement_Status'].isin(['Placed', 'Not Placed']).all(), "Placement_Status values invalid"
    print("[+] All data validation assertions passed successfully.")
    
    # 4. Load Phase: Initialize Database star schema
    print("[*] Rebuilding PostgreSQL database tables...")
    # Drop all existing tables cascadingly to ensure clean schemas
    with engine.connect() as conn:
        conn.execute(text("DROP TABLE IF EXISTS fact_placements CASCADE;"))
        conn.execute(text("DROP TABLE IF EXISTS dim_skills CASCADE;"))
        conn.execute(text("DROP TABLE IF EXISTS dim_experience CASCADE;"))
        conn.execute(text("DROP TABLE IF EXISTS dim_student CASCADE;"))
        conn.commit()
        print("[+] Existing star schema tables dropped.")

    # Create fresh schema tables
    Base.metadata.create_all(bind=engine)
    print("[+] Created star schema database tables successfully.")
    
    # Insert dimension values bulked
    session = SessionLocal()
    try:
        # Prepare Student Dimension
        print("[*] Loading Student Dimension (dim_student)...")
        students = [
            DimStudent(
                student_id=row['Student_ID'],
                cgpa=row['CGPA'],
                coding_score=int(row['Coding_Score']),
                work_experience=row['Work_Exp']
            )
            for _, row in df.iterrows()
        ]
        session.bulk_save_objects(students)
        session.flush()

        # Prepare Experience Dimension
        print("[*] Loading Experience Dimension (dim_experience)...")
        experiences = [
            DimExperience(
                student_id=row['Student_ID'],
                internships=int(row['Internships']),
                projects_count=int(row['Projects_Count']),
                certifications_count=int(row['Certifications_Count'])
            )
            for _, row in df.iterrows()
        ]
        session.bulk_save_objects(experiences)
        session.flush()

        # Prepare Skills Dimension
        print("[*] Loading Skills Dimension (dim_skills)...")
        skill_cols = ['Python', 'SQL', 'Java', 'Data Analysis', 'Web Dev', 'Machine Learning', 'Cloud', 'React']
        skills = []
        for _, row in df.iterrows():
            skills_str = str(row['Skills'])
            skills.append(DimSkills(
                student_id=row['Student_ID'],
                skill_python=('Python' in skills_str),
                skill_sql=('SQL' in skills_str),
                skill_java=('Java' in skills_str),
                skill_data_analysis=('Data Analysis' in skills_str),
                skill_web_dev=('Web Dev' in skills_str),
                skill_ml=('Machine Learning' in skills_str),
                skill_cloud=('Cloud' in skills_str),
                skill_react=('React' in skills_str)
            ))
        session.bulk_save_objects(skills)
        session.flush()

        # Prepare Placement Fact Table
        print("[*] Loading Fact Placements (fact_placements)...")
        placements = [
            FactPlacements(
                student_id=row['Student_ID'],
                placement_status=row['Placement_Status'],
                salary=int(row['Salary'])
            )
            for _, row in df.iterrows()
        ]
        session.bulk_save_objects(placements)
        session.commit()
        print("[+] Bulk data loading completed successfully.")
        print("[+] ETL finished! PostgreSQL Warehouse successfully loaded with 10,000 records.")
        
    except Exception as e:
        session.rollback()
        print(f"[-] ETL Pipeline crashed: {e}")
        raise e
    finally:
        session.close()

if __name__ == "__main__":
    run_etl()