import os
import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules
from sqlalchemy import text
from database.connection import engine

def get_placement_rules():
    """
    Extracts high-impact placement patterns from the PostgreSQL warehouse
    using the Apriori Algorithm. Computes Support, Confidence, and Lift.
    """
    print("[*] Starting Apriori Association Rule Mining...")
    
    # 1. Pull data from database
    query = """
    SELECT 
        s.student_id,
        s.work_experience,
        e.internships,
        t.skill_python, t.skill_sql, t.skill_java, t.skill_data_analysis, 
        t.skill_web_dev, t.skill_ml, t.skill_cloud, t.skill_react,
        f.placement_status
    FROM fact_placements f
    JOIN dim_student s ON f.student_id = s.student_id
    JOIN dim_experience e ON f.student_id = e.student_id
    JOIN dim_skills t ON f.student_id = t.student_id;
    """
    try:
        with engine.connect() as conn:
            df = pd.read_sql(text(query), conn)
    except Exception as e:
        print(f"[-] Database query failed in mining engine: {e}")
        # Fallback to local CSV if database is offline or not configured
        current_dir = os.path.dirname(os.path.abspath(__file__))
        csv_path = os.path.join(current_dir, '..', 'data', 'processed', 'master_student_dataset.csv')
        if os.path.exists(csv_path):
             df = pd.read_csv(csv_path)
             # Adapt CSV column names to schema formats
             df['skill_python'] = df['Skills'].apply(lambda x: 'Python' in str(x))
             df['skill_sql'] = df['Skills'].apply(lambda x: 'SQL' in str(x))
             df['skill_java'] = df['Skills'].apply(lambda x: 'Java' in str(x))
             df['skill_data_analysis'] = df['Skills'].apply(lambda x: 'Data Analysis' in str(x))
             df['skill_web_dev'] = df['Skills'].apply(lambda x: 'Web Dev' in str(x))
             df['skill_ml'] = df['Skills'].apply(lambda x: 'Machine Learning' in str(x))
             df['skill_cloud'] = df['Skills'].apply(lambda x: 'Cloud' in str(x))
             df['skill_react'] = df['Skills'].apply(lambda x: 'React' in str(x))
             df['work_experience'] = df['Work_Exp']
             df['placement_status'] = df['Placement_Status']
        else:
             return []

    # 2. Build boolean matrix representing presence of features
    dataset = []
    skill_mapping = {
        'skill_python': 'Python',
        'skill_sql': 'SQL',
        'skill_java': 'Java',
        'skill_data_analysis': 'Data Analysis',
        'skill_web_dev': 'Web Dev',
        'skill_ml': 'Machine Learning',
        'skill_cloud': 'Cloud',
        'skill_react': 'React'
    }
    
    for _, row in df.iterrows():
        row_data = {}
        
        # Skills
        for db_col, clean_name in skill_mapping.items():
            row_data[clean_name] = bool(row[db_col])
            
        # Core boosting attributes
        row_data['Has Internship'] = int(row['internships']) > 0
        row_data['Has Work Experience'] = row['work_experience'] == 'Yes'
        row_data['Placed'] = row['placement_status'] == 'Placed'
        
        dataset.append(row_data)
        
    df_mining = pd.DataFrame(dataset)
    
    # 3. Apply Apriori
    # Support represents frequency of pattern in whole dataset
    frequent_itemsets = apriori(df_mining, min_support=0.03, use_colnames=True)
    
    if frequent_itemsets.empty:
        print("[-] No frequent itemsets found with min_support=0.03")
        return []
        
    # Generate association rules
    rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.30)
    
    if rules.empty:
        print("[-] No association rules found with min_threshold=0.30")
        return []
        
    # 4. Filter for Placed consequent
    placement_rules = rules[rules['consequents'] == frozenset({'Placed'})]
    
    # Remove empty/obvious rules, and filter rules with lift > 1.0 (positive correlation)
    placement_rules = placement_rules[placement_rules['lift'] > 1.0]
    
    # Sort by Lift (highest statistical leverage first)
    placement_rules = placement_rules.sort_values(by='lift', ascending=False)
    
    # 5. Format strategic rule interpretations
    results = []
    for _, row in placement_rules.head(10).iterrows():
        antecedents = list(row['antecedents'])
        if not antecedents:
            continue
            
        support = float(row['support'])
        confidence = float(row['confidence'])
        lift = float(row['lift'])
        
        # Build logical advice string
        features_str = " + ".join(antecedents)
        lift_percent = round((lift - 1.0) * 100, 1)
        interpretation = (
            f"Candidates with [{features_str}] are {lift_percent}% more likely "
            f"to secure placements compared to the average student base, with a "
            f"placement success probability of {round(confidence * 100, 1)}%."
        )
        
        results.append({
            "antecedents": antecedents,
            "support": round(support, 4),
            "confidence": round(confidence, 3),
            "lift": round(lift, 3),
            "interpretation": interpretation
        })
        
    print(f"[+] Discovered {len(results)} high-confidence strategic talent insights.")
    return results

if __name__ == "__main__":
    rules = get_placement_rules()
    for idx, rule in enumerate(rules[:3]):
        print(f"Rule #{idx+1} (Lift: {rule['lift']}): {rule['interpretation']}")
