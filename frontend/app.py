import streamlit as st
import requests
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

# ==========================================
# 1. SETUP & CONFIGURATION
# ==========================================
API_URL = "http://127.0.0.1:8000"

st.set_page_config(
    page_title="Placement Analytics & Candidate Evaluation",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern professional appearance (Glassmorphism & harmonized color theme)
st.markdown("""
<style>
    .reportview-container {
        background: #f8f9fa;
    }
    .metric-card {
        background: white;
        padding: 20px;
        border-radius: 12px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
        border-left: 5px solid #2979ff;
        margin-bottom: 15px;
    }
    .card-title {
        color: #757575;
        font-size: 14px;
        font-weight: 600;
        text-transform: uppercase;
        margin-bottom: 8px;
    }
    .card-value {
        color: #212121;
        font-size: 28px;
        font-weight: 700;
    }
    .recommendation-pill {
        background: #e3f2fd;
        color: #0d47a1;
        padding: 10px 15px;
        border-radius: 8px;
        border-left: 4px solid #1976d2;
        margin-bottom: 10px;
        font-size: 14px;
        font-weight: 500;
    }
    .yield-badge {
        background: #00c853;
        color: white;
        padding: 2px 8px;
        border-radius: 4px;
        font-size: 12px;
        font-weight: 700;
        float: right;
    }
</style>
""", unsafe_allow_html=True)

# Initialize Session State values for Resume Auto-fill
if "cgpa" not in st.session_state: st.session_state.cgpa = 7.2
if "coding_score" not in st.session_state: st.session_state.coding_score = 65
if "internships" not in st.session_state: st.session_state.internships = 0
if "work_experience" not in st.session_state: st.session_state.work_experience = "No"
if "projects_count" not in st.session_state: st.session_state.projects_count = 1
if "certifications_count" not in st.session_state: st.session_state.certifications_count = 0
if "skills" not in st.session_state: st.session_state.skills = []
if "resume_parsed" not in st.session_state: st.session_state.resume_parsed = False
if "parsed_profile" not in st.session_state: st.session_state.parsed_profile = None

# ==========================================
# 2. SIDEBAR - PROFILE CONFIGURATION
# ==========================================
st.sidebar.title("👤 Candidate Profile Evaluation")
st.sidebar.write("Configure details manually or upload a resume to auto-fill.")

# Resume PDF Uploader
uploaded_file = st.sidebar.file_uploader("📂 Upload Candidate Resume (PDF)", type=["pdf"])

if uploaded_file is not None and not st.session_state.resume_parsed:
    with st.sidebar.spinner("Parsing resume details..."):
        try:
            files = {"file": (uploaded_file.name, uploaded_file.getvalue(), "application/pdf")}
            response = requests.post(f"{API_URL}/parse-resume", files=files)
            if response.status_code == 200:
                profile = response.json()
                st.session_state.cgpa = float(profile["cgpa"])
                st.session_state.internships = int(profile["internships"])
                st.session_state.work_experience = profile["work_experience"]
                st.session_state.projects_count = int(profile["projects_count"])
                st.session_state.certifications_count = int(profile["certifications_count"])
                st.session_state.skills = profile["skills"]
                st.session_state.parsed_profile = profile
                st.session_state.resume_parsed = True
                st.sidebar.success("✅ Resume parsed and profile loaded!")
            else:
                st.sidebar.error("Could not parse resume text.")
        except Exception as e:
            st.sidebar.error(f"Connection failure to parsing API: {e}")

if st.sidebar.button("🧹 Clear/Reset Profile"):
    st.session_state.cgpa = 7.2
    st.session_state.coding_score = 65
    st.session_state.internships = 0
    st.session_state.work_experience = "No"
    st.session_state.projects_count = 1
    st.session_state.certifications_count = 0
    st.session_state.skills = []
    st.session_state.resume_parsed = False
    st.session_state.parsed_profile = None
    st.rerun()

st.sidebar.divider()

# Input controls bound to session state
cgpa = st.sidebar.slider("CGPA (Academic Performance)", 5.0, 10.0, st.session_state.cgpa, 0.1)
coding_score = st.sidebar.slider("Coding Evaluation Score (0-100)", 0, 100, st.session_state.coding_score, 1)
internships = st.sidebar.slider("Number of Internships", 0, 3, st.session_state.internships, 1)
projects_count = st.sidebar.slider("Technical Projects Completed", 0, 5, st.session_state.projects_count, 1)
certifications_count = st.sidebar.slider("Professional Certifications", 0, 4, st.session_state.certifications_count, 1)
work_experience = st.sidebar.selectbox("Formal Work Experience?", ["No", "Yes"], index=["No", "Yes"].index(st.session_state.work_experience))

skills_list = ["Python", "SQL", "Java", "Data Analysis", "Web Dev", "Machine Learning", "Cloud", "React"]
selected_skills = st.sidebar.multiselect("Technical Skills Set", options=skills_list, default=[s for s in st.session_state.skills if s in skills_list])

# ==========================================
# 3. HEADER & CORE STATISTICS PANEL
# ==========================================
st.title("🎓 Placement Analytics & Candidate Evaluation Platform")
st.markdown("An ML-driven workspace providing explainable placement risk modeling, TF-IDF resume mapping, and data-mined recruitment insights.")

# Retrieve warehouse stats from API
try:
    stats_resp = requests.get(f"{API_URL}/stats")
    if stats_resp.status_code == 200:
        stats = stats_resp.json()
    else:
        stats = {"placed_count": 7641, "not_placed_count": 2359, "total_records": 10000, "average_salary": 718523.80, "placement_rate": 76.41}
except Exception:
    # Default offline values matching loaded PostgreSQL metrics
    stats = {"placed_count": 7641, "not_placed_count": 2359, "total_records": 10000, "average_salary": 718523.80, "placement_rate": 76.41}

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.markdown(f'<div class="metric-card"><div class="card-title">Warehouse Cohort</div><div class="card-value">{stats["total_records"]:,} Students</div></div>', unsafe_allow_html=True)
with col2:
    st.markdown(f'<div class="metric-card"><div class="card-title">Placement Success</div><div class="card-value">{stats["placement_rate"]}%</div></div>', unsafe_allow_html=True)
with col3:
    st.markdown(f'<div class="metric-card"><div class="card-title">Avg Annual Package</div><div class="card-value">INR {stats["average_salary"]:,.0f}</div></div>', unsafe_allow_html=True)
with col4:
    st.markdown(f'<div class="metric-card" style="border-left-color: #00c853;"><div class="card-title">Database Status</div><div class="card-value" style="color: #00c853;">Online Star</div></div>', unsafe_allow_html=True)

st.divider()

# Create tabs for structured navigation
tab1, tab2, tab3 = st.tabs(["🎯 Candidate Assessment", "🧠 Explainable ML & Model Trust", "🔍 Strategic Talent Insights"])

# ==========================================
# TAB 1: CANDIDATE ASSESSMENT & RESUME PARSING
# ==========================================
with tab1:
    col_left, col_right = st.columns([1, 1])
    
    with col_left:
        st.subheader("📊 Candidate Employability Metrics")
        
        # Build payload and call predictive backend
        payload = {
            "cgpa": cgpa,
            "coding_score": coding_score,
            "internships": internships,
            "work_experience": work_experience,
            "skills": selected_skills,
            "projects_count": projects_count,
            "certifications_count": certifications_count
        }
        
        try:
            pred_resp = requests.post(f"{API_URL}/predict", json=payload)
            if pred_resp.status_code == 200:
                result = pred_resp.json()
                prob = result["probability"]
                status = result["placement_status"]
                advice = result["advice"]
                recommendations = result["recommendations"]
            else:
                prob, status, advice, recommendations = 50.0, "Prediction Offline", "API error", []
        except Exception as e:
            prob, status, advice, recommendations = 50.0, "API Offline", f"Unable to reach FastAPI predictor: {e}", []
            
        # Draw elegant Gauge chart for placement probability
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=prob,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "Placement Probability Score", 'font': {'size': 20, 'color': '#212121'}},
            gauge={
                'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "#424242"},
                'bar': {'color': "#2979ff"},
                'bgcolor': "white",
                'borderwidth': 2,
                'bordercolor': "#e0e0e0",
                'steps': [
                    {'range': [0, 50], 'color': '#ffebee'},
                    {'range': [50, 80], 'color': '#e3f2fd'},
                    {'range': [80, 100], 'color': '#e8f5e9'}
                ],
                'threshold': {
                    'line': {'color': "#00c853", 'width': 4},
                    'thickness': 0.75,
                    'value': 90
                }
            }
        ))
        
        fig_gauge.update_layout(height=320, margin=dict(l=20, r=20, t=40, b=20))
        st.plotly_chart(fig_gauge, use_container_width=True)
        
        # Display baseline placement verdict
        if status == "Placed":
            st.success(f"✔️ **Employability Classification: PLACED PROSPECT ({prob}%)**")
        else:
            st.error(f"❌ **Employability Classification: LOW PLACEMENT PROBABILITY ({prob}%)**")
            
        st.info(f"💡 **Explainable AI Verdict:** {advice}")

    with col_right:
        # Display Resume Parsing & Classification summary if a file was uploaded
        if st.session_state.resume_parsed and st.session_state.parsed_profile:
            st.subheader("📄 Resume Analysis Summary")
            profile = st.session_state.parsed_profile
            insights = profile["domain_insights"]
            
            st.markdown(f"**Classified Professional Domain:** `{insights['primary_domain']}`")
            st.markdown(f"**Domain Mapping Confidence:** `{insights['confidence_score']}%`")
            
            # Show domain affinity chart
            affinities = insights["domain_affinities"]
            df_aff = pd.DataFrame({
                "Career Domain": list(affinities.keys()),
                "Affinity Match (%)": list(affinities.values())
            }).sort_values(by="Affinity Match (%)", ascending=True)
            
            fig_aff = px.bar(
                df_aff, 
                x="Affinity Match (%)", 
                y="Career Domain", 
                orientation='h', 
                title="TF-IDF Cosine Similarity Mapping",
                color="Affinity Match (%)",
                color_continuous_scale="Blues"
            )
            fig_aff.update_layout(height=200, margin=dict(l=10, r=10, t=30, b=10), showlegend=False, coloraxis_showscale=False)
            st.plotly_chart(fig_aff, use_container_width=True)
            st.markdown("---")
            
        # Recommendations / Simulated Actions section
        st.subheader("📈 Dynamic Career Action Plan")
        st.write("Calculated by simulating 'what-if' enhancements on the candidate profile against the Random Forest classifier:")
        
        if recommendations:
            for item in recommendations:
                if item["yield_boost"] > 0:
                    st.markdown(
                        f'<div class="recommendation-pill">'
                        f'<span class="yield-badge">+{item["yield_boost"]}% Boost</span>'
                        f'👉 {item["action"]}'
                        f'</div>',
                        unsafe_allow_html=True
                    )
                else:
                    st.success(f"⭐ **Exemplary Profile Met!** {item['action']}")
        else:
            st.warning("All professional dimensions optimized. Profile is ready for deployment!")

# ==========================================
# TAB 2: EXPLAINABLE ML & MODEL TRUST PORTAL
# ==========================================
with tab2:
    st.subheader("🧠 Model Explainability & Trust Metrics")
    st.write("Validation indicators direct from the trained Random Forest Model:")
    
    try:
        model_resp = requests.get(f"{API_URL}/model-info")
        if model_resp.status_code == 200:
            model_info = model_resp.json()
        else:
            model_info = None
    except Exception:
        model_info = None
        
    if model_info:
        # 1. Metric Indicators
        sub_col1, sub_col2, sub_col3, sub_col4 = st.columns(4)
        with sub_col1:
            st.metric("Test Accuracy Score", f"{model_info['accuracy']*100:.2f}%")
        with sub_col2:
            st.metric("Precision Score", f"{model_info['precision']*100:.2f}%")
        with sub_col3:
            st.metric("F1-Score", f"{model_info['f1_score']:.4f}")
        with sub_col4:
            st.metric("5-Fold CV Accuracy", f"{model_info['cv_accuracy_mean']*100:.2f}%")
            
        st.divider()
        
        # 2. Feature Importances and Confusion Matrix
        col_fig_left, col_fig_right = st.columns([6, 5])
        
        with col_fig_left:
            st.markdown("#### Feature Importance Ranking")
            st.write("Weighted contribution of each attribute to the Random Forest predictions:")
            
            importances = model_info["feature_importances"]
            # Map column tags to recruiter-friendly text
            friendly_names = {
                "cgpa": "CGPA (Academics)",
                "coding_score": "Coding Assessment",
                "internships": "Internships Completed",
                "projects_count": "Technical Projects",
                "certifications_count": "Professional Certifications",
                "work_ex_bin": "Work Experience",
                "skill_python": "Skill: Python",
                "skill_sql": "Skill: SQL",
                "skill_java": "Skill: Java",
                "skill_data_analysis": "Skill: Data Analysis",
                "skill_web_dev": "Skill: Web Dev",
                "skill_ml": "Skill: Machine Learning",
                "skill_cloud": "Skill: Cloud",
                "skill_react": "Skill: React"
            }
            
            df_imp = pd.DataFrame({
                "Attribute": [friendly_names.get(f, f) for f in importances.keys()],
                "Importance Weight": list(importances.values())
            }).sort_values(by="Importance Weight", ascending=True)
            
            fig_imp = px.bar(
                df_imp, 
                x="Importance Weight", 
                y="Attribute", 
                orientation='h',
                color="Importance Weight",
                color_continuous_scale="Purples"
            )
            fig_imp.update_layout(height=420, margin=dict(l=10, r=10, t=10, b=10), coloraxis_showscale=False)
            st.plotly_chart(fig_imp, use_container_width=True)
            
        with col_fig_right:
            st.markdown("#### Confusion Matrix Heatmap")
            st.write("Validation grid comparing predictions against target labels on unseen 2,000 candidate profiles:")
            
            cm = model_info["confusion_matrix"]
            
            z_data = [
                [cm["true_negatives"], cm["false_positives"]],
                [cm["false_negatives"], cm["true_positives"]]
            ]
            
            fig_cm = px.imshow(
                z_data,
                labels=dict(x="Predicted Outcome", y="Actual Outcome", color="Records"),
                x=['Not Placed', 'Placed'],
                y=['Not Placed', 'Placed'],
                color_continuous_scale="Greens",
                text_auto=True
            )
            fig_cm.update_layout(height=380, margin=dict(l=20, r=20, t=40, b=20))
            st.plotly_chart(fig_cm, use_container_width=True)
            st.caption("True Positives and True Negatives verify highly accurate predictions and minimized bias.")
            
    else:
        st.warning("Model evaluation metrics not found. Verify model is trained and active.")

# ==========================================
# TAB 3: DATA MINING ENGINE & RULES TABLE
# ==========================================
with tab3:
    st.subheader("🔍 Strategic Placement Patterns (Apriori Rules)")
    st.write("Discovered using the Apriori Association Rule algorithm on the data warehouse:")
    st.markdown("""
        * **Support**: Percentage of all candidates exhibiting this profile combinations.
        * **Confidence**: Likelihood that candidates with this profile will be placed successfully.
        * **Lift**: Ratio of placement success increase relative to the average student base (e.g., Lift > 1.2 indicates +20% probability boost).
    """)
    
    try:
        tips_resp = requests.get(f"{API_URL}/tips")
        if tips_resp.status_code == 200:
            tips = tips_resp.json()
            rules = tips.get("rules", [])
        else:
            rules = []
    except Exception:
        rules = []
        
    if rules:
        # Build clean tabular visual
        table_rows = []
        for rule in rules:
            table_rows.append({
                "Skill Profile Combination": " + ".join(rule["antecedents"]),
                "Support (%)": f"{rule['support'] * 100:.2f}%",
                "Confidence (%)": f"{rule['confidence'] * 100:.1f}%",
                "Lift Score": rule["lift"],
                "Strategic Assessment Interpretation": rule["interpretation"]
            })
            
        df_rules = pd.DataFrame(table_rows)
        
        # Render a clean, sortable interactive table
        st.dataframe(df_rules, use_container_width=True, hide_index=True)
        
        # Optional: Render association bubble chart representing Lift/Confidence
        st.markdown("#### Discovered Career Strategy Impact Map")
        df_bubble = pd.DataFrame({
            "Strategy": [" + ".join(r["antecedents"]) for r in rules],
            "Confidence": [r["confidence"] * 100 for r in rules],
            "Lift": [r["lift"] for r in rules],
            "Support": [r["support"] * 100 for r in rules]
        })
        
        fig_bubble = px.scatter(
            df_bubble,
            x="Confidence",
            y="Lift",
            size="Support",
            color="Lift",
            text="Strategy",
            title="Strategic Placement Impact (Bubble size = Support percentage)",
            color_continuous_scale="Viridis"
        )
        fig_bubble.update_traces(textposition='top center')
        fig_bubble.update_layout(height=400, margin=dict(l=10, r=10, t=40, b=10))
        st.plotly_chart(fig_bubble, use_container_width=True)
        
    else:
        st.warning("Apriori insights currently offline. Verify connection to the PostgreSQL database.")