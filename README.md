# 🎓 ML-Driven Placement Analytics & Candidate Evaluation Platform

An end-to-end Machine Learning, Data Engineering, and Predictive Analytics prototype platform designed to evaluate candidate employability, classify career domains using NLP, and discover strategic success rules.

Built around a **PostgreSQL Data Warehouse (Star Schema)**, this system integrates a **Random Forest Classification Pipeline**, a **TF-IDF + Cosine Similarity Resume Domain Classifier**, an **Apriori Association Engine**, and a dynamic mathematical **"What-If" Recommender**, served via a **FastAPI** backend and recruiter-friendly **Streamlit** dashboard.

---

## 📂 Project Architecture

```text
placement-prediction-system/
│
├── api/
│   ├── main.py             # FastAPI REST service & route wiring
│   └── schemas.py          # Pydantic validation schemas
│
├── ml/
│   ├── data_generator.py   # Statistical dataset synthesizer (10k records)
│   ├── pipeline.py         # SKLearn column preprocessors & model Pipeline
│   ├── train.py            # Training pipeline (ETL -> Train -> CV -> Eval -> Save)
│   ├── mining_engine.py    # Apriori Association Engine (Support, Confidence, Lift)
│   ├── resume_parser.py    # TF-IDF Cosine Similarity & regex skill extractor
│   ├── recommender.py      # "What-If" dynamic career plan simulator
│   └── models/             # Serialized sklearn pipeline & validation logs
│
├── database/
│   ├── connection.py       # SQLAlchemy connection pool manager & env config
│   └── etl_load.py         # Star Schema builder & bulk load pipeline
│
├── frontend/
│   └── app.py              # Recruiter-facing Streamlit dashboard
│
├── .env.example            # Environment variable template
└── requirements.txt        # Project dependencies
```

---

## 🛠 Tech Stack

* **Machine Learning:** Scikit-Learn, Pandas, NumPy, Joblib
* **Data Mining:** MLxtend (Apriori Algorithm)
* **Database & ORM:** PostgreSQL, SQLAlchemy, Psycopg2-binary
* **Backend:** FastAPI, Uvicorn, Pydantic, PyPDF
* **Frontend UI:** Streamlit, Plotly Express

---

## 📈 Database Schema (Star Schema)

The database utilizes a clean **Star Schema** relational design in PostgreSQL to optimize query performance and isolate metrics:

* **`dim_student` (Dimension Table):** Contains academic markers (`cgpa`, `coding_score`) and categorical descriptors (`work_experience`).
* **`dim_experience` (Dimension Table):** Houses extracurricular performance metrics (`internships`, `projects_count`, `certifications_count`).
* **`dim_skills` (Dimension Table):** A one-hot encoded boolean matrix of technical proficiencies (`skill_python`, `skill_sql`, `skill_react`, etc.).
* **`fact_placements` (Fact Table):** Acts as the central anchor mapping foreign keys to target variables (`placement_status`, `salary`).

---

## 🚀 Execution & Run Guide

### 1. Database Initialization & Environment Configuration
Ensure your local PostgreSQL server is active, then create the database:
```sql
CREATE DATABASE placement_db;
```

Copy the example environment file and configure your local credentials:
```bash
cp .env.example .env
```

Set your database configuration inside `.env`:
```env
DB_USER=postgres
DB_PASS=your_secure_password
DB_HOST=localhost
DB_PORT=5432
DB_NAME=placement_db
```
*(Alternatively, you can provide a full `DATABASE_URL=postgresql+psycopg2://user:pass@host:5432/dbname`.)*

> **Security Note:** Never commit `.env` or hardcode credentials into source code or documentation. Always keep secrets in environment variables excluded via `.gitignore`.

### 2. Statistical Data Synthesis
Generate 10,000 statistically realistic candidate records mapping complex correlations:
```bash
python -m ml.data_generator
```

### 3. Star Schema ETL Loader
Validate types/bounds and bulk-load generated records into PostgreSQL:
```bash
python -m database.etl_load
```

### 4. Modular ML Training
Extract records from PostgreSQL, execute 5-Fold Cross-Validation, train the Random Forest model, and serialize performance logs:
```bash
python -m ml.train
```

### 5. Start Backend REST API
Launch the FastAPI server (accessible at `http://127.0.0.1:8000`):
```bash
python -m uvicorn api.main:app --reload
```

### 6. Start Dashboard UI
Launch the interactive Streamlit recruiter interface (accessible at `http://localhost:8501`):
```bash
streamlit run frontend/app.py
```

---

## ⚖️ Project Maturity & Production Readiness Roadmap

This repository is designed as an **end-to-end prototype and proof-of-concept (POC)** demonstrating data engineering, applied machine learning, and interactive analytics.

It is **not** an enterprise production-ready deployment. In technical interviews, discussing the explicit differences between an analytical prototype and a hardened production system demonstrates engineering maturity. Key production gaps and roadmap items include:

1. **Secrets Management:** In a true production environment, credentials should not reside in static local `.env` files. Instead, use a dedicated secrets manager (e.g., HashiCorp Vault, AWS Secrets Manager, or GCP Secret Manager) with automated secret rotation and IAM role authentication.
2. **Authentication & Authorization:** The REST API currently does not enforce authentication. Production deployment requires OAuth2 / JWT authentication, role-based access control (RBAC), and API rate-limiting / DDoS protection.
3. **Database Hardening & Migrations:** The ETL script currently recreates tables on initial load. Production requires declarative schema migrations (via Alembic), SSL/TLS-encrypted connections (`sslmode=verify-full`), read replicas, and strict least-privilege database user permissions.
4. **Data Governance & Real-World Validation:** The model is trained on synthesized data to simulate hiring patterns. Production deployment requires vetting on real, anonymized historical hiring data, bias audits (Fairness/Disparate Impact), and ongoing data drift monitoring (e.g., Evidently AI or Great Expectations).
5. **Observability & Infrastructure:** Production systems need centralized logging (ELK / CloudWatch), distributed tracing (OpenTelemetry), health probes, containerization (Docker Compose / Kubernetes), and automated CI/CD pipelines.

---

## 🧠 Interview Guide: Explaining the Engineering Decisions

When speaking with hiring managers or software/ML engineers during technical interviews, use these structured explanations to defend your technical decisions:

### 1. Why Random Forest?
> *"We chose a **Random Forest Classifier** because the candidate evaluation problem operates on tabular data with non-linear decision boundaries and discrete binary features (like one-hot skills). Random Forest handles mixed data types natively, is robust to outliers, and requires minimal feature scaling. Furthermore, its ensemble nature provides structural protection against overfitting, which we validated with a 5-Fold Cross-Validation score of **94.8%** matching our test set accuracy of **94.8%**, demonstrating strong baseline consistency across folds on our benchmark dataset."*

### 2. How is the synthetic data generated?
> *"Instead of simple random sampling, the data synthesis uses academic and industry-aligned statistical distributions. CGPA is normally distributed around a 7.2 mean, and coding scores are generated with a positive correlation boost from the CGPA. The final placement outcome is assigned using a **logistic sigmoid probability formula** based on weighted contributions from all variables, plus Gaussian random noise to represent real-life interview luck. This establishes a clean, mathematically sound ground truth for our model to learn."*

### 3. How does the TF-IDF Resume Domain Classifier work?
> *"To replace simple keyword matching, we engineered an NLP vector space model. We defined idealized target corpuses for 4 major domains (Data Science, Frontend, Backend, and DevOps). We fit a **TfidfVectorizer** on these corpuses to extract term importance weights. When a resume PDF is uploaded, it is cleaned and converted to a TF-IDF vector. We then compute the **Cosine Similarity** between the resume vector and our four target domain vectors, assigning the candidate to the domain with the highest cosine score. This provides a lightweight classification system without the massive memory footprint of a deep learning model."*

### 4. How does the "What-If" Candidate Recommendation Engine operate?
> *"Instead of hardcoded static rules, the recommendations are calculated dynamically. The engine queries the trained Random Forest pipeline to get the candidate's baseline probability. It then performs **what-if simulations** by systematically cloning the profile and incrementing features (e.g. adding a missing skill, increasing the coding index, completing an internship). It calculates the mathematical probability delta ($\Delta$ Probability) for each action. The actions are then sorted and returned based on the highest predictive yield, giving candidates a precise, mathematically-justified career action plan."*

### 5. Why Apriori?
> *"We chose the **Apriori Algorithm** to perform market trend mining on our PostgreSQL database records. While the Random Forest model is excellent for predictive scoring, Apriori extracts explicit association rules (calculating Support, Confidence, and **Lift**). This lets us discover strategic co-occurrence patterns (e.g., Candidates combining SQL + Web Dev are 26% more likely to be placed compared to the baseline population). This provides recruiters and educators with clear, actionable strategic insights."*

### 6. Is this application production-ready?
> *"No, and being upfront about that distinction is important. This is an end-to-end analytical prototype demonstrating ETL pipelines, ML modeling, and REST APIs. Moving to production would require cloud secrets management (Vault / AWS Secrets Manager) instead of `.env`, OAuth2/JWT authentication, database migration management with Alembic, encrypted TLS connections, and replacing the synthetic data generator with validated institutional datasets subject to fairness and drift audits."*

### 7. How are credentials and database connections managed?
> *"Following 12-factor application methodology, all database connection details are decoupled from code and read from environment variables via `python-dotenv`. The database module safely parses parameters or a direct `DATABASE_URL`, handles URL-encoding for complex passwords, provides safe fallbacks if the database is temporarily unreachable, and provides a clean `.env.example` template while ensuring real `.env` files are strictly ignored by git."*
