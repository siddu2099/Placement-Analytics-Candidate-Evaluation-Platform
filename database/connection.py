import os
from pathlib import Path
from urllib.parse import quote_plus

from dotenv import load_dotenv
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker, declarative_base

# Load environment variables: check project root first, then database directory, then standard cwd
current_file = Path(__file__).resolve()
project_root = current_file.parent.parent
root_env = project_root / ".env"
db_env = current_file.parent / ".env"

if root_env.exists():
    load_dotenv(dotenv_path=root_env)
elif db_env.exists():
    load_dotenv(dotenv_path=db_env)
else:
    load_dotenv()

# Support full connection string or individual components via environment variables
DATABASE_URI = os.getenv("DATABASE_URL") or os.getenv("DATABASE_URI")

if DATABASE_URI:
    if DATABASE_URI.startswith("postgres://"):
        DATABASE_URI = DATABASE_URI.replace("postgres://", "postgresql+psycopg2://", 1)
    elif DATABASE_URI.startswith("postgresql://") and not DATABASE_URI.startswith("postgresql+"):
        DATABASE_URI = DATABASE_URI.replace("postgresql://", "postgresql+psycopg2://", 1)
else:
    DB_USER = os.getenv("DB_USER", "postgres")
    DB_PASS = os.getenv("DB_PASS", "")
    DB_HOST = os.getenv("DB_HOST", "localhost")
    DB_NAME = os.getenv("DB_NAME", "placement_db")
    DB_PORT = os.getenv("DB_PORT", "5432")

    if DB_PASS:
        encoded_password = quote_plus(str(DB_PASS))
        auth = f"{DB_USER}:{encoded_password}" if DB_USER else f":{encoded_password}"
    elif DB_USER:
        auth = f"{DB_USER}"
    else:
        auth = ""

    auth_prefix = f"{auth}@" if auth else ""
    DATABASE_URI = f"postgresql+psycopg2://{auth_prefix}{DB_HOST}:{DB_PORT}/{DB_NAME}"

engine = create_engine(
    DATABASE_URI,
    pool_size=5,
    max_overflow=10,
    pool_timeout=10,
    pool_recycle=1800,
    connect_args={"connect_timeout": 3}
)

SessionLocal = sessionmaker(
    autocommit=False,
    autoflush=False,
    bind=engine
)

Base = declarative_base()


def get_db():
    """Dependency for obtaining database sessions in FastAPI routes"""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def check_connection():
    """Verifies that the PostgreSQL server is running and database is accessible"""
    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        return True
    except Exception as e:
        print(f"[-] Database connection failure: {e}")
        return False
