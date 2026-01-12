import os
from datetime import datetime
from pathlib import Path
from sqlalchemy import (
    create_engine,
    Column,
    Integer,
    Float,
    String,
    DateTime,
    Boolean,
)
from sqlalchemy.orm import declarative_base, sessionmaker

env_db_url = os.getenv("DB_URL")
if env_db_url:
    DATABASE_URL = env_db_url
else:
    BASE_DIR = Path(__file__).resolve().parent
    db_path = BASE_DIR / "history.db"
    db_path.parent.mkdir(parents=True, exist_ok=True)
    DATABASE_URL = f"sqlite:///{db_path.as_posix()}"

engine = create_engine(
    DATABASE_URL,
    connect_args=(
        {"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {}
    ),
)

SessionLocal = sessionmaker(
    autocommit=False,
    autoflush=False,
    bind=engine,
)

Base = declarative_base()


class RequestHistory(Base):
    __tablename__ = "request_history"

    id = Column(Integer, primary_key=True, index=True)
    timestamp = Column(DateTime, default=datetime.utcnow)

    elapsed_ms = Column(Float)
    image_width = Column(Integer)
    image_height = Column(Integer)

    predicted_class = Column(String, nullable=True)


def init_db() -> None:
    Base.metadata.create_all(bind=engine)
