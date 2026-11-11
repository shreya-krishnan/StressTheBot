import os
from pathlib import Path

from sqlalchemy import create_engine, event
from sqlalchemy.orm import declarative_base, sessionmaker


DEFAULT_DATABASE_PATH = Path(__file__).resolve().parents[1] / "data" / "stressthebot.db"
DATABASE_URL = os.getenv("STRESSTHEBOT_DATABASE_URL", f"sqlite:///{DEFAULT_DATABASE_PATH.as_posix()}")

if DATABASE_URL.startswith("sqlite:///"):
    database_path = Path(DATABASE_URL.removeprefix("sqlite:///"))
    if not database_path.is_absolute():
        database_path = Path.cwd() / database_path
    database_path.parent.mkdir(parents=True, exist_ok=True)

engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {},
)

if DATABASE_URL.startswith("sqlite"):
    @event.listens_for(engine, "connect")
    def _enable_sqlite_foreign_keys(dbapi_connection, _connection_record):
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.close()

SessionLocal = sessionmaker(bind=engine, autoflush=False, expire_on_commit=False)
Base = declarative_base()


def get_db():
    session = SessionLocal()
    try:
        yield session
    finally:
        session.close()


def init_db():
    from app.database import models  # noqa: F401
    Base.metadata.create_all(bind=engine)
