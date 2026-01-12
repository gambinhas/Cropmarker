from __future__ import annotations

from pathlib import Path

from sqlalchemy import create_engine, event
from sqlalchemy.orm import DeclarativeBase, sessionmaker


class Base(DeclarativeBase):
    pass


def create_sqlite_engine(db_path: Path):
    db_path.parent.mkdir(parents=True, exist_ok=True)
    engine = create_engine(
        f"sqlite:///{db_path}",
        connect_args={"check_same_thread": False},
        pool_pre_ping=True,
    )

    @event.listens_for(engine, "connect")
    def _set_sqlite_pragma(dbapi_connection, connection_record):  # noqa: ANN001
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA journal_mode=WAL;")
        cursor.execute("PRAGMA foreign_keys=ON;")
        cursor.close()

    return engine


def create_session_factory(engine):
    return sessionmaker(autoflush=False, autocommit=False, bind=engine)


def ensure_sqlite_migrations(engine) -> None:  # noqa: ANN001
    """Apply minimal schema migrations for SQLite without Alembic.

    Safe to call on every start.
    """

    with engine.begin() as conn:
        # users.expertise_score
        cols = {row[1] for row in conn.exec_driver_sql("PRAGMA table_info(users)").fetchall()}
        if cols and "expertise_score" not in cols:
            conn.exec_driver_sql("ALTER TABLE users ADD COLUMN expertise_score INTEGER NOT NULL DEFAULT 0")
