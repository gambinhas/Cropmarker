from __future__ import annotations

from datetime import datetime

from sqlalchemy import Boolean, DateTime, Float, ForeignKey, Integer, String, Text, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column, relationship

from .db import Base


class User(Base):
    __tablename__ = "users"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    username: Mapped[str] = mapped_column(String(64), unique=True, index=True)
    password_hash: Mapped[str] = mapped_column(String(255))
    is_admin: Mapped[bool] = mapped_column(Boolean, default=False)
    expertise_score: Mapped[int] = mapped_column(Integer, default=0)
    access_token_hash: Mapped[str] = mapped_column(String(64), default="", index=True)
    # Admin-visible encrypted token (so it can be shown in the admin panel). Login still uses access_token_hash.
    access_token_encrypted: Mapped[str] = mapped_column(Text, default="")
    last_login_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)

    annotations: Mapped[list[Annotation]] = relationship(back_populates="user")  # type: ignore[name-defined]
    user_tasks: Mapped[list[UserTask]] = relationship(back_populates="user")  # type: ignore[name-defined]


class Task(Base):
    __tablename__ = "tasks"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    site: Mapped[str] = mapped_column(String(255))
    rel_path: Mapped[str] = mapped_column(String(1024), unique=True)
    filename: Mapped[str] = mapped_column(String(255))
    unique_id: Mapped[str] = mapped_column(String(1024), unique=True, index=True)
    display_order: Mapped[int] = mapped_column(Integer, index=True)

    user_tasks: Mapped[list[UserTask]] = relationship(back_populates="task")  # type: ignore[name-defined]


class UserTask(Base):
    __tablename__ = "user_tasks"
    __table_args__ = (
        UniqueConstraint("user_id", "instance_id", name="uq_user_instance"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id"), index=True)
    task_id: Mapped[int] = mapped_column(ForeignKey("tasks.id"), index=True)

    # Unique per-user task instance id. Originals use Task.unique_id;
    # QC duplicates use Task.unique_id + "_qc".
    instance_id: Mapped[str] = mapped_column(String(1100))

    # For QC duplicates, points to the original instance_id (Task.unique_id).
    qc_reference: Mapped[str | None] = mapped_column(String(1100), nullable=True)
    display_order: Mapped[int] = mapped_column(Integer, index=True)

    user: Mapped[User] = relationship(back_populates="user_tasks")  # type: ignore[name-defined]
    task: Mapped[Task] = relationship(back_populates="user_tasks")
    annotations: Mapped[list[Annotation]] = relationship(back_populates="user_task")  # type: ignore[name-defined]


class Annotation(Base):
    __tablename__ = "annotations"
    __table_args__ = (UniqueConstraint("user_id", "user_task_id", name="uq_user_usertask"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id"), index=True)
    user_task_id: Mapped[int] = mapped_column(ForeignKey("user_tasks.id"), index=True)

    cropmark: Mapped[int] = mapped_column(Integer)  # 0/1/2
    drawing_json: Mapped[str] = mapped_column(Text)  # JSON list of strokes

    brightness: Mapped[float] = mapped_column(Float, default=100.0)  # percent
    contrast: Mapped[float] = mapped_column(Float, default=100.0)  # percent

    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    user: Mapped[User] = relationship(back_populates="annotations")
    user_task: Mapped[UserTask] = relationship(back_populates="annotations")
