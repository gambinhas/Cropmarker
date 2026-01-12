from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path

from sqlalchemy import and_, select

from .db import Base, create_session_factory, create_sqlite_engine, ensure_sqlite_migrations
from .models import Annotation, Task, User, UserTask
from .task_build import build_tasks


_MONTHS = {
    "jan": "01",
    "feb": "02",
    "mar": "03",
    "apr": "04",
    "may": "05",
    "jun": "06",
    "jul": "07",
    "aug": "08",
    "sep": "09",
    "oct": "10",
    "nov": "11",
    "dec": "12",
}


def _parse_mm_yyyy_from_filename(filename: str) -> str:
    m = re.search(r"(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)(\d{4})", (filename or "").lower())
    if not m:
        return ""
    mm = _MONTHS.get(m.group(1), "")
    yyyy = m.group(2)
    return f"{mm}/{yyyy}" if mm else ""


def _safe_filename_part(s: str) -> str:
    s = (s or "").strip()
    s = re.sub(r"[\\/:*?\"<>|]+", "_", s)
    s = s.strip()
    return s or "user"


def main() -> None:
    parser = argparse.ArgumentParser(description="Cropmarker WebApp admin CLI")
    parser.add_argument("--db", default="webapp_data/webapp.sqlite3", help="SQLite DB path")
    parser.add_argument("--dataset-root", default="../image_dataset", help="Path to image_dataset")

    sub = parser.add_subparsers(dest="cmd", required=True)

    sub.add_parser("init-db", help="Create tables")

    p_import = sub.add_parser("import-tasks", help="Import original tasks from dataset")
    # QC duplicates are created per-user at login; import only originals here.

    p_user = sub.add_parser("create-user", help="Create a user")
    p_user.add_argument("--username", required=True)
    p_user.add_argument("--admin", action="store_true")
    p_user.add_argument("--expertise-score", type=int, default=0, choices=[0, 1, 3, 5])

    p_export = sub.add_parser("export-csv", help="Export annotations as CSV")
    p_export.add_argument("--out", default="export.csv")

    p_export_xlsx = sub.add_parser(
        "export-xlsx",
        help="Export per-user XLSX in the requested format",
    )
    p_export_xlsx.add_argument(
        "--out-dir",
        default="export_xlsx",
        help="Output directory for XLSX file(s)",
    )
    p_export_xlsx.add_argument(
        "--username",
        default=None,
        help="Export only this user (default: export all users)",
    )

    args = parser.parse_args()

    db_path = Path(args.db).resolve()
    engine = create_sqlite_engine(db_path)
    Base.metadata.create_all(engine)
    ensure_sqlite_migrations(engine)
    Session = create_session_factory(engine)

    if args.cmd == "init-db":
        # tables already created above
        print(f"Initialized DB at: {db_path}")
        return

    if args.cmd == "import-tasks":
        dataset_root = Path(args.dataset_root).resolve()
        rows = build_tasks(dataset_root)
        with Session() as s:
            existing = {t.unique_id for t in s.execute(select(Task.unique_id)).scalars().all()}
            to_add = []
            for r in rows:
                if r.unique_id in existing:
                    continue
                to_add.append(
                    Task(
                        site=r.site,
                        rel_path=r.rel_path,
                        filename=r.filename,
                        unique_id=r.unique_id,
                        display_order=r.order,
                    )
                )
            s.add_all(to_add)
            s.commit()
        print(f"Imported {len(to_add)} tasks (skipped {len(rows) - len(to_add)} existing).")
        return

    if args.cmd == "create-user":
        with Session() as s:
            existing = s.execute(select(User).where(User.username == args.username)).scalar_one_or_none()
            if existing:
                # For convenience, allow updating the admin flag.
                if bool(args.admin) and not existing.is_admin:
                    existing.is_admin = True
                if int(args.expertise_score) != int(getattr(existing, "expertise_score", 0)):
                    existing.expertise_score = int(args.expertise_score)
                    s.commit()
                print(
                    f"User already exists: {args.username} admin={bool(existing.is_admin)} expertise_score={int(getattr(existing, 'expertise_score', 0))}"
                )
                return

            # No passwords by design (mirrors the original Tkinter workflow).
            user = User(
                username=args.username,
                password_hash="",
                is_admin=bool(args.admin),
                expertise_score=int(args.expertise_score),
            )
            s.add(user)
            s.commit()
        print(f"Created user: {args.username} admin={bool(args.admin)} expertise_score={int(args.expertise_score)}")
        return

    if args.cmd == "export-csv":
        out_path = Path(args.out).resolve()
        with Session() as s:
            rows = s.execute(
                select(Annotation, User, UserTask, Task)
                .join(User, Annotation.user_id == User.id)
                .join(UserTask, Annotation.user_task_id == UserTask.id)
                .join(Task, UserTask.task_id == Task.id)
                .order_by(Annotation.created_at.asc())
            ).all()

        with out_path.open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(
                [
                    "annotation_id",
                    "created_at",
                    "username",
                    "expertise_score",
                    "user_task_id",
                    "instance_id",
                    "qc_reference",
                    "base_task_id",
                    "base_unique_id",
                    "site",
                    "rel_path",
                    "filename",
                    "order",
                    "cropmark",
                    "brightness",
                    "contrast",
                    "drawing_json",
                ]
            )
            for ann, user, user_task, task in rows:
                w.writerow(
                    [
                        ann.id,
                        ann.created_at.isoformat(),
                        user.username,
                        getattr(user, "expertise_score", 0),
                        user_task.id,
                        user_task.instance_id,
                        user_task.qc_reference or "",
                        task.id,
                        task.unique_id,
                        task.site,
                        task.rel_path,
                        task.filename,
                        user_task.display_order,
                        ann.cropmark,
                        ann.brightness,
                        ann.contrast,
                        ann.drawing_json,
                    ]
                )

        print(f"Wrote: {out_path}")
        return

    if args.cmd == "export-xlsx":
        try:
            from openpyxl import Workbook
        except Exception as e:
            raise SystemExit(
                "Missing dependency 'openpyxl'. Install it with: pip install -r webapp/requirements.txt"
            ) from e

        out_dir = Path(args.out_dir).resolve()
        out_dir.mkdir(parents=True, exist_ok=True)

        with Session() as s:
            users_q = select(User).order_by(User.username.asc())
            if args.username:
                users_q = users_q.where(User.username == args.username)
            users = s.execute(users_q).scalars().all()

            if not users:
                print("No users found.")
                return

            for user in users:
                rows = s.execute(
                    select(UserTask, Task, Annotation)
                    .join(Task, UserTask.task_id == Task.id)
                    .outerjoin(
                        Annotation,
                        and_(Annotation.user_task_id == UserTask.id, Annotation.user_id == user.id),
                    )
                    .where(UserTask.user_id == user.id)
                    .order_by(UserTask.display_order.asc())
                ).all()

                wb = Workbook()
                ws = wb.active
                ws.title = "cropmarks"

                headers = [
                    "Site",
                    "Date",
                    "Cropmark",
                    "Image",
                    "Saved",
                    "Drawing",
                    "QC_Reference",
                    "UniqueID",
                    "Order",
                ]
                ws.append(headers)

                for user_task, task, ann in rows:
                    saved = "S" if ann is not None else ""
                    cropmark = ann.cropmark if ann is not None else ""

                    drawing = ""
                    if ann is not None and int(ann.cropmark) in (1, 2):
                        drawing = ann.drawing_json or ""

                    ws.append(
                        [
                            task.site,
                            _parse_mm_yyyy_from_filename(task.filename),
                            cropmark,
                            task.filename,
                            saved,
                            drawing,
                            user_task.qc_reference or "",
                            task.unique_id,
                            int(user_task.display_order),
                        ]
                    )

                ws.freeze_panes = "A2"
                ws.auto_filter.ref = f"A1:I{ws.max_row}"
                for col, width in {
                    "A": 26,
                    "B": 10,
                    "C": 10,
                    "D": 36,
                    "E": 8,
                    "F": 60,
                    "G": 26,
                    "H": 36,
                    "I": 8,
                }.items():
                    ws.column_dimensions[col].width = width

                expertise_score = int(getattr(user, "expertise_score", 0) or 0)
                filename = f"cropmarks_{_safe_filename_part(user.username)}_{expertise_score}.xlsx"
                out_path = out_dir / filename
                wb.save(out_path)
                print(f"Wrote: {out_path}")

        return


if __name__ == "__main__":
    main()
