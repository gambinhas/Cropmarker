from __future__ import annotations

import json
import os
import random
import re
import secrets
import hashlib
import zipfile
from io import BytesIO
from pathlib import Path
from datetime import datetime

from fastapi import Depends, FastAPI, Form, HTTPException, Request
from fastapi.responses import HTMLResponse, RedirectResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from sqlalchemy import and_, delete, func, select
from starlette.middleware.sessions import SessionMiddleware

from .auth import login_session, logout_session, require_user
from .db import Base, create_session_factory, create_sqlite_engine, ensure_sqlite_migrations
from .models import Annotation, Task, User, UserTask

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_DB = BASE_DIR / "webapp_data" / "webapp.sqlite3"
DEFAULT_DATASET_ROOT = (BASE_DIR / ".." / "image_dataset").resolve()
RESOURCES_DIR = (BASE_DIR / ".." / "resources").resolve()
SECRET_KEY = os.environ.get("CROPMARKER_SECRET_KEY", "dev-secret-change-me")
ADMIN_EXPORT_TOKEN = os.environ.get("CROPMARKER_ADMIN_EXPORT_TOKEN", "")

DB_PATH = Path(os.environ.get("CROPMARKER_DB_PATH", str(DEFAULT_DB)))
DATASET_ROOT = Path(os.environ.get("CROPMARKER_DATASET_ROOT", str(DEFAULT_DATASET_ROOT)))

# Make the service bootable in fresh environments (e.g., Render) where the disk may start empty.
DB_PATH.parent.mkdir(parents=True, exist_ok=True)
DATASET_ROOT.mkdir(parents=True, exist_ok=True)

engine = create_sqlite_engine(DB_PATH)
SessionLocal = create_session_factory(engine)
Base.metadata.create_all(engine)
ensure_sqlite_migrations(engine)

app = FastAPI(title="Cropmarker Web")
app.add_middleware(SessionMiddleware, secret_key=SECRET_KEY)

templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))

# Serve images as /images/<site>/<file>.jpg
app.mount("/images", StaticFiles(directory=str(DATASET_ROOT)), name="images")
app.mount("/static", StaticFiles(directory=str(BASE_DIR / "static")), name="static")
if RESOURCES_DIR.exists():
    app.mount("/resources", StaticFiles(directory=str(RESOURCES_DIR)), name="resources")


QC_DUPLICATES_PER_USER = 39
EXPECTED_ORIGINALS = 381

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


def _require_admin_export_token(request: Request, token_param: str | None) -> str:
    # If the token is not configured, do not expose this endpoint at all.
    if not ADMIN_EXPORT_TOKEN:
        raise HTTPException(status_code=404)

    token = (token_param or "").strip()
    if not token:
        token = (request.headers.get("X-Admin-Token") or "").strip()
    if not token:
        auth = (request.headers.get("Authorization") or "").strip()
        if auth.lower().startswith("bearer "):
            token = auth[7:].strip()

    if not token or not secrets.compare_digest(token, ADMIN_EXPORT_TOKEN):
        raise HTTPException(status_code=403, detail="Forbidden")

    return token


def _fmt_dt(dt) -> str:
    if not dt:
        return ""
    try:
        return dt.strftime("%Y-%m-%d %H:%M")
    except Exception:
        return str(dt)


def _hash_access_token(token: str) -> str:
    # Store only a non-reversible hash in the DB.
    return hashlib.sha256((token or "").encode("utf-8")).hexdigest()


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def ensure_user_task_list(db, user_id: int) -> None:
    existing_count = db.execute(select(UserTask.id).where(UserTask.user_id == user_id)).scalars().first()
    if existing_count is not None:
        return

    base_tasks = db.execute(select(Task).order_by(Task.id.asc())).scalars().all()
    if len(base_tasks) != EXPECTED_ORIGINALS:
        raise HTTPException(
            status_code=500,
            detail=f"Expected {EXPECTED_ORIGINALS} original tasks, found {len(base_tasks)}. Re-import tasks from dataset.",
        )

    rng = random.SystemRandom()
    qc_sample = set(rng.sample(base_tasks, QC_DUPLICATES_PER_USER))

    items: list[tuple[Task, bool]] = [(t, False) for t in base_tasks] + [(t, True) for t in qc_sample]
    rng.shuffle(items)

    user_tasks: list[UserTask] = []
    for idx, (t, is_qc) in enumerate(items, start=1):
        instance_id = f"{t.unique_id}_qc" if is_qc else t.unique_id
        qc_reference = t.unique_id if is_qc else None
        user_tasks.append(
            UserTask(
                user_id=user_id,
                task_id=t.id,
                instance_id=instance_id,
                qc_reference=qc_reference,
                display_order=idx,
            )
        )

    db.add_all(user_tasks)
    db.commit()


@app.get("/", response_class=HTMLResponse)
def index(request: Request):
    user = request.session.get("user")
    if not user:
        return RedirectResponse(url="/login", status_code=302)
    return RedirectResponse(url="/tasks/next", status_code=302)


@app.get("/login", response_class=HTMLResponse)
def login_get(request: Request):
    return templates.TemplateResponse("login.html", {"request": request, "error": None})


@app.get("/new-user", response_class=HTMLResponse)
def new_user_get(request: Request, username: str):
    raise HTTPException(status_code=404)


@app.post("/new-user")
def new_user_post(
    request: Request,
    username: str = Form(...),
    expertise_score: int = Form(...),
    db=Depends(get_db),
):
    raise HTTPException(status_code=404)


def _admin_user_rows(db):
    ut_counts = {
        int(user_id): int(count)
        for user_id, count in db.execute(select(UserTask.user_id, func.count()).group_by(UserTask.user_id)).all()
    }
    ann_counts = {
        int(user_id): int(count)
        for user_id, count in db.execute(select(Annotation.user_id, func.count()).group_by(Annotation.user_id)).all()
    }
    last_saved = {
        int(user_id): last
        for user_id, last in db.execute(
            select(Annotation.user_id, func.max(Annotation.created_at)).group_by(Annotation.user_id)
        ).all()
    }

    users = db.execute(select(User).where(User.is_admin == False).order_by(User.username.asc())).scalars().all()  # noqa: E712
    rows = []
    for u in users:
        total = int(ut_counts.get(int(u.id), 0))
        done = int(ann_counts.get(int(u.id), 0))
        remaining = max(0, total - done)
        rows.append(
            {
                "username": u.username,
                "expertise_score": int(getattr(u, "expertise_score", 0) or 0),
                "total": total,
                "done": done,
                "remaining": remaining,
                "last_login_at": _fmt_dt(getattr(u, "last_login_at", None)),
                "last_saved_at": _fmt_dt(last_saved.get(int(u.id))),
                "has_token": bool(getattr(u, "access_token_hash", "")),
            }
        )
    return rows


@app.get("/admin", response_class=HTMLResponse)
def admin_get(request: Request, token: str | None = None, db=Depends(get_db)):
    admin_token = _require_admin_export_token(request, token)
    users = _admin_user_rows(db)
    return templates.TemplateResponse(
        "admin.html",
        {
            "request": request,
            "admin_token": admin_token,
            "users": users,
            "message": None,
            "token_to_copy": None,
        },
    )


@app.post("/admin/users/create", response_class=HTMLResponse)
def admin_create_user(
    request: Request,
    token: str = Form(...),
    username: str = Form(...),
    expertise_score: int = Form(...),
    access_token: str = Form(""),
    db=Depends(get_db),
):
    token = _require_admin_export_token(request, token)
    username = (username or "").strip()
    if not username:
        raise HTTPException(status_code=400)
    if expertise_score not in (0, 1, 3, 5):
        raise HTTPException(status_code=400)

    existing = db.execute(select(User).where(User.username == username)).scalar_one_or_none()
    if existing:
        users = _admin_user_rows(db)
        return templates.TemplateResponse(
            "admin.html",
            {
                "request": request,
                "admin_token": token,
                "users": users,
                "message": f"User already exists: {username}",
                "token_to_copy": None,
            },
            status_code=400,
        )

    token_plain = (access_token or "").strip() or secrets.token_urlsafe(24)
    user = User(
        username=username,
        password_hash="",
        is_admin=False,
        expertise_score=int(expertise_score),
        access_token_hash=_hash_access_token(token_plain),
    )
    db.add(user)
    db.commit()
    db.refresh(user)
    ensure_user_task_list(db, user.id)

    users = _admin_user_rows(db)
    return templates.TemplateResponse(
        "admin.html",
        {
            "request": request,
            "admin_token": token,
            "users": users,
            "message": f"Created user: {username}",
            "token_to_copy": token_plain,
        },
    )


@app.post("/admin/users/update", response_class=HTMLResponse)
def admin_update_user(
    request: Request,
    token: str = Form(...),
    username: str = Form(...),
    expertise_score: int = Form(...),
    db=Depends(get_db),
):
    token = _require_admin_export_token(request, token)
    if expertise_score not in (0, 1, 3, 5):
        raise HTTPException(status_code=400)
    user = db.execute(select(User).where(User.username == username)).scalar_one_or_none()
    if not user:
        raise HTTPException(status_code=404)
    user.expertise_score = int(expertise_score)
    db.commit()

    users = _admin_user_rows(db)
    return templates.TemplateResponse(
        "admin.html",
        {
            "request": request,
            "admin_token": token,
            "users": users,
            "message": f"Updated expertise for: {username}",
            "token_to_copy": None,
        },
    )


@app.post("/admin/users/rotate-token", response_class=HTMLResponse)
def admin_rotate_token(
    request: Request,
    token: str = Form(...),
    username: str = Form(...),
    db=Depends(get_db),
):
    token = _require_admin_export_token(request, token)
    user = db.execute(select(User).where(User.username == username)).scalar_one_or_none()
    if not user:
        raise HTTPException(status_code=404)

    token_plain = secrets.token_urlsafe(24)
    user.access_token_hash = _hash_access_token(token_plain)
    db.commit()

    users = _admin_user_rows(db)
    return templates.TemplateResponse(
        "admin.html",
        {
            "request": request,
            "admin_token": token,
            "users": users,
            "message": f"Rotated token for: {username}",
            "token_to_copy": token_plain,
        },
    )


@app.post("/admin/users/reset", response_class=HTMLResponse)
def admin_reset_user(
    request: Request,
    token: str = Form(...),
    username: str = Form(...),
    db=Depends(get_db),
):
    token = _require_admin_export_token(request, token)
    user = db.execute(select(User).where(User.username == username)).scalar_one_or_none()
    if not user:
        raise HTTPException(status_code=404)

    db.execute(delete(Annotation).where(Annotation.user_id == user.id))
    db.commit()

    users = _admin_user_rows(db)
    return templates.TemplateResponse(
        "admin.html",
        {
            "request": request,
            "admin_token": token,
            "users": users,
            "message": f"Reset annotations for: {username}",
            "token_to_copy": None,
        },
    )


@app.post("/admin/users/delete", response_class=HTMLResponse)
def admin_delete_user(
    request: Request,
    token: str = Form(...),
    username: str = Form(...),
    db=Depends(get_db),
):
    token = _require_admin_export_token(request, token)
    user = db.execute(select(User).where(User.username == username)).scalar_one_or_none()
    if not user:
        raise HTTPException(status_code=404)

    db.execute(delete(Annotation).where(Annotation.user_id == user.id))
    db.execute(delete(UserTask).where(UserTask.user_id == user.id))
    db.execute(delete(User).where(User.id == user.id))
    db.commit()

    users = _admin_user_rows(db)
    return templates.TemplateResponse(
        "admin.html",
        {
            "request": request,
            "admin_token": token,
            "users": users,
            "message": f"Deleted user: {username}",
            "token_to_copy": None,
        },
    )


@app.get("/intro", response_class=HTMLResponse)
def intro_get(request: Request):
    user = require_user(request)
    return templates.TemplateResponse(
        "intro.html",
        {
            "request": request,
            "username": user.username,
        },
    )


@app.post("/login")
def login_post(
    request: Request,
    access_token: str = Form(...),
    db=Depends(get_db),
):
    access_token = (access_token or "").strip()
    if not access_token:
        return templates.TemplateResponse(
            "login.html",
            {"request": request, "error": "Please enter an access token"},
            status_code=400,
        )

    token_hash = _hash_access_token(access_token)
    user = db.execute(select(User).where(User.access_token_hash == token_hash)).scalar_one_or_none()
    if not user:
        return templates.TemplateResponse(
            "login.html",
            {"request": request, "error": "Invalid access token"},
            status_code=403,
        )

    user.last_login_at = datetime.utcnow()
    db.commit()

    ensure_user_task_list(db, user.id)

    login_session(request, user)
    return RedirectResponse(url="/tasks/next", status_code=302)


@app.post("/logout")
def logout_post(request: Request):
    logout_session(request)
    return RedirectResponse(url="/login", status_code=302)


@app.get("/admin/exports.zip")
def admin_exports_zip(
    request: Request,
    token: str | None = None,
    username: str | None = None,
    include_admin: bool = False,
    db=Depends(get_db),
):
    """Token-protected download of a ZIP containing one XLSX per user.

    This is intended for admin-only use on Render where downloading files from disk is inconvenient.
    """

    _require_admin_export_token(request, token)

    try:
        from openpyxl import Workbook
    except Exception as e:
        raise HTTPException(status_code=500, detail="Missing dependency 'openpyxl'") from e

    users_q = select(User).order_by(User.username.asc())
    if not include_admin:
        users_q = users_q.where(User.is_admin == False)  # noqa: E712
    if username:
        users_q = users_q.where(User.username == username)
    users = db.execute(users_q).scalars().all()
    if not users:
        raise HTTPException(status_code=404, detail="No users found")

    zip_buf = BytesIO()
    with zipfile.ZipFile(zip_buf, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for user in users:
            rows = db.execute(
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
            ws.append(
                [
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
            )

            for user_task, task, ann in rows:
                saved = "S" if ann is not None else ""
                cropmark = ann.cropmark if ann is not None else ""

                drawing = ""
                if ann is not None and int(ann.cropmark) in (1, 2):
                    drawing = ann.drawing_json or ""

                # Compatibility with the existing supervisor tool / legacy offline exports:
                # - UniqueID is the image filename (not the dataset rel_path)
                # - QC duplicates use "<filename>_qc" and QC_Reference points to the original filename
                is_qc = bool(user_task.qc_reference)
                legacy_unique_id = f"{task.filename}_qc" if is_qc else task.filename
                legacy_qc_ref = task.filename if is_qc else ""

                ws.append(
                    [
                        task.site,
                        _parse_mm_yyyy_from_filename(task.filename),
                        cropmark,
                        task.filename,
                        saved,
                        drawing,
                        legacy_qc_ref,
                        legacy_unique_id,
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
            xlsx_name = f"cropmarks_{_safe_filename_part(user.username)}_{expertise_score}.xlsx"
            xlsx_buf = BytesIO()
            wb.save(xlsx_buf)
            zf.writestr(xlsx_name, xlsx_buf.getvalue())

    zip_buf.seek(0)
    headers = {
        "Content-Disposition": "attachment; filename=cropmarker_exports.zip",
        "Cache-Control": "no-store",
    }
    return StreamingResponse(zip_buf, media_type="application/zip", headers=headers)


@app.get("/tasks/next")
def next_task(request: Request, db=Depends(get_db)):
    user = require_user(request)

    ensure_user_task_list(db, user.id)

    annotated_user_task_ids = db.execute(
        select(Annotation.user_task_id).where(Annotation.user_id == user.id)
    ).scalars().all()

    q = select(UserTask).where(UserTask.user_id == user.id).order_by(UserTask.display_order.asc())
    if annotated_user_task_ids:
        q = q.where(UserTask.id.not_in(annotated_user_task_ids))

    next_user_task = db.execute(q.limit(1)).scalars().first()
    if not next_user_task:
        return templates.TemplateResponse(
            "done.html",
            {
                "request": request,
                "username": user.username,
            },
        )
    return RedirectResponse(url=f"/annotate/{next_user_task.id}", status_code=302)


@app.get("/annotate/{user_task_id}", response_class=HTMLResponse)
def annotate_get(user_task_id: int, request: Request, db=Depends(get_db)):
    user = require_user(request)

    ensure_user_task_list(db, user.id)

    user_task = db.execute(
        select(UserTask).where(and_(UserTask.id == user_task_id, UserTask.user_id == user.id))
    ).scalar_one_or_none()
    if not user_task:
        raise HTTPException(status_code=404)

    task = db.execute(select(Task).where(Task.id == user_task.task_id)).scalar_one_or_none()
    if not task:
        raise HTTPException(status_code=404)

    existing = db.execute(
        select(Annotation).where(and_(Annotation.user_id == user.id, Annotation.user_task_id == user_task_id))
    ).scalar_one_or_none()

    existing_payload_json = "null"
    if existing:
        try:
            existing_payload_json = json.dumps(
                {
                    "cropmark": int(existing.cropmark),
                    "drawing": json.loads(existing.drawing_json or "[]"),
                    "brightness": float(existing.brightness),
                    "contrast": float(existing.contrast),
                },
                ensure_ascii=False,
            )
        except Exception:
            existing_payload_json = "null"

    total = int(db.execute(select(func.count()).select_from(UserTask).where(UserTask.user_id == user.id)).scalar_one())
    done = int(db.execute(select(func.count()).select_from(Annotation).where(Annotation.user_id == user.id)).scalar_one())
    remaining = max(0, total - done)

    # Prev/Next navigation in the user's randomized order (wrap-around).
    prev_user_task = db.execute(
        select(UserTask)
        .where(and_(UserTask.user_id == user.id, UserTask.display_order < user_task.display_order))
        .order_by(UserTask.display_order.desc())
        .limit(1)
    ).scalars().first()
    if not prev_user_task:
        prev_user_task = db.execute(
            select(UserTask)
            .where(UserTask.user_id == user.id)
            .order_by(UserTask.display_order.desc())
            .limit(1)
        ).scalars().first()

    next_user_task = db.execute(
        select(UserTask)
        .where(and_(UserTask.user_id == user.id, UserTask.display_order > user_task.display_order))
        .order_by(UserTask.display_order.asc())
        .limit(1)
    ).scalars().first()
    if not next_user_task:
        next_user_task = db.execute(
            select(UserTask)
            .where(UserTask.user_id == user.id)
            .order_by(UserTask.display_order.asc())
            .limit(1)
        ).scalars().first()

    image_url = f"/images/{task.rel_path}"

    return templates.TemplateResponse(
        "annotate.html",
        {
            "request": request,
            "task": task,
            "user_task": user_task,
            "prev_user_task": prev_user_task,
            "next_user_task": next_user_task,
            "remaining": remaining,
            "total": total,
            "image_url": image_url,
            "existing": existing,
            "existing_payload_json": existing_payload_json,
        },
    )


@app.post("/api/annotations/{user_task_id}")
async def save_annotation(user_task_id: int, request: Request, db=Depends(get_db)):
    user = require_user(request)

    ensure_user_task_list(db, user.id)

    user_task = db.execute(
        select(UserTask).where(and_(UserTask.id == user_task_id, UserTask.user_id == user.id))
    ).scalar_one_or_none()
    if not user_task:
        raise HTTPException(status_code=404)

    payload = await request.json()
    cropmark = int(payload.get("cropmark"))
    drawing = payload.get("drawing")
    brightness = float(payload.get("brightness", 100))
    contrast = float(payload.get("contrast", 100))

    if cropmark not in (0, 1, 2):
        raise HTTPException(status_code=400, detail="cropmark must be 0/1/2")

    if cropmark in (1, 2):
        if not isinstance(drawing, list) or len(drawing) == 0:
            raise HTTPException(status_code=400, detail="drawing required for cropmark 1/2")

    drawing_json = json.dumps(drawing if drawing is not None else [], ensure_ascii=False)

    existing = db.execute(
        select(Annotation).where(and_(Annotation.user_id == user.id, Annotation.user_task_id == user_task_id))
    ).scalar_one_or_none()

    if existing:
        existing.cropmark = cropmark
        existing.drawing_json = drawing_json
        existing.brightness = brightness
        existing.contrast = contrast
    else:
        db.add(
            Annotation(
                user_id=user.id,
                user_task_id=user_task_id,
                cropmark=cropmark,
                drawing_json=drawing_json,
                brightness=brightness,
                contrast=contrast,
            )
        )

    db.commit()
    return {"ok": True}
