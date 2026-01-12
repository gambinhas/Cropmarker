# Cropmarker Web (custom)

This is a small web app that mimics the **workflow rules** of the original Tkinter GUI:

- Cropmark choice: 0/1/2
- Save enabled only when a choice is selected
- For 1/2, drawing is required
- Right mouse button draws strokes
- Mouse wheel zoom + left button pan
- Brightness/contrast sliders

Additionally, it mimics the session logic:

- Each user gets a **random order** of images
- Each user gets **39 random QC duplicates** (total: 381 originals + 39 duplicates = 420)

## Quick start (Windows)

1) Open a terminal in `Cropmarker/`.
2) Initialize DB, import tasks, create users:

- Initialize:
  - `./webapp/.venv/Scripts/python.exe -m webapp.manage --db webapp/webapp_data/webapp.sqlite3 init-db`
- Import tasks:
  - `./webapp/.venv/Scripts/python.exe -m webapp.manage --db webapp/webapp_data/webapp.sqlite3 --dataset-root image_dataset import-tasks`
- Create an admin:
  - `./webapp/.venv/Scripts/python.exe -m webapp.manage --db webapp/webapp_data/webapp.sqlite3 create-user --username admin --admin`

3) Run:

- `webapp/run_webapp.cmd`

4) Open:

- http://localhost:8000

## Notes

- Login is **username-only** (no passwords). If it's the first time for that username, the app asks for an **expertise score** (0/1/3/5).
- Images are served from `Cropmarker/image_dataset` under `/images/<site>/<file>.jpg`.
- Annotations are stored in `Cropmarker/webapp/webapp_data/webapp.sqlite3`.

## Deploy online (Render)

This project includes a Render blueprint at `Cropmarker/render.yaml`.

Key points:
- You need **persistent storage** for both:
  - the SQLite DB (annotations)
  - the `image_dataset` folder (the JPEGs)
- Render persistent disks are mounted at `/var/data` in the blueprint.

### Steps

1) Put the `Cropmarker/` folder in a Git repo and push to GitHub.

2) In Render, create a **New Blueprint Instance** from your repo.

3) After the first deploy, use Render Shell to bootstrap data:
- Upload or download your dataset into `/var/data/image_dataset` (must contain subfolders per site).
- Import tasks into the DB:
  - `python -m webapp.manage --db /var/data/webapp.sqlite3 --dataset-root /var/data/image_dataset import-tasks`
- Create an admin user:
  - `python -m webapp.manage --db /var/data/webapp.sqlite3 create-user --username admin --admin`

4) Open the service URL and login.

Environment variables used:
- `CROPMARKER_SECRET_KEY` (required in production)
- `CROPMARKER_DB_PATH` (defaults to `webapp_data/webapp.sqlite3`)
- `CROPMARKER_DATASET_ROOT` (defaults to `../image_dataset`)
