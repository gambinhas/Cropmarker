# Cropmark Annotation GUI (`phd_cropmarks3.py`)

A **Tkinter/Pillow/OpenCV** tool for crowdsourced annotation of aerial images for **cropmark** detection. The goal is to classify each image as **No Mark (0)**, **Faint Mark (1)**, or **Clear Mark (2)** and, for positive cases, **draw** a rough outline/position of the cropmarks. Annotations are stored in a **per‑user Excel workbook**; the app supports **session restore** and injects a small set of **duplicates** for intra‑annotator quality control (QC).

> This GUI is **only** for data collection. Cross‑annotator validation and supervisory review are handled in a separate tool.

---

## Key features

- **Fast, minimal annotation:** 0/1/2; **drawing required** for 1 and 2; negatives can be saved without drawing.  
- **Navigation & inspection:** mouse‑wheel zoom, left‑button pan; adjustable brightness/contrast.  
- **Deterministic persistence:** pre‑populates the Excel sheet with the task list and **QC duplicates** (stable sample using `random.seed(42)`), fixed presentation order via `Order`, and resume at the exact point you left off.  
- **Portable:** all paths are relative to the script/executable directory; works with PyInstaller bundles.  
- **Lean and opinionated:** **JPEG only** (*.jpg*). Image adjustment limited to brightness/contrast (shown to be sufficient in practice).

---

## Requirements

- Python **3.9+**
- `pip install pillow opencv-python openpyxl`
- `tkinter` (bundled with standard CPython on Windows/macOS; on Linux install `python3-tk` via your distro)

---

## Folder structure

```
project/
├─ phd_cropmarks3.py
├─ image_dataset/
│  ├─ SITE_A/
│  │  ├─ source_monYear_site.jpg
│  │  └─ ...
│  └─ SITE_B/ ...
├─ resources/
│  ├─ checked.png
│  ├─ unchecked.png
│  └─ instructions.png
└─ annotations/            # created automatically
   └─ cropmarks_{username}_{score}.xlsx
```

- **`image_dataset/`**: subfolders per site; **only** `.jpg` files are considered.  
- **`resources/`**: radio‑button icons and the instructions image used by the GUI.  
- **`annotations/`**: per‑user Excel sheets and `app.log`.

---

## Quick start

```bash
python phd_cropmarks3.py
```

1) **Username**: on first run, enter your username.  
2) **Experience**: if it’s a new user, pick a level (0–5). This level is used **only** in the Excel filename (`cropmarks_{username}_{score}.xlsx`).  
3) The app **pre‑populates** the Excel with all found images and a sample of **QC duplicates**.  
4) It opens at the **first unsaved** task.

---

## Quality Control (QC)

- The app selects **up to 39** base images to **duplicate** (deterministic sampling with `seed=42`).  
- Each duplicate gets `UniqueID = "{filename}_qc"` and keeps `QC_Reference = filename` of the original.  
- Goal: measure **intra‑annotator consistency** without depending on the external validation flow.

> Inter‑annotator and gold‑standard assessments are **not** handled here—those belong to the supervisor tool.

---

## Controls

| Action                       | Mouse/Keyboard                             |
|-----------------------------|--------------------------------------------|
| Zoom                         | Mouse wheel                                |
| Pan                          | **Left button** drag                        |
| Draw (for 1/2)               | **Right button** press‑drag‑release         |
| Save                         | **S**                                       |
| Previous / Next image        | **A** / **D**                               |
| Delete drawing               | **Delete**                                  |
| Exit                         | **Esc**                                     |

- **Brightness/Contrast**: sliders at the bottom bar.  
- **Save** remains **disabled** for 1/2 until a drawing exists.

---

## Data schema (Excel)

Each row represents **one task** (original image or QC duplicate):

| Column          | Description                                                                 |
|-----------------|-----------------------------------------------------------------------------|
| `Site`          | Subfolder (site) name                                                       |
| `Date`          | `MM/YYYY` parsed from filename (`_mmmYYYY_`, PT/EN month names supported)   |
| `Cropmark`      | 0 = No Mark, 1 = Faint Mark, 2 = Clear Mark                                 |
| `Image`         | Filename (e.g., `source_oct2021_site.jpg`)                                  |
| `Saved`         | `"S"` when saved                                                            |
| `Drawing`       | JSON list of **strokes**; each stroke is a list of `(x, y)` pairs           |
| `QC_Reference`  | Original filename, when this row is a duplicate (`*_qc`)                    |
| `UniqueID`      | `filename` or `filename_qc`                                                 |
| `Order`         | Presentation index (deterministic session order)                            |

### `Drawing` format
```json
[
  [[x1, y1], [x2, y2], ...],      // stroke 1
  [[x1, y1], [x2, y2], ...]       // stroke 2, ...
]
```
Coordinates are in the **original image space** (independent of zoom/pan). On reload, drawings are re‑rendered over the image.

---

## Session flow

- **First run**: creates (or opens) the user’s Excel; lists all images and adds QC duplicates; defines `Order`.  
- **Subsequent runs**:  
  - if the Excel already exists, the app **rebuilds** the task list from the sheet (respecting `Order` and `Saved`), and  
  - **jumps** to the **first** row without `"S"`.  
- **Save** writes directly into the row matching the current `UniqueID`.

> If you move/rename images **after** launching the project structure, that’s on you: the app assumes a stable layout.

---

## Platform notes

- **DPI/scale**: fonts adapt to the OS scaling factor.  
- **Ultrawide 3440×1440**: window fixed at **1920×1080** (avoids layout artifacts); other resolutions launch maximized.  
- **PyInstaller**: path resolution uses `sys.executable` when frozen; otherwise `__file__`.

---

## Common issues (short answers)

- **Nothing happens on Save** → the Excel file is probably **open** in Excel (file lock). Close it and try again.  
- **“resources/… not found”** → `checked.png`, `unchecked.png`, or `instructions.png` missing from `resources/`.  
- **“No images found”** → ensure there are subfolders in `image_dataset/` with `.jpg` files.  
- **Lag with very large rasters** → this is a lean GUI; extremely large images and thick strokes can feel heavy. Simplify the view.

---

## Usage notes

- Ensure **unique filenames** across the dataset (already enforced in this collection).  
- Don’t change the **folder structure** after starting the campaign.  
- Keep `annotations/` under versioning or regular backups (each user has their own Excel).  
- Export to CSV/Parquet (outside this app) for statistical analysis.

---

## Running online (recommended): Label Studio (web)

If you want multiple people (different digital literacy levels) to annotate online, use **Label Studio**.
It provides a web UI, user accounts, and exports.

### What you get

- A simple browser-based workflow (open link → label 0/1/2 → draw polygon → submit).
- Centralized annotations (no per-user Excel files).
- Easy export to JSON/CSV.

### Files included in this folder

- `docker-compose.labelstudio.yml` — starts Label Studio with local file serving.
- `labelstudio_label_config.xml` — project UI (0/1/2 + polygon drawing).
- `labelstudio_tasks_generator.py` — generates tasks JSON from `image_dataset/` and injects QC duplicates.
- `labelstudio_export_to_csv.py` — converts Label Studio JSON export to a flat CSV.
- `start_labelstudio.ps1` — starts Label Studio without Docker (recommended on Windows if Docker/WSL is problematic).

### Step-by-step (Windows)

1) Install Docker Desktop (and enable WSL2 if asked).

2) Open a terminal in this folder (the one that contains `docker-compose.labelstudio.yml`).

3) Start Label Studio:

```bash
docker compose -f docker-compose.labelstudio.yml up -d
```

4) Open Label Studio in your browser:

- http://localhost:8080

5) Create your project:

- Create Project → set the **Labeling Interface** using the contents of `labelstudio_label_config.xml`.

6) Create tasks (with QC duplicates) from your dataset:

```bash
python labelstudio_tasks_generator.py --dataset-root image_dataset --out labelstudio_tasks.json --qc-duplicates 39 --local-files-prefix images
```

7) Import tasks:

- In the project → Data Manager → Import → upload `labelstudio_tasks.json`.

8) Ensure local storage is configured:

- Project → Settings → Cloud Storage → Add Source Storage → Local Files
- Absolute local path (inside container): `/label-studio/files/images`
- Import method: **Tasks** (because we import a JSON tasks file that references local images)

9) Create user accounts for annotators (optional but recommended):

- Settings → Users

10) Export annotations when done:

- Project → Export → JSON

11) Convert export JSON to CSV:

```bash
python labelstudio_export_to_csv.py --in export.json --out export.csv
```

---

### If Docker installation fails (no Docker path)

If you cannot install Docker Desktop (WSL/VirtualMachinePlatform issues), you can run Label Studio directly in Python.

Important (Windows): Label Studio local storage is much more reliable if `LABEL_STUDIO_LOCAL_FILES_DOCUMENT_ROOT` has **no spaces**.
Because this project lives under OneDrive (spaces), use the included drive-letter launcher.

1) Install Label Studio in the dedicated venv:

```powershell
./.venv-labelstudio/Scripts/python.exe -m pip install -U pip
./.venv-labelstudio/Scripts/python.exe -m pip install label-studio
```

2) Start Label Studio (no PowerShell scripts needed):

```bat
start_labelstudio.cmd
```

3) In Label Studio (web UI), configure Local Files storage (required for `/data/local-files/?d=...` to work):

- Project → Settings → Cloud Storage → Add Source Storage
- Storage type: **Local Files**
- Absolute local path: `L:\image_dataset` (use single backslashes in the UI)
- Import method: **Tasks**
- Click **Add Storage**

Do not click **Sync Storage** (we import tasks JSON manually).

4) Generate tasks:

```bash
python labelstudio_tasks_generator.py --dataset-root image_dataset --out labelstudio_tasks.json --local-files-prefix image_dataset --qc-duplicates 39
```

5) Import and annotate:

- Data Manager → Import → upload `labelstudio_tasks.json`

If images still don’t render, make sure you’re using `http://localhost:8080` (not `http://0.0.0.0:8080`).

---

### Multi-user over LAN (keep this PC on)

If you want other annotators to use their own computers while this machine stays on for a few days, run Label Studio on your PC and let them connect over the local network.

1) Start Label Studio in LAN mode:

```bat
start_labelstudio_lan.cmd
```

2) In Windows Firewall, allow inbound port 8080 (only if needed):

- Windows Security → Firewall & network protection → Advanced settings
- Inbound Rules → New Rule…
- Port → TCP → Specific local ports: `8080`
- Allow the connection
- Profile: Private (recommended)
- Name: `Label Studio 8080`

3) Share the URL with annotators:

- The launcher prints something like `http://192.168.x.y:8080`
- Annotators open that URL in a browser and log in.

4) Keep the PC awake for 3 days:

- Settings → System → Power & battery → Screen and sleep
- Set Sleep = Never (while plugged in)
- Avoid Windows restarts/updates during the campaign

Security note: LAN mode exposes Label Studio to anyone on the same network. Use strong passwords and only share the URL with your annotators.

---

### Docker / WSL troubleshooting (VirtualMachinePlatform)

If you want to retry Docker Desktop later, run PowerShell as Administrator and try:

```powershell
Enable-WindowsOptionalFeature -Online -FeatureName Microsoft-Windows-Subsystem-Linux -All -NoRestart
Enable-WindowsOptionalFeature -Online -FeatureName VirtualMachinePlatform -All -NoRestart
```

Reboot, then:

```powershell
wsl --update
```

If those features still fail to enable, check the DISM log referenced by the installer error.

### Notes

- This replicates your dataset structure (`image_dataset/<SITE>/*.jpg`) without uploading 381 images manually.
- The QC logic is deterministic via seed 42 and `*_qc` unique IDs.
- If filenames repeat across different site folders, this setup still works: `unique_id` is based on the relative path (`<SITE>/<file>.jpg`).
- Enforcing “drawing required when label is 1/2” is best handled by training + review; Label Studio doesn’t strictly enforce conditional UI constraints by default.

---

## License & citation

- License: choose one (e.g., MIT/Apache‑2.0).  
- Suggested citation (example):  
  > Leal, B. M. G. (2025). *Cropmark Annotation GUI* (vX.Y). GitHub repository. Available at: `<repo-url>`

---

## Contributions

Bug‑fix PRs, robustness improvements (e.g., cross‑platform wheel events), and helper scripts for export/aggregation are welcome. Avoid adding image controls that complicate the UI without clear detection gains.

---

### Data integrity checklist

- [ ] **Unique** filenames in the dataset.  
- [ ] `resources/` contains the three required images.  
- [ ] Do **not** open the Excel while annotating.  
- [ ] Periodically verify `annotations/cropmarks_{username}_{score}.xlsx`.
