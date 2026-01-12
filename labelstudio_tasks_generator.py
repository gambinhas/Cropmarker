import argparse
import json
import os
import random
import urllib.parse
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


@dataclass(frozen=True)
class TaskRow:
    site: str
    rel_path: str  # relative to dataset root (posix style)
    filename: str
    unique_id: str
    qc_reference: str | None
    order: int


def iter_jpgs(dataset_root: Path) -> Iterable[tuple[str, Path]]:
    """Yields (site, full_path) for all .jpg under dataset_root/site/*.jpg"""
    for site_dir in sorted([p for p in dataset_root.iterdir() if p.is_dir()], key=lambda p: p.name.lower()):
        site = site_dir.name
        images = [p for p in site_dir.iterdir() if p.is_file() and p.suffix.lower() == ".jpg"]
        for img in sorted(images, key=lambda p: p.name.lower()):
            yield site, img


def to_posix_relpath(path: Path) -> str:
    return path.as_posix()


def build_tasks(
    dataset_root: Path,
    qc_duplicates: int,
    qc_seed: int,
) -> list[TaskRow]:
    all_items: list[tuple[str, Path]] = list(iter_jpgs(dataset_root))
    if not all_items:
        raise SystemExit(f"No .jpg files found under: {dataset_root}")

    # Base tasks
    tasks: list[TaskRow] = []
    for site, full_path in all_items:
        rel_path = to_posix_relpath(full_path.relative_to(dataset_root))
        tasks.append(
            TaskRow(
                site=site,
                rel_path=rel_path,
                filename=full_path.name,
                # Use relative path to guarantee uniqueness even when filenames repeat across sites
                unique_id=rel_path,
                qc_reference=None,
                order=-1,
            )
        )

    # Deterministic QC selection
    random.seed(qc_seed)
    candidates = [full_path for _, full_path in all_items]
    candidates_sorted = sorted(candidates, key=lambda p: p.as_posix().lower())
    to_sample = min(qc_duplicates, len(candidates_sorted))
    sampled = random.sample(candidates_sorted, to_sample)

    for full_path in sampled:
        rel_path = to_posix_relpath(full_path.relative_to(dataset_root))
        tasks.append(
            TaskRow(
                site=full_path.parent.name,
                rel_path=rel_path,
                filename=full_path.name,
                unique_id=f"{rel_path}_qc",
                qc_reference=rel_path,
                order=-1,
            )
        )

    # Stable order
    tasks_sorted = sorted(tasks, key=lambda t: (t.site.lower(), t.filename.lower(), 1 if t.qc_reference else 0, t.unique_id.lower()))
    tasks_final: list[TaskRow] = []
    for idx, t in enumerate(tasks_sorted, start=1):
        tasks_final.append(TaskRow(**{**t.__dict__, "order": idx}))
    return tasks_final


def make_labelstudio_json(tasks: list[TaskRow], local_files_prefix: str) -> list[dict]:
    """Creates Label Studio importable JSON.

    local_files_prefix should match how you mount your dataset in Label Studio.

    Example (Docker mount):
      -v C:\\...\\image_dataset:/data/images
    Then local_files_prefix could be: images

    Task will reference:
      /data/local-files/?d=images/SITE_A/file.jpg
    """
    prefix = local_files_prefix.strip("/ ")
    out: list[dict] = []
    for t in tasks:
        # Label Studio local files serving uses POSIX paths
        d_value = f"{prefix}/{t.rel_path}"
        d_value_encoded = urllib.parse.quote(d_value, safe="/")
        image_ref = f"/data/local-files/?d={d_value_encoded}"
        out.append(
            {
                "data": {
                    "image": image_ref,
                    "site": t.site,
                    "filename": t.filename,
                    "unique_id": t.unique_id,
                    "qc_reference": t.qc_reference,
                    "order": t.order,
                }
            }
        )
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate Label Studio tasks from Cropmarker image_dataset.")
    parser.add_argument("--dataset-root", default="image_dataset", help="Path to image_dataset folder")
    parser.add_argument("--out", default="labelstudio_tasks.json", help="Output JSON file path")
    parser.add_argument("--qc-duplicates", type=int, default=39, help="How many QC duplicates to inject")
    parser.add_argument("--qc-seed", type=int, default=42, help="Seed for deterministic QC selection")
    parser.add_argument(
        "--local-files-prefix",
        default="images",
        help="Prefix used by Label Studio local-files serving (matches Docker mount under /data).",
    )
    args = parser.parse_args()

    dataset_root = Path(args.dataset_root).resolve()
    tasks = build_tasks(dataset_root, qc_duplicates=args.qc_duplicates, qc_seed=args.qc_seed)
    payload = make_labelstudio_json(tasks, local_files_prefix=args.local_files_prefix)

    out_path = Path(args.out).resolve()
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote {len(payload)} tasks to: {out_path}")


if __name__ == "__main__":
    main()
