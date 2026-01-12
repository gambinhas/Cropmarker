from __future__ import annotations
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
    for site_dir in sorted([p for p in dataset_root.iterdir() if p.is_dir()], key=lambda p: p.name.lower()):
        site = site_dir.name
        images = [p for p in site_dir.iterdir() if p.is_file() and p.suffix.lower() == ".jpg"]
        for img in sorted(images, key=lambda p: p.name.lower()):
            yield site, img


def to_posix_relpath(path: Path) -> str:
    return path.as_posix()


def build_tasks(dataset_root: Path, qc_duplicates: int = 39, qc_seed: int = 42) -> list[TaskRow]:
    all_items: list[tuple[str, Path]] = list(iter_jpgs(dataset_root))
    if not all_items:
        raise RuntimeError(f"No .jpg files found under: {dataset_root}")

    tasks: list[TaskRow] = []
    for site, full_path in all_items:
        rel_path = to_posix_relpath(full_path.relative_to(dataset_root))
        tasks.append(
            TaskRow(
                site=site,
                rel_path=rel_path,
                filename=full_path.name,
                unique_id=rel_path,
                qc_reference=None,
                order=-1,
            )
        )

    tasks_sorted = sorted(tasks, key=lambda t: (t.site.lower(), t.filename.lower(), t.unique_id.lower()))
    tasks_final: list[TaskRow] = []
    for idx, t in enumerate(tasks_sorted, start=1):
        tasks_final.append(TaskRow(**{**t.__dict__, "order": idx}))
    return tasks_final
