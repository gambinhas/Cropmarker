import argparse
import csv
import json
from pathlib import Path
from typing import Any


def _get(obj: dict, key: str, default: Any = None) -> Any:
    return obj.get(key, default) if isinstance(obj, dict) else default


def extract_annotation_fields(annotation: dict) -> dict:
    result = _get(annotation, "result", []) or []

    cropmark_value = ""
    cropmark_from_name = ""

    drawings: list[dict] = []

    for item in result:
        itype = _get(item, "type", "")
        from_name = _get(item, "from_name", "")
        value = _get(item, "value", {}) or {}

        if itype == "choices":
            # Expected: value = {"choices": ["0"]} etc
            choices = _get(value, "choices", []) or []
            if choices and not cropmark_value:
                cropmark_value = str(choices[0])
                cropmark_from_name = from_name

        # Any shape tool
        if itype in {"polygonlabels", "brushlabels", "rectanglelabels", "ellipselabels", "keypointlabels", "polyline"}:
            drawings.append(item)

    return {
        "cropmark": cropmark_value,
        "cropmark_from_name": cropmark_from_name,
        "drawing_json": json.dumps(drawings, ensure_ascii=False),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert Label Studio JSON export to a flat CSV.")
    parser.add_argument("--in", dest="in_path", required=True, help="Path to Label Studio JSON export")
    parser.add_argument("--out", dest="out_path", default="labelstudio_export.csv", help="Output CSV path")
    args = parser.parse_args()

    in_path = Path(args.in_path)
    out_path = Path(args.out_path)

    data = json.loads(in_path.read_text(encoding="utf-8"))

    rows: list[dict] = []

    # Export can be list[task] or dict with "tasks"; handle both.
    tasks = data
    if isinstance(data, dict) and "tasks" in data:
        tasks = data["tasks"]

    if not isinstance(tasks, list):
        raise SystemExit("Unexpected export format: expected a list of tasks")

    for task in tasks:
        task_id = _get(task, "id", "")
        task_data = _get(task, "data", {}) or {}

        base = {
            "task_id": task_id,
            "unique_id": _get(task_data, "unique_id", ""),
            "qc_reference": _get(task_data, "qc_reference", ""),
            "site": _get(task_data, "site", ""),
            "filename": _get(task_data, "filename", ""),
            "order": _get(task_data, "order", ""),
            "image": _get(task_data, "image", ""),
        }

        annotations = _get(task, "annotations", []) or []
        if not annotations:
            rows.append(
                {
                    **base,
                    "annotation_id": "",
                    "completed_by": "",
                    "created_at": "",
                    "updated_at": "",
                    "cropmark": "",
                    "cropmark_from_name": "",
                    "drawing_json": "[]",
                }
            )
            continue

        for ann in annotations:
            ann_fields = extract_annotation_fields(ann)
            rows.append(
                {
                    **base,
                    "annotation_id": _get(ann, "id", ""),
                    "completed_by": _get(ann, "completed_by", ""),
                    "created_at": _get(ann, "created_at", ""),
                    "updated_at": _get(ann, "updated_at", ""),
                    **ann_fields,
                }
            )

    fieldnames = [
        "task_id",
        "unique_id",
        "qc_reference",
        "site",
        "filename",
        "order",
        "image",
        "annotation_id",
        "completed_by",
        "created_at",
        "updated_at",
        "cropmark",
        "cropmark_from_name",
        "drawing_json",
    ]

    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)

    print(f"Wrote {len(rows)} rows to: {out_path.resolve()}")


if __name__ == "__main__":
    main()
