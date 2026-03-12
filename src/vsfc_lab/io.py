from __future__ import annotations

import csv
import json
import logging
from pathlib import Path
from typing import Any


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def setup_logger(log_file: Path) -> logging.Logger:
    logger = logging.getLogger(f"vsfc_lab.{log_file.stem}")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    return logger


def dump_json(path: Path, obj: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as file:
        json.dump(obj, file, ensure_ascii=False, indent=2)


def append_jsonl(path: Path, obj: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(obj, ensure_ascii=False) + "\n")


def write_summary_csv(
    path: Path,
    rows: list[dict[str, Any]],
    default_fieldnames: list[str] | None = None,
) -> None:
    if rows:
        fieldnames = list(rows[0].keys())
    elif default_fieldnames:
        fieldnames = default_fieldnames
    else:
        fieldnames = ["metric", "mean", "std", "n_runs"]
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        if rows:
            writer.writerows(rows)
