from __future__ import annotations
import os
import csv
from typing import Dict


def save_csv(path: str, row: Dict[str, float]):
    os.makedirs(os.path.dirname(path), exist_ok=True)

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(row.keys())
        writer.writerow(row.values())


def latex_row(name: str, mean: float, std: float) -> str:
    return f"{name} & {mean:.6f} $\\pm$ {std:.6f} \\\\"