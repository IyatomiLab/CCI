from __future__ import annotations

import csv
import math
from pathlib import Path
from typing import Any, Dict, List

BASE_ORDER = ["text", "model_id", "target_culture", "cci"]
PREFIX_ORDER = ["p_", "p_error"]


def _is_valid_number(value: Any) -> bool:
    """Return True only for non-NaN numeric values."""
    return isinstance(value, (int, float)) and not (
        isinstance(value, float) and math.isnan(value)
    )


def flatten_p(p: Dict[str, Any]) -> Dict[str, Any]:
    """Flatten {'Japan': 0.3, ...} into {'p_Japan': 0.3, ...}."""
    if not isinstance(p, dict):
        return {"p_error": f"type={type(p).__name__}"}
    if "error" in p:
        return {"p_error": str(p["error"])}

    out: Dict[str, Any] = {}
    for key, value in p.items():
        out[f"p_{key}"] = float(value) if _is_valid_number(value) else None
    return out


def build_output_row(
    text: str,
    model_id: str,
    target_culture: str,
    p: Dict[str, Any],
    cci: Any,
) -> Dict[str, Any]:
    """Build one output row and flatten per-culture scores into CSV-friendly columns."""
    row: Dict[str, Any] = {
        "text": text,
        "model_id": model_id,
        "target_culture": target_culture,
        "cci": cci,
    }
    row.update(flatten_p(p))
    return row


def _order_columns(keys: List[str]) -> List[str]:
    """Build deterministic header order: base keys -> prefixed groups -> remaining keys."""
    seen: set[str] = set()
    ordered: List[str] = []

    for key in BASE_ORDER:
        if key in keys and key not in seen:
            ordered.append(key)
            seen.add(key)

    sorted_keys = sorted(keys)
    for prefix in PREFIX_ORDER:
        for key in sorted_keys:
            if key not in seen and (key == prefix or key.startswith(prefix)):
                ordered.append(key)
                seen.add(key)

    for key in sorted_keys:
        if key not in seen:
            ordered.append(key)
            seen.add(key)

    return ordered


def save_output_csv(
    out_path: Path,
    new_rows: List[Dict[str, Any]],
) -> None:

    out_path.parent.mkdir(parents=True, exist_ok=True)

    existing: List[Dict[str, Any]] = []
    if out_path.exists():
        with out_path.open("r", encoding="utf-8", newline="") as f:
            existing.extend(csv.DictReader(f))

    all_rows = existing + new_rows
    all_keys: set[str] = set()
    for row in all_rows:
        all_keys.update(row.keys())

    fieldnames = _order_columns(list(all_keys))
    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in all_rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})
