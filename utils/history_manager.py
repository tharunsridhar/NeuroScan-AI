from __future__ import annotations

import json
from pathlib import Path

from utils.config import HISTORY_FILE


def load_history(history_file: Path = HISTORY_FILE) -> list[dict]:
    if history_file.exists():
        try:
            return json.loads(history_file.read_text(encoding="utf-8"))
        except Exception:
            return []
    return []


def save_history(record: dict, history_file: Path = HISTORY_FILE) -> None:
    history = load_history(history_file)
    history.insert(0, record)
    history_file.write_text(json.dumps(history[:100], indent=2), encoding="utf-8")


def compare_with_prior(history: list, filename: str, current_area: float) -> dict:
    matches = [row for row in history if row.get('filename') == filename]
    prior_area = None
    for row in reversed(matches):
        area = row.get('area_cm2')
        if isinstance(area, (int, float)):
            prior_area = float(area)
            break
    if prior_area is None or prior_area <= 0:
        return {'prior_available': False, 'change_percent': None, 'progression_flag': 'Unknown'}
    change = (float(current_area) - prior_area) / prior_area * 100.0
    return {'prior_available': True, 'change_percent': round(float(change), 2), 'progression_flag': 'Progression' if change >= 20 else 'Regression' if change <= -20 else 'Stable'}
