"""Read prepared queue snapshots without importing acquisition/sequence drivers."""
import json
from pathlib import Path
from typing import Any, Dict


def load_prepared_queue(run_dir: Path, batch_id: str) -> Dict[str, Any]:
    normalized = str(batch_id or '').strip()
    if not normalized or any(character not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-' for character in normalized):
        raise ValueError('Invalid prepared queue id')
    path = Path(run_dir) / 'mid_fringe_schedules' / f'{normalized}.json'
    if not path.is_file():
        raise FileNotFoundError('Prepared mid-fringe queue was not found')
    payload = json.loads(path.read_text(encoding='utf-8'))
    if not isinstance(payload, dict) or payload.get('batch_id') != normalized:
        raise ValueError('Prepared mid-fringe queue is invalid')
    return payload
