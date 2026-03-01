from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

try:
    import numpy as np
except Exception:
    np = None


def _to_json_safe(obj: Any) -> Any:
    if np is not None:
        if isinstance(obj, np.generic):
            return obj.item()
        if isinstance(obj, np.ndarray):
            return obj.tolist()
    return str(obj)


def log_event(event: dict[str, Any], filename: str = "events.jsonl") -> None:
    root = Path(__file__).resolve().parents[1]
    log_dir = root / "logs"
    log_dir.mkdir(exist_ok=True)

    path = log_dir / filename

    payload = {
        "ts_utc": datetime.utcnow().isoformat(),
        **event,
    }

    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=False, default=_to_json_safe) + "\n")
