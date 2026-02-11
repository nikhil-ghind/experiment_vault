from __future__ import annotations
import json
import os
import sqlite3
from dataclasses import dataclass, asdict, field
from datetime import datetime
from typing import Optional, List, Dict, Any


@dataclass
class Experiment:
    name: str
    model_type: str
    params: Dict[str, Any]
    metrics: Dict[str, float]
    artifact_path: str
    status: str = "completed"
    run_id: Optional[str] = None
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    tags: Dict[str, str] = field(default_factory=dict)


class ExperimentStore:
    """Lightweight SQLite-backed experiment store (wraps MLflow run IDs)."""

    def __init__(self, db_path: str = "experiments.db"):
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self._init_schema()

    def _init_schema(self):
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS experiments (
                id       INTEGER PRIMARY KEY AUTOINCREMENT,
                name     TEXT NOT NULL,
                run_id   TEXT,
                model_type TEXT,
                params   TEXT,
                metrics  TEXT,
                artifact_path TEXT,
                status   TEXT DEFAULT 'completed',
                tags     TEXT,
                created_at TEXT
            )
        """)
        self.conn.commit()

    def save(self, exp: Experiment) -> int:
        cur = self.conn.execute(
            """INSERT INTO experiments
               (name, run_id, model_type, params, metrics, artifact_path, status, tags, created_at)
               VALUES (?,?,?,?,?,?,?,?,?)""",
            (exp.name, exp.run_id, exp.model_type,
             json.dumps(exp.params), json.dumps(exp.metrics),
             exp.artifact_path, exp.status,
             json.dumps(exp.tags), exp.created_at)
        )
        self.conn.commit()
        return cur.lastrowid

    def list(self, model_type: str = None, status: str = None) -> List[Dict]:
        query = "SELECT * FROM experiments WHERE 1=1"
        params = []
        if model_type:
            query += " AND model_type=?"; params.append(model_type)
        if status:
            query += " AND status=?"; params.append(status)
        query += " ORDER BY id DESC"
        rows = self.conn.execute(query, params).fetchall()
        cols = [c[0] for c in self.conn.execute(query, params).description] if rows else []
        return [dict(zip(cols, r)) for r in rows]

    def best(self, metric: str = "val_loss", lower_is_better: bool = True) -> Optional[Dict]:
        rows = self.conn.execute(
            "SELECT * FROM experiments WHERE status='completed'"
        ).fetchall()
        if not rows:
            return None
        col_names = [c[0] for c in self.conn.execute(
            "SELECT * FROM experiments WHERE status='completed'"
        ).description]
        records = [dict(zip(col_names, r)) for r in rows]
        parsed = []
        for r in records:
            m = json.loads(r["metrics"] or "{}")
            if metric in m:
                r["_sort_val"] = m[metric]
                parsed.append(r)
        if not parsed:
            return None
        return sorted(parsed, key=lambda x: x["_sort_val"], reverse=not lower_is_better)[0]
