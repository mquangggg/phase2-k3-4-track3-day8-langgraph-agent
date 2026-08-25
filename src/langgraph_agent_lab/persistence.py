"""Checkpointer adapter."""

from __future__ import annotations

from pathlib import Path

from langgraph.checkpoint.base import BaseCheckpointSaver


def build_checkpointer(
    kind: str = "memory",
    database_url: str | None = None,
) -> BaseCheckpointSaver | None:
    """Return a LangGraph checkpointer."""
    if kind == "none":
        return None
    if kind == "memory":
        from langgraph.checkpoint.memory import MemorySaver

        return MemorySaver()
    if kind == "sqlite":
        import sqlite3

        try:
            from langgraph.checkpoint.sqlite import (  # type: ignore[import-not-found]
                SqliteSaver,
            )
        except ImportError as exc:
            raise RuntimeError(
                "SQLite checkpoint backend requested but langgraph-checkpoint-sqlite "
                "is not installed. Install it with: pip install langgraph-checkpoint-sqlite"
            ) from exc


        db_path = database_url or "outputs/checkpoints.sqlite"
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(db_path, check_same_thread=False)
        return SqliteSaver(conn=conn)

    if kind == "postgres":
        raise NotImplementedError(
            "TODO(student): implement Postgres checkpointer (optional extension)"
        )
    raise ValueError(f"Unknown checkpointer kind: {kind}")

