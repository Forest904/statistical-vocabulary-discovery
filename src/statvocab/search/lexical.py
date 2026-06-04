"""SQLite FTS5 lexical indexing and BM25 retrieval."""

from __future__ import annotations

import re
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

TOKEN_RE = re.compile(r"[\w][\w'-]*", re.UNICODE)


@dataclass(frozen=True)
class LexicalHit:
    """One lexical retrieval candidate."""

    table_id: str
    score: float
    rank: float


def _connect(path: Path) -> sqlite3.Connection:
    connection = sqlite3.connect(path)
    connection.execute("PRAGMA journal_mode=DELETE")
    connection.execute("PRAGMA synchronous=NORMAL")
    return connection


def _fts_query(query: str) -> str:
    tokens = [match.group(0).replace('"', '""') for match in TOKEN_RE.finditer(query)]
    if not tokens:
        return ""
    return " OR ".join(f'"{token}"' for token in tokens[:32])


def build_lexical_index(rows: list[dict[str, object]], output_path: Path) -> Path:
    """Build a SQLite FTS5 index for title-only and all-vocabulary retrieval."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        output_path.unlink()
    wal_path = output_path.with_suffix(output_path.suffix + "-wal")
    shm_path = output_path.with_suffix(output_path.suffix + "-shm")
    for sidecar in (wal_path, shm_path):
        if sidecar.exists():
            sidecar.unlink()

    connection = _connect(output_path)
    try:
        connection.execute(
            """
            CREATE VIRTUAL TABLE search_docs USING fts5(
                table_id UNINDEXED,
                title,
                all_vocab,
                tokenize='unicode61 remove_diacritics 2'
            )
            """
        )
        connection.executemany(
            "INSERT INTO search_docs(table_id, title, all_vocab) VALUES (?, ?, ?)",
            [
                (
                    str(row["table_id"]),
                    str(row.get("title_lexical") or ""),
                    str(row.get("all_vocabulary_lexical") or ""),
                )
                for row in rows
            ],
        )
        connection.commit()
    finally:
        connection.close()
    return output_path


def search_lexical(
    index_path: Path,
    query: str,
    *,
    top_k: int,
    mode: Literal["title", "all"] = "all",
) -> list[LexicalHit]:
    """Search the FTS index and return higher-is-better BM25 scores."""

    match_query = _fts_query(query)
    if not match_query or not index_path.exists():
        return []
    column = "title" if mode == "title" else "all_vocab"
    sql = (
        "SELECT table_id, bm25(search_docs) AS rank "
        f"FROM search_docs WHERE {column} MATCH ? ORDER BY rank LIMIT ?"
    )
    connection = sqlite3.connect(index_path)
    try:
        rows = connection.execute(sql, (match_query, top_k)).fetchall()
    except sqlite3.OperationalError:
        return []
    finally:
        connection.close()
    return [
        LexicalHit(table_id=str(table_id), score=max(0.0, -float(rank)), rank=float(rank))
        for table_id, rank in rows
    ]
