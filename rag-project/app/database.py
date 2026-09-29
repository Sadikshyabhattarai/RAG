from pathlib import Path
import sqlite3
from typing import Any
import sqlite_vec

DATABASE_PATH = Path(__file__).resolve().parent.parent / "db" / "rag.db"
EMBEDDING_DIM = 768  # Standard for nomic-embed-text


def get_connection() -> sqlite3.Connection:
    DATABASE_PATH.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(DATABASE_PATH)
    connection.enable_load_extension(True)
    sqlite_vec.load(connection)
    connection.enable_load_extension(False)
    connection.execute("PRAGMA journal_mode = WAL;")
    connection.row_factory = sqlite3.Row
    return connection


def initialize_database() -> None:
    with get_connection() as connection:
        # Document chunks metadata table
        connection.execute("""
            CREATE TABLE IF NOT EXISTS chunks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                source TEXT NOT NULL,
                chunk_index INTEGER NOT NULL,
                content TEXT NOT NULL,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(source, chunk_index)
            );
        """)
        # Native sqlite-vec virtual table for vector similarity search
        connection.execute(f"""
            CREATE VIRTUAL TABLE IF NOT EXISTS vec_chunks USING vec0(
                chunk_id INTEGER PRIMARY KEY,
                embedding float[{EMBEDDING_DIM}] distance_metric=cosine
            );
        """)
        connection.commit()


def list_sources() -> list[dict[str, Any]]:
    with get_connection() as connection:
        rows = connection.execute("""
            SELECT source, COUNT(*) as chunk_count, MIN(created_at) as created_at
            FROM chunks
            GROUP BY source
        """).fetchall()
        return [dict(row) for row in rows]


def replace_source_chunks(source: str, chunk_records: list[dict[str, Any]]) -> int:
    with get_connection() as connection:
        cursor = connection.cursor()
        
        # 1. Clean up existing chunks and vectors for this source
        old_ids = cursor.execute("SELECT id FROM chunks WHERE source = ?", (source,)).fetchall()
        if old_ids:
            id_list = [row["id"] for row in old_ids]
            cursor.execute(f"DELETE FROM vec_chunks WHERE chunk_id IN ({','.join(['?']*len(id_list))})", id_list)
            cursor.execute("DELETE FROM chunks WHERE source = ?", (source,))

        # 2. Insert new chunks and vector entries
        for item in chunk_records:
            cursor.execute(
                "INSERT INTO chunks (source, chunk_index, content) VALUES (?, ?, ?)",
                (source, item["chunk_index"], item["content"]),
            )
            chunk_id = cursor.lastrowid

            cursor.execute(
                "INSERT INTO vec_chunks (chunk_id, embedding) VALUES (?, ?)",
                (chunk_id, sqlite_vec.serialize_float32(item["embedding"])),
            )

        connection.commit()
    return len(chunk_records)


def search_vector(query_embedding: list[float], limit: int = 4) -> list[dict[str, Any]]:
    """Native vector similarity search directly inside SQLite."""
    with get_connection() as connection:
        cursor = connection.cursor()
        rows = cursor.execute(
            """
            SELECT c.source, c.content, v.distance
            FROM vec_chunks v
            JOIN chunks c ON c.id = v.chunk_id
            WHERE v.embedding MATCH ? AND k = ?
            ORDER BY distance
            """,
            (sqlite_vec.serialize_float32(query_embedding), limit),
        ).fetchall()
        
        return [
            {"source": row["source"], "content": row["content"], "score": round(1.0 - row["distance"], 4)}
            for row in rows
        ]