"""Transactional index replacement; failures preserve the previous index."""
import hashlib
import sqlite3
from contextlib import closing
from threading import RLock

import numpy as np

from config import DATABASE_PATH, DOCUMENTS_PATH, EMBEDDING_MODEL, project_path
from ingest import create_chunks, load_documents, pdf_files, read_pdf_bytes
from vector_store import create_embeddings

INDEX_LOCK = RLock()


def connect(database_path=DATABASE_PATH):
    path = project_path(database_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    return sqlite3.connect(path, timeout=30)


def _schema(connection):
    connection.execute("""CREATE TABLE IF NOT EXISTS chunks (
        id INTEGER PRIMARY KEY AUTOINCREMENT, source TEXT NOT NULL,
        text TEXT NOT NULL, embedding BLOB NOT NULL, page INTEGER)""")
    columns = {row[1] for row in connection.execute("PRAGMA table_info(chunks)")}
    if "page" not in columns:
        connection.execute("ALTER TABLE chunks ADD COLUMN page INTEGER")
    connection.execute("CREATE TABLE IF NOT EXISTS metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL)")
    connection.execute("CREATE TABLE IF NOT EXISTS documents (source TEXT PRIMARY KEY, digest TEXT NOT NULL)")


def create_database(database_path=DATABASE_PATH):
    with closing(connect(database_path)) as connection, connection:
        _schema(connection)


def document_manifest(folder_path=DOCUMENTS_PATH):
    return {p.name: hashlib.sha256(read_pdf_bytes(p)).hexdigest() for p in pdf_files(folder_path)}


def rebuild_database(folder_path=DOCUMENTS_PATH, database_path=DATABASE_PATH):
    with INDEX_LOCK:
        manifest = document_manifest(folder_path)
        chunks = create_chunks(load_documents(folder_path))
        if not chunks:
            raise ValueError("İndekslenecek metin içeren PDF yok. Mevcut indeks korundu.")
        embeddings = np.asarray(create_embeddings(chunks), dtype=np.float32)
        if (embeddings.ndim != 2 or embeddings.shape[0] != len(chunks)
                or embeddings.shape[1] == 0 or not np.isfinite(embeddings).all()
                or (np.linalg.norm(embeddings, axis=1) == 0).any()):
            raise ValueError("Embedding çıktısı geçersiz. Mevcut indeks korundu.")
        if document_manifest(folder_path) != manifest:
            raise ValueError("İndeksleme sırasında PDF'ler değişti; yeniden deneyin.")
        with closing(connect(database_path)) as connection, connection:
            connection.execute("BEGIN IMMEDIATE")
            _schema(connection)
            connection.execute("DELETE FROM chunks")
            connection.executemany(
                "INSERT INTO chunks (source,text,embedding,page) VALUES (?,?,?,?)",
                [(c["source"], c["text"], e.tobytes(), c.get("page"))
                 for c, e in zip(chunks, embeddings)],
            )
            connection.execute("DELETE FROM documents")
            connection.executemany("INSERT INTO documents VALUES (?,?)", manifest.items())
            connection.execute("INSERT OR REPLACE INTO metadata VALUES ('embedding_model',?)", (EMBEDDING_MODEL,))
        return len(chunks)


def save_chunks():
    return rebuild_database()


def index_status(database_path=DATABASE_PATH, folder_path=DOCUMENTS_PATH):
    manifest = document_manifest(folder_path)
    if not project_path(database_path).exists():
        return {"chunks": 0, "documents": len(manifest), "stale": bool(manifest)}
    with closing(connect(database_path)) as connection:
        tables = {r[0] for r in connection.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        count = connection.execute("SELECT COUNT(*) FROM chunks").fetchone()[0] if "chunks" in tables else 0
        stored = dict(connection.execute("SELECT source,digest FROM documents")) if "documents" in tables else None
    return {"chunks": count, "documents": len(manifest), "stale": stored != manifest}


if __name__ == "__main__":
    print(f"Kaydedilen parça: {rebuild_database()}")
