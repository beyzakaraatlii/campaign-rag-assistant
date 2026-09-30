"""Cosine retrieval with exact-identifier priority and legacy-index support."""

import re
import sqlite3
from contextlib import closing

import numpy as np

from config import (
    DATABASE_PATH,
    EMBEDDING_MODEL,
    MIN_SIMILARITY,
    TOP_K,
    project_path,
)
from vector_store import encode_texts


def load_chunks_from_database(database_path=DATABASE_PATH):
    path = project_path(database_path)

    if not path.exists():
        return [], np.empty((0, 0), dtype=np.float32)

    with closing(
        sqlite3.connect(
            path.as_uri() + "?mode=ro",
            uri=True
        )
    ) as connection:

        columns = {
            row[1]
            for row in connection.execute(
                "PRAGMA table_info(chunks)"
            )
        }

        if not columns:
            return [], np.empty((0, 0), dtype=np.float32)

        tables = {
            row[0]
            for row in connection.execute(
                """
                SELECT name
                FROM sqlite_master
                WHERE type='table'
                """
            )
        }

        if "metadata" in tables:
            row = connection.execute(
                """
                SELECT value
                FROM metadata
                WHERE key='embedding_model'
                """
            ).fetchone()

            if row and row[0] != EMBEDDING_MODEL:
                raise ValueError(
                    "Embedding modeli değişmiş; "
                    "bilgi tabanını yeniden indeksleyin."
                )

        page_column = (
            "page"
            if "page" in columns
            else "NULL"
        )

        rows = connection.execute(
            f"""
            SELECT source, text, embedding, {page_column}
            FROM chunks
            ORDER BY id
            """
        ).fetchall()

    chunks = []
    embeddings = []

    for source, text, blob, page in rows:
        chunks.append({
            "source": source,
            "text": text,
            "page": page
        })

        embeddings.append(
            np.frombuffer(
                blob,
                dtype=np.float32
            )
        )

    if not embeddings:
        return [], np.empty((0, 0), dtype=np.float32)

    dimensions = {
        len(embedding)
        for embedding in embeddings
    }

    if len(dimensions) != 1:
        raise ValueError(
            "İndeksteki vektör boyutları tutarsız; "
            "yeniden indeksleyin."
        )

    return chunks, np.asarray(
        embeddings,
        dtype=np.float32
    )


def extract_identifiers(query):
    """
    LUNA-582, ORBIT-739 gibi özel kodları yakalar.

    Bu tür kodlar bir kaynakta birebir geçiyorsa,
    semantic similarity skorundan önce o kaynak
    önceliklendirilir.
    """

    return re.findall(
        r"\b[A-Za-zÇĞİÖŞÜçğıöşü]+-\d+\b",
        query
    )


def expand_query(query):
    """
    Genel reklam sorularını İngilizce kaynaklarla
    eşleşmeyi kolaylaştırmak için genişletir.

    Özel kampanya kodu içeren sorgular genişletilmez.
    """

    identifiers = extract_identifiers(query)

    if identifiers:
        return query

    query_lower = query.lower()

    creative_keywords = [
        "dikkat",
        "kreatif",
        "yaratıcı",
        "içerik",
        "video",
        "görsel",
        "ses",
        "reklam fikri",
    ]

    measurement_keywords = [
        "ölç",
        "performans",
        "kpi",
        "başarı",
        "dönüşüm",
        "cpa",
        "roi",
        "roas",
        "ctr",
        "metrik",
    ]

    audience_keywords = [
        "hedef kitle",
        "kitle",
        "persona",
        "müşteri",
        "segment",
        "target",
    ]

    if any(
        word in query_lower
        for word in creative_keywords
    ):
        expansion = (
            " creative attention hook engaging "
            "authentic sound video content"
        )

    elif any(
        word in query_lower
        for word in measurement_keywords
    ):
        expansion = (
            " measurement KPI performance conversion "
            "CPA attribution effectiveness"
        )

    elif any(
        word in query_lower
        for word in audience_keywords
    ):
        expansion = (
            " audience targeting target audience "
            "persona customer segmentation"
        )

    else:
        expansion = (
            " advertising campaign marketing"
        )

    return f"{query} {expansion}".strip()


def retrieve(
    query,
    database_path=DATABASE_PATH,
    top_k=TOP_K,
    min_score=MIN_SIMILARITY,
):
    if not query.strip():
        return []

    if top_k <= 0:
        return []

    chunks, embeddings = load_chunks_from_database(
        database_path
    )

    if not chunks:
        return []

    clean_query = query.strip()
    search_query = expand_query(clean_query)

    query_embedding = encode_texts(
        [search_query]
    )[0]

    if embeddings.ndim != 2:
        raise ValueError(
            "Embedding verisi geçersiz; "
            "bilgi tabanını yeniden indeksleyin."
        )

    if embeddings.shape[1] != query_embedding.size:
        raise ValueError(
            "Embedding boyutu uyumsuz; "
            "bilgi tabanını yeniden indeksleyin."
        )

    if not np.isfinite(embeddings).all():
        raise ValueError(
            "İndekste geçersiz embedding değerleri var; "
            "bilgi tabanını yeniden indeksleyin."
        )

    if not np.isfinite(query_embedding).all():
        raise ValueError(
            "Sorgu embedding'i geçersiz."
        )

    embedding_norms = np.linalg.norm(
        embeddings,
        axis=1
    )

    query_norm = np.linalg.norm(
        query_embedding
    )

    norms = embedding_norms * query_norm

    similarities = np.divide(
        embeddings @ query_embedding,
        norms,
        out=np.full(
            len(chunks),
            -1.0,
            dtype=np.float32
        ),
        where=norms > 0,
    )

    identifiers = [
        identifier.casefold()
        for identifier in extract_identifiers(
            clean_query
        )
    ]

    def has_exact_identifier(index):
        if not identifiers:
            return False

        text = chunks[index]["text"].casefold()

        return any(
            identifier in text
            for identifier in identifiers
        )

    def ranking_key(index):
        exact_match = has_exact_identifier(
            index
        )

        return (
            1 if exact_match else 0,
            float(similarities[index]),
        )

    ranked_indices = sorted(
        range(len(chunks)),
        key=ranking_key,
        reverse=True,
    )

    results = []

    for index in ranked_indices:

        exact_match = has_exact_identifier(
            index
        )

        similarity = float(
            similarities[index]
        )

        # Özel kod birebir eşleşiyorsa similarity
        # eşiğinin altında olsa bile kaynak korunur.
        if (
            not exact_match
            and similarity < min_score
        ):
            continue

        results.append({
            **chunks[index],
            "score": similarity,
        })

        if len(results) >= top_k:
            break

    return results


if __name__ == "__main__":
    question = input(
        "Reklam sorunu yaz: "
    )

    results = retrieve(
        question
    )

    for result in results:
        print(
            "\nKaynak:",
            result["source"]
        )

        print(
            "Sayfa:",
            result["page"]
        )

        print(
            "Benzerlik:",
            round(
                result["score"],
                4
            )
        )

        print(
            result["text"]
        )

        print("-" * 60)