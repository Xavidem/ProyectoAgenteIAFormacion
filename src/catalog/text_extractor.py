from __future__ import annotations
import os
import json
import uuid
import re
from pathlib import Path
from typing import List, Dict, Any
from io import BytesIO
from functools import lru_cache
from sentence_transformers import SentenceTransformer

import requests
import fitz
from docx import Document
from tqdm import tqdm

from sharepoint_fetcher import download_bytes

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    VectorParams,
    PointStruct,
    Filter,
    FieldCondition,
    MatchValue,
)

METADATA_PATH = Path(os.getenv("METADATA_PATH", "/app/metadata_master_1.json"))
QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", "6333"))
QDRANT_COLLECTION_NAME = os.getenv("QDRANT_COLLECTION_NAME", "docs")

# Chunking
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "280"))
CHUNK_STRIDE = int(os.getenv("CHUNK_STRIDE", "200"))

# Embeddings vía API externa (obligatoria)
EMBEDDINGS_URL = os.getenv("EMBEDDINGS_URL", "").strip()
EMBEDDING_DIM = int(os.getenv("EMBEDDING_DIM", "384"))
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL", "/app/models/all-MiniLM-L6-v2")
if not EMBEDDINGS_URL:
    raise RuntimeError(
        "[Indexer] EMBEDDINGS_URL no está definido; "
        "este indexador requiere un servicio de embeddings HTTP."
    )

# ========================================
# INICIO: CARGA DE DOCUMENTOS
# ========================================
def load_document_bytes(rec: dict) -> bytes:
    extra = rec.get("extra") or {}
    server_rel = extra.get("server_relative_url")

    if not server_rel:
        raise RuntimeError(
            f"[Indexer] Registro {rec.get('id')} sin server_relative_url en extra; "
            "este indexador sólo soporta documentos de SharePoint."
        )

    return download_bytes(server_rel)

# ========================================
# FIN: CARGA DE DOCUMENTOS
# ========================================

# ========================================
# INICIO: UTILIDADES DEL TEXTO
# ========================================
def clean_text(txt: str) -> str:
    txt = txt.replace("\x00", " ").replace("\r", "\n")
    txt = re.sub(r"[ \t]+", " ", txt)
    txt = re.sub(r"\n{3,}", "\n\n", txt)
    return txt.strip()


def chunk_tokens(tokens: List[str], size: int, stride: int) -> List[List[str]]:
    if size <= 0:
        return [tokens]
    out = []
    start = 0
    n = len(tokens)
    while start < n:
        end = min(start + size, n)
        out.append(tokens[start:end])
        if end == n:
            break
        step = size - stride if stride < size else size
        start = start + step
    return out

# ========================================
# INICIO: UTILIDADES DE TEXTO
# ========================================

# ========================================
# INICIO: EXTRACCION DE TEXTO
# ========================================
def extract_text_from_pdf_bytes(content: bytes) -> str:
    doc = fitz.open(stream=content, filetype="pdf")
    try:
        parts = []
        for page in doc:
            parts.append(page.get_text())
        return "".join(parts)
    finally:
        doc.close()


def extract_text_from_docx_bytes(content: bytes) -> str:
    bio = BytesIO(content)
    d = Document(bio)
    return "\n".join(p.text for p in d.paragraphs)


#def extract_text_from_pptx_bytes(content: bytes) -> str:
#    prs = Presentation(BytesIO(content))
#    texts = []
#    for slide in prs.slides:
#        for shape in slide.shapes:
#            if hasattr(shape, "text"):
#                texts.append(shape.text)
#    return "\n".join(texts)


# ========================================
# INICIO: EMBEDDINGS
# ========================================
@lru_cache(maxsize=1)
def _get_embedding_model():
    print(f"[Indexer] Cargando modelo de embeddings: {EMBEDDING_MODEL_NAME}")
    model = SentenceTransformer(EMBEDDING_MODEL_NAME, local_files_only=True)
    return model

def embed_texts(texts: list[str]) -> list[list[float]]:
    if not texts:
        return []

    model = _get_embedding_model()
    emb = model.encode(
        texts,
        batch_size=32,
        show_progress_bar=False,
        normalize_embeddings=True,
    )
    return emb.tolist()

# ========================================
# FIN: EMBEDDINGS
# ========================================

# ========================================
# INICIO: QDRANT
# ========================================
def ensure_collection(client: QdrantClient, size: int) -> None:
    exists = False
    try:
        info = client.get_collection(QDRANT_COLLECTION_NAME)
        exists = info is not None
    except Exception:
        exists = False
    if not exists:
        client.create_collection(
            collection_name=QDRANT_COLLECTION_NAME,
            vectors_config=VectorParams(size=size, distance=Distance.COSINE),
        )


def upsert_points(
    client: QdrantClient,
    vectors: List[List[float]],
    payloads: List[Dict[str, Any]],
    ids: List[str],
) -> None:
    points = [PointStruct(id=pid, vector=vec, payload=pl) for pid, vec, pl in zip(ids, vectors, payloads)]
    client.upsert(collection_name=QDRANT_COLLECTION_NAME, points=points)


def delete_doc_chunks(client: QdrantClient, doc_id: str) -> None:
    flt = Filter(must=[FieldCondition(key="doc_id", match=MatchValue(value=doc_id))])
    client.delete(collection_name=QDRANT_COLLECTION_NAME, points_selector=flt)


# ========================================
# FIN: QDRANT
# ========================================

# ========================================
# INICIO: PROCESO PRINCIPAL
# ========================================
def process_and_index():
    # Carga de metadatos
    if not METADATA_PATH.exists():
        raise FileNotFoundError(f"[Indexer] No existe METADATA_PATH: {METADATA_PATH}")
    records = json.loads(METADATA_PATH.read_text(encoding="utf-8"))
    if not isinstance(records, list):
        raise ValueError("[Indexer] metadata_master.json no es una lista JSON")

    # Cliente Qdrant
    client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
    ensure_collection(client, EMBEDDING_DIM)

    for rec in tqdm(records, desc="Indexando los documentos"):
        try:
            ext = (rec.get("type") or "").lower()
            # soportamos pdf, docx y pptx
            if ext not in {"pdf", "docx", "pptx"}:
                continue

            doc_id = rec["id"]
            extra_info = rec.get("extra") or {}
            display_path = rec.get("path") or ""
            title = rec.get("title") or os.path.basename(display_path)

            if not extra_info.get("server_relative_url"):
                print(f"[Indexer] WARN: sin server_relative_url para {display_path}, skipping")
                continue

            # Elimina chunks previos del doc
            delete_doc_chunks(client, doc_id)

            # Cargar bytes del documento (SharePoint)
            content = load_document_bytes(rec)

            # Extraer texto según extensión
            if ext == "pdf":
                full_text = extract_text_from_pdf_bytes(content)
            elif ext == "docx":
                full_text = extract_text_from_docx_bytes(content)
            #elif ext == "pptx":
            #    full_text = extract_text_from_pptx_bytes(content)
            else:
                continue

            # Limpieza y chunking
            cleaned = clean_text(full_text)
            if not cleaned:
                continue
            tokens = cleaned.split()
            chunks = chunk_tokens(tokens, CHUNK_SIZE, CHUNK_STRIDE)
            texts = [" ".join(chunk) for chunk in chunks if chunk]
            if not texts:
                continue

            # Embeddings
            vectors = embed_texts(texts)
            if not vectors:
                continue

            # Snippet del primer chunk
            snippet = " ".join(texts[0].split()[:40]) if texts else ""

            ids: List[str] = []
            payloads: List[Dict[str, Any]] = []

            for idx, vec in enumerate(vectors):
                pid = str(uuid.uuid5(uuid.UUID(doc_id), str(idx)))
                ids.append(pid)

                payloads.append({
                    "doc_id": doc_id,
                    "chunk_id": idx,
                    "title": title,
                    "path": display_path,             
                    "snippet": snippet,
                    "author": rec.get("author", ""),
                    "modified": rec.get("modified", ""),
                    "source": "sharepoint",
                    "sp_server_relative_url": extra_info.get("server_relative_url"),
                    "sp_length_bytes": extra_info.get("length_bytes"),
                })

            upsert_points(client, vectors, payloads, ids)

        except Exception as e:
            print(f"[Indexer] ERROR con {rec.get('path')}: {e}")

    print("[Indexer] Proceso completado.")


# ---------- main ----------
def main():
    process_and_index()

if __name__ == "__main__":
    main()

