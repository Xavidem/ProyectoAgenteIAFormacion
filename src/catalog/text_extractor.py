from __future__ import annotations

import datetime as dt
import json
import os
import re
import uuid
from functools import lru_cache
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List, Optional

import fitz  # PyMuPDF
from docx import Document
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

try:
    from langdetect import DetectorFactory, LangDetectException, detect as _ld_detect

    DetectorFactory.seed = 0
except ImportError:  # pragma: no cover
    _ld_detect = None
    LangDetectException = Exception  # type: ignore[assignment, misc]

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchValue,
    PointStruct,
    VectorParams,
)


METADATA_PATH = Path(os.getenv("METADATA_PATH", "metadata_master_1.json")).expanduser().resolve()

QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", "6333"))
QDRANT_COLLECTION_NAME = os.getenv("QDRANT_COLLECTION_NAME", "docs")

CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "280"))
CHUNK_STRIDE = int(os.getenv("CHUNK_STRIDE", "200"))
STRUCTURAL_CHUNKING = os.getenv("STRUCTURAL_CHUNKING", "1") not in {"0", "false", "False", ""}

EMBEDDING_MODEL_NAME = os.getenv(
    "EMBEDDING_MODEL",
    "sentence-transformers/all-MiniLM-L6-v2",
)

DATA_ROOT = Path(os.getenv("DATA_ROOT", "/app/data/pdfs"))


def load_document_bytes(rel_path: str) -> bytes:
    rel_path = (rel_path or "").lstrip("/").replace("\\", "/")
    local_path = (DATA_ROOT / rel_path).resolve()

    if not str(local_path).startswith(str(DATA_ROOT.resolve())):
        raise ValueError(f"[Indexer] Ruta fuera de DATA_ROOT: {rel_path}")

    if not local_path.is_file():
        raise FileNotFoundError(f"[Indexer] No existe el fichero local: {local_path}")

    return local_path.read_bytes()


def clean_text(txt: str) -> str:
    txt = txt.replace("\x00", " ").replace("\r", "\n")
    txt = re.sub(r"[ \t]+", " ", txt)
    txt = re.sub(r"\n{3,}", "\n\n", txt)
    return txt.strip()


def chunk_tokens(tokens: List[str], size: int, stride: int) -> List[List[str]]:
    if size <= 0:
        return [tokens]
    out: List[List[str]] = []
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


def chunk_segments(
    segments: List[str],
    size: int,
    stride: int,
) -> List[List[str]]:
    """Agrupa segmentos (parrafos / bloques) en chunks de hasta `size` tokens.

    Respeta los limites estructurales: nunca corta dentro de un segmento; si un
    segmento por si solo supera `size`, se trocea con `chunk_tokens`. Se aplica
    solapamiento por palabras al inicio del siguiente chunk para no perder
    contexto en los bordes.
    """
    if size <= 0:
        joined = " ".join(segments).split()
        return [joined] if joined else []

    chunks: List[List[str]] = []
    current: List[str] = []
    overlap_words: List[str] = []

    for seg in segments:
        seg_tokens = seg.split()
        if not seg_tokens:
            continue

        if len(seg_tokens) > size:
            if current:
                chunks.append(current)
                overlap_words = current[-stride:] if 0 < stride < len(current) else []
                current = []
            for sub in chunk_tokens(seg_tokens, size, stride):
                if overlap_words:
                    candidate = overlap_words + sub
                    candidate = candidate[:size]
                    chunks.append(candidate)
                    overlap_words = []
                else:
                    chunks.append(sub)
            if chunks:
                last = chunks[-1]
                overlap_words = last[-stride:] if 0 < stride < len(last) else []
            continue

        if len(current) + len(seg_tokens) <= size:
            current.extend(seg_tokens)
        else:
            chunks.append(current)
            overlap_words = current[-stride:] if 0 < stride < len(current) else []
            current = list(overlap_words) + seg_tokens

    if current:
        chunks.append(current)

    return [c for c in chunks if c]


def detect_language(text: str, sample_chars: int = 4000) -> str:
    """Detecta idioma sobre una muestra del texto. Devuelve codigo ISO-639-1 o ''."""
    if not text or _ld_detect is None:
        return ""
    sample = text.strip()[:sample_chars]
    if len(sample) < 30:
        return ""
    try:
        return _ld_detect(sample)
    except LangDetectException:
        return ""


def _normalize_pdf_date(raw: Optional[str]) -> str:
    """Convierte fechas de PDF (D:YYYYMMDDHHmmSS...) a ISO 8601 cuando es posible."""
    if not raw:
        return ""
    s = raw.strip()
    if s.startswith("D:"):
        s = s[2:]
    digits = re.match(r"(\d{4})(\d{2})?(\d{2})?(\d{2})?(\d{2})?(\d{2})?", s)
    if not digits:
        return raw
    y, mo, d, h, mi, se = (digits.group(i) or "" for i in range(1, 7))
    iso = y
    if mo:
        iso += f"-{mo}"
    if d:
        iso += f"-{d}"
    if h:
        iso += f"T{h}"
        if mi:
            iso += f":{mi}"
        if se:
            iso += f":{se}"
    return iso


def _pdf_segments_from_blocks(doc) -> List[str]:
    """Extrae segmentos respetando paginas y bloques. Marca saltos de pagina con cadena vacia."""
    segments: List[str] = []
    for page in doc:
        try:
            blocks = page.get_text("blocks") or []
        except Exception:
            blocks = []
        page_segments: List[str] = []
        for blk in blocks:
            text = blk[4] if len(blk) >= 5 else ""
            if not isinstance(text, str):
                continue
            cleaned = re.sub(r"\s+", " ", text).strip()
            if cleaned:
                page_segments.append(cleaned)
        if not page_segments:
            try:
                full = page.get_text() or ""
            except Exception:
                full = ""
            for para in re.split(r"\n\s*\n", full):
                p = para.strip()
                if p:
                    page_segments.append(p)
        segments.extend(page_segments)
    return segments


def extract_pdf(content: bytes) -> Dict[str, Any]:
    doc = fitz.open(stream=content, filetype="pdf")
    try:
        segments = _pdf_segments_from_blocks(doc)
        full_text = "\n\n".join(segments)
        meta = doc.metadata or {}
        return {
            "text": full_text,
            "segments": segments,
            "author": (meta.get("author") or "").strip(),
            "created": _normalize_pdf_date(meta.get("creationDate")),
            "modified": _normalize_pdf_date(meta.get("modDate")),
        }
    finally:
        doc.close()


def extract_docx(content: bytes) -> Dict[str, Any]:
    bio = BytesIO(content)
    d = Document(bio)
    segments: List[str] = []
    for p in d.paragraphs:
        text = (p.text or "").strip()
        if text:
            segments.append(text)
    text = "\n".join(segments)
    cp = d.core_properties
    created = cp.created.isoformat() if getattr(cp, "created", None) else ""
    modified = cp.modified.isoformat() if getattr(cp, "modified", None) else ""
    author = (getattr(cp, "author", "") or "").strip()
    return {
        "text": text,
        "segments": segments,
        "author": author,
        "created": created,
        "modified": modified,
    }


@lru_cache(maxsize=1)
def _get_embedding_model() -> SentenceTransformer:
    print(f"[Indexer] Cargando modelo de embeddings: {EMBEDDING_MODEL_NAME}")
    return SentenceTransformer(EMBEDDING_MODEL_NAME)


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


def ensure_collection(client: QdrantClient) -> int:
    model = _get_embedding_model()
    dim = model.get_sentence_embedding_dimension()

    try:
        info = client.get_collection(QDRANT_COLLECTION_NAME)
        exists = info is not None
    except Exception:
        exists = False

    if not exists:
        client.create_collection(
            collection_name=QDRANT_COLLECTION_NAME,
            vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
        )
    return dim


def upsert_points(
    client: QdrantClient,
    vectors: List[List[float]],
    payloads: List[Dict[str, Any]],
    ids: List[str],
) -> None:
    points = [
        PointStruct(id=pid, vector=vec, payload=pl)
        for pid, vec, pl in zip(ids, vectors, payloads)
    ]
    client.upsert(collection_name=QDRANT_COLLECTION_NAME, points=points)


def delete_doc_chunks(client: QdrantClient, doc_id: str) -> None:
    flt = Filter(must=[FieldCondition(key="doc_id", match=MatchValue(value=doc_id))])
    client.delete(collection_name=QDRANT_COLLECTION_NAME, points_selector=flt)


def count_doc_chunks(client: QdrantClient, doc_id: str) -> int:
    flt = Filter(must=[FieldCondition(key="doc_id", match=MatchValue(value=doc_id))])
    try:
        res = client.count(
            collection_name=QDRANT_COLLECTION_NAME,
            count_filter=flt,
            exact=True,
        )
        return int(getattr(res, "count", 0))
    except Exception:
        return 0


def _save_metadata(records: List[Dict[str, Any]]) -> None:
    METADATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    METADATA_PATH.write_text(
        json.dumps(records, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _now_iso() -> str:
    return dt.datetime.now(tz=dt.timezone.utc).isoformat(timespec="seconds")


def _build_chunks(
    cleaned_text: str,
    segments: Optional[List[str]],
) -> List[List[str]]:
    """Devuelve chunks como listas de tokens, usando estructura cuando es posible."""
    if STRUCTURAL_CHUNKING and segments:
        struct = chunk_segments(segments, CHUNK_SIZE, CHUNK_STRIDE)
        if struct:
            return struct
    return chunk_tokens(cleaned_text.split(), CHUNK_SIZE, CHUNK_STRIDE)


def process_and_index() -> None:
    if not METADATA_PATH.exists():
        raise FileNotFoundError(f"[Indexer] No existe METADATA_PATH: {METADATA_PATH}")
    records = json.loads(METADATA_PATH.read_text(encoding="utf-8"))
    if not isinstance(records, list):
        raise ValueError("[Indexer] metadata_master_1.json no es una lista JSON")

    client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
    ensure_collection(client)

    n_total = len(records)
    n_skipped = 0
    n_indexed = 0
    n_failed = 0

    for rec in tqdm(records, desc="Indexando los documentos"):
        try:
            ext = (rec.get("type") or "").lower()
            if ext not in {"pdf", "docx"}:
                continue

            doc_id = rec["id"]
            display_path = rec.get("path") or ""
            title = rec.get("title") or os.path.basename(display_path)
            extra = rec.setdefault("extra", {})
            current_sha = extra.get("sha256") or ""
            indexed_sha = extra.get("indexed_sha256") or ""

            if current_sha and current_sha == indexed_sha and count_doc_chunks(client, doc_id) > 0:
                n_skipped += 1
                continue

            delete_doc_chunks(client, doc_id)

            content = load_document_bytes(display_path)

            if ext == "pdf":
                extracted = extract_pdf(content)
            elif ext == "docx":
                extracted = extract_docx(content)
            else:
                continue

            full_text = extracted.get("text") or ""
            if extracted.get("author") and not rec.get("author"):
                rec["author"] = extracted["author"]
            if extracted.get("created") and not rec.get("created"):
                rec["created"] = extracted["created"]
            if extracted.get("modified") and not rec.get("modified"):
                rec["modified"] = extracted["modified"]

            cleaned = clean_text(full_text)
            if not cleaned:
                continue

            if not rec.get("language"):
                detected_lang = detect_language(cleaned)
                if detected_lang:
                    rec["language"] = detected_lang

            chunks = _build_chunks(cleaned, extracted.get("segments"))
            texts = [" ".join(chunk) for chunk in chunks if chunk]
            if not texts:
                continue

            vectors = embed_texts(texts)
            if not vectors:
                continue

            ids: List[str] = []
            payloads: List[Dict[str, Any]] = []
            for idx, _ in enumerate(vectors):
                pid = str(uuid.uuid5(uuid.UUID(doc_id), str(idx)))
                ids.append(pid)
                snippet_idx = " ".join(texts[idx].split()[:40])
                payloads.append({
                    "doc_id": doc_id,
                    "chunk_id": idx,
                    "title": title,
                    "path": display_path,
                    "ext": ext,
                    "snippet": snippet_idx,
                    "author": rec.get("author", ""),
                    "modified": rec.get("modified", ""),
                    "language": rec.get("language", ""),
                    "source": "local",
                })

            upsert_points(client, vectors, payloads, ids)

            extra["chunks_count"] = len(vectors)
            extra["indexed_at"] = _now_iso()
            extra["indexed_sha256"] = current_sha
            n_indexed += 1

        except Exception as e:
            print(f"[Indexer] ERROR con {rec.get('path')}: {e}")
            n_failed += 1

    _save_metadata(records)

    print(
        f"[Indexer] Proceso completado. "
        f"total={n_total} indexados={n_indexed} saltados={n_skipped} fallidos={n_failed}"
    )


def main():
    process_and_index()


if __name__ == "__main__":
    main()
