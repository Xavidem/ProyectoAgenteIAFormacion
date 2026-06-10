"""Chat API: punto de entrada FastAPI que orquesta SBERT + Qdrant + LLaVA.

Este modulo expone:
  - POST /chat               -> consulta principal
  - POST /session/start      -> crea/recupera sesion
  - POST /session/clear      -> limpia historial de una sesion
  - POST /session/cleanup    -> elimina sesiones caducadas (TTL)
  - GET  /files/{doc_id}     -> sirve el documento original
  - GET  /health             -> liveness probe (incluye estado LLaVA)
  - GET  /admin/stats        -> conteo de docs/chunks y ultimo indexado
  - POST /admin/reindex      -> dispara la reindexacion (modo asincrono)
"""
from __future__ import annotations

import json
import logging
import math
import mimetypes
import re
import subprocess
import time
import unicodedata
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import requests
from fastapi import BackgroundTasks, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict
from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels
from qdrant_client.http.exceptions import UnexpectedResponse
from sentence_transformers import SentenceTransformer

import mem.store as memo


SYSTEM_PROMPT_ES = (
    "Eres un asistente de formacion. Responde EXCLUSIVAMENTE con la informacion contenida "
    "en la EVIDENCIA proporcionada por el sistema RAG. "
    "Si la evidencia no contiene la respuesta, di literalmente: "
    "\"No tengo informacion suficiente en los documentos disponibles.\" "
    "No inventes, no completes con conocimiento general, no cites fuentes externas."
)
SYSTEM_PROMPT_EN = (
    "You are a training assistant. Answer ONLY with information contained in the EVIDENCE "
    "provided by the RAG system. If the evidence does not contain the answer, say literally: "
    "\"I do not have enough information in the available documents.\" "
    "Do not invent, do not complete with general knowledge, do not cite external sources."
)
NO_EVIDENCE_RESPONSE_ES = "No tengo informacion suficiente en los documentos disponibles."
NO_EVIDENCE_RESPONSE_EN = "I do not have enough information in the available documents."


class Settings(BaseSettings):
    """Configuracion del servicio chat-api leida de variables de entorno o .env."""

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    log_level: str = Field("INFO", alias="LOG_LEVEL")

    embedding_model_path: str = Field(
        "./models/all-MiniLM-L6-v2", alias="EMBEDDING_MODEL_PATH"
    )

    llava_infer_url: str = Field(
        "http://llava-service:8001/infer", alias="LLAVA_INFER_URL"
    )
    llava_read_timeout: float = Field(50.0, alias="LLAVA_READ_TIMEOUT")
    llava_retries: int = Field(1, alias="LLAVA_RETRIES")
    llava_max_tokens_chat: int = Field(320, alias="LLAVA_MAX_TOKENS_CHAT")
    llava_max_tokens_summary: int = Field(220, alias="LLAVA_MAX_TOKENS_SUMMARY")

    qdrant_host: str = Field("qdrant", alias="QDRANT_HOST")
    qdrant_port: int = Field(6333, alias="QDRANT_PORT")
    qdrant_collection_name: str = Field("docs", alias="QDRANT_COLLECTION_NAME")

    search_multiplier: int = Field(6, alias="SEARCH_MULTIPLIER")
    prompt_char_budget: int = Field(1200, alias="PROMPT_CHAR_BUDGET")
    default_k: int = Field(5, alias="DEFAULT_K")

    # Diversificacion: lambda de MMR (1.0 = solo relevancia, 0.0 = solo diversidad).
    mmr_lambda: float = Field(0.7, alias="MMR_LAMBDA", ge=0.0, le=1.0)
    # Si es True, hasta 2 chunks por doc cuando el segundo score >= MULTI_CHUNK_RATIO * mejor.
    allow_multi_chunk: bool = Field(True, alias="ALLOW_MULTI_CHUNK")
    multi_chunk_ratio: float = Field(0.9, alias="MULTI_CHUNK_RATIO", ge=0.0, le=1.0)

    metadata_path: str = Field("metadata_master_1.json", alias="METADATA_PATH")
    official_links_path: str = Field("/app/official_urls.json", alias="OFFICIAL_LINKS_PATH")

    docs_root: str = Field("/app/data/pdfs", alias="DOCS_ROOT")

    cors_origins: str = Field("*", alias="CORS_ORIGINS")

    reindex_command: str = Field("", alias="REINDEX_COMMAND")

    @property
    def cors_origins_list(self) -> List[str]:
        if self.cors_origins.strip() == "*":
            return ["*"]
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]


settings = Settings()

logging.basicConfig(
    level=getattr(logging, settings.log_level.upper(), logging.INFO),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("chat-api")


class ChatRequest(BaseModel):
    query: str = Field(
        ...,
        min_length=1,
        max_length=2000,
        description="Texto de consulta del usuario.",
    )
    k: int = Field(default=5, ge=1, le=10, description="Numero de vecinos semanticos a recuperar.")
    threshold: float = Field(0.0, ge=0.0, le=1.0, description="Score minimo para considerar un vecino.")
    session_id: Optional[str] = Field(None, description="Identificador de usuario en memoria.")
    folder: Optional[str] = Field(
        None,
        max_length=200,
        description="Subcarpeta de data/pdfs para restringir la busqueda (p.ej. 'carpeta1').",
    )
    doc_type: Optional[str] = Field(
        None,
        pattern=r"^(pdf|docx)$",
        description="Filtrar por tipo de documento ('pdf' o 'docx').",
    )


class DocumentItem(BaseModel):
    doc_id: str
    title: Optional[str] = None
    path: Optional[str] = None
    snippet: Optional[str] = None
    chunk_id: Optional[int] = None
    score: float


class ChatResponse(BaseModel):
    documents: List[DocumentItem]
    response: str
    fallback_urls: Optional[List[str]] = None
    session_id: str


class SessionStartResp(BaseModel):
    session_id: str


class SessionClearReq(BaseModel):
    session_id: str


class StatsResponse(BaseModel):
    documents: int
    chunks: int
    last_indexed_at: Optional[str] = None
    collection: str


class ReindexResponse(BaseModel):
    triggered: bool
    detail: str


def normalize_text(s: str) -> str:
    s = s.lower()
    s = unicodedata.normalize("NFKD", s)
    s = "".join(c for c in s if not unicodedata.combining(c))
    return s


_LANG_HINTS = {
    "es": {
        "que", "como", "donde", "cuando", "cual", "para", "porque", "este",
        "esta", "estos", "estas", "una", "unos", "unas", "del", "los", "las",
        "necesito", "quiero", "puedo", "explicame", "ayudame", "muestrame",
        "dame", "sobre", "tambien", "aqui", "alli", "ejemplo", "ejercicio",
    },
    "en": {
        "what", "how", "where", "when", "which", "for", "because", "this",
        "that", "these", "those", "the", "and", "with", "i", "need", "want",
        "can", "show", "give", "explain", "about", "also", "here", "there",
        "example", "exercise",
    },
}


def detect_query_language(query: str) -> str:
    """Devuelve 'es' o 'en' (o '' si no se puede inferir)."""
    if not query:
        return ""
    norm = normalize_text(query)
    tokens = set(re.findall(r"\w+", norm))
    if not tokens:
        return ""
    scores = {lang: len(tokens & hints) for lang, hints in _LANG_HINTS.items()}
    best_lang, best_score = max(scores.items(), key=lambda x: x[1])
    return best_lang if best_score >= 1 else ""


def truncate_prompt(s: str, budget: Optional[int] = None) -> str:
    budget = budget or settings.prompt_char_budget
    if len(s) <= budget:
        return s
    return s[:budget] + "\n[...truncado para ajustar el contexto...]"


def load_official_links(path: str) -> Dict[str, Any]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        topics = data.get("topics") or []
        for t in topics:
            t["__norm_keywords"] = [normalize_text(k) for k in (t.get("keywords") or [])]
        return {"topics": topics}
    except FileNotFoundError:
        logger.warning("official_urls.json no encontrado en %s; fallback deshabilitado", path)
        return {"topics": []}
    except json.JSONDecodeError as e:
        logger.error("official_urls.json invalido (%s): %s", path, e)
        return {"topics": []}


def official_links_for_query(query: str, catalog: Dict[str, Any]) -> Optional[List[str]]:
    q_norm = normalize_text(query)
    tokens = set(re.findall(r"\w+", q_norm))
    urls: List[str] = []
    seen = set()
    for topic in catalog.get("topics", []):
        kws = topic.get("__norm_keywords") or []
        if any(kw in q_norm or kw in tokens for kw in kws):
            for u in topic.get("urls", []):
                if u not in seen:
                    urls.append(u)
                    seen.add(u)
        if len(urls) >= 3:
            break
    return urls[:3] or None


def _load_metadata_dict(path: str) -> Dict[str, Dict[str, Any]]:
    try:
        with open(path, encoding="utf-8") as f:
            recs = json.load(f)
    except FileNotFoundError:
        logger.warning("metadata_path no encontrado en %s; metadata vacia", path)
        return {}
    except json.JSONDecodeError as e:
        logger.error("metadata_path invalido (%s): %s", path, e)
        return {}
    return {rec["id"]: rec for rec in recs if isinstance(rec, dict) and rec.get("id")}


def call_llava(prompt: str, max_tokens: int = 64, temperature: float = 0.7) -> Optional[str]:
    """Llama al servicio LLaVA con reintentos. Devuelve None si todos los intentos fallan."""
    base = (max_tokens, settings.llava_read_timeout)
    fallback = (max(32, max_tokens // 2), max(5.0, settings.llava_read_timeout / 2))
    attempts = [base] + [fallback] * max(0, settings.llava_retries)

    for i, (tok, to) in enumerate(attempts, 1):
        payload = {"prompt": prompt, "max_tokens": tok, "temperature": temperature}
        try:
            response = requests.post(settings.llava_infer_url, json=payload, timeout=to)
            response.raise_for_status()
            data = response.json() or {}
            text = data.get("text")
            if text:
                return text
            logger.warning("LLaVA intento %d/%d devolvio respuesta vacia", i, len(attempts))
        except requests.exceptions.RequestException as e:
            logger.warning(
                "LLaVA intento %d/%d (tokens=%d, timeout=%.1fs) fallo: %s",
                i, len(attempts), tok, to, e,
            )
        if i < len(attempts):
            time.sleep(0.5)
    return None


def _build_qdrant_filter(folder: Optional[str], doc_type: Optional[str]) -> Optional[qmodels.Filter]:
    """Construye un Filter de Qdrant para 'folder' y 'doc_type'. None si no hay filtros."""
    must: List[qmodels.FieldCondition] = []
    if doc_type:
        must.append(
            qmodels.FieldCondition(
                key="ext",
                match=qmodels.MatchValue(value=doc_type.lower()),
            )
        )
    if folder:
        folder_clean = folder.strip().strip("/").strip("\\")
        if folder_clean:
            must.append(
                qmodels.FieldCondition(
                    key="path",
                    match=qmodels.MatchText(text=folder_clean),
                )
            )
    return qmodels.Filter(must=must) if must else None


def _cosine_sim(a: List[float], b: List[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a)) or 1.0
    nb = math.sqrt(sum(x * x for x in b)) or 1.0
    return dot / (na * nb)


def mmr_rerank(
    query_vec: List[float],
    candidates: List[Tuple[float, List[float], Any]],
    top_n: int,
    lambda_mult: float,
) -> List[Any]:
    """Maximal Marginal Relevance.

    candidates: lista de tuplas (score_relevancia, vector_chunk, payload_o_hit).
    Devuelve los `top_n` items reordenados favoreciendo diversidad.
    """
    if not candidates:
        return []
    if top_n >= len(candidates) or lambda_mult >= 1.0:
        return [c[2] for c in candidates[:top_n]]

    selected: List[Tuple[float, List[float], Any]] = []
    pool: List[Tuple[float, List[float], Any]] = list(candidates)

    pool.sort(key=lambda x: x[0], reverse=True)
    selected.append(pool.pop(0))

    while pool and len(selected) < top_n:
        best_idx = 0
        best_score = -float("inf")
        for i, (rel, vec, _) in enumerate(pool):
            sim_to_selected = max(_cosine_sim(vec, s[1]) for s in selected)
            score = lambda_mult * rel - (1.0 - lambda_mult) * sim_to_selected
            if score > best_score:
                best_score = score
                best_idx = i
        selected.append(pool.pop(best_idx))

    return [c[2] for c in selected]


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Carga recursos pesados una sola vez al arrancar la API."""
    logger.info("Iniciando chat-api")

    memo.init_db()
    try:
        n = memo.prune_old_sessions()
        logger.info("Limpieza inicial de sesiones expiradas: %d", n)
    except Exception:
        logger.exception("Fallo al limpiar sesiones expiradas en arranque")

    logger.info("Cargando modelo de embeddings desde %s", settings.embedding_model_path)
    app.state.embedder = SentenceTransformer(
        settings.embedding_model_path, local_files_only=True
    )

    logger.info("Conectando con Qdrant en %s:%d", settings.qdrant_host, settings.qdrant_port)
    app.state.qdrant = QdrantClient(host=settings.qdrant_host, port=settings.qdrant_port)

    logger.info("Cargando metadatos desde %s", settings.metadata_path)
    app.state.metadata = _load_metadata_dict(settings.metadata_path)

    logger.info("Cargando catalogo de URLs oficiales desde %s", settings.official_links_path)
    app.state.official_links = load_official_links(settings.official_links_path)

    app.state.memory_ready = True
    logger.info("chat-api listo")
    yield
    logger.info("chat-api apagandose")


app = FastAPI(
    title="Chat API",
    description="Endpoint /chat que une SBERT, Qdrant y LLaVA para responder preguntas",
    version="1.2.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def qdrant_search(
    client: QdrantClient,
    query_vector: List[float],
    limit: int,
    qfilter: Optional[qmodels.Filter] = None,
) -> List[Any]:
    """Busca vecinos en Qdrant usando el cliente nativo, con vectores en payload."""
    try:
        return client.search(
            collection_name=settings.qdrant_collection_name,
            query_vector=query_vector,
            limit=limit,
            with_payload=True,
            with_vectors=True,
            query_filter=qfilter,
        )
    except UnexpectedResponse as e:
        logger.error("Qdrant respondio con error inesperado: %s", e)
        raise HTTPException(status_code=502, detail="Qdrant no disponible") from e
    except Exception as e:
        logger.exception("Error inesperado consultando Qdrant")
        raise HTTPException(status_code=502, detail="Error consultando indice") from e


def qdrant_count(client: QdrantClient) -> int:
    try:
        res = client.count(
            collection_name=settings.qdrant_collection_name,
            exact=False,
        )
        return int(getattr(res, "count", 0))
    except Exception:
        logger.exception("No se pudo contar puntos en Qdrant")
        return 0


def _resolve_doc_disk_path(rel_or_abs: str) -> Path:
    if not rel_or_abs:
        raise HTTPException(status_code=404, detail="Fichero no existe en disco")

    docs_root = Path(settings.docs_root).resolve()
    p = Path(rel_or_abs)
    candidate = (p if p.is_absolute() else (docs_root / p)).resolve()

    try:
        candidate.relative_to(docs_root)
    except ValueError:
        raise HTTPException(status_code=403, detail="Ruta no permitida")

    if not candidate.exists() or not candidate.is_file():
        raise HTTPException(status_code=404, detail="Fichero no existe en disco")

    return candidate


@app.get("/health")
def health():
    ok_llava = False
    try:
        health_url = settings.llava_infer_url.replace("/infer", "/health")
        rr = requests.get(health_url, timeout=5)
        ok_llava = rr.ok
    except requests.exceptions.RequestException:
        ok_llava = False
    return {"ok": True, "llava": ok_llava}


@app.post("/session/start", response_model=SessionStartResp)
def session_start():
    sid = memo.start_session(None)
    return {"session_id": sid}


@app.post("/session/clear")
def session_clear(req: SessionClearReq):
    memo.clear_session(req.session_id)
    return {"ok": True}


@app.post("/session/cleanup")
def session_cleanup():
    n = memo.prune_old_sessions()
    return {"deleted_sessions": n}


def _consolidate_by_doc(
    hits_sorted: List[Any],
    threshold: float,
    k: int,
    allow_multi: bool,
    multi_ratio: float,
) -> List[Tuple[str, float, Dict[str, Any]]]:
    """Consolida los hits por doc_id respetando umbral.

    Si allow_multi: permite hasta 2 chunks por documento siempre que el segundo
    chunk tenga score >= multi_ratio * mejor_chunk_del_doc. Tope total = k.
    """
    grouped: Dict[str, List[Tuple[float, Dict[str, Any]]]] = {}
    for h in hits_sorted:
        score = float(getattr(h, "score", 0.0))
        if score < threshold:
            continue
        pay = getattr(h, "payload", None) or {}
        doc_id = pay.get("doc_id")
        if not doc_id:
            continue
        grouped.setdefault(doc_id, []).append((score, pay))

    selected: List[Tuple[str, float, Dict[str, Any]]] = []
    primary: List[Tuple[str, float, Dict[str, Any]]] = []
    secondary: List[Tuple[str, float, Dict[str, Any]]] = []
    for doc_id, items in grouped.items():
        items.sort(key=lambda x: x[0], reverse=True)
        best_score, best_pay = items[0]
        primary.append((doc_id, best_score, best_pay))
        if allow_multi and len(items) > 1:
            second_score, second_pay = items[1]
            if second_score >= multi_ratio * best_score:
                secondary.append((doc_id, second_score, second_pay))

    primary.sort(key=lambda x: x[1], reverse=True)
    selected = primary[:k]

    if allow_multi and len(selected) < k and secondary:
        secondary.sort(key=lambda x: x[1], reverse=True)
        for item in secondary:
            if len(selected) >= k:
                break
            selected.append(item)
        selected.sort(key=lambda x: x[1], reverse=True)

    return selected


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    sid = memo.start_session(request.session_id)
    convo_ctx = memo.context_text(sid)

    embedder: SentenceTransformer = app.state.embedder
    qdrant: QdrantClient = app.state.qdrant
    metadata: Dict[str, Any] = app.state.metadata

    query_vec = embedder.encode(request.query).tolist()

    qfilter = _build_qdrant_filter(request.folder, request.doc_type)
    search_limit = max(request.k * settings.search_multiplier, request.k)
    hits = qdrant_search(qdrant, query_vec, search_limit, qfilter=qfilter)

    if hits and 0.0 <= settings.mmr_lambda < 1.0:
        candidates: List[Tuple[float, List[float], Any]] = []
        for h in hits:
            score = float(getattr(h, "score", 0.0))
            if score < request.threshold:
                continue
            vec = getattr(h, "vector", None) or []
            if not isinstance(vec, list) or not vec:
                candidates.append((score, query_vec, h))
            else:
                candidates.append((score, vec, h))
        if candidates:
            hits = mmr_rerank(
                query_vec,
                candidates,
                top_n=min(len(candidates), request.k * 2),
                lambda_mult=settings.mmr_lambda,
            )

    top_docs = _consolidate_by_doc(
        hits_sorted=hits,
        threshold=request.threshold,
        k=request.k,
        allow_multi=settings.allow_multi_chunk,
        multi_ratio=settings.multi_chunk_ratio,
    )

    documents: List[DocumentItem] = []
    for doc_id, score, pay in top_docs:
        meta = metadata.get(doc_id, {})
        path = meta.get("path") or pay.get("path")
        title = meta.get("title") or pay.get("title")
        snippet = pay.get("snippet")
        if not title and path:
            title = Path(path).name
        documents.append(DocumentItem(
            doc_id=doc_id,
            title=title,
            path=path,
            score=score,
            snippet=snippet,
            chunk_id=pay.get("chunk_id"),
        ))

    lang = detect_query_language(request.query)
    if not documents:
        no_evidence_msg = NO_EVIDENCE_RESPONSE_EN if lang == "en" else NO_EVIDENCE_RESPONSE_ES
        memo.append_message(sid, "user", request.query)
        memo.append_message(sid, "assistant", no_evidence_msg)
        catalog = app.state.official_links
        curated_urls = official_links_for_query(request.query, catalog)
        return ChatResponse(
            documents=[],
            response=no_evidence_msg,
            fallback_urls=curated_urls,
            session_id=sid,
        )

    system_prompt = SYSTEM_PROMPT_EN if lang == "en" else SYSTEM_PROMPT_ES

    prompt_parts: List[str] = [system_prompt]
    if convo_ctx:
        prompt_parts.append("Contexto de conversacion (resumen + ultimos turnos):\n" + convo_ctx)
    prompt_parts.append(f"Consulta: {request.query}")
    prompt_parts.append("EVIDENCIA (documentos relevantes):")
    for d in documents:
        prompt_parts.append(
            f"- {d.title or d.path or d.doc_id} (score {d.score:.4f}) | {d.snippet or ''}"
        )
    prompt_parts.append("Responde solo con la EVIDENCIA anterior.")
    prompt_text = truncate_prompt("\n\n".join(prompt_parts))

    llava_resp = call_llava(
        prompt_text,
        max_tokens=settings.llava_max_tokens_chat,
        temperature=0.4,
    )
    if not llava_resp:
        llava_resp = (
            "No tengo informacion suficiente en los documentos disponibles."
            if lang != "en"
            else "I do not have enough information in the available documents."
        )

    memo.append_message(sid, "user", request.query)
    memo.append_message(sid, "assistant", llava_resp)

    try:
        if memo.should_summarize(sid):
            full = memo.compact_history_text(sid)
            summary_prompt = (
                "Resume la conversacion de forma densa (<=150 palabras). "
                "Prioriza hechos, decisiones, restricciones, preferencias del usuario y contexto. "
                "Texto:\n" + full
            )
            new_summary = call_llava(
                summary_prompt,
                max_tokens=settings.llava_max_tokens_summary,
                temperature=0.2,
            )
            if new_summary:
                memo.set_summary(sid, new_summary)
    except Exception:
        logger.exception("Fallo al resumir la conversacion para sesion %s", sid)

    return ChatResponse(
        documents=documents,
        response=llava_resp,
        fallback_urls=None,
        session_id=sid,
    )


@app.get("/files/{doc_id}")
def get_file(doc_id: str):
    rec = app.state.metadata.get(doc_id)
    if not rec:
        raise HTTPException(status_code=404, detail="Documento no encontrado")

    rel_path = rec.get("path") or (rec.get("extra") or {}).get("local_rel_path") or ""
    disk_path = _resolve_doc_disk_path(rel_path)

    media_type, _ = mimetypes.guess_type(str(disk_path))
    media_type = media_type or "application/octet-stream"
    headers = {"Content-Disposition": f'inline; filename="{disk_path.name}"'}

    return FileResponse(
        path=str(disk_path),
        media_type=media_type,
        filename=disk_path.name,
        headers=headers,
    )


@app.get("/admin/stats", response_model=StatsResponse)
def admin_stats():
    metadata: Dict[str, Any] = app.state.metadata
    n_docs = len(metadata)
    last_indexed = ""
    for rec in metadata.values():
        ts = (rec.get("extra") or {}).get("indexed_at") or ""
        if ts and ts > last_indexed:
            last_indexed = ts
    return StatsResponse(
        documents=n_docs,
        chunks=qdrant_count(app.state.qdrant),
        last_indexed_at=last_indexed or None,
        collection=settings.qdrant_collection_name,
    )


def _run_reindex_command(cmd: str) -> None:
    logger.info("Disparando reindexacion: %s", cmd)
    try:
        result = subprocess.run(
            cmd, shell=True, check=False, capture_output=True, text=True, timeout=3600,
        )
        if result.returncode != 0:
            logger.error("Reindexacion fallo (rc=%d): %s", result.returncode, result.stderr[:1000])
        else:
            logger.info("Reindexacion completada con exito")
    except Exception:
        logger.exception("Error ejecutando comando de reindexacion")


@app.post("/admin/reindex", response_model=ReindexResponse)
def admin_reindex(background_tasks: BackgroundTasks):
    cmd = (settings.reindex_command or "").strip()
    if not cmd:
        raise HTTPException(
            status_code=503,
            detail="REINDEX_COMMAND no configurado. Define la variable para habilitar este endpoint.",
        )
    background_tasks.add_task(_run_reindex_command, cmd)
    return ReindexResponse(triggered=True, detail="Reindexacion lanzada en segundo plano")
