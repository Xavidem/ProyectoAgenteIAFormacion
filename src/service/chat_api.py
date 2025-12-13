from fastapi import FastAPI
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any, cast
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from contextlib import asynccontextmanager
import unicodedata
import re
import requests
import json
import os
import time
import logging
import mem.store as memo
from urllib.parse import quote

def env_int(name: str, default: int) -> int:
    try: return int(os.getenv(name, str(default)))
    except: return default
def env_float(name: str, default: float) -> float:
    try: return float(name, str(default))
    except: return default

# Configuracion inicial de las constantes
MODEL_EMBEDDING_PATH = os.getenv("EMBEDDING_MODEL_PATH", "./models/all-MiniLM-L6-v2")
LLAVA_INFER_URL = os.getenv("LLAVA_INFER_URL", "http://llava-service:8001/infer")
QDRANT_HOST = os.getenv("QDRANT_HOST", "qdrant")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", "6333"))
COLLECTION_NAME = os.getenv("QDRANT_COLLECTION_NAME", "docs")
METADATA_PATH = os.getenv("METADATA_PATH", "metadata_master_1.json")
SEARCH_MULTIPLIER = int(os.getenv("SEARCH_MULTIPLIER", "6"))
PROMPT_CHAR_BUDGET = int(os.getenv("PROMPT_CHAR_BUDGET", "6000"))
LLAVA_READ_TIMEOUT = env_float("LLAVA_READ_TIMEOUT", 50.0)
LLAVA_RETRIES = env_int("LLAVA_RETRIES", 1)
OFFICIAL_LINKS_PATH = os.getenv("OFFICIAL_LINKS_PATH", "/app/official_urls.json")
LOCAL_DATA_ROOT = os.getenv("LOCAL_DATA_ROOT", "").replace("\\", "/").rstrip("/")
SHAREPOINT_BASE = os.getenv("SHAREPOINT_BASE", "").rstrip("/")
SHAREPOINT_UNDERSCORE_AS_SPACE = os.getenv("SHAREPOINT_UNDERSCORE_AS_SPACE", "1") == "1"
default_k = 5

@asynccontextmanager
async def lifespan(app: FastAPI):
    memo.init_db()
    try:
        memo.prune_old_sessions()
    except Exception:
        pass
    setattr(app.state, "official_links", load_official_links(OFFICIAL_LINKS_PATH))
    setattr(app.state, "memory_ready", True)
    yield

# Inicializar la app de FastAPI
app = FastAPI(
    title = "Chat API",
    description = "Endpoint /chat que une SBERT, Qdrant y LLaVA para responder preguntas",
    version = "1.0.0",
    lifespan=lifespan
)

# Carga del modelo de embeddings
embedder = SentenceTransformer(MODEL_EMBEDDING_PATH, local_files_only=True)

# Conectamos con Qdrant
qdrant = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)

# Cargamos metadatos locales
with open(METADATA_PATH, encoding='utf-8') as f:
    metadata = {rec['id']: rec for rec in json.load(f)}

logger = logging.getLogger("chat-api")
session = requests.Session()

# ========================================
# INICIO: PYDANTIC MODELS
# ========================================
class ChatRequest(BaseModel):
    query: str = Field(..., description="Texto de consulta del usuario.")
    k: int = Field(default_k, description="Numero de vecinos semanticos a recuperar.")
    threshold: float = Field(0.0, description="Score minimo para considerar un vecino.")
    session_id: Optional[str] = Field(None, description="Identificador de usuario en memoria")

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

# ========================================
# FIN: PYDANTIC MODELS
# ========================================

# ========================================
# INICIO: FUNCIONES DE MEMORIA
# ========================================
class SessionStartResp(BaseModel):
    session_id: str

class SessionClearReq(BaseModel):
    session_id: str

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

# ========================================
# FIN: FUNCIONES DE MEMORIA
# ========================================

# ========================================
# INICIO: FUNCIONES DE UTILIDAD
# ========================================
# Funcion para llamar a LLaVA
def call_llava(prompt: str, max_tokens: int = 64, temperature: float = 0.7) -> str:
    payload = {"prompt": prompt, "max_tokens": max_tokens, "temperature": temperature}
    attempts = [(max_tokens, LLAVA_READ_TIMEOUT), (max(32, max_tokens // 2), max(5.0, LLAVA_READ_TIMEOUT/2))]
    for i, (tok, to) in enumerate(attempts, 1):
        try:
            response = requests.post(LLAVA_INFER_URL, json=payload, timeout=LLAVA_READ_TIMEOUT)
            response.raise_for_status()
            data = response.json() or {}
            return data.get("text")
        except requests.exceptions.RequestException as e:
            logger.warning(f"LLaVA intento (tokens={tok}, timeout={to}s) falló: {e}")
            if i < len(attempts) and LLAVA_RETRIES:
                time.sleep(0.5)
            continue
    return None

def normalize_snippet(val) -> str:
    if isinstance(val, list):
        return " ".join(map(str, val))[:1000]
    if val is None:
        return ""
    return str(val)

def truncate_prompt(s: str, budget: int = PROMPT_CHAR_BUDGET) -> str:
    if len(s) <= budget:
        return s
    return s[:budget] + "\n[...truncado para ajustar el contexto...]"

def normalize_text(s: str) -> str:
    s = s.lower()
    s = unicodedata.normalize("NFKD", s)
    s = "".join(c for c in s if not unicodedata.combining(c))
    return s

# ========================================
# FIN: FUNCIONES DE UTILIDAD
# ========================================

# ========================================
# INICIO: CARGA DE LINKS
# ========================================

def load_official_links(path: str) -> Dict[str, Any]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
            topics = data.get("topics") or []
            for t in topics:
                t["__norm_keywords"] = [normalize_text(k) for k in (t.get("keywords") or [])]
            return {"topics": topics}
    except Exception:
        return {"topics": []}

def official_links_for_query(query: str, catalog: Dict[str, Any]) -> List[str] | None:
    q_norm = normalize_text(query)
    tokens = set(re.findall(r"\w+", q_norm))
    urls: List[str] = []
    seen = set()
    for topic in catalog.get("topics", []):
        kws = topic.get("__norm_keywords") or []
        matched = False
        for kw in kws:
            if kw in q_norm or kw in tokens:
                matched = True
                break
        if matched:
            for u in topic.get("urls", []):
                if u not in seen:
                    urls.append(u)
                    seen.add(u)
        if len(urls) >= 3:
            break
    return urls[:3] or None

# ========================================
# FIN: CARGA DE LINKS
# ========================================

# ========================================
# INICIO: URLS
# ========================================
SHAREPOINT_BASE = os.getenv("SHAREPOINT_BASE", "").rstrip("/")

"""DESCRIPCION DE LA FUNCION
Parsea la ruta local definida en el environment"""
def _parse_local_roots() -> list[str]:
    raw = os.getenv("LOCAL_DATA_ROOTS", "[]")
    try:
        roots = json.loads(raw)
        if isinstance(roots, str): roots = [roots]
    except json.JSONDecodeError:
        roots = [s.strip() for s in raw.replace(",", ";").split(";") if s.strip()]
    roots = [r.replace("\\", "/").rstrip("/") for r in roots if isinstance(r, str)]
    return sorted(roots, key=len, reverse=True)

LOCAL_DATA_ROOTS = _parse_local_roots()

"""DESCRIPCION DE LA FUNCION
Carga el mapa temático definido en environment"""
def _load_map(name: str) -> dict:
    raw = os.getenv(name, "")
    if not raw: return {}
    try: return json.loads(raw)
    except json.JSONDecodeError as e:
        logging.warning("%s inválido, usando {}. Error: %s", name, e)
        return {}

TOPIC_MAP = _load_map("SHAREPOINT_TOPIC_MAP")
ADD_WEB_PARAM = os.getenv("SHAREPOINT_OPEN_WEB", "1") == "1"

"""DESCRIPCION DE LA FUNCION
Helper que crea la url que va a ser usado por el chatbot"""
def path_to_sharepoint_url(local_path: str) -> str | None:
    if not local_path or not SHAREPOINT_BASE:
        return None

    p = (local_path or "").replace("\\", "/")
    lp = p.lower()

    # Elegimos prefijo
    rel = None
    for root in LOCAL_DATA_ROOTS:
        rl = root.lower()
        if lp == rl or lp.startswith(rl + "/"):
            rel = p[len(root):].lstrip("/")
            break

    # Fallback, cortamos la url donde queramos
    if rel is None:
        marker = "/proyecto-chatbot-materiales/pdfs/"
        idx = lp.find(marker)
        if idx >= 0:
            rel = p[idx + len(marker):].lstrip("/")

    if not rel:
        logging.warning("No pude mapear a SP: %s", p)
        return None

    parts = rel.split("/")
    if not parts:
        return None

    # Mapeamos la carpeta temática
    if parts[0] in TOPIC_MAP:
        parts[0] = TOPIC_MAP[parts[0]]

    # Creamos la url codificando cada segmento
    rel_enc = "/".join(quote(seg, safe="") for seg in parts)
    url = f"{SHAREPOINT_BASE}/{rel_enc}"
    if ADD_WEB_PARAM:
        url += "?web=1"
    return url
# ========================================
# INICIO: URLS
# ========================================

# ========================================
# INICIO: FUNCIONES DEL ENDPOINT
# ========================================
# Funcion para conocer el estado de LLaVa
@app.get("/health")
def health():
    ok_llava = False
    try:
        health_url = LLAVA_INFER_URL.replace("/infer", "/health")
        rr = requests.get(health_url, timeout=5)
        ok_llava = rr.ok
    except Exception:
        ok_llava = False
    return {"ok": True, "llava": ok_llava}

# Llamada al endpoint /chat
@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    # Primero comprobamos que se ha inicializado la base de datos
    if not getattr(app.state, "memory_ready", False):
        memo.init_db()
        setattr(app.state, "memory_ready", True)

    # Comprobamos que tenemos el catalogo de URLs
    if not hasattr(app.state, "official_links"):
        setattr(app.state, "official_links", load_official_links(OFFICIAL_LINKS_PATH))
        
    # Sesion
    sid = memo.start_session(request.session_id)

    # Memoria y contexto del prompt
    convo_ctx = memo.context_text(sid)

    # Generamos el embedding de la consulta
    query_vec = embedder.encode(request.query)

    # Metemos el catalogo de URLs
    catalog = getattr(app.state, "official_links", None)
    if catalog is None:
        catalog = load_official_links(OFFICIAL_LINKS_PATH)
        setattr(app.state, "official_links", catalog)
    curated_urls = official_links_for_query(request.query, catalog)

    # Conectamos con Qdrant
    hits = qdrant.search(
        collection_name=COLLECTION_NAME,
        query_vector=query_vec.tolist(),
        limit=max(request.k * 5, request.k),
        with_payload=True
    )
    
    best_by_doc = {}
    for hit in hits:
        if hit.score < request.threshold:
            continue
        pay = hit.payload or {}
        doc_id = pay.get("doc_id")
        if not doc_id:
            continue
        prev = best_by_doc.get(doc_id)
        if (prev is None) or (hit.score < prev[0]):
            best_by_doc[doc_id] = (hit.score, hit)

    top_docs = sorted(best_by_doc.items(), key=lambda x: x[1][0], reverse=True)[:request.k]

    documents = []
    for doc_id, (score, hit) in top_docs:
        pay = hit.payload or {}
        meta = metadata.get(doc_id, {})
        path = meta.get("path") or pay.get("path")
        sp_url = path_to_sharepoint_url(path) or path
        title = meta.get("title") or pay.get("title")
        snippet = pay.get("snippet")

        if not title and path:
            from pathlib import Path
            title = Path(path).name


        documents.append(DocumentItem(
            doc_id=doc_id,
            title=title,
            path=sp_url,
            score=score,
            snippet=snippet,
            chunk_id=pay.get("chunk_id", None)
        ))

    # Prompt para LLaVA
    prompt_parts = []
    if convo_ctx:
        prompt_parts.append("Contexto de conversacion (resumen + ultimos turnos):\n" + convo_ctx)
    prompt_parts.append(f"Consulta: {request.query}")
    if documents:
        prompt_parts.append("Resultados RAG (documentos relevantes): ")
        for d in documents:
            prompt_parts.append(f"- {d.title or d.path or d.doc_id} (score {d.score:.4f}) | {d.snippet or ''}")
    else:
        prompt_parts.append("No hay documentos internos relevantes.")
    prompt_text = "\n\n".join(prompt_parts)
    llava_resp = call_llava(prompt_text, max_tokens=256, temperature=0.4)
    if not llava_resp:
        if documents:
            resumen = "Aquí tienes tus documentos, ¡pregúntame sobre otro tema que quieras saber!"
            llava_resp = resumen
        else:
            llava_resp = "No hay generación de respuesta disponible ahora mismo."
    memo.append_message(sid, "user", request.query)
    memo.append_message(sid, "assistant", llava_resp)

    try:
        if memo.should_summarize(sid):
            full = memo.compact_history_text(sid)
            summary_prompt = (
                "Resume la conversacion de forma densa (<=150 palabras). "
                "Prioriza hechos, decisiones, restricciones, preferencias del usuario y contexto."
                "Texto:\n" + full
            )
            new_summary = call_llava(summary_prompt, max_tokens=220, temperature=0.2)
            if new_summary:
                memo.set_summary(sid, new_summary)
    except Exception:
        pass

    return ChatResponse(
        documents=documents,
        response= llava_resp or "No se pudo generar una respuesta",
        fallback_urls=curated_urls,
        session_id=sid
    )



