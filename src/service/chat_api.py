from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Optional, List, Dict
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
import requests
import json
import os
import time
from urllib.parse import quote

# Configuracion inicial de las constantes
MODEL_EMBEDDING_PATH = os.getenv("EMBEDDING_MODEL_PATH", "./models/all-MiniLM-L6-v2")
LLAVA_INFER_URL = os.getenv("LLAVA_INFER_URL", "http://localhost:8001/infer")
QDRANT_HOST = os.getenv("QDRANT_HOST", "qdrant")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", "6333"))
COLLECTION_NAME = os.getenv("QDRANT_COLLECTION_NAME", "docs")
METADATA_PATH = os.getenv("METADATA_PATH", "metadata_master.json")
SEARCH_MULTIPLIER = int(os.getenv("SEARCH_MULTIPLIER", "6"))
PROMPT_CHAR_BUDGET = int(os.getenv("PROMPT_CHAR_BUDGET", "6000"))
LLAVA_CONNECT_TIMEOUT = float(os.getenv("LLAVA_CONNECT_TIMEOUT", "5"))
LLAVA_READ_TIMEOUT = float(os.getenv("LLAVA_READ_TIMEOUT", "180"))

default_k = 5

# Inicializar la app de FastAPI
app = FastAPI(
    title = "Chat API",
    description = "Endpoint /chat que une SBERT, Qdrant y LLaVA para responder preguntas",
    version = "1.0.0"
)

# Carga del modelo de embeddings
embedder = SentenceTransformer(MODEL_EMBEDDING_PATH, local_files_only=True)

# Conectamos con Qdrant
qdrant = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)

# Cargamos metadatos locales
with open(METADATA_PATH, encoding='utf-8') as f:
    metadata = {rec['id']: rec for rec in json.load(f)}


# PYDANTIC MODELS
class ChatRequest(BaseModel):
    query: str = Field(..., description="Texto de consulta del usuario.")
    k: int = Field(default_k, description="Numero de vecinos semanticos a recuperar.")
    threshold: float = Field(0.0, description="Score minimo para considerar un vecino.")

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

"""FUNCIONES DE UTILIDAD"""
# Funcion para el fallback a la web
def fallback_web(query: str) -> List[str]:
    url = f"https://duckduckgo.com/?q={quote(query)}"
    return [url]


# Funcion para llamar a LLaVA
def call_llava(prompt: str, max_tokens: int = 64, temperature: float = 0.7) -> str:
    payload = {"prompt": prompt, "max_tokens": max_tokens, "temperature": temperature}
    last_error = None
    for _ in range(2):
        try:
            response = requests.post(LLAVA_INFER_URL, json=payload, timeout=(LLAVA_CONNECT_TIMEOUT, LLAVA_READ_TIMEOUT))
            response.raise_for_status()
            data = response.json()
            return data.get("text", "")
        except requests.exceptions.RequestException as e:
            last_error = e
            time.sleep(1)
    raise HTTPException(status_code=502, detail=f"Error al llamar a LLaVA: {str(last_error)}")

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

"""FUNCIONES DE ENDPOINTS"""
# Llamada al endpoint /chat
@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    # Generamos el embedding de la consulta
    query_vec = embedder.encode(request.query)

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
        title = meta.get("title") or pay.get("titlee")
        snippet = pay.get("snippet")

        if not title and path:
            from pathlib import Path
            title = Path(path).name

        documents.append(DocumentItem(
            doc_id=doc_id,
            title=title,
            path=path,
            score=score,
            snippet=snippet,
            chunk_id=pay.get("chunk_id", None)
        ))

    # Parte del fallback
    fallback_urls = None if documents else fallback_web(request.query)

    # Prompt para LLaVA
    prompt_parts = [f"Consulta: {request.query}"]
    prompt_parts += [ "Resultados encontrados:"] + [f" - {d.title or d.path or d.doc_id})" for d in documents] if documents else ["No se encontraron resultados relevantes, vamos con búsqueda en la web..."]
    llava_resp = call_llava("\n".join(prompt_parts))

    return ChatResponse(
        documents=documents,
        response= llava_resp or "No se pudo generar una respuesta",
        fallback_urls=fallback_urls
    )