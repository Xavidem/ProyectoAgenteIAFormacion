"""Configuracion comun para los tests del proyecto.

Al cargarse el paquete `tests`, este modulo:
  1) Anade `src/` al sys.path para poder importar los modulos del proyecto.
  2) Registra stubs para dependencias externas pesadas (sentence-transformers,
     qdrant-client, pymupdf, python-docx, langdetect, tqdm) de modo que los
     tests no requieran instalarlas en el venv local.

De esta forma cualquier test del paquete simplemente puede hacer
`from catalog.text_extractor import ...` sin trabajo adicional.
"""
from __future__ import annotations
import sys
import types
from pathlib import Path
from unittest.mock import MagicMock

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
# `chat_api.py` y `llm_service.py` viven bajo src/service/. Los containerfiles los
# copian planos a /app/, asi que en runtime se importan como `chat_api` directamente.
# Replicamos esa misma convencion en tests para no tener que reorganizar el codigo.
SERVICE = SRC / "service"
if str(SERVICE) not in sys.path:
    sys.path.insert(0, str(SERVICE))


def _ensure_stub(module_name: str, attrs: dict | None = None) -> types.ModuleType:
    if module_name in sys.modules:
        return sys.modules[module_name]
    mod = types.ModuleType(module_name)
    for k, v in (attrs or {}).items():
        setattr(mod, k, v)
    sys.modules[module_name] = mod
    return mod


_ensure_stub(
    "sentence_transformers",
    {
        "SentenceTransformer": MagicMock(
            return_value=MagicMock(
                get_sentence_embedding_dimension=lambda: 384,
                encode=MagicMock(return_value=MagicMock(tolist=lambda: [[0.0] * 384])),
            )
        )
    },
)

_qc = _ensure_stub("qdrant_client", {"QdrantClient": MagicMock(return_value=MagicMock())})

_QDRANT_MODEL_NAMES = (
    "Distance",
    "VectorParams",
    "PointStruct",
    "Filter",
    "FieldCondition",
    "MatchValue",
    "MatchText",
)
_qc_models = _ensure_stub("qdrant_client.models")
_qc_http = _ensure_stub("qdrant_client.http")
_qc_http_models = _ensure_stub("qdrant_client.http.models")
for _name in _QDRANT_MODEL_NAMES:
    sentinel = MagicMock(name=_name)
    if not hasattr(_qc_models, _name):
        setattr(_qc_models, _name, sentinel)
    if not hasattr(_qc_http_models, _name):
        setattr(_qc_http_models, _name, sentinel)

_qc_http_exc = _ensure_stub("qdrant_client.http.exceptions")
if not hasattr(_qc_http_exc, "UnexpectedResponse"):
    class _UnexpectedResponse(Exception):
        pass
    setattr(_qc_http_exc, "UnexpectedResponse", _UnexpectedResponse)

_ensure_stub("fitz", {"open": MagicMock()})
_ensure_stub("docx", {"Document": MagicMock()})


def _tqdm_passthrough(iterable=None, *args, **kwargs):
    return iterable if iterable is not None else iter([])


_ensure_stub("tqdm", {"tqdm": _tqdm_passthrough})

_ensure_stub(
    "langdetect",
    {
        "DetectorFactory": types.SimpleNamespace(seed=0),
        "LangDetectException": Exception,
        "detect": MagicMock(return_value="es"),
    },
)
