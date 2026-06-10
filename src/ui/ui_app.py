import logging
import os
import uuid
from typing import Any, Dict, List

import gradio as gr
import requests

logging.basicConfig(
    level=getattr(logging, os.getenv("LOG_LEVEL", "INFO").upper(), logging.INFO),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("ui-app")

BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000/chat")
BASE_URL = BACKEND_URL.rsplit("/chat", 1)[0]

FILES_PUBLIC_BASE = os.getenv("FILES_PUBLIC_BASE", "http://localhost:8000")

DEFAULT_THRESHOLD = float(os.getenv("UI_DEFAULT_THRESHOLD", "0.4"))
DEFAULT_K = int(os.getenv("UI_DEFAULT_K", "5"))

# ============================================================
#  INICIO: CSS
# ============================================================
CSS = """
#consulta-box textarea, #consulta-box input {
  background-color: #0b1220 !important;
  color: #f9fafb !important;
  border: 2px solid #2563eb !important;
}
#consulta-box textarea::placeholder, #consulta-box input::placeholder {
  color: #94a3b8 !important;
}
#consulta-box label {
  color: #e5e7eb !important;
  font-weight: 600;
}
#links-panel {
  border: 1px solid #334155;
  border-radius: 12px;
  padding: 10px 12px;
  background: #0b1220;
}
.footer-credit {
  font-size: 0.75em;
  color: #94a3b8;
  text-align: right;
  margin-top: 1rem;
}
"""

# ============================================================
#  INICIO: METODO DE SESION
# ============================================================
def ensure_session_id(current: str | None) -> str:
  if current:
    return current
  try:
    r = requests.post(f"{BASE_URL}/session/start", timeout=20)
    r.raise_for_status()
    return r.json().get("session_id") or str(uuid.uuid4())
  except requests.exceptions.RequestException as e:
    logger.warning("No se pudo crear sesion en el backend, usando UUID local: %s", e)
    return str(uuid.uuid4())
  
# ============================================================
#  INICIO: LLAMADA AL BACKEND
# ============================================================

class BackendError(Exception):
  """Error con un mensaje legible para mostrar en la UI."""

def call_backend(query: str, k: int, threshold: float, session_id: str) -> Dict[str, Any]:
  payload = {
    "query": query,
    "k": int(k),
    "threshold": float(threshold),
    "session_id": session_id,
  }
  try:
    r = requests.post(BACKEND_URL, json=payload, timeout=90)
  except requests.exceptions.ConnectionError as e:
    logger.error("No se pudo conectar al backend %s: %s", BACKEND_URL, e)
    raise BackendError(f"No se pudo conectar al backend ({BACKEND_URL}). Verifica que el servicio chat-api esté arrancado.") from e
  except requests.exceptions.Timeout as e:
    logger.error("Timeout llamando al backend: %s", e)
    raise BackendError("La respuesta del backend tardó demasiado (timeout). Intenta una consulta más corta o reduce 'k'.") from e
  except requests.exceptions.RequestException as e:
    logger.error("Fallo de red llamando al backend: %s", e)
    raise BackendError(f"Fallo de red llamando al backend: {e}") from e

  if not r.ok:
    detail = ""
    try:
      data = r.json()
      detail = data.get("detail") or data.get("error") or str(data)
    except Exception:
      detail = (r.text or "").strip()[:500]
    msg = f"Error {r.status_code} del backend: {detail or '(sin detalle)'}"
    logger.warning(msg)
    raise BackendError(msg)

  try:
    return r.json()
  except ValueError as e:
    logger.exception("El backend devolvió JSON inválido")
    raise BackendError("El backend devolvió una respuesta no válida (JSON malformado).") from e

# ============================================================
#  INICIO: FORMATEO DE RESULTADOS
# ============================================================

def format_documents_rows(documents: List[Dict[str, Any]]):
  rows = []
  for d in documents or []:
    title = d.get("title") or d.get("path") or d.get("doc_id") or "Sin título"
    snippet = d.get("snippet") or ""
    path = d.get("path") or ""
    score = d.get("score")
    try:
      score = round(float(score), 4) if score is not None else None
    except Exception:
      pass
    rows.append([title, snippet, path, score])
  return rows

def format_links_md(documents: List[Dict[str, Any]]) -> str:
  if not documents:
    return "_Escribe una consulta para ver enlaces a documentos._"
  items = []
  for d in documents:
    doc_id = d.get("doc_id")
    if not doc_id:
      continue
    title = d.get("title") or d.get("path") or doc_id
    url = f"{FILES_PUBLIC_BASE}/files/{doc_id}"
    items.append(f"- [{title}]({url})")
  return "### Enlaces a documentos\n" + ("\n".join(items) if items else "_No hay enlaces disponibles._")

def format_fallback_md(urls: List[str] | None) -> str:
  if not urls:
    return ""
  items = "\n".join([f"- {u}" for u in urls])
  return f"### Resultados de búsqueda web (fallback)\n{items}"

# ============================================================
#  INICIO: INTERFAZ GRADIO
# ============================================================
def on_submit(message: str, chat_history: list, k: int, threshold: float, sid: str):
  sid = ensure_session_id(sid)
  chat_history = chat_history or []

  if not (message or "").strip():
    return chat_history, gr.update(value=format_links_md([])), [], gr.update(value="", visible=False), sid, ""

  # Gradio Chatbot (formato mensajes dict)
  chat_history = chat_history + [
    {"role": "user", "content": message},
    {"role": "assistant", "content": "…"}
  ]

  try:
    data = call_backend(message, int(k), float(threshold), sid)
    assistant_text = (data.get("response") if isinstance(data, dict) else None) or "Aquí tienes los resultados:"
    documents = data.get("documents", []) if isinstance(data, dict) else []
    fallback_urls = data.get("fallback_urls")

    chat_history[-1] = {"role": "assistant", "content": assistant_text}

    docs_rows = format_documents_rows(documents)
    links_md = format_links_md(documents)

    fb_md = format_fallback_md(fallback_urls) if fallback_urls else ""
    fb_update = gr.update(value=fb_md, visible=bool(fallback_urls))

    return chat_history, gr.update(value=links_md), docs_rows, fb_update, sid, ""

  except BackendError as e:
    chat_history[-1] = {"role": "assistant", "content": str(e)}
    return chat_history, gr.update(value=format_links_md([])), [], gr.update(value="", visible=False), sid, ""
  except Exception as e:
    logger.exception("Error inesperado en on_submit")
    chat_history[-1] = {"role": "assistant", "content": f"Error inesperado en la UI: {e}"}
    return chat_history, gr.update(value=format_links_md([])), [], gr.update(value="", visible=False), sid, ""

def on_clear():
  return [], gr.update(value=format_links_md([])), [], gr.update(value="", visible=False), "", ""

with gr.Blocks(css=CSS, title="Agente IA - UI") as demo:
  gr.Markdown(
    "# Chatbot para la búsqueda de ficheros\n"
    "Encuentra los documentos que necesitas de forma rápida"
  )

  chatbot = gr.Chatbot(height=420, label="Chat")

  with gr.Row():
    k = gr.Slider(1, 10, value=DEFAULT_K, step=1, label="Número de documentos a recuperar")
    threshold = gr.Slider(0.0, 0.9, value=DEFAULT_THRESHOLD, step=0.05, label="Score mínimo de similitud")

  message = gr.Textbox(
    placeholder="Escribe tu consulta…",
    label="Consulta",
    elem_id="consulta-box"
  )

  with gr.Row():
    send = gr.Button("Enviar", variant="primary")
    clear = gr.Button("Limpiar chat")

  links_md = gr.Markdown(value=format_links_md([]), elem_id="links-panel")

  gr.Markdown("### Documentos recuperados")
  docs_df = gr.DataFrame(
    headers=["Título", "Snippet", "Ruta", "Score"],
    row_count=(0, "dynamic"),
    column_count=(4, "fixed"),
    wrap=True,
    interactive=False,
  )

  with gr.Accordion("URLs oficiales (fallback)", open=False):
    fb_md = gr.Markdown(visible=False)

  session_id_box = gr.Textbox(value="", visible=False)
  dummy_clear = gr.Textbox(value="", visible=False)  # para limpiar el textbox

  send.click(
    fn=on_submit,
    inputs=[message, chatbot, k, threshold, session_id_box],
    outputs=[chatbot, links_md, docs_df, fb_md, session_id_box, dummy_clear],
  )
  message.submit(
    fn=on_submit,
    inputs=[message, chatbot, k, threshold, session_id_box],
    outputs=[chatbot, links_md, docs_df, fb_md, session_id_box, dummy_clear],
  )
  clear.click(
    fn=on_clear,
    inputs=[],
    outputs=[chatbot, links_md, docs_df, fb_md, session_id_box, dummy_clear],
  )

  gr.Markdown("Creado por Javier Vals", elem_classes="footer-credit")

demo.launch(server_name="0.0.0.0", server_port=7860)
