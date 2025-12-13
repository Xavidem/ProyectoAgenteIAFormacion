import os
import requests
import re
import gradio as gr
import uuid
import json
from typing import List, Dict, Any

BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000/chat")
BASE_URL = BACKEND_URL.rsplit("/chat", 1)[0] #esto es para "/session/start"

def ensure_session_id(current: str | None) -> str:
    if current:
        return current
    try:
        r = requests.post(f"{BASE_URL}/session/start", timeout=20)
        r.raise_for_status()
        return r.json().get("session_id") or str(uuid.uuid4())
    except Exception:
        return str(uuid.uuid4())
    
def validate_corporate_email(email: str) -> tuple[str, str]:
    email = (email or "").strip().lower()
    if not email:
        return "Introduce tu correo corporativo para acceder.", ""
    if not email.endswith("@emeal.nttdata.com"):
        return "Correo no válido", ""
    return f"Sesión iniciada como {email}", email
# ============================================================
#  INICIO: FORMATEO DE DOCUMENTOS
# ============================================================

def format_documents_md(documents: List[Dict[str, Any]]):
    if not documents:
        return "Documentos\n No hay documentos disponibles"
    rows = []
    for d in documents or []:
        title = d.get("title") or d.get("path") or d.get("doc_id") or "Sin titulo"
        snippet = d.get("snippet") or "No hay vista previa del documento"
        path = d.get("path") or "No hay ruta disponible"
        score = d.get("score")
        try:
            score = round(float(score), 4) if score is not None else None
        except Exception:
            pass
        rows.append([title, snippet, path, score])
    return rows

def format_fallback_md(urls: List[str] | None) -> str:
    if not urls:
        return "No se encontraron URLs"
    items = "\n".join([f"- {u}" for u in urls])
    return f"### Resultados de busqueda web (fallback)\n{items}"

# ============================================================
#  FIN: FORMATEO DE DOCUMENTOS
# ============================================================

# ============================================================
#  INICIO: PRESETS
# ============================================================
def _load_presets() -> list[str]:
    raw = os.getenv("PROMPT_PRESETS", "")
    if not raw:
        return [
            "Necesito documentos de Postman",
            "Dame información sobre Java",
            "Dame material de información de Gherkin",
            "Dame documentos de Selenium",
            "Necesito que me des documentos de Hoppscotch",
            "Dame información sobre Bases de Datos"
        ]
    try:
        data = json.loads(raw)
        if isinstance(data, list):
            return [str(x) for x in data if isinstance(x, (str, int, float))]
    except json.JSONDecodeError:
        pass
    parts = [p.strip() for p in raw.replace(";", ",").split(",") if p.strip()]
    return parts or []

PROMPT_PRESETS = _load_presets()

def apply_presets(preset: str) -> str:
    return preset or ""

# ============================================================
#  FIN: PRESETS
# ============================================================

# ============================================================
#  INICIO: FORMATEO DE ENLACES y UI LOCAL
# ============================================================
def docs_link_md(documents: List[Dict[str, Any]]) -> str:
    if not documents:
        return ""
    lines = []
    for d in documents:
        title = d.get("title") or d.get("path") or d.get("doc_id") or "Documento"
        url = d.get("path") or ""
        if isinstance(url, str) and url.startswith("http"):
            lines.append(f"- [{title}]({url})")
        else:
            lines.append(f"- {title}: {url}")
    return "\n".join(lines)


def call_backend(query:str, k: int, threshold: float, session_id: str) -> Dict[str, Any]:
    payload = {"query": query, "k": int(k), "threshold": float(threshold), "session_id": session_id}
    r = requests.post(BACKEND_URL, json=payload, timeout=90)
    r.raise_for_status()
    return r.json()

def on_submit(message: str, chat_history: List, k: int, threshold: float, sid, user_email: str) -> str:
    sid = ensure_session_id(sid)
    if not message.strip():
        return chat_history, [], gr.update(value="", visible=False), sid, gr.update(value="Escribe una consulta...", visible=True)

    if not user_email:
        warning = "Debes iniciar sesión con tu correo corporativo antes de usar el chatbot."
        return chat_history + [(message, warning)], [], gr.update(value="", visible=False), sid, gr.update(value="", visible=False)
    
    chat_history = chat_history + [(message, "...")]

    try:
        data = call_backend(message, int(k), float(threshold), sid)
        assistant_text = data.get("response") if isinstance(data, dict) else None
        documents = data.get("documents", []) if isinstance(data, dict) else []
        fallback_urls = data.get("fallback_urls")

        assistant_text = assistant_text or "Aqui tienes los resultados de tu consulta: "

        docs_md = format_documents_md(documents)
        links_md = docs_link_md(documents)
        fb_md = format_fallback_md(fallback_urls) if fallback_urls else ""
        fb_update = gr.update(value=fb_md, visible=bool(fallback_urls))

        chat_history[-1] = (message, assistant_text)

        if links_md.strip():
            links_update = gr.update(value=links_md, visible=True)
        else:
            links_update = gr.update(value="", visible=False)
        
        return chat_history, docs_md, fb_update, sid, links_update
    except requests.RequestException as e:
        error = f"Error al llamar al backend: {e}"
        chat_history[-1] = (message, error)
        return chat_history, [], gr.update(value="", visible=False), sid, gr.update(value="", visible=False)


with gr.Blocks(
    css="""
#docs-panel {border: 1px solid #e5e7eb; border-radius: 12px; padding: 12px;}
#fb-panel {border: 1px dashed #e5e7eb; border-radius: 12px; padding: 12px; background: #fafafa;}

#consulta-box textarea {
  border: 2px solid #2563eb;
  background-color: #eff6ff;
}

#consulta-box label {
  color: #1d4ed8;
  font-weight: 600;
}

#presets-accordion > div {
  background-color: #f9fafb;
  border-radius: 12px;
  border: 1px dashed #d1d5db;
}

.footer-credit {
  font-size: 0.75em;
  color: #6b7280;
  text-align: right;
  margin-top: 1rem;
}
""",
    title="Agente IA - UI",
) as demo:
    gr.Markdown(
        "# Chatbot del departamento Q&A de NTT DATA \n"
        "Encuentra los documentos que necesitas de forma rápida"
    )

    user_email_state = gr.State("")

    with gr.Column(visible=True) as login_panel:
        gr.Markdown(
            "## Acceso al chatbot\n"
            "Introduce tu correo corporativo para usar el asistente."
        )
        with gr.Row():
            with gr.Column(scale=2):
                login_email = gr.Textbox(
                    label="Correo corporativo",
                    placeholder="tu.nombre@emeal.nttdata.com",
                )
            with gr.Column(scale=1):
                login_button = gr.Button("Acceder al chatbot", variant="secondary")
        login_status = gr.Markdown("")

    def do_login(email: str, current_email: str):
        msg, normalized = validate_corporate_email(email)
        if not normalized:
            return (
                msg,
                current_email,
                gr.update(visible=False), 
                gr.update(visible=True),   
            )
        # Login correcto
        return (
            msg,
            normalized,
            gr.update(visible=True),      
            gr.update(visible=False),  
        )

    with gr.Column(visible=False) as app_panel:
        chatbot = gr.Chatbot(height=420, label="Chat")

        with gr.Row():
            k = gr.Slider(
                1, 10, value=5, step=1,
                label="Número de documentos a recuperar"
            )
            threshold = gr.Slider(
                0.0, 0.9, value=0.0, step=0.05,
                label="Score mínimo de similitud"
            )

        message = gr.Textbox(
            placeholder="Escribe tu consulta...",
            label="Consulta",
            elem_id="consulta-box"
        )

        with gr.Row():
            send = gr.Button("Enviar", variant="primary")
            clear = gr.Button("Limpiar chat")

        with gr.Accordion(
            "Prompts sugeridos (opcional)",
            open=False,
            elem_id="presets-accordion"
        ):
            gr.Markdown(
                "Selecciona un prompt para rellenar la consulta de arriba "
                "y luego pulsa **Usar prompt y enviar**."
            )
            with gr.Row():
                presets = gr.Dropdown(
                    choices=PROMPT_PRESETS,
                    label="Selecciona un prompt",
                    interactive=True,
                    scale=3,
                )
                use_preset = gr.Button(
                    "Usar prompt y enviar",
                    variant="secondary",
                    scale=1,
                )

        gr.Markdown("### Enlaces a Documentos (SharePoint)")
        docs_links = gr.Markdown(visible=False)

        gr.Markdown("### Documentos recuperados")
        docs_md = gr.DataFrame(
            headers=["Título", "Snippet", "Ruta"],
            row_count=(0, "dynamic"),
            col_count=(3, "fixed"),
            wrap=True,
            interactive=False,
        )

        with gr.Accordion("URLs oficiales (fallback)", open=False):
            fb_md = gr.Markdown(visible=False)

        session_id_box = gr.Textbox(value="", visible=False)

        gr.Markdown("Creado por Javier Vals", elem_classes="footer-credit")

    login_button.click(
        fn=do_login,
        inputs=[login_email, user_email_state],
        outputs=[login_status, user_email_state, app_panel, login_panel],
    )
    send.click(
        fn=on_submit,
        inputs=[message, chatbot, k, threshold, session_id_box, user_email_state],
        outputs=[chatbot, docs_md, fb_md, session_id_box, docs_links],
    )
    message.submit(
        fn=on_submit,
        inputs=[message, chatbot, k, threshold, session_id_box, user_email_state],
        outputs=[chatbot, docs_md, fb_md, session_id_box, docs_links],
    )
    presets.change(
        fn=apply_presets,
        inputs=[presets],
        outputs=[message],
    )

    use_preset.click(
        fn=on_submit,
        inputs=[message, chatbot, k, threshold, session_id_box, user_email_state],
        outputs=[chatbot, docs_md, fb_md, session_id_box, docs_links],
    )


    gr.Markdown("Creado por Javier Vals", elem_classes="footer-credit")

demo.launch(server_name="0.0.0.0", server_port=7860)

        