import os, sqlite3, uuid, time
from pathlib import Path
from typing import List, Optional, Tuple

DB_PATH = os.getenv("MEMORY_DB_PATH", "/app/memory.sqlite")
TTL_DAYS = int(os.getenv("MEMORY_TTL_DAYS", "5"))
_MAX_MSGS_FOR_CONTEXT = int(os.getenv("MEMORY_MAX_TURNS", "8"))
_SUMMARY_TARGET_CHARS = int(os.getenv("MEMORY_SUMMARY_TARGET_CHARS", "1200"))

"""DESCRIPCION PRINCIPAL DE LA FUNCION: Crea una conexion con SQLite para cada operacion """
def _conn():
    Path(DB_PATH).parent.mkdir(parents=True, exist_ok=True)
    return sqlite3.connect(DB_PATH, isolation_level=None, check_same_thread=False)

"""DESCRIPCION PRINCIPAL DE LA FUNCION: Inicializa el esquema si no existe con las tablas de datos"""
def init_db():
    with _conn() as cx:
        cx.execute("""
            CREATE TABLE IF NOT EXISTS sessions (
                session_id TEXT PRIMARY KEY,
                created_at INTEGER NOT NULL,
                last_seen_at INTEGER NOT NULL
            )""")
        cx.execute("""
            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                role TEXT NOT NULL CHECK(role IN ('user', 'assistant', 'system')),
                text TEXT NOT NULL,
                timestamp INTEGER NOT NULL,
                FOREIGN KEY(session_id) REFERENCES sessions(session_id)
        )""")
        cx.execute("""
            CREATE TABLE IF NOT EXISTS summaries (
                session_id TEXT NOT NULL,
                summary_text TEXT NOT NULL,
                updated_at INTEGER NOT NULL,
                FOREIGN KEY(session_id) REFERENCES sessions(session_id)
        )""")

"""DESCRIPCION PRINCIPAL DE LA FUNCION: Hace una limpieza de las sesiones antiguas por TTL"""
def prune_old_sessions(ttl_days: int = TTL_DAYS) -> int:
    cutoff = int(time.time()) - ttl_days * 86400
    with _conn() as cx:
        cur = cx.execute("SELECT session_id FROM sessions WHERE last_seen_at < ?", (cutoff,))
        old = [r[0] for r in cur.fetchall()]
        for sid in old:
            cx.execute("DELETE FROM summaries WHERE session_id = ?", (sid,))
            cx.execute("DELETE FROM messages WHERE session_id = ?", (sid,))
            cx.execute("DELETE FROM sessions WHERE session_id = ?", (sid,))
        return len(old)

"""DESCRIPCION PRINCIPAL DE LA FUNCION: Crea o modifica una sesion"""
def start_session(session_id: Optional[str] = None) -> str:
    sid = session_id or str(uuid.uuid4())
    now = int(time.time())
    with _conn() as cx:
        cx.execute("INSERT OR IGNORE INTO sessions (session_id, created_at, last_seen_at) VALUES (?, ?, ?)",
                   (sid, now, now))
        cx.execute("UPDATE sessions SET last_seen_at = ? WHERE session_id=?", (now, sid))
    return sid

def clear_session(session_id: str):
    with _conn() as cx:
        cx.execute("DELETE FROM summaries WHERE session_id=?", (session_id,))
        cx.execute("DELETE FROM messages WHERE session_id=?", (session_id,))
        cx.execute("UPDATE sessions SET last_seen_at=? WHERE session_id=?", (int(time.time()), session_id))

"""DESCRIPCION PRINCIPAL DE LA FUNCION: Añade un turno al historial """
def append_message(session_id: str, role: str, text: str):
    now = int(time.time())
    with _conn() as cx:
        cx.execute("INSERT INTO messages (session_id, role, text, timestamp) VALUES (?, ?, ?, ?)", (session_id, role, text, now))
        cx.execute("UPDATE sessions SET last_seen_at = ? WHERE session_id = ?", (now, session_id))

"""DESCRIPCION PRINCIPAL DE LA FUNCION: Lee los ultimos N turnos de una sesion"""
def get_messages(session_id: str, limit: int = _MAX_MSGS_FOR_CONTEXT) -> List[Tuple[str, str,]]:
    with _conn() as cx:
        cur = cx.execute("SELECT role, text FROM messages WHERE session_id = ? ORDER BY id DESC LIMIT ?", (session_id, limit))
        rows = cur.fetchall()[::-1]
    return rows

"""DESCRIPCION PRINCIPAL DE LA FUNCION: Obtiene el resumen guardado si existe para la sesion activa"""
def get_summary(session_id: str) -> str:
    with _conn() as cx:
        cur = cx.execute("SELECT summary_text FROM summaries WHERE session_id = ?", (session_id,))
        row = cur.fetchone()
        return row[0] if row else ""

"""DESCRIPCION PRINCIPAL DE LA FUNCION: Guarda/actualiza un resumen de una sesion"""
def set_summary(session_id: str, summary_text: str):
    now = int(time.time())
    with _conn() as cx:
        cx.execute("INSERT INTO summaries (session_id, summary_text, updated_at) VALUES (?, ?, ?)", (session_id, summary_text, now))

"""DESCRIPCION PRINCIPAL DE LA FUNCION: Construye el contexto donde inyectamos el prompt del LLM"""
def context_text(session_id: str, max_turns: int = _MAX_MSGS_FOR_CONTEXT) -> str:
    summary = get_summary(session_id)
    msgs = get_messages(session_id, limit=max_turns)
    lines = []
    if summary:
        lines.append(f"[Resumen]: {summary}")
    if msgs:
        lines.append("[Ultimos turnos]:")
        for role, text in msgs:
            lines.append(f"- {role}: {text}")
    return "\n".join(lines)

"""DESCRIPCION PRINCIPAL DE LA FUNCION: Comprueba si es hora de hacer un sumario"""
def should_summarize(session_id: str, budget_chars: int = _SUMMARY_TARGET_CHARS) -> bool:
    summary = get_summary(session_id)
    msgs = get_messages(session_id, limit=50)
    total_chars = len(summary) + sum(len(text) for _, text in msgs)
    return total_chars > budget_chars

"""DESCRIPCION PRINCIPAL DE LA FUNCION: Prepara el material a resumir para añadir """
def compact_history_text(session_id: str) -> str:
    summary = get_summary(session_id)
    msgs = get_messages(session_id, limit=50)
    parts = []
    if summary:
        parts.append(f"[Resumen]:\n {summary}")
    if msgs:
        parts.append("[Turnos recientes]:\n" + "n".join(f"{role}: {text}" for role, text in msgs))
    return "\n".join(parts)