"""Tests para src/mem/store.py: TTL, UPSERT del resumen y formato del historial."""
from __future__ import annotations

import gc
import importlib
import os
import sqlite3
import time
import unittest
import uuid
from pathlib import Path


class _MemoryTestBase(unittest.TestCase):
    """Carga `mem.store` apuntando a una DB SQLite temporal por test.

    Maneja la limpieza de forma compatible con Windows: cierra conexiones,
    fuerza GC y borra el fichero al final.
    """

    def setUp(self) -> None:
        self._db_path = Path(os.getenv("TEMP", os.getcwd())) / f"mem_{uuid.uuid4().hex}.sqlite"

        os.environ["MEMORY_DB_PATH"] = str(self._db_path)
        os.environ["MEMORY_TTL_DAYS"] = "5"
        os.environ["MEMORY_MAX_TURNS"] = "8"
        os.environ["MEMORY_SUMMARY_TARGET_CHARS"] = "1200"

        import mem.store as store
        importlib.reload(store)
        store.init_db()
        self.store = store

    def tearDown(self) -> None:
        gc.collect()
        try:
            self._db_path.unlink(missing_ok=True)
        except PermissionError:
            # En Windows con ANTIVIRUS algunos handles tardan.
            time.sleep(0.1)
            gc.collect()
            try:
                self._db_path.unlink(missing_ok=True)
            except Exception:
                pass


class TestSession(_MemoryTestBase):
    def test_start_session_returns_uuid_when_none_provided(self):
        sid = self.store.start_session(None)
        self.assertTrue(sid)

    def test_start_session_idempotent_with_explicit_id(self):
        sid = self.store.start_session("abc123")
        again = self.store.start_session("abc123")
        self.assertEqual(sid, again)

    def test_clear_session_removes_messages_and_summary(self):
        sid = self.store.start_session(None)
        self.store.append_message(sid, "user", "hola")
        self.store.set_summary(sid, "resumen actual")
        self.store.clear_session(sid)
        self.assertEqual(self.store.get_messages(sid), [])
        self.assertEqual(self.store.get_summary(sid), "")


class TestSummaryUpsert(_MemoryTestBase):
    """A3: el resumen no debe acumularse en filas nuevas."""

    def test_set_summary_upsert_keeps_only_one_row(self):
        sid = self.store.start_session(None)
        self.store.set_summary(sid, "v1")
        self.store.set_summary(sid, "v2")
        self.store.set_summary(sid, "v3")

        self.assertEqual(self.store.get_summary(sid), "v3")

        cx = sqlite3.connect(str(self._db_path))
        try:
            cur = cx.execute("SELECT COUNT(*) FROM summaries WHERE session_id=?", (sid,))
            self.assertEqual(cur.fetchone()[0], 1)
        finally:
            cx.close()


class TestCompactHistoryNewlines(_MemoryTestBase):
    """A2: el separador entre turnos debe ser '\\n', no la letra 'n'."""

    def test_compact_history_uses_real_newlines(self):
        sid = self.store.start_session(None)
        self.store.append_message(sid, "user", "primera")
        self.store.append_message(sid, "assistant", "respuesta")

        text = self.store.compact_history_text(sid)

        self.assertIn("user: primera", text)
        self.assertIn("assistant: respuesta", text)
        self.assertNotIn("primeran assistant", text)


class TestPruneOldSessions(_MemoryTestBase):
    def test_prune_removes_only_expired_sessions(self):
        sid_old = self.store.start_session("old")
        sid_new = self.store.start_session("new")

        ten_days_ago = int(time.time()) - 10 * 86400

        cx = sqlite3.connect(str(self._db_path))
        try:
            cx.execute(
                "UPDATE sessions SET last_seen_at=? WHERE session_id=?",
                (ten_days_ago, sid_old),
            )
            cx.commit()
        finally:
            cx.close()

        deleted = self.store.prune_old_sessions(ttl_days=5)
        self.assertEqual(deleted, 1)

        cx = sqlite3.connect(str(self._db_path))
        try:
            cur = cx.execute("SELECT session_id FROM sessions ORDER BY session_id")
            ids = [r[0] for r in cur.fetchall()]
        finally:
            cx.close()
        self.assertEqual(ids, [sid_new])


if __name__ == "__main__":
    unittest.main()
