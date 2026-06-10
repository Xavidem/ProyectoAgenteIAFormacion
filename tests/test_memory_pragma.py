"""Tests para los PRAGMAs aplicados a SQLite en mem.store."""
from __future__ import annotations

import gc
import os
import sqlite3
import tempfile
import unittest


class TestSqlitePragmas(unittest.TestCase):
    def setUp(self) -> None:
        fd, self.db_path = tempfile.mkstemp(suffix="_pragma.sqlite")
        os.close(fd)
        os.unlink(self.db_path)
        os.environ["MEMORY_DB_PATH"] = self.db_path

        import importlib
        import sys as _sys
        if "mem.store" in _sys.modules:
            importlib.reload(_sys.modules["mem.store"])
        else:
            import mem.store  # noqa: F401
        from mem import store
        self.store = store
        self.store.init_db()

    def tearDown(self) -> None:
        gc.collect()
        for ext in ("", "-wal", "-shm", "-journal"):
            p = self.db_path + ext
            try:
                if os.path.exists(p):
                    os.unlink(p)
            except PermissionError:
                pass

    def test_journal_mode_is_wal(self):
        cx = sqlite3.connect(self.db_path)
        try:
            cur = cx.execute("PRAGMA journal_mode")
            mode = cur.fetchone()[0]
        finally:
            cx.close()
        self.assertEqual(mode.lower(), "wal")

    def test_foreign_keys_pragma_active_in_connection(self):
        with self.store._conn() as cx:
            cur = cx.execute("PRAGMA foreign_keys")
            self.assertEqual(cur.fetchone()[0], 1)

    def test_busy_timeout_set_in_connection(self):
        with self.store._conn() as cx:
            cur = cx.execute("PRAGMA busy_timeout")
            self.assertEqual(cur.fetchone()[0], 5000)

    def test_indexes_created(self):
        with self.store._conn() as cx:
            cur = cx.execute("PRAGMA index_list('messages')")
            names = {row[1] for row in cur.fetchall()}
            self.assertIn("idx_messages_session", names)

            cur2 = cx.execute("PRAGMA index_list('sessions')")
            names2 = {row[1] for row in cur2.fetchall()}
            self.assertIn("idx_sessions_last_seen", names2)

    def test_concurrent_read_during_write_works(self):
        """En modo WAL, otra conexion puede leer datos ya commiteados durante una escritura."""
        sid = self.store.start_session(None)
        self.store.append_message(sid, "user", "hola")
        with self.store._conn() as writer:
            writer.execute("BEGIN")
            writer.execute(
                "INSERT INTO messages (session_id, role, text, timestamp) VALUES (?, ?, ?, ?)",
                (sid, "assistant", "in-flight", 0),
            )
            msgs = self.store.get_messages(sid, limit=10)
            self.assertTrue(any(t == "hola" for _, t in msgs))
            writer.execute("ROLLBACK")


if __name__ == "__main__":
    unittest.main()
