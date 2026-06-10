"""Tests para el corto-circuito sin evidencia y para los filtros aceptados por /chat."""
from __future__ import annotations

import os
import tempfile
import unittest
from unittest.mock import MagicMock, patch

try:
    from fastapi.testclient import TestClient
    HAS_FASTAPI_TESTCLIENT = True
except Exception:
    HAS_FASTAPI_TESTCLIENT = False


class TestChatEndpointNoEvidence(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        if not HAS_FASTAPI_TESTCLIENT:
            raise unittest.SkipTest("fastapi.testclient no disponible")
        fd, cls.db_path = tempfile.mkstemp(suffix="_chat_test.sqlite")
        os.close(fd)
        os.unlink(cls.db_path)
        os.environ["MEMORY_DB_PATH"] = cls.db_path

        import importlib
        import sys as _sys
        if "mem.store" in _sys.modules:
            importlib.reload(_sys.modules["mem.store"])
        if "chat_api" in _sys.modules:
            del _sys.modules["chat_api"]
        import chat_api  # type: ignore[import-not-found]
        cls.chat_api = chat_api
        import mem.store as memo
        memo.init_db()

    @classmethod
    def tearDownClass(cls) -> None:
        for ext in ("", "-wal", "-shm", "-journal"):
            p = cls.db_path + ext
            try:
                if os.path.exists(p):
                    os.unlink(p)
            except PermissionError:
                pass

    def _build_client(self, hits=None, official_topics=None):
        app = self.chat_api.app
        app.state.metadata = {}
        app.state.qdrant = MagicMock()
        app.state.qdrant.search = MagicMock(return_value=hits or [])
        app.state.qdrant.count = MagicMock(return_value=MagicMock(count=0))

        embedder = MagicMock()
        encoded = MagicMock()
        encoded.tolist = MagicMock(return_value=[0.0] * 384)
        embedder.encode = MagicMock(return_value=encoded)
        app.state.embedder = embedder

        topics = official_topics if official_topics is not None else [
            {"name": "java", "keywords": ["java"], "urls": ["https://www.java.com/"]}
        ]
        for t in topics:
            t["__norm_keywords"] = [self.chat_api.normalize_text(k) for k in t.get("keywords", [])]
        app.state.official_links = {"topics": topics}
        app.state.memory_ready = True
        return TestClient(app)

    def test_no_hits_short_circuits_no_llava_call(self):
        client = self._build_client(hits=[])
        with patch.object(self.chat_api, "call_llava") as mock_llava:
            resp = client.post("/chat", json={"query": "Que es Python?", "k": 3, "threshold": 0.4})
        self.assertEqual(resp.status_code, 200)
        body = resp.json()
        self.assertEqual(body["documents"], [])
        self.assertIn("No tengo informacion suficiente", body["response"])
        mock_llava.assert_not_called()

    def test_no_hits_returns_fallback_urls_only_when_no_docs(self):
        client = self._build_client(hits=[])
        with patch.object(self.chat_api, "call_llava"):
            resp = client.post("/chat", json={"query": "ayuda con java por favor", "k": 3, "threshold": 0.4})
        self.assertEqual(resp.status_code, 200)
        body = resp.json()
        self.assertEqual(body["documents"], [])
        self.assertEqual(body["fallback_urls"], ["https://www.java.com/"])

    def test_with_hits_does_not_send_fallback_urls(self):
        from types import SimpleNamespace
        hits = [
            SimpleNamespace(
                score=0.85,
                payload={"doc_id": "abc", "chunk_id": 0, "title": "Manual", "snippet": "java es lenguaje"},
                vector=[1.0, 0.0, 0.0],
            )
        ]
        client = self._build_client(hits=hits)
        with patch.object(self.chat_api, "call_llava", return_value="Respuesta del LLM"):
            resp = client.post("/chat", json={"query": "java", "k": 3, "threshold": 0.4})
        self.assertEqual(resp.status_code, 200)
        body = resp.json()
        self.assertGreaterEqual(len(body["documents"]), 1)
        self.assertIsNone(body["fallback_urls"])
        self.assertEqual(body["response"], "Respuesta del LLM")

    def test_query_too_long_rejected(self):
        client = self._build_client(hits=[])
        long_q = "a" * 3000
        resp = client.post("/chat", json={"query": long_q, "k": 3, "threshold": 0.0})
        self.assertEqual(resp.status_code, 422)

    def test_invalid_doc_type_rejected(self):
        client = self._build_client(hits=[])
        resp = client.post("/chat", json={"query": "hola", "doc_type": "xlsx"})
        self.assertEqual(resp.status_code, 422)

    def test_valid_doc_type_accepted(self):
        client = self._build_client(hits=[])
        with patch.object(self.chat_api, "call_llava"):
            resp = client.post("/chat", json={"query": "hola", "doc_type": "pdf", "folder": "carpeta1"})
        self.assertEqual(resp.status_code, 200)


if __name__ == "__main__":
    unittest.main()
