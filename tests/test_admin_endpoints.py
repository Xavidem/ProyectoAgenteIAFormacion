"""Tests para los endpoints /admin/stats y /admin/reindex."""
from __future__ import annotations

import unittest
from unittest.mock import MagicMock

try:
    from fastapi.testclient import TestClient
    HAS_FASTAPI_TESTCLIENT = True
except Exception:
    HAS_FASTAPI_TESTCLIENT = False


class TestAdminEndpoints(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        if not HAS_FASTAPI_TESTCLIENT:
            raise unittest.SkipTest("fastapi.testclient no disponible")
        import chat_api  # type: ignore[import-not-found]
        cls.chat_api = chat_api

    def _build_client(self, *, metadata=None, count=0, reindex_command=""):
        app = self.chat_api.app
        app.state.metadata = metadata or {}
        app.state.qdrant = MagicMock()
        app.state.qdrant.count = MagicMock(
            return_value=MagicMock(count=count)
        )
        app.state.embedder = MagicMock()
        app.state.official_links = {"topics": []}
        app.state.memory_ready = True
        self.chat_api.settings.reindex_command = reindex_command
        return TestClient(app)

    def test_stats_empty(self):
        client = self._build_client(metadata={}, count=0)
        resp = client.get("/admin/stats")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(data["documents"], 0)
        self.assertEqual(data["chunks"], 0)
        self.assertIsNone(data["last_indexed_at"])

    def test_stats_with_data(self):
        metadata = {
            "doc1": {"id": "doc1", "extra": {"indexed_at": "2024-01-01T00:00:00"}},
            "doc2": {"id": "doc2", "extra": {"indexed_at": "2024-06-15T12:00:00"}},
            "doc3": {"id": "doc3"},
        }
        client = self._build_client(metadata=metadata, count=42)
        resp = client.get("/admin/stats")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(data["documents"], 3)
        self.assertEqual(data["chunks"], 42)
        self.assertEqual(data["last_indexed_at"], "2024-06-15T12:00:00")

    def test_reindex_disabled_returns_503(self):
        client = self._build_client(reindex_command="")
        resp = client.post("/admin/reindex")
        self.assertEqual(resp.status_code, 503)
        self.assertIn("REINDEX_COMMAND", resp.json()["detail"])

    def test_reindex_triggers_background(self):
        client = self._build_client(reindex_command="echo dummy")
        resp = client.post("/admin/reindex")
        self.assertEqual(resp.status_code, 200)
        self.assertTrue(resp.json()["triggered"])

    def test_health_endpoint(self):
        client = self._build_client()
        resp = client.get("/health")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertTrue(data["ok"])
        self.assertIn("llava", data)


if __name__ == "__main__":
    unittest.main()
