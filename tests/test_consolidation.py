"""Tests para _consolidate_by_doc (D8) y _build_qdrant_filter (D4)."""
from __future__ import annotations

import unittest
from types import SimpleNamespace


def _import_chat_api():
    import chat_api  # type: ignore[import-not-found]
    return chat_api


def _hit(score: float, payload: dict):
    return SimpleNamespace(score=score, payload=payload)


class TestConsolidateByDoc(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.chat_api = _import_chat_api()

    def test_below_threshold_excluded(self):
        hits = [
            _hit(0.2, {"doc_id": "d1", "chunk_id": 0}),
            _hit(0.7, {"doc_id": "d2", "chunk_id": 0}),
        ]
        result = self.chat_api._consolidate_by_doc(
            hits_sorted=hits, threshold=0.5, k=5,
            allow_multi=False, multi_ratio=0.9,
        )
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0][0], "d2")

    def test_picks_best_chunk_per_doc(self):
        hits = [
            _hit(0.6, {"doc_id": "d1", "chunk_id": 0}),
            _hit(0.85, {"doc_id": "d1", "chunk_id": 3}),
            _hit(0.5, {"doc_id": "d1", "chunk_id": 5}),
        ]
        result = self.chat_api._consolidate_by_doc(
            hits_sorted=hits, threshold=0.0, k=5,
            allow_multi=False, multi_ratio=0.9,
        )
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0][0], "d1")
        self.assertEqual(result[0][1], 0.85)
        self.assertEqual(result[0][2]["chunk_id"], 3)

    def test_allow_multi_chunk_keeps_close_second(self):
        hits = [
            _hit(0.95, {"doc_id": "d1", "chunk_id": 0}),
            _hit(0.90, {"doc_id": "d1", "chunk_id": 2}),
            _hit(0.30, {"doc_id": "d1", "chunk_id": 5}),
        ]
        result = self.chat_api._consolidate_by_doc(
            hits_sorted=hits, threshold=0.0, k=5,
            allow_multi=True, multi_ratio=0.9,
        )
        self.assertEqual(len(result), 2)
        chunk_ids = {r[2]["chunk_id"] for r in result}
        self.assertEqual(chunk_ids, {0, 2})

    def test_allow_multi_chunk_skips_far_second(self):
        hits = [
            _hit(0.95, {"doc_id": "d1", "chunk_id": 0}),
            _hit(0.50, {"doc_id": "d1", "chunk_id": 2}),
        ]
        result = self.chat_api._consolidate_by_doc(
            hits_sorted=hits, threshold=0.0, k=5,
            allow_multi=True, multi_ratio=0.9,
        )
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0][2]["chunk_id"], 0)

    def test_caps_total_to_k(self):
        hits = [_hit(0.9 - i * 0.05, {"doc_id": f"d{i}", "chunk_id": 0}) for i in range(8)]
        result = self.chat_api._consolidate_by_doc(
            hits_sorted=hits, threshold=0.0, k=3,
            allow_multi=True, multi_ratio=0.9,
        )
        self.assertEqual(len(result), 3)

    def test_ignores_hits_without_doc_id(self):
        hits = [
            _hit(0.9, {"chunk_id": 0}),
            _hit(0.8, {"doc_id": "d1", "chunk_id": 0}),
        ]
        result = self.chat_api._consolidate_by_doc(
            hits_sorted=hits, threshold=0.0, k=5,
            allow_multi=False, multi_ratio=0.9,
        )
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0][0], "d1")


class TestBuildQdrantFilter(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.chat_api = _import_chat_api()

    def test_no_filters_returns_none(self):
        self.assertIsNone(self.chat_api._build_qdrant_filter(None, None))
        self.assertIsNone(self.chat_api._build_qdrant_filter("", ""))

    def test_doc_type_only(self):
        f = self.chat_api._build_qdrant_filter(None, "pdf")
        self.assertIsNotNone(f)

    def test_folder_only(self):
        f = self.chat_api._build_qdrant_filter("carpeta1", None)
        self.assertIsNotNone(f)

    def test_both(self):
        f = self.chat_api._build_qdrant_filter("carpeta2", "docx")
        self.assertIsNotNone(f)


if __name__ == "__main__":
    unittest.main()
