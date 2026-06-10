"""Tests para mmr_rerank y _cosine_sim de chat_api."""
from __future__ import annotations

import unittest


def _import_chat_api():
    import chat_api  # type: ignore[import-not-found]
    return chat_api


class TestCosineSim(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.chat_api = _import_chat_api()

    def test_identical_vectors_returns_one(self):
        v = [1.0, 0.0, 0.0]
        self.assertAlmostEqual(self.chat_api._cosine_sim(v, v), 1.0, places=6)

    def test_orthogonal_vectors_returns_zero(self):
        a = [1.0, 0.0]
        b = [0.0, 1.0]
        self.assertAlmostEqual(self.chat_api._cosine_sim(a, b), 0.0, places=6)

    def test_zero_vector_does_not_crash(self):
        a = [0.0, 0.0]
        b = [1.0, 1.0]
        self.assertEqual(self.chat_api._cosine_sim(a, b), 0.0)


class TestMmrRerank(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.chat_api = _import_chat_api()

    def test_empty_returns_empty(self):
        self.assertEqual(self.chat_api.mmr_rerank([1.0], [], 5, 0.5), [])

    def test_lambda_one_returns_relevance_top(self):
        cands = [
            (0.9, [1.0, 0.0], "a"),
            (0.5, [0.0, 1.0], "b"),
            (0.7, [0.5, 0.5], "c"),
        ]
        out = self.chat_api.mmr_rerank([1.0, 0.0], cands, top_n=3, lambda_mult=1.0)
        # top_n >= len(cands) y lambda=1.0: cortocircuito que devuelve por relevancia.
        self.assertEqual(out[0], "a")
        self.assertIn(out[1], {"b", "c"})
        self.assertEqual(set(out), {"a", "b", "c"})

    def test_lambda_low_diversifies(self):
        # Candidatos a y b casi iguales, c diverso. Con lambda bajo, c sube.
        cands = [
            (0.95, [1.0, 0.0, 0.0], "a"),
            (0.94, [1.0, 0.001, 0.0], "b_similar_to_a"),
            (0.50, [0.0, 1.0, 0.0], "c_different"),
        ]
        out = self.chat_api.mmr_rerank([1.0, 0.0, 0.0], cands, top_n=2, lambda_mult=0.1)
        self.assertEqual(out[0], "a")
        self.assertEqual(out[1], "c_different")

    def test_top_n_larger_than_candidates_returns_all(self):
        cands = [
            (0.9, [1.0, 0.0], "a"),
            (0.5, [0.0, 1.0], "b"),
        ]
        out = self.chat_api.mmr_rerank([1.0, 0.0], cands, top_n=10, lambda_mult=0.5)
        self.assertEqual(set(out), {"a", "b"})


if __name__ == "__main__":
    unittest.main()
