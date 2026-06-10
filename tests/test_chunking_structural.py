"""Tests para chunk_segments (chunking estructural)."""
from __future__ import annotations

import unittest

from catalog.text_extractor import chunk_segments, chunk_tokens


class TestChunkSegments(unittest.TestCase):
    def test_empty_input(self):
        self.assertEqual(chunk_segments([], 100, 20), [])

    def test_short_segments_pack_into_one_chunk(self):
        segments = ["uno dos tres", "cuatro cinco", "seis siete ocho"]
        chunks = chunk_segments(segments, size=20, stride=2)
        self.assertEqual(len(chunks), 1)
        self.assertEqual(
            chunks[0],
            ["uno", "dos", "tres", "cuatro", "cinco", "seis", "siete", "ocho"],
        )

    def test_multiple_chunks_when_size_is_exceeded(self):
        segments = [" ".join([f"w{i}" for i in range(10)])] * 6
        chunks = chunk_segments(segments, size=20, stride=4)
        self.assertGreaterEqual(len(chunks), 3)
        for chunk in chunks:
            self.assertLessEqual(len(chunk), 20)

    def test_segment_larger_than_size_is_split(self):
        big = " ".join([f"t{i}" for i in range(50)])
        chunks = chunk_segments([big], size=10, stride=2)
        self.assertGreater(len(chunks), 1)
        for chunk in chunks:
            self.assertLessEqual(len(chunk), 10)

    def test_overlap_between_chunks(self):
        segments = [
            "alpha beta gamma delta epsilon zeta eta theta iota kappa",
            "lambda mu nu xi omicron pi rho sigma tau upsilon",
            "phi chi psi omega",
        ]
        chunks = chunk_segments(segments, size=10, stride=3)
        self.assertGreaterEqual(len(chunks), 2)
        first_tail = set(chunks[0][-3:])
        second_head = set(chunks[1][:3])
        self.assertTrue(first_tail & second_head)

    def test_does_not_break_within_segment_when_possible(self):
        segments = ["primer parrafo corto", "segundo parrafo distinto"]
        chunks = chunk_segments(segments, size=100, stride=10)
        self.assertEqual(len(chunks), 1)

    def test_size_zero_returns_full_text(self):
        segments = ["uno dos", "tres cuatro"]
        out = chunk_segments(segments, size=0, stride=0)
        self.assertEqual(out, [["uno", "dos", "tres", "cuatro"]])


class TestChunkTokensFallbackStillWorks(unittest.TestCase):
    """Smoke check: el fallback sigue funcionando si no hay segmentos estructurales."""

    def test_no_segments_falls_back_to_word_chunks(self):
        text = "uno dos tres cuatro cinco seis siete ocho"
        tokens = text.split()
        chunks = chunk_tokens(tokens, size=3, stride=1)
        self.assertGreater(len(chunks), 1)


if __name__ == "__main__":
    unittest.main()
