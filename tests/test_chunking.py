"""Tests para chunk_tokens y clean_text en src/catalog/text_extractor.py."""
from __future__ import annotations

import unittest

from catalog.text_extractor import chunk_tokens, clean_text


class TestChunkTokens(unittest.TestCase):
    def test_empty_input_returns_empty_list(self):
        self.assertEqual(chunk_tokens([], 280, 200), [])

    def test_size_le_zero_returns_full_in_one_chunk(self):
        toks = ["a", "b", "c"]
        self.assertEqual(chunk_tokens(toks, 0, 200), [toks])
        self.assertEqual(chunk_tokens(toks, -5, 200), [toks])

    def test_size_greater_than_input_returns_single_chunk(self):
        toks = ["a", "b", "c"]
        self.assertEqual(chunk_tokens(toks, 280, 200), [toks])

    def test_no_overlap_when_stride_zero(self):
        toks = ["a", "b", "c", "d", "e"]
        chunks = chunk_tokens(toks, size=2, stride=0)
        self.assertEqual(chunks, [["a", "b"], ["c", "d"], ["e"]])

    def test_overlap_with_stride(self):
        toks = ["a", "b", "c", "d", "e"]
        # size=3, stride=1 => step=2
        chunks = chunk_tokens(toks, size=3, stride=1)
        self.assertEqual(chunks[0], ["a", "b", "c"])
        self.assertEqual(chunks[1], ["c", "d", "e"])

    def test_stride_ge_size_falls_back_to_no_overlap(self):
        toks = ["a", "b", "c", "d"]
        chunks = chunk_tokens(toks, size=2, stride=2)
        # step = size cuando stride >= size, por tanto no hay solapamiento
        self.assertEqual(chunks, [["a", "b"], ["c", "d"]])

    def test_does_not_loop_forever(self):
        toks = list("abcdefghij")
        chunks = chunk_tokens(toks, size=4, stride=3)
        flat = [t for c in chunks for t in c]
        self.assertGreaterEqual(len(flat), len(toks))
        self.assertLess(len(chunks), 50)

    def test_default_project_settings(self):
        # Valores reales del proyecto: size=280, stride=200 -> step=80
        toks = list(range(1000))
        chunks = chunk_tokens([str(t) for t in toks], size=280, stride=200)
        self.assertGreater(len(chunks), 1)
        for c in chunks:
            self.assertLessEqual(len(c), 280)


class TestCleanText(unittest.TestCase):
    def test_strips_nulls(self):
        self.assertEqual(clean_text("hola\x00mundo"), "hola mundo")

    def test_collapses_spaces_and_tabs(self):
        self.assertEqual(clean_text("hola      mundo"), "hola mundo")
        self.assertEqual(clean_text("hola\t\t\tmundo"), "hola mundo")

    def test_collapses_excess_newlines(self):
        out = clean_text("hola\n\n\n\nmundo")
        self.assertEqual(out, "hola\n\nmundo")

    def test_empty_input(self):
        self.assertEqual(clean_text(""), "")
        self.assertEqual(clean_text("   "), "")


if __name__ == "__main__":
    unittest.main()
