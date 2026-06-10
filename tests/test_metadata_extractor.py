"""Tests para metadata_extractor: doc_id determinista, deduplicacion via JSON previo."""
from __future__ import annotations

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from catalog import metadata_extractor as mex


class TestDocIdFromHash(unittest.TestCase):
    def test_deterministic_id_for_same_hash(self):
        sha = "a" * 64
        self.assertEqual(mex._doc_id_from_hash(sha), mex._doc_id_from_hash(sha))

    def test_different_hash_yields_different_id(self):
        self.assertNotEqual(
            mex._doc_id_from_hash("a" * 64),
            mex._doc_id_from_hash("b" * 64),
        )


class TestSha256OfFile(unittest.TestCase):
    def test_known_content(self):
        with TemporaryDirectory() as td:
            p = Path(td) / "hello.txt"
            p.write_bytes(b"hello world")
            # hash conocido SHA-256 de "hello world"
            expected = (
                "b94d27b9934d3e08a52e52d7da7dabfac484efe37a5380ee9088f7ace2efcde9"
            )
            self.assertEqual(mex._sha256_of_file(p), expected)


class TestLoadPrevious(unittest.TestCase):
    def test_missing_file_returns_empty_dict(self):
        self.assertEqual(mex._load_previous(Path("/no/existe/123.json")), {})

    def test_invalid_json_returns_empty_dict(self):
        with TemporaryDirectory() as td:
            p = Path(td) / "broken.json"
            p.write_text("{not json")
            self.assertEqual(mex._load_previous(p), {})

    def test_loads_records_by_id(self):
        with TemporaryDirectory() as td:
            p = Path(td) / "ok.json"
            data = [
                {"id": "abc", "title": "A"},
                {"id": "def", "title": "B"},
            ]
            p.write_text(json.dumps(data), encoding="utf-8")
            loaded = mex._load_previous(p)
            self.assertEqual(set(loaded.keys()), {"abc", "def"})
            self.assertEqual(loaded["abc"]["title"], "A")


class TestCollectMetadataLocal(unittest.TestCase):
    def test_collects_only_supported_extensions(self):
        with TemporaryDirectory() as td:
            root = Path(td)
            (root / "a.pdf").write_bytes(b"pdf-bytes")
            (root / "b.docx").write_bytes(b"docx-bytes")
            (root / "ignorame.txt").write_text("plain")
            (root / "imagen.png").write_bytes(b"PNG")

            with patch.object(mex, "DATA_ROOT", root):
                recs = mex.collect_metadata_local()

            paths = sorted(r["path"] for r in recs)
            self.assertEqual(paths, ["a.pdf", "b.docx"])
            for r in recs:
                self.assertIn("sha256", r["extra"])
                self.assertIn("size_bytes", r["extra"])
                self.assertEqual(r["extra"]["chunks_count"], 0)
                self.assertEqual(r["extra"]["indexed_at"], "")

    def test_preserves_enrichment_when_doc_id_unchanged(self):
        with TemporaryDirectory() as td:
            root = Path(td)
            f = root / "doc.pdf"
            f.write_bytes(b"contenido fijo")

            with patch.object(mex, "DATA_ROOT", root):
                first = mex.collect_metadata_local()

            previous = {first[0]["id"]: {**first[0], "author": "ya rellenado",
                                          "language": "es",
                                          "extra": {**first[0]["extra"],
                                                    "indexed_at": "2025-01-01T00:00:00",
                                                    "chunks_count": 7,
                                                    "indexed_sha256": first[0]["extra"]["sha256"]}}}

            with patch.object(mex, "DATA_ROOT", root):
                second = mex.collect_metadata_local(previous=previous)

            self.assertEqual(second[0]["id"], first[0]["id"])
            self.assertEqual(second[0]["author"], "ya rellenado")
            self.assertEqual(second[0]["language"], "es")
            self.assertEqual(second[0]["extra"]["chunks_count"], 7)
            self.assertEqual(second[0]["extra"]["indexed_at"], "2025-01-01T00:00:00")

    def test_changed_content_resets_indexed_state(self):
        with TemporaryDirectory() as td:
            root = Path(td)
            f = root / "doc.pdf"
            f.write_bytes(b"v1")

            with patch.object(mex, "DATA_ROOT", root):
                first = mex.collect_metadata_local()
            old_id = first[0]["id"]

            previous = {old_id: {**first[0],
                                  "extra": {**first[0]["extra"],
                                            "indexed_sha256": first[0]["extra"]["sha256"],
                                            "chunks_count": 5,
                                            "indexed_at": "2025-01-01T00:00:00"}}}

            f.write_bytes(b"v2 contenido distinto")

            with patch.object(mex, "DATA_ROOT", root):
                second = mex.collect_metadata_local(previous=previous)

            # Nuevo id porque el sha256 cambio
            self.assertNotEqual(second[0]["id"], old_id)
            self.assertEqual(second[0]["extra"]["chunks_count"], 0)
            self.assertEqual(second[0]["extra"]["indexed_at"], "")


if __name__ == "__main__":
    unittest.main()
