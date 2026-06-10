"""Tests para helpers internos de chat_api."""
from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path


def _import_chat_api():
    import chat_api  # type: ignore[import-not-found]
    return chat_api


class TestTruncatePrompt(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.chat_api = _import_chat_api()

    def test_short_prompt_unchanged(self):
        text = "hola mundo"
        self.assertEqual(self.chat_api.truncate_prompt(text, budget=100), text)

    def test_long_prompt_truncated(self):
        text = "x" * 5000
        out = self.chat_api.truncate_prompt(text, budget=1000)
        self.assertTrue(out.startswith("x" * 1000))
        self.assertIn("[...truncado", out)

    def test_default_budget_uses_settings(self):
        text = "y" * (self.chat_api.settings.prompt_char_budget + 100)
        out = self.chat_api.truncate_prompt(text)
        self.assertLess(len(out), len(text))
        self.assertIn("[...truncado", out)


class TestSettingsCorsOrigins(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.chat_api = _import_chat_api()

    def test_wildcard_returns_star(self):
        s = self.chat_api.Settings(CORS_ORIGINS="*")
        self.assertEqual(s.cors_origins_list, ["*"])

    def test_comma_separated_list(self):
        s = self.chat_api.Settings(
            CORS_ORIGINS="http://localhost:7860, http://localhost:3000 ,"
        )
        self.assertEqual(
            s.cors_origins_list,
            ["http://localhost:7860", "http://localhost:3000"],
        )

    def test_empty_string_returns_empty_list(self):
        s = self.chat_api.Settings(CORS_ORIGINS="")
        self.assertEqual(s.cors_origins_list, [])


class TestLoadMetadataDict(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.chat_api = _import_chat_api()

    def test_missing_file_returns_empty_dict(self):
        result = self.chat_api._load_metadata_dict("/path/that/does/not/exist.json")
        self.assertEqual(result, {})

    def test_invalid_json_returns_empty_dict(self):
        with tempfile.NamedTemporaryFile("w", delete=False, suffix=".json") as f:
            f.write("not a valid json {{{")
            tmp = f.name
        try:
            result = self.chat_api._load_metadata_dict(tmp)
            self.assertEqual(result, {})
        finally:
            os.unlink(tmp)

    def test_valid_json_indexed_by_id(self):
        records = [
            {"id": "abc", "title": "Doc A"},
            {"id": "def", "title": "Doc B"},
            {"no_id": True},
        ]
        with tempfile.NamedTemporaryFile("w", delete=False, suffix=".json", encoding="utf-8") as f:
            json.dump(records, f)
            tmp = f.name
        try:
            result = self.chat_api._load_metadata_dict(tmp)
            self.assertEqual(set(result.keys()), {"abc", "def"})
            self.assertEqual(result["abc"]["title"], "Doc A")
        finally:
            os.unlink(tmp)


class TestResolveDocDiskPath(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.chat_api = _import_chat_api()
        cls.tmp = tempfile.TemporaryDirectory()
        cls.docs_root = Path(cls.tmp.name).resolve()
        (cls.docs_root / "ok.pdf").write_bytes(b"x")
        (cls.docs_root / "subdir").mkdir()
        (cls.docs_root / "subdir" / "nested.pdf").write_bytes(b"y")
        cls._original_docs_root = cls.chat_api.settings.docs_root
        cls.chat_api.settings.docs_root = str(cls.docs_root)

    @classmethod
    def tearDownClass(cls) -> None:
        cls.chat_api.settings.docs_root = cls._original_docs_root
        cls.tmp.cleanup()

    def test_empty_path_raises_404(self):
        with self.assertRaises(self.chat_api.HTTPException) as cm:
            self.chat_api._resolve_doc_disk_path("")
        self.assertEqual(cm.exception.status_code, 404)

    def test_relative_path_inside_root(self):
        p = self.chat_api._resolve_doc_disk_path("ok.pdf")
        self.assertEqual(p.name, "ok.pdf")

    def test_nested_path_inside_root(self):
        p = self.chat_api._resolve_doc_disk_path("subdir/nested.pdf")
        self.assertEqual(p.name, "nested.pdf")

    def test_path_traversal_blocked(self):
        with self.assertRaises(self.chat_api.HTTPException) as cm:
            self.chat_api._resolve_doc_disk_path("../etc/passwd")
        self.assertEqual(cm.exception.status_code, 403)

    def test_absolute_path_outside_blocked(self):
        outside = Path(tempfile.gettempdir()).resolve() / "totally_outside.pdf"
        outside.write_bytes(b"z")
        try:
            with self.assertRaises(self.chat_api.HTTPException) as cm:
                self.chat_api._resolve_doc_disk_path(str(outside))
            self.assertEqual(cm.exception.status_code, 403)
        finally:
            outside.unlink(missing_ok=True)

    def test_nonexistent_path_returns_404(self):
        with self.assertRaises(self.chat_api.HTTPException) as cm:
            self.chat_api._resolve_doc_disk_path("does_not_exist.pdf")
        self.assertEqual(cm.exception.status_code, 404)


if __name__ == "__main__":
    unittest.main()
