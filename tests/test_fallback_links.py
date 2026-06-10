"""Tests para el matching de URLs curadas (fallback)."""
from __future__ import annotations

import unittest
from unittest.mock import patch


def _import_chat_api():
    import json as _json
    with patch.object(_json, "load", return_value=[]):
        import chat_api  # type: ignore[import-not-found]
        return chat_api


class TestOfficialLinks(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.chat_api = _import_chat_api()

    def _build_catalog(self):
        catalog = {
            "topics": [
                {"name": "java", "keywords": ["java", "jdk", "jvm"],
                 "urls": ["https://www.java.com/es/"]},
                {"name": "python", "keywords": ["python"],
                 "urls": ["https://www.learnpython.org/es/"]},
            ]
        }
        for t in catalog["topics"]:
            t["__norm_keywords"] = [self.chat_api.normalize_text(k) for k in t["keywords"]]
        return catalog

    def test_match_returns_urls(self):
        catalog = self._build_catalog()
        urls = self.chat_api.official_links_for_query("Quiero aprender Java", catalog)
        self.assertEqual(urls, ["https://www.java.com/es/"])

    def test_no_match_returns_none(self):
        catalog = self._build_catalog()
        self.assertIsNone(
            self.chat_api.official_links_for_query("dame una receta de cocina", catalog)
        )

    def test_match_is_case_insensitive_and_accent_insensitive(self):
        catalog = self._build_catalog()
        urls = self.chat_api.official_links_for_query("PYTHON con acentos áéí", catalog)
        self.assertEqual(urls, ["https://www.learnpython.org/es/"])


if __name__ == "__main__":
    unittest.main()
