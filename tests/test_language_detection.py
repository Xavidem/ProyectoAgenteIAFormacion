"""Tests para detect_query_language y normalize_text de chat_api."""
from __future__ import annotations

import unittest
from unittest.mock import patch


def _import_chat_api():
    """Importa chat_api parcheando la lectura del JSON de metadatos."""
    import json as _json
    with patch.object(_json, "load", return_value=[]):
        import chat_api  # type: ignore[import-not-found]
        return chat_api


class TestQueryLanguage(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.chat_api = _import_chat_api()

    def test_detect_spanish(self):
        self.assertEqual(self.chat_api.detect_query_language("¿Como puedo hacer un bucle?"), "es")
        self.assertEqual(self.chat_api.detect_query_language("Necesito un ejemplo de Java"), "es")

    def test_detect_english(self):
        self.assertEqual(self.chat_api.detect_query_language("What is a Python list?"), "en")
        self.assertEqual(self.chat_api.detect_query_language("Show me an example of SQL"), "en")

    def test_empty_query_returns_empty(self):
        self.assertEqual(self.chat_api.detect_query_language(""), "")

    def test_no_clear_signal_returns_empty(self):
        self.assertEqual(self.chat_api.detect_query_language("Madrid"), "")


class TestNormalizeText(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.chat_api = _import_chat_api()

    def test_lowercase_and_strips_accents(self):
        self.assertEqual(self.chat_api.normalize_text("Á É Í Ó Ú Ñ"), "a e i o u n")

    def test_keeps_basic_words(self):
        norm = self.chat_api.normalize_text("¡Hola, mundo!")
        self.assertIn("hola", norm)
        self.assertIn("mundo", norm)


if __name__ == "__main__":
    unittest.main()
