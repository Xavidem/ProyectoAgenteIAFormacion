"""Tests para call_backend de ui_app: extraccion de errores reales del backend."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import requests

ROOT = Path(__file__).resolve().parents[1]
UI_DIR = ROOT / "src" / "ui"
if str(UI_DIR) not in sys.path:
    sys.path.insert(0, str(UI_DIR))


def _import_ui_app():
    # Stub gradio antes del import para no cargar la libreria real ni lanzar el server.
    import types
    if "gradio" not in sys.modules:
        gr = types.ModuleType("gradio")
        gr.Blocks = MagicMock()
        gr.update = MagicMock()
        gr.Slider = MagicMock()
        gr.Textbox = MagicMock()
        gr.Markdown = MagicMock()
        gr.DataFrame = MagicMock()
        gr.Chatbot = MagicMock()
        gr.Button = MagicMock()
        gr.Row = MagicMock()
        gr.Accordion = MagicMock()
        sys.modules["gradio"] = gr
    # Forzamos demo.launch a no hacer nada
    import gradio as _gr
    _gr.Blocks = MagicMock(return_value=MagicMock(launch=MagicMock()))
    if "ui_app" in sys.modules:
        del sys.modules["ui_app"]
    import ui_app  # type: ignore[import-not-found]
    return ui_app


class TestCallBackend(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.ui_app = _import_ui_app()

    def test_connection_error_raises_backend_error(self):
        with patch.object(requests, "post", side_effect=requests.exceptions.ConnectionError("boom")):
            with self.assertRaises(self.ui_app.BackendError) as cm:
                self.ui_app.call_backend("hola", 5, 0.4, "sid-1")
            self.assertIn("conectar", str(cm.exception).lower())

    def test_timeout_raises_backend_error(self):
        with patch.object(requests, "post", side_effect=requests.exceptions.Timeout("slow")):
            with self.assertRaises(self.ui_app.BackendError) as cm:
                self.ui_app.call_backend("hola", 5, 0.4, "sid-1")
            self.assertIn("timeout", str(cm.exception).lower())

    def test_4xx_with_detail_extracts_message(self):
        mock_resp = MagicMock()
        mock_resp.ok = False
        mock_resp.status_code = 422
        mock_resp.json = MagicMock(return_value={"detail": "query es requerido"})
        with patch.object(requests, "post", return_value=mock_resp):
            with self.assertRaises(self.ui_app.BackendError) as cm:
                self.ui_app.call_backend("", 5, 0.4, "sid-1")
            msg = str(cm.exception)
            self.assertIn("422", msg)
            self.assertIn("query es requerido", msg)

    def test_5xx_with_text_fallback(self):
        mock_resp = MagicMock()
        mock_resp.ok = False
        mock_resp.status_code = 500
        mock_resp.json = MagicMock(side_effect=ValueError())
        mock_resp.text = "Internal Server Error"
        with patch.object(requests, "post", return_value=mock_resp):
            with self.assertRaises(self.ui_app.BackendError) as cm:
                self.ui_app.call_backend("hola", 5, 0.4, "sid-1")
            msg = str(cm.exception)
            self.assertIn("500", msg)
            self.assertIn("Internal Server Error", msg)

    def test_success_returns_json(self):
        mock_resp = MagicMock()
        mock_resp.ok = True
        mock_resp.status_code = 200
        expected = {"response": "hola", "documents": [], "session_id": "sid-1"}
        mock_resp.json = MagicMock(return_value=expected)
        with patch.object(requests, "post", return_value=mock_resp):
            result = self.ui_app.call_backend("hola", 5, 0.4, "sid-1")
        self.assertEqual(result, expected)

    def test_invalid_json_raises_backend_error(self):
        mock_resp = MagicMock()
        mock_resp.ok = True
        mock_resp.status_code = 200
        mock_resp.json = MagicMock(side_effect=ValueError("not json"))
        with patch.object(requests, "post", return_value=mock_resp):
            with self.assertRaises(self.ui_app.BackendError) as cm:
                self.ui_app.call_backend("hola", 5, 0.4, "sid-1")
            self.assertIn("no válida", str(cm.exception).lower() + str(cm.exception))


if __name__ == "__main__":
    unittest.main()
