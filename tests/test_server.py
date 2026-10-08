import http.client
import json
import socket
import tempfile
import threading
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from clef_ane.bundle import BundleError
from clef_ane.engine.errors import RequestError, TooLong
from clef_ane.server import create_server, serve, termination_signals


class StubService:
    name = "test-clef"
    backbone = SimpleNamespace(max_tokens=1024)

    def systemone(self, request):
        if "error" in request:
            raise {"invalid": RequestError, "long": TooLong, "internal": RuntimeError}[request["error"]]("test error")
        return {"model": self.name, "answers": {"q": {"type": "noul", "noul": 0.75}}, "usage": {"input_tokens": 12, "output_tokens": 0}}


class HTTPTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.log = patch("clef_ane.server.Handler.log_message")
        cls.log.start()
        cls.server = create_server(StubService(), port=0, max_body_mb=1)
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.thread.join(timeout=5)
        cls.server.server_close()
        cls.log.stop()

    def request(self, method="POST", path="/v1/systemone", body="{}", headers=None):
        connection = http.client.HTTPConnection("127.0.0.1", self.server.server_port, timeout=5)
        try:
            connection.request(method, path, body=body, headers=headers or {})
            response = connection.getresponse()
            return response.status, json.loads(response.read())
        finally:
            connection.close()

    def test_health(self):
        status, body = self.request("GET", "/health", None)
        self.assertEqual(status, 200)
        self.assertEqual(body, {"status": "ok", "model": "test-clef", "max_tokens": 1024})

    def test_model_listing(self):
        status, body = self.request("GET", "/v1/models", None)
        self.assertEqual(status, 200)
        self.assertEqual(body["data"][0]["id"], "test-clef")

    def test_request_response(self):
        status, body = self.request(body=json.dumps({"state": "hello", "questions": {"q": {"type": "noul"}}}))
        self.assertEqual(status, 200)
        self.assertEqual(body["answers"]["q"]["noul"], 0.75)

    def test_invalid_json_or_nonobject(self):
        for body in ("broken", "[]", "null", b"\xff"):
            with self.subTest(body=body):
                self.assertEqual(self.request(body=body)[0], 400)

    def test_excessive_content_length_digits(self):
        self.assertEqual(self.request(headers={"Content-Length": "1" * 5000})[0], 400)

    def test_json_recursion_error_returns_400(self):
        with patch("clef_ane.server.json", wraps=json) as decoder:
            decoder.loads.side_effect = RecursionError("too deeply nested")
            self.assertEqual(self.request(body='{"x": []}')[0], 400)

    def test_termination_signals_use_the_cleanup_path(self):
        import signal
        old = signal.getsignal(signal.SIGTERM)
        cleaned_up = False
        with self.assertRaises(KeyboardInterrupt):
            with termination_signals():
                try:
                    signal.raise_signal(signal.SIGTERM)
                finally:
                    cleaned_up = True
        self.assertTrue(cleaned_up)
        self.assertEqual(signal.getsignal(signal.SIGTERM), old)

    def test_negative_content_length(self):
        self.assertEqual(self.request(headers={"Content-Length": "-1"})[0], 400)

    def test_oversized_body_is_rejected_before_reading(self):
        self.assertEqual(self.request(headers={"Content-Length": "1048577"})[0], 413)

    def test_schema_errors_have_expected_statuses(self):
        for error, status in (("invalid", 400), ("long", 413), ("internal", 500)):
            with self.subTest(error=error), patch("sys.stderr"):
                self.assertEqual(self.request(body=json.dumps({"error": error}))[0], status)

    def test_unknown_routes(self):
        self.assertEqual(self.request("GET", "/missing", None)[0], 404)
        self.assertEqual(self.request("POST", "/missing")[0], 404)

    def test_duplicate_content_length_is_rejected(self):
        raw = b"POST /v1/systemone HTTP/1.0\r\nContent-Length: 2\r\nContent-Length: 3\r\n\r\n{}"
        with socket.create_connection(("127.0.0.1", self.server.server_port), timeout=5) as connection:
            connection.sendall(raw)
            self.assertIn(b"400", connection.recv(512).split(b"\r\n")[0])

    def test_shutdown_waits_for_request_threads(self):
        self.assertFalse(self.server.daemon_threads)

    def test_missing_model_fails_without_native_build_or_download(self):
        with tempfile.TemporaryDirectory() as root, patch("clef_ane.server.require_apple_silicon"), \
                patch("clef_ane.server.build_bridge") as build, patch("clef_ane.download.download_model") as download:
            with self.assertRaisesRegex(BundleError, "download"):
                serve(Path(root) / "not-downloaded")
            build.assert_not_called()
            download.assert_not_called()
