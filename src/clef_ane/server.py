"""Local HTTP adapter. The heavy model is imported only when starting serve."""
from __future__ import annotations

import json
import os
import re
import signal
import socket
import sys
import threading
import time
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from filelock import FileLock

from .bundle import IMAGE_FORMAT_VERSION, load_bundle
from .engine.errors import RequestError, TooLong
from .native import build_bridge, require_apple_silicon


class Handler(BaseHTTPRequestHandler):
    server_version = "clef-ane"

    def setup(self):
        super().setup()
        self.connection.settimeout(30)

    def _send(self, status: int, body: dict) -> None:
        data = json.dumps(body, ensure_ascii=False, allow_nan=False).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        try:
            self.wfile.write(data)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def _error(self, status: int, message: str) -> None:
        self._send(status, {"error": {"message": message}})

    def do_GET(self):
        service = self.server.service
        if self.path == "/health":
            vision = getattr(service, "vision", None)
            modalities = ["text"] + (["image"] if vision is not None else [])
            if getattr(vision, "supports_video", False) is True:
                modalities.append("video")
            self._send(200, {"status": "ok", "model": service.name, "max_tokens": service.backbone.max_tokens,
                             "input_modalities": modalities})
        elif self.path == "/v1/models":
            self._send(200, {"object": "list", "data": [{"id": service.name, "object": "model"}]})
        else:
            self._error(404, "not found")

    def do_POST(self):
        if not self.server.request_slots.acquire(blocking=False):
            return self._error(503, "server is busy; wait for an active request to finish and try again")
        try:
            return self._post()
        finally:
            self.server.request_slots.release()

    def _post(self):
        if self.path != "/v1/systemone":
            return self._error(404, "not found")
        values = self.headers.get_all("Content-Length", [])
        if self.headers.get("Transfer-Encoding") or len(values) != 1 or not re.fullmatch(r"[0-9]{1,12}", values[0]):
            return self._error(400, "send a JSON request with one valid Content-Length header")
        length = int(values[0])
        if length > self.server.max_body_bytes:
            return self._error(413, "request body exceeds the configured size limit")
        try:
            raw = self.rfile.read(length)
            if len(raw) != length:
                return self._error(400, "request body is incomplete")
            request = json.loads(raw)
            if not isinstance(request, dict):
                return self._error(400, "request body must be a JSON object")
        except (ValueError, UnicodeDecodeError, RecursionError):
            return self._error(400, "request body must contain valid JSON")
        except socket.timeout:
            return self._error(408, "request body timed out")
        started = time.monotonic()
        try:
            response = self.server.service.systemone(request)
            self._send(200, response)
        except TooLong as error:
            self._error(413, str(error))
        except RequestError as error:
            self._error(400, str(error))
        except Exception as error:
            print(f"Request failed: {error!r}", file=sys.stderr, flush=True)
            self._error(500, "model request failed; check the server log")
        else:
            self.log_message("processed %d tokens in %.2fs", response["usage"]["input_tokens"], time.monotonic() - started)


def create_server(service, host: str = "127.0.0.1", port: int = 8080, max_body_mb: int = 32) -> ThreadingHTTPServer:
    server = ThreadingHTTPServer((host, port), Handler)
    # Drain active requests before freeing the native kernels on shutdown.
    server.daemon_threads = False
    server.service = service
    server.max_body_bytes = max_body_mb * 1024 * 1024
    server.request_slots = threading.BoundedSemaphore(2)  # one active inference plus one buffered request
    return server


def load_service(root: Path, manifest: dict, max_tokens: int, name: str, vision_device: str = "ane"):
    from .engine.ref_backbone import Head
    from .engine.service import Service
    from .model import BundleBackbone

    backbone = BundleBackbone(root, manifest, max_tokens)
    try:
        head = Head(model_dir=root)
        vision = None
        if manifest["format_version"] == IMAGE_FORMAT_VERSION:
            from .vision import Vision
            vision = Vision(root, head.tokenizer, device=vision_device)
        service = Service(backbone, head, name, vision=vision)
        service.systemone({"state": "warmup", "questions": {"w": {"type": "noul", "instructions": "Is this a warmup?"}}})
        return service
    except BaseException:
        backbone.close()
        raise


@contextmanager
def termination_signals():
    """Give launchd/kill and terminal shutdown the same cleanup path as Ctrl-C."""
    if threading.current_thread() is not threading.main_thread():
        yield
        return
    def stop(signum, frame):
        raise KeyboardInterrupt
    previous = {sig: signal.signal(sig, stop) for sig in (signal.SIGTERM, signal.SIGHUP)}
    try:
        yield
    finally:
        for sig, handler in previous.items():
            signal.signal(sig, handler)


def serve(model_dir: Path, *, host: str = "127.0.0.1", port: int = 8080, max_tokens: int = 16384,
          name: str = "clef-flash", max_body_mb: int = 32, vision_device: str = "ane") -> None:
    require_apple_silicon()
    root = model_dir.expanduser().resolve()
    if not root.is_dir():
        load_bundle(root)  # report the actionable missing-model error, without creating a directory
    with termination_signals(), FileLock(str(root / ".clef-ane.lock"), timeout=0):
        print(f"Checking model files in {root}...", flush=True)
        manifest = load_bundle(root)
        # Bind before loading many gigabytes onto the ANE, so an occupied port fails fast.
        with create_server(None, host, port, max_body_mb) as server:
            if host not in ("127.0.0.1", "localhost"):
                print("Warning: this server has no authentication. Only expose it on a trusted network.", file=sys.stderr)
            os.environ["CLEF_ANE_BRIDGE"] = str(build_bridge())
            started = time.monotonic()
            service = load_service(root, manifest, max_tokens, name, vision_device)
            server.service = service
            print(f"Ready in {time.monotonic() - started:.0f}s; LUT-6; up to {max_tokens} input tokens", flush=True)
            if service.vision is not None:
                media = "Images and video" if service.vision.supports_video else "Images"
                print(f"{media} enabled: {service.vision.device.upper()} encoder loads on the first media request", flush=True)
            print(f"Listening on http://{host}:{server.server_port}", flush=True)
            try:
                server.serve_forever(poll_interval=0.2)
            finally:
                server.server_close()
                if service.vision is not None:
                    service.vision.close()
                service.backbone.close()
