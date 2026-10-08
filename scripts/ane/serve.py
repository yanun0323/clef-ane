"""HTTP server for Clef-Flash on the ANE: POST /v1/systemone, GET /health, GET /v1/models.

    .venv/bin/python -I scripts/ane/serve.py [--port 8080] [--host 127.0.0.1] [--buckets 1024,2048,4096,8192,16384]

The model stays loaded on the ANE between requests. There is no authentication, so it listens on localhost only
unless you pass --host explicitly.
"""
import argparse
import json
import sys
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from service import DEFAULT_LUT_BITS, RequestError, Service, TooLong

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--host", default="127.0.0.1")
parser.add_argument("--port", type=int, default=8080)
parser.add_argument("--name", default="clef-flash", help="model name reported in responses")
parser.add_argument("--buckets", default="1024,2048,4096,8192,16384", help="context sizes to compile (comma separated)")
parser.add_argument("--max-body-mb", type=int, default=8)
parser.add_argument("--lut-bits", type=int, choices=(0, 4, 6), default=DEFAULT_LUT_BITS,
                    help="store the linear layers as 4- or 6-bit lookup tables, 0 for fp16 (default %(default)s). "
                         "The first start with a new setting quantises every layer (about 7 minutes) and caches it in data/lutcache")
args = parser.parse_args()

service = Service.load(args.name, tuple(int(b) for b in args.buckets.split(",")), lut_bits=args.lut_bits)
max_body = args.max_body_mb * 1024 * 1024


class Handler(BaseHTTPRequestHandler):
    server_version = "clef-ane"

    def _send(self, status: int, body: dict) -> None:
        data = json.dumps(body, ensure_ascii=False).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _error(self, status: int, message: str) -> None:
        self._send(status, {"error": {"message": message}})

    def log_message(self, fmt, *a):  # one line per request on stderr
        sys.stderr.write(f"{self.log_date_time_string()} {fmt % a}\n")

    def do_GET(self):
        if self.path == "/health":
            self._send(200, {"status": "ok", "model": service.name, "max_tokens": service.backbone.max_tokens})
        elif self.path == "/v1/models":
            self._send(200, {"object": "list", "data": [{"id": service.name, "object": "model"}]})
        else:
            self._error(404, f"not found: {self.path}")

    def do_POST(self):
        if self.path != "/v1/systemone":
            return self._error(404, f"not found: {self.path}")
        try:
            length = int(self.headers.get("Content-Length") or 0)
        except ValueError:
            return self._error(400, "bad Content-Length")
        if length > max_body:
            return self._error(413, f"request body larger than {args.max_body_mb} MB")
        try:
            request = json.loads(self.rfile.read(length))
            if not isinstance(request, dict):
                raise ValueError("request body must be a JSON object")
        except (json.JSONDecodeError, ValueError) as error:
            return self._error(400, str(error))
        started = time.perf_counter()
        try:
            response = service.systemone(request)
        except TooLong as error:
            return self._error(413, f"maximum context length exceeded: {error}")
        except RequestError as error:
            return self._error(400, str(error))
        except Exception as error:  # a failed ANE call must not take the server down silently
            sys.stderr.write(f"internal error: {error!r}\n")
            return self._error(500, "internal error")
        self.log_message("systemone %d tokens in %.2fs", response["usage"]["input_tokens"], time.perf_counter() - started)
        self._send(200, response)


server = ThreadingHTTPServer((args.host, args.port), Handler)
print(f"listening on http://{args.host}:{args.port}", flush=True)
try:
    server.serve_forever()
except KeyboardInterrupt:
    pass
