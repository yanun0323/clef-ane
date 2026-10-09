"""Send a local short clip as embedded bytes to a video-capable clef-ane server."""
import argparse
import base64
import json
import sys
import urllib.error
import urllib.request
from pathlib import Path

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("video", type=Path, help="local MP4, MOV, or WebM file (up to 16 MiB and 30 seconds)")
parser.add_argument("--question", default="Does anything move in the video?", help="a yes/no question about the video; audio is ignored")
parser.add_argument("--url", default="http://127.0.0.1:8080", help="clef-ane server address")
args = parser.parse_args()
mime = {".mp4": "video/mp4", ".mov": "video/quicktime", ".webm": "video/webm"}.get(args.video.suffix.lower())
if mime is None:
    parser.error("Use an MP4, MOV, or WebM file")
try:
    with args.video.open("rb") as file:
        raw = file.read(16 * 1024 * 1024 + 1)
except OSError as error:
    parser.error(f"Cannot read the video: {error}")
if len(raw) > 16 * 1024 * 1024:
    parser.error("Video exceeds 16 MiB; send a smaller clip")
body = {"state": "Decide from the attached video, in time order.",
        "videos": [f"data:{mime};base64," + base64.b64encode(raw).decode()],
        "questions": {"answer": {"type": "noul", "instructions": args.question}}}
request = urllib.request.Request(args.url.rstrip("/") + "/v1/systemone", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
try:
    with urllib.request.urlopen(request, timeout=180) as response:
        print(json.dumps(json.load(response), indent=2, ensure_ascii=False))
except urllib.error.HTTPError as error:
    print(error.read().decode(), file=sys.stderr)
    raise SystemExit(1)
except urllib.error.URLError as error:
    print(f"Video request failed: {error.reason}", file=sys.stderr)
    raise SystemExit(1)
