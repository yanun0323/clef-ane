"""Send local images as embedded data to an image-capable clef-ane server."""
import argparse
import base64
import json
import mimetypes
import sys
import urllib.error
import urllib.request
from pathlib import Path

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("images", nargs="+", type=Path, help="local PNG, JPEG, or WebP files (up to four)")
parser.add_argument("--question", default="Is there readable text in the image?", help="a yes/no question about the images")
parser.add_argument("--url", default="http://127.0.0.1:8080", help="clef-ane server address")
args = parser.parse_args()
images = []
for path in args.images:
    mime = mimetypes.guess_type(path.name)[0]
    if mime not in ("image/png", "image/jpeg", "image/webp"):
        parser.error(f"Use a PNG, JPEG, or WebP file: {path}")
    images.append(f"data:{mime};base64," + base64.b64encode(path.read_bytes()).decode())
body = {"state": "Decide from the attached images.", "images": images,
        "questions": {"answer": {"type": "noul", "instructions": args.question}}}
request = urllib.request.Request(args.url.rstrip("/") + "/v1/systemone", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
try:
    with urllib.request.urlopen(request, timeout=120) as response:
        print(json.dumps(json.load(response), indent=2, ensure_ascii=False))
except urllib.error.HTTPError as error:
    print(error.read().decode(), file=sys.stderr)
    raise SystemExit(1)
