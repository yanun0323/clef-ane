# clef-ane

Run Clef-Flash's decision API on the Apple Neural Engine with text, images, and short videos. The 32 text layers run on the ANE; the decision head runs on the CPU. Images and videos share the original vision encoder, which also runs on the ANE by default, so serving leaves the GPU idle. This is a research runtime using Apple's private ANE API, not an officially supported Apple deployment format.

## Install

Serving requires an Apple Silicon Mac, Python 3.12 or newer, and Xcode Command Line Tools (`xcode-select --install`). Hardware validation has so far been on the project's M5 Pro running macOS 27; other machines are not yet verified.

From a checkout:

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install .
```

The `clef-ane` command works from any directory while this environment is active. For development, use `python -m pip install -e .` instead.

## Download and serve

```sh
clef-ane download
clef-ane serve
```

The default model is `Yanun/clef-flash-ane-lut6` on Hugging Face (about 9.74 GiB), including shared image/video weights. Use clef-ane 0.2.0 or newer for video input. It is public, so downloading does not require a Hugging Face account. Use `--repo` to select another compatible bundle; subsequent downloads remember that repository. This command accepts only the exported clef-ane model format, not the original Cloudflare or MLX checkpoints.

Downloads can be resumed by running the same command. Each download pins a single Hub commit and checks file sizes and SHA-256 hashes. No Python code is downloaded or executed from the model repository. The LUT-6 model is already converted; serving does not need the original 17 GB checkpoint or the research LUT cache. On startup, the ANE programs are still compiled for the local machine. Serving also needs temporary disk space for those compiled programs, in addition to the model download; normal shutdown removes the program files.

The default address is `127.0.0.1:8080`. There is **no authentication**. Keep it on localhost unless you have a trusted network or an authenticated reverse proxy.

```sh
clef-ane serve --port 9000 --max-tokens 4096
clef-ane download --repo Yanun/clef-flash-ane-lut6 --revision COMMIT --model-dir /path/to/model
clef-ane serve --model-dir /path/to/model
clef-ane --help
```

The previous text-only model (about 8.90 GiB) remains available; format-1 bundles are still supported:

```sh
clef-ane download --revision text-v1 --model-dir /path/to/text-model
clef-ane serve --model-dir /path/to/text-model
```

Serving is offline: it never starts a download automatically. Default files go under `~/Library/Caches/clef-ane` on macOS. Set `CLEF_ANE_HOME` to move the cache, or pass `--model-dir` to both commands. `CLEF_ANE_REPO` overrides the default download repository.

## Send a request

```sh
curl http://127.0.0.1:8080/v1/systemone \
  -H 'Content-Type: application/json' \
  -d '{"state":"The invoice is overdue.","questions":{"late":{"type":"noul","instructions":"Is the payment late?"}}}'
```

Endpoints: `POST /v1/systemone`, `GET /health`, `GET /v1/models`. Supports yes/no (`noul`), choice, and score questions, up to 64 questions per request and 16,384 input tokens. Long state text is truncated to leave room for the media and questions. Audio and text/image/video generation are not supported. Requests are processed one at a time; one additional request may wait, and a full queue returns HTTP 503. `/health` reports `input_modalities` for the loaded bundle.

## Image input

The default download supports images and videos (bundle format 2). To upgrade an older text-only download, update the CLI and run `clef-ane download` again. The existing LUT-6 ANE programs are unchanged. By default the 27 vision layers run on the ANE in fp16 (not LUT-6), so the server does no GPU work. Patch embedding, positions, and the final merge run on the CPU. The first image or video request compiles the vision programs, which takes about 10 seconds; later requests reuse them. Each ANE call handles up to 1024 patches (one 512×512 image or four video frame pairs); larger requests use several calls.

`--vision-device mps` uses the Metal GPU instead: it is faster (about 0.4 s instead of 0.6 s for one image) but competes with other GPU work. `--vision-device cpu` is the slowest. Serving remains offline.

In another terminal, send a local image using the example client (the file is read on the client, not by the server):

```sh
python examples/image_request.py receipt.png --question "Is the receipt marked as paid?"
```

The JSON request uses `images: ["data:image/png;base64,..."]` alongside the existing `state` and `questions` fields. Multiple images stay in list order.

- PNG, JPEG, or WebP still images only; no remote URLs, server file paths, animations, or request-supplied `media_kwargs`.
- Up to 4 images; 4 MiB compressed per image / 8 MiB combined; 16 megapixels per image / 32 megapixels combined.
- The default HTTP body limit is 32 MiB **including base64 overhead**; `--max-body-mb` changes that outer limit, not the media limits.
- Images use the upstream **Pillow** preprocessing backend and are resized to at most about 262,144 pixels each (256 visual tokens). Small text and fine detail can be lost; resizing may differ from the upstream default torchvision backend.
- Images and questions must fit the selected context. They are never silently dropped; oversized input returns HTTP 413. Invalid image data returns HTTP 400.

## Video input

Use the same downloaded model and server. The example reads the file on the client and embeds its bytes in the request:

```sh
python examples/video_request.py clip.mp4 --question "Does the person enter the room?"
```

Send `videos: ["data:video/mp4;base64,..."]` with `state` and `questions`. The MIME types are `video/mp4`, `video/quicktime` for MOV, and `video/webm`. A request may also contain images; the model sees all images first, then the video. **Audio is ignored.**

- One MP4, MOV, or WebM clip per request, up to 16 MiB and 30 seconds. MP4/MOV supports H.264, HEVC, MPEG4, and AV1; WebM supports VP8, VP9, and AV1. No URLs, server paths, or caller-supplied decoding options.
- Limits: 1800 source frames, 4 megapixels per frame, 4096 pixels per side, and 4 billion decoded pixels total. Clips need increasing timestamps, fixed dimensions, square pixels, and one video track. Trim audio-only tails if the declared duration exceeds the video. Mirrored display metadata is rejected; re-encode the displayed orientation. Damage checks cannot distinguish a cleanly cut clip without an original duration/frame count from a shorter complete clip.
- The decoder samples the first frame in each half-second time bucket plus the final frame (at most 64). It keeps actual presentation times relative to the first frame, including variable-frame-rate clips.
- Frames are resized with Pillow to at most 65,536 pixels each. Each pair becomes one visual block labelled with its mean timestamp, rounded to one decimal. An odd final frame is repeated. Fine detail and events between sampled frames can be missed.
- Sampled video, images, and questions must fit the selected context. They are never silently dropped. HTTP 413 means a size, time, or context limit was exceeded; HTTP 400 means invalid input. `usage.video_frames` reports the sampled frame count before odd-frame padding.
- Decoding runs in a child process with a 45-second deadline and file-output limits. This is **not an OS security sandbox**; keep the unauthenticated server on localhost or behind a trusted proxy.

Image/video regression checks compare synthetic examples with the original BF16 multimodal forward under the same bounded preprocessing. They are not general accuracy or energy-efficiency benchmarks, nor a claim of equivalence to upstream default video sampling or torchvision resizing. One retained mixed-media fixture exposes a source-model error: both BF16 and ANE treat the green still image as the beginning of a red-to-blue video. Matching the source model does not mean every answer is correct.

## Development

```sh
make test             # fast CLI, download, bundle and HTTP tests; no model or ANE required
make test-model       # existing ANE service regression tests; needs original research weights
make run              # clef-ane serve
make download         # clef-ane download
make test-images -- --model-dir bundles/clef-flash-ane-media
make test-videos -- --model-dir bundles/clef-flash-ane-media
                      # BF16 vs ANE media and HTTP checks; require original research weights
```

To add image/video weights to a format-1 text bundle without downloading the original 17 GB decoder checkpoint:

```sh
make fetch-vision  # fixed upstream revision; about 0.85 GiB
make export-images -- --text-bundle /path/to/text-model --output bundles/clef-flash-ane-media
```

To export a release from the existing research weights and LUT-6 cache:

```sh
python scripts/export_bundle.py --output /path/to/clef-flash-ane-lut6
```

The export contains packed LUT-6 ANE programs, tokenizer files, CPU weights, upstream attribution, and a checksummed manifest. Add `--with-images` to include shared image/video weights and processor settings, or use `--with-images --text-bundle /path/to/text-bundle` to reuse existing text programs. Vision weights must have the matching receipt created by `make fetch-vision`.

Exports do not contain the original decoder weights, research inputs, local paths, or credentials. Only the large ANE linear-layer weights are LUT-6; embeddings and the CPU decision head keep their original stored precision. This bundle is specific to `clef-ane`, not a Transformers or MLX model.

To publish the exported bundle to the logged-in account's **public** Hugging Face repository:

```sh
hf auth login
python scripts/publish_bundle.py --model-dir /path/to/clef-flash-ane-lut6
```

The publisher verifies the files, obtains the account name from Hugging Face, and uploads only the manifest, listed model files, and model card. It refuses an existing private repository or an unrelated model, and checks anonymous access after upload. The release already defaults to `Yanun/clef-flash-ane-lut6`. If publishing a fork, update `DEFAULT_REPO_ID` in `src/clef_ane/paths.py` before building its wheel.

The pinned Cloudflare encoder is included under `src/clef_ane/_vendor/` with its Apache-2.0 license. Model attribution and the upstream model license are included in every exported bundle.
