# clef-ane

Run Clef-Flash's text decision API on the Apple Neural Engine. The 32 text layers run on the ANE; the decision head runs on the CPU. This is a research runtime using Apple's private ANE API, not an officially supported Apple deployment format.

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

The default model is `Yanun/clef-flash-ane-lut6` on Hugging Face (about 8.90 GiB). It is public, so downloading does not require a Hugging Face account. Use `--repo` to select another compatible bundle; subsequent downloads remember that repository. This command accepts only the exported clef-ane model format, not the original Cloudflare or MLX checkpoints.

Downloads can be resumed by running the same command. Each download pins a single Hub commit and checks file sizes and SHA-256 hashes. No Python code is downloaded or executed from the model repository. The LUT-6 model is already converted; serving does not need the original 17 GB checkpoint or the research LUT cache. On startup, the ANE programs are still compiled for the local machine. Serving also needs temporary disk space for those compiled programs, in addition to the model download; normal shutdown removes the program files.

The default address is `127.0.0.1:8080`. There is **no authentication**. Keep it on localhost unless you have a trusted network or an authenticated reverse proxy.

```sh
clef-ane serve --port 9000 --max-tokens 4096
clef-ane download --repo Yanun/clef-flash-ane-lut6 --revision COMMIT --model-dir /path/to/model
clef-ane serve --model-dir /path/to/model
clef-ane --help
```

Serving is offline: it never starts a download automatically. Default files go under `~/Library/Caches/clef-ane` on macOS. Set `CLEF_ANE_HOME` to move the cache, or pass `--model-dir` to both commands. `CLEF_ANE_REPO` overrides the default download repository.

## Send a request

```sh
curl http://127.0.0.1:8080/v1/systemone \
  -H 'Content-Type: application/json' \
  -d '{"state":"The invoice is overdue.","questions":{"late":{"type":"noul","instructions":"Is the payment late?"}}}'
```

Endpoints: `POST /v1/systemone`, `GET /health`, `GET /v1/models`. Supports yes/no (`noul`), choice, and score questions, up to 64 questions per request and 16,384 input tokens. Long state text is truncated to leave room for the questions. Images and videos are not supported. Requests are processed one at a time.

## Development

```sh
make test             # fast CLI, download, bundle and HTTP tests; no model or ANE required
make test-model       # existing ANE service regression tests; needs original research weights
make run              # clef-ane serve
make download         # clef-ane download
```

To export a release from the existing research weights and LUT-6 cache:

```sh
python scripts/export_bundle.py --output /path/to/clef-flash-ane-lut6
```

The export contains packed LUT-6 ANE programs, tokenizer files, CPU weights, upstream attribution, and a checksummed manifest. It does not contain the original decoder weights, research inputs, local paths, or credentials. Only the large ANE linear-layer weights are LUT-6; embeddings and the CPU decision head keep their original stored precision. This bundle is specific to `clef-ane`, not a Transformers or MLX model.

To publish the exported bundle to the logged-in account's **public** Hugging Face repository:

```sh
hf auth login
python scripts/publish_bundle.py --model-dir /path/to/clef-flash-ane-lut6
```

The publisher verifies the files, obtains the account name from Hugging Face, and uploads only the manifest, listed model files, and model card. It refuses an existing private repository or an unrelated model, and checks anonymous access after upload. The release already defaults to `Yanun/clef-flash-ane-lut6`. If publishing a fork, update `DEFAULT_REPO_ID` in `src/clef_ane/paths.py` before building its wheel.

The pinned Cloudflare encoder is included under `src/clef_ane/_vendor/` with its Apache-2.0 license. Model attribution and the upstream model license are included in every exported bundle.
