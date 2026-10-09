# clef-ane

<p align="center">
  <a href="README.md"><img src="https://img.shields.io/badge/English-Click-yellow" alt="English"></a>
  <a href="README-tw.md"><img src="https://img.shields.io/badge/繁體中文-點擊查看-orange" alt="繁體中文"></a>
  <a href="README-cn.md"><img src="https://img.shields.io/badge/简体中文-点击查看-orange" alt="简体中文"></a>
  <a href="README-ja.md"><img src="https://img.shields.io/badge/日本語-クリック-blue" alt="日本語"></a>
  <a href="README-ko.md"><img src="https://img.shields.io/badge/한국어-클릭-yellow" alt="한국어"></a>
</p>

clef-ane runs Cloudflare's [Clef-Flash](https://huggingface.co/Cloudflare/clef-flash) on your Mac. Clef-Flash is a model that makes decisions: you describe a situation and ask questions, and it tells you how likely each answer is. You can describe the situation with text, images, or a short video.

It runs on the Mac's Neural Engine (ANE), a chip built for AI work, so your graphics card (GPU) stays free for other things. Everything runs on your Mac, without an internet connection.

This is an experiment. It uses parts of macOS that Apple does not officially support, so a macOS update could break it.

## What it can do

- Answer three kinds of questions: yes/no, pick one option, or rate on a scale.
- Read text, JSON, up to 4 images, and one short video in the same request.
- Leave the GPU completely unused.
- Work offline after a one-time download.
- Use the same request format as Cloudflare's SystemOne API.

## What you need

- A Mac with Apple Silicon (M1 or newer) and macOS 14 or newer. So far it has only been tested on an M5 Pro with macOS 27.
- About 11 GiB of free disk space: about 1 GiB for the program and about 10 GiB for the model.

## Install

Run this in Terminal:

```sh
curl -fsSL https://github.com/yanun0323/clef-ane/releases/latest/download/install.sh | sh
```

It downloads the program (about 260 MB), checks that the download is intact, and puts the `clef-ane` command in `~/.local/bin`. You don't need Python or Xcode. If the installer says `~/.local/bin` is not on your PATH, follow the line it prints, then open a new terminal.

Run the same command again to update. To uninstall, delete `~/.local/share/clef-ane` and `~/.local/bin/clef-ane`; downloaded models are in `~/Library/Caches/clef-ane`.

To install from the source code instead (needs Python 3.12 or newer and `xcode-select --install`), run this in the project folder:

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install .
```

## Start

Download the model once (about 9.74 GiB), then start the server:

```sh
clef-ane download
clef-ane serve
```

- No Hugging Face account is needed. If the download stops, run the same command again to continue.
- Starting takes about 1.5 minutes. It's ready when you see `Listening on http://127.0.0.1:8080`.
- The first request with an image or video takes about 10 seconds longer. After that it's fast.
- Press `Ctrl-C` to stop.

> [!WARNING]
> The server has no password. By default only your own Mac can reach it. Don't open it to other computers unless you trust the network.

## Ask a question

In another terminal window:

```sh
curl http://127.0.0.1:8080/v1/systemone \
  -H 'Content-Type: application/json' \
  -d @examples/request.json
```

[`examples/request.json`](examples/request.json) describes a customer who was charged twice and asks three questions. You get back:

```json
{
  "model": "clef",
  "answers": {
    "team": {"type": "choice", "choice": "billing", "confidence": 0.9647,
             "probabilities": {"billing": 0.9647, "technical": 0.0142, "other": 0.0211}},
    "refund": {"type": "noul", "noul": 0.8871},
    "urgency": {"type": "score", "score": 1.6367, "confidence": 0.7115,
                "legend": {"0": "Routine", "1": "Soon", "2": "Urgent"},
                "probabilities": {"0": 0.0748, "1": 0.2137, "2": 0.7115}}
  },
  "usage": {"input_tokens": 343, "output_tokens": 0}
}
```

This means: the ticket should go to billing (96% likely), the customer is asking for a refund (89% likely), and it is most likely urgent.

### How to write a request

A request has a `state` (the situation, as text or JSON) and up to 64 `questions`. Each question has:

| Field | Meaning |
|---|---|
| `type` | `noul` for yes/no, `choice` to pick one option, `score` to rate on a scale |
| `instructions` | The question itself. Optional |
| `criteria` | The options. For `choice`, names and descriptions; for `score`, a list from lowest to highest |

The answer for `noul` is the chance of "yes". For `choice` and `score`, you get the chance of every option.

## Images and video

The example scripts read a file on your Mac and send it:

```sh
python examples/image_request.py receipt.png --question "Is the receipt marked as paid?"
python examples/video_request.py clip.mp4 --question "Does the person enter the room?"
```

To send files yourself, put them in the request as text, encoded with base64:

```json
{
  "state": "Decide from the attached image.",
  "images": ["data:image/png;base64,iVBORw0..."],
  "questions": {"paid": {"type": "noul", "instructions": "Is the receipt marked as paid?"}}
}
```

For video, use `"videos": ["data:video/mp4;base64,..."]`. MOV uses `video/quicktime` and WebM uses `video/webm`.

| | Images | Video |
|---|---|---|
| File types | PNG, JPEG, WebP | MP4, MOV, WebM |
| How many | Up to 4 | 1 |
| Largest file | 4 MiB each, 8 MiB total | 16 MiB |
| Longest or biggest | 16 megapixels each | 30 seconds |
| What the model sees | Each image shrunk to about 512×512 | About 2 frames per second, each shrunk to about 256×256 |

**If a file is too big, you get an error.** Images and video within the limits are shrunk automatically (never cropped). Anything over the limits is rejected; it is not cut or compressed for you. Trim long videos and compress large files first.

Also good to know:

- Sound in videos is ignored.
- Small text and quick movements can be missed, because images are shrunk and only some video frames are used.
- Links and file paths don't work. Send the file contents in the request.

## Options

| Option | Default | What it does |
|---|---|---|
| `--port` | `8080` | Port number to listen on |
| `--host` | `127.0.0.1` | Which address to listen on. The default only allows your own Mac |
| `--max-tokens` | `16384` | Longest input allowed. A smaller number uses less memory |
| `--max-body-mb` | `32` | Largest request size in MiB |
| `--vision-device` | `ane` | Where images and video are processed. `ane` keeps the GPU free; `mps` uses the GPU and is a little faster; `cpu` is slowest |
| `--model-dir` | in `~/Library/Caches/clef-ane` | Where the model is stored |

See all options with `clef-ane --help`. You can also set `CLEF_ANE_HOME` to choose where models are stored.

The older text-only model (about 8.90 GiB) is still available:

```sh
clef-ane download --revision text-v1 --model-dir ~/clef-text-model
clef-ane serve --model-dir ~/clef-text-model
```

## Known limits

- One request is handled at a time. One more can wait; others get a "busy" error.
- On our 27 test questions, it gave the same answers as the original model. We have not run the official Clef-Flash benchmarks, so overall accuracy may differ.
- Images and video are used at a lower resolution than the original model allows, so detailed pictures may be less accurate.
- It only scores the options you give it. It does not write text.

## To do

The original Clef-Flash can do these, but clef-ane can't yet:

- [ ] Handle several requests at once.
- [ ] Use sharper images and more than 4 images.
- [ ] Use longer videos, more frames, and more than one video.
- [ ] Let you adjust how images and video are processed (`media_kwargs`).
- [ ] Let you limit how much of the situation text is used (`max_state_tokens`).
- [ ] Return the raw scores behind each answer (logits).
- [ ] Optionally shrink or trim files that are too big, instead of rejecting them.
- [ ] Support the larger Clef model.
- [ ] Run the official Decision Index benchmarks.

## For developers

```sh
make install   # install for development
make test      # quick tests, no model needed
make run       # start the server
make request   # send examples/request.json to the server
make package   # build the installer files into dist/
make help      # list all commands
```

Comparing results with the original model (`make test-model`, `make test-images`, `make test-videos`) needs the original model files. Scripts for building and uploading the model are in [`scripts/`](scripts/).

## License

clef-ane is released under the [Apache-2.0 license](LICENSE). The installer bundle also contains third-party software under its own licenses, listed in `THIRD_PARTY_NOTICES.txt` inside the install folder; this includes FFmpeg codec libraries (x264 and x265) under the GPL.

Clef-Flash was made by Cloudflare from Qwen's Qwen3.5-9B. Both use the Apache-2.0 license. The downloaded model includes the `LICENSE` and `NOTICE` files, and Cloudflare's code in [`src/clef_ane/_vendor/`](src/clef_ane/_vendor/) keeps its license. This is not an official Cloudflare, Qwen, or Apple project.
