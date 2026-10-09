.DEFAULT_GOAL := help

-include Makefile.env
export

.PHONY: $(wildcard *)

# Everything after the target name is passed to the command: `make run -- --port 9000`
ARGS := $(wordlist 2,$(words $(MAKECMDGOALS)),$(MAKECMDGOALS))

HOST ?= 127.0.0.1
PORT ?= 8080
PY := .venv/bin/python -I
CLI := .venv/bin/clef-ane
HF := https://huggingface.co/Cloudflare/clef-flash/resolve/main

## help: show help
help:
	@echo ""
	@echo "Usage:"
	@echo ""
	@sed -n 's/^## //p' Makefile | column -t -s ':' | sed -e 's/^/\t/'
	@echo ""

## build: compile the ANE bridge (build/libane_bridge.dylib)
build:
	@mkdir -p build
	xcrun --sdk macosx clang -O2 -fno-objc-arc -dynamiclib -framework Foundation -framework IOSurface scripts/ane/ane_bridge.m -o build/libane_bridge.dylib

## install: install the clef-ane CLI into .venv for development
install:
	$(PY) -m pip install -e .

## run: start clef-ane serve (HOST and PORT set the address; pass options after --)
run:
	@test -x $(CLI) || { echo "CLI not installed: run make install"; exit 1; }
	$(CLI) serve --host $(HOST) --port $(PORT) $(ARGS)

## download: download and verify the published LUT-6 model (pass --repo or other options after --)
download:
	@test -x $(CLI) || { echo "CLI not installed: run make install"; exit 1; }
	$(CLI) download $(ARGS)

## package: build the standalone installer bundle and install.sh into dist/ (network and Xcode Command Line Tools needed)
package:
	$(PY) scripts/build_standalone.py $(ARGS)

## request: send examples/request.json to the running server (another file with `make request -- file.json`)
request:
	curl -sS http://$(HOST):$(PORT)/v1/systemone -d @$(or $(ARGS),examples/request.json) | $(PY) -m json.tool

## test: run fast CLI, download, bundle and HTTP regression tests (no model or ANE required)
test:
	$(PY) -m unittest discover -s tests -v

## test-model: run the existing ANE service regression tests against the bf16 reference (requires research weights)
test-model: build
	$(PY) scripts/ane/test_service.py

## fetch: download original research weights (about 17 GB); not needed by clef-ane serve
fetch:
	@mkdir -p data hf-meta
	@for f in config.json tokenizer.json tokenizer_config.json; do [ -f data/$$f ] || curl -fL $(HF)/$$f -o data/$$f; done
	@[ -f hf-meta/joint_schema_model.py ] || curl -fL $(HF)/joint_schema_model.py -o hf-meta/joint_schema_model.py
	@for j in 0 1 2 3 4 5; do $(PY) scripts/fetch_all.py $$j 6 & done; wait

## fetch-vision: download the pinned vision weights and settings (about 0.85 GiB; no text weights)
fetch-vision:
	$(PY) scripts/fetch_vision.py $(ARGS)

## export-images: add image/video weights to a text bundle (pass --text-bundle and --output after --)
export-images:
	$(PY) scripts/export_bundle.py --with-images $(ARGS)

## test-images: compare image decisions with BF16 and exercise HTTP (research weights; pass --model-dir after --)
test-images: build
	$(PY) scripts/verify_images.py $(ARGS)

## test-videos: compare video, mixed-media and image decisions with BF16 (research weights; pass --model-dir after --)
test-videos: build
	$(PY) scripts/verify_videos.py $(ARGS)

## power: measure idle / ANE / GPU rail power with powermetrics (asks for your sudo password)
power:
	sudo scripts/measure_power.sh

## power-system: measure whole-machine power with macmon (no sudo, needs brew install macmon, about 9 minutes)
power-system:
	scripts/measure_system_power.sh


# NOTE: Add command before this Footer
%:
	@:
