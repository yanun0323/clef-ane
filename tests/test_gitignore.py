"""Keep private/generated files out of Git without hiding the distributable runtime."""
import os
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path


@unittest.skipUnless(shutil.which("git"), "Git is not installed")
class GitIgnoreTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        temporary = tempfile.TemporaryDirectory()
        cls.addClassCleanup(temporary.cleanup)
        cls.root = Path(temporary.name)
        cls.env = {key: value for key, value in os.environ.items() if not key.startswith("GIT_")}
        cls.env.update({"GIT_CONFIG_NOSYSTEM": "1", "GIT_CONFIG_GLOBAL": os.devnull})
        subprocess.run(["git", "-c", "init.defaultBranch=main", "init", "--quiet", "--template=", str(cls.root)],
                       env=cls.env, check=True, capture_output=True)
        shutil.copyfile(Path(__file__).resolve().parents[1] / ".gitignore", cls.root / ".gitignore")

    def ignored(self, paths):
        result = subprocess.run(["git", "-c", f"core.excludesFile={os.devnull}", "check-ignore", "--no-index", "--stdin", "-z"],
                                cwd=self.root, env=self.env, input="\0".join(paths) + "\0",
                                text=True, capture_output=True)
        self.assertIn(result.returncode, (0, 1), result.stderr)
        return set(filter(None, result.stdout.split("\0")))

    def test_credentials_and_local_settings_are_ignored(self):
        paths = [
            ".env", ".env.local", ".env.production", "config/development.env", "config/app.env.local",
            "Makefile.env", "Makefile.local.env", "private.key", "private.pem", "certificate.p12", "certificate.pfx",
            ".netrc", ".pypirc", ".ssh/id_ed25519", ".aws/credentials", ".huggingface/token", ".hf/token",
            "hf-home/token", "hf-home/stored_tokens", ".hf_token", "hf_token", "token", "stored_tokens",
            "secrets/token.json", "credentials/hf.json", "secrets.json", "credentials.json",
            ".vscode/settings.json", ".idea/workspace.xml", "local.code-workspace",
            ".pi/sessions/session.jsonl", ".claude/settings.local.json", ".codex/auth.json", "CLAUDE.md", "AGENTS.md",
        ]
        self.assertEqual(self.ignored(paths), set(paths))

    def test_model_downloads_and_generated_outputs_are_ignored(self):
        paths = [
            ".venv/bin/python", ".venv-mlx/bin/python", ".venv.test/bin/python", "venv/bin/python",
            "src/clef_ane/__pycache__/cli.cpython-312.pyc", ".pytest_cache/results", ".ruff_cache/results",
            ".mypy_cache/results", ".coverage", ".coverage.local", "htmlcov/index.html",
            "build/libane_bridge.dylib", "dist/clef_ane.whl", "clef_ane.egg-info/PKG-INFO",
            ".tmp/upload.log", "results/power/idle.jsonl", "data/weights/layer_00.safetensors",
            "docs/power-measurements.md", "docs/power-chart.svg",
            "models/clef-flash-4bit/config.json", "hf-meta/model.json", "sources/forge_README.md",
            "vendor/maderix-ANE/README.md", "exports/clef-ane.json", "bundles/clef-ane.json",
            "accidental-export/layer.safetensors", "accidental-export/programs/00/mlp.bin",
            "scratch/input.npy", "scratch/activation.npz", "scratch/model.mlpackage/Manifest.json",
            "scratch/model.mlmodelc/coremldata.bin", "scratch/model.mlmodel",
            "server.pid", ".clef-ane.lock", ".build.lock", ".source.json",
        ]
        self.assertEqual(self.ignored(paths), set(paths))

    def test_source_licenses_examples_and_dependency_locks_stay_visible(self):
        paths = [
            ".gitignore", "README.md", "Makefile", "pyproject.toml", "setup.py", "uv.lock",
            "src/clef_ane/cli.py", "src/clef_ane/paths.py", "src/clef_ane/_vendor/LICENSE",
            "src/clef_ane/_vendor/SOURCE", "src/clef_ane/_vendor/joint_schema_model.py",
            "scripts/ane/ane_bridge.m", "scripts/ane/runtime.py", "scripts/export_bundle.py", "scripts/publish_bundle.py",
            "tests/test_cli.py", "tests/test_gitignore.py", "examples/request.json",
            ".env.example", ".env.sample", ".env.template", "Makefile.env.example",
            "examples/client.sample.env", "examples/client.template.env", "config/app.env.example",
        ]
        self.assertEqual(self.ignored(paths), set())
