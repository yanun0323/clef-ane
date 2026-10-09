import ast
import re
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def packaged_engine_modules() -> set[str]:
    tree = ast.parse((ROOT / "setup.py").read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and any(getattr(t, "id", None) == "runtime_modules" for t in node.targets):
            return set(ast.literal_eval(node.value))
    raise AssertionError("runtime_modules not found in setup.py")


def module_level_relative_imports(path: Path) -> list[str]:
    """`from .x import y` at module level (including `if __package__:` blocks); lazy research imports are skipped."""
    names, statements = [], list(ast.parse(path.read_text()).body)
    while statements:
        node = statements.pop()
        if isinstance(node, ast.If):
            statements += node.body
        elif isinstance(node, ast.ImportFrom) and node.level == 1 and node.module:
            names.append(node.module.split(".")[0])
    return names


class PackagingTests(unittest.TestCase):
    def test_every_engine_module_the_runtime_imports_is_packaged(self):
        """setup.py allowlists engine modules; a missing entry only fails in non-editable installs."""
        packaged = packaged_engine_modules()
        needed, pending = set(), []
        for path in (ROOT / "src/clef_ane").glob("*.py"):
            text = path.read_text()
            pending += re.findall(r"(?:from|import) (?:clef_ane)?\.?engine\.(\w+)", text)
            pending += [name for group in re.findall(r"from \.?engine import ([\w, ]+)", text) for name in group.replace(" ", "").split(",")]
        while pending:
            name = pending.pop()
            if name in needed:
                continue
            needed.add(name)
            source = ROOT / "scripts/ane" / f"{name}.py"
            self.assertTrue(source.is_file(), name)
            pending += module_level_relative_imports(source)
        self.assertIn("vision_encoder", needed)
        self.assertEqual(needed - packaged, set())


if __name__ == "__main__":
    unittest.main()
