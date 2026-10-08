"""Find the first op of a graph that the ANE compiler rejects. Usage: bisect_compile.py module.function [args]"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from ane import CompileError, Kernel


def first_failure(graph, verbose=False):
    for tensor, _ in graph.defined:
        try:
            Kernel(graph.upto(tensor)).close()
            if verbose: print("ok  ", tensor.name, tensor.shape)
        except CompileError as e:
            return tensor, str(e)[:200]
    return None, ""


if __name__ == "__main__":
    import delta
    g = delta.build()
    t, msg = first_failure(g, verbose=True)
    print("first failing op:", t, "\n", msg if t else "all ops compile")


def minimize(graph, output):
    """Greedily replace intermediate tensors by inputs while the program still fails to compile."""
    cuts: list = []
    candidates = [t for t, _ in graph.defined if t.name != output.name]
    for t in reversed(candidates):
        trial = graph.upto(output, tuple(cuts + [t]))
        if not trial.lines:
            continue
        try:
            Kernel(trial).close()
        except CompileError:
            cuts.append(t)
    return graph.upto(output, tuple(cuts))
