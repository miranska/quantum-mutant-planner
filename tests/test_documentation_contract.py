"""Static tests for type annotations and the requested Sphinx docstring style."""

import ast
from pathlib import Path
import re


def test_source_functions_are_typed_and_use_only_allowed_sphinx_fields() -> None:
    """Every source function should satisfy the documented annotation contract."""

    root = Path(__file__).resolve().parents[1] / "src" / "quantum_mutant_planner"
    failures: list[str] = []
    for source_path in sorted(root.rglob("*.py")):
        tree = ast.parse(source_path.read_text(encoding="utf-8"), filename=str(source_path))
        for node in ast.walk(tree):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            location = f"{source_path.relative_to(root)}:{node.lineno}:{node.name}"
            docstring = ast.get_docstring(node)
            if not docstring:
                failures.append(f"{location} has no docstring")
                continue
            if node.returns is None:
                failures.append(f"{location} has no return annotation")
            arguments = [*node.args.posonlyargs, *node.args.args, *node.args.kwonlyargs]
            if node.args.vararg is not None:
                arguments.append(node.args.vararg)
            if node.args.kwarg is not None:
                arguments.append(node.args.kwarg)
            for argument in arguments:
                if argument.arg not in {"self", "cls"} and argument.annotation is None:
                    failures.append(f"{location} parameter {argument.arg} has no annotation")
            fields = set(re.findall(r":([A-Za-z_]+)(?:\s+[^:]+)?:", docstring))
            disallowed = fields.difference({"param", "raises", "return"})
            if disallowed:
                failures.append(f"{location} uses disallowed Sphinx fields {sorted(disallowed)}")
    assert not failures, "\n".join(failures)
