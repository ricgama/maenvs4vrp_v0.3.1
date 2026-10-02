"""Docstrings follow the Google-style convention described in AGENTS.md."""
import ast
import re

import pytest

from tests.helpers import PKG_DIR, REPO_DIR

FILES = sorted(p for p in PKG_DIR.rglob("*.py") if "learning_notebooks" not in p.parts)
ENTRY_RE = re.compile(r"^(\*{0,2}\w+)(?: \(([^)]*(?:\([^)]*\))?[^)]*)\))?: \S")
BAD_HEADER_RE = re.compile(r"^\s*(Args|Returns|Raises|Yields)\s*$", re.M)


def args_section(doc: str) -> list[str] | None:
    """Return the top-level entries of the Args section, or None when there is no Args section."""
    m = re.search(r"^Args:\n(.*?)(?=^\S|\Z)", doc, re.M | re.S)
    if not m:
        return None
    return [line[4:] for line in m.group(1).splitlines() if line.startswith("    ") and not line.startswith("     ")]


def check_function(fn, qualname):
    problems = []
    public = not fn.name.startswith("_") or fn.name == "__init__"
    doc = ast.get_docstring(fn)
    positional = [a.arg for a in fn.args.posonlyargs + fn.args.args if a.arg not in ("self", "cls")]
    params = positional + [a.arg for a in fn.args.kwonlyargs]
    if fn.args.vararg:
        params.insert(len(positional), "*" + fn.args.vararg.arg)
    if fn.args.kwarg:
        params.append("**" + fn.args.kwarg.arg)
    if doc is None:
        return [f"{qualname}: missing docstring"] if public else []
    if BAD_HEADER_RE.search(doc):
        problems.append(f"{qualname}: section header without ':'")
    if re.search(r"\bN/a\b|^\s*n/a\.?\s*$", doc, re.M):
        problems.append(f"{qualname}: uses N/a")
    if fn.name == "__init__" and re.search(r"^Returns:", doc, re.M):
        problems.append(f"{qualname}: __init__ has a Returns section")
    entries = args_section(doc)
    if entries is None:
        if params and public:
            problems.append(f"{qualname}: parameters {params} are not documented")
        return problems
    names = []
    for entry in entries:
        m = ENTRY_RE.match(entry)
        if not m:
            problems.append(f"{qualname}: malformed Args entry {entry!r}")
            continue
        names.append(m.group(1))
    if [n.lstrip("*") for n in names] != [p.lstrip("*") for p in params]:
        problems.append(f"{qualname}: Args {names} do not match the signature {params}")
    return problems


@pytest.mark.parametrize("path", FILES, ids=[str(p.relative_to(PKG_DIR)) for p in FILES])
def test_docstrings(path):
    tree = ast.parse(path.read_text())
    problems = [] if ast.get_docstring(tree) else ["module: missing docstring"]
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            problems += check_function(node, node.name)
        elif isinstance(node, ast.ClassDef):
            if not ast.get_docstring(node):
                problems.append(f"{node.name}: missing class docstring")
            for item in node.body:
                if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    problems += check_function(item, f"{node.name}.{item.name}")
    assert not problems, f"{path.relative_to(REPO_DIR)}:\n  " + "\n  ".join(problems)
