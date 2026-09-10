"""Safe, vectorised arithmetic expressions for job-scoring formulas.

Expressions come from YAML. We parse to an AST, whitelist node types and
function names, then compile once. Evaluation binds names to numpy arrays, so
one call scores every pending job. Not allowed: attribute access, subscripts,
lambdas, comprehensions, `if` expressions and `and`/`or` (not vectorisable —
use where(cond, a, b) and & / | instead), and any name not in the namespace.
"""
from __future__ import annotations

import ast
from typing import Callable

import numpy as np

FUNCS = {
    "min": np.minimum, "max": np.maximum, "abs": np.abs, "exp": np.exp,
    "log": np.log, "log1p": np.log1p, "sqrt": np.sqrt, "floor": np.floor,
    "ceil": np.ceil, "pow": np.power, "where": np.where, "clip": np.clip,
}

_ALLOWED = (
    ast.Expression, ast.BinOp, ast.UnaryOp, ast.Compare, ast.Call, ast.Name,
    ast.Load, ast.Constant,
    ast.Add, ast.Sub, ast.Mult, ast.Div, ast.FloorDiv, ast.Mod, ast.Pow,
    ast.BitAnd, ast.BitOr, ast.Invert, ast.UAdd, ast.USub,
    ast.Lt, ast.LtE, ast.Gt, ast.GtE, ast.Eq, ast.NotEq,
)


class ExprError(ValueError):
    pass


def compile_expr(expr: str, variables: set[str]) -> Callable[[dict], np.ndarray]:
    try:
        tree = ast.parse(expr, mode="eval")
    except SyntaxError as e:
        raise ExprError(f"syntax error in expression {expr!r}: {e}") from e
    names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            if not isinstance(node.func, ast.Name) or node.func.id not in FUNCS:
                raise ExprError(f"only {sorted(FUNCS)} may be called")
            if node.keywords:
                raise ExprError("keyword arguments are not allowed")
        elif not isinstance(node, _ALLOWED):
            raise ExprError(f"disallowed syntax {type(node).__name__} in {expr!r}")
        if isinstance(node, ast.Name) and not (
                isinstance(getattr(node, "ctx", None), ast.Load)):
            raise ExprError("assignment is not allowed")
        if isinstance(node, ast.Name):
            names.add(node.id)
        if isinstance(node, ast.Constant) and not isinstance(node.value, (int, float)):
            raise ExprError("only numeric literals are allowed")
    free = names - set(FUNCS)
    unknown = free - set(variables)
    if unknown:
        raise ExprError(f"unknown variable(s) {sorted(unknown)}; "
                        f"available: {sorted(variables)}")
    code = compile(tree, "<expr>", "eval")
    g = {"__builtins__": {}, **FUNCS}

    def fn(ns: dict) -> np.ndarray:
        return eval(code, g, ns)  # noqa: S307 — AST-validated above

    fn.variables = free
    fn.source = expr
    return fn
