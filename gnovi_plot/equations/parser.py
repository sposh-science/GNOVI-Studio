"""Safe parsing of user-entered calculated-column formulas.

The only place formula text is parsed -- never a raw Python `eval()`. Every
token in the formula is checked against a strict whitelist before the text
is ever handed to SymPy: SymPy's own parser evaluates a transformed Python
expression string internally, and by default treats any unrecognized bare
name as an inert symbol rather than raising, which is not strict enough on
its own to reject things like `__import__`, `open`, or attribute access.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import sympy
from sympy.parsing.sympy_parser import parse_expr, standard_transformations


class FormulaError(Exception):
    """Raised for any formula that fails safe parsing or evaluation."""


_ALLOWED_FUNCTIONS = {
    "sqrt": sympy.sqrt,
    "exp": sympy.exp,
    "abs": sympy.Abs,
    "ln": sympy.log,
    "log": lambda x: sympy.log(x, 10),
}

_BRACKET_RE = re.compile(r"\[([^\[\]]+)\]")
_IDENTIFIER_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
_PLACEHOLDER_RE = re.compile(r"_col\d+")

# Ordered so numeric literals (including scientific notation, e.g. 1e-5) are
# consumed whole before the identifier pattern can mistake a trailing "e"
# for a name. Any character that matches none of these (quotes, semicolons,
# braces, stray brackets, etc.) falls through to OTHER and is rejected.
_TOKEN_RE = re.compile(
    r"(?P<NUMBER>\d+\.\d+(?:[eE][+-]?\d+)?|\.\d+(?:[eE][+-]?\d+)?|\d+(?:[eE][+-]?\d+)?)"
    r"|(?P<IDENT>[A-Za-z_][A-Za-z0-9_]*)"
    r"|(?P<OP>\*\*|[+\-*/(),])"
    r"|(?P<SPACE>\s+)"
    r"|(?P<OTHER>.)"
)


@dataclass
class ParsedFormula:
    expression: sympy.Expr
    column_symbols: dict[str, str]  # placeholder name -> real column name


def parse_formula(formula: str, columns: list[str]) -> ParsedFormula:
    """Safely parse `formula` into a SymPy expression.

    Columns are referenced either as a bare identifier (when the column name
    is itself a valid identifier, e.g. `ScanRate`) or bracketed
    (`[Current/A]`) for names that aren't valid identifiers -- the raw
    column name itself is never altered. Supports `+ - * / ** ( )` and the
    functions `sqrt`, `log` (base 10), `ln` (natural log), `exp`, `abs`.
    """
    if not formula or not formula.strip():
        raise FormulaError("Formula must not be empty")

    placeholder_to_column: dict[str, str] = {}
    column_to_placeholder: dict[str, str] = {}

    def _register_column(name: str) -> str:
        placeholder = column_to_placeholder.get(name)
        if placeholder is None:
            placeholder = f"_col{len(placeholder_to_column)}"
            placeholder_to_column[placeholder] = name
            column_to_placeholder[name] = placeholder
        return placeholder

    def _bracket_sub(match: re.Match) -> str:
        name = match.group(1).strip()
        if name not in columns:
            raise FormulaError(f"Unknown column: '{name}'")
        return _register_column(name)

    working = _BRACKET_RE.sub(_bracket_sub, formula)
    valid_bare_columns = {c for c in columns if _IDENTIFIER_RE.fullmatch(c)}

    out_tokens: list[str] = []
    for match in _TOKEN_RE.finditer(working):
        kind = match.lastgroup
        text = match.group()
        if kind in ("NUMBER", "OP"):
            out_tokens.append(text)
        elif kind == "SPACE":
            out_tokens.append(" ")
        elif kind == "IDENT":
            if _PLACEHOLDER_RE.fullmatch(text) and text in placeholder_to_column:
                out_tokens.append(text)
            elif text in _ALLOWED_FUNCTIONS:
                out_tokens.append(text)
            elif text in valid_bare_columns:
                out_tokens.append(_register_column(text))
            else:
                raise FormulaError(f"Unknown name in formula: '{text}'")
        else:
            raise FormulaError(f"Formula contains a character that is not permitted: '{text}'")

    working = "".join(out_tokens)

    local_dict = {name: sympy.Symbol(name) for name in placeholder_to_column}
    local_dict.update(_ALLOWED_FUNCTIONS)

    # `standard_transformations` (auto_number/auto_symbol) generates code that
    # calls sympy's own literal constructors -- these are inert math-object
    # constructors, not a route to arbitrary execution, so they're safe to
    # expose even with `__builtins__` blocked below.
    safe_global_dict = {
        "__builtins__": {},
        "Integer": sympy.Integer,
        "Float": sympy.Float,
        "Symbol": sympy.Symbol,
        "Rational": sympy.Rational,
    }

    try:
        expr = parse_expr(
            working,
            local_dict=local_dict,
            global_dict=safe_global_dict,
            transformations=standard_transformations,
            evaluate=True,
        )
    except FormulaError:
        raise
    except Exception as exc:
        raise FormulaError(f"Invalid formula: {exc}") from exc

    allowed_symbols = {local_dict[name] for name in placeholder_to_column}
    unresolved = expr.free_symbols - allowed_symbols
    if unresolved:
        raise FormulaError(f"Unknown name in formula: '{unresolved.pop()}'")

    return ParsedFormula(expression=expr, column_symbols=placeholder_to_column)
