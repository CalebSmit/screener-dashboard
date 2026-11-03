"""The dashboard's half of the computed-sentence sweep: a small JavaScript reader.

`tests/test_claims_register.py` sweeps `research_page.py` and
`run_screener.build_screener_overview` by walking their Python f-strings, because
those generators write their prose in Python. `generate_dashboard.py` does not:
its Python f-strings hold one log line and no computed figure, and every sentence
a reader sees with a number in it is built **in the browser**, by JavaScript the
generator emits - template literals (`` `... ${x.toFixed(1)} ...` ``) and `+`
chains (`'The median this run is ' + k.median_composite.toFixed(1) + ...`). A
Python sweep of that file finds nothing to register, which is the state the
audit's "biggest of the three surfaces" was in until 2026-11-02.

So this reads the JavaScript **the generator emits** (`generate_html()`'s inline
scripts - rule 10: the generator's output, never the committed `index.html`),
and turns every string-building expression into the same shape the Python sweep
works on: literal text with holes, each hole its own source expression. The unit
splitting, the tag rules and the stat-card merge are the Python sweep's, shared,
so the two surfaces cannot drift apart on what a "sentence" is.

It is a reader for **this generator's** JavaScript, not a JavaScript parser. It
lexes strings, template literals (nested), comments and regex literals, and finds
`+` chains by operand; that is all a sentence needs. If the generator ever emits
something it cannot lex it raises, rather than skipping text silently - a sweep
that quietly reads less passes vacuously.
"""

from __future__ import annotations

import re

_IDENT = re.compile(r"[A-Za-z_$][\w$]*")
_NUM = re.compile(r"0[xX][0-9a-fA-F]+|\d[\d_]*\.?\d*(?:[eE][+-]?\d+)?|\.\d+(?:[eE][+-]?\d+)?")
_PUNCT = re.compile(
    r">>>=|===|!==|\*\*=|<<=|>>=|>>>|\.\.\.|=>|==|!=|<=|>=|&&|\|\||\?\?|\?\.(?!\d)|\+\+|--|"
    r"\+=|-=|\*=|/=|%=|&=|\|=|\^=|\*\*|<<|>>|[{}()\[\];,<>+\-*/%&|^!~?:=.@#]")
# a `/` after one of these starts a regex literal, not a division
_REGEX_AFTER = frozenset("(,=:[!&|?{};+-*%<>~^") | {
    "return", "typeof", "case", "do", "else", "in", "of", "=>", "&&", "||", "??",
    "==", "===", "!=", "!==", "<=", ">="}
_ESCAPES = {"n": "\n", "t": "\t", "r": "\r", "b": "\b", "f": "\f", "v": "\v", "0": "\0"}


class Tok:
    __slots__ = ("kind", "val", "start", "end")

    def __init__(self, kind: str, val, start: int, end: int):
        self.kind, self.val, self.start, self.end = kind, val, start, end

    def is_(self, *vals: str) -> bool:
        return self.kind in ("punct", "word") and self.val in vals

    def __repr__(self) -> str:      # pragma: no cover - debugging aid
        return f"{self.kind}:{self.val!r}"


def _escape(src: str, j: int) -> tuple[str, int]:
    """The character a backslash escape at src[j] stands for, and where it ends."""
    nxt = src[j + 1]
    if nxt == "u":
        if src[j + 2] == "{":
            k = src.index("}", j)
            return chr(int(src[j + 3:k], 16)), k + 1
        return chr(int(src[j + 2:j + 6], 16)), j + 6
    if nxt == "x":
        return chr(int(src[j + 2:j + 4], 16)), j + 4
    if nxt == "\n":
        return "", j + 2
    return _ESCAPES.get(nxt, nxt), j + 2


def _read_string(src: str, i: int) -> tuple[str, int]:
    quote, out, j = src[i], [], i + 1
    while j < len(src):
        c = src[j]
        if c == "\\":
            ch, j = _escape(src, j)
            out.append(ch)
            continue
        if c == quote:
            return "".join(out), j + 1
        if c == "\n":
            break
        out.append(c)
        j += 1
    raise ValueError(f"unterminated string at {i}: {src[i:i + 60]!r}")


def _expr_end(src: str, i: int) -> int:
    """From just after a template's `${`, the index of the `}` that closes it."""
    depth, j = 0, i
    while j < len(src):
        c = src[j]
        if c in "'\"":
            j = _read_string(src, j)[1]
            continue
        if c == "`":
            j = _read_template(src, j)[1]
            continue
        if src.startswith("//", j):
            j = src.index("\n", j)
            continue
        if src.startswith("/*", j):
            j = src.index("*/", j) + 2
            continue
        if c == "{":
            depth += 1
        elif c == "}":
            if depth == 0:
                return j
            depth -= 1
        j += 1
    raise ValueError(f"unterminated ${{ at {i}")


def _read_template(src: str, i: int) -> tuple[list[tuple[str, str]], int]:
    """A template literal at src[i] as ("text", s) / ("hole", source) parts."""
    parts: list[tuple[str, str]] = []
    buf: list[str] = []
    j = i + 1
    while j < len(src):
        c = src[j]
        if c == "\\":
            ch, j = _escape(src, j)
            buf.append(ch)
            continue
        if c == "`":
            if buf:
                parts.append(("text", "".join(buf)))
            return parts, j + 1
        if c == "$" and src[j + 1:j + 2] == "{":
            if buf:
                parts.append(("text", "".join(buf)))
                buf = []
            k = _expr_end(src, j + 2)
            parts.append(("hole", src[j + 2:k].strip()))
            j = k + 1
            continue
        buf.append(c)
        j += 1
    raise ValueError(f"unterminated template at {i}")


def tokenize(src: str) -> list[Tok]:
    toks: list[Tok] = []
    i, n = 0, len(src)
    while i < n:
        c = src[i]
        if c.isspace():
            i += 1
        elif src.startswith("//", i):
            k = src.find("\n", i)
            i = n if k < 0 else k
        elif src.startswith("/*", i):
            i = src.index("*/", i) + 2
        elif c in "'\"":
            v, j = _read_string(src, i)
            toks.append(Tok("str", v, i, j))
            i = j
        elif c == "`":
            parts, j = _read_template(src, i)
            toks.append(Tok("tpl", parts, i, j))
            i = j
        elif c == "/" and (not toks or (toks[-1].kind in ("punct", "word")
                                        and toks[-1].val in _REGEX_AFTER)):
            j, in_class = i + 1, False
            while src[j] != "/" or in_class:
                if src[j] == "\\":
                    j += 1
                elif src[j] == "[":
                    in_class = True
                elif src[j] == "]":
                    in_class = False
                elif src[j] == "\n":
                    raise ValueError(f"cannot lex a regex at {i}: {src[i:i + 60]!r}")
                j += 1
            j += 1
            while j < n and src[j].isalpha():
                j += 1
            toks.append(Tok("regex", src[i:j], i, j))
            i = j
        elif m := _IDENT.match(src, i):
            toks.append(Tok("word", m.group(), i, m.end()))
            i = m.end()
        elif (m := _NUM.match(src, i)) and m.group():
            toks.append(Tok("num", m.group(), i, m.end()))
            i = m.end()
        elif m := _PUNCT.match(src, i):
            toks.append(Tok("punct", m.group(), i, m.end()))
            i = m.end()
        else:
            raise ValueError(f"cannot lex {src[i:i + 40]!r} at {i}")
    return toks


_CLOSE = {"(": ")", "[": "]", "{": "}"}
_PREFIX = frozenset({"-", "+", "!", "~", "typeof", "void", "await", "new", "...", "delete"})
_NOT_OPERAND = frozenset({"return", "if", "else", "case", "in", "of", "var", "let", "const",
                          "function", "for", "while", "do", "switch", "throw", "try", "catch",
                          "break", "continue", "default", "instanceof"})


def _match(toks: list[Tok], i: int) -> int:
    opn, close, depth = toks[i].val, _CLOSE[toks[i].val], 0
    for j in range(i, len(toks)):
        if toks[j].kind == "punct":
            if toks[j].val == opn:
                depth += 1
            elif toks[j].val == close:
                depth -= 1
                if depth == 0:
                    return j
    raise ValueError(f"unbalanced {opn!r} at {toks[i].start}")


def _primary(toks: list[Tok], i: int) -> int | None:
    n = len(toks)
    while i < n and toks[i].kind in ("punct", "word") and toks[i].val in _PREFIX:
        i += 1
    if i >= n:
        return None
    t = toks[i]
    if t.kind in ("str", "tpl", "num", "regex") or (t.kind == "word" and t.val not in _NOT_OPERAND):
        i += 1
    elif t.is_("(", "["):
        i = _match(toks, i) + 1
    else:
        return None
    while i < n:                                   # member access, calls, indexing
        t = toks[i]
        if t.is_(".", "?.") and i + 1 < n and toks[i + 1].kind == "word":
            i += 2
        elif t.is_("?.") and i + 1 < n and toks[i + 1].is_("(", "["):
            i = _match(toks, i + 1) + 1
        elif t.is_("(", "["):
            i = _match(toks, i) + 1
        else:
            break
    return i


def _operand(toks: list[Tok], i: int) -> int | None:
    """One operand of a `+` chain: a primary, joined to others only by operators
    that bind tighter than `+` (so `a * b + 'x'` is the operand `a * b`)."""
    j = _primary(toks, i)
    while j is not None and j < len(toks) and toks[j].is_("*", "/", "%", "**"):
        k = _primary(toks, j + 1)
        if k is None:
            break
        j = k
    return j


def string_exprs(src: str, at: int | None = None) -> list[tuple[int, list[tuple[str, str]]]]:
    """Every expression in ``src`` that builds a string: each template literal and
    each `+` chain with a string operand, as ("text", s) / ("hole", source) parts.

    A hole's own source is read again, so a template or chain nested inside a
    `${...}` or a parenthesised ternary is its own entry - which is what lets the
    caller abbreviate it in the outer key without losing its words.

    Each entry comes with its offset in ``src`` (a nested one, with its parent's),
    which is how a hole printing a variable is judged by the assignment before it.
    """
    toks = tokenize(src)
    out: list[tuple[int, list[tuple[str, str]]]] = []
    i, n = 0, len(toks)
    while i < n:
        j = _operand(toks, i)
        if j is None:
            i += 1
            continue
        ops = [(i, j, "+")]
        while j < n and toks[j].is_("+", "-"):
            k = _operand(toks, j + 1)
            if k is None:
                break
            ops.append((j + 1, k, toks[j].val))
            j = k
        literal = [b - a == 1 and toks[a].kind in ("str", "tpl") for a, b, _ in ops]
        if any(literal) and (len(ops) > 1 or toks[i].kind == "tpl"):
            parts: list[tuple[str, str]] = []
            for (a, b, op), lit in zip(ops, literal):
                expr = src[toks[a].start:toks[b - 1].end]
                if op == "-":                      # numeric: belongs to the hole before it
                    if parts and parts[-1][0] == "hole":
                        parts[-1] = ("hole", parts[-1][1] + " - " + expr)
                    else:
                        parts.append(("hole", "-" + expr))
                elif lit and toks[a].kind == "str":
                    parts.append(("text", toks[a].val))
                elif lit:
                    parts.extend(toks[a].val)
                else:
                    parts.append(("hole", expr))
            pos = toks[i].start if at is None else at
            out.append((pos, parts))
            for kind, v in parts:
                if kind == "hole":
                    out.extend(string_exprs(v, pos))
            i = j
            continue
        i += 1
    return out


# ---------------------------------------------------------------------------
# what counts as a figure
# ---------------------------------------------------------------------------
# The page's own number formatters. A call to one of these is a figure this repo
# formatted, the JavaScript counterpart of a numeric format spec in an f-string.
# The date formatters are listed too, as **not** numbers, so a new `fmt*` helper
# fails `test_every_formatter_the_sweep_meets_is_classified` until someone decides
# which it is - a helper missing from both lists would otherwise make every
# sentence it formats invisible.
NUMBER_HELPERS = frozenset({
    "fmt", "fmtWeight", "fmtMetric", "fmtDollar", "fmtPct", "fmtPctChg", "fmtBig", "fmtPE",
    "fmtPct1", "fmtDE", "fmtInput", "fmtRG", "cPct", "cPrice", "cMoney", "cPlain", "cPp", "pc",
    "pct", "sp", "ordinal", "ord", "mkChg", "mkOrd", "mkVal",
})
# a called name of these shapes must be in one of the two lists above
FORMATTER_SHAPE = re.compile(r"^(?:fmt|c[A-Z]|mk[A-Z])")
NOT_NUMBER_HELPERS = frozenset({"fmtD", "fmtDate", "fmtTime", "prettyDate", "cDate"})
_NUMBER_WORDS = frozenset({"toFixed", "toPrecision", "toLocaleString", "length", "Math"})


def _split_ternary(toks: list[Tok]) -> list[list[Tok]] | None:
    """The two branches of a top-level `c ? a : b`, or None. The condition is not
    printed, so `x.length ? ... : ''` is not a figure because of its `.length`."""
    depth, q = 0, None
    for k, t in enumerate(toks):
        if t.is_("(", "[", "{"):
            depth += 1
        elif t.is_(")", "]", "}"):
            depth -= 1
        elif depth == 0 and t.is_("?") and q is None:
            q, nest = k, 0
        elif depth == 0 and q is not None and t.is_("?"):
            nest += 1
        elif depth == 0 and q is not None and t.is_(":"):
            if nest == 0:
                return [toks[q + 1:k], toks[k + 1:]]
            nest -= 1
    return None


def _figure(toks: list[Tok]) -> bool:
    while len(toks) > 2 and toks[0].is_("(") and _match(toks, 0) == len(toks) - 1:
        toks = toks[1:-1]
    branches = _split_ternary(toks)
    if branches is not None:
        return any(_figure(b) for b in branches)
    k = 0
    while k < len(toks):
        t = toks[k]
        nxt = toks[k + 1] if k + 1 < len(toks) else None
        if t.is_("(", "{"):
            # a group is judged on its own, ternary and all - so the condition of a
            # nested `(n === 1 ? 'is' : 'are')` is not a figure either - except the
            # arguments of a date formatter, which print a date
            end = _match(toks, k)
            prev = toks[k - 1] if k else None
            if not (prev is not None and prev.kind == "word" and prev.val in NOT_NUMBER_HELPERS):
                if _figure(toks[k + 1:end]):
                    return True
            k = end + 1
            continue
        if t.is_("["):                 # an index picks a value; it is not printed
            k = _match(toks, k) + 1
            continue
        if t.kind == "word" and t.val in _NUMBER_WORDS:
            return True
        if t.kind == "word" and t.val in NUMBER_HELPERS and nxt is not None and nxt.is_("("):
            return True
        if t.is_("*", "/", "%", "-", "**"):
            return True
        if t.is_("+") and not (_stringy(toks, k - 1, -1) or _stringy(toks, k + 1, +1)):
            return True
        k += 1
    return False


def _stringy(toks: list[Tok], k: int, step: int) -> bool:
    """Whether the operand beside a `+` is a string: a literal, or a bracketed group
    holding one - `(a ? 'x' : '') + (b ? 'y' : '')` is concatenation, not addition."""
    if not 0 <= k < len(toks):
        return False
    t = toks[k]
    if t.kind in ("str", "tpl"):
        return True
    if step > 0 and t.is_("("):
        return any(x.kind in ("str", "tpl") for x in toks[k:_match(toks, k) + 1])
    if step < 0 and t.is_(")"):
        depth = 0
        for j in range(k, -1, -1):
            if toks[j].is_(")"):
                depth += 1
            elif toks[j].is_("("):
                depth -= 1
                if depth == 0:
                    return any(x.kind in ("str", "tpl") for x in toks[j:k + 1])
    return False


def is_figure(expr: str, assigned: dict | None = None, at: int = 0) -> bool:
    """Whether a hole prints a number this repo formatted or computed.

    A hole that prints a bare variable is judged by the **last assignment to that
    name before the sentence** (``assigned``, from `assignments`): a figure if it
    was assigned one (`const covPct = (cov.n / cov.of * 100).toFixed(0)`).
    Otherwise computing a number one line before the sentence that prints it
    would take the sentence out of the sweep. "Last before" rather than "anywhere"
    because the page reuses short names: `t` is a ticker in the holdings toasts and
    a figure in a chart.
    """
    toks = tokenize(expr)
    if assigned and len(toks) == 1 and toks[0].kind == "word" and toks[0].val in assigned:
        before = [fig for pos, fig in assigned[toks[0].val] if pos < at]
        return bool(before) and before[-1]
    return _figure(toks)


_ASSIGN = re.compile(r"\b(?:const|let|var)\s+([A-Za-z_$][\w$]*)\s*=\s*([^;\n]+)")


def assignments(src: str) -> dict[str, list[tuple[int, bool]]]:
    """Every one-line `const/let/var name = ...` in a script: its offset, and
    whether the value is a figure. A multi-line initialiser it cannot lex is
    recorded as not a figure."""
    out: dict[str, list[tuple[int, bool]]] = {}
    for m in _ASSIGN.finditer(src):
        expr = m.group(2).strip()
        try:
            fig = "=>" not in expr[:40] and _figure(tokenize(expr))
        except (ValueError, IndexError, KeyError):
            fig = False
        out.setdefault(m.group(1), []).append((m.start(), fig))
    return out


def is_markup(expr: str) -> bool:
    """Whether a hole inserts markup (a literal containing a tag). Such a hole is
    a block of html, not a word, so it ends the unit; its own text is swept as
    its own entry."""
    for t in tokenize(expr):
        if t.kind == "str" and "<" in t.val:
            return True
        if t.kind == "tpl" and any(k == "text" and "<" in v for k, v in t.val):
            return True
    return False


def key_expr(expr: str) -> str:
    """A hole's source as it appears in a key, with each nested template literal
    shown as `...`. The nested template is swept as its own entry, so its words
    are keyed there; left in place it would make the outer key a block of markup."""
    out, last = [], 0
    for t in tokenize(expr):
        if t.kind == "tpl":
            out.append(expr[last:t.start])
            out.append("`...`")
            last = t.end
    out.append(expr[last:])
    return "".join(out)


def called_names(expr: str) -> set[str]:
    toks = tokenize(expr)
    return {t.val for k, t in enumerate(toks[:-1])
            if t.kind == "word" and toks[k + 1].is_("(")
            and not (k and toks[k - 1].is_(".", "?."))}
