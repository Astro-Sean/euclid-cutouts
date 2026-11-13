"""Terminal output helpers for scripts and CLIs.

Dependency-free status lines, framed headers, tables, boxes, and progress
output with automatic color fallbacks. Status lines use plain bracketed
tags ([ok], [warn], [error], [info]) so output stays readable on dumb
terminals, pipes, and NO_COLOR environments.

    import sys
    sys.path.insert(0, "<optimize_prints skill dir>/scripts")
    import termprint as tp

    tp.header("Reduction pipeline")
    tp.ok("bias frames stacked")
    tp.warn("3 flats saturated; skipped")
    tp.fail("could not read SP001.fits")          # goes to stderr
    tp.table(["file", "exptime", "status"], rows)
    tp.box("QA summary", ["7 files ok", "1 skipped", "0 failed"])

Rules these helpers enforce so output stays scannable:
- One status tag per line, always in the same column: ok/warn/error/info.
- Errors go to stderr; results stay on stdout so pipes work.
- No emoji or Unicode symbols; bracketed ASCII tags are portable.
- Tables cap at six columns; numbers right-align, text left-aligns.
- Headers print a full-width divider above and below the title, with one
  blank line before each opening divider and none before the first.
"""

from __future__ import annotations

import os
import sys
import time

_OK = "[ok]"
_WARN = "[warn]"
_FAIL = "[error]"
_INFO = "[info]"

_CODES = {
    "ok": "32",      # green
    "warn": "33",    # yellow
    "fail": "31",    # red
    "info": "36",    # cyan
    "dim": "2",
    "bold": "1",
}

WIDTH = 70


def supports_color(stream=None):
    """Honor NO_COLOR, dumb terminals, and non-tty output."""
    stream = stream or sys.stdout
    if os.environ.get("NO_COLOR"):
        return False
    term = os.environ.get("TERM", "")
    if term in ("", "dumb"):
        return False
    return hasattr(stream, "isatty") and stream.isatty()


_COLOR = supports_color()
_COLOR_ERR = supports_color(sys.stderr)

# Tracks whether anything has been printed, so headers can emit exactly one
# blank line before their opening divider and none at the start of output.
_emitted = False


def _mark():
    global _emitted
    _emitted = True


def _tag(tag, color, stream):
    colored = (stream is sys.stderr and _COLOR_ERR) or (
        stream is not sys.stderr and _COLOR
    )
    return f"\033[{_CODES[color]}m{tag}\033[0m" if colored else tag


def _line(tag, color, msg, stream):
    if stream is sys.stderr:
        sys.stdout.flush()
    print(f"{_tag(tag, color, stream)} {msg}", file=stream, flush=True)
    _mark()


def ok(msg):
    _line(_OK, "ok", msg, sys.stdout)


def warn(msg):
    _line(_WARN, "warn", msg, sys.stdout)


def info(msg):
    _line(_INFO, "info", msg, sys.stdout)


def fail(msg):
    _line(_FAIL, "fail", msg, sys.stderr)


def die(msg, code=1):
    fail(msg)
    raise SystemExit(code)


def ask(msg, default=False):
    """Yes/no prompt as a [warn] line. Enter takes `default`.

    Non-interactive stdin returns `default` without prompting.
    KeyboardInterrupt propagates so callers keep their exit path.
    """
    suffix = "[Y/n]" if default else "[y/N]"
    if not (hasattr(sys.stdin, "isatty") and sys.stdin.isatty()):
        return default
    try:
        reply = input(f"{_tag(_WARN, 'warn', sys.stdout)} {msg} {suffix} ")
    except EOFError:
        print()
        return default
    yes = reply.strip().lower()
    if not yes:
        return default
    return yes in ("y", "yes")


def gap():
    """One blank line between logical blocks; no-op at start of output."""
    if _emitted:
        print()


def header(title, width=WIDTH, char="="):
    """Framed section banner: full-width rule, title, rule."""
    if _emitted:
        print()
    print(char * width)
    print(title)
    print(char * width)
    _mark()


def subheader(title, width=WIDTH):
    """Lighter phase banner framed with '-' rules; use under a header."""
    header(title, width=width, char="-")


def rule(char="-", width=WIDTH):
    print(char * width)
    _mark()


def kv(key, value, width=24):
    """Aligned key-value pair: 'exptime        : 60 s'."""
    print(f"{key:<{width}} : {value}")
    _mark()


def table(headers, rows, aligns=None, max_cols=6):
    """Print an aligned table. Text left-aligns, numbers right-align.

    headers : list of column names (max max_cols)
    rows    : iterable of row tuples/lists
    aligns  : optional 'l'/'r' per column; default auto-detects numbers
    """
    headers = [str(h) for h in headers[:max_cols]]
    rows = [[str(c) for c in r[:max_cols]] for r in rows]
    if not headers:
        return
    ncols = len(headers)

    def is_num(s):
        try:
            float(s)
            return True
        except (ValueError, TypeError):
            return False

    if aligns is None:
        aligns = [
            "r" if all(is_num(r[i]) for r in rows) and rows else "l"
            for i in range(ncols)
        ]

    widths = [len(headers[i]) for i in range(ncols)]
    for r in rows:
        for i in range(ncols):
            widths[i] = max(widths[i], len(r[i]))

    def fmt_row(cells):
        parts = []
        for i in range(ncols):
            w = widths[i]
            parts.append(
                f"{cells[i]:>{w}}" if aligns[i] == "r" else f"{cells[i]:<{w}}"
            )
        return "  ".join(parts).rstrip()

    print(fmt_row(headers))
    print(fmt_row(["-" * w for w in widths]))
    for r in rows:
        print(fmt_row(r))
    _mark()


def box(title, lines, width=None, stream=None):
    """Multi-check summary box; use for >= 3 results, not one-liners."""
    stream = stream or sys.stdout
    body = [title] + [""] + [str(l) for l in lines]
    w = width or (max(len(l) for l in body) + 4)
    edge = "+" + "-" * (w - 2) + "+"
    print(edge, file=stream)
    for l in body:
        print(f"| {l:<{w - 4}} |", file=stream)
    print(edge, file=stream)
    _mark()


def progress(i, total, label="", every=1, width=30):
    """In-place progress line ending with a newline at completion.

    Throttle-heavy loops print at most 'every' iterations; final call
    always emits. Call with i == total once to close the line.
    """
    if i % every and i != total:
        return
    frac = min(1.0, i / total) if total else 1.0
    # on a pipe there is no carriage return; emit a line only when a
    # quartile boundary is crossed so logs stay readable
    if not (hasattr(sys.stdout, "isatty") and sys.stdout.isatty()):
        prev = max(i - every, 0) / total if total else 0
        if i == total or int(frac * 4) != int(prev * 4):
            print(f"{label} {100 * frac:5.1f}%")
            _mark()
        return
    filled = int(width * frac)
    bar = "#" * filled + "-" * (width - filled)
    print(
        f"\r{label} [{bar}] {100 * frac:5.1f}%",
        end="\n" if i >= total else "",
        flush=True,
    )
    _mark()


class timed:
    """Context manager: 'label... done in 0.42 s' on exit."""

    def __init__(self, label):
        self.label = label

    def __enter__(self):
        print(f"{self.label}... ", end="", flush=True)
        self._t0 = time.monotonic()
        return self

    def __exit__(self, exc_type, *_):
        dt = time.monotonic() - self._t0
        if exc_type is None:
            print(f"done in {dt:.2f} s")
        else:
            print(f"FAILED after {dt:.2f} s")
        _mark()
        return False


__all__ = [
    "ok", "warn", "fail", "info", "die", "ask", "header", "subheader",
    "rule", "gap", "kv", "table", "box", "progress", "timed",
    "supports_color", "WIDTH",
]
