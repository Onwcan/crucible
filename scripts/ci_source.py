#!/usr/bin/env python3
"""Read-only, standard-library checks for the files tracked by Git.

Run from any directory with ``python scripts/ci_source.py``. Python files are
compiled in memory; no dependencies are imported and no bytecode is written.
Markdown checks cover inline/image links and reference definitions, excluding
code examples. They validate local files, not remote URLs or heading anchors.
"""
from __future__ import annotations

import argparse
import json
import os
import posixpath
import re
import subprocess
import sys
from pathlib import Path
from urllib.parse import unquote, urlsplit


MAX_TRACKED_BYTES = 10 * 1024 * 1024
ARTIFACT_DIRECTORIES = {
    "__pycache__", ".venv", "venv", "node_modules", "baseline-src",
    "baseline-source", "checkpoints", "model-exports", "raw-benchmarks",
    "benchmark-output", "profiler-output", "target",
}
ARTIFACT_SUFFIXES = {
    ".pyc", ".pyo", ".pt", ".pth", ".ckpt", ".safetensors", ".gguf",
    ".ggml", ".onnx", ".bin", ".nsys-rep", ".qdrep", ".qdstrm",
    ".ncu-rep", ".nvvp", ".cubin", ".fatbin", ".exe", ".dll", ".so",
    ".dylib", ".a", ".o",
}
SOURCE_SUFFIXES = {
    ".py", ".sh", ".bash", ".ps1", ".rs", ".cu", ".c", ".h", ".cpp",
    ".toml", ".yml", ".yaml", ".ini", ".cfg",
}
# Literal user-home paths in executable/configuration files are not portable.
# Documentation and evidence deliberately retain historical measurement context.
# "user" is the existing model-ID rejection test's generic placeholder.
HOME_ACCOUNT = r"(?!user(?=[/\\]|$|[\s'\"]))[A-Za-z0-9_.-]+(?=[/\\]|$|[\s'\"])"
MACHINE_PATH = re.compile(
    r"(?:/home/|/mnt/[a-z]/Users/|[A-Za-z]:(?:\\{1,2}|/)Users(?:\\{1,2}|/))"
    + HOME_ACCOUNT
)
CONFLICT_EDGE = re.compile(r"^\s*(?:<{7}|>{7}|\|{7})(?:\s.*)?$")


def git(root: Path, *args: str) -> bytes:
    return subprocess.run(
        ["git", "-C", str(root), *args], check=True, capture_output=True,
        timeout=30,
    ).stdout


def tracked_files(root: Path) -> list[str]:
    # NUL separation and argv avoid shell quoting or newline-in-filename bugs.
    return [os.fsdecode(name) for name in git(root, "ls-files", "-z").split(b"\0") if name]


def reject_constant(value: str) -> None:
    raise ValueError(f"non-JSON constant {value}")


def unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate object key {key!r}")
        result[key] = value
    return result


def prose_only(source: str) -> str:
    """Blank fenced/indented/inline code while retaining line numbers."""
    lines = []
    fence = ""
    for line in source.splitlines(keepends=True):
        marker = re.match(r"^ {0,3}(`{3,}|~{3,})", line)
        if fence:
            if re.match(r"^ {0,3}" + re.escape(fence[0]) + "{" + str(len(fence)) + r",}\s*$", line):
                fence = ""
            lines.append("\n" if line.endswith("\n") else "")
        elif marker:
            fence = marker[1]
            lines.append("\n" if line.endswith("\n") else "")
        elif line.startswith(("    ", "\t")):
            lines.append("\n" if line.endswith("\n") else "")
        else:
            lines.append(line)
    text = "".join(lines)
    return re.sub(r"(`+)(.+?)\1(?!`)", lambda m: "\n" * m[0].count("\n"), text, flags=re.DOTALL)


def destination(text: str, start: int) -> str:
    """Read a Markdown destination, including escaped/balanced parentheses."""
    index = start
    while index < len(text) and text[index].isspace():
        index += 1
    if index < len(text) and text[index] == "<":
        end = text.find(">", index + 1)
        return text[index + 1:end] if end >= 0 else ""
    result = []
    depth = 0
    while index < len(text):
        char = text[index]
        if char == "\\" and index + 1 < len(text):
            index += 1
            result.append(text[index])
        elif char == "(":
            depth += 1
            result.append(char)
        elif char == ")":
            if not depth:
                break
            depth -= 1
            result.append(char)
        elif char.isspace() and not depth:
            break
        else:
            result.append(char)
        index += 1
    return "".join(result)


def markdown_links(source: str) -> list[tuple[int, str]]:
    text = prose_only(source)
    starts = [match.end() for match in re.finditer(r"(?<!\\)\]\(", text)]
    starts += [match.end() for match in re.finditer(r"(?m)^ {0,3}\[[^\]\n]+\]:[ \t]*", text)]
    return [(text.count("\n", 0, start) + 1, destination(text, start)) for start in sorted(starts)]


def check_link(root: Path, name: str, target: str, tracked: set[str]) -> str | None:
    try:
        parsed = urlsplit(target)
    except ValueError:
        return "malformed link destination"
    if not target or parsed.scheme or parsed.netloc or not parsed.path:
        return None
    path = unquote(parsed.path)
    if path.startswith("/"):
        return "absolute local file link is not portable"
    relative = posixpath.normpath(posixpath.join(posixpath.dirname(name), path))
    if relative == ".." or relative.startswith("../"):
        return "local link escapes the repository"
    if relative not in tracked:
        return "local link does not name a tracked file"
    try:
        resolved = (root / relative).resolve()
    except (OSError, ValueError, RuntimeError):
        return "local link target is invalid or cannot be resolved"
    if not resolved.is_relative_to(root.resolve()) or not resolved.is_file():
        return "local link target is missing or leaves the repository"
    return None


def check_repository(root: Path, *, check_clean: bool = False) -> tuple[list[str], dict[str, int]]:
    names = tracked_files(root)
    tracked = set(names)
    errors: list[str] = []
    counts = {"tracked": len(names), "python": 0, "json": 0, "markdown": 0, "local_links": 0}
    for name in names:
        path = root / name
        label = repr(name)
        parts = [part.lower() for part in name.split("/")]
        suffix = path.suffix.lower()
        if set(parts[:-1]) & ARTIFACT_DIRECTORIES or suffix in ARTIFACT_SUFFIXES:
            errors.append(f"{label}: tracked generated/model/profiler artifact")
        if not path.is_file():
            errors.append(f"{label}: tracked file is missing or is not a regular file")
            continue
        if not path.resolve().is_relative_to(root.resolve()):
            errors.append(f"{label}: tracked symlink leaves the repository")
            continue
        if path.stat().st_size > MAX_TRACKED_BYTES:
            errors.append(f"{label}: tracked file exceeds the 10 MiB artifact limit")
            continue
        data = path.read_bytes()
        if suffix == ".py":
            counts["python"] += 1
            try:
                compile(data, name, "exec", dont_inherit=True)
            except (SyntaxError, ValueError, UnicodeError) as error:
                errors.append(f"{label}: Python syntax: {error}")
        if suffix == ".json":
            counts["json"] += 1
            try:
                json.loads(data, parse_constant=reject_constant, object_pairs_hook=unique_object)
            except (ValueError, UnicodeError) as error:
                errors.append(f"{label}: JSON: {error}")
        try:
            source = data.decode("utf-8")
        except UnicodeError:
            continue  # Images and other permitted binary assets have no text checks.
        if "\0" in source:
            continue
        for number, line in enumerate(source.splitlines(), 1):
            if CONFLICT_EDGE.fullmatch(line) or (suffix != ".md" and line.strip() == "======="):
                errors.append(f"{label}:{number}: unresolved merge-conflict marker")
            # A standalone ======= can be a legitimate Markdown heading underline.
            # Elsewhere an orphan separator is also a likely unresolved conflict.
            if suffix in SOURCE_SUFFIXES and MACHINE_PATH.search(line):
                errors.append(f"{label}:{number}: literal machine-specific user-home path")
        if suffix == ".md":
            counts["markdown"] += 1
            for number, target in markdown_links(source):
                problem = check_link(root, name, target, tracked)
                if problem:
                    errors.append(f"{label}:{number}: {problem}: {target!r}")
                elif target and not urlsplit(target).scheme and not urlsplit(target).netloc and urlsplit(target).path:
                    counts["local_links"] += 1
    if check_clean and git(root, "status", "--porcelain=v1", "-z", "--untracked-files=all"):
        errors.append("working tree/index is not clean (git status --short --untracked-files=all)")
    return errors, counts


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--check-clean", action="store_true", help="also reject index/worktree changes and untracked files")
    args = parser.parse_args()
    try:
        errors, counts = check_repository(args.root, check_clean=args.check_clean)
    except (OSError, subprocess.SubprocessError) as error:
        print(f"source checks could not run: {error}", file=sys.stderr)
        return 1
    for error in errors:
        print(f"ERROR: {error}", file=sys.stderr)
    print("Source checks: " + ", ".join(f"{key}={value}" for key, value in counts.items()))
    print(f"{'FAIL' if errors else 'PASS'}: {len(errors)} error(s); no files written")
    return int(bool(errors))


if __name__ == "__main__":
    raise SystemExit(main())
