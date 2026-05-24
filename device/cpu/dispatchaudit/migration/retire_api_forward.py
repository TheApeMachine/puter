#!/usr/bin/env python3
"""Retire api_forward.go: export Default receiver and rewrite intra-package callers."""

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

TYPE_FILES = {
    "activation": "activation.go",
    "elementwise": "elementwise.go",
    "reduction": "reduction.go",
    "dot": "dot.go",
    "matmul": "matmul.go",
    "pool": "pool.go",
    "convolution": "convolution.go",
    "dropout": "dropout.go",
    "losses": "losses.go",
    "sampling": "sampling.go",
    "embedding": "embedding.go",
    "normalization": "normalization.go",
    "layernorm": "layernorm.go",
    "rope": "rope.go",
    "hawkes": "hawkes.go",
    "physics": "physics.go",
    "causal": "receiver.go",
    "masking": "masking.go",
    "attention": "receiver.go",
    "vsa": "receiver.go",
    "active_inference": "receiver.go",
    "predictive_coding": "receiver.go",
    "dequant": "dequant.go",
    "quant": "quant.go",
    "pospop": "pospop.go",
}

FORWARD_RE = re.compile(r"^func ([A-Z]\w*)\(", re.MULTILINE)
METHOD_RE = re.compile(r"^func \([^)]+ \w+\) ([A-Z]\w*)\(", re.MULTILINE)


def ensure_default(type_path: Path) -> None:
    text = type_path.read_text()
    if "var Default = New()" in text:
        return
    if "func New()" not in text:
        raise RuntimeError(f"New() missing in {type_path}")
    text = text.rstrip() + "\n\nvar Default = New()\n"
    type_path.write_text(text)


def collect_methods(api_forward: Path) -> list[str]:
    text = api_forward.read_text()
    return FORWARD_RE.findall(text)


def collect_receiver_methods(pkg_dir: Path) -> set[str]:
    names: set[str] = set()
    for path in pkg_dir.glob("*.go"):
        if path.name == "api_forward.go":
            continue
        names.update(METHOD_RE.findall(path.read_text()))
    return names


def rewrite_calls(path: Path, methods: list[str]) -> bool:
    text = path.read_text()
    original = text
    for method in sorted(methods, key=len, reverse=True):
        text = re.sub(
            rf"(?<![.\w]){re.escape(method)}\(",
            f"Default.{method}(",
            text,
        )
    if text != original:
        path.write_text(text)
        return True
    return False


def main() -> None:
    for pkg, type_file in TYPE_FILES.items():
        pkg_dir = ROOT / pkg
        api_forward = pkg_dir / "api_forward.go"
        if not api_forward.exists():
            continue

        methods = collect_methods(api_forward)
        receiver_methods = collect_receiver_methods(pkg_dir)
        forward_only = [m for m in methods if m in receiver_methods]
        type_path = pkg_dir / type_file
        ensure_default(type_path)

        changed_files = []
        for path in sorted(pkg_dir.glob("*.go")):
            if path.name in {"api_forward.go", "ops.go"}:
                continue
            if rewrite_calls(path, forward_only):
                changed_files.append(path.name)

        api_forward.unlink()
        print(
            f"{pkg}: removed api_forward.go, "
            f"rewrote {len(changed_files)} files ({', '.join(changed_files) or 'none'})"
        )


if __name__ == "__main__":
    main()
