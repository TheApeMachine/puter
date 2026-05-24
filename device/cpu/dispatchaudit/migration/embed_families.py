#!/usr/bin/env python3
"""Convert CPU family package-level API funcs to struct methods."""

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

FAMILIES = [
    ("activation", "Activation", "activation", ["ops.go", "ops_extra.go", "softmax.go", "gated.go", "param.go"]),
    ("elementwise", "Elementwise", "elementwise", ["ops.go"]),
    ("reduction", "Reduction", "reduction", ["ops.go"]),
    ("dot", "Product", "product", ["ops.go"]),
    ("matmul", "Gemm", "gemm", ["ops.go"]),
    ("pool", "Pool", "pool", ["ops.go"]),
    ("convolution", "Convolution", "convolution", ["ops.go"]),
    ("dropout", "DropoutLayer", "dropoutLayer", ["ops.go"]),
    ("losses", "Losses", "losses", ["ops.go"]),
    ("sampling", "Sampling", "sampling", ["ops.go"]),
    ("embedding", "Embedding", "embedding", ["ops.go"]),
    ("normalization", "Normalization", "normalization", ["ops.go"]),
    ("layernorm", "Norm", "norm", ["ops.go"]),
    ("rope", "RotaryEmbedding", "rotaryEmbedding", ["ops.go"]),
    ("hawkes", "Hawkes", "hawkes", ["ops.go"]),
    ("physics", "Physics", "physics", ["ops.go"]),
    ("causal", "Causal", "causal", ["ops.go"]),
    ("masking", "Masking", "masking", ["ops.go"]),
    ("attention", "Attention", "attention", ["ops.go"]),
    ("vsa", "VSA", "vsa", ["ops.go"]),
    ("active_inference", "ActiveInference", "activeInference", ["ops.go"]),
    ("predictive_coding", "PredictiveCoding", "predictiveCoding", ["ops.go"]),
    ("dequant", "Dequantization", "dequantization", ["ops.go"]),
    ("quant", "Quantization", "quantization", ["ops.go"]),
    ("pospop", "PosPop", "posPop", ["dispatch.go"]),
]

FUNC_RE = re.compile(r"^func ([A-Z]\w*)\(")


def write_type_file(pkg_dir: Path, type_name: str) -> None:
    path = pkg_dir / f"{type_name.lower() if type_name != 'VSA' else 'vsa'}.go"
    if type_name == "Activation":
        path = pkg_dir / "activation.go"
    elif type_name == "Elementwise":
        path = pkg_dir / "elementwise.go"
    elif type_name == "Reduction":
        path = pkg_dir / "reduction.go"
    elif type_name == "Product":
        path = pkg_dir / "dot.go"
    elif type_name == "Gemm":
        path = pkg_dir / "matmul.go"
    elif type_name == "Pool":
        path = pkg_dir / "pool.go"
    elif type_name == "Convolution":
        path = pkg_dir / "convolution.go"
    elif type_name == "DropoutLayer":
        path = pkg_dir / "dropout.go"
    elif type_name == "Losses":
        path = pkg_dir / "losses.go"
    elif type_name == "Sampling":
        path = pkg_dir / "sampling.go"
    elif type_name == "Embedding":
        path = pkg_dir / "embedding.go"
    elif type_name == "Normalization":
        path = pkg_dir / "normalization.go"
    elif type_name == "Norm":
        path = pkg_dir / "layernorm.go"
    elif type_name == "RotaryEmbedding":
        path = pkg_dir / "rope.go"
    elif type_name == "Hawkes":
        path = pkg_dir / "hawkes.go"
    elif type_name == "Physics":
        path = pkg_dir / "physics.go"
    elif type_name == "Causal":
        path = pkg_dir / "causal.go"
    elif type_name == "Masking":
        path = pkg_dir / "masking.go"
    elif type_name == "Attention":
        path = pkg_dir / "attention.go"
    elif type_name == "VSA":
        path = pkg_dir / "vsa.go"
    elif type_name == "ActiveInference":
        path = pkg_dir / "active_inference.go"
    elif type_name == "PredictiveCoding":
        path = pkg_dir / "predictive_coding.go"
    elif type_name == "Dequantization":
        path = pkg_dir / "dequant.go"
    elif type_name == "Quantization":
        path = pkg_dir / "quant.go"
    elif type_name == "PosPop":
        path = pkg_dir / "pospop.go"

    if path.exists():
        return

    pkg = pkg_dir.name
    content = f"""package {pkg}

/*
{type_name} implements device.{type_name if type_name not in {"Product", "Gemm", "Norm", "RotaryEmbedding", "DropoutLayer", "Dequantization", "Quantization", "PosPop"} else {
    "Dot" if type_name == "Product" else
    "Matmul" if type_name == "Gemm" else
    "LayerNorm" if type_name == "Norm" else
    "RoPE" if type_name == "RotaryEmbedding" else
    "Dropout" if type_name == "DropoutLayer" else
    "Dequant" if type_name == "Dequantization" else
    "Quant" if type_name == "Quantization" else
    "PosPop"
}} for the CPU backend.
*/
type {type_name} struct{{}}

/*
New constructs a {type_name} receiver for CPU dispatch.
*/
func New() {type_name} {{
\treturn {type_name}{{}}
}}
"""
    path.write_text(content)


def convert_file(path: Path, type_name: str, receiver: str) -> int:
    text = path.read_text()
    lines = text.splitlines(keepends=True)
    changed = 0
    out = []
    for line in lines:
        match = FUNC_RE.match(line)
        if match and f"func ({receiver} {type_name})" not in line:
            name = match.group(1)
            if name.startswith("Test") or name.startswith("Benchmark"):
                out.append(line)
                continue
            newline = line.replace(f"func {name}(", f"func ({receiver} {type_name}) {name}(")
            out.append(newline)
            changed += 1
        else:
            out.append(line)
    if changed:
        path.write_text("".join(out))
    return changed


def main() -> int:
    total = 0
    for pkg, type_name, receiver, api_files in FAMILIES:
        pkg_dir = ROOT / pkg
        if not pkg_dir.is_dir():
            print(f"missing package: {pkg_dir}", file=sys.stderr)
            continue
        write_type_file(pkg_dir, type_name)
        for name in api_files:
            path = pkg_dir / name
            if not path.exists():
                print(f"skip missing: {path}")
                continue
            count = convert_file(path, type_name, receiver)
            total += count
            print(f"{path.relative_to(ROOT)}: {count} methods")
    print(f"total methods converted: {total}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
