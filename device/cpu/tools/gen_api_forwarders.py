#!/usr/bin/env python3
"""Generate package-level forwarders for embedded family methods."""

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

FAMILIES = [
    ("activation", "Activation", "defaultActivation", ["ops.go", "ops_extra.go", "softmax.go", "gated.go", "param.go"]),
    ("elementwise", "Elementwise", "defaultElementwise", ["ops.go"]),
    ("reduction", "Reduction", "defaultReduction", ["ops.go"]),
    ("dot", "Product", "defaultProduct", ["ops.go"]),
    ("matmul", "Gemm", "defaultGemm", ["ops.go"]),
    ("pool", "Pool", "defaultPool", ["ops.go"]),
    ("convolution", "Convolution", "defaultConvolution", ["ops.go"]),
    ("dropout", "DropoutLayer", "defaultDropoutLayer", ["ops.go"]),
    ("losses", "Losses", "defaultLosses", ["ops.go"]),
    ("sampling", "Sampling", "defaultSampling", ["ops.go"]),
    ("embedding", "Embedding", "defaultEmbedding", ["ops.go"]),
    ("normalization", "Normalization", "defaultNormalization", ["ops.go"]),
    ("layernorm", "Norm", "defaultNorm", ["ops.go"]),
    ("rope", "RotaryEmbedding", "defaultRotaryEmbedding", ["ops.go"]),
    ("hawkes", "Hawkes", "defaultHawkes", ["ops.go"]),
    ("physics", "Physics", "defaultPhysics", ["ops.go"]),
    ("causal", "Causal", "defaultCausal", ["ops.go"]),
    ("masking", "Masking", "defaultMasking", ["ops.go"]),
    ("attention", "Attention", "defaultAttention", ["ops.go"]),
    ("vsa", "VSA", "defaultVSA", ["ops.go"]),
    ("active_inference", "ActiveInference", "defaultActiveInference", ["ops.go"]),
    ("predictive_coding", "PredictiveCoding", "defaultPredictiveCoding", ["ops.go"]),
    ("dequant", "Dequantization", "defaultDequantization", ["ops.go"]),
    ("quant", "Quantization", "defaultQuantization", ["ops.go"]),
    ("pospop", "PosPop", "defaultPosPop", ["dispatch.go"]),
]

METHOD_RE = re.compile(
    r"^func \([^)]+ " + r"(\w+)\) (\w+)\(([\s\S]*?)\)\s*(\([^)]*\)\s*)?\{",
    re.MULTILINE,
)


def extract_methods(path: Path, type_name: str) -> list[tuple[str, str, str, str]]:
    text = path.read_text()
    methods = []
    for match in METHOD_RE.finditer(text):
        if match.group(1) != type_name:
            continue
        name = match.group(2)
        params = match.group(3).strip()
        results = (match.group(4) or "").strip()
        methods.append((name, params, results, ""))
    return methods


def main() -> None:
    for pkg, type_name, var_name, api_files in FAMILIES:
        pkg_dir = ROOT / pkg
        methods: dict[str, tuple[str, str]] = {}
        for name in api_files:
            path = pkg_dir / name
            if not path.exists():
                continue
            for method_name, params, results, _ in extract_methods(path, type_name):
                methods[method_name] = (params, results)

        if not methods:
            continue

        lines = [
            f"package {pkg}",
            "",
            f"var {var_name} = New()",
            "",
        ]
        for method_name in sorted(methods):
            params, results = methods[method_name]
            sig_params = params
            call_args = ", ".join(part.strip().split()[0] for part in params.split(",") if part.strip())
            if results:
                lines.append(f"func {method_name}({sig_params}) {results} {{")
                lines.append(f"\treturn {var_name}.{method_name}({call_args})")
            else:
                lines.append(f"func {method_name}({sig_params}) {{")
                lines.append(f"\t{var_name}.{method_name}({call_args})")
            lines.append("}")
            lines.append("")

        out_path = pkg_dir / "api_forward.go"
        out_path.write_text("\n".join(lines))
        print(f"wrote {out_path.relative_to(ROOT)} ({len(methods)} forwarders)")


if __name__ == "__main__":
    main()
