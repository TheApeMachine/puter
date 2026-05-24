// Command gapmatrix emits a family×backend×dtype coverage CSV for puter device backends.
package main

import (
	"encoding/csv"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strings"
)

type dtypeRow struct {
	name string
}

type matrixRow struct {
	family         string
	backend        string
	dtypeName      string
	dispatchStatus string
	kernelStatus   string
	testStatus     string
	notes          string
}

var families = []string{
	"activation", "elementwise", "reduction", "dot", "matmul", "pool",
	"convolution", "dropout", "losses", "sampling", "embedding",
	"normalization", "layernorm", "rope", "hawkes", "physics", "causal",
	"masking", "attention", "vsa", "active_inference", "predictive_coding",
	"dequant", "quant",
}

var dtypes = []string{
	"f64", "f32", "f16", "bf16", "fp8e4m3", "fp8e5m2",
	"i64", "i32", "i16", "i8", "i4",
	"u64", "u32", "u16", "u8", "bool", "c64", "c128",
}

var backends = []string{"cpu", "metal", "cuda", "xla"}

// dispatchMatrix encodes curated dispatch coverage from device dispatch wiring.
// Status: native | lut | generic | panic | silent | stub | unavailable | missing
var dispatchMatrix = map[string]map[string]map[string]string{
	"activation": {
		"cpu": {
			"f32": "native", "f16": "lut", "bf16": "lut",
			"f64": "silent", "fp8e4m3": "silent", "fp8e5m2": "silent",
			"i64": "silent", "i32": "silent", "i16": "silent", "i8": "silent",
			"i4": "silent", "u64": "silent", "u32": "silent", "u16": "silent",
			"u8": "silent", "bool": "silent", "c64": "silent", "c128": "silent",
		},
		"metal": {
			"f32": "native", "f16": "native", "bf16": "native",
			"f64": "unavailable", "fp8e4m3": "unavailable", "fp8e5m2": "unavailable",
			"i64": "unavailable", "i32": "unavailable", "i16": "unavailable", "i8": "unavailable",
			"i4": "unavailable", "u64": "unavailable", "u32": "unavailable", "u16": "unavailable",
			"u8": "unavailable", "bool": "unavailable", "c64": "unavailable", "c128": "unavailable",
		},
		"cuda": {
			"f32": "native", "f16": "native", "bf16": "native",
			"f64": "native", "fp8e4m3": "native", "fp8e5m2": "native",
			"i64": "stub", "i32": "stub", "i16": "stub", "i8": "stub",
			"i4": "stub", "u64": "stub", "u32": "stub", "u16": "stub",
			"u8": "stub", "bool": "stub", "c64": "stub", "c128": "stub",
		},
		"xla": {
			"f64": "lower", "f32": "lower", "f16": "lower", "bf16": "lower",
			"fp8e4m3": "lower", "fp8e5m2": "lower", "i64": "lower", "i32": "lower",
			"i16": "lower", "i8": "lower", "u64": "lower", "u32": "lower",
			"u16": "lower", "u8": "lower", "bool": "lower",
			"i4": "missing", "c64": "missing", "c128": "missing",
		},
	},
	"sampling": {
		"cpu": {
			"f32": "native", "f16": "panic", "bf16": "panic",
		},
		"metal": {"f32": "shader", "f16": "shader", "bf16": "shader"},
		"cuda":  {"f32": "native", "f16": "native", "bf16": "native"},
		"xla":   {"f32": "lower", "f16": "lower", "bf16": "lower"},
	},
	"dropout": {
		"cpu":   {"f32": "native", "f16": "panic", "bf16": "panic"},
		"metal": {"f32": "shader", "f16": "shader", "bf16": "shader"},
		"cuda":  {"f32": "native", "f16": "native", "bf16": "native"},
		"xla":   {"f32": "lower", "f16": "lower", "bf16": "lower"},
	},
	"physics": {
		"cpu":   {"f32": "native", "f16": "panic", "bf16": "panic"},
		"metal": {"f32": "shader"},
		"cuda":  {"f32": "native", "f16": "native", "bf16": "native"},
		"xla":   {"f32": "lower", "f16": "lower", "bf16": "lower"},
	},
	"causal": {
		"cpu":   {"f32": "native"},
		"metal": {"f32": "shader"},
		"cuda":  {"f32": "native", "f16": "native", "bf16": "native"},
		"xla":   {"f32": "lower", "f16": "lower", "bf16": "lower"},
	},
	"hawkes": {
		"cpu":   {"f32": "native"},
		"metal": {"f32": "shader"},
		"cuda":  {"f32": "native", "f16": "native", "bf16": "native"},
		"xla":   {"f32": "lower", "f16": "lower", "bf16": "lower"},
	},
	"vsa": {
		"cpu":   {"f32": "native"},
		"metal": {"f32": "shader"},
		"cuda":  {"f32": "native", "f16": "native", "bf16": "native"},
		"xla":   {"f32": "lower", "f16": "lower", "bf16": "lower"},
	},
	"predictive_coding": {
		"cpu":   {"f32": "native"},
		"metal": {"f32": "shader"},
		"cuda":  {"f32": "native", "f16": "native", "bf16": "native"},
		"xla":   {"f32": "lower", "f16": "lower", "bf16": "lower"},
	},
	"masking": {
		"cpu":   {"f32": "native", "f16": "native", "bf16": "native"},
		"metal": {"f32": "missing", "f16": "missing", "bf16": "missing"},
		"cuda":  {"f32": "native", "f16": "native", "bf16": "native"},
		"xla":   {"f32": "lower", "f16": "lower", "bf16": "lower"},
	},
}

var defaultCoreFloat = map[string]string{
	"f32": "native", "f16": "native", "bf16": "native",
}

var defaultCoreFloatCPU = map[string]string{
	"f32": "native", "f16": "native", "bf16": "native",
	"f64": "panic", "fp8e4m3": "panic", "fp8e5m2": "panic",
}

func main() {
	repoRoot := findRepoRoot()
	testStatus := probeTests(repoRoot)

	outputPath := filepath.Join(repoRoot, "BACKEND_DTYPE_GAP_MATRIX.csv")
	outputFile, err := os.Create(outputPath)

	if err != nil {
		fmt.Fprintf(os.Stderr, "gapmatrix: create output: %v\n", err)
		os.Exit(1)
	}
	defer outputFile.Close()

	writer := csv.NewWriter(outputFile)
	defer writer.Flush()

	header := []string{
		"family", "backend", "dtype",
		"dispatch_status", "kernel_status", "test_status", "notes",
	}

	if err := writer.Write(header); err != nil {
		fmt.Fprintf(os.Stderr, "gapmatrix: write header: %v\n", err)
		os.Exit(1)
	}

	for _, family := range families {
		for _, backend := range backends {
			for _, dtypeName := range dtypes {
				row := buildRow(family, backend, dtypeName, testStatus)
				record := []string{
					row.family, row.backend, row.dtypeName,
					row.dispatchStatus, row.kernelStatus, row.testStatus, row.notes,
				}

				if err := writer.Write(record); err != nil {
					fmt.Fprintf(os.Stderr, "gapmatrix: write row: %v\n", err)
					os.Exit(1)
				}
			}
		}
	}

	fmt.Printf("wrote %s (%d families × %d backends × %d dtypes)\n",
		outputPath, len(families), len(backends), len(dtypes))
}

func buildRow(family, backend, dtypeName string, testStatus map[string]string) matrixRow {
	dispatch := lookupDispatch(family, backend, dtypeName)
	kernel := kernelStatus(family, backend, dtypeName, dispatch)
	testKey := fmt.Sprintf("%s/%s", backendPath(backend, family), family)
	test := testStatus[testKey]

	notes := ""

	if backend == "metal" && family == "masking" {
		notes = "masking ops live under metal/attention; not on metal.Backend"
	}

	if backend == "xla" && (dtypeName == "i4" || dtypeName == "c64" || dtypeName == "c128") {
		notes = "MapDType rejects dtype"
	}

	if dispatch == "missing" && kernel == "missing" {
		test = "n/a"
	}

	return matrixRow{
		family:         family,
		backend:        backend,
		dtypeName:      dtypeName,
		dispatchStatus: dispatch,
		kernelStatus:   kernel,
		testStatus:     test,
		notes:          notes,
	}
}

func lookupDispatch(family, backend, dtypeName string) string {
	if familyMap, ok := dispatchMatrix[family]; ok {
		if backendMap, ok := familyMap[backend]; ok {
			if status, ok := backendMap[dtypeName]; ok {
				return status
			}
		}
	}

	switch backend {
	case "cpu":
		if status, ok := defaultCoreFloatCPU[dtypeName]; ok {
			return status
		}
	case "metal", "cuda":
		if status, ok := defaultCoreFloat[dtypeName]; ok {
			return status
		}
		if backend == "metal" {
			return "unavailable"
		}
		return "stub"
	case "xla":
		if dtypeName == "i4" || dtypeName == "c64" || dtypeName == "c128" {
			return "missing"
		}
		return "lower"
	}

	if dtypeName == "f32" || dtypeName == "f16" || dtypeName == "bf16" {
		return "native"
	}

	return "missing"
}

func kernelStatus(family, backend, dtypeName, dispatch string) string {
	switch dispatch {
	case "native", "lut", "generic", "lower":
		return "implemented"
	case "shader":
		return "shader_only"
	case "stub":
		return "stub"
	case "unavailable":
		return "host_unwired"
	case "silent":
		return "silent_noop"
	case "panic":
		return "panic"
	case "missing":
		return "missing"
	default:
		return dispatch
	}
}

func backendPath(backend, family string) string {
	return fmt.Sprintf("device/%s/%s", backend, family)
}

func probeTests(repoRoot string) map[string]string {
	status := map[string]string{}

	for _, backend := range backends {
		packagePattern := filepath.Join(repoRoot, "device", backend, "...")
		command := exec.Command("go", "test", packagePattern, "-count=1")
		command.Dir = repoRoot
		output, err := command.CombinedOutput()
		text := string(output)

		matches, _ := filepath.Glob(filepath.Join(repoRoot, "device", backend, "*"))
		for _, path := range matches {
			info, statErr := os.Stat(path)

			if statErr != nil || !info.IsDir() {
				continue
			}

			family := filepath.Base(path)
			key := fmt.Sprintf("device/%s/%s", backend, family)
			pkgLine := fmt.Sprintf("github.com/theapemachine/puter/device/%s/%s", backend, family)

			switch {
			case strings.Contains(text, pkgLine+" [no test files]"):
				status[key] = "no_tests"
			case strings.Contains(text, "FAIL\t"+pkgLine):
				status[key] = "fail"
			case strings.Contains(text, "ok  \t"+pkgLine):
				status[key] = "pass"
			case strings.Contains(text, pkgLine+" [build failed]"):
				status[key] = "build_fail"
			default:
				if err != nil {
					status[key] = "unknown"
				} else {
					status[key] = "pass_or_skip"
				}
			}
		}
	}

	return status
}

func findRepoRoot() string {
	if wd, err := os.Getwd(); err == nil {
		candidate := wd

		for {
			if _, err := os.Stat(filepath.Join(candidate, "go.mod")); err == nil {
				if _, err := os.Stat(filepath.Join(candidate, "device", "interface.go")); err == nil {
					return candidate
				}
			}

			parent := filepath.Dir(candidate)

			if parent == candidate {
				break
			}

			candidate = parent
		}
	}

	_, file, _, _ := runtime.Caller(0)
	return filepath.Clean(filepath.Join(filepath.Dir(file), "../.."))
}
