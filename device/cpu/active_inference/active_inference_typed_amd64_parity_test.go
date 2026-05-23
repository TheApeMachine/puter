//go:build amd64

package active_inference

import (
	"testing"

	"github.com/theapemachine/manifesto/dtype"
	"github.com/theapemachine/puter/device/cpu/parity"
	"golang.org/x/sys/cpu"
)

func avx512ActiveInferenceBF16Available() bool {
	return cpu.X86.HasAVX512F
}

func TestBeliefUpdateBF16AVX512Parity(t *testing.T) {
	if !avx512ActiveInferenceBF16Available() {
		t.Skip("AVX-512F required")
	}

	for _, length := range parity.Lengths {
		likelihood := make([]dtype.BF16, length)
		prior := make([]dtype.BF16, length)
		want := make([]dtype.BF16, length)
		got := make([]dtype.BF16, length)
		for index := range likelihood {
			likelihood[index] = dtype.NewBfloat16FromFloat32(float32(index%17+1) * 0.1)
			prior[index] = dtype.NewBfloat16FromFloat32(float32(index%11+1) * 0.08)
		}
		BeliefUpdateBFloat16Scalar(likelihood, prior, want)
		BeliefUpdateBF16AVX512(likelihood, prior, got)
		for index := range got {
			if got[index] != want[index] {
				t.Fatalf("N=%d i=%d got=%v want=%v", length, index, got[index], want[index])
			}
		}
	}
}

func TestBeliefUpdateBF16AVX2Parity(t *testing.T) {
	if !avx2ActiveInferenceAvailable() {
		t.Skip("AVX2+FMA required")
	}

	for _, length := range parity.Lengths {
		likelihood := make([]dtype.BF16, length)
		prior := make([]dtype.BF16, length)
		want := make([]dtype.BF16, length)
		got := make([]dtype.BF16, length)
		for index := range likelihood {
			likelihood[index] = dtype.NewBfloat16FromFloat32(float32(index%17+1) * 0.1)
			prior[index] = dtype.NewBfloat16FromFloat32(float32(index%11+1) * 0.08)
		}
		BeliefUpdateBFloat16Scalar(likelihood, prior, want)
		BeliefUpdateBF16AVX2(likelihood, prior, got)
		for index := range got {
			if got[index] != want[index] {
				t.Fatalf("N=%d i=%d got=%v want=%v", length, index, got[index], want[index])
			}
		}
	}
}

func TestBeliefUpdateBF16SSE2Parity(t *testing.T) {
	if !sse2ActiveInferenceAvailable() {
		t.Skip("SSE2 required")
	}

	for _, length := range parity.Lengths {
		likelihood := make([]dtype.BF16, length)
		prior := make([]dtype.BF16, length)
		want := make([]dtype.BF16, length)
		got := make([]dtype.BF16, length)
		for index := range likelihood {
			likelihood[index] = dtype.NewBfloat16FromFloat32(float32(index%17+1) * 0.1)
			prior[index] = dtype.NewBfloat16FromFloat32(float32(index%11+1) * 0.08)
		}
		BeliefUpdateBFloat16Scalar(likelihood, prior, want)
		BeliefUpdateBF16SSE2(likelihood, prior, got)
		for index := range got {
			if got[index] != want[index] {
				t.Fatalf("N=%d i=%d got=%v want=%v", length, index, got[index], want[index])
			}
		}
	}
}

func TestPrecisionWeightBF16AVX512Parity(t *testing.T) {
	if !avx512ActiveInferenceBF16Available() {
		t.Skip("AVX-512F required")
	}

	for _, length := range parity.Lengths {
		errorsVec := make([]dtype.BF16, length)
		precision := make([]dtype.BF16, length)
		want := make([]dtype.BF16, length)
		got := make([]dtype.BF16, length)
		for index := range errorsVec {
			errorsVec[index] = dtype.NewBfloat16FromFloat32(float32(index%7+1) * 0.2)
			precision[index] = dtype.NewBfloat16FromFloat32(float32(index%5+1) * 0.3)
		}
		PrecisionWeightBFloat16Scalar(errorsVec, precision, want)
		PrecisionWeightBF16AVX512(errorsVec, precision, got)
		for index := range got {
			if got[index] != want[index] {
				t.Fatalf("N=%d i=%d got=%v want=%v", length, index, got[index], want[index])
			}
		}
	}
}

func TestPrecisionWeightBF16AVX2Parity(t *testing.T) {
	if !avx2ActiveInferenceAvailable() {
		t.Skip("AVX2+FMA required")
	}

	for _, length := range parity.Lengths {
		errorsVec := make([]dtype.BF16, length)
		precision := make([]dtype.BF16, length)
		want := make([]dtype.BF16, length)
		got := make([]dtype.BF16, length)
		for index := range errorsVec {
			errorsVec[index] = dtype.NewBfloat16FromFloat32(float32(index%7+1) * 0.2)
			precision[index] = dtype.NewBfloat16FromFloat32(float32(index%5+1) * 0.3)
		}
		PrecisionWeightBFloat16Scalar(errorsVec, precision, want)
		PrecisionWeightBF16AVX2(errorsVec, precision, got)
		for index := range got {
			if got[index] != want[index] {
				t.Fatalf("N=%d i=%d got=%v want=%v", length, index, got[index], want[index])
			}
		}
	}
}

func TestPrecisionWeightBF16SSE2Parity(t *testing.T) {
	if !sse2ActiveInferenceAvailable() {
		t.Skip("SSE2 required")
	}

	for _, length := range parity.Lengths {
		errorsVec := make([]dtype.BF16, length)
		precision := make([]dtype.BF16, length)
		want := make([]dtype.BF16, length)
		got := make([]dtype.BF16, length)
		for index := range errorsVec {
			errorsVec[index] = dtype.NewBfloat16FromFloat32(float32(index%7+1) * 0.2)
			precision[index] = dtype.NewBfloat16FromFloat32(float32(index%5+1) * 0.3)
		}
		PrecisionWeightBFloat16Scalar(errorsVec, precision, want)
		PrecisionWeightBF16SSE2(errorsVec, precision, got)
		for index := range got {
			if got[index] != want[index] {
				t.Fatalf("N=%d i=%d got=%v want=%v", length, index, got[index], want[index])
			}
		}
	}
}
