//go:build arm64

package active_inference

import (
	"testing"
	"unsafe"

	"github.com/theapemachine/manifesto/dtype"
)

func TestTypedFEDebugCompare(t *testing.T) {
	likelihood := []dtype.BF16{dtype.NewBfloat16FromFloat32(0.05)}
	posterior := []dtype.BF16{dtype.NewBfloat16FromFloat32(0.07)}
	prior := []dtype.BF16{dtype.NewBfloat16FromFloat32(0.06)}

	scalar := FreeEnergyBFloat16Scalar(likelihood, posterior, prior)
	goF64 := freeEnergyBFloat16F64NEON(likelihood, posterior, prior)
	bridge := dtype.BF16(freeEnergyBFloat16NEONBridge(
		uintptr(unsafe.Pointer(&likelihood[0])),
		uintptr(unsafe.Pointer(&posterior[0])),
		uintptr(unsafe.Pointer(&prior[0])),
		1,
	))
	neon := FreeEnergyBF16NEON(likelihood, posterior, prior)

	t.Logf("scalar=%v goF64=%v bridge=%v neon=%v", scalar, goF64, bridge, neon)
}
