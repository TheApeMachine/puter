//go:build xla

package xla

import (
	"testing"
	"unsafe"

	"github.com/theapemachine/manifesto/dtype"
	"github.com/theapemachine/manifesto/tensor"
	xlaparity "github.com/theapemachine/puter/device/xla/internal/parity"
)

/*
ParityHarness runs XLA programs against CPU production references through the real backend.
*/
type ParityHarness struct {
	backend *Backend
}

/*
NewParityHarness opens the XLA backend or skips the test when unavailable.
*/
func NewParityHarness(testingTB testing.TB) *ParityHarness {
	testingTB.Helper()

	backend, err := NewBackend()

	if err != nil {
		testingTB.Skipf("XLA backend unavailable: %v", err)
	}

	return &ParityHarness{backend: backend}
}

/*
Close releases the XLA backend.
*/
func (parityHarness *ParityHarness) Close() {
	if parityHarness == nil || parityHarness.backend == nil {
		return
	}

	_ = parityHarness.backend.Close()
	parityHarness.backend = nil
}

/*
Backend exposes the opened XLA backend for dispatch calls.
*/
func (parityHarness *ParityHarness) Backend() *Backend {
	return parityHarness.backend
}

/*
UploadVector copies host values into an XLA-resident tensor.
*/
func (parityHarness *ParityHarness) UploadVector(values []float32, format dtype.DType) *DeviceTensor {
	bytesIn, err := xlaparity.EncodeVector(values, format)

	if err != nil {
		panic(err)
	}

	shape, err := tensor.NewShape([]int{len(values)})

	if err != nil {
		panic(err)
	}

	deviceTensor, err := parityHarness.backend.Upload(shape, format, bytesIn)

	if err != nil {
		panic(err)
	}

	residentTensor, ok := deviceTensor.(*DeviceTensor)

	if !ok {
		panic("xla parity: upload did not return DeviceTensor")
	}

	return residentTensor
}

/*
DownloadBytes reads an XLA-resident tensor back to raw storage bytes.
*/
func (parityHarness *ParityHarness) DownloadBytes(deviceTensor *DeviceTensor) []byte {
	_, bytesOut, err := parityHarness.backend.Download(deviceTensor)

	if err != nil {
		panic(err)
	}

	return bytesOut
}

/*
DownloadFloat32 reads an XLA-resident tensor back as float32 lanes.
*/
func (parityHarness *ParityHarness) DownloadFloat32(deviceTensor *DeviceTensor, format dtype.DType) []float32 {
	return xlaparity.DecodeFloat32Vector(parityHarness.DownloadBytes(deviceTensor), format)
}

/*
ResidentPointer exposes the unsafe.Pointer token ComputeHost resolves.
*/
func ResidentPointer(deviceTensor *DeviceTensor) unsafe.Pointer {
	return unsafe.Pointer(deviceTensor)
}
