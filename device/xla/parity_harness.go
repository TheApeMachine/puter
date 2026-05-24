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
UploadMatrix copies a row-major matrix into an XLA-resident tensor.
*/
func (parityHarness *ParityHarness) UploadMatrix(
	values []float32,
	rows, cols int,
	format dtype.DType,
) *DeviceTensor {
	bytesIn, err := xlaparity.EncodeVector(values, format)

	if err != nil {
		panic(err)
	}

	shape, err := tensor.NewShape([]int{rows, cols})

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
UploadVolume copies a dense rank-3 tensor into an XLA-resident buffer.
*/
func (parityHarness *ParityHarness) UploadVolume(
	values []float32,
	dimension0, dimension1, dimension2 int,
	format dtype.DType,
) *DeviceTensor {
	bytesIn, err := xlaparity.EncodeVector(values, format)

	if err != nil {
		panic(err)
	}

	shape, err := tensor.NewShape([]int{dimension0, dimension1, dimension2})

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
UploadInt32Vector copies int32 lanes into an XLA-resident tensor.
*/
func (parityHarness *ParityHarness) UploadInt32Vector(values []int32) *DeviceTensor {
	bytesIn := make([]byte, len(values)*4)

	for index, value := range values {
		offset := index * 4
		bytesIn[offset] = byte(value)
		bytesIn[offset+1] = byte(value >> 8)
		bytesIn[offset+2] = byte(value >> 16)
		bytesIn[offset+3] = byte(value >> 24)
	}

	shape, err := tensor.NewShape([]int{len(values)})

	if err != nil {
		panic(err)
	}

	deviceTensor, err := parityHarness.backend.Upload(shape, dtype.Int32, bytesIn)

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
ResidentPointer exposes the unsafe.Pointer token ComputeHost resolves.
*/
func ResidentPointer(deviceTensor *DeviceTensor) unsafe.Pointer {
	return unsafe.Pointer(deviceTensor)
}
