//go:build cuda

package physics

import (
	_ "embed"
	"math"
	"strings"
	"unsafe"

	"github.com/theapemachine/manifesto/dtype"
)

/*
#include "physics_dispatch.h"
*/
import "C"

//go:embed physics.cuh
var physicsHubSource string

//go:embed differential.cu
var differentialDomainSource string

//go:embed spectral.cu
var spectralDomainSource string

func moduleSource() string {
	parts := []string{
		physicsHubSource,
		differentialDomainSource,
		spectralDomainSource,
	}
	return strings.Join(parts, "\n")
}

func registerModuleSource() {
	source := C.CString(moduleSource())
	C.cuda_physics_register_module_source(source)
}

func init() {
	registerModuleSource()
}

func elementDType(format dtype.DType) C.int {
	switch format {
	case dtype.Float32:
		return C.CUDAElementDTypeFloat32
	case dtype.Float16:
		return C.CUDAElementDTypeFloat16
	case dtype.BFloat16:
		return C.CUDAElementDTypeBFloat16
	default:
		return -1
	}
}

func cudaStatusError(status C.CUDAStatus) error {
	if status.code == 0 {
		return nil
	}

	message := C.GoString(&status.message[0])
	return &dispatchError{code: int(status.code), message: message}
}

type dispatchError struct {
	code    int
	message string
}

func (dispatchError *dispatchError) Error() string {
	return dispatchError.message
}

var errUnsupportedDType = &dispatchError{code: -6, message: "unsupported CUDA physics dtype"}

const (
	physicsOpLaplacian4 C.int = 0
	physicsOpGrad1D     C.int = 1
	physicsOpDivergence1D C.int = 2
	physicsOpQuantumPotential C.int = 3
	physicsOpBohmianVelocity C.int = 4
)

func dispatchPhysicsVector(
	operation C.int,
	contextRef C.CUDADeviceRef,
	inputRef C.CUDABufferRef,
	spacingRef C.CUDABufferRef,
	outputRef C.CUDABufferRef,
	format dtype.DType,
	count uint32,
) error {
	elementFormat := elementDType(format)

	if elementFormat < 0 {
		return errUnsupportedDType
	}

	var status C.CUDAStatus
	var code C.int

	switch operation {
	case physicsOpLaplacian4:
		code = C.cuda_dispatch_laplacian4(
			contextRef,
			elementFormat,
			inputRef,
			spacingRef,
			outputRef,
			C.uint32_t(count),
			0,
			&status,
		)
	case physicsOpGrad1D:
		code = C.cuda_dispatch_grad1d(
			contextRef,
			elementFormat,
			inputRef,
			spacingRef,
			outputRef,
			C.uint32_t(count),
			0,
			&status,
		)
	case physicsOpDivergence1D:
		code = C.cuda_dispatch_divergence1d(
			contextRef,
			elementFormat,
			inputRef,
			spacingRef,
			outputRef,
			C.uint32_t(count),
			0,
			&status,
		)
	case physicsOpQuantumPotential:
		code = C.cuda_dispatch_quantum_potential(
			contextRef,
			elementFormat,
			inputRef,
			spacingRef,
			outputRef,
			C.uint32_t(count),
			0,
			&status,
		)
	case physicsOpBohmianVelocity:
		code = C.cuda_dispatch_bohmian_velocity(
			contextRef,
			elementFormat,
			inputRef,
			spacingRef,
			outputRef,
			C.uint32_t(count),
			0,
			&status,
		)
	default:
		return &dispatchError{code: -6, message: "unknown CUDA physics vector operation"}
	}

	if code != 0 {
		return cudaStatusError(status)
	}

	return nil
}

func DispatchLaplacian(
	contextRef C.CUDADeviceRef,
	inputRef C.CUDABufferRef,
	spacingRef C.CUDABufferRef,
	outputRef C.CUDABufferRef,
	format dtype.DType,
	count uint32,
	rank uint32,
	dim0 uint32,
	dim1 uint32,
	dim2 uint32,
) error {
	elementFormat := elementDType(format)

	if elementFormat < 0 {
		return errUnsupportedDType
	}

	var status C.CUDAStatus
	code := C.cuda_dispatch_laplacian(
		contextRef,
		elementFormat,
		inputRef,
		spacingRef,
		outputRef,
		C.uint32_t(count),
		C.uint32_t(rank),
		C.uint32_t(dim0),
		C.uint32_t(dim1),
		C.uint32_t(dim2),
		0,
		&status,
	)

	if code != 0 {
		return cudaStatusError(status)
	}

	return nil
}

func DispatchLaplacian4(
	contextRef C.CUDADeviceRef,
	inputRef C.CUDABufferRef,
	spacingRef C.CUDABufferRef,
	outputRef C.CUDABufferRef,
	format dtype.DType,
	count uint32,
) error {
	return dispatchPhysicsVector(
		physicsOpLaplacian4,
		contextRef,
		inputRef,
		spacingRef,
		outputRef,
		format,
		count,
	)
}

func DispatchGrad1D(
	contextRef C.CUDADeviceRef,
	inputRef C.CUDABufferRef,
	spacingRef C.CUDABufferRef,
	outputRef C.CUDABufferRef,
	format dtype.DType,
	count uint32,
) error {
	return dispatchPhysicsVector(
		physicsOpGrad1D,
		contextRef,
		inputRef,
		spacingRef,
		outputRef,
		format,
		count,
	)
}

func DispatchDivergence1D(
	contextRef C.CUDADeviceRef,
	inputRef C.CUDABufferRef,
	spacingRef C.CUDABufferRef,
	outputRef C.CUDABufferRef,
	format dtype.DType,
	count uint32,
) error {
	return dispatchPhysicsVector(
		physicsOpDivergence1D,
		contextRef,
		inputRef,
		spacingRef,
		outputRef,
		format,
		count,
	)
}

func DispatchQuantumPotential(
	contextRef C.CUDADeviceRef,
	densityRef C.CUDABufferRef,
	spacingRef C.CUDABufferRef,
	outputRef C.CUDABufferRef,
	format dtype.DType,
	count uint32,
) error {
	return dispatchPhysicsVector(
		physicsOpQuantumPotential,
		contextRef,
		densityRef,
		spacingRef,
		outputRef,
		format,
		count,
	)
}

func DispatchBohmianVelocity(
	contextRef C.CUDADeviceRef,
	phaseRef C.CUDABufferRef,
	spacingRef C.CUDABufferRef,
	outputRef C.CUDABufferRef,
	format dtype.DType,
	count uint32,
) error {
	return dispatchPhysicsVector(
		physicsOpBohmianVelocity,
		contextRef,
		phaseRef,
		spacingRef,
		outputRef,
		format,
		count,
	)
}

func DispatchMadelungContinuity(
	contextRef C.CUDADeviceRef,
	densityRef C.CUDABufferRef,
	velocityRef C.CUDABufferRef,
	spacingRef C.CUDABufferRef,
	outputRef C.CUDABufferRef,
	format dtype.DType,
	count uint32,
) error {
	elementFormat := elementDType(format)

	if elementFormat < 0 {
		return errUnsupportedDType
	}

	var status C.CUDAStatus
	code := C.cuda_dispatch_madelung_continuity(
		contextRef,
		elementFormat,
		densityRef,
		velocityRef,
		spacingRef,
		outputRef,
		C.uint32_t(count),
		0,
		&status,
	)

	if code != 0 {
		return cudaStatusError(status)
	}

	return nil
}

func IsPowerOfTwo(value uint32) bool {
	return value > 0 && (value&(value-1)) == 0
}

func naiveFFTTwiddles(count uint32, inverse bool) (real []float32, imag []float32) {
	real = make([]float32, count*count)
	imag = make([]float32, count*count)
	sign := -1.0

	if inverse {
		sign = 1.0
	}

	for index := uint32(0); index < count; index++ {
		for source := uint32(0); source < count; source++ {
			angle := sign * 2.0 * math.Pi * float64(index) * float64(source) / float64(count)
			offset := index*count + source
			real[offset] = float32(math.Cos(angle))
			imag[offset] = float32(math.Sin(angle))
		}
	}

	return real, imag
}

func DispatchFFT1D(
	contextRef C.CUDADeviceRef,
	realInRef C.CUDABufferRef,
	imagInRef C.CUDABufferRef,
	realOutRef C.CUDABufferRef,
	imagOutRef C.CUDABufferRef,
	twiddleRealRef C.CUDABufferRef,
	twiddleImagRef C.CUDABufferRef,
	format dtype.DType,
	count uint32,
) error {
	return dispatchFFT(contextRef, realInRef, imagInRef, realOutRef, imagOutRef, twiddleRealRef, twiddleImagRef, format, count, false)
}

func DispatchIFFT1D(
	contextRef C.CUDADeviceRef,
	realInRef C.CUDABufferRef,
	imagInRef C.CUDABufferRef,
	realOutRef C.CUDABufferRef,
	imagOutRef C.CUDABufferRef,
	twiddleRealRef C.CUDABufferRef,
	twiddleImagRef C.CUDABufferRef,
	format dtype.DType,
	count uint32,
) error {
	return dispatchFFT(contextRef, realInRef, imagInRef, realOutRef, imagOutRef, twiddleRealRef, twiddleImagRef, format, count, true)
}

func dispatchFFT(
	contextRef C.CUDADeviceRef,
	realInRef C.CUDABufferRef,
	imagInRef C.CUDABufferRef,
	realOutRef C.CUDABufferRef,
	imagOutRef C.CUDABufferRef,
	twiddleRealRef C.CUDABufferRef,
	twiddleImagRef C.CUDABufferRef,
	format dtype.DType,
	count uint32,
	inverse bool,
) error {
	elementFormat := elementDType(format)

	if elementFormat < 0 {
		return errUnsupportedDType
	}

	inverseFlag := C.int(0)

	if inverse {
		inverseFlag = 1
	}

	var status C.CUDAStatus
	code := C.cuda_dispatch_fft1d(
		contextRef,
		elementFormat,
		realInRef,
		imagInRef,
		realOutRef,
		imagOutRef,
		twiddleRealRef,
		twiddleImagRef,
		C.uint32_t(count),
		inverseFlag,
		0,
		&status,
	)

	if code != 0 {
		return cudaStatusError(status)
	}

	return nil
}

func TwiddleHostBytes(count uint32, inverse bool) ([]byte, []byte) {
	realValues, imagValues := naiveFFTTwiddles(count, inverse)
	realBytes := unsafe.Slice((*byte)(unsafe.Pointer(&realValues[0])), len(realValues)*4)
	imagBytes := unsafe.Slice((*byte)(unsafe.Pointer(&imagValues[0])), len(imagValues)*4)
	return realBytes, imagBytes
}
