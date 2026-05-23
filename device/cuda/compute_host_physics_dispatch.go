//go:build cuda

package cuda

import (
	"unsafe"

	"github.com/theapemachine/manifesto/dtype"
	"github.com/theapemachine/puter/device/cuda/physics"
)

func laplacianLaunchShape(dims []int) (count uint32, rank uint32, dim0 uint32, dim1 uint32, dim2 uint32) {
	rank = uint32(len(dims))

	if rank == 0 {
		return 0, 0, 1, 1, 1
	}

	dim0 = uint32(dims[0])
	dim1 = 1
	dim2 = 1
	count = dim0

	if rank >= 2 {
		dim1 = uint32(dims[1])
		count *= dim1
	}

	if rank >= 3 {
		dim2 = uint32(dims[2])
		count *= dim2
	}

	return count, rank, dim0, dim1, dim2
}

func (host *ComputeHost) physicsSpacingBuffer(spacing float32, format dtype.DType) C.CUDABufferRef {
	if host.bridge == nil {
		return nil
	}

	return host.bridge.uploadHostBytes(physicsSpacingBytes(spacing, format))
}

func (host *ComputeHost) dispatchPhysicsVector(
	input unsafe.Pointer,
	output unsafe.Pointer,
	count int,
	spacing float32,
	format dtype.DType,
	dispatch func(
		contextRef C.CUDADeviceRef,
		inputRef C.CUDABufferRef,
		spacingRef C.CUDABufferRef,
		outputRef C.CUDABufferRef,
		format dtype.DType,
		elementCount uint32,
	) error,
) {
	if count == 0 || host.bridge == nil {
		return
	}

	spacingBuffer := host.physicsSpacingBuffer(spacing, format)

	defer host.bridge.releaseScratch(spacingBuffer)

	if err := dispatch(
		host.contextRef(),
		resolveBufferRef(input),
		spacingBuffer,
		resolveBufferRef(output),
		format,
		uint32(count),
	); err != nil {
		host.dispatchError(err)
	}
}

func (host *ComputeHost) DispatchBohmianVelocity(phase, output unsafe.Pointer, count int, spacing float32, format dtype.DType) {
	host.dispatchPhysicsVector(phase, output, count, spacing, format, physics.DispatchBohmianVelocity)
}

func (host *ComputeHost) DispatchDivergence1D(input, output unsafe.Pointer, count int, spacing float32, format dtype.DType) {
	host.dispatchPhysicsVector(input, output, count, spacing, format, physics.DispatchDivergence1D)
}

func (host *ComputeHost) DispatchGrad1D(input, output unsafe.Pointer, count int, spacing float32, format dtype.DType) {
	host.dispatchPhysicsVector(input, output, count, spacing, format, physics.DispatchGrad1D)
}

func (host *ComputeHost) DispatchLaplacian(input, output unsafe.Pointer, dims []int, spacing float32, format dtype.DType) {
	count, rank, dim0, dim1, dim2 := laplacianLaunchShape(dims)

	if count == 0 || host.bridge == nil {
		return
	}

	spacingBuffer := host.physicsSpacingBuffer(spacing, format)

	defer host.bridge.releaseScratch(spacingBuffer)

	if err := physics.DispatchLaplacian(
		host.contextRef(),
		resolveBufferRef(input),
		spacingBuffer,
		resolveBufferRef(output),
		format,
		count,
		rank,
		dim0,
		dim1,
		dim2,
	); err != nil {
		host.dispatchError(err)
	}
}

func (host *ComputeHost) DispatchLaplacian4(input, output unsafe.Pointer, count int, spacing float32, format dtype.DType) {
	host.dispatchPhysicsVector(input, output, count, spacing, format, physics.DispatchLaplacian4)
}

func (host *ComputeHost) DispatchMadelungContinuity(
	density, velocity, residual unsafe.Pointer,
	count int,
	spacing float32,
	format dtype.DType,
) {
	if count == 0 || host.bridge == nil {
		return
	}

	spacingBuffer := host.physicsSpacingBuffer(spacing, format)

	defer host.bridge.releaseScratch(spacingBuffer)

	if err := physics.DispatchMadelungContinuity(
		host.contextRef(),
		resolveBufferRef(density),
		resolveBufferRef(velocity),
		spacingBuffer,
		resolveBufferRef(residual),
		format,
		uint32(count),
	); err != nil {
		host.dispatchError(err)
	}
}

func (host *ComputeHost) DispatchQuantumPotential(density, output unsafe.Pointer, count int, spacing float32, format dtype.DType) {
	host.dispatchPhysicsVector(density, output, count, spacing, format, physics.DispatchQuantumPotential)
}

func (host *ComputeHost) dispatchFFT1D(
	realIn, imagIn, realOut, imagOut unsafe.Pointer,
	count int,
	format dtype.DType,
	inverse bool,
) {
	if count == 0 || host.bridge == nil {
		return
	}

	elementCount := uint32(count)
	var twiddleReal C.CUDABufferRef
	var twiddleImag C.CUDABufferRef

	if !physics.IsPowerOfTwo(elementCount) {
		realBytes, imagBytes := physics.TwiddleHostBytes(elementCount, inverse)
		twiddleReal = host.bridge.uploadHostBytes(realBytes)
		twiddleImag = host.bridge.uploadHostBytes(imagBytes)

		defer host.bridge.releaseScratch(twiddleReal)
		defer host.bridge.releaseScratch(twiddleImag)
	}

	dispatch := physics.DispatchFFT1D

	if inverse {
		dispatch = physics.DispatchIFFT1D
	}

	if err := dispatch(
		host.contextRef(),
		resolveBufferRef(realIn),
		resolveBufferRef(imagIn),
		resolveBufferRef(realOut),
		resolveBufferRef(imagOut),
		twiddleReal,
		twiddleImag,
		format,
		elementCount,
	); err != nil {
		host.dispatchError(err)
	}
}

func (host *ComputeHost) DispatchFFT1D(realIn, imagIn, realOut, imagOut unsafe.Pointer, count int, format dtype.DType) {
	host.dispatchFFT1D(realIn, imagIn, realOut, imagOut, count, format, false)
}

func (host *ComputeHost) DispatchIFFT1D(realIn, imagIn, realOut, imagOut unsafe.Pointer, count int, format dtype.DType) {
	host.dispatchFFT1D(realIn, imagIn, realOut, imagOut, count, format, true)
}
