//go:build darwin && cgo

package metal

import (
	"encoding/binary"
	"math"
	"unsafe"

	"github.com/theapemachine/manifesto/dtype"
	"github.com/theapemachine/manifesto/tensor"
	"github.com/theapemachine/puter/device"
)

func devicePanic(err error) {
	if err != nil {
		panic(err)
	}
}

func (backend *Backend) tensorAtPanic(pointer unsafe.Pointer) *metalTensor {
	target, err := backend.tensorAt(pointer)

	devicePanic(err)

	return target
}

func (backend *Backend) tensorsAtPanic(pointers ...unsafe.Pointer) []*metalTensor {
	tensors, err := backend.tensorsAt(pointers...)

	devicePanic(err)

	return tensors
}

func (backend *Backend) syncTensor(target *metalTensor) {
	devicePanic(target.Sync(backend.ctx))
}

func (backend *Backend) unaryElementwisePanic(
	dst, src unsafe.Pointer,
	format dtype.DType,
	operation metalUnaryFloat32Operation,
) {
	_ = format
	tensors := backend.tensorsAtPanic(src, dst)

	devicePanic(runMetalUnaryElementwise(operation, tensors[0], tensors[1]))
}

func (backend *Backend) emptyScalar(format dtype.DType) *metalTensor {
	shape, err := tensor.NewShape([]int{1})

	devicePanic(err)

	target, err := backend.bridge.empty(shape, format)

	devicePanic(err)

	return target
}

func (backend *Backend) readFloat32Scalar(pointer unsafe.Pointer) float32 {
	target := backend.tensorAtPanic(pointer)

	backend.syncTensor(target)

	storageDType, bytes, err := backend.Download(target)

	devicePanic(err)

	if storageDType != dtype.Float32 || len(bytes) < 4 {
		panic("metal: scalar output must be float32")
	}

	return math.Float32frombits(binary.LittleEndian.Uint32(bytes))
}

func (backend *Backend) readInt32Scalar(pointer unsafe.Pointer) int32 {
	target := backend.tensorAtPanic(pointer)

	backend.syncTensor(target)

	storageDType, bytes, err := backend.Download(target)

	devicePanic(err)

	if storageDType != dtype.Int32 || len(bytes) < 4 {
		panic("metal: scalar output must be int32")
	}

	return int32(binary.LittleEndian.Uint32(bytes))
}

func (backend *Backend) reductionScalar(
	values unsafe.Pointer,
	count int,
	format dtype.DType,
	operation metalReductionOp,
) float32 {
	_ = count
	input := backend.tensorAtPanic(values)
	out := backend.emptyScalar(format)

	devicePanic(runMetalReduction(operation, input, out))

	return backend.readFloat32Scalar(out.residentPointer())
}

func (backend *Backend) pairLossScalar(
	predictions, targets unsafe.Pointer,
	format dtype.DType,
	operation metalLossOp,
) float32 {
	tensors := backend.tensorsAtPanic(predictions, targets)
	out := backend.emptyScalar(format)

	devicePanic(runMetalPairLoss(operation, tensors[0], tensors[1], out))

	return backend.readFloat32Scalar(out.residentPointer())
}

func (backend *Backend) dotProduct(
	left, right unsafe.Pointer,
	count int,
	format dtype.DType,
) float32 {
	_ = count
	tensors := backend.tensorsAtPanic(left, right)
	out := backend.uploadFloat32Scalar(0.0, format)

	devicePanic(runMetalDot(tensors[0], tensors[1], out))

	return backend.readFloat32Scalar(out.residentPointer())
}

func (backend *Backend) gluInvoke(
	dst, gate, up unsafe.Pointer,
	run func(tensor.Tensor, tensor.Tensor, tensor.Tensor) error,
) {
	tensors := backend.tensorsAtPanic(gate, up, dst)

	devicePanic(run(tensors[0], tensors[1], tensors[2]))
}

func (backend *Backend) gluPackedInvoke(
	dst, packed unsafe.Pointer,
	batch, halfCount int,
	format dtype.DType,
	run func(tensor.Tensor, tensor.Tensor, tensor.Tensor) error,
) {
	_ = format
	packedTensor := backend.tensorAtPanic(packed)
	dstTensor := backend.tensorAtPanic(dst)

	if dstTensor.shape.Len() != batch*halfCount {
		devicePanic(tensor.ErrShapeMismatch)
	}

	halfShape := dstTensor.shape

	gate, err := backend.bridge.empty(halfShape, packedTensor.dtype)

	devicePanic(err)

	up, err := backend.bridge.empty(halfShape, packedTensor.dtype)

	devicePanic(err)

	devicePanic(runMetalSplit2(packedTensor, gate, up))
	devicePanic(run(gate, up, dstTensor))

	_ = gate.Close()
	_ = up.Close()
}

func (backend *Backend) samplingIndex(
	operation metalSamplingOp,
	config device.SamplingConfig,
	logits unsafe.Pointer,
	vocabSize int,
	format dtype.DType,
) int32 {
	_ = vocabSize
	_ = format
	logitTensor := backend.tensorAtPanic(logits)
	outShape, err := tensor.NewShape([]int{1})

	devicePanic(err)

	out, err := backend.bridge.empty(outShape, dtype.Int32)

	devicePanic(err)

	devicePanic(runMetalSamplingWithConfig(operation, logitTensor, out, config))
	backend.syncTensor(out)

	index := backend.readInt32Scalar(out.residentPointer())
	_ = out.Close()

	return index
}

func (backend *Backend) uploadFloat32Scalar(val float32, format dtype.DType) *metalTensor {
	shape, err := tensor.NewShape([]int{1})
	devicePanic(err)

	bytes := make([]byte, 4)
	binary.LittleEndian.PutUint32(bytes, math.Float32bits(val))

	t, err := backend.Upload(shape, format, bytes)
	devicePanic(err)

	return t.(*metalTensor)
}

func (backend *Backend) uploadInt32Scalar(val int32, format dtype.DType) *metalTensor {
	shape, err := tensor.NewShape([]int{1})
	devicePanic(err)

	bytes := make([]byte, 4)
	binary.LittleEndian.PutUint32(bytes, uint32(val))

	t, err := backend.Upload(shape, format, bytes)
	devicePanic(err)

	return t.(*metalTensor)
}
