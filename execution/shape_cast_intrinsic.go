package execution

import (
	"encoding/binary"
	"fmt"
	"math"

	"github.com/theapemachine/manifesto/dtype"
	"github.com/theapemachine/manifesto/dtype/convert"
	"github.com/theapemachine/manifesto/tensor"
)

func runCastIntrinsic(resolver *bindResolver) (any, error) {
	input, err := resolver.resolveInputTensor("0")

	if err != nil {
		return nil, err
	}

	liveInput, err := resolver.liveInputTensor("0", input)

	if err != nil {
		return nil, err
	}

	sourceDType, sourceBytes, err := liveInput.RawBytes()

	if err != nil {
		return nil, err
	}

	targetDType := resolver.outputDType

	if sourceDType == targetDType {
		if liveInput.Shape().Equal(resolver.outputShape) {
			return liveInput, nil
		}

		return liveInput.Reshape(resolver.outputShape.Dims())
	}

	converted, err := castTensorBytes(sourceDType, targetDType, sourceBytes)

	if err != nil {
		return nil, fmt.Errorf("shape.cast %q: %w", resolver.node.ID, err)
	}

	output, err := tensor.NewFromBytes(resolver.outputShape, targetDType, converted)

	if err != nil {
		return nil, err
	}

	return output, nil
}

func castTensorBytes(
	sourceDType dtype.DType,
	targetDType dtype.DType,
	sourceBytes []byte,
) ([]byte, error) {
	float64Values, err := convert.BytesToFloat64(sourceDType, sourceBytes)

	if err != nil {
		return nil, err
	}

	return encodeFloat64Values(targetDType, float64Values)
}

func encodeFloat64Values(targetDType dtype.DType, values []float64) ([]byte, error) {
	switch targetDType {
	case dtype.Float64:
		return convert.Float64ToBytes(values), nil
	case dtype.Float32:
		out := make([]float32, len(values))

		for index, value := range values {
			out[index] = float32(value)
		}

		return convert.Float32ToBytes(out), nil
	case dtype.BFloat16:
		out := make([]dtype.BF16, len(values))

		for index, value := range values {
			out[index] = dtype.NewBfloat16FromFloat32(float32(value))
		}

		return convert.BFloat16ToBytes(out), nil
	case dtype.Float16:
		out := make([]dtype.F16, len(values))

		for index, value := range values {
			out[index] = dtype.Fromfloat32(float32(value))
		}

		return convert.Float16ToBytes(out), nil
	case dtype.Float8E4M3:
		out := make([]dtype.F8E4M3, len(values))

		for index, value := range values {
			out[index] = dtype.NewF8E4M3FromFloat32(float32(value))
		}

		return convert.Float8E4M3ToBytes(out), nil
	case dtype.Float8E5M2:
		out := make([]dtype.F8E5M2, len(values))

		for index, value := range values {
			out[index] = dtype.NewF8E5M2FromFloat32(float32(value))
		}

		return convert.Float8E5M2ToBytes(out), nil
	case dtype.Int64:
		out := make([]byte, len(values)*8)

		for index, value := range values {
			binary.LittleEndian.PutUint64(out[index*8:], uint64(int64(value)))
		}

		return out, nil
	case dtype.Int32:
		out := make([]byte, len(values)*4)

		for index, value := range values {
			binary.LittleEndian.PutUint32(out[index*4:], uint32(int32(value)))
		}

		return out, nil
	case dtype.Int16:
		out := make([]byte, len(values)*2)

		for index, value := range values {
			binary.LittleEndian.PutUint16(out[index*2:], uint16(int16(value)))
		}

		return out, nil
	case dtype.Int8:
		int8Values, err := convert.BytesToInt8(dtype.Float64, convert.Float64ToBytes(values))

		if err != nil {
			return nil, err
		}

		out := make([]byte, len(int8Values))

		for index, value := range int8Values {
			out[index] = byte(value)
		}

		return out, nil
	case dtype.Uint64:
		out := make([]byte, len(values)*8)

		for index, value := range values {
			if value < 0 {
				return nil, fmt.Errorf("shape.cast: negative value %g cannot encode as uint64", value)
			}

			binary.LittleEndian.PutUint64(out[index*8:], uint64(value))
		}

		return out, nil
	case dtype.Uint32:
		out := make([]byte, len(values)*4)

		for index, value := range values {
			if value < 0 || value > float64(math.MaxUint32) {
				return nil, fmt.Errorf("shape.cast: value %g out of uint32 range", value)
			}

			binary.LittleEndian.PutUint32(out[index*4:], uint32(value))
		}

		return out, nil
	case dtype.Uint16:
		out := make([]byte, len(values)*2)

		for index, value := range values {
			if value < 0 || value > float64(math.MaxUint16) {
				return nil, fmt.Errorf("shape.cast: value %g out of uint16 range", value)
			}

			binary.LittleEndian.PutUint16(out[index*2:], uint16(value))
		}

		return out, nil
	case dtype.Uint8:
		out := make([]byte, len(values))

		for index, value := range values {
			if value < 0 || value > math.MaxUint8 {
				return nil, fmt.Errorf("shape.cast: value %g out of uint8 range", value)
			}

			out[index] = byte(value)
		}

		return out, nil
	default:
		return nil, fmt.Errorf("shape.cast: unsupported target dtype %s", targetDType)
	}
}
