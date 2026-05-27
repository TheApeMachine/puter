//go:build darwin && cgo

package metal

import (
	"context"
	"math"
	"testing"

	"github.com/smartystreets/goconvey/convey"
	"github.com/theapemachine/manifesto/dtype"
	"github.com/theapemachine/manifesto/dtype/convert"
	"github.com/theapemachine/manifesto/tensor"
	"github.com/theapemachine/puter/device"
	cpuparity "github.com/theapemachine/puter/device/cpu/parity"
)

func TestLlamaCoreMetalEmbeddingLookupParity(testingObject *testing.T) {
	backend := newMetalTestBackend(testingObject)
	defer backend.Close()

	convey.Convey("Given token embedding inputs", testingObject, func() {
		table := uploadRoPETensor(testingObject, backend, []float32{
			1, 2, 3,
			4, 5, 6,
			7, 8, 9,
		})
		defer table.Close()
		indices := uploadInt32MetalTensor(testingObject, backend, []int32{2, 0})
		defer indices.Close()
		output := uploadRoPETensor(testingObject, backend, make([]float32, 6))
		defer output.Close()

		backend.Lookup(
			table.DispatchPointer(),
			indices.DispatchPointer(),
			output.DispatchPointer(),
			3,
			3,
			2,
			dtype.Float32,
		)
		backend.SyncDevice()

		convey.Convey("It should match row lookup semantics", func() {
			got := downloadFloat32MetalTensor(testingObject, output)

			convey.So(got, convey.ShouldResemble, []float32{7, 8, 9, 1, 2, 3})
		})
	})
}

func TestLlamaCoreMetalMatmulParity(testingObject *testing.T) {
	backend := newMetalTestBackend(testingObject)
	defer backend.Close()

	convey.Convey("Given non-square matmul inputs", testingObject, func() {
		rows := 3
		inner := 5
		cols := 4
		leftValues := []float32{
			1, 2, 3, 4, 5,
			-1, 0.5, 2, -0.5, 3,
			0, -2, 1, 1.5, -1,
		}
		rightValues := []float32{
			0.5, 1, -1, 2,
			-2, 0.25, 0.5, -1,
			1.5, -0.75, 2, 0,
			0, 3, -0.5, 1,
			2, -1, 1, 0.5,
		}
		want := referenceMatmul(leftValues, rightValues, rows, inner, cols)
		left := uploadRoPETensor(testingObject, backend, leftValues)
		defer left.Close()
		right := uploadRoPETensor(testingObject, backend, rightValues)
		defer right.Close()
		output := uploadRoPETensor(testingObject, backend, make([]float32, rows*cols))
		defer output.Close()

		backend.Matmul(
			output.DispatchPointer(),
			left.DispatchPointer(),
			right.DispatchPointer(),
			rows,
			inner,
			cols,
			dtype.Float32,
		)
		backend.SyncDevice()

		convey.Convey("It should match row-major CPU matmul", func() {
			got := downloadFloat32MetalTensor(testingObject, output)

			cpuparity.AssertFloat32SlicesWithinULP(testingObject, got, want, 1)
		})
	})
}

func TestLlamaCoreMetalMatmulBFloat16Parity(testingObject *testing.T) {
	backend := newMetalTestBackend(testingObject)
	defer backend.Close()

	convey.Convey("Given BF16 matmul inputs", testingObject, func() {
		testCases := []metalMatmulBFloat16Case{
			newSmallBFloat16MatmulCase(),
			newAlignedBFloat16MatmulCase(),
		}

		for _, testCase := range testCases {
			convey.Convey(testCase.name, func() {
				runMetalMatmulBFloat16Case(testingObject, backend, testCase)
			})
		}
	})
}

func TestLlamaCoreMetalRMSNormParity(testingObject *testing.T) {
	backend := newMetalTestBackend(testingObject)
	defer backend.Close()

	convey.Convey("Given RMSNorm inputs", testingObject, func() {
		rows := 2
		cols := 4
		epsilon := float32(1.0e-5)
		inputValues := []float32{
			1, -2, 3, -4,
			0.5, 1.5, -2.5, 3.5,
		}
		scaleValues := []float32{1, 0.5, -1, 2}
		want := referenceRMSNorm(inputValues, scaleValues, rows, cols, epsilon)
		input := uploadRoPETensor(testingObject, backend, inputValues)
		defer input.Close()
		scale := uploadRoPETensor(testingObject, backend, scaleValues)
		defer scale.Close()
		output := uploadRoPETensor(testingObject, backend, make([]float32, len(inputValues)))
		defer output.Close()

		backend.RMSNorm(
			device.RMSNormConfig{Epsilon: float64(epsilon)},
			input.DispatchPointer(),
			scale.DispatchPointer(),
			output.DispatchPointer(),
			rows,
			cols,
			dtype.Float32,
		)
		backend.SyncDevice()

		convey.Convey("It should match CPU RMSNorm semantics", func() {
			got := downloadFloat32MetalTensor(testingObject, output)

			cpuparity.AssertFloat32SlicesWithinULP(testingObject, got, want, 4)
		})
	})
}

func TestLlamaCoreMetalSwiGLUTensorsParity(testingObject *testing.T) {
	backend := newMetalTestBackend(testingObject)
	defer backend.Close()

	convey.Convey("Given SwiGLU tensor inputs", testingObject, func() {
		gateValues := []float32{-2, -0.5, 0, 1, 3}
		upValues := []float32{4, -3, 2, 0.5, -1}
		want := referenceSwiGLU(gateValues, upValues)
		gate := uploadRoPETensor(testingObject, backend, gateValues)
		defer gate.Close()
		up := uploadRoPETensor(testingObject, backend, upValues)
		defer up.Close()
		output := uploadRoPETensor(testingObject, backend, make([]float32, len(gateValues)))
		defer output.Close()

		backend.SwiGLUTensors(
			output.DispatchPointer(),
			gate.DispatchPointer(),
			up.DispatchPointer(),
			len(gateValues),
			dtype.Float32,
		)
		backend.SyncDevice()

		convey.Convey("It should match silu(gate) times up", func() {
			got := downloadFloat32MetalTensor(testingObject, output)

			cpuparity.AssertFloat32SlicesWithinULP(testingObject, got, want, 4)
		})
	})
}

func BenchmarkLlamaCoreMetalMatmul(benchmark *testing.B) {
	backend := newMetalBenchmarkBackend(benchmark)
	defer backend.Close()

	rows := 128
	inner := 2048
	cols := 2048
	left := uploadRoPETensor(benchmark, backend, make([]float32, rows*inner))
	defer left.Close()
	right := uploadRoPETensor(benchmark, backend, make([]float32, inner*cols))
	defer right.Close()
	output := uploadRoPETensor(benchmark, backend, make([]float32, rows*cols))
	defer output.Close()

	benchmark.ResetTimer()

	for benchmark.Loop() {
		backend.Matmul(
			output.DispatchPointer(),
			left.DispatchPointer(),
			right.DispatchPointer(),
			rows,
			inner,
			cols,
			dtype.Float32,
		)
	}

	backend.SyncDevice()
}

func newMetalTestBackend(testingObject *testing.T) *Backend {
	testingObject.Helper()

	backend, err := NewBackend(context.Background(), nil)
	if err != nil {
		testingObject.Skipf("Metal backend unavailable: %v", err)
	}

	return backend
}

func newMetalBenchmarkBackend(benchmark *testing.B) *Backend {
	benchmark.Helper()

	backend, err := NewBackend(context.Background(), nil)
	if err != nil {
		benchmark.Skipf("Metal backend unavailable: %v", err)
	}

	return backend
}

func referenceMatmul(left []float32, right []float32, rows int, inner int, cols int) []float32 {
	output := make([]float32, rows*cols)

	for rowIndex := 0; rowIndex < rows; rowIndex++ {
		for colIndex := 0; colIndex < cols; colIndex++ {
			sum := float32(0)

			for innerIndex := 0; innerIndex < inner; innerIndex++ {
				sum += left[rowIndex*inner+innerIndex] * right[innerIndex*cols+colIndex]
			}

			output[rowIndex*cols+colIndex] = sum
		}
	}

	return output
}

func referenceMatmulBFloat16(left []float32, right []float32, rows int, inner int, cols int) []float32 {
	roundedLeft := roundFloat32ToBFloat16(left)
	roundedRight := roundFloat32ToBFloat16(right)
	output := referenceMatmul(roundedLeft, roundedRight, rows, inner, cols)

	return roundFloat32ToBFloat16(output)
}

func roundFloat32ToBFloat16(values []float32) []float32 {
	rounded := make([]float32, len(values))

	for valueIndex, value := range values {
		rounded[valueIndex] = dtype.NewBfloat16FromFloat32(value).Float32()
	}

	return rounded
}

type metalMatmulBFloat16Case struct {
	name        string
	rows        int
	inner       int
	cols        int
	leftValues  []float32
	rightValues []float32
}

func newSmallBFloat16MatmulCase() metalMatmulBFloat16Case {
	return metalMatmulBFloat16Case{
		name:  "It should match a small BF16 row-major CPU matmul",
		rows:  3,
		inner: 5,
		cols:  4,
		leftValues: []float32{
			1, 2, 3, 4, 5,
			-1, 0.5, 2, -0.5, 3,
			0, -2, 1, 1.5, -1,
		},
		rightValues: []float32{
			0.5, 1, -1, 2,
			-2, 0.25, 0.5, -1,
			1.5, -0.75, 2, 0,
			0, 3, -0.5, 1,
			2, -1, 1, 0.5,
		},
	}
}

func newAlignedBFloat16MatmulCase() metalMatmulBFloat16Case {
	rows := 2
	inner := 128
	cols := 128

	return metalMatmulBFloat16Case{
		name:        "It should match an aligned BF16 row-major CPU matmul",
		rows:        rows,
		inner:       inner,
		cols:        cols,
		leftValues:  quantizedBFloat16Inputs(rows * inner),
		rightValues: identityMatmulValues(inner, cols),
	}
}

func runMetalMatmulBFloat16Case(
	testingObject *testing.T,
	backend *Backend,
	testCase metalMatmulBFloat16Case,
) {
	testingObject.Helper()

	want := referenceMatmulBFloat16(
		testCase.leftValues,
		testCase.rightValues,
		testCase.rows,
		testCase.inner,
		testCase.cols,
	)
	left := uploadMetalFloatTensor(testingObject, backend, testCase.leftValues, dtype.BFloat16)
	defer left.Close()
	right := uploadMetalFloatTensor(testingObject, backend, testCase.rightValues, dtype.BFloat16)
	defer right.Close()
	output := uploadMetalFloatTensor(
		testingObject,
		backend,
		make([]float32, testCase.rows*testCase.cols),
		dtype.BFloat16,
	)
	defer output.Close()

	backend.Matmul(
		output.DispatchPointer(),
		left.DispatchPointer(),
		right.DispatchPointer(),
		testCase.rows,
		testCase.inner,
		testCase.cols,
		dtype.BFloat16,
	)
	backend.SyncDevice()

	got := downloadFloat32MetalTensor(testingObject, output)

	cpuparity.AssertFloat32SlicesWithinULP(testingObject, got, want, 0)
}

func quantizedBFloat16Inputs(count int) []float32 {
	values := make([]float32, count)

	for valueIndex := range values {
		values[valueIndex] = float32(valueIndex%17-8) * 0.25
	}

	return values
}

func identityMatmulValues(rows int, cols int) []float32 {
	values := make([]float32, rows*cols)

	for rowIndex := range rows {
		if rowIndex >= cols {
			return values
		}

		values[rowIndex*cols+rowIndex] = 1
	}

	return values
}

func uploadMetalFloatTensor(
	testingHandle interface {
		Helper()
		Fatalf(string, ...any)
	},
	backend *Backend,
	values []float32,
	format dtype.DType,
) *DeviceTensor {
	testingHandle.Helper()

	shape, err := tensor.NewShape([]int{len(values)})
	if err != nil {
		testingHandle.Fatalf("uploadMetalFloatTensor: shape: %v", err)
	}

	rawBytes, err := encodeMetalFloatTensor(values, format)
	if err != nil {
		testingHandle.Fatalf("uploadMetalFloatTensor: encode: %v", err)
	}

	resident, err := backend.Upload(shape, format, rawBytes)
	if err != nil {
		testingHandle.Fatalf("uploadMetalFloatTensor: upload: %v", err)
	}

	deviceTensor, ok := resident.(*DeviceTensor)
	if !ok {
		testingHandle.Fatalf("uploadMetalFloatTensor: got %T", resident)
	}

	return deviceTensor
}

func encodeMetalFloatTensor(values []float32, format dtype.DType) ([]byte, error) {
	switch format {
	case dtype.Float32:
		return convert.Float32ToBytes(values), nil
	case dtype.BFloat16:
		encoded := make([]dtype.BF16, len(values))

		for valueIndex, value := range values {
			encoded[valueIndex] = dtype.NewBfloat16FromFloat32(value)
		}

		return convert.BFloat16ToBytes(encoded), nil
	default:
		return nil, tensor.ErrDTypeMismatch
	}
}

func referenceRMSNorm(
	input []float32,
	scale []float32,
	rows int,
	cols int,
	epsilon float32,
) []float32 {
	output := make([]float32, len(input))

	for rowIndex := 0; rowIndex < rows; rowIndex++ {
		rowOffset := rowIndex * cols
		sumSquares := float32(0)

		for colIndex := 0; colIndex < cols; colIndex++ {
			value := input[rowOffset+colIndex]
			sumSquares += value * value
		}

		invRMS := float32(1.0 / math.Sqrt(float64(sumSquares/float32(cols)+epsilon)))

		for colIndex := 0; colIndex < cols; colIndex++ {
			output[rowOffset+colIndex] = input[rowOffset+colIndex] * invRMS * scale[colIndex]
		}
	}

	return output
}

func referenceSwiGLU(gate []float32, up []float32) []float32 {
	output := make([]float32, len(gate))

	for index, value := range gate {
		silu := value / (1.0 + float32(math.Exp(float64(-value))))
		output[index] = silu * up[index]
	}

	return output
}
