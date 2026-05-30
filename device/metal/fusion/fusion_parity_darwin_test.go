//go:build darwin && cgo

package fusion_test

import (
	"context"
	"fmt"
	"math"
	"testing"

	"github.com/smartystreets/goconvey/convey"
	"github.com/theapemachine/manifesto/codegen"
	"github.com/theapemachine/manifesto/dtype"
	"github.com/theapemachine/manifesto/dtype/convert"
	"github.com/theapemachine/manifesto/optimizer"
	"github.com/theapemachine/manifesto/tensor"
	cpuparity "github.com/theapemachine/puter/device/cpu/parity"
	"github.com/theapemachine/puter/device/metal"
)

var deviceParitySizes = []int{1, 7, 64, 1024, 8192}

func TestFusionProgramDeviceParity(testingObject *testing.T) {
	convey.Convey("Given ReLU(Add) on device buffers", testingObject, func() {
		backend := newFusionTestBackend(testingObject)
		defer backend.Close()

		fusionAST := reluAddFusion()
		sourceKernel, err := codegen.EmitMetal(fusionAST)
		convey.So(err, convey.ShouldBeNil)

		reference, err := codegen.EmitReferenceCPU(fusionAST)
		convey.So(err, convey.ShouldBeNil)

		program, err := backend.FusionCache().Program(
			sourceKernel.Source(),
			sourceKernel.KernelName(),
		)
		convey.So(err, convey.ShouldBeNil)

		for _, count := range deviceParitySizes {
			convey.Convey(fmt.Sprintf("It should match the scalar reference at N=%d", count), func() {
				left, right, output, expected := deviceParityBuffers(testingObject, backend, count, reference)
				defer left.Close()
				defer right.Close()
				defer output.Close()

				err := program.Dispatch(
					backend.MetalContextRef(),
					[]uintptr{
						metal.BufferRefFromDispatch(left.DispatchPointer()),
						metal.BufferRefFromDispatch(right.DispatchPointer()),
					},
					metal.BufferRefFromDispatch(output.DispatchPointer()),
					count,
				)
				convey.So(err, convey.ShouldBeNil)

				backend.SyncDevice()

				got := downloadFloat32DeviceTensor(testingObject, output)
				convey.So(got, convey.ShouldResemble, expected)
			})
		}
	})
}

func newFusionTestBackend(testingObject *testing.T) *metal.Backend {
	testingObject.Helper()

	backend, err := metal.NewBackend(context.Background(), nil)

	if err != nil {
		testingObject.Skipf("Metal backend unavailable: %v", err)
	}

	return backend
}

func reluAddFusion() *optimizer.FusionAST {
	return &optimizer.FusionAST{
		InputPorts: []string{"x", "y"},
		OutputPort: "result",
		Root: &optimizer.ASTNode{
			Type: optimizer.NodeReLU,
			Children: []*optimizer.ASTNode{
				{
					Type: optimizer.NodeAdd,
					Children: []*optimizer.ASTNode{
						{Type: optimizer.NodeInput, InputIndex: 0},
						{Type: optimizer.NodeInput, InputIndex: 1},
					},
				},
			},
		},
	}
}

func deviceParityBuffers(
	testingObject *testing.T,
	backend *metal.Backend,
	count int,
	reference *codegen.CPUKernel,
) (left *metal.DeviceTensor, right *metal.DeviceTensor, output *metal.DeviceTensor, expected []float32) {
	testingObject.Helper()

	leftValues := make([]float32, count)
	rightValues := make([]float32, count)
	expected = make([]float32, count)

	for index := 0; index < count; index++ {
		leftValues[index] = float32(math.Sin(float64(index)))
		rightValues[index] = float32(math.Cos(float64(index)))
	}

	if err := reference.Run([][]float32{leftValues, rightValues}, expected, count); err != nil {
		testingObject.Fatal(err)
	}

	left = uploadFloat32Tensor(testingObject, backend, leftValues)
	right = uploadFloat32Tensor(testingObject, backend, rightValues)
	output = uploadFloat32Tensor(testingObject, backend, make([]float32, count))

	return left, right, output, expected
}

func uploadFloat32Tensor(
	testingHandle interface {
		Helper()
		Fatalf(string, ...any)
	},
	backend *metal.Backend,
	values []float32,
) *metal.DeviceTensor {
	testingHandle.Helper()

	shape, err := tensor.NewShape([]int{len(values)})

	if err != nil {
		testingHandle.Fatalf("uploadFloat32Tensor: shape: %v", err)
	}

	resident, err := backend.Upload(shape, dtype.Float32, convert.Float32ToBytes(values))

	if err != nil {
		testingHandle.Fatalf("uploadFloat32Tensor: upload: %v", err)
	}

	deviceTensor, ok := resident.(*metal.DeviceTensor)

	if !ok {
		testingHandle.Fatalf("uploadFloat32Tensor: got %T", resident)
	}

	return deviceTensor
}

func downloadFloat32DeviceTensor(testingObject *testing.T, resident *metal.DeviceTensor) []float32 {
	testingObject.Helper()

	elementFormat, rawBytes, err := resident.RawBytes()

	if err != nil {
		testingObject.Fatal(err)
	}

	values, err := convert.BytesToFloat32(elementFormat, rawBytes)

	if err != nil {
		testingObject.Fatal(err)
	}

	return values
}

func BenchmarkFusionProgramDeviceDispatch(benchmark *testing.B) {
	backend := newFusionBenchmarkBackend(benchmark)
	defer backend.Close()

	fusionAST := reluAddFusion()
	sourceKernel, err := codegen.EmitMetal(fusionAST)

	if err != nil {
		benchmark.Fatal(err)
	}

	program, err := backend.FusionCache().Program(
		sourceKernel.Source(),
		sourceKernel.KernelName(),
	)

	if err != nil {
		benchmark.Fatal(err)
	}

	count := 8192
	left := uploadFloat32Tensor(benchmark, backend, make([]float32, count))
	defer left.Close()
	right := uploadFloat32Tensor(benchmark, backend, make([]float32, count))
	defer right.Close()
	output := uploadFloat32Tensor(benchmark, backend, make([]float32, count))
	defer output.Close()

	benchmark.ResetTimer()

	for benchmark.Loop() {
		if err := program.Dispatch(
			backend.MetalContextRef(),
			[]uintptr{
				metal.BufferRefFromDispatch(left.DispatchPointer()),
				metal.BufferRefFromDispatch(right.DispatchPointer()),
			},
			metal.BufferRefFromDispatch(output.DispatchPointer()),
			count,
		); err != nil {
			benchmark.Fatal(err)
		}
	}
}

func newFusionBenchmarkBackend(benchmark *testing.B) *metal.Backend {
	benchmark.Helper()

	backend, err := metal.NewBackend(context.Background(), nil)

	if err != nil {
		benchmark.Skipf("Metal backend unavailable: %v", err)
	}

	return backend
}

func TestFusionProgramDeviceParityULP(testingObject *testing.T) {
	convey.Convey("Given ReLU(Add) with tight ULP bounds", testingObject, func() {
		backend := newFusionTestBackend(testingObject)
		defer backend.Close()

		fusionAST := reluAddFusion()
		sourceKernel, err := codegen.EmitMetal(fusionAST)
		convey.So(err, convey.ShouldBeNil)

		reference, err := codegen.EmitReferenceCPU(fusionAST)
		convey.So(err, convey.ShouldBeNil)

		program, err := backend.FusionCache().Program(
			sourceKernel.Source(),
			sourceKernel.KernelName(),
		)
		convey.So(err, convey.ShouldBeNil)

		count := 1024
		left, right, output, expected := deviceParityBuffers(testingObject, backend, count, reference)
		defer left.Close()
		defer right.Close()
		defer output.Close()

		err = program.Dispatch(
			backend.MetalContextRef(),
			[]uintptr{
				metal.BufferRefFromDispatch(left.DispatchPointer()),
				metal.BufferRefFromDispatch(right.DispatchPointer()),
			},
			metal.BufferRefFromDispatch(output.DispatchPointer()),
			count,
		)
		convey.So(err, convey.ShouldBeNil)

		backend.SyncDevice()

		got := downloadFloat32DeviceTensor(testingObject, output)
		cpuparity.AssertFloat32SlicesWithinULP(testingObject, got, expected, 2)
	})
}
