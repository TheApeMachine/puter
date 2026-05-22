package metal

import (
	"fmt"
	"testing"

	"github.com/theapemachine/manifesto/dtype"
	"github.com/theapemachine/manifesto/dtype/convert"
	"github.com/theapemachine/manifesto/tensor"
)

func BenchmarkBackend_DotDTypes(benchmark *testing.B) {
	backend := newBackendForDeviceTest(benchmark)
	defer func() {
		if err := backend.Close(); err != nil {
			benchmark.Fatalf("Close failed: %v", err)
		}
	}()

	for _, storageDType := range append([]dtype.DType{dtype.Float32}, elementwiseStorageDTypes...) {
		storageDType := storageDType

		benchmark.Run(storageDType.Name(), func(benchmark *testing.B) {
			elementCount := parityElementCounts[len(parityElementCounts)-1]
			shape, err := tensor.NewShape([]int{elementCount})
			if err != nil {
				benchmark.Fatal(err)
			}

			leftValues, rightValues, _ := binaryFloat32ParityValues(elementCount, "mul")
			leftBytes := encodeFloat32ValuesAsDType(leftValues, storageDType)
			rightBytes := encodeFloat32ValuesAsDType(rightValues, storageDType)

			if storageDType == dtype.Float32 {
				leftBytes = convert.Float32ToBytes(leftValues)
				rightBytes = convert.Float32ToBytes(rightValues)
			}

			left := uploadDTypeTensorForTest(benchmark, backend, shape, storageDType, leftBytes)
			right := uploadDTypeTensorForTest(benchmark, backend, shape, storageDType, rightBytes)
			defer closeBenchmarkTensors(left, right)

			benchmark.ResetTimer()

			for benchmark.Loop() {
				_ = backend.Dot(Resident(left), Resident(right), elementCount, storageDType)
			}
		})
	}
}

func BenchmarkBackend_AxpyDTypes(benchmark *testing.B) {
	backend := newBackendForDeviceTest(benchmark)
	defer func() {
		if err := backend.Close(); err != nil {
			benchmark.Fatalf("Close failed: %v", err)
		}
	}()

	for _, storageDType := range append([]dtype.DType{dtype.Float32}, elementwiseStorageDTypes...) {
		storageDType := storageDType

		benchmark.Run(fmt.Sprintf("%s/N=%d", storageDType.Name(), 8192), func(benchmark *testing.B) {
			elementCount := 8192
			shape, err := tensor.NewShape([]int{elementCount})
			if err != nil {
				benchmark.Fatal(err)
			}

			yValues := make([]float32, elementCount)
			xValues := make([]float32, elementCount)
			for index := range yValues {
				yValues[index] = float32(index)*0.01 + 1
				xValues[index] = float32(index)*0.02 - 0.5
			}

			yBytes := encodeFloat32ValuesAsDType(yValues, storageDType)
			xBytes := encodeFloat32ValuesAsDType(xValues, storageDType)

			if storageDType == dtype.Float32 {
				yBytes = convert.Float32ToBytes(yValues)
				xBytes = convert.Float32ToBytes(xValues)
			}

			y := uploadDTypeTensorForTest(benchmark, backend, shape, storageDType, yBytes)
			x := uploadDTypeTensorForTest(benchmark, backend, shape, storageDType, xBytes)
			defer closeBenchmarkTensors(y, x)

			alpha := float32(0.5)

			benchmark.ResetTimer()

			for benchmark.Loop() {
				backend.Axpy(Resident(y), Resident(x), elementCount, alpha, storageDType)
			}
		})
	}
}
