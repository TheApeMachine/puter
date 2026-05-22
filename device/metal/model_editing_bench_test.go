package metal

import (
	"context"
	"testing"

	"github.com/theapemachine/manifesto/dtype"
)

func BenchmarkKernel_RunWeightGraftAddFloat32(benchmark *testing.B) {
	backend := newBackendForBenchmark(benchmark)
	defer func() {
		_ = backend.Close()
	}()

	elementCount := 8192
	fixture := weightGraftFixtureForTest(elementCount, dtype.Float32)
	shape := mustShapeForTest(benchmark, []int{elementCount})
	weights := uploadDTypeTensorForTest(benchmark, backend, shape, dtype.Float32, fixture.weightsBytes)
	injection := uploadDTypeTensorForTest(benchmark, backend, shape, dtype.Float32, fixture.injectionBytes)
	defer closeBenchmarkTensors(weights, injection)

	kernel := lookupWeightGraftAddKernel(benchmark, dtype.Float32)
	benchmark.SetBytes(int64(elementCount * 8))
	benchmark.ResetTimer()

	for benchmark.Loop() {
		if err := kernel.Run(weights, injection); err != nil {
			benchmark.Fatal(err)
		}

		if err := weights.Sync(context.Background()); err != nil {
			benchmark.Fatal(err)
		}
	}
}
