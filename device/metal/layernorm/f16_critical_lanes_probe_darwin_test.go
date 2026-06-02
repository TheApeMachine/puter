//go:build darwin && cgo

package layernorm

import (
	"testing"

	"github.com/theapemachine/manifesto/dtype"
	cpuparity "github.com/theapemachine/puter/device/cpu/parity"
	metalparity "github.com/theapemachine/puter/device/metal/internal/parity"
)

func TestLayerNormF16CriticalLanesProbe(t *testing.T) {
	rows := 7
	cols := 8192
	elementCount := rows * cols
	seedBase := int64(0x4F00 + rows*1000 + cols)

	input := randomLayerNormVector(elementCount, seedBase)
	scale := randomLayerNormVector(cols, seedBase+1)
	bias := randomLayerNormVector(cols, seedBase+2)
	want := metalparity.LayerNormReference(input, scale, bias, rows, cols, dtype.Float16)

	harness := metalparity.NewHarness(t)
	defer harness.Close()

	inputTensor := harness.UploadVector(input, dtype.Float16)
	scaleTensor := harness.UploadVector(scale, dtype.Float16)
	biasTensor := harness.UploadVector(bias, dtype.Float16)
	outputTensor := harness.UploadVector(make([]float32, elementCount), dtype.Float16)
	defer inputTensor.Close()
	defer scaleTensor.Close()
	defer biasTensor.Close()
	defer outputTensor.Close()

	if err := DispatchLayerNormRefs(
		harness.ContextRef(),
		inputTensor.Ref(),
		scaleTensor.Ref(),
		biasTensor.Ref(),
		outputTensor.Ref(),
		dtype.Float16,
		uint32(rows),
		uint32(cols),
	); err != nil {
		t.Fatalf("dispatch: %v", err)
	}

	got := harness.DownloadFloat32(outputTensor, dtype.Float16)

	for _, laneIndex := range []int{306, 3289} {
		t.Logf(
			"lane %d got=%.9g want=%.9g ulp=%d",
			laneIndex,
			got[laneIndex],
			want[laneIndex],
			cpuparity.Float32ULPDistance(got[laneIndex], want[laneIndex]),
		)
	}
}
