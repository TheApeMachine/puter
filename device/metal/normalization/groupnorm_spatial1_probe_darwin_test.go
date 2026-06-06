//go:build darwin && cgo

package normalization

import (
	"testing"

	"github.com/theapemachine/manifesto/dtype"
	metalparity "github.com/theapemachine/puter/device/metal/internal/parity"
)

func TestGroupNormF16Spatial1Probe(t *testing.T) {
	batch := 2
	channels := 8
	groups := 2
	spatial := 1
	elementCount := batch * channels * spatial
	seedBase := int64(0x4E00 + batch*100 + channels*10 + groups + spatial)

	input := randomGroupNormVector(elementCount, seedBase)
	scale := randomGroupNormVector(channels, seedBase+1)
	bias := randomGroupNormVector(channels, seedBase+2)
	want := metalparity.GroupNormReference(input, scale, bias, batch, channels, spatial, groups, dtype.Float16)

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

	if err := DispatchGroupNormRefs(
		harness.ContextRef(),
		inputTensor.Ref(),
		scaleTensor.Ref(),
		biasTensor.Ref(),
		outputTensor.Ref(),
		dtype.Float16,
		uint32(batch),
		uint32(channels),
		uint32(spatial),
		uint32(groups),
	); err != nil {
		t.Fatalf("dispatch groupnorm: %v", err)
	}

	got := harness.DownloadFloat32(outputTensor, dtype.Float16)
	metalparity.AssertFloat32SlicesWithinULP(t, got, want, groupNormMaxULP(dtype.Float16))
}
