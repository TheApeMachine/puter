package metal

import (
	"math"
	"testing"

	"github.com/theapemachine/manifesto/dtype"
)

func TestBatchNormElement41992Probe(t *testing.T) {
	backend := newBackendForDeviceTest(t)
	defer func() {
		if err := backend.Close(); err != nil {
			t.Fatalf("Close failed: %v", err)
		}
	}()

	spatial := 1024
	batch, channels := norm3DShape()
	fixture := norm3DFixtureForTest(batch, channels, spatial, dtype.Float32)
	input, scale, bias, mean, variance, out := batchNormEvalTensorsForTest(
		t, backend, dtype.Float32, batch, channels, spatial, fixture,
	)
	defer closeBenchmarkTensors(input, scale, bias, mean, variance, out)

	err := lookupBatchNormEvalKernel(t, dtype.Float32).Run(input, scale, bias, mean, variance, out)
	if err != nil {
		t.Fatalf("batchnorm_eval Run failed: %v", err)
	}

	_, actualBytes, err := backend.Download(out)
	if err != nil {
		t.Fatalf("Download failed: %v", err)
	}

	actual := decodeDTypeBytesToFloat32(actualBytes, dtype.Float32)
	inputStored := decodeDTypeBytesToFloat32(fixture.inputBytes, dtype.Float32)
	scaleStored := decodeDTypeBytesToFloat32(fixture.scaleBytes, dtype.Float32)
	biasStored := decodeDTypeBytesToFloat32(fixture.biasBytes, dtype.Float32)
	meanStored := decodeDTypeBytesToFloat32(fixture.meanBytes, dtype.Float32)
	varianceStored := decodeDTypeBytesToFloat32(fixture.varianceBytes, dtype.Float32)

	const index = 41992
	channel := (index / spatial) % channels

	metalExpected := expectedBatchNormEvalValuesMetalSqrt(
		t, backend,
		inputStored, scaleStored, biasStored, meanStored, varianceStored,
		batch, channels, spatial,
	)

	invMetal := metalInvStdDevsForTest(
		t, backend, []float32{varianceStored[channel] + layerNormEpsilonMetalForTest},
	)[0]
	manualMetal := (inputStored[index]-meanStored[channel])*invMetal*scaleStored[channel] + biasStored[channel]
	manualHost := (inputStored[index]-meanStored[channel])*normInvStdDev(varianceStored[channel])*scaleStored[channel] + biasStored[channel]
	delta := inputStored[index] - meanStored[channel]
	product := delta * invMetal * scaleStored[channel]
	recomputed := product + biasStored[channel]

	t.Logf("channel=%d", channel)
	t.Logf(
		"input=%08x mean=%08x scale=%08x bias=%08x variance=%08x",
		math.Float32bits(inputStored[index]),
		math.Float32bits(meanStored[channel]),
		math.Float32bits(scaleStored[channel]),
		math.Float32bits(biasStored[channel]),
		math.Float32bits(varianceStored[channel]),
	)
	t.Logf("invMetal=%08x %g", math.Float32bits(invMetal), invMetal)
	t.Logf(
		"delta=%08x product=%08x manualMetal=%08x recomputed=%08x sum=%08x",
		math.Float32bits(delta),
		math.Float32bits(product),
		math.Float32bits(manualMetal),
		math.Float32bits(recomputed),
		math.Float32bits(product+biasStored[channel]),
	)
	t.Logf("gpu=%08x %g metalExpected=%08x %g manualMetal=%08x %g manualHost=%08x %g",
		math.Float32bits(actual[index]), actual[index],
		math.Float32bits(metalExpected[index]), metalExpected[index],
		math.Float32bits(manualMetal), manualMetal,
		math.Float32bits(manualHost), manualHost,
	)
	t.Logf("ULP gpu vs metalExpected=%d manualMetal=%d manualHost=%d",
		float32ULPDistance(actual[index], metalExpected[index]),
		float32ULPDistance(actual[index], manualMetal),
		float32ULPDistance(metalExpected[index], manualMetal),
	)

	rowStart := channel * spatial
	twoStepExpected := applyNorm3DRowTwoStepForTest(
		inputStored[rowStart:rowStart+spatial],
		scaleStored[channel],
		biasStored[channel],
		meanStored[channel],
		invMetal,
	)[spatialIndexAt41992(spatial)]

	t.Logf("twoStepExpected=%08x metalExpected=%08x", math.Float32bits(twoStepExpected), math.Float32bits(metalExpected[index]))
}

func spatialIndexAt41992(spatial int) int {
	const index = 41992
	return index % spatial
}

func applyNorm3DRowTwoStepForTest(
	input []float32,
	scale float32,
	bias float32,
	mean float32,
	invStdDev float32,
) []float32 {
	out := make([]float32, len(input))

	for index, value := range input {
		normalized := (value - mean) * invStdDev
		out[index] = normalized*scale + bias
	}

	return out
}
