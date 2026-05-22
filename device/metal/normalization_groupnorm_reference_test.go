package metal

import (
	"math"
	"testing"
)

func TestGroupNormReferenceMatchesMetalSerialAlgorithm(t *testing.T) {
	batch, channels := norm3DShape()
	spatial := 7
	groups := metalDefaultGroupNormGroups
	input, scale, bias, _, _ := norm3DValues(batch, channels, spatial)

	want := expectedGroupNormValues(input, scale, bias, batch, channels, spatial)
	got := groupNormMetalSerialReference(input, scale, bias, batch, channels, spatial, groups)

	maxDistance, maxIndex := maxNormalizationFloat32ULPDistance(got, want)
	if maxDistance > 0 {
		t.Fatalf(
			"reference mismatch at %d: got %08x (%g), want %08x (%g), distance %d",
			maxIndex,
			math.Float32bits(got[maxIndex]),
			got[maxIndex],
			math.Float32bits(want[maxIndex]),
			want[maxIndex],
			maxDistance,
		)
	}
}

func groupNormMetalSerialReference(
	input []float32,
	scale []float32,
	bias []float32,
	batch int,
	channels int,
	spatial int,
	groups int,
) []float32 {
	out := make([]float32, len(input))
	channelsPerGroup := channels / groups

	for batchIndex := range batch {
		for groupIndex := range groups {
			channelStart := groupIndex * channelsPerGroup
			groupStart := (batchIndex*channels + channelStart) * spatial
			groupSize := channelsPerGroup * spatial
			group := input[groupStart : groupStart+groupSize]

			var sum float32
			for _, value := range group {
				sum += value
			}

			mean := sum / float32(groupSize)
			var variance float32

			for _, value := range group {
				delta := value - mean
				variance += delta * delta
			}

			invStdDev := normInvStdDev(variance / float32(groupSize))

			for channelIndex := range channelsPerGroup {
				for spatialIndex := range spatial {
					index := groupStart + channelIndex*spatial + spatialIndex
					channel := channelStart + channelIndex
					normalized := (input[index] - mean) * invStdDev
					out[index] = normalized*scale[channel] + bias[channel]
				}
			}
		}
	}

	return out
}
