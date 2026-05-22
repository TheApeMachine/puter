package metal

import (
	"math"
	"testing"
)

func TestBatchNormInvStdDevMetalPaths(t *testing.T) {
	backend := newBackendForDeviceTest(t)
	defer func() {
		if err := backend.Close(); err != nil {
			t.Fatalf("Close failed: %v", err)
		}
	}()

	variance := float32(0.9375)
	variancePlusEpsilon := variance + layerNormEpsilonMetalForTest

	metalSqrtRecip := metalInvStdDevsForTest(t, backend, []float32{variancePlusEpsilon})[0]
	hostMath := normInvStdDev(variance)
	hostMetalSqrt := 1 / sqrtFloat32(variancePlusEpsilon)

	t.Logf(
		"variancePlusEpsilon=%g metalSqrtRecip=%08x (%g) hostMath=%08x (%g) hostMetalSqrt=%08x (%g)",
		variancePlusEpsilon,
		math.Float32bits(metalSqrtRecip),
		metalSqrtRecip,
		math.Float32bits(hostMath),
		hostMath,
		math.Float32bits(hostMetalSqrt),
		hostMetalSqrt,
	)
	t.Logf(
		"ULP metalSqrtRecip vs hostMath=%d hostMetalSqrt=%d",
		float32ULPDistance(metalSqrtRecip, hostMath),
		float32ULPDistance(metalSqrtRecip, hostMetalSqrt),
	)
}
