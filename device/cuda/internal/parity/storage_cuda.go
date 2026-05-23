//go:build cuda

package parity

import (
	"math/rand"
)

/*
AssertEncodedSlicesEqual fails when reduced-precision storage bytes differ.
*/
func AssertEncodedSlicesEqual(
	testingTB interface {
		Helper()
		Fatal(args ...any)
	},
	got, want []byte,
) {
	testingTB.Helper()

	if len(got) != len(want) {
		testingTB.Fatal("byte length mismatch got=", len(got), " want=", len(want))
	}

	for index := range got {
		if got[index] == want[index] {
			continue
		}

		testingTB.Fatal(
			"storage byte ", index,
			" got=", got[index],
			" want=", want[index],
		)
	}
}

/*
RandomUnaryInput fills a deterministic float32 vector for unary parity tests.
*/
func RandomUnaryInput(count int, seed int64) []float32 {
	rng := rand.New(rand.NewSource(seed))
	values := make([]float32, count)

	for index := range values {
		values[index] = rng.Float32()*4.0 - 2.0
	}

	return values
}
