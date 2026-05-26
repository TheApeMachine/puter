//go:build arm64

package activation

import (
	"fmt"
	"testing"

	. "github.com/smartystreets/goconvey/convey"
	"github.com/theapemachine/puter/device/cpu/parity"
)

func TestSwiGLUTensorsF32NEON(testingObject *testing.T) {
	Convey("Given SwiGLU NEON tensors", testingObject, func() {
		for _, length := range parity.Lengths {
			gate := make([]float32, length)
			up := make([]float32, length)
			destination := make([]float32, length)
			reference := make([]float32, length)

			for index := range gate {
				gate[index] = float32(index)*0.1 - 0.5
				up[index] = float32(index)*0.1 + 0.5
			}

			SwiGLUTensorsF32Generic(&reference[0], &gate[0], &up[0], length)
			SwiGLUTensorsF32NEON(&destination[0], &gate[0], &up[0], length)

			Convey(fmt.Sprintf("It should match generic within 2 ULP at N=%d", length), func() {
				parity.AssertFloat32SlicesWithinULP(testingObject, destination, reference, 2)
			})
		}
	})
}

func BenchmarkSwiGLUTensorsF32NEON(benchmark *testing.B) {
	length := 8192
	gate := make([]float32, length)
	up := make([]float32, length)
	destination := make([]float32, length)

	for benchmark.Loop() {
		SwiGLUTensorsF32NEON(&destination[0], &gate[0], &up[0], length)
	}
}
