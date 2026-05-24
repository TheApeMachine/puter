//go:build xla

package causal_test

import (
	"fmt"
	"testing"
	"unsafe"

	"github.com/smartystreets/goconvey/convey"
	"github.com/theapemachine/manifesto/dtype"
	cpucausal "github.com/theapemachine/puter/device/cpu/causal"
	"github.com/theapemachine/puter/device/xla"
	xlaparity "github.com/theapemachine/puter/device/xla/internal/parity"
)

var referenceCausal = cpucausal.New()

func TestCausalXLAParity(t *testing.T) {
	harness := xla.NewParityHarness(t)
	defer harness.Close()

	convey.Convey("Given XLA CATE", t, func() {
		for _, count := range xlaparity.Lengths {
			convey.Convey(fmt.Sprintf("N=%d", count), func() {
				treated := xlaparity.RandomUnaryInput(count, 0xb100+int64(count))
				control := xlaparity.RandomUnaryInput(count, 0xb200+int64(count))
				want := make([]float32, count)
				referenceCausal.CATE(
					unsafe.Pointer(&treated[0]),
					unsafe.Pointer(&control[0]),
					unsafe.Pointer(&want[0]),
					count,
					dtype.Float32,
				)

				treatedTensor := harness.UploadVector(treated, dtype.Float32)
				controlTensor := harness.UploadVector(control, dtype.Float32)
				outputTensor := harness.UploadVector(make([]float32, count), dtype.Float32)
				defer treatedTensor.Close()
				defer controlTensor.Close()
				defer outputTensor.Close()

				harness.Backend().CATE(
					xla.ResidentPointer(treatedTensor),
					xla.ResidentPointer(controlTensor),
					xla.ResidentPointer(outputTensor),
					count,
					dtype.Float32,
				)

				got := harness.DownloadFloat32(outputTensor, dtype.Float32)
				xlaparity.AssertFloat32SlicesWithinULP(t, got, want, 2)
			})
		}
	})
}

func BenchmarkCATEXLAParity(b *testing.B) {
	harness := xla.NewParityHarness(b)
	defer harness.Close()

	count := 8192
	treated := xlaparity.RandomUnaryInput(count, 0xb300)
	control := xlaparity.RandomUnaryInput(count, 0xb400)
	treatedTensor := harness.UploadVector(treated, dtype.Float32)
	controlTensor := harness.UploadVector(control, dtype.Float32)
	outputTensor := harness.UploadVector(make([]float32, count), dtype.Float32)
	defer treatedTensor.Close()
	defer controlTensor.Close()
	defer outputTensor.Close()

	for b.Loop() {
		harness.Backend().CATE(
			xla.ResidentPointer(treatedTensor),
			xla.ResidentPointer(controlTensor),
			xla.ResidentPointer(outputTensor),
			count,
			dtype.Float32,
		)
	}
}
