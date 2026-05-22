//go:build arm64

package embedding

import (
	"fmt"
	"testing"
	"unsafe"

	"github.com/smartystreets/goconvey/convey"
	"github.com/theapemachine/puter/device/cpu/parity"
)

const embeddingNEONMaxULP = 0

func TestLookupF32NEONParity(t *testing.T) {
	convey.Convey("Given Lookup float32 NEON", t, func() {
		const vocab = 128
		const indexCount = 7

		for _, hidden := range parity.Lengths {
			convey.Convey(fmt.Sprintf("It should match generic for hidden=%d", hidden), func() {
				table := randomEmbeddingTable(vocab, hidden, 0xE220+int64(hidden))
				indices := randomEmbeddingIndices(indexCount, vocab, 0xE221+int64(hidden))

				want := make([]float32, indexCount*hidden)
				got := make([]float32, indexCount*hidden)

				runLookupF32Generic(
					unsafe.Pointer(&table[0]),
					unsafe.Pointer(&indices[0]),
					unsafe.Pointer(&want[0]),
					vocab, hidden, indexCount,
				)
				runLookupF32NEON(
					unsafe.Pointer(&table[0]),
					unsafe.Pointer(&indices[0]),
					unsafe.Pointer(&got[0]),
					vocab, hidden, indexCount,
				)

				parity.AssertFloat32SlicesWithinULP(t, got, want, embeddingNEONMaxULP)
			})
		}

		convey.Convey("It should match generic via direct CopyRow asm at parity.Lengths", func() {
			for _, hidden := range parity.Lengths {
				table := randomEmbeddingTable(4, hidden, 0xE222+int64(hidden))
				want := make([]float32, hidden)
				got := make([]float32, hidden)
				copy(want, table[hidden:hidden*2])

				copyRowF32NEON(&got[0], &table[hidden], hidden)

				parity.AssertFloat32SlicesWithinULP(t, got, want, embeddingNEONMaxULP)
			}
		})
	})
}

func TestBagF32NEONParity(t *testing.T) {
	convey.Convey("Given Bag float32 NEON", t, func() {
		const vocab = 64
		const bagCount = 3
		const indexCount = 11

		for _, hidden := range parity.Lengths {
			convey.Convey(fmt.Sprintf("It should match generic for hidden=%d", hidden), func() {
				table := randomEmbeddingTable(vocab, hidden, 0xE230+int64(hidden))
				indices := randomEmbeddingIndices(indexCount, vocab, 0xE231+int64(hidden))
				offsets := []int32{0, 4, 7}

				want := make([]float32, bagCount*hidden)
				got := make([]float32, bagCount*hidden)

				runBagF32Generic(
					unsafe.Pointer(&table[0]),
					unsafe.Pointer(&indices[0]),
					unsafe.Pointer(&offsets[0]),
					unsafe.Pointer(&want[0]),
					vocab, hidden, bagCount, indexCount,
				)
				runBagF32NEON(
					unsafe.Pointer(&table[0]),
					unsafe.Pointer(&indices[0]),
					unsafe.Pointer(&offsets[0]),
					unsafe.Pointer(&got[0]),
					vocab, hidden, bagCount, indexCount,
				)

				parity.AssertFloat32SlicesWithinULP(t, got, want, embeddingNEONMaxULP)
			})
		}

		convey.Convey("It should match generic via direct AddRow asm at parity.Lengths", func() {
			for _, hidden := range parity.Lengths {
				table := randomEmbeddingTable(4, hidden, 0xE232+int64(hidden))
				want := make([]float32, hidden)
				got := make([]float32, hidden)

				for dimIndex := range hidden {
					want[dimIndex] = float32(dimIndex) * 0.25
					got[dimIndex] = want[dimIndex]
				}

				for dimIndex := range hidden {
					want[dimIndex] += table[hidden+dimIndex]
				}

				addRowF32NEON(&got[0], &table[hidden], hidden)

				parity.AssertFloat32SlicesWithinULP(t, got, want, embeddingNEONMaxULP)
			}
		})
	})
}
