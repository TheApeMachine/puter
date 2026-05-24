//go:build arm64

package random

import (
	"testing"

	"github.com/smartystreets/goconvey/convey"
)

/*
NEON Philox parity: the 4-lane NEON kernel must produce, for any
(key0, key1, ctrBase), the exact same 16 uint32s as four sequential
calls to the scalar Philox4x32 with counters ctrBase, ctrBase+1,
ctrBase+2, ctrBase+3, concatenated in that order.

This is bitwise parity, not ULP tolerance. The kernel uses the same
algorithm; any divergence is a kernel bug.
*/

func scalarPhilox4Counters(key0, key1 uint32, ctrBase uint64) [16]uint32 {
	var out [16]uint32

	for lane := uint64(0); lane < 4; lane++ {
		state := NewPhiloxState(uint64(key1)<<32|uint64(key0), ctrBase+lane)
		w0, w1, w2, w3 := Philox4x32(state)
		out[lane*4+0] = w0
		out[lane*4+1] = w1
		out[lane*4+2] = w2
		out[lane*4+3] = w3
	}

	return out
}

func TestPhilox4x32x4NEONBitwiseParity(t *testing.T) {
	convey.Convey("Given Philox4x32x4NEON across representative inputs", t, func() {
		cases := []struct {
			name             string
			key0, key1       uint32
			ctrBase          uint64
		}{
			{"AllZero", 0, 0, 0},
			{"SmallSeed", 0xDEADBEEF, 0xCAFEBABE, 0x1000},
			{"LargeCtr", 0xA4093822, 0x299F31D0, 0x0000_0000_FFFF_0000},
			{"KAT-mixed", 0xa4093822, 0x299f31d0, 0x85a308d3_243f6a88},
		}

		for _, testCase := range cases {
			testCase := testCase

			convey.Convey("It matches the scalar reference bitwise for "+testCase.name, func() {
				var neonOut [16]uint32
				Philox4x32x4NEON(&neonOut[0], testCase.key0, testCase.key1, testCase.ctrBase)

				scalarOut := scalarPhilox4Counters(testCase.key0, testCase.key1, testCase.ctrBase)

				for index := 0; index < 16; index++ {
					convey.So(neonOut[index], convey.ShouldEqual, scalarOut[index])
				}
			})
		}
	})
}
