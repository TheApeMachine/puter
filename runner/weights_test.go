package runner

import (
	"encoding/binary"
	"testing"

	"github.com/smartystreets/goconvey/convey"
	"github.com/theapemachine/manifesto/dtype"
	"github.com/theapemachine/manifesto/ir"
	"github.com/theapemachine/manifesto/tensor"
	"github.com/theapemachine/manifesto/weights"
)

func TestSliceWeightBytes(testingObject *testing.T) {
	convey.Convey("Given a packed 2D projection weight", testingObject, func() {
		shape, err := tensor.NewShape([]int{2, 6})
		convey.So(err, convey.ShouldBeNil)

		node := manifestComputeNode("linear", "projection.linear", ir.OpFused, shape)
		node.SetAttribute("in_features", ir.IntAttribute(2))
		node.SetAttribute("out_features", ir.IntAttribute(2))

		raw := float32SequenceBytes(12)
		meta := weights.TensorMeta{
			DType: "F32",
			Shape: []int64{2, 6},
		}

		convey.Convey("It should slice output rows contiguously", func() {
			sliced, slicedMeta, err := sliceWeightBytes(raw, meta, dtype.Float32, node, astWeightSlice{
				Axis:  "output",
				Start: 0,
			})

			convey.So(err, convey.ShouldBeNil)
			convey.So(slicedMeta.Shape, convey.ShouldResemble, []int64{2, 6})
			convey.So(float32ValuesFromBytes(sliced), convey.ShouldResemble, []float32{
				0, 1, 2, 3, 4, 5,
				6, 7, 8, 9, 10, 11,
			})
		})

		convey.Convey("It should slice input columns per row", func() {
			sliced, slicedMeta, err := sliceWeightBytes(raw, meta, dtype.Float32, node, astWeightSlice{
				Axis:  "input",
				Start: 2,
			})

			convey.So(err, convey.ShouldBeNil)
			convey.So(slicedMeta.Shape, convey.ShouldResemble, []int64{2, 2})
			convey.So(float32ValuesFromBytes(sliced), convey.ShouldResemble, []float32{
				2, 3,
				8, 9,
			})
		})
	})
}

func float32SequenceBytes(count int) []byte {
	bytes := make([]byte, count*4)

	for index := range count {
		binary.LittleEndian.PutUint32(bytes[index*4:], uint32(index)<<23)
	}

	return bytes
}

func float32ValuesFromBytes(bytes []byte) []float32 {
	values := make([]float32, len(bytes)/4)

	for index := range values {
		values[index] = float32(binary.LittleEndian.Uint32(bytes[index*4:]) >> 23)
	}

	return values
}
