package execution

import (
	"encoding/binary"
	"encoding/json"
	"math"
	"testing"

	"github.com/smartystreets/goconvey/convey"
	"github.com/theapemachine/manifesto/dtype"
	"github.com/theapemachine/manifesto/tensor"
	"github.com/theapemachine/manifesto/types"
)

func TestResidentStoreLookupAndTranspose(t *testing.T) {
	convey.Convey("Given a rank-2 resident weight", t, func() {
		memory := tensor.NewHostBackend()
		defer memory.Close()

		shape, err := tensor.NewShape([]int{2, 3})

		convey.So(err, convey.ShouldBeNil)

		rawBytes := make([]byte, 24)
		for index := range 6 {
			binary.LittleEndian.PutUint32(rawBytes[index*4:], math.Float32bits(float32(index)+1))
		}

		resident, err := memory.Upload(shape, dtype.Float32, rawBytes)

		convey.So(err, convey.ShouldBeNil)

		store := NewResidentStore(memory)
		store.RegisterTensor("linear.weight", types.Token{
			Kind:      types.KindTensor,
			Name:      "linear.weight",
			Shape:     []int64{2, 3},
			Precision: dtype.Float32,
		}, resident)

		convey.Convey("Lookup returns the registered tensor", func() {
			got, err := store.Lookup("linear.weight")

			convey.So(err, convey.ShouldBeNil)
			convey.So(got.Shape().Dims(), convey.ShouldResemble, []int{2, 3})
		})

		convey.Convey("LookupTransposed swaps the matrix dimensions", func() {
			transposed, err := store.LookupTransposed("linear.weight")

			convey.So(err, convey.ShouldBeNil)
			convey.So(transposed.Shape().Dims(), convey.ShouldResemble, []int{3, 2})
		})

		convey.Convey("LookupSlice extracts an output-axis range", func() {
			sliced, err := store.LookupSlice("linear.weight", "output", 1, 2)

			convey.So(err, convey.ShouldBeNil)
			convey.So(sliced.Shape().Dims(), convey.ShouldResemble, []int{1, 3})
		})
	})
}

func TestLoadSafetensorsArchive(t *testing.T) {
	convey.Convey("Given a minimal safetensors archive", t, func() {
		memory := tensor.NewHostBackend()
		defer memory.Close()

		values := []float32{1, 2, 3, 4}
		dataSection := make([]byte, len(values)*4)

		for index, value := range values {
			binary.LittleEndian.PutUint32(dataSection[index*4:], math.Float32bits(value))
		}

		header := map[string]any{
			"model.weight": map[string]any{
				"dtype":        "F32",
				"shape":        []int64{2, 2},
				"data_offsets": []int64{0, int64(len(dataSection))},
			},
		}

		headerBytes, err := json.Marshal(header)

		convey.So(err, convey.ShouldBeNil)

		archive := make([]byte, 8+len(headerBytes)+len(dataSection))
		binary.LittleEndian.PutUint64(archive[:8], uint64(len(headerBytes)))
		copy(archive[8:], headerBytes)
		copy(archive[8+len(headerBytes):], dataSection)

		bundle, err := LoadSafetensorsArchive(archive, memory)

		convey.So(err, convey.ShouldBeNil)
		convey.So(bundle, convey.ShouldNotBeNil)
		convey.So(bundle.Parser, convey.ShouldNotBeNil)
		convey.So(bundle.Store, convey.ShouldNotBeNil)

		convey.Convey("The store resolves the uploaded tensor", func() {
			resident, err := bundle.Store.Lookup("model.weight")

			convey.So(err, convey.ShouldBeNil)
			convey.So(resident.Shape().Dims(), convey.ShouldResemble, []int{2, 2})
		})
	})
}
