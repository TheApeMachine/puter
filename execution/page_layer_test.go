package execution

import (
	"testing"

	"github.com/smartystreets/goconvey/convey"
	"github.com/theapemachine/manifesto/dtype"
	"github.com/theapemachine/manifesto/tensor"
)

func TestLayerStorageView(testingObject *testing.T) {
	convey.Convey("Given stacked layer KV storage", testingObject, func() {
		shape, err := tensor.NewShape([]int{2, 4, 8, 3, 5})

		convey.So(err, convey.ShouldBeNil)

		storage, err := tensor.NewHostBackend().Upload(
			shape,
			dtype.Float32,
			make([]byte, shape.Len()*4),
		)

		convey.So(err, convey.ShouldBeNil)

		convey.Convey("It should return one layer slice without copying", func() {
			layer, err := layerStorageView(storage, 1)

			convey.So(err, convey.ShouldBeNil)
			convey.So(layer.Shape().Dims(), convey.ShouldResemble, []int{4, 8, 3, 5})
		})
	})
}
