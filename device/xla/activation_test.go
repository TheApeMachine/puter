package xla

import (
	"testing"

	"github.com/smartystreets/goconvey/convey"
	"github.com/theapemachine/manifesto/dtype"
	"github.com/theapemachine/manifesto/tensor"
)

func TestUnaryActivationLoweringProgramKey(t *testing.T) {
	convey.Convey("Given a unary activation lowering", t, func() {
		builder := NewDefaultBuilder("gpu")
		inputShape := mustShape(t, 1024)
		context := LoweringContext{
			InputDTypes: []dtype.DType{dtype.Float16},
			InputShapes: []tensor.Shape{inputShape},
			OutputDType: dtype.Float16,
			OutputShape: inputShape,
		}

		programKey, err := builder.ProgramKeyFor("exp", context, nil, nil)

		convey.Convey("It should build a stable program key", func() {
			convey.So(err, convey.ShouldBeNil)
			convey.So(programKey.Operation, convey.ShouldEqual, "exp")
			convey.So(programKey.Target, convey.ShouldEqual, "gpu")
		})
	})
}

func TestBinaryElementwiseLoweringProgramKey(t *testing.T) {
	convey.Convey("Given a binary elementwise lowering", t, func() {
		builder := NewDefaultBuilder("gpu")
		leftShape := mustShape(t, 4, 1)
		rightShape := mustShape(t, 3)
		broadcastShape, err := BroadcastShape(leftShape, rightShape)
		convey.So(err, convey.ShouldBeNil)

		context := LoweringContext{
			InputDTypes: []dtype.DType{dtype.BFloat16, dtype.BFloat16},
			InputShapes: []tensor.Shape{leftShape, rightShape},
			OutputDType: dtype.BFloat16,
			OutputShape: broadcastShape,
		}

		programKey, err := builder.ProgramKeyFor("add", context, nil, nil)

		convey.Convey("It should include broadcast shapes in the key", func() {
			convey.So(err, convey.ShouldBeNil)
			convey.So(programKey.Operation, convey.ShouldEqual, "add")
			convey.So(len(programKey.Shapes), convey.ShouldEqual, 4)
		})
	})
}
