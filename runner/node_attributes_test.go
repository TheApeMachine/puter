package runner

import (
	"encoding/binary"
	"testing"

	"github.com/smartystreets/goconvey/convey"
	"github.com/theapemachine/manifesto/dtype"
	"github.com/theapemachine/manifesto/ir"
	"github.com/theapemachine/manifesto/tensor"
	"github.com/theapemachine/puter/kernels"
)

func TestNodeOptionalFloatAttribute(testingObject *testing.T) {
	convey.Convey("Given integer IR attributes", testingObject, func() {
		shape, err := tensor.NewShape([]int{1})
		convey.So(err, convey.ShouldBeNil)

		node := manifestComputeNode("rope_q_0", "positional.rope", ir.OpFused, shape)
		node.SetAttribute("base", ir.IntAttribute(500000))

		value, ok := nodeOptionalFloatAttribute(node, "base")

		convey.Convey("It should parse them as float values", func() {
			convey.So(ok, convey.ShouldBeTrue)
			convey.So(value, convey.ShouldEqual, 500000)
		})
	})
}

func TestAppendKernelAttributesSlice(testingObject *testing.T) {
	convey.Convey("Given a slice node with dim, start, and end attributes", testingObject, func() {
		backend := tensor.NewHostBackend()
		defer backend.Close()

		shape, err := tensor.NewShape([]int{1})
		convey.So(err, convey.ShouldBeNil)

		input, err := backend.Upload(shape, dtype.Float32, make([]byte, 4))
		convey.So(err, convey.ShouldBeNil)
		defer input.Close()

		output, err := backend.Upload(shape, dtype.Float32, make([]byte, 4))
		convey.So(err, convey.ShouldBeNil)
		defer output.Close()

		node := manifestComputeNode("slice", "shape.slice", ir.OpFused, shape)
		node.SetAttribute("dim", ir.IntAttribute(1))
		node.SetAttribute("start", ir.IntAttribute(1024))
		node.SetAttribute("end", ir.IntAttribute(0))

		args, err := appendKernelAttributes(backend, node, "slice", []tensor.Tensor{input, output})

		convey.Convey("It should append the scalar attributes before the output", func() {
			convey.So(err, convey.ShouldBeNil)
			convey.So(kernelSignature(args), convey.ShouldResemble, kernels.Signature{
				Layout: tensor.LayoutDense,
				Inputs: []dtype.DType{
					dtype.Float32,
					dtype.Int32,
					dtype.Int32,
					dtype.Int32,
				},
				Outputs: []dtype.DType{dtype.Float32},
			})
			convey.So(int32ScalarValue(testingObject, backend, args[1]), convey.ShouldEqual, 1)
			convey.So(int32ScalarValue(testingObject, backend, args[2]), convey.ShouldEqual, 1024)
			convey.So(int32ScalarValue(testingObject, backend, args[3]), convey.ShouldEqual, 0)
		})
	})
}

func TestAppendKernelAttributesTranspose(testingObject *testing.T) {
	convey.Convey("Given a transpose node with two swap dimensions", testingObject, func() {
		backend := tensor.NewHostBackend()
		defer backend.Close()

		shape, err := tensor.NewShape([]int{1, 64, 64, 128})
		convey.So(err, convey.ShouldBeNil)

		input, err := backend.Upload(shape, dtype.BFloat16, make([]byte, shape.Len()*2))
		convey.So(err, convey.ShouldBeNil)
		defer input.Close()

		output, err := backend.Upload(shape, dtype.BFloat16, make([]byte, shape.Len()*2))
		convey.So(err, convey.ShouldBeNil)
		defer output.Close()

		inputNode := manifestComputeNode("packed_grid", "input", ir.OpInput, shape)
		node := manifestComputeNode("vae.unpack.pack_t23", "shape.transpose", ir.OpFused, shape)
		node.AddInput(inputNode)
		node.SetAttribute("dim0", ir.IntAttribute(2))
		node.SetAttribute("dim1", ir.IntAttribute(3))

		args, err := appendKernelAttributes(backend, node, "transpose", []tensor.Tensor{input, output})

		convey.Convey("It should append a permutation tensor before the output", func() {
			convey.So(err, convey.ShouldBeNil)
			convey.So(kernelSignature(args), convey.ShouldResemble, kernels.Signature{
				Layout:  tensor.LayoutDense,
				Inputs:  []dtype.DType{dtype.BFloat16, dtype.Int32},
				Outputs: []dtype.DType{dtype.BFloat16},
			})
			convey.So(int32VectorValue(testingObject, backend, args[1]), convey.ShouldResemble, []int32{0, 1, 3, 2})
		})
	})
}

func int32ScalarValue(
	testingObject *testing.T,
	backend *tensor.HostBackend,
	value tensor.Tensor,
) int32 {
	actualDType, bytes, err := backend.Download(value)
	if err != nil {
		testingObject.Fatalf("download int32 scalar: %v", err)
	}

	if actualDType != dtype.Int32 || len(bytes) != 4 {
		testingObject.Fatalf("invalid int32 scalar dtype=%s bytes=%d", actualDType.Name(), len(bytes))
	}

	return int32(binary.LittleEndian.Uint32(bytes))
}

func int32VectorValue(
	testingObject *testing.T,
	backend *tensor.HostBackend,
	value tensor.Tensor,
) []int32 {
	actualDType, bytes, err := backend.Download(value)
	if err != nil {
		testingObject.Fatalf("download int32 vector: %v", err)
	}

	if actualDType != dtype.Int32 || len(bytes)%4 != 0 {
		testingObject.Fatalf("invalid int32 vector dtype=%s bytes=%d", actualDType.Name(), len(bytes))
	}

	values := make([]int32, len(bytes)/4)

	for index := range values {
		values[index] = int32(binary.LittleEndian.Uint32(bytes[index*4:]))
	}

	return values
}
