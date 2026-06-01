package shape

import cpushape "github.com/theapemachine/puter/device/cpu/shape"

/*
Shape implements device.Shape on CUDA by delegating to the CPU scalar
reference until dedicated device paths land.
*/
type Shape struct {
	cpushape.Shape
}

func New() Shape {
	return Shape{Shape: cpushape.New()}
}
