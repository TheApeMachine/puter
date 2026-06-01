package shape

/*
Shape implements device.Shape for the CPU backend.
*/
type Shape struct{}

/*
New constructs a Shape receiver for CPU dispatch.
*/
func New() Shape {
	return Shape{}
}
