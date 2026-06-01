package math

/*
Math implements device.Math for the CPU backend.
*/
type Math struct{}

/*
New constructs a Math receiver for CPU dispatch.
*/
func New() Math {
	return Math{}
}
