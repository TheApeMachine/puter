package physics

/*
Physics implements device.Physics for the CPU backend.
*/
type Physics struct{}

/*
New constructs a Physics receiver for CPU dispatch.
*/
func New() Physics {
	return Physics{}
}

var Default = New()
