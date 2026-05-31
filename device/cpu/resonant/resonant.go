package resonant

/*
Resonant implements device.Resonant for the CPU backend.
*/
type Resonant struct{}

/*
New constructs a Resonant receiver for CPU dispatch.
*/
func New() Resonant {
	return Resonant{}
}

var Default = New()
