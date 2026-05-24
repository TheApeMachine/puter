package hawkes

/*
Hawkes implements device.Hawkes for the CPU backend.
*/
type Hawkes struct{}

/*
New constructs a Hawkes receiver for CPU dispatch.
*/
func New() Hawkes {
	return Hawkes{}
}

var Default = New()
