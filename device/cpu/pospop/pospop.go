package pospop

/*
PosPop implements device.{'PosPop'} for the CPU backend.
*/
type PosPop struct{}

/*
New constructs a PosPop receiver for CPU dispatch.
*/
func New() PosPop {
	return PosPop{}
}

var Default = New()
