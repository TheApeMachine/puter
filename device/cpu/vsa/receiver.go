package vsa

/*
VSA implements device.VSA for the CPU backend.
*/
type VSA struct{}

/*
New constructs a VSA receiver for CPU dispatch.
*/
func New() VSA {
	return VSA{}
}

var Default = New()
