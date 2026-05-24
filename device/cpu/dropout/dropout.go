package dropout

/*
DropoutLayer implements device.{'Dropout'} for the CPU backend.
*/
type DropoutLayer struct{}

/*
New constructs a DropoutLayer receiver for CPU dispatch.
*/
func New() DropoutLayer {
	return DropoutLayer{}
}

var Default = New()
