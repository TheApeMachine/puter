package attention

/*
Attention implements device.Attention for the CPU backend.
*/
type Attention struct{}

/*
New constructs an Attention receiver for CPU dispatch.
*/
func New() Attention {
	return Attention{}
}

var Default = New()
