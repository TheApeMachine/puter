package losses

/*
Losses implements device.Losses for the CPU backend.
*/
type Losses struct{}

/*
New constructs a Losses receiver for CPU dispatch.
*/
func New() Losses {
	return Losses{}
}

var Default = New()
