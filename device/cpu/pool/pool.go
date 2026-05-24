package pool

/*
Pool implements device.Pool for the CPU backend.
*/
type Pool struct{}

/*
New constructs a Pool receiver for CPU dispatch.
*/
func New() Pool {
	return Pool{}
}

var Default = New()
