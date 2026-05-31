package optimizer

/*
Stepper implements device.Optimizer for the CPU backend.
*/
type Stepper struct{}

/*
NewStepper constructs a Stepper receiver for CPU dispatch.
*/
func NewStepper() Stepper {
	return Stepper{}
}
