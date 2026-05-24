package active_inference

/*
ActiveInference implements device.ActiveInference for the CPU backend.
*/
type ActiveInference struct{}

/*
New constructs an ActiveInference receiver for CPU dispatch.
*/
func New() ActiveInference {
	return ActiveInference{}
}

var Default = New()
