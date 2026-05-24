package predictive_coding

/*
PredictiveCoding implements device.PredictiveCoding for the CPU backend.
*/
type PredictiveCoding struct{}

/*
New constructs a PredictiveCoding receiver for CPU dispatch.
*/
func New() PredictiveCoding {
	return PredictiveCoding{}
}
