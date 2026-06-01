package interpretability

import cpuinterpretability "github.com/theapemachine/puter/device/cpu/interpretability"

/*
Interpretability implements device.Interpretability on XLA by delegating
to the CPU scalar reference until dedicated device paths land.
*/
type Interpretability struct {
	cpuinterpretability.Interpretability
}

func New() Interpretability {
	return Interpretability{Interpretability: cpuinterpretability.New()}
}
