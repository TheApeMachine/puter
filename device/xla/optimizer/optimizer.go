package optimizer

import cpuoptimizer "github.com/theapemachine/puter/device/cpu/optimizer"

/*
Optimizer implements device.Optimizer on XLA by delegating to the CPU
scalar reference until dedicated device paths land.
*/
type Optimizer struct {
	cpuoptimizer.Stepper
}

func New() Optimizer {
	return Optimizer{Stepper: cpuoptimizer.NewStepper()}
}
