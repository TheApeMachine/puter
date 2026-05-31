package resonant

import cpuresonant "github.com/theapemachine/puter/device/cpu/resonant"

/*
Resonant implements device.Resonant on CUDA by delegating to the CPU
scalar reference until dedicated device paths land.
*/
type Resonant struct {
	cpuresonant.Resonant
}

func New() Resonant {
	return Resonant{Resonant: cpuresonant.New()}
}
