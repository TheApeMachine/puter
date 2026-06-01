package peel

import cpupeel "github.com/theapemachine/puter/device/cpu/peel"

/*
Peel implements device.Peel on XLA by delegating to the CPU reference
until dedicated device paths land.
*/
type Peel struct {
	cpupeel.Peel
}

func New() Peel {
	return Peel{Peel: cpupeel.New()}
}
