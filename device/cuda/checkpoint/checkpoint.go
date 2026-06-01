package checkpoint

import cpucheckpoint "github.com/theapemachine/puter/device/cpu/checkpoint"

/*
Checkpoint implements device.Checkpoint on CUDA by delegating to the CPU
scalar reference until dedicated device paths land.
*/
type Checkpoint struct {
	cpucheckpoint.Checkpoint
}

func New() Checkpoint {
	return Checkpoint{Checkpoint: cpucheckpoint.New()}
}
