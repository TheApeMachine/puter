package math

import cpumath "github.com/theapemachine/puter/device/cpu/math"

/*
Math implements device.Math on XLA by delegating to the CPU scalar
reference until dedicated device paths land.
*/
type Math struct {
	cpumath.Math
}

func New() Math {
	return Math{Math: cpumath.New()}
}
