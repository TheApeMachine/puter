package dropout

import "github.com/theapemachine/puter/device"

/*
DropoutConfig carries the drop rate and deterministic seed for
inverted dropout scaling.
*/
type DropoutConfig = device.DropoutConfig

func DefaultDropoutConfig() DropoutConfig {
	return DropoutConfig{Rate: 0.1, Seed: 0xc0ffee}
}
