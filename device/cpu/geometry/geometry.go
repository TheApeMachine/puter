package geometry

import (
	"github.com/theapemachine/puter/device"
)

/*
Geometry implements device.Geometry for the CPU backend.
*/
type Geometry struct{}

/*
New constructs a Geometry receiver for CPU dispatch.
*/
func New() Geometry {
	return Geometry{}
}

var Default = New()

var _ device.Geometry = Geometry{}
