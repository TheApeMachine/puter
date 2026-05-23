//go:build darwin && cgo

package physics

/*
#cgo CFLAGS: -x objective-c -fobjc-arc -I${SRCDIR} -I${SRCDIR}/../internal/bridge
#cgo LDFLAGS: -framework Metal -framework Foundation -framework CoreFoundation

#include "native/physics.m"
#include "native/differential.m"
#include "native/spectral.m"
*/
import "C"
