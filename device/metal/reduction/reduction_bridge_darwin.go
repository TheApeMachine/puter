//go:build darwin && cgo

package reduction

/*
#cgo CFLAGS: -x objective-c -fobjc-arc -I${SRCDIR} -I${SRCDIR}/../internal/bridge
#cgo LDFLAGS: -framework Metal -framework Foundation -framework CoreFoundation

#include "native/reduction.m"
#include "native/aggregate.m"
*/
import "C"
