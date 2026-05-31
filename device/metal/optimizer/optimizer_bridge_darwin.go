//go:build darwin && cgo

package optimizer

/*
#cgo CFLAGS: -x objective-c -fobjc-arc -I${SRCDIR} -I${SRCDIR}/../internal/bridge
#cgo LDFLAGS: -framework Metal -framework Foundation -framework CoreFoundation

#include "../internal/runtime/bridge_optimizer_darwin.m"
#include "../internal/runtime/bridge_optimizer_extra_darwin.m"
*/
import "C"
