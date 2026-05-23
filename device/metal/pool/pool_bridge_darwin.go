//go:build darwin && cgo

package pool

/*
#cgo CFLAGS: -x objective-c -fobjc-arc -I${SRCDIR} -I${SRCDIR}/../internal/bridge
#cgo LDFLAGS: -framework Metal -framework Foundation -framework CoreFoundation

#include "native/pool.m"
#include "native/adaptive.m"
#include "native/avgpool.m"
#include "native/maxpool.m"
*/
import "C"
