//go:build darwin && cgo

package embedding

/*
#cgo CFLAGS: -x objective-c -fobjc-arc -I${SRCDIR} -I${SRCDIR}/../internal/bridge
#cgo LDFLAGS: -framework Metal -framework Foundation -framework CoreFoundation

#include "native/embedding.m"
#include "native/bag.m"
#include "native/lookup.m"
#include "native/timestep.m"
*/
import "C"
