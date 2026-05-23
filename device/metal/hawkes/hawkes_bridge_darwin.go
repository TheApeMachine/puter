//go:build darwin && cgo

package hawkes

/*
#cgo CFLAGS: -x objective-c -fobjc-arc -I${SRCDIR} -I${SRCDIR}/../internal/bridge
#cgo LDFLAGS: -framework Metal -framework Foundation -framework CoreFoundation

#include "native/hawkes.m"
#include "native/intensity.m"
#include "native/kernel.m"
#include "native/likelihood.m"
#include "native/markov.m"
*/
import "C"
