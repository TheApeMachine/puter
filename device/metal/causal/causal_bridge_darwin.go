//go:build darwin && cgo

package causal

/*
#cgo CFLAGS: -x objective-c -fobjc-arc -I${SRCDIR} -I${SRCDIR}/../internal/bridge
#cgo LDFLAGS: -framework Metal -framework Foundation -framework CoreFoundation

#include "native/causal.m"
#include "native/adjustment.m"
#include "native/dag.m"
#include "native/intervention.m"
#include "native/matrix.m"
*/
import "C"
