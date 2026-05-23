//go:build darwin && cgo

package layernorm

/*
#cgo CFLAGS: -x objective-c -fobjc-arc -I${SRCDIR} -I${SRCDIR}/../internal/bridge
#cgo LDFLAGS: -framework Metal -framework Foundation -framework CoreFoundation

#include "native/layernorm.m"
#include "native/layer.m"
*/
import "C"
