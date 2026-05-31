//go:build darwin && cgo

package masking

/*
#cgo CFLAGS: -x objective-c -fobjc-arc -I${SRCDIR} -I${SRCDIR}/../internal/bridge -I${SRCDIR}/../attention
#cgo LDFLAGS: -framework Metal -framework Foundation -framework CoreFoundation

#include "../attention/native/masking.m"
*/
import "C"
