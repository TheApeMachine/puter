//go:build darwin && cgo

package vsa

/*
#cgo CFLAGS: -x objective-c -fobjc-arc -I${SRCDIR} -I${SRCDIR}/../internal/bridge
#cgo LDFLAGS: -framework Metal -framework Foundation -framework CoreFoundation

#include "native/vsa.m"
#include "native/bind.m"
#include "native/bundle.m"
#include "native/permute.m"
#include "native/similarity.m"
*/
import "C"
