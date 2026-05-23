//go:build cuda

package vsa

/*
#cgo cuda CFLAGS: -I${SRCDIR} -I${SRCDIR}/../internal/bridge
#cgo cuda LDFLAGS: -lcudart -lnvrtc -lcuda -lpthread

#include "native/bind.c"
#include "native/bundle.c"
#include "native/permute.c"
#include "native/similarity.c"
#include "native/vsa.c"
*/
import "C"
