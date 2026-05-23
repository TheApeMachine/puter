//go:build cuda

package active_inference

/*
#cgo cuda CFLAGS: -I${SRCDIR} -I${SRCDIR}/../internal/bridge
#cgo cuda LDFLAGS: -lcudart -lnvrtc -lcuda -lpthread

#include "native/active_inference.c"
#include "native/belief.c"
#include "native/free_energy.c"
*/
import "C"
