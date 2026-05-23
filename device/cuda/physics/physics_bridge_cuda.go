//go:build cuda

package physics

/*
#cgo cuda CFLAGS: -I${SRCDIR} -I${SRCDIR}/../internal/bridge
#cgo cuda LDFLAGS: -lcudart -lnvrtc -lcuda -lpthread

#include "native/differential.c"
#include "native/physics.c"
#include "native/spectral.c"
*/
import "C"
