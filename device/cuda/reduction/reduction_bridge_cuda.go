//go:build cuda

package reduction

/*
#cgo cuda CFLAGS: -I${SRCDIR} -I${SRCDIR}/../internal/bridge
#cgo cuda LDFLAGS: -lcudart -lnvrtc -lcuda -lpthread

#include "native/aggregate.c"
#include "native/reduction.c"
*/
import "C"
