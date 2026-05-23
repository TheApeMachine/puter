//go:build cuda

package pool

/*
#cgo cuda CFLAGS: -I${SRCDIR} -I${SRCDIR}/../internal/bridge
#cgo cuda LDFLAGS: -lcudart -lnvrtc -lcuda -lpthread

#include "native/adaptive.c"
#include "native/avgpool.c"
#include "native/maxpool.c"
#include "native/pool.c"
*/
import "C"
