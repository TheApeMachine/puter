//go:build cuda

package pool

/*
#cgo cuda CFLAGS: -I${SRCDIR} -I${SRCDIR}/../internal/bridge
#cgo cuda LDFLAGS: -lcudart -lnvrtc -lcuda -lpthread

#include "native/pool.c"
#include "native/avgpool.c"
#include "native/maxpool.c"
*/
import "C"
