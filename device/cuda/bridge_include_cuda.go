//go:build cuda

package cuda

/*
#cgo cuda CFLAGS: -I${SRCDIR}/internal/bridge
#cgo cuda LDFLAGS: -lcuda -lcudart -lnvrtc -lpthread

#include "internal/bridge/status.c"
#include "internal/bridge/context.c"
#include "internal/bridge/runtime.c"
*/
import "C"
