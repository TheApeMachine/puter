//go:build cuda

package layernorm

/*
#cgo cuda CFLAGS: -I${SRCDIR} -I${SRCDIR}/../internal/bridge
#cgo cuda LDFLAGS: -lcudart -lnvrtc -lcuda -lpthread

#include "native/layer.c"
#include "native/layernorm.c"
*/
import "C"
