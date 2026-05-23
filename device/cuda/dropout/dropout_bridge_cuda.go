//go:build cuda

package dropout

/*
#cgo cuda CFLAGS: -I${SRCDIR} -I${SRCDIR}/../internal/bridge
#cgo cuda LDFLAGS: -lcudart -lnvrtc -lcuda -lpthread

#include "native/dropout.c"
#include "native/mask.c"
*/
import "C"
