//go:build cuda

package convolution

/*
#cgo cuda CFLAGS: -I${SRCDIR} -I${SRCDIR}/../internal/bridge
#cgo cuda LDFLAGS: -lcudart -lnvrtc -lcuda -lpthread

#include "native/conv2d.c"
#include "native/convolution.c"
*/
import "C"
