package cpu

import (
	"context"
	"testing"

	. "github.com/smartystreets/goconvey/convey"
	"github.com/theapemachine/qpool"
)

func TestBackendAllocateAligned(t *testing.T) {
	Convey("Given a CPU backend", t, func() {
		workerPool := qpool.NewQ(context.Background(), 1, 1, qpool.NewConfig())
		defer workerPool.Close()

		backend, err := NewBackend(context.Background(), workerPool)
		So(err, ShouldBeNil)
		defer backend.Close()

		Convey("It should return 64-byte aligned workspace memory", func() {
			pointer, err := backend.allocateAligned(128)
			So(err, ShouldBeNil)
			So(pointer, ShouldNotBeNil)
			So(uintptr(pointer)%workspaceAlign, ShouldEqual, 0)
		})
	})
}
