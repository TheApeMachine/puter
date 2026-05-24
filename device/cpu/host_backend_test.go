package cpu

import (
	"testing"

	. "github.com/smartystreets/goconvey/convey"
	"github.com/theapemachine/puter/device"
)

func TestNewHostBackend(t *testing.T) {
	Convey("Given NewHostBackend", t, func() {
		hostBackend := NewHostBackend()

		Convey("It should satisfy device.HostBackend", func() {
			var _ device.HostBackend = hostBackend
		})

		Convey("It should count set bits in a host string", func() {
			var counts [8]int
			hostBackend.CountString(&counts, "abc")

			So(counts, ShouldNotResemble, [8]int{})
		})
	})
}
