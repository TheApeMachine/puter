package pool

import (
	"context"
	"testing"

	. "github.com/smartystreets/goconvey/convey"
	"github.com/theapemachine/manifesto/tensor"
	"github.com/theapemachine/qpool"
)

func TestParseDeviceID(test *testing.T) {
	Convey("Given device id strings", test, func() {
		Convey("It should accept host aliases", func() {
			deviceID, err := ParseDeviceID("cpu")
			So(err, ShouldBeNil)
			So(deviceID, ShouldResemble, DeviceID{Location: tensor.Host, Index: 0})
		})

		Convey("It should parse indexed gpu ids", func() {
			deviceID, err := ParseDeviceID("metal:2")
			So(err, ShouldBeNil)
			So(deviceID, ShouldResemble, DeviceID{Location: tensor.Metal, Index: 2})
		})
	})
}

func TestNew(test *testing.T) {
	Convey("Given a device pool", test, func() {
		devicePool, err := New(context.Background(), nil)
		So(err, ShouldBeNil)

		defer func() {
			So(devicePool.Close(), ShouldBeNil)
		}()

		Convey("It should always discover host", func() {
			hostDevice, err := devicePool.Device(DeviceID{Location: tensor.Host, Index: 0})
			So(err, ShouldBeNil)
			So(hostDevice, ShouldNotBeNil)
		})

		Convey("It should expose host preprocessing", func() {
			hostBackend := devicePool.HostBackend()
			So(hostBackend, ShouldNotBeNil)
		})

		Convey("It should reject unknown device ids", func() {
			_, err := devicePool.Device(DeviceID{Location: tensor.CUDA, Index: 9})
			So(err, ShouldNotBeNil)
			So(err.Error(), ShouldContainSubstring, "device not found")
		})
	})
}

func TestNew_Heterogeneous(test *testing.T) {
	Convey("Given discovery on a host with optional accelerators", test, func() {
		devicePool, err := New(context.Background(), qpool.NewQ(context.Background(), 1, 1, nil))
		So(err, ShouldBeNil)

		defer func() {
			So(devicePool.Close(), ShouldBeNil)
		}()

		Convey("It should keep host resident while metal memory is present", func() {
			hostDevice, err := devicePool.Device(DeviceID{Location: tensor.Host, Index: 0})
			So(err, ShouldBeNil)
			So(hostDevice, ShouldNotBeNil)

			metalDevice, err := devicePool.Device(DeviceID{Location: tensor.Metal, Index: 0})
			So(err, ShouldBeNil)
			So(metalDevice, ShouldNotBeNil)
		})
	})
}
