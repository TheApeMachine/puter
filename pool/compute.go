package pool

import (
	"github.com/theapemachine/manifesto/tensor"
	"github.com/theapemachine/puter/device"
)

/*
ComputeDevice returns the preferred compute backend in device precedence order.
*/
func (devicePool *Pool) ComputeDevice() (device.Backend, DeviceID, error) {
	for _, deviceID := range devicePool.DeviceIDs() {
		backend, err := devicePool.Device(deviceID)

		if err != nil {
			continue
		}

		return backend, deviceID, nil
	}

	return nil, DeviceID{}, ErrDeviceNotFound
}

/*
ComputeMemory returns tensor storage on the same device as ComputeDevice.
*/
func (devicePool *Pool) ComputeMemory() (tensor.Backend, DeviceID, error) {
	for _, deviceID := range devicePool.DeviceIDs() {
		backend, err := devicePool.Device(deviceID)

		if err != nil {
			continue
		}

		memory, ok := backend.(tensor.Backend)

		if !ok {
			continue
		}

		return memory, deviceID, nil
	}

	return devicePool.MemoryBackend()
}
