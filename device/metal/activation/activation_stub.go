//go:build !darwin || !cgo

package activation

func (activation *Activation) stubHost() {
	activation.host.NeedsPlatform()
}
