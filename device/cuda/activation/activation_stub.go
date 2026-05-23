//go:build !cuda

package activation

func (activation *Activation) stubHost() {
	activation.host.NeedsPlatform()
}
