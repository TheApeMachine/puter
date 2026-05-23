//go:build !cuda

package elementwise

func (elementwise *Elementwise) stubHost() {
	elementwise.host.NeedsPlatform()
}
