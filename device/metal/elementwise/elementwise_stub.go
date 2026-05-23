//go:build !darwin || !cgo

package elementwise

func (elementwise *Elementwise) stubHost() {
	elementwise.host.NeedsPlatform()
}
