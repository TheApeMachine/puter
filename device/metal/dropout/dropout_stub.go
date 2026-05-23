//go:build !darwin || !cgo

package dropout

func (dropoutLayer *DropoutLayer) stubHost() {
	dropoutLayer.host.NeedsPlatform()
}
