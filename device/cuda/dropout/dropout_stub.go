//go:build !cuda

package dropout

func (dropoutLayer *DropoutLayer) stubHost() {
	dropoutLayer.host.NeedsPlatform()
}
