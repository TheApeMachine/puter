//go:build !cuda

package losses

func (losses *Losses) stubHost() {
	losses.host.NeedsPlatform()
}
