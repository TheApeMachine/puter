//go:build !darwin || !cgo

package losses

func (losses *Losses) stubHost() {
	losses.host.NeedsPlatform()
}
