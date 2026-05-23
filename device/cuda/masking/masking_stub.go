//go:build !cuda

package masking

func (masking *Masking) stubHost() {
	masking.host.NeedsPlatform()
}
