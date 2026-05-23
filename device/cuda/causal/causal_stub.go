//go:build !cuda

package causal

func (causal *Causal) stubHost() {
	causal.host.NeedsPlatform()
}
