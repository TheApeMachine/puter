//go:build !darwin || !cgo

package causal

func (causal *Causal) stubHost() {
	causal.host.NeedsPlatform()
}
