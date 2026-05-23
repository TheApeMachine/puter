//go:build !darwin || !cgo

package embedding

func (embedding *Embedding) stubHost() {
	embedding.host.NeedsPlatform()
}
