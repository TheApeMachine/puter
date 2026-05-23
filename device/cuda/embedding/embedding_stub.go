//go:build !cuda

package embedding

func (embedding *Embedding) stubHost() {
	embedding.host.NeedsPlatform()
}
