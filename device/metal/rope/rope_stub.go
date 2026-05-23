//go:build !darwin || !cgo

package rope

func (rotaryEmbedding *RotaryEmbedding) stubHost() {
	rotaryEmbedding.host.NeedsPlatform()
}
