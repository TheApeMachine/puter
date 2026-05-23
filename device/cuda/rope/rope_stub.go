//go:build !cuda

package rope

func (rotaryEmbedding *RotaryEmbedding) stubHost() {
	rotaryEmbedding.host.NeedsPlatform()
}
