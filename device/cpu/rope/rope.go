package rope

/*
RotaryEmbedding implements device.{'RoPE'} for the CPU backend.
*/
type RotaryEmbedding struct{}

/*
New constructs a RotaryEmbedding receiver for CPU dispatch.
*/
func New() RotaryEmbedding {
	return RotaryEmbedding{}
}

var Default = New()
