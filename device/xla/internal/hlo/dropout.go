package hlo

import (
	"fmt"
	"math"

	"github.com/theapemachine/manifesto/dtype"
	"github.com/theapemachine/manifesto/tensor"
)

func RenderDropout(
	moduleName string,
	elementFormat dtype.DType,
	vectorShape tensor.Shape,
	rate float32,
	seed uint64,
) (string, error) {
	elementType, err := elementToken(elementFormat)

	if err != nil {
		return "", err
	}

	if rate <= 0 {
		return renderDropoutPassthrough(moduleName, elementType, vectorShape)
	}

	keepProb := float32(1.0 - rate)
	scale := float32(1.0 / keepProb)
	threshold := uint32(float64(keepProb) * (1 << 32))
	elementCount := vectorShape.Dims()[0]
	vectorLiteral := reductionInputLiteral(elementType, vectorShape)
	entryLayout := fmt.Sprintf("%s->%s", vectorLiteral, vectorLiteral)

	seedX := uint32(seed)
	_ = uint32(seed >> 32)
	_ = uint32(seed ^ 0x9e3779b9)
	_ = uint32((seed >> 32) ^ 0x6c078965)

	return fmt.Sprintf(`HloModule %s, entry_computation_layout={%s}

%%xorshift {
  state = u32[] parameter(0)
  s1 = u32[] shift-left(state, u32[] constant(13))
  x1 = u32[] xor(state, s1)
  s2 = u32[] shift-right-logical(x1, u32[] constant(17))
  x2 = u32[] xor(x1, s2)
  s3 = u32[] shift-left(x2, u32[] constant(5))
  ROOT next = u32[] xor(x2, s3)
}

%%update {
  state = u32[] parameter(0)
  elem = %s parameter(1)
  advanced = u32[] call(state), to_apply=%%xorshift
  ROOT out = (u32[], u32[]) tuple(advanced, advanced)
}

ENTRY main {
  src = %s parameter(0)
  seed_x = u32[] constant(%d)
  states = u32[%d] scan(seed_x, src), to_apply=%%update
  threshold = u32[] constant(%d)
  scale_c = %s[] constant(%g)
  scale_b = %s broadcast(scale_c), dimensions={}
  keep = pred[%d]{0} compare(states, threshold), direction=LT
  scaled = %s multiply(src, scale_b)
  zero = %s[] constant(0)
  zero_b = %s broadcast(zero), dimensions={}
  ROOT result = %s select(keep, scaled, zero_b)
}
`, moduleName, entryLayout,
		vectorLiteral,
		vectorLiteral,
		seedX, elementCount, threshold,
		elementType, scale, vectorLiteral,
		elementCount, vectorLiteral,
		elementType, vectorLiteral, vectorLiteral), nil
}

func renderDropoutPassthrough(
	moduleName string,
	elementType string,
	vectorShape tensor.Shape,
) (string, error) {
	vectorLiteral := reductionInputLiteral(elementType, vectorShape)
	entryLayout := fmt.Sprintf("%s->%s", vectorLiteral, vectorLiteral)

	return fmt.Sprintf(`HloModule %s, entry_computation_layout={%s}

ENTRY main {
  src = %s parameter(0)
  ROOT result = %s copy(src)
}
`, moduleName, entryLayout, vectorLiteral, vectorLiteral), nil
}

func DropoutThreshold(keepProb float32) uint32 {
	if keepProb <= 0 {
		return 0
	}

	if keepProb >= 1 {
		return math.MaxUint32
	}

	return uint32(float64(keepProb) * (1 << 32))
}
