//go:build amd64

package model_editing

//go:noescape
func WeightGraftAddFloat32AVX2Asm(weights, injection *float32, count int)

func WeightGraftAddFloat32AVX2(weights, injection *float32, count int) {
	if count == 0 {
		return
	}

	WeightGraftAddFloat32AVX2Asm(weights, injection, count)
}

func weightGraftAddFloat32AVX2(weights, injection []float32, count int) {
	WeightGraftAddFloat32AVX2(&weights[0], &injection[0], count)
}
