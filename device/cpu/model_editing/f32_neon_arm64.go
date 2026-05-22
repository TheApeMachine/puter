//go:build arm64

package model_editing

//go:noescape
func WeightGraftAddFloat32NEONAsm(weights, injection *float32, count int)

/*
WeightGraftAddFloat32NEON adds injection in-place: weights[i] += injection[i].
*/
func WeightGraftAddFloat32NEON(weights, injection *float32, count int) {
	if count == 0 {
		return
	}

	WeightGraftAddFloat32NEONAsm(weights, injection, count)
}
