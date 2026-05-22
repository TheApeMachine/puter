//go:build arm64

package sampling

import "testing"

func TestDebugSoftmaxCompare(t *testing.T) {
	logits := randomSamplingLogits(7, 0x3610)
	want := make([]float32, 7)
	got := make([]float32, 7)

	SamplingSoftmaxRowGeneric(logits, want, 0.85)
	SamplingSoftmaxRowFloat32NEONAsm(&logits[0], &got[0], 0.85, 7)

	var wantSum, gotSum float32
	for index := range want {
		wantSum += want[index]
		gotSum += got[index]
	}

	t.Logf("logits=%v", logits)
	t.Logf("want=%v sum=%v", want, wantSum)
	t.Logf("got=%v sum=%v", got, gotSum)
}
