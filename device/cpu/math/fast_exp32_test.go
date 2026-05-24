package math

import (
	"fmt"
	stdmath "math"
	"testing"
	"unsafe"
)

func fastExp32Inline(x float32) float32 {
	z := x * float32(1.4426950408889634)
	k := int32(z)
	if z < 0 {
		k--
	}
	f := z - float32(k)
	poly := float32(1.0) + f*(float32(0.69314718)+f*(float32(0.24022650)+f*(float32(0.05550410)+f*(float32(0.00961812)+f*float32(0.00133389)))))
	bits := stdmath.Float32bits(poly)
	bits += uint32(k) << 23
	return stdmath.Float32frombits(bits)
}

func TestFastExp32MatchesInline(t *testing.T) {
	x := float32(-2.3492558)
	got := FastExp32(x)
	want := fastExp32Inline(x)
	fmt.Printf("FastExp32=0x%08x inline=0x%08x\n", float32Bits(got), float32Bits(want))
	if got != want {
		t.Fatalf("mismatch")
	}
}

func float32Bits(value float32) uint32 {
	return *(*uint32)(unsafe.Pointer(&value))
}
