#include "textflag.h"

// CPUID.07H.01H:EDX[23] reports AVX512-FP16 support.
TEXT ·hasAVX512FP16Asm(SB), NOSPLIT, $0-1
	MOVL  $7, AX
	MOVL  $1, CX
	CPUID
	TESTL $(1<<23), EDX
	JE    has_avx512_fp16_no
	MOVB  $1, ret+0(FP)
	RET

has_avx512_fp16_no:
	MOVB  $0, ret+0(FP)
	RET
