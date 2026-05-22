// SPDX-License-Identifier: Apache-2.0
// NEON int32 tokenizer pack: contiguous copy (tokenizer_pack_int32).
#include "textflag.h"

// func TokenizerPackInt32NEONAsm(dst, src *int32, count int)
TEXT ·TokenizerPackInt32NEONAsm(SB), NOSPLIT, $0-24
	MOVD dst+0(FP), R0
	MOVD src+8(FP), R1
	MOVD count+16(FP), R2

tok_pack_loop16:
	CMP  $16, R2
	BLT  tok_pack_loop4

	VLD1 (R1), [V0.S4, V1.S4, V2.S4, V3.S4]
	VST1 [V0.S4, V1.S4, V2.S4, V3.S4], (R0)

	ADD  $64, R0
	ADD  $64, R1
	SUB  $16, R2
	B    tok_pack_loop16

tok_pack_loop4:
	CMP  $4, R2
	BLT  tok_pack_scalar_tail

	VLD1 (R1), [V0.S4]
	VST1 [V0.S4], (R0)

	ADD  $16, R0
	ADD  $16, R1
	SUB  $4, R2
	B    tok_pack_loop4

tok_pack_scalar_tail:
	CBZ  R2, tok_pack_done

tok_pack_scalar_loop:
	MOVW (R1), R3
	MOVW R3, (R0)
	ADD  $4, R0
	ADD  $4, R1
	SUB  $1, R2
	CBNZ R2, tok_pack_scalar_loop

tok_pack_done:
	RET
