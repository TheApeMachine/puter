// SPDX-License-Identifier: Apache-2.0
// NEON checkpoint f32 payload: little-endian float32 bytes copy (encode/decode data).
#include "textflag.h"

// func CheckpointEncodeFloat32DataNEONAsm(dst *byte, src *float32, count int)
TEXT ·CheckpointEncodeFloat32DataNEONAsm(SB), NOSPLIT, $0-24
	MOVD dst+0(FP), R0
	MOVD src+8(FP), R1
	MOVD count+16(FP), R2

ckpt_enc_loop16:
	CMP  $16, R2
	BLT  ckpt_enc_loop4

	VLD1 (R1), [V0.S4, V1.S4, V2.S4, V3.S4]
	VST1 [V0.S4, V1.S4, V2.S4, V3.S4], (R0)

	ADD  $64, R0
	ADD  $64, R1
	SUB  $16, R2
	B    ckpt_enc_loop16

ckpt_enc_loop4:
	CMP  $4, R2
	BLT  ckpt_enc_scalar_tail

	VLD1 (R1), [V0.S4]
	VST1 [V0.S4], (R0)

	ADD  $16, R0
	ADD  $16, R1
	SUB  $4, R2
	B    ckpt_enc_loop4

ckpt_enc_scalar_tail:
	CBZ  R2, ckpt_enc_done

ckpt_enc_scalar_loop:
	FMOVS (R1), F0
	FMOVS F0, (R0)
	ADD  $4, R0
	ADD  $4, R1
	SUB  $1, R2
	CBNZ R2, ckpt_enc_scalar_loop

ckpt_enc_done:
	RET

// func CheckpointDecodeFloat32DataNEONAsm(dst *float32, src *byte, count int)
TEXT ·CheckpointDecodeFloat32DataNEONAsm(SB), NOSPLIT, $0-24
	MOVD dst+0(FP), R0
	MOVD src+8(FP), R1
	MOVD count+16(FP), R2

ckpt_dec_loop16:
	CMP  $16, R2
	BLT  ckpt_dec_loop4

	VLD1 (R1), [V0.S4, V1.S4, V2.S4, V3.S4]
	VST1 [V0.S4, V1.S4, V2.S4, V3.S4], (R0)

	ADD  $64, R0
	ADD  $64, R1
	SUB  $16, R2
	B    ckpt_dec_loop16

ckpt_dec_loop4:
	CMP  $4, R2
	BLT  ckpt_dec_scalar_tail

	VLD1 (R1), [V0.S4]
	VST1 [V0.S4], (R0)

	ADD  $16, R0
	ADD  $16, R1
	SUB  $4, R2
	B    ckpt_dec_loop4

ckpt_dec_scalar_tail:
	CBZ  R2, ckpt_dec_done

ckpt_dec_scalar_loop:
	FMOVS (R1), F0
	FMOVS F0, (R0)
	ADD  $4, R0
	ADD  $4, R1
	SUB  $1, R2
	CBNZ R2, ckpt_dec_scalar_loop

ckpt_dec_done:
	RET
