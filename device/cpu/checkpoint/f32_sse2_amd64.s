#include "textflag.h"

// func CheckpointEncodeFloat32DataSSE2Asm(dst *byte, src *float32, count int)
TEXT ·CheckpointEncodeFloat32DataSSE2Asm(SB), NOSPLIT, $0-24
	MOVQ dst+0(FP), DI
	MOVQ src+8(FP), SI
	MOVQ count+16(FP), CX

ckpt_enc_sse2_w4:
	CMPQ CX, $4
	JL   ckpt_enc_sse2_tail

	VMOVUPS (SI), X0
	VMOVUPS X0, (DI)

	ADDQ $16, SI
	ADDQ $16, DI
	SUBQ $4, CX
	JMP  ckpt_enc_sse2_w4

ckpt_enc_sse2_tail:
	TESTQ CX, CX
	JZ   ckpt_enc_sse2_done

ckpt_enc_sse2_scalar:
	MOVSS (SI), X0
	MOVSS X0, (DI)
	ADDQ  $4, SI
	ADDQ  $4, DI
	DECQ  CX
	JNZ  ckpt_enc_sse2_scalar

ckpt_enc_sse2_done:
	RET

// func CheckpointDecodeFloat32DataSSE2Asm(dst *float32, src *byte, count int)
TEXT ·CheckpointDecodeFloat32DataSSE2Asm(SB), NOSPLIT, $0-24
	MOVQ dst+0(FP), DI
	MOVQ src+8(FP), SI
	MOVQ count+16(FP), CX

ckpt_dec_sse2_w4:
	CMPQ CX, $4
	JL   ckpt_dec_sse2_tail

	VMOVUPS (SI), X0
	VMOVUPS X0, (DI)

	ADDQ $16, SI
	ADDQ $16, DI
	SUBQ $4, CX
	JMP  ckpt_dec_sse2_w4

ckpt_dec_sse2_tail:
	TESTQ CX, CX
	JZ   ckpt_dec_sse2_done

ckpt_dec_sse2_scalar:
	MOVSS (SI), X0
	MOVSS X0, (DI)
	ADDQ  $4, SI
	ADDQ  $4, DI
	DECQ  CX
	JNZ  ckpt_dec_sse2_scalar

ckpt_dec_sse2_done:
	RET
