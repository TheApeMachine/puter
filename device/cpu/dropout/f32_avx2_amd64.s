#include "textflag.h"

DATA dropSignFlipAVX2<>+0(SB)/4, $0x80000000
DATA dropSignFlipAVX2<>+4(SB)/4, $0x80000000
DATA dropSignFlipAVX2<>+8(SB)/4, $0x80000000
DATA dropSignFlipAVX2<>+12(SB)/4, $0x80000000
GLOBL dropSignFlipAVX2<>(SB), RODATA|NOPTR, $16

// func DropoutFloat32AVX2Asm(
//     dst, src *float32, count int,
//     seedLane *uint32, scale, threshold float32,
// )
TEXT ·DropoutFloat32AVX2Asm(SB), NOSPLIT, $0-40
	MOVQ dst+0(FP), DI
	MOVQ src+8(FP), SI
	MOVQ count+16(FP), CX
	MOVQ seedLane+24(FP), R8
	MOVSS scale+32(FP), X15
	MOVSS threshold+36(FP), X14

	VBROADCASTSS X15, Y15
	VBROADCASTSS X14, Y14
	VBROADCASTSS dropSignFlipAVX2<>(SB), Y13

	XORPS X11, X11
	MOVL  (R8), AX
	MOVD  AX, X11

	TESTQ CX, CX
	JZ   drop_avx2_done

drop_avx2_w8:
	CMPQ CX, $8
	JL   drop_avx2_w4

	MOVQ $8, R9
	LEAQ 0(SP), R10
drop_avx2_gen8:
	VPSLLD $13, X11, X12
	VPXORD X12, X11, X11
	VPSRLD $17, X11, X12
	VPXORD X12, X11, X11
	VPSLLD $5, X11, X12
	VPXORD X12, X11, X11
	VMOVD X11, (R10)
	ADDQ $4, R10
	DECQ R9
	JNZ  drop_avx2_gen8

	VMOVUPS (SI), Y0
	VMOVDQU (SP), Y2
	VPXOR Y13, Y2, Y2
	VPXOR Y13, Y14, Y1
	VPCMPGTD Y1, Y2, Y12
	VANDPS Y12, Y0, Y0
	VMULPS Y15, Y0, Y0
	VMOVUPS Y0, (DI)

	ADDQ $32, DI
	ADDQ $32, SI
	SUBQ $8, CX
	JMP  drop_avx2_w8

drop_avx2_w4:
	CMPQ CX, $4
	JL   drop_avx2_tail

	MOVQ $4, R9
	LEAQ 0(SP), R10
drop_avx2_gen4:
	VPSLLD $13, X11, X12
	VPXORD X12, X11, X11
	VPSRLD $17, X11, X12
	VPXORD X12, X11, X11
	VPSLLD $5, X11, X12
	VPXORD X12, X11, X11
	VMOVD X11, (R10)
	ADDQ $4, R10
	DECQ R9
	JNZ  drop_avx2_gen4

	VMOVUPS (SI), X0
	VMOVDQU (SP), X2
	VPXOR X13, X2, X2
	VPXOR X13, X14, X1
	VPCMPGTD X1, X2, X12
	VANDPS X12, X0, X0
	MULPS X15, X0
	VMOVUPS X0, (DI)

	ADDQ $16, DI
	ADDQ $16, SI
	SUBQ $4, CX
	JMP  drop_avx2_w4

drop_avx2_tail:
	TESTQ CX, CX
	JZ   drop_avx2_done

	MOVQ CX, R9
drop_avx2_scalar:
	VPSLLD $13, X11, X12
	VPXORD X12, X11, X11
	VPSRLD $17, X11, X12
	VPXORD X12, X11, X11
	VPSLLD $5, X11, X12
	VPXORD X12, X11, X11
	VMOVD X11, AX
	MOVD X14, DX
	CMPL AX, DX
	JA   drop_avx2_keep_lane
	XORPS X0, X0
	JMP  drop_avx2_store_lane
drop_avx2_keep_lane:
	MOVSS (SI), X0
	MULSS X15, X0
drop_avx2_store_lane:
	MOVSS X0, (DI)
	ADDQ $4, DI
	ADDQ $4, SI
	DECQ R9
	JNZ  drop_avx2_scalar

drop_avx2_done:
	VMOVD X11, AX
	MOVL AX, (R8)
	RET
