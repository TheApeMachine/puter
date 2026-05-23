// SPDX-License-Identifier: Apache-2.0
// AVX2 uint16 LUT gather via VPMOVZXWD index widen and VGATHERDPS table load.
#include "textflag.h"

DATA lutGatherMaskAVX2<>+0(SB)/4, $0x0000ffff
DATA lutGatherMaskAVX2<>+4(SB)/4, $0x0000ffff
DATA lutGatherMaskAVX2<>+8(SB)/4, $0x0000ffff
DATA lutGatherMaskAVX2<>+12(SB)/4, $0x0000ffff
DATA lutGatherMaskAVX2<>+16(SB)/4, $0x0000ffff
DATA lutGatherMaskAVX2<>+20(SB)/4, $0x0000ffff
DATA lutGatherMaskAVX2<>+24(SB)/4, $0x0000ffff
DATA lutGatherMaskAVX2<>+28(SB)/4, $0x0000ffff
GLOBL lutGatherMaskAVX2<>(SB), 8, $32

// func ApplyF16LUTAVX2(dst, src *uint16, count int, lut *[65536]uint16)
TEXT ·ApplyF16LUTAVX2(SB), NOSPLIT, $0-32
	MOVQ dst+0(FP), DI
	MOVQ src+8(FP), SI
	MOVQ count+16(FP), CX
	MOVQ lut+24(FP), BX

	TESTQ CX, CX
	JZ done

	VPBROADCASTD ·lutGatherMaskAVX2<>(SB), Y15

avx2_loop8:
	CMPQ CX, $8
	JL avx2_loop4

	VMOVDQU (SI), X0
	VPMOVZXWD X0, Y1
	VPSLLD $1, Y1, Y2
	VXORPS Y13, Y13, Y13
	VGATHERDPS Y13, (BX)(Y2*1), Y3
	VPAND Y3, Y15, Y3

	MOVL X3, AX
	MOVW AX, (DI)
	PEXTRD $1, X3, AX
	MOVW AX, 2(DI)
	PEXTRD $2, X3, AX
	MOVW AX, 4(DI)
	PEXTRD $3, X3, AX
	MOVW AX, 6(DI)

	VEXTRACTI128 $1, Y3, X4
	MOVL X4, AX
	MOVW AX, 8(DI)
	PEXTRD $1, X4, AX
	MOVW AX, 10(DI)
	PEXTRD $2, X4, AX
	MOVW AX, 12(DI)
	PEXTRD $3, X4, AX
	MOVW AX, 14(DI)

	ADDQ $16, SI
	ADDQ $16, DI
	SUBQ $8, CX
	JMP avx2_loop8

avx2_loop4:
	CMPQ CX, $4
	JL avx2_scalar_tail

	VMOVDQU (SI), X0
	VPMOVZXWD X0, Y1
	VPSLLD $1, Y1, Y2
	VXORPS X13, X13, X13
	VGATHERDPS X13, (BX)(X2*1), X3
	VPAND X3, X15, X3

	MOVL X3, AX
	MOVW AX, (DI)
	PEXTRD $1, X3, AX
	MOVW AX, 2(DI)
	PEXTRD $2, X3, AX
	MOVW AX, 4(DI)
	PEXTRD $3, X3, AX
	MOVW AX, 6(DI)

	ADDQ $8, SI
	ADDQ $8, DI
	SUBQ $4, CX
	JMP avx2_loop4

avx2_scalar_tail:
	TESTQ CX, CX
	JZ done

avx2_scalar_loop:
	MOVWLZX (SI), R8
	LEAQ (BX)(R8*2), R9
	MOVWLZX (R9), R10
	MOVW R10, (DI)
	ADDQ $2, SI
	ADDQ $2, DI
	DECQ CX
	JNZ avx2_scalar_loop

done:
	RET
