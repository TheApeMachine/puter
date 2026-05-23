// SPDX-License-Identifier: Apache-2.0
// AVX-512 uint16 LUT gather via VPMOVZXWD index widen and VGATHERDPS table load.
#include "textflag.h"

DATA lutGatherMaskAVX512<>+0(SB)/4, $0x0000ffff
DATA lutGatherMaskAVX512<>+4(SB)/4, $0x0000ffff
DATA lutGatherMaskAVX512<>+8(SB)/4, $0x0000ffff
DATA lutGatherMaskAVX512<>+12(SB)/4, $0x0000ffff
DATA lutGatherMaskAVX512<>+16(SB)/4, $0x0000ffff
DATA lutGatherMaskAVX512<>+20(SB)/4, $0x0000ffff
DATA lutGatherMaskAVX512<>+24(SB)/4, $0x0000ffff
DATA lutGatherMaskAVX512<>+28(SB)/4, $0x0000ffff
GLOBL lutGatherMaskAVX512<>(SB), 8, $32

// func ApplyF16LUTAVX512(dst, src *uint16, count int, lut *[65536]uint16)
TEXT ·ApplyF16LUTAVX512(SB), NOSPLIT, $0-32
	MOVQ dst+0(FP), DI
	MOVQ src+8(FP), SI
	MOVQ count+16(FP), CX
	MOVQ lut+24(FP), BX

	TESTQ CX, CX
	JZ done

	VPBROADCASTD ·lutGatherMaskAVX512<>(SB), Z15

avx512_loop16:
	CMPQ CX, $16
	JL avx512_loop8

	VMOVDQU16 (SI), Z0
	VPMOVZXWD Z0, Z1
	VPSLLD $1, Z1, Z2
	KXNORW K1, K1, K1
	VGATHERDPS (BX)(Z2*1), K1, Z3
	VPANDD Z3, Z15, Z3

	VPSRLD $0, Z3, Z3
	VMOVDQU16 Z3, (DI)

	ADDQ $32, SI
	ADDQ $32, DI
	SUBQ $16, CX
	JMP avx512_loop16

avx512_loop8:
	CMPQ CX, $8
	JL avx512_loop4

	VMOVDQU (SI), X0
	VPMOVZXWD X0, Y1
	VPSLLD $1, Y1, Y2
	KXNORW K1, K1, K1
	VGATHERDPS (BX)(Y2*1), K1, Y3
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
	JMP avx512_loop8

avx512_loop4:
	CMPQ CX, $4
	JL avx512_scalar_tail

	VMOVDQU (SI), X0
	VPMOVZXWD X0, Y1
	VPSLLD $1, Y1, Y2
	KXNORW K1, K1, K1
	VGATHERDPS (BX)(X2*1), K1, X3
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
	JMP avx512_loop4

avx512_scalar_tail:
	TESTQ CX, CX
	JZ done

avx512_scalar_loop:
	MOVWLZX (SI), R8
	LEAQ (BX)(R8*2), R9
	MOVWLZX (R9), R10
	MOVW R10, (DI)
	ADDQ $2, SI
	ADDQ $2, DI
	DECQ CX
	JNZ avx512_scalar_loop

done:
	RET
