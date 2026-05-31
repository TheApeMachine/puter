// SSE2 bf16 lane helpers: bit-cast load into .PS, RNE pack back to .H.

DATA bf16RNEOneSSE2<>+0(SB)/4, $0x00000001
DATA bf16RNEOneSSE2<>+4(SB)/4, $0x00000001
DATA bf16RNEOneSSE2<>+8(SB)/4, $0x00000001
DATA bf16RNEOneSSE2<>+12(SB)/4, $0x00000001
GLOBL bf16RNEOneSSE2<>(SB), RODATA|NOPTR, $16

DATA bf16RNEBiasSSE2<>+0(SB)/4, $0x00007fff
DATA bf16RNEBiasSSE2<>+4(SB)/4, $0x00007fff
DATA bf16RNEBiasSSE2<>+8(SB)/4, $0x00007fff
DATA bf16RNEBiasSSE2<>+12(SB)/4, $0x00007fff
GLOBL bf16RNEBiasSSE2<>(SB), RODATA|NOPTR, $16

#define BF16_WIDEN_SSE2_4(src, xLo, xHi) \
	VMOVDQU X2, (src); \
	VPXOR  X3, X3, X3; \
	VPUNPCKLWD X3, X2, xLo; \
	VPUNPCKHWD X3, X2, xHi; \
	VPSLLD $16, xLo, xLo; \
	VPSLLD $16, xHi, xHi

#define BF16_WIDEN_X2_LOW4(dstX) \
	VPXOR X3, X3, X3; \
	VPUNPCKLWD X3, X2, dstX; \
	VPSLLD $16, dstX, dstX

#define BF16_DOT_W4_SSE2(srcLeft, srcRight) \
	VMOVDQU X1, (srcLeft); \
	VMOVDQU X2, (srcRight); \
	VPXOR X3, X3, X3; \
	VPUNPCKLWD X3, X1, X4; \
	VPUNPCKLWD X3, X2, X5; \
	VPSLLD $16, X4, X4; \
	VPSLLD $16, X5, X5; \
	MULPS X4, X5; \
	ADDPS X5, X0; \
	VPUNPCKHWD X3, X1, X4; \
	VPUNPCKHWD X3, X2, X5; \
	VPSLLD $16, X4, X4; \
	VPSLLD $16, X5, X5; \
	MULPS X4, X5; \
	ADDPS X5, X0

#define BF16_LOAD_4H_TO_X4(srcPtr, dstX) \
	VMOVDQU X2, (srcPtr); \
	VPMOVZXWD X2, dstX; \
	VPSLLD $16, dstX, dstX

#define BF16_RNE_X4(xReg) \
	VPSRLD $16, xReg, X3; \
	VPANDD bf16RNEOneSSE2<>(SB), X3, X3; \
	VPADDD bf16RNEBiasSSE2<>(SB), X3, X3; \
	VPADDD X3, xReg, xReg; \
	VPSRLD $16, xReg, xReg

#define PACK_BF16_SSE2_8H(xLo, xHi, dst) \
	BF16_RNE_X4(xLo); \
	BF16_RNE_X4(xHi); \
	MOVL  xLo, AX; \
	MOVW  AX, (dst); \
	PEXTRD $1, xLo, AX; \
	MOVW  AX, 2(dst); \
	MOVL  xHi, AX; \
	MOVW  AX, 4(dst); \
	PEXTRD $1, xHi, AX; \
	MOVW  AX, 6(dst)

#define PACK_BF16_X1_4H(dstReg) \
	VPSRLD $16, X1, X2; \
	VPANDD bf16RNEOneSSE2<>(SB), X2, X2; \
	VPADDD bf16RNEBiasSSE2<>(SB), X2, X2; \
	VPADDD X2, X1, X1; \
	VPSRLD $16, X1, X1; \
	MOVL X1, AX; \
	MOVW AX, (dstReg); \
	PEXTRD $1, X1, AX; \
	MOVW AX, 2(dstReg); \
	PEXTRD $2, X1, AX; \
	MOVW AX, 4(dstReg); \
	PEXTRD $3, X1, AX; \
	MOVW AX, 6(dstReg)

#define PACK_BF16_SCALAR_X2(dstPtr) \
	VPSRLD $16, X2, X3; \
	VPANDD bf16RNEOneSSE2<>(SB), X3, X3; \
	VPADDD bf16RNEBiasSSE2<>(SB), X3, X3; \
	VPADDD X3, X2, X2; \
	VPSRLD $16, X2, X2; \
	MOVL X2, AX; \
	MOVW AX, (dstPtr)

#define PACK_BF16_X1_2H(dstPtr) \
	VPSRLD $16, X1, X2; \
	VPANDD bf16RNEOneSSE2<>(SB), X2, X2; \
	VPADDD bf16RNEBiasSSE2<>(SB), X2, X2; \
	VPADDD X2, X1, X1; \
	VPSRLD $16, X1, X1; \
	MOVL X1, AX; \
	MOVW AX, (dstPtr); \
	PEXTRD $2, X1, AX; \
	MOVW AX, 2(dstPtr)

#define PACK_BF16_ACCUM_X0_4H(dst) \
	BF16_RNE_X4(X0); \
	MOVL X0, AX; \
	MOVW AX, (dst); \
	PEXTRD $1, X0, AX; \
	MOVW AX, 2(dst); \
	PEXTRD $2, X0, AX; \
	MOVW AX, 4(dst); \
	PEXTRD $3, X0, AX; \
	MOVW AX, 6(dst)

#define PACK_BF16_SCALAR_F32_X0_TO(dstFP) \
	VPSRLD $16, X0, X1; \
	VPANDD bf16RNEOneSSE2<>(SB), X1, X1; \
	VPADDD bf16RNEBiasSSE2<>(SB), X1, X1; \
	VPADDD X1, X0, X0; \
	VPSRLD $16, X0, X0; \
	MOVL X0, AX; \
	MOVW AX, dstFP
