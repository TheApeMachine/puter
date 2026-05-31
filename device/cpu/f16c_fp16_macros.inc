// F16C fp16 lane helpers (SSE4.1/AVX2): VCVTPH2PS widen, VCVTPS2PH narrow.
// Used where native AVX512-FP16 (.PH) is unavailable.

#define VCVTPS2PH_Y0_X2 WORD $0xC4E3; WORD $0x7D1D; BYTE $0xD8; BYTE $0x00
#define VCVTPS2PH_X0_X2 WORD $0xC4E3; WORD $0x7D1D; BYTE $0xD0; BYTE $0x00

#define FP16_WIDEN_SSE2_4(src, dst) \
	VMOVDQU X2, (src); \
	VCVTPH2PS X2, dst

#define FP16_NARROW_SSE2_4(src, dst) \
	MOVAPS src, X0; \
	VCVTPS2PH_X0_X2; \
	VMOVDQU X2, (dst)

#define FP16_NARROW_SCALAR_F32_X0_TO(dstFP) \
	VCVTPS2PH_X0_X2; \
	MOVL X2, AX; \
	MOVW AX, dstFP

#define FP16_WIDEN_AVX2_8H(baseReg, dstY) \
	VMOVDQU X1, (baseReg); \
	VCVTPH2PS X1, dstY; \
	VPSRLDQ $8, X1, X2; \
	VCVTPH2PS X2, X5; \
	VINSERTF128 $1, X5, dstY, dstY

#define FP16_WIDEN_AVX2_4H(baseReg, dstY) \
	VMOVDQU X2, (baseReg); \
	VCVTPH2PS X2, dstY

#define FP16_NARROW_AVX2_Y8(dstReg) \
	VCVTPS2PH_Y0_X2; \
	VMOVDQU X2, (dstReg); \
	VEXTRACTF128 $1, Y0, X0; \
	VCVTPS2PH_X0_X2; \
	VMOVDQU X2, 8(dstReg)

#define FP16_NARROW_AVX2_Y4(dstReg) \
	VCVTPS2PH_Y0_X2; \
	VMOVDQU X2, (dstReg)

#define FP16_WIDEN_AVX512_8H(baseReg, dstY) \
	VMOVDQU X2, (baseReg); \
	VCVTPH2PS X2, dstY; \
	VPSRLDQ $8, X2, X3; \
	VCVTPH2PS X3, Y4; \
	VEXTRACTI128 $0, Y4, X4; \
	VINSERTF128 $1, X4, dstY, dstY

#define FP16_WIDEN_AVX512_4H(baseReg, dstY) \
	VMOVDQU X2, (baseReg); \
	VCVTPH2PS X2, dstY

#define FP16_NARROW_AVX512_Y8(dstReg) \
	VCVTPS2PH_Y0_X2; \
	VMOVDQU X2, (dstReg); \
	VEXTRACTF128 $1, Y0, X0; \
	VCVTPS2PH_X0_X2; \
	VMOVDQU X2, 8(dstReg)

#define FP16_NARROW_AVX512_Y4(dstReg) \
	VCVTPS2PH_Y0_X2; \
	VMOVDQU X2, (dstReg)

#define FP16_LOAD_4H_TO_X4(srcPtr, dstX) \
	VMOVDQU X2, (srcPtr); \
	VCVTPH2PS X2, dstX

#define FP16_NARROW_X0_TO_4H(dst) \
	VCVTPS2PH_X0_X2; \
	VMOVDQU X2, (dst)

#define FP16_NARROW_X1_TO_4H(dstPtr) \
	MOVAPS X1, X0; \
	VCVTPS2PH_X0_X2; \
	VMOVDQU X2, (dstPtr)

#define FP16_NARROW_X1_TO_2H(dstPtr) \
	MOVAPS X1, X0; \
	VCVTPS2PH_X0_X2; \
	MOVL X2, AX; \
	MOVW AX, (dstPtr); \
	PEXTRD $2, X2, AX; \
	MOVW AX, 2(dstPtr)
