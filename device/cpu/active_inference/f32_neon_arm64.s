// SPDX-License-Identifier: Apache-2.0
// NEON active inference float32 kernels with shared log helpers.
#include "textflag.h"

#define VFADD_S4(m, n, d)   WORD $(0x4E20D400 | ((m) << 16) | ((n) << 5) | (d))
#define VFSUB_S4(m, n, d)   WORD $(0x4EA0D400 | ((m) << 16) | ((n) << 5) | (d))
#define VFMUL_S4(m, n, d)   WORD $(0x6E20DC00 | ((m) << 16) | ((n) << 5) | (d))
#define VFMLA_S4(m, n, d)   WORD $(0x4E20CC00 | ((m) << 16) | ((n) << 5) | (d))
#define VFDIV_S4(m, n, d)   WORD $(0x6E20FC00 | ((m) << 16) | ((n) << 5) | (d))
#define VFMLA_D2(m, n, d)   WORD $(0x4E60CC00 | ((m) << 16) | ((n) << 5) | (d))
#define FCVTL_2D(n, d)      WORD $(0x0E617800 | ((n) << 5) | (d))
#define FCVTL2_2D(n, d)     WORD $(0x4E617800 | ((n) << 5) | (d))
#define VFADD_D2(m, n, d)   WORD $(0x4E60D400 | ((m) << 16) | ((n) << 5) | (d))
#define FADDP_D(n, d)       WORD $(0x7E70D800 | ((n) << 5) | (d))
#define VUSHR_S4_BY23(n, d) WORD $(0x6F290400 | ((n) << 5) | (d))
#define VISUB_S4(m, n, d)   WORD $(0x6EA08400 | ((m) << 16) | ((n) << 5) | (d))
#define VAND_B16(m, n, d)   WORD $(0x4E201C00 | ((m) << 16) | ((n) << 5) | (d))
#define VORR_B16(m, n, d)   WORD $(0x4EA01C00 | ((m) << 16) | ((n) << 5) | (d))
#define VSCVTF_S4(n, d)     WORD $(0x4E21D800 | ((n) << 5) | (d))
#define VMAX_S4(m, n, d)    WORD $(0x4E20F400 | ((m) << 16) | ((n) << 5) | (d))
#define VMOV_B16(src, dst)  WORD $(0x4EA01C00 | ((src) << 16) | ((src) << 5) | (dst))

DATA aiEps<>+0(SB)/4, $0x358637bd
GLOBL aiEps<>(SB), RODATA|NOPTR, $4

DATA aiLogC<>+0(SB)/4, $0.6931471805599453
DATA aiLogC<>+4(SB)/4, $1.0
DATA aiLogC<>+8(SB)/4, $0.09090909
DATA aiLogC<>+12(SB)/4, $0.11111111
DATA aiLogC<>+16(SB)/4, $0.14285715
DATA aiLogC<>+20(SB)/4, $0.20000000
DATA aiLogC<>+24(SB)/4, $0.33333334
DATA aiLogC<>+28(SB)/4, $2.0
GLOBL aiLogC<>(SB), RODATA|NOPTR, $32

DATA aiOneBits<>+0(SB)/4, $0x3F800000
GLOBL aiOneBits<>(SB), RODATA|NOPTR, $4

#define AI_LOAD_LOG_CONSTS \
    MOVD $aiLogC<>(SB), R3 ;\
    FMOVS 0(R3), F16 ;\
    FMOVS 4(R3), F17 ;\
    FMOVS 8(R3), F18 ;\
    FMOVS 12(R3), F19 ;\
    FMOVS 16(R3), F20 ;\
    FMOVS 20(R3), F21 ;\
    FMOVS 24(R3), F22 ;\
    FMOVS 28(R3), F23 ;\
    MOVD $0x007FFFFF, R6 ;\
    VMOV R6, V24.S[0] ;\
    VDUP V24.S[0], V24.S4 ;\
    MOVD $0x3F800000, R6 ;\
    VMOV R6, V25.S[0] ;\
    VDUP V25.S[0], V25.S4 ;\
    MOVD $127, R6 ;\
    VMOV R6, V26.S[0] ;\
    VDUP V26.S[0], V26.S4

#define AI_NEON_LOG(in, out) \
    VUSHR_S4_BY23(in, 1) ;\
    VISUB_S4(26, 1, 1) ;\
    VAND_B16(24, in, 2) ;\
    VORR_B16(25, 2, 2) ;\
    VSCVTF_S4(1, 1) ;\
    VFSUB_S4(17, 2, 3) ;\
    VFADD_S4(17, 2, 4) ;\
    VFDIV_S4(4, 3, 5) ;\
    VFMUL_S4(5, 5, 6) ;\
    VMOV_B16(18, 7) ;\
    VMOV_B16(19, 8) ; VFMLA_S4(6, 7, 8) ;\
    VMOV_B16(20, 7) ; VFMLA_S4(6, 8, 7) ;\
    VMOV_B16(21, 8) ; VFMLA_S4(6, 7, 8) ;\
    VMOV_B16(22, 7) ; VFMLA_S4(6, 8, 7) ;\
    VMOV_B16(17, 8) ; VFMLA_S4(6, 7, 8) ;\
    VFMUL_S4(5, 8, 8) ;\
    VFMUL_S4(23, 8, 8) ;\
    VFMLA_S4(16, 1, out)

#define AI_F32X4_TO_F64_ADD(src, acc) \
    FCVTL_2D(src, 8) ;\
    FCVTL2_2D(src, 9) ;\
    VFADD_D2(8, acc, acc) ;\
    VFADD_D2(9, acc, acc)

// func PrecisionWeightFloat32NEONAsm(errors, precision, output *float32, count int)
TEXT ·PrecisionWeightFloat32NEONAsm(SB), NOSPLIT, $0-32
	MOVD errors+0(FP), R0
	MOVD precision+8(FP), R1
	MOVD output+16(FP), R2
	MOVD count+24(FP), R3
	CBZ  R3, ai_pw_done

ai_pw_loop4:
	CMP  $4, R3
	BLT  ai_pw_scalar

	VLD1 (R0), [V0.S4]
	VLD1 (R1), [V1.S4]
	VFMUL_S4(1, 0, 0)
	VST1 [V0.S4], (R2)
	ADD  $16, R0
	ADD  $16, R1
	ADD  $16, R2
	SUB  $4, R3
	B    ai_pw_loop4

ai_pw_scalar:
	CBZ  R3, ai_pw_done

ai_pw_scalar_loop:
	FMOVS (R0), F0
	FMOVS (R1), F1
	FMULS F1, F0, F0
	FMOVS F0, (R2)
	ADD  $4, R0
	ADD  $4, R1
	ADD  $4, R2
	SUB  $1, R3
	CBNZ R3, ai_pw_scalar_loop

ai_pw_done:
	RET

// func BeliefUpdateFloat32NEONAsm(likelihood, prior, output *float32, count int)
TEXT ·BeliefUpdateFloat32NEONAsm(SB), NOSPLIT, $0-32
	MOVD likelihood+0(FP), R0
	MOVD prior+8(FP), R1
	MOVD output+16(FP), R2
	MOVD count+24(FP), R3
	CBZ  R3, ai_bu_done

	VEOR V16.B16, V16.B16, V16.B16

ai_bu_loop4:
	CMP  $4, R3
	BLT  ai_bu_tail

	VLD1 (R0), [V0.S4]
	VLD1 (R1), [V1.S4]
	VFMUL_S4(1, 0, 0)
	VST1 [V0.S4], (R2)
	AI_F32X4_TO_F64_ADD(0, 16)
	ADD  $16, R0
	ADD  $16, R1
	ADD  $16, R2
	SUB  $4, R3
	B    ai_bu_loop4

ai_bu_tail:
	CBZ  R3, ai_bu_reduce

ai_bu_tail_loop:
	FMOVS (R0), F0
	FMOVS (R1), F1
	FMULS F1, F0, F0
	FMOVS F0, (R2)
	FCVTSD F0, F0
	FADDD F0, F16, F16
	ADD  $4, R0
	ADD  $4, R1
	ADD  $4, R2
	SUB  $1, R3
	CBNZ R3, ai_bu_tail_loop

ai_bu_reduce:
	FADDP_D(16, 0)
	FMOVS $0.0, F11
	FCMPS F0, F11
	BEQ  ai_bu_done

	FMOVS aiOneBits<>(SB), F3
	FDIVS F0, F3, F3
	VDUP V3.S[0], V3.S4

	MOVD output+16(FP), R2
	MOVD count+24(FP), R3

ai_bu_scale_loop4:
	CMP  $4, R3
	BLT  ai_bu_scale_scalar

	VLD1 (R2), [V0.S4]
	VFMUL_S4(3, 0, 0)
	VST1 [V0.S4], (R2)
	ADD  $16, R2
	SUB  $4, R3
	B    ai_bu_scale_loop4

ai_bu_scale_scalar:
	CBZ  R3, ai_bu_done

ai_bu_scale_scalar_loop:
	FMOVS (R2), F0
	FMULS F3, F0, F0
	FMOVS F0, (R2)
	ADD  $4, R2
	SUB  $1, R3
	CBNZ R3, ai_bu_scale_scalar_loop

ai_bu_done:
	RET

// func FreeEnergyFloat32NEONAsm(likelihood, posterior, prior *float32, count int) float32
TEXT ·FreeEnergyFloat32NEONAsm(SB), NOSPLIT, $0-32
	MOVD likelihood+0(FP), R0
	MOVD posterior+8(FP), R1
	MOVD prior+16(FP), R2
	MOVD count+24(FP), R3

	VEOR V16.B16, V16.B16, V16.B16
	VEOR V17.B16, V17.B16, V17.B16
	FMOVS aiEps<>(SB), F30
	VDUP V30.S[0], V30.S4
	AI_LOAD_LOG_CONSTS
	CBZ  R3, ai_fe_store

ai_fe_loop4:
	CMP  $4, R3
	BLT  ai_fe_tail

	VLD1 (R0), [V3.S4]
	VLD1 (R1), [V4.S4]
	VLD1 (R2), [V5.S4]
	VMAX_S4(30, 3, 3)
	VMAX_S4(30, 4, 4)
	VMAX_S4(30, 5, 5)

	VMOV V3.B16, V0.B16
	AI_NEON_LOG(0, 6)
	VMOV V4.B16, V0.B16
	AI_NEON_LOG(0, 7)
	VMOV V5.B16, V0.B16
	AI_NEON_LOG(0, 8)

	VEOR V9.B16, V9.B16, V9.B16
	VFSUB_S4(6, 9, 10)
	VFMUL_S4(4, 10, 10)
	AI_F32X4_TO_F64_ADD(10, 16)

	VFSUB_S4(8, 7, 11)
	VFMUL_S4(4, 11, 11)
	AI_F32X4_TO_F64_ADD(11, 17)

	ADD  $16, R0
	ADD  $16, R1
	ADD  $16, R2
	SUB  $4, R3
	B    ai_fe_loop4

ai_fe_tail:
	CBZ  R3, ai_fe_store

ai_fe_tail_loop:
	FMOVS (R0), F0
	FMOVS (R1), F1
	FMOVS (R2), F2
	FMAXS F30, F0, F0
	FMAXS F30, F1, F1
	FMAXS F30, F2, F2
	VDUP V0.S[0], V0.S4
	AI_NEON_LOG(0, 6)
	VDUP V1.S[0], V1.S4
	AI_NEON_LOG(1, 7)
	VDUP V2.S[0], V2.S4
	AI_NEON_LOG(2, 8)
	FMOVS F6, F10
	FMOVS $-1.0, F14
	FMULS F14, F10, F10
	FMULS F1, F10, F10
	FCVTSD F10, F10
	FADDD F10, F16, F16
	FMOVS F7, F11
	FMOVS F8, F12
	FSUBS F12, F11, F11
	FMULS F1, F11, F11
	FCVTSD F11, F11
	FADDD F11, F17, F17
	ADD  $4, R0
	ADD  $4, R1
	ADD  $4, R2
	SUB  $1, R3
	CBNZ R3, ai_fe_tail_loop

ai_fe_store:
	VFADD_D2(17, 16, 16)
	FADDP_D(16, 0)
	FCVTDS F0, F0
	FMOVS F0, ret+32(FP)
	RET

// func ExpectedFreeEnergyFloat32NEONAsm(
//     predictedObs, preferredObs, predictedState *float32,
//     obsCount, stateCount int,
// ) float32
TEXT ·ExpectedFreeEnergyFloat32NEONAsm(SB), NOSPLIT, $0-40
	MOVD predictedObs+0(FP), R0
	MOVD preferredObs+8(FP), R1
	MOVD predictedState+16(FP), R2
	MOVD obsCount+24(FP), R3
	MOVD stateCount+32(FP), R4

	VEOR V16.B16, V16.B16, V16.B16
	VEOR V17.B16, V17.B16, V17.B16
	FMOVS aiEps<>(SB), F30
	VDUP V30.S[0], V30.S4
	AI_LOAD_LOG_CONSTS

ai_efe_obs_loop4:
	CMP  $4, R3
	BLT  ai_efe_obs_done

	VLD1 (R0), [V3.S4]
	VLD1 (R1), [V4.S4]
	VMAX_S4(30, 3, 3)
	VMAX_S4(30, 4, 4)

	VMOV V3.B16, V0.B16
	AI_NEON_LOG(0, 6)
	VMOV V4.B16, V0.B16
	AI_NEON_LOG(0, 7)

	VFSUB_S4(7, 6, 10)
	VFMUL_S4(3, 10, 10)
	AI_F32X4_TO_F64_ADD(10, 16)

	ADD  $16, R0
	ADD  $16, R1
	SUB  $4, R3
	B    ai_efe_obs_loop4

ai_efe_obs_done:
	MOVD predictedState+16(FP), R2
	MOVD stateCount+32(FP), R4

ai_efe_state_loop4:
	CMP  $4, R4
	BLT  ai_efe_store

	VLD1 (R2), [V3.S4]
	VMAX_S4(30, 3, 3)
	VMOV V3.B16, V0.B16
	AI_NEON_LOG(0, 6)

	VEOR V9.B16, V9.B16, V9.B16
	VFSUB_S4(6, 9, 10)
	VFMUL_S4(3, 10, 10)
	AI_F32X4_TO_F64_ADD(10, 17)

	ADD  $16, R2
	SUB  $4, R4
	B    ai_efe_state_loop4

ai_efe_store:
	VFADD_D2(17, 16, 16)
	FADDP_D(16, 0)
	FCVTDS F0, F0
	FMOVS F0, ret+36(FP)
	RET
