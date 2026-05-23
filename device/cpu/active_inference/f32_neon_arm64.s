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

DATA aiEps<>+0(SB)/4, $0x2b8cbccc
GLOBL aiEps<>(SB), RODATA|NOPTR, $4

DATA aiLogC<>+0(SB)/4, $0.6931471805599453
DATA aiLogC<>+4(SB)/4, $1.0
DATA aiLogC<>+8(SB)/4, $0.09090909
DATA aiLogC<>+12(SB)/4, $0.11111111
DATA aiLogC<>+16(SB)/4, $0.14285715
DATA aiLogC<>+20(SB)/4, $0.20000000
DATA aiLogC<>+24(SB)/4, $0.33333334
DATA aiLogC<>+28(SB)/4, $2.0
GLOBL aiLogC<>(SB), 8, $32

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

#define AI_F32X4_TO_F64_ADD_F14(src) \
    FCVTL_2D(src, 8) ;\
    FCVTL2_2D(src, 9) ;\
    FADDD F8, F14, F14 ;\
    FADDD F9, F14, F14

// func BeliefUpdateFloat32NEONAsm(likelihood, prior, output *float32, count int)
TEXT ·BeliefUpdateFloat32NEONAsm(SB), NOSPLIT, $0-32
	MOVD likelihood+0(FP), R0
	MOVD prior+8(FP), R1
	MOVD output+16(FP), R2
	MOVD count+24(FP), R3
	CBZ  R3, ai_bu_done

	FMOVD $0, F14

ai_bu_loop4:
	CMP  $4, R3
	BLT  ai_bu_tail

	VLD1 (R0), [V0.S4]
	VLD1 (R1), [V1.S4]
	VFMUL_S4(1, 0, 0)
	VST1 [V0.S4], (R2)
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
	ADD  $4, R0
	ADD  $4, R1
	ADD  $4, R2
	SUB  $1, R3
	CBNZ R3, ai_bu_tail_loop

ai_bu_reduce:
	MOVD output+16(FP), R2
	MOVD count+24(FP), R3
	FMOVD $0, F14

ai_bu_sum_loop:
	CBZ  R3, ai_bu_sum_done

	FMOVS (R2), F0
	FCVTSD F0, F0
	FADDD F0, F14, F14
	ADD  $4, R2
	SUB  $1, R3
	B    ai_bu_sum_loop

ai_bu_sum_done:
	FMOVD $0, F15
	FCMPD F14, F15
	BEQ  ai_bu_done

	FMOVD $1.0, F3
	FDIVD F14, F3, F3
	FCVTDS F3, F3
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

#define AI_F32X4_TO_F64_ADD_CE(src) \
	FCVTL_2D(src, 8) ;\
	FCVTL2_2D(src, 9) ;\
	FADDD F8, F14, F14 ;\
	FADDD F9, F14, F14 ;\
	FADDD F10, F14, F14 ;\
	FADDD F11, F14, F14

#define AI_F32X4_TO_F64_ADD_KL(src) \
	FCVTL_2D(src, 8) ;\
	FCVTL2_2D(src, 9) ;\
	FADDD F8, F15, F15 ;\
	FADDD F9, F15, F15 ;\
	FADDD F10, F15, F15 ;\
	FADDD F11, F15, F15

#define AI_F32_LANE0_F64_ADD_CE \
	FCVTSD F6, F8 ;\
	FADDD F8, F14, F14

#define AI_F32_LANE0_F64_ADD_KL \
	FCVTSD F6, F8 ;\
	FADDD F8, F15, F15

#define AI_F32_LANE0_V7_F64_ADD_KL \
	FCVTSD F7, F8 ;\
	FADDD F8, F15, F15

#define AI_FE_STORE_RESULT \
	FADDD F15, F14, F0 ;\
	FCVTDS F0, F0 ;\
	FMOVS F0, ret+32(FP)

#define AI_EFE_STORE_RESULT \
	FADDD F15, F14, F0 ;\
	FCVTDS F0, F0 ;\
	FMOVS F0, ret+40(FP)

#define AI_LOAD_LOG_MASKS \
	MOVD $0x007FFFFF, R6 ;\
	VMOV R6, V24.S[0] ;\
	VDUP V24.S[0], V24.S4 ;\
	MOVD $0x3F800000, R6 ;\
	VMOV R6, V25.S[0] ;\
	VDUP V25.S[0], V25.S4 ;\
	MOVD $127, R6 ;\
	VMOV R6, V26.S[0] ;\
	VDUP V26.S[0], V26.S4

#define AI_RELOAD_LOG_POLY \
	MOVD $aiLogC<>(SB), R14 ;\
	FMOVS  0(R14), F16 ;\
	VDUP V16.S[0], V16.S4 ;\
	FMOVS  4(R14), F17 ;\
	VDUP V17.S[0], V17.S4 ;\
	FMOVS  8(R14), F18 ;\
	VDUP V18.S[0], V18.S4 ;\
	FMOVS 12(R14), F19 ;\
	VDUP V19.S[0], V19.S4 ;\
	FMOVS 16(R14), F20 ;\
	VDUP V20.S[0], V20.S4 ;\
	FMOVS 20(R14), F21 ;\
	VDUP V21.S[0], V21.S4 ;\
	FMOVS 24(R14), F22 ;\
	VDUP V22.S[0], V22.S4 ;\
	FMOVS 28(R14), F23 ;\
	VDUP V23.S[0], V23.S4

#define AI_LOAD_LOG_CONSTS \
	AI_RELOAD_LOG_POLY ;\
	AI_LOAD_LOG_MASKS

// Natural log on four clamped lanes in `in`, result in `out`. Staging in V9 preserves V20 log coefficients.
#define AI_NEON_LOG4(in, out) \
	VMOV_B16(in, 9) ;\
	VUSHR_S4_BY23(9, 1) ;\
	VISUB_S4(26, 1, 1) ;\
	VAND_B16(24, 9, 2) ;\
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
	VFMLA_S4(16, 1, 8) ;\
	VMOV_B16(8, out)

#define AI_FE_BLOCK4 \
	VLD1 (R10), [V3.S4] ;\
	VLD1 (R11), [V4.S4] ;\
	VLD1 (R12), [V5.S4] ;\
	VMOV_B16(4, 27) ;\
	VMAX_S4(31, 3, 3) ;\
	VMAX_S4(31, 4, 4) ;\
	VMAX_S4(31, 5, 5) ;\
	AI_LOAD_LOG_CONSTS ;\
	AI_NEON_LOG4(3, 10) ;\
	AI_LOAD_LOG_CONSTS ;\
	AI_NEON_LOG4(4, 11) ;\
	AI_LOAD_LOG_CONSTS ;\
	AI_NEON_LOG4(5, 12) ;\
	VEOR V30.B16, V30.B16, V30.B16 ;\
	VFSUB_S4(10, 30, 6) ;\
	VFMUL_S4(27, 6, 6) ;\
	AI_F32X4_TO_F64_ADD_CE(6) ;\
	VFSUB_S4(12, 11, 7) ;\
	VFMUL_S4(27, 7, 7) ;\
	AI_F32X4_TO_F64_ADD_KL(7) ;\
	ADD  $16, R10 ;\
	ADD  $16, R11 ;\
	ADD  $16, R12 ;\
	SUB  $4, R3

#define AI_FE_TAIL_ONE \
	FMOVS (R10), F3 ;\
	VDUP V3.S[0], V3.S4 ;\
	FMOVS (R11), F4 ;\
	VDUP V4.S[0], V4.S4 ;\
	FMOVS (R12), F5 ;\
	VDUP V5.S[0], V5.S4 ;\
	VMOV_B16(4, 27) ;\
	VMAX_S4(31, 3, 3) ;\
	VMAX_S4(31, 4, 4) ;\
	VMAX_S4(31, 5, 5) ;\
	AI_LOAD_LOG_CONSTS ;\
	AI_NEON_LOG4(3, 10) ;\
	AI_LOAD_LOG_CONSTS ;\
	AI_NEON_LOG4(4, 11) ;\
	AI_LOAD_LOG_CONSTS ;\
	AI_NEON_LOG4(5, 12) ;\
	VEOR V30.B16, V30.B16, V30.B16 ;\
	VFSUB_S4(10, 30, 6) ;\
	VFMUL_S4(27, 6, 6) ;\
	AI_F32_LANE0_F64_ADD_CE ;\
	VFSUB_S4(12, 11, 7) ;\
	VFMUL_S4(27, 7, 7) ;\
	AI_F32_LANE0_V7_F64_ADD_KL ;\
	ADD  $4, R10 ;\
	ADD  $4, R11 ;\
	ADD  $4, R12 ;\
	SUB  $1, R3

#define AI_EFE_OBS_BLOCK4 \
	VLD1 (R0), [V3.S4] ;\
	VLD1 (R1), [V4.S4] ;\
	VMOV_B16(3, 27) ;\
	VMAX_S4(31, 3, 3) ;\
	VMAX_S4(31, 4, 4) ;\
	AI_LOAD_LOG_CONSTS ;\
	AI_NEON_LOG4(3, 10) ;\
	AI_LOAD_LOG_CONSTS ;\
	AI_NEON_LOG4(4, 11) ;\
	VFSUB_S4(11, 10, 6) ;\
	VFMUL_S4(27, 6, 6) ;\
	AI_F32X4_TO_F64_ADD_CE(6) ;\
	ADD  $16, R0 ;\
	ADD  $16, R1 ;\
	SUB  $4, R3

#define AI_EFE_OBS_TAIL_ONE \
	FMOVS (R0), F3 ;\
	VDUP V3.S[0], V3.S4 ;\
	FMOVS (R1), F4 ;\
	VDUP V4.S[0], V4.S4 ;\
	VMOV_B16(3, 27) ;\
	VMAX_S4(31, 3, 3) ;\
	VMAX_S4(31, 4, 4) ;\
	AI_LOAD_LOG_CONSTS ;\
	AI_NEON_LOG4(3, 10) ;\
	AI_LOAD_LOG_CONSTS ;\
	AI_NEON_LOG4(4, 11) ;\
	VFSUB_S4(11, 10, 6) ;\
	VFMUL_S4(27, 6, 6) ;\
	AI_F32_LANE0_F64_ADD_CE ;\
	ADD  $4, R0 ;\
	ADD  $4, R1 ;\
	SUB  $1, R3

#define AI_EFE_STATE_BLOCK4 \
	VLD1 (R2), [V3.S4] ;\
	VMOV_B16(3, 27) ;\
	VMAX_S4(31, 3, 3) ;\
	AI_LOAD_LOG_CONSTS ;\
	AI_NEON_LOG4(3, 10) ;\
	VEOR V30.B16, V30.B16, V30.B16 ;\
	VFSUB_S4(10, 30, 6) ;\
	VFMUL_S4(27, 6, 6) ;\
	AI_F32X4_TO_F64_ADD_KL(6) ;\
	ADD  $16, R2 ;\
	SUB  $4, R4

#define AI_EFE_STATE_TAIL_ONE \
	FMOVS (R2), F3 ;\
	VDUP V3.S[0], V3.S4 ;\
	VMOV_B16(3, 27) ;\
	VMAX_S4(31, 3, 3) ;\
	AI_LOAD_LOG_CONSTS ;\
	AI_NEON_LOG4(3, 10) ;\
	VEOR V30.B16, V30.B16, V30.B16 ;\
	VFSUB_S4(10, 30, 6) ;\
	VFMUL_S4(27, 6, 6) ;\
	AI_F32_LANE0_F64_ADD_KL ;\
	ADD  $4, R2 ;\
	SUB  $1, R4

// AiNeonLogProbeAsm writes natural log of max(x, eps) into out.
TEXT ·AiNeonLogProbeAsm(SB), NOSPLIT, $0-16
	MOVD x+0(FP), R10
	MOVD out+8(FP), R11
	FMOVS R10, F3
	VDUP V3.S[0], V3.S4
	MOVD $0x2b8cbccc, R17
	VMOV R17, V31.S[0]
	VDUP V31.S[0], V31.S4
	VMAX_S4(31, 3, 3)
	AI_LOAD_LOG_CONSTS
	AI_NEON_LOG4(3, 10)
	FMOVS F10, (R11)
	RET

// AiNeonFeLogLikeProbeAsm runs tail clamp+log on likelihood and writes logLike lane0.
TEXT ·AiNeonFeLogLikeProbeAsm(SB), NOSPLIT, $0-16
	MOVD likelihood+0(FP), R10
	MOVD out+8(FP), R11
	FMOVS (R10), F3
	VDUP V3.S[0], V3.S4
	MOVD $0x2b8cbccc, R17
	VMOV R17, V31.S[0]
	VDUP V31.S[0], V31.S4
	VMAX_S4(31, 3, 3)
	AI_LOAD_LOG_CONSTS
	AI_NEON_LOG4(3, 10)
	FMOVS F10, (R11)
	RET

// AiNeonFeTailProbeAsm runs one tail element and writes ceAcc and klAcc.
TEXT ·AiNeonFeTailProbeAsm(SB), NOSPLIT, $0-40
	MOVD likelihood+0(FP), R10
	MOVD posterior+8(FP), R11
	MOVD prior+16(FP), R12
	MOVD ceAcc+24(FP), R19
	MOVD klAcc+32(FP), R20
	FMOVD $0, F14
	FMOVD $0, F15
	FMOVS (R10), F3
	VDUP V3.S[0], V3.S4
	FMOVS (R11), F4
	VDUP V4.S[0], V4.S4
	FMOVS (R12), F5
	VDUP V5.S[0], V5.S4
	MOVD $0x2b8cbccc, R17
	VMOV R17, V31.S[0]
	VDUP V31.S[0], V31.S4
	VMOV_B16(4, 27)
	VMAX_S4(31, 3, 3)
	VMAX_S4(31, 4, 4)
	VMAX_S4(31, 5, 5)
	AI_LOAD_LOG_CONSTS
	AI_NEON_LOG4(3, 10)
	AI_LOAD_LOG_CONSTS
	AI_NEON_LOG4(4, 11)
	AI_LOAD_LOG_CONSTS
	AI_NEON_LOG4(5, 12)
	VEOR V30.B16, V30.B16, V30.B16
	VFSUB_S4(10, 30, 6)
	VFMUL_S4(27, 6, 6)
	AI_F32_LANE0_F64_ADD_CE
	VFSUB_S4(12, 11, 7)
	VFMUL_S4(27, 7, 7)
	AI_F32_LANE0_V7_F64_ADD_KL
	FMOVD F14, 0(R19)
	FMOVD F15, 0(R20)
	RET

// func FreeEnergyFloat32NEONAsm(likelihood, posterior, prior *float32, count int) float32
TEXT ·FreeEnergyFloat32NEONAsm(SB), NOSPLIT, $0-40
	MOVD likelihood+0(FP), R10
	MOVD posterior+8(FP), R11
	MOVD prior+16(FP), R12
	MOVD count+24(FP), R3

	MOVD $0x2b8cbccc, R17
	VMOV R17, V31.S[0]
	VDUP V31.S[0], V31.S4
	AI_LOAD_LOG_CONSTS
	FMOVD $0, F14
	FMOVD $0, F15
	CBZ  R3, ai_fe_store

ai_fe_loop4:
	CMP  $4, R3
	BLT  ai_fe_tail

	AI_FE_BLOCK4
	B    ai_fe_loop4

ai_fe_tail:
	CBZ  R3, ai_fe_store

ai_fe_tail_loop:
	AI_FE_TAIL_ONE
	CBNZ R3, ai_fe_tail_loop

ai_fe_store:
	AI_FE_STORE_RESULT
	RET

// func ExpectedFreeEnergyFloat32NEONAsm(
//     predictedObs, preferredObs, predictedState *float32,
//     obsCount, stateCount int,
// ) float32
TEXT ·ExpectedFreeEnergyFloat32NEONAsm(SB), NOSPLIT, $0-48
	MOVD predictedObs+0(FP), R0
	MOVD preferredObs+8(FP), R1
	MOVD predictedState+16(FP), R2
	MOVD obsCount+24(FP), R3
	MOVD stateCount+32(FP), R4

	MOVD $0x2b8cbccc, R17
	VMOV R17, V31.S[0]
	VDUP V31.S[0], V31.S4
	AI_LOAD_LOG_CONSTS
	FMOVD $0, F14
	FMOVD $0, F15

ai_efe_obs_loop4:
	CMP  $4, R3
	BLT  ai_efe_obs_tail

	AI_EFE_OBS_BLOCK4
	B    ai_efe_obs_loop4

ai_efe_obs_tail:
	CBZ  R3, ai_efe_state_init

ai_efe_obs_tail_loop:
	AI_EFE_OBS_TAIL_ONE
	CBNZ R3, ai_efe_obs_tail_loop

ai_efe_state_init:
	MOVD predictedState+16(FP), R2
	MOVD stateCount+32(FP), R4

ai_efe_state_loop4:
	CMP  $4, R4
	BLT  ai_efe_state_tail

	AI_EFE_STATE_BLOCK4
	B    ai_efe_state_loop4

ai_efe_state_tail:
	CBZ  R4, ai_efe_store

ai_efe_state_tail_loop:
	AI_EFE_STATE_TAIL_ONE
	CBNZ R4, ai_efe_state_tail_loop

ai_efe_store:
	AI_EFE_STORE_RESULT
	RET
