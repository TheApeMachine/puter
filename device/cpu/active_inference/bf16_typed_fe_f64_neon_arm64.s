// SPDX-License-Identifier: Apache-2.0
// bfloat16 typed FreeEnergy / ExpectedFreeEnergy: load bf16 → f64, math.Log in f64,
// f64 accumulate, single bf16 store. No f32 compute or per-element narrow.
#include "textflag.h"

DATA aiEpsF64<>+0(SB)/8, $0x3F50600000000000
GLOBL aiEpsF64<>(SB), 8, $8

#define AI_BF16_LOAD_F64(ptr, fd) \
	MOVHU (ptr), R6 ;\
	LSL  $16, R6, R6 ;\
	FMOVS R6, F1 ;\
	FCVTSD F1, fd

#define AI_F64_CLAMP_POS(fd) \
	FMOVD aiEpsF64<>(SB), F2 ;\
	FMAXD fd, F2, fd

#define AI_F64_LOG_TO(in, out) \
	FMOVD F14, acc14+56(SP) ;\
	FMOVD F15, acc15+64(SP) ;\
	MOVD R0, saveR0+72(SP) ;\
	MOVD R1, saveR1+80(SP) ;\
	MOVD R2, saveR2+88(SP) ;\
	MOVD R3, saveR3+96(SP) ;\
	MOVD R4, saveR4+104(SP) ;\
	MOVD R10, saveR10+112(SP) ;\
	MOVD R11, saveR11+120(SP) ;\
	MOVD R12, saveR12+48(SP) ;\
	FMOVD in, F0 ;\
	CALL ·activeInferenceStdLogF64(SB) ;\
	FMOVD F0, out ;\
	FMOVD acc14+56(SP), F14 ;\
	FMOVD acc15+64(SP), F15 ;\
	MOVD saveR0+72(SP), R0 ;\
	MOVD saveR1+80(SP), R1 ;\
	MOVD saveR2+88(SP), R2 ;\
	MOVD saveR3+96(SP), R3 ;\
	MOVD saveR4+104(SP), R4 ;\
	MOVD saveR10+112(SP), R10 ;\
	MOVD saveR11+120(SP), R11 ;\
	MOVD saveR12+48(SP), R12

#define AI_BF16_STORE_FE_SUM(fd) \
	FCVTSD fd, F0 ;\
	FMOVS F0, R6 ;\
	LSR  $16, R6, R6 ;\
	MOVH R6, ret+32(FP)

#define AI_BF16_STORE_EFE_SUM(fd) \
	FCVTSD fd, F0 ;\
	FMOVS F0, R6 ;\
	LSR  $16, R6, R6 ;\
	MOVH R6, ret+40(FP)

// func FreeEnergyBFloat16NEONAsm(likelihood, posterior, prior *uint16, count int) uint16
TEXT ·FreeEnergyBFloat16NEONAsm(SB), $512-34
	MOVD likelihood+0(FP), R10
	MOVD posterior+8(FP), R11
	MOVD prior+16(FP), R12
	MOVD count+24(FP), R3
	FMOVD $0, F14
	FMOVD $0, F15
	CBZ  R3, ai_bf16_fe_f64_store

ai_bf16_fe_f64_loop:
	AI_BF16_LOAD_F64(R10, F8)
	AI_BF16_LOAD_F64(R11, F9)
	AI_BF16_LOAD_F64(R12, F10)
	FMOVD F8, F12
	FMOVD F9, F11
	FMOVD F10, F13
	AI_F64_CLAMP_POS(F12)
	AI_F64_CLAMP_POS(F11)
	AI_F64_CLAMP_POS(F13)
	AI_F64_LOG_TO(F12, F4)
	AI_F64_LOG_TO(F11, F5)
	AI_F64_LOG_TO(F13, F6)
	FNEGD F4, F4
	FMULD F9, F4, F4
	FADDD F4, F14, F14
	FSUBD F6, F5, F5
	FMULD F9, F5, F5
	FADDD F5, F15, F15
	ADD  $2, R10
	ADD  $2, R11
	ADD  $2, R12
	SUB  $1, R3
	CBNZ R3, ai_bf16_fe_f64_loop

ai_bf16_fe_f64_store:
	FADDD F15, F14, F14
	AI_BF16_STORE_FE_SUM(F14)
	RET

// func ExpectedFreeEnergyBFloat16NEONAsm(
//     predictedObs, preferredObs, predictedState *uint16,
//     obsCount, stateCount int,
// ) uint16
TEXT ·ExpectedFreeEnergyBFloat16NEONAsm(SB), $512-42
	MOVD predictedObs+0(FP), R0
	MOVD preferredObs+8(FP), R1
	MOVD predictedState+16(FP), R2
	MOVD obsCount+24(FP), R3
	MOVD stateCount+32(FP), R4
	FMOVD $0, F14

ai_bf16_efe_f64_obs:
	CBZ  R3, ai_bf16_efe_f64_state

ai_bf16_efe_f64_obs_loop:
	AI_BF16_LOAD_F64(R0, F8)
	AI_BF16_LOAD_F64(R1, F9)
	FMOVD F8, F12
	FMOVD F9, F13
	AI_F64_CLAMP_POS(F12)
	AI_F64_CLAMP_POS(F13)
	AI_F64_LOG_TO(F12, F4)
	AI_F64_LOG_TO(F13, F5)
	FSUBD F5, F4, F4
	FMULD F8, F4, F4
	FADDD F4, F14, F14
	ADD  $2, R0
	ADD  $2, R1
	SUB  $1, R3
	CBNZ R3, ai_bf16_efe_f64_obs_loop
	B    ai_bf16_efe_f64_state

ai_bf16_efe_f64_state:
	MOVD predictedState+16(FP), R2
	MOVD stateCount+32(FP), R4
	CBZ  R4, ai_bf16_efe_f64_store

ai_bf16_efe_f64_state_loop:
	AI_BF16_LOAD_F64(R2, F8)
	FMOVD F8, F9
	FMOVD F8, F12
	AI_F64_CLAMP_POS(F12)
	AI_F64_LOG_TO(F12, F4)
	FNEGD F4, F4
	FMULD F9, F4, F4
	FADDD F4, F14, F14
	ADD  $2, R2
	SUB  $1, R4
	CBNZ R4, ai_bf16_efe_f64_state_loop

ai_bf16_efe_f64_store:
	AI_BF16_STORE_EFE_SUM(F14)
	RET
