// SPDX-License-Identifier: Apache-2.0
// bfloat16 typed FreeEnergy / ExpectedFreeEnergy: load bf16 → f64, log via aiNeonLogF64, f64 accumulate, bf16 store.
#include "textflag.h"

#define AI_BF16_LOAD_F64(ptr, fd) \
	MOVHU (ptr), R6 ;\
	LSL  $16, R6, R6 ;\
	FMOVS R6, F1 ;\
	FCVTSD F1, fd

#define AI_F64_CLAMP_POS(fd) \
	MOVD  $0x3F506024DD2F1AA0, R6 ;\
	FMOVD R6, F2 ;\
	FMAXD fd, F2, fd

#define AI_BF16_STORE_FE_SUM(fd) \
	FCVTDS fd, F0 ;\
	FMOVS F0, R6 ;\
	LSR  $16, R6, R6 ;\
	MOVH R6, ret+32(FP)

#define AI_BF16_STORE_EFE_SUM(fd) \
	FCVTDS fd, F0 ;\
	FMOVS F0, R6 ;\
	LSR  $16, R6, R6 ;\
	MOVH R6, ret+40(FP)

#define AI_F64_ADD_TO_CE(term) \
	FMOVD 48(RSP), F14 ;\
	FADDD term, F14, F14 ;\
	FMOVD F14, 48(RSP)

#define AI_F64_ADD_TO_KL(term) \
	FMOVD 56(RSP), F15 ;\
	FADDD term, F15, F15 ;\
	FMOVD F15, 56(RSP)

// func freeEnergyBFloat16NEONBridge(likelihood, posterior, prior *uint16, count int) uint16
TEXT ·freeEnergyBFloat16NEONBridge(SB), NOSPLIT, $96-34
	MOVD likelihood+0(FP), R19
	MOVD posterior+8(FP), R20
	MOVD prior+16(FP), R21
	MOVD count+24(FP), R22
	FMOVD $0, F14
	FMOVD F14, 48(RSP)
	FMOVD F14, 56(RSP)
	CBZ  R22, ai_bf16_fe_f64_store

ai_bf16_fe_f64_loop:
	AI_BF16_LOAD_F64(R19, F8)
	AI_BF16_LOAD_F64(R20, F9)
	AI_BF16_LOAD_F64(R21, F10)
	FMOVD F8, F12
	FMOVD F9, F11
	FMOVD F10, F13
	AI_F64_CLAMP_POS(F12)
	AI_F64_CLAMP_POS(F11)
	AI_F64_CLAMP_POS(F13)
	FMOVD F12, 64(RSP)
	FMOVD F11, 72(RSP)
	FMOVD F13, 80(RSP)
	FMOVD 64(RSP), F0
	CALL  aiNeonLogF64(SB)
	FMOVD F0, F4
	FNEGD F4, F4
	AI_BF16_LOAD_F64(R20, F9)
	FMULD F9, F4, F4
	AI_F64_ADD_TO_CE(F4)
	FMOVD 72(RSP), F0
	CALL  aiNeonLogF64(SB)
	FMOVD F0, F5
	FMOVD 80(RSP), F0
	CALL  aiNeonLogF64(SB)
	FMOVD F0, F6
	FSUBD F6, F5, F5
	AI_BF16_LOAD_F64(R20, F9)
	FMULD F9, F5, F5
	AI_F64_ADD_TO_KL(F5)
	ADD  $2, R19
	ADD  $2, R20
	ADD  $2, R21
	SUB  $1, R22
	CBNZ R22, ai_bf16_fe_f64_loop

ai_bf16_fe_f64_store:
	FMOVD 48(RSP), F14
	FMOVD 56(RSP), F15
	FADDD F15, F14, F14
	AI_BF16_STORE_FE_SUM(F14)
	RET

// func expectedFreeEnergyBFloat16NEONBridge(
//     predictedObs, preferredObs, predictedState *uint16,
//     obsCount, stateCount int,
// ) uint16
TEXT ·expectedFreeEnergyBFloat16NEONBridge(SB), NOSPLIT, $96-42
	MOVD predictedObs+0(FP), R19
	MOVD preferredObs+8(FP), R20
	MOVD predictedState+16(FP), R21
	MOVD obsCount+24(FP), R22
	MOVD stateCount+32(FP), R23
	FMOVD $0, F14
	FMOVD F14, 48(RSP)
	CBZ  R22, ai_bf16_efe_f64_state

ai_bf16_efe_f64_obs_loop:
	AI_BF16_LOAD_F64(R19, F8)
	AI_BF16_LOAD_F64(R20, F9)
	FMOVD F8, F12
	FMOVD F9, F13
	AI_F64_CLAMP_POS(F12)
	AI_F64_CLAMP_POS(F13)
	FMOVD F12, 64(RSP)
	FMOVD F13, 72(RSP)
	FMOVD 64(RSP), F0
	CALL  aiNeonLogF64(SB)
	FMOVD F0, F4
	FMOVD 72(RSP), F0
	CALL  aiNeonLogF64(SB)
	FMOVD F0, F5
	FSUBD F5, F4, F4
	AI_BF16_LOAD_F64(R19, F8)
	FMULD F8, F4, F4
	FMOVD 48(RSP), F14
	FADDD F4, F14, F14
	FMOVD F14, 48(RSP)
	ADD  $2, R19
	ADD  $2, R20
	SUB  $1, R22
	CBNZ R22, ai_bf16_efe_f64_obs_loop

ai_bf16_efe_f64_state:
	CBZ  R23, ai_bf16_efe_f64_store

ai_bf16_efe_f64_state_loop:
	AI_BF16_LOAD_F64(R21, F8)
	FMOVD F8, F12
	AI_F64_CLAMP_POS(F12)
	FMOVD F12, 64(RSP)
	FMOVD 64(RSP), F0
	CALL  aiNeonLogF64(SB)
	FMOVD F0, F4
	FNEGD F4, F4
	AI_BF16_LOAD_F64(R21, F8)
	FMULD F8, F4, F4
	FMOVD 48(RSP), F14
	FADDD F4, F14, F14
	FMOVD F14, 48(RSP)
	ADD  $2, R21
	SUB  $1, R23
	CBNZ R23, ai_bf16_efe_f64_state_loop

ai_bf16_efe_f64_store:
	FMOVD 48(RSP), F14
	AI_BF16_STORE_EFE_SUM(F14)
	RET
