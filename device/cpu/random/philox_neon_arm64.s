// SPDX-License-Identifier: Apache-2.0
//
// NEON 4-lane Philox-4×32-10 kernel for arm64.
//
// Algorithm contract is identical to Philox4x32 in philox.go — same M
// and W constants, same 10 rounds, same round function. The only
// difference is parallelism: this kernel runs four independent lanes
// at once, each with a counter offset by its lane index from ctrBase.
//
// Bitwise parity vs the scalar reference is the verification gate.
// Every lane's 4-uint32 output must equal what Philox4x32 returns when
// called with (key0, key1, ctrBase + lane).
//
// Register usage:
//   R0    out pointer (16 uint32s = 64 bytes)
//   R1    key0
//   R2    key1
//   R3    ctrBaseLow  (uint32 low half of ctrBase)
//   R4    ctrBaseHigh (uint32 high half of ctrBase)
//   R5    k0_scalar (running, broadcast each round)
//   R6    k1_scalar (running)
//   R7    round counter (10 → 0)
//   R8    scratch (W constants, lane-offset constants)
//   V0    V_C0 = [ctrBaseLow+0, +1, +2, +3]
//   V1    V_C1 = broadcast(ctrBaseHigh)
//   V2    V_C2 = zero
//   V3    V_C3 = zero
//   V4    V_M0 broadcast(0xD2511F53)
//   V5    V_M1 broadcast(0xCD9E8D57)
//   V6    UMULL low product (2 lanes of 64-bit)
//   V7    UMULL2 high product (2 lanes of 64-bit)
//   V8    V_LO0 = mulhilo_lo32(M0, C0) per lane
//   V9    V_HI0 = mulhilo_hi32(M0, C0) per lane
//   V10   V_LO1 = mulhilo_lo32(M1, C2) per lane
//   V11   V_HI1 = mulhilo_hi32(M1, C2) per lane
//   V12   V_K0 = broadcast(k0_scalar)
//   V13   V_K1 = broadcast(k1_scalar)
//   V14   scratch (new c0 build)
//   V15   scratch (new c2 build)
//
// Instructions encoded via WORD because Go's arm64 assembler does not
// expose them as mnemonics:
//   UMULL  Vd.2D, Vn.2S, Vm.2S   0x2EA0C000 | (Rm<<16) | (Rn<<5) | Rd
//   UMULL2 Vd.2D, Vn.4S, Vm.4S   0x6EA0C000 | (Rm<<16) | (Rn<<5) | Rd
//   UZP1   Vd.4S, Vn.4S, Vm.4S   0x4E801800 | (Rm<<16) | (Rn<<5) | Rd
//   UZP2   Vd.4S, Vn.4S, Vm.4S   0x4E805800 | (Rm<<16) | (Rn<<5) | Rd

#include "textflag.h"

#define UMULL_2D_2S(m, n, d)  WORD $(0x2EA0C000 | ((m) << 16) | ((n) << 5) | (d))
#define UMULL2_2D_4S(m, n, d) WORD $(0x6EA0C000 | ((m) << 16) | ((n) << 5) | (d))
#define UZP1_4S(m, n, d)      WORD $(0x4E801800 | ((m) << 16) | ((n) << 5) | (d))
#define UZP2_4S(m, n, d)      WORD $(0x4E805800 | ((m) << 16) | ((n) << 5) | (d))

DATA ·philoxLaneOffset+0x00(SB)/4, $0x00000000
DATA ·philoxLaneOffset+0x04(SB)/4, $0x00000001
DATA ·philoxLaneOffset+0x08(SB)/4, $0x00000002
DATA ·philoxLaneOffset+0x0C(SB)/4, $0x00000003
GLOBL ·philoxLaneOffset(SB), RODATA, $16

// func Philox4x32x4NEON(out *uint32, key0 uint32, key1 uint32, ctrBase uint64)
TEXT ·Philox4x32x4NEON(SB), NOSPLIT, $0-24
    MOVD out+0(FP), R0
    MOVWU key0+8(FP), R1
    MOVWU key1+12(FP), R2
    MOVD ctrBase+16(FP), R3
    LSR  $32, R3, R4         // R4 = ctrBaseHigh
    AND  $0xFFFFFFFF, R3, R3 // R3 = ctrBaseLow (zero-extended)

    // Build V0 = [ctrBaseLow+0, ctrBaseLow+1, ctrBaseLow+2, ctrBaseLow+3]
    VDUP R3, V0.S4           // V0 = [base, base, base, base]
    MOVD $·philoxLaneOffset(SB), R8
    VLD1 (R8), [V16.S4]      // V16 = [0, 1, 2, 3]
    VADD V16.S4, V0.S4, V0.S4

    // V1 = broadcast(ctrBaseHigh); assumption documented: no lane overflow into Ctr1.
    VDUP R4, V1.S4

    // V2, V3 = zero
    VEOR V2.B16, V2.B16, V2.B16
    VEOR V3.B16, V3.B16, V3.B16

    // V4 = broadcast(0xD2511F53)
    MOVD $0xD2511F53, R8
    VDUP R8, V4.S4

    // V5 = broadcast(0xCD9E8D57)
    MOVD $0xCD9E8D57, R8
    VDUP R8, V5.S4

    // Running scalar key parts.
    MOVW R1, R5
    MOVW R2, R6

    // 10 rounds.
    MOVD $10, R7

round_loop:
    // Broadcast current key parts.
    VDUP R5, V12.S4
    VDUP R6, V13.S4

    // mulhilo(M0, C0):
    //   V6 = UMULL  V4.2S, V0.2S  (lanes 0,1 of V0 × V4 → 2 × uint64)
    //   V7 = UMULL2 V4.4S, V0.4S  (lanes 2,3 → 2 × uint64)
    //   V8 = UZP1 V6.4S, V7.4S   → 4 × uint32 of low halves
    //   V9 = UZP2 V6.4S, V7.4S   → 4 × uint32 of high halves
    UMULL_2D_2S(4, 0, 6)
    UMULL2_2D_4S(4, 0, 7)
    UZP1_4S(7, 6, 8)
    UZP2_4S(7, 6, 9)

    // mulhilo(M1, C2):
    UMULL_2D_2S(5, 2, 6)
    UMULL2_2D_4S(5, 2, 7)
    UZP1_4S(7, 6, 10)
    UZP2_4S(7, 6, 11)

    // new C0 = HI1 XOR C1 XOR K0
    VEOR V11.B16, V1.B16, V14.B16
    VEOR V12.B16, V14.B16, V14.B16

    // new C2 = HI0 XOR C3 XOR K1
    VEOR V9.B16, V3.B16, V15.B16
    VEOR V13.B16, V15.B16, V15.B16

    // new C1 = LO1 (in V10), new C3 = LO0 (in V8)
    VMOV V10.B16, V1.B16
    VMOV V8.B16, V3.B16
    VMOV V14.B16, V0.B16
    VMOV V15.B16, V2.B16

    // Key Weyl bump.
    MOVD $0x9E3779B9, R8
    ADDW R8, R5, R5
    MOVD $0xBB67AE85, R8
    ADDW R8, R6, R6

    SUB $1, R7, R7
    CBNZ R7, round_loop

    // Store interleaved: ST4 { V0.4S, V1.4S, V2.4S, V3.4S }, [R0]
    VST4 [V0.S4, V1.S4, V2.S4, V3.S4], (R0)
    RET
