// NEON dequant widen/convert helpers shared by int4 and int8 kernels.

#define VSXTL_H8(n, d)       WORD $(0x0F08A400 | ((n) << 5) | (d))
#define VSXTL2_H8(n, d)      WORD $(0x4F08A400 | ((n) << 5) | (d))
#define VSXTL_S4(n, d)       WORD $(0x0F10A400 | ((n) << 5) | (d))
#define VSXTL2_S4(n, d)      WORD $(0x4F10A400 | ((n) << 5) | (d))
#define VSCVTF_S4(n, d)      WORD $(0x4E21D800 | ((n) << 5) | (d))
#define VFMUL_S4(m, n, d)    WORD $(0x6E20DC00 | ((m) << 16) | ((n) << 5) | (d))
#define VSUB_H8(m, n, d)     WORD $(0x6E608400 | ((m) << 16) | ((n) << 5) | (d))
