#include <metal_stdlib>
using namespace metal;

struct ResonantUpdateParams {
  uint32_t n;
  uint32_t D;
  uint32_t H;
  float inv_D;
  float scale;
  float damping;
  uint32_t zero_diag;
};

static inline bfloat resonant_bf16_sqrt(bfloat value) {
  if (value <= bfloat(0.0)) {
    return bfloat(0.0);
  }

  bfloat guess = value;

  for (int index = 0; index < 5; index++) {
    guess = bfloat(0.5) * (guess + value / guess);
  }

  return guess;
}

static inline float resonant_inv_r_float(float accumReal, float accumImag, float epsilon) {
  return 1.0f / precise::sqrt(accumReal * accumReal + accumImag * accumImag + epsilon);
}

static inline half resonant_inv_r_half(half accumReal, half accumImag, half epsilon) {
  return half(1.0h) / sqrt(accumReal * accumReal + accumImag * accumImag + epsilon);
}

static inline bfloat resonant_inv_r_bfloat(
  bfloat accumReal,
  bfloat accumImag,
  bfloat epsilon
) {
  return bfloat(1.0) / resonant_bf16_sqrt(
    accumReal * accumReal + accumImag * accumImag + epsilon
  );
}

#define RESONANT_UPDATE_FWD_KERNEL(name, storage_type, scalar_type, inv_r_fn, epsilon_literal) \
kernel void name( \
    device const storage_type* x [[buffer(0)]], \
    device const storage_type* y [[buffer(1)]], \
    device const storage_type* vr [[buffer(2)]], \
    device const storage_type* vi [[buffer(3)]], \
    device const storage_type* diag [[buffer(4)]], \
    device storage_type* xo [[buffer(5)]], \
    device storage_type* yo [[buffer(6)]], \
    device storage_type* a_out [[buffer(7)]], \
    device storage_type* b_out [[buffer(8)]], \
    device storage_type* inv_r_out [[buffer(9)]], \
    constant ResonantUpdateParams& p [[buffer(10)]], \
    uint gid [[thread_position_in_grid]]) { \
  if (gid >= p.n) { \
    return; \
  } \
  const uint32_t d = gid % p.D; \
  const uint32_t tmp = gid / p.D; \
  const uint32_t h = tmp % p.H; \
  const scalar_type invDim = scalar_type(p.inv_D); \
  const scalar_type scale = scalar_type(p.scale); \
  const scalar_type damping = scalar_type(p.damping); \
  const scalar_type oneMinus = scalar_type(1.0) - damping; \
  const scalar_type epsilon = scalar_type(epsilon_literal); \
  const scalar_type diagValue = scalar_type(diag[h * p.D + d]); \
  scalar_type couplingReal = scalar_type(vr[gid]) * invDim; \
  scalar_type couplingImag = scalar_type(vi[gid]) * invDim; \
  if (p.zero_diag) { \
    couplingReal -= diagValue * scalar_type(x[gid]); \
    couplingImag -= diagValue * scalar_type(y[gid]); \
  } \
  const scalar_type accumReal = scalar_type(x[gid]) * oneMinus + scale * couplingReal; \
  const scalar_type accumImag = scalar_type(y[gid]) * oneMinus + scale * couplingImag; \
  const scalar_type invRadius = inv_r_fn(accumReal, accumImag, epsilon); \
  xo[gid] = storage_type(accumReal * invRadius); \
  yo[gid] = storage_type(accumImag * invRadius); \
  a_out[gid] = storage_type(accumReal); \
  b_out[gid] = storage_type(accumImag); \
  inv_r_out[gid] = storage_type(invRadius); \
}

#define RESONANT_UPDATE_BWD_KERNEL(name, storage_type, scalar_type) \
kernel void name( \
    device const storage_type* gxo [[buffer(0)]], \
    device const storage_type* gyo [[buffer(1)]], \
    device const storage_type* x [[buffer(2)]], \
    device const storage_type* y [[buffer(3)]], \
    device const storage_type* diag [[buffer(4)]], \
    device const storage_type* a [[buffer(5)]], \
    device const storage_type* b [[buffer(6)]], \
    device const storage_type* inv_r [[buffer(7)]], \
    device storage_type* gvr [[buffer(8)]], \
    device storage_type* gvi [[buffer(9)]], \
    device storage_type* gx [[buffer(10)]], \
    device storage_type* gy [[buffer(11)]], \
    constant ResonantUpdateParams& p [[buffer(12)]], \
    uint gid [[thread_position_in_grid]]) { \
  if (gid >= p.n) { \
    return; \
  } \
  const uint32_t d = gid % p.D; \
  const uint32_t tmp = gid / p.D; \
  const uint32_t h = tmp % p.H; \
  const scalar_type scale = scalar_type(p.scale); \
  const scalar_type invDim = scalar_type(p.inv_D); \
  const scalar_type damping = scalar_type(p.damping); \
  const scalar_type diagValue = scalar_type(diag[h * p.D + d]); \
  const scalar_type inverseRadius = scalar_type(inv_r[gid]); \
  const scalar_type inverseRadiusCubed = inverseRadius * inverseRadius * inverseRadius; \
  const scalar_type gradXOutValue = scalar_type(gxo[gid]); \
  const scalar_type gradYOutValue = scalar_type(gyo[gid]); \
  const scalar_type aValue = scalar_type(a[gid]); \
  const scalar_type bValue = scalar_type(b[gid]); \
  const scalar_type dotProduct = gradXOutValue * aValue + gradYOutValue * bValue; \
  const scalar_type gradAccumReal = \
      gradXOutValue * inverseRadius - aValue * dotProduct * inverseRadiusCubed; \
  const scalar_type gradAccumImag = \
      gradYOutValue * inverseRadius - bValue * dotProduct * inverseRadiusCubed; \
  scalar_type stateCoeff = scalar_type(1.0) - damping; \
  if (p.zero_diag) { \
    stateCoeff -= scale * diagValue; \
  } \
  gx[gid] = storage_type(gradAccumReal * stateCoeff); \
  gy[gid] = storage_type(gradAccumImag * stateCoeff); \
  gvr[gid] = storage_type(gradAccumReal * (scale * invDim)); \
  gvi[gid] = storage_type(gradAccumImag * (scale * invDim)); \
  (void)x[gid]; \
  (void)y[gid]; \
}

RESONANT_UPDATE_FWD_KERNEL(
  resonant_update_fwd_fp32,
  float,
  float,
  resonant_inv_r_float,
  1.0e-6f
)

RESONANT_UPDATE_FWD_KERNEL(
  resonant_update_fwd_fp16,
  half,
  half,
  resonant_inv_r_half,
  1.0e-6h
)

kernel void resonant_update_fwd_bfloat16(
    device const ushort* x [[buffer(0)]],
    device const ushort* y [[buffer(1)]],
    device const ushort* vr [[buffer(2)]],
    device const ushort* vi [[buffer(3)]],
    device const ushort* diag [[buffer(4)]],
    device ushort* xo [[buffer(5)]],
    device ushort* yo [[buffer(6)]],
    device ushort* a_out [[buffer(7)]],
    device ushort* b_out [[buffer(8)]],
    device ushort* inv_r_out [[buffer(9)]],
    constant ResonantUpdateParams& p [[buffer(10)]],
    uint gid [[thread_position_in_grid]]) {
  if (gid >= p.n) {
    return;
  }

  const uint32_t d = gid % p.D;
  const uint32_t tmp = gid / p.D;
  const uint32_t h = tmp % p.H;
  const bfloat invDim = bfloat(p.inv_D);
  const bfloat scale = bfloat(p.scale);
  const bfloat damping = bfloat(p.damping);
  const bfloat oneMinus = bfloat(1.0) - damping;
  const bfloat epsilon = bfloat(1.0e-6);
  const bfloat diagValue = as_type<bfloat>(diag[h * p.D + d]);
  bfloat couplingReal = as_type<bfloat>(vr[gid]) * invDim;
  bfloat couplingImag = as_type<bfloat>(vi[gid]) * invDim;

  if (p.zero_diag) {
    couplingReal -= diagValue * as_type<bfloat>(x[gid]);
    couplingImag -= diagValue * as_type<bfloat>(y[gid]);
  }

  const bfloat accumReal =
      as_type<bfloat>(x[gid]) * oneMinus + scale * couplingReal;
  const bfloat accumImag =
      as_type<bfloat>(y[gid]) * oneMinus + scale * couplingImag;
  const bfloat invRadius = resonant_inv_r_bfloat(accumReal, accumImag, epsilon);

  xo[gid] = as_type<ushort>(accumReal * invRadius);
  yo[gid] = as_type<ushort>(accumImag * invRadius);
  a_out[gid] = as_type<ushort>(accumReal);
  b_out[gid] = as_type<ushort>(accumImag);
  inv_r_out[gid] = as_type<ushort>(invRadius);
}

RESONANT_UPDATE_BWD_KERNEL(resonant_update_bwd_fp32, float, float)
RESONANT_UPDATE_BWD_KERNEL(resonant_update_bwd_fp16, half, half)

kernel void resonant_update_bwd_bfloat16(
    device const ushort* gxo [[buffer(0)]],
    device const ushort* gyo [[buffer(1)]],
    device const ushort* x [[buffer(2)]],
    device const ushort* y [[buffer(3)]],
    device const ushort* diag [[buffer(4)]],
    device const ushort* a [[buffer(5)]],
    device const ushort* b [[buffer(6)]],
    device const ushort* inv_r [[buffer(7)]],
    device ushort* gvr [[buffer(8)]],
    device ushort* gvi [[buffer(9)]],
    device ushort* gx [[buffer(10)]],
    device ushort* gy [[buffer(11)]],
    constant ResonantUpdateParams& p [[buffer(12)]],
    uint gid [[thread_position_in_grid]]) {
  if (gid >= p.n) {
    return;
  }

  const uint32_t d = gid % p.D;
  const uint32_t tmp = gid / p.D;
  const uint32_t h = tmp % p.H;
  const bfloat scale = bfloat(p.scale);
  const bfloat invDim = bfloat(p.inv_D);
  const bfloat damping = bfloat(p.damping);
  const bfloat diagValue = as_type<bfloat>(diag[h * p.D + d]);
  const bfloat inverseRadius = as_type<bfloat>(inv_r[gid]);
  const bfloat inverseRadiusCubed =
      inverseRadius * inverseRadius * inverseRadius;
  const bfloat gradXOutValue = as_type<bfloat>(gxo[gid]);
  const bfloat gradYOutValue = as_type<bfloat>(gyo[gid]);
  const bfloat aValue = as_type<bfloat>(a[gid]);
  const bfloat bValue = as_type<bfloat>(b[gid]);
  const bfloat dotProduct = gradXOutValue * aValue + gradYOutValue * bValue;
  const bfloat gradAccumReal =
      gradXOutValue * inverseRadius - aValue * dotProduct * inverseRadiusCubed;
  const bfloat gradAccumImag =
      gradYOutValue * inverseRadius - bValue * dotProduct * inverseRadiusCubed;
  bfloat stateCoeff = bfloat(1.0) - damping;

  if (p.zero_diag) {
    stateCoeff -= scale * diagValue;
  }

  gx[gid] = as_type<ushort>(gradAccumReal * stateCoeff);
  gy[gid] = as_type<ushort>(gradAccumImag * stateCoeff);
  gvr[gid] = as_type<ushort>(gradAccumReal * (scale * invDim));
  gvi[gid] = as_type<ushort>(gradAccumImag * (scale * invDim));
  (void)x[gid];
  (void)y[gid];
}
