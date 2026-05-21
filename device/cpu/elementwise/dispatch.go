// Package elementwise implements dense element-wise binary and unary
// operations for float64, float32, float16, and bfloat16.
//
// Public entry points take unsafe.Pointer buffers and a dtype.DType,
// matching pkg/backend/device/cpu/activation. Float32 kernels follow
// the pick-at-init model: select_{arm64,amd64,generic}.go register
// ISA candidates; f32_kernels.go binds the picked implementation.
// Float16 and bfloat16 paths route through pick-at-init uint16 kernels:
// NEON on arm64, dtype-correct generic elsewhere.
package elementwise
