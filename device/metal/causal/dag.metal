#include <metal_stdlib>
#include "causal.metal"

using namespace metal;

DAG_KERNELS(dag_markov_factorization_float32, Float32CausalStorage, float)
DAG_KERNELS(dag_markov_factorization_float16, Float16CausalStorage, half)
DAG_KERNELS(dag_markov_factorization_bfloat16, BFloat16CausalStorage, ushort)
