#ifndef PUTER_DEVICE_CUDA_HAWKES_HAWKES_CUH
#define PUTER_DEVICE_CUDA_HAWKES_HAWKES_CUH

#include <cuda_runtime.h>
#include <cuda_fp16.h>

static constexpr unsigned int cudaHawkesMarkovThreadCount = 256U;

__device__ __forceinline__ float hawkes_markov_bf16_to_float(unsigned short value) {
    unsigned int bits = static_cast<unsigned int>(value) << 16U;
    return __uint_as_float(bits);
}

__device__ __forceinline__ unsigned short hawkes_markov_float_to_bf16(float value) {
    return static_cast<unsigned short>(__float_as_uint(value) >> 16U);
}

__device__ __forceinline__ float hawkes_markov_safe_positive(float value) {
    return fmaxf(value, 1.0e-12f);
}

struct Float32HawkesMarkovStorage {
    __device__ static float load(const float* values, unsigned int index) {
        return values[index];
    }

    __device__ static void store(float* values, unsigned int index, float value) {
        values[index] = value;
    }
};

struct Float16HawkesMarkovStorage {
    __device__ static float load(const __half* values, unsigned int index) {
        return __half2float(values[index]);
    }

    __device__ static void store(__half* values, unsigned int index, float value) {
        values[index] = __float2half(value);
    }
};

struct BFloat16HawkesMarkovStorage {
    __device__ static float load(const unsigned short* values, unsigned int index) {
        return hawkes_markov_bf16_to_float(values[index]);
    }

    __device__ static void store(unsigned short* values, unsigned int index, float value) {
        values[index] = hawkes_markov_float_to_bf16(value);
    }
};

__device__ __forceinline__ float hawkes_kahan_add(float sum, float addend, float* compensation) {
    float y = addend - *compensation;
    float t = sum + y;
    *compensation = (t - sum) - y;
    return t;
}

__device__ __forceinline__ float hawkes_markov_reduce(float* reduction, unsigned int threadIndex) {
    __syncthreads();

    for (unsigned int stride = cudaHawkesMarkovThreadCount / 2U; stride > 0U; stride >>= 1U) {
        if (threadIndex < stride) {
            reduction[threadIndex] += reduction[threadIndex + stride];
        }

        __syncthreads();
    }

    return reduction[0];
}

// Matches cpu/hawkes hawkesExpC + activation.ExpF32NEON (VFRINTN, VFCVTZS, VFMLA Horner, VFMUL scale).
__device__ __forceinline__ float metal_hawkes_exp32(float value) {
    const float log2e = 1.4426950408889634f;
    const float ln2 = 0.6931471805599453f;
    float scaled = value * log2e;
    float roundedK = rintf(scaled);
    float fraction = value - roundedK * ln2;
    float poly = 0.00019841270f;

    poly = fmaf(fraction, poly, 0.0013888889f);
    poly = fmaf(fraction, poly, 0.008333334f);
    poly = fmaf(fraction, poly, 0.041666667f);
    poly = fmaf(fraction, poly, 0.16666667f);
    poly = fmaf(fraction, poly, 0.5f);
    poly = fmaf(fraction, poly, 1.0f);
    poly = fmaf(fraction, poly, 1.0f);

    int exponentInt = static_cast<int>(roundedK);
    unsigned int scaleBits = static_cast<unsigned int>(exponentInt + 127) << 23U;

    return poly * __uint_as_float(scaleBits);
}

template <typename Storage, typename Scalar>
__device__ __forceinline__ void hawkes_intensity_kernel(
    const Scalar* events,
    const Scalar* queryTimes,
    const Scalar* baseline,
    const Scalar* alpha,
    const Scalar* beta,
    Scalar* out,
    float* reduction,
    unsigned int eventCount,
    unsigned int queryIndex,
    unsigned int threadIndex
) {
    float queryTime = Storage::load(queryTimes, queryIndex);
    float alphaValue = Storage::load(alpha, 0);
    float betaValue = Storage::load(beta, 0);
    float intensitySum = 0.0f;

    for (unsigned int base = 0; base < eventCount; base += cudaHawkesMarkovThreadCount) {
        float localValue = 0.0f;
        unsigned int eventIndex = base + threadIndex;
        unsigned int waveEnd = min(base + cudaHawkesMarkovThreadCount, eventCount);

        if (eventIndex < waveEnd) {
            float eventTime = Storage::load(events, eventIndex);

            if (eventTime <= queryTime) {
                float exponentArg = -betaValue * (queryTime - eventTime);
                localValue = alphaValue * metal_hawkes_exp32(exponentArg);
            }
        }

        reduction[threadIndex] = localValue;
        __syncthreads();

        if (threadIndex == 0) {
            unsigned int activeEvents = waveEnd - base;

            for (unsigned int threadOffset = 0; threadOffset < activeEvents; threadOffset++) {
                intensitySum += reduction[threadOffset];
            }
        }

        __syncthreads();
    }

    if (threadIndex == 0) {
        Storage::store(out, queryIndex, Storage::load(baseline, 0) + intensitySum);
    }
}

template <typename Storage, typename Scalar>
__device__ __forceinline__ void hawkes_kernel_matrix_kernel(
    const Scalar* events,
    const Scalar* alpha,
    const Scalar* beta,
    Scalar* out,
    unsigned int eventCount,
    unsigned int index
) {
    unsigned int total = eventCount * eventCount;

    if (index >= total) {
        return;
    }

    unsigned int row = index / eventCount;
    unsigned int col = index - row * eventCount;
    float value = 0.0f;

    if (col < row) {
        float delta = Storage::load(events, row) - Storage::load(events, col);
        value = Storage::load(alpha, 0) * expf(-Storage::load(beta, 0) * delta);
    }

    Storage::store(out, index, value);
}

template <typename Storage, typename Scalar>
__device__ __forceinline__ void hawkes_log_likelihood_partial_kernel(
    const Scalar* events,
    const Scalar* totalTime,
    const Scalar* baseline,
    const Scalar* alpha,
    const Scalar* beta,
    float* scratch,
    float* reduction,
    unsigned int eventCount,
    unsigned int groupIndex,
    unsigned int threadIndex
) {
    float mu = Storage::load(baseline, 0);
    float alphaValue = Storage::load(alpha, 0);
    float betaValue = Storage::load(beta, 0);
    float totalTimeValue = Storage::load(totalTime, 0);
    unsigned int baseEventIndex = groupIndex * cudaHawkesMarkovThreadCount;

    for (unsigned int offset = 0; offset < cudaHawkesMarkovThreadCount; offset++) {
        unsigned int eventIndex = baseEventIndex + offset;

        if (eventIndex >= eventCount) {
            break;
        }

        float eventTime = Storage::load(events, eventIndex);
        float historySum = 0.0f;

        for (unsigned int previousIndex = threadIndex; previousIndex < eventIndex; previousIndex += cudaHawkesMarkovThreadCount) {
            float delta = eventTime - Storage::load(events, previousIndex);
            historySum += metal_hawkes_exp32(-betaValue * delta);
        }

        reduction[threadIndex] = historySum;
        float reducedHistory = hawkes_markov_reduce(reduction, threadIndex);

        if (threadIndex == 0) {
            float intensity = mu + alphaValue * reducedHistory;
            float compensator = (alphaValue / betaValue) *
                (1.0f - expf(-betaValue * (totalTimeValue - eventTime)));

            scratch[eventIndex] = logf(hawkes_markov_safe_positive(intensity)) - compensator;
        }

        __syncthreads();
    }
}

template <typename Storage, typename Scalar>
__device__ __forceinline__ void hawkes_log_likelihood_finalize_kernel(
    const float* scratch,
    const Scalar* totalTime,
    const Scalar* baseline,
    Scalar* out,
    float* reduction,
    unsigned int eventCount,
    unsigned int threadIndex
) {
    float localSum = 0.0f;
    float localCompensation = 0.0f;

    for (unsigned int index = threadIndex; index < eventCount; index += cudaHawkesMarkovThreadCount) {
        localSum = hawkes_kahan_add(localSum, scratch[index], &localCompensation);
    }

    reduction[threadIndex] = localSum;
    float sum = hawkes_markov_reduce(reduction, threadIndex);

    if (threadIndex == 0) {
        float baselineValue = Storage::load(baseline, 0);
        float totalTimeValue = Storage::load(totalTime, 0);
        Storage::store(out, 0, sum - baselineValue * totalTimeValue);
    }
}

template <typename Storage, typename Scalar>
__device__ __forceinline__ void markov_mutual_information_partial_kernel(
    const Scalar* joint,
    float* scratch,
    float* reduction,
    unsigned int rows,
    unsigned int cols,
    unsigned int groupIndex,
    unsigned int threadIndex
) {
    unsigned int total = rows * cols;
    unsigned int flatIndex = groupIndex * cudaHawkesMarkovThreadCount + threadIndex;
    float localValue = 0.0f;

    if (flatIndex < total) {
        unsigned int row = flatIndex / cols;
        unsigned int col = flatIndex - row * cols;
        float jointValue = Storage::load(joint, flatIndex);

        if (jointValue > 1.0e-12f) {
            float marginalRow = 0.0f;
            float marginalCol = 0.0f;

            for (unsigned int colIndex = 0; colIndex < cols; colIndex++) {
                marginalRow += Storage::load(joint, row * cols + colIndex);
            }

            for (unsigned int rowIndex = 0; rowIndex < rows; rowIndex++) {
                marginalCol += Storage::load(joint, rowIndex * cols + col);
            }

            localValue = jointValue * logf(jointValue / (marginalRow * marginalCol + 1.0e-12f));
        }
    }

    reduction[threadIndex] = localValue;
    float sum = hawkes_markov_reduce(reduction, threadIndex);

    if (threadIndex == 0) {
        scratch[groupIndex] = sum;
    }
}

template <typename Storage, typename Scalar>
__device__ __forceinline__ void hawkes_markov_finalize_kernel(
    const float* scratch,
    Scalar* out,
    float* reduction,
    unsigned int partialCount,
    unsigned int threadIndex
) {
    float localValue = 0.0f;

    for (unsigned int index = threadIndex; index < partialCount; index += cudaHawkesMarkovThreadCount) {
        localValue += scratch[index];
    }

    reduction[threadIndex] = localValue;
    float sum = hawkes_markov_reduce(reduction, threadIndex);

    if (threadIndex == 0) {
        Storage::store(out, 0, sum);
    }
}

template <typename Storage, typename Scalar>
__device__ __forceinline__ void markov_blanket_partition_kernel(
    const Scalar* adjacency,
    const int* internalNodes,
    int* out,
    unsigned int nodeCount,
    unsigned int internalCount,
    unsigned int nodeIndex
) {
    if (nodeIndex >= nodeCount) {
        return;
    }

    bool isInternal = false;
    bool incomingFromInternal = false;
    bool outgoingToInternal = false;

    for (unsigned int index = 0; index < internalCount; index++) {
        int internalNode = internalNodes[index];

        if (internalNode < 0 || static_cast<unsigned int>(internalNode) >= nodeCount) {
            continue;
        }

        unsigned int other = static_cast<unsigned int>(internalNode);
        isInternal = isInternal || other == nodeIndex;
        incomingFromInternal = incomingFromInternal ||
            Storage::load(adjacency, other * nodeCount + nodeIndex) != 0.0f;
        outgoingToInternal = outgoingToInternal ||
            Storage::load(adjacency, nodeIndex * nodeCount + other) != 0.0f;
    }

    if (isInternal) {
        out[nodeIndex] = 0;
        return;
    }

    if (incomingFromInternal && outgoingToInternal) {
        out[nodeIndex] = 2;
        return;
    }

    if (outgoingToInternal) {
        out[nodeIndex] = 1;
        return;
    }

    out[nodeIndex] = 3;
}

template <typename Storage, typename Scalar>
__device__ __forceinline__ void markov_flow_kernel(
    const Scalar* mutualInformation,
    const int* partition,
    Scalar* out,
    unsigned int nodeCount,
    int targetLabel,
    unsigned int nodeIndex
) {
    if (nodeIndex >= nodeCount) {
        return;
    }

    if (partition[nodeIndex] != targetLabel) {
        Storage::store(out, nodeIndex, 0.0f);
        return;
    }

    float sum = 0.0f;

    for (unsigned int otherIndex = 0; otherIndex < nodeCount; otherIndex++) {
        if (partition[otherIndex] == 0) {
            sum += Storage::load(mutualInformation, nodeIndex * nodeCount + otherIndex);
        }
    }

    Storage::store(out, nodeIndex, sum);
}

#endif
