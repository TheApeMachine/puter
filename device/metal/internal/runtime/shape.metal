#include <metal_stdlib>

using namespace metal;

static inline void copy_tail_bytes(
    device const uchar* input,
    device uchar* out,
    uint byteCount,
    uint base
) {
    for (uint offset = 0; offset < 16; offset++) {
        uint byteIndex = base + offset;

        if (byteIndex < byteCount) {
            out[byteIndex] = input[byteIndex];
        }
    }
}

static inline bool shape_mask_bit(device const uchar* mask, uint index) {
    return ((mask[index >> 3u] >> (index & 7u)) & 1u) != 0u;
}

static inline void shape_set_error(device atomic_uint* errorFlag) {
    atomic_store_explicit(errorFlag, 1u, memory_order_relaxed);
}

template <typename Storage>
static inline void gather_kernel(
    device const Storage* source,
    device const int* indices,
    device Storage* out,
    device atomic_uint* errorFlag,
    constant uint& sourceRows,
    constant uint& inner,
    constant uint& outRows,
    uint index
) {
    uint count = outRows * inner;

    if (index >= count) {
        return;
    }

    uint outRow = index / inner;
    uint col = index - outRow * inner;
    int sourceRow = indices[outRow];

    if (sourceRow < 0 || uint(sourceRow) >= sourceRows) {
        shape_set_error(errorFlag);
        return;
    }

    out[index] = source[uint(sourceRow) * inner + col];
}

template <typename Storage>
static inline void scatter_kernel(
    device const Storage* target,
    device const int* indices,
    device const Storage* updates,
    device Storage* out,
    device atomic_uint* errorFlag,
    constant uint& targetRows,
    constant uint& inner,
    constant uint& updateRows,
    uint index
) {
    uint count = targetRows * inner;

    if (index >= count) {
        return;
    }

    uint targetRow = index / inner;
    uint col = index - targetRow * inner;
    Storage value = target[index];

    for (int updateRow = int(updateRows) - 1; updateRow >= 0; updateRow--) {
        int indexedRow = indices[uint(updateRow)];

        if (indexedRow < 0 || uint(indexedRow) >= targetRows) {
            shape_set_error(errorFlag);
            continue;
        }

        if (uint(indexedRow) == targetRow) {
            value = updates[uint(updateRow) * inner + col];
            break;
        }
    }

    out[index] = value;
}

template <typename Storage>
static inline void page_write_kernel(
    device const Storage* storage,
    device const Storage* values,
    device const int* pageIDs,
    device const int* offsets,
    device Storage* out,
    device atomic_uint* errorFlag,
    constant uint& pageCount,
    constant uint& pageSize,
    constant uint& inner,
    constant uint& valueRows,
    constant uint& storageOffset,
    constant uint& outOffset,
    uint index
) {
    uint count = pageCount * pageSize * inner;

    if (index >= count) {
        return;
    }

    uint storageRow = index / inner;
    uint col = index - storageRow * inner;
    Storage value = storage[storageOffset + index];

    for (int row = int(valueRows) - 1; row >= 0; row--) {
        int pageID = pageIDs[uint(row)];
        int pageOffset = offsets[uint(row)];

        if (pageID < 0 || uint(pageID) >= pageCount || pageOffset < 0 || uint(pageOffset) >= pageSize) {
            shape_set_error(errorFlag);
            continue;
        }

        uint writeRow = uint(pageID) * pageSize + uint(pageOffset);

        if (writeRow == storageRow) {
            value = values[uint(row) * inner + col];
            break;
        }
    }

    out[outOffset + index] = value;
}

template <typename Storage>
static inline void page_gather_kernel(
    device const Storage* storage,
    device const int* pageTable,
    device Storage* out,
    device atomic_uint* errorFlag,
    constant uint& pageCount,
    constant uint& pageSize,
    constant uint& inner,
    constant uint& outRows,
    constant uint& storageOffset,
    constant uint& outOffset,
    uint index
) {
    uint count = outRows * inner;

    if (index >= count) {
        return;
    }

    uint row = index / inner;
    uint col = index - row * inner;
    uint tableIndex = row / pageSize;
    uint pageOffset = row - tableIndex * pageSize;
    int pageID = pageTable[tableIndex];

    if (pageID < 0 || uint(pageID) >= pageCount) {
        shape_set_error(errorFlag);
        return;
    }

    uint storageRow = uint(pageID) * pageSize + pageOffset;
    out[outOffset + index] = storage[storageOffset + storageRow * inner + col];
}

template <typename Scalar, typename Vec>
static inline void where_kernel(
    device const uchar* mask,
    device const Vec* positive,
    device const Vec* negative,
    device Vec* out,
    constant uint& count,
    uint index
) {
    uint base = index * 4u;
    Vec positiveValues = positive[index];
    Vec negativeValues = negative[index];
    Vec result = negativeValues;

    if (base < count && shape_mask_bit(mask, base)) {
        result.x = positiveValues.x;
    }

    if (base + 1u < count && shape_mask_bit(mask, base + 1u)) {
        result.y = positiveValues.y;
    }

    if (base + 2u < count && shape_mask_bit(mask, base + 2u)) {
        result.z = positiveValues.z;
    }

    if (base + 3u < count && shape_mask_bit(mask, base + 3u)) {
        result.w = positiveValues.w;
    }

    out[index] = result;
}

template <typename Scalar, typename Vec>
static inline void masked_fill_kernel(
    device const Vec* input,
    device const uchar* mask,
    device const Scalar* scalar,
    device Vec* out,
    constant uint& count,
    uint index
) {
    uint base = index * 4u;
    Vec result = input[index];
    Scalar fillValue = scalar[0];

    if (base < count && shape_mask_bit(mask, base)) {
        result.x = fillValue;
    }

    if (base + 1u < count && shape_mask_bit(mask, base + 1u)) {
        result.y = fillValue;
    }

    if (base + 2u < count && shape_mask_bit(mask, base + 2u)) {
        result.z = fillValue;
    }

    if (base + 3u < count && shape_mask_bit(mask, base + 3u)) {
        result.w = fillValue;
    }

    out[index] = result;
}

template <typename Storage>
static inline void transpose_kernel(
    device const Storage* input,
    device Storage* out,
    device atomic_uint* errorFlag,
    constant uint& rank,
    constant uint& count,
    constant uint* permutation,
    constant uint* inputStrides,
    constant uint* outStrides,
    uint index
) {
    if (index >= count) {
        return;
    }

    if (rank == 0u) {
        shape_set_error(errorFlag);
        return;
    }

    uint remainder = index;
    uint outIndex = 0u;

    for (uint inAxis = 0u; inAxis < rank; inAxis++) {
        uint stride = inputStrides[inAxis];

        if (stride == 0u) {
            shape_set_error(errorFlag);
            return;
        }

        uint coordinate = remainder / stride;
        remainder -= coordinate * stride;

        for (uint outAxis = 0u; outAxis < rank; outAxis++) {
            if (permutation[outAxis] == inAxis) {
                outIndex += coordinate * outStrides[outAxis];
            }
        }
    }

    out[outIndex] = input[index];
}

static inline void copy_bytes_kernel(
    device const uint4* inputVector,
    device uint4* outVector,
    constant uint& byteCount,
    uint index
) {
    uint base = index * 16;

    if (base + 15 < byteCount) {
        outVector[index] = inputVector[index];
        return;
    }

    device const uchar* input = reinterpret_cast<device const uchar*>(inputVector);
    device uchar* out = reinterpret_cast<device uchar*>(outVector);
    copy_tail_bytes(input, out, byteCount, base);
}

static inline void concat_bytes_kernel(
    device const uint4* leftVector,
    device const uint4* rightVector,
    device uint4* outVector,
    constant uint& leftBytes,
    constant uint& totalBytes,
    uint index
) {
    uint base = index * 16;

    if (base + 15 < leftBytes) {
        outVector[index] = leftVector[index];
        return;
    }

    if (base >= leftBytes && base + 15 < totalBytes && leftBytes % 16 == 0) {
        outVector[index] = rightVector[(base - leftBytes) / 16];
        return;
    }

    device const uchar* left = reinterpret_cast<device const uchar*>(leftVector);
    device const uchar* right = reinterpret_cast<device const uchar*>(rightVector);
    device uchar* out = reinterpret_cast<device uchar*>(outVector);

    for (uint offset = 0; offset < 16; offset++) {
        uint outIndex = base + offset;

        if (outIndex >= totalBytes) {
            continue;
        }

        if (outIndex < leftBytes) {
            out[outIndex] = left[outIndex];
            continue;
        }

        out[outIndex] = right[outIndex - leftBytes];
    }
}

static inline void concat_last_dim_bytes_kernel(
    device const uchar* left,
    device const uchar* right,
    device uchar* out,
    constant uint& leftRowBytes,
    constant uint& rightRowBytes,
    constant uint& rowBytes,
    constant uint& totalBytes,
    uint index
) {
    uint base = index * 16;

    for (uint offset = 0; offset < 16; offset++) {
        uint outIndex = base + offset;

        if (outIndex >= totalBytes) {
            continue;
        }

        uint row = outIndex / rowBytes;
        uint col = outIndex - row * rowBytes;

        if (col < leftRowBytes) {
            out[outIndex] = left[row * leftRowBytes + col];
            continue;
        }

        out[outIndex] = right[row * rightRowBytes + (col - leftRowBytes)];
    }
}

static inline void split2_bytes_kernel(
    device const uint4* inputVector,
    device uint4* leftVector,
    device uint4* rightVector,
    constant uint& leftBytes,
    constant uint& totalBytes,
    uint index
) {
    uint base = index * 16;

    if (base + 15 < leftBytes) {
        leftVector[index] = inputVector[index];
        return;
    }

    if (base >= leftBytes && base + 15 < totalBytes && leftBytes % 16 == 0) {
        rightVector[(base - leftBytes) / 16] = inputVector[index];
        return;
    }

    device const uchar* input = reinterpret_cast<device const uchar*>(inputVector);
    device uchar* left = reinterpret_cast<device uchar*>(leftVector);
    device uchar* right = reinterpret_cast<device uchar*>(rightVector);

    for (uint offset = 0; offset < 16; offset++) {
        uint inputIndex = base + offset;

        if (inputIndex >= totalBytes) {
            continue;
        }

        if (inputIndex < leftBytes) {
            left[inputIndex] = input[inputIndex];
            continue;
        }

        right[inputIndex - leftBytes] = input[inputIndex];
    }
}

static inline void slice_bytes_kernel(
    device const uint4* inputVector,
    device uint4* outVector,
    constant uint& sliceLen,
    constant uint& inputDimSize,
    constant uint& innerBytes,
    constant uint& start,
    constant uint& outBytes,
    uint index
) {
    uint base = index * 16;

    if (base >= outBytes) {
        return;
    }

    uint blockBytes = sliceLen * innerBytes;
    uint inputBlockStride = inputDimSize * innerBytes;
    device const uchar* input = reinterpret_cast<device const uchar*>(inputVector);
    device uchar* out = reinterpret_cast<device uchar*>(outVector);

    if (innerBytes >= 16u && innerBytes % 16u == 0u) {
        uint outIndex = base;
        uint outerIdx = outIndex / blockBytes;
        uint within = outIndex - outerIdx * blockBytes;
        uint sliceCoord = within / innerBytes;
        uint innerOff = within - sliceCoord * innerBytes;
        uint inIndex = outerIdx * inputBlockStride + (start + sliceCoord) * innerBytes + innerOff;

        if (base + 15 < outBytes && innerOff + 16u <= innerBytes) {
            outVector[index] = inputVector[inIndex / 16u];
            return;
        }
    }

    for (uint offset = 0; offset < 16; offset++) {
        uint outIndex = base + offset;

        if (outIndex >= outBytes) {
            continue;
        }

        uint outerIdx = outIndex / blockBytes;
        uint within = outIndex - outerIdx * blockBytes;
        uint sliceCoord = within / innerBytes;
        uint innerOff = within - sliceCoord * innerBytes;
        uint inIndex = outerIdx * inputBlockStride + (start + sliceCoord) * innerBytes + innerOff;
        out[outIndex] = input[inIndex];
    }
}

static inline void last_token_bytes_kernel(
    device const uint4* inputVector,
    device uint4* outVector,
    constant uint& seq,
    constant uint& hiddenBytes,
    constant uint& outBytes,
    uint index
) {
    uint base = index * 16;
    uint batchIndex = base / hiddenBytes;
    uint hiddenOffset = base - batchIndex * hiddenBytes;
    uint inputBase = (batchIndex * seq + (seq - 1)) * hiddenBytes + hiddenOffset;

    if (base + 15 < outBytes && hiddenOffset + 15 < hiddenBytes && inputBase % 16 == 0) {
        outVector[index] = inputVector[inputBase / 16];
        return;
    }

    device const uchar* input = reinterpret_cast<device const uchar*>(inputVector);
    device uchar* out = reinterpret_cast<device uchar*>(outVector);

    for (uint offset = 0; offset < 16; offset++) {
        uint outIndex = base + offset;

        if (outIndex >= outBytes) {
            continue;
        }

        batchIndex = outIndex / hiddenBytes;
        hiddenOffset = outIndex - batchIndex * hiddenBytes;
        uint inputIndex = (batchIndex * seq + (seq - 1)) * hiddenBytes + hiddenOffset;
        out[outIndex] = input[inputIndex];
    }
}

template <typename Storage>
static inline void transpose2d_kernel(
    device const Storage* input,
    device Storage* out,
    constant uint& rows,
    constant uint& cols,
    uint index
) {
    uint elementCount = rows * cols;

    if (index >= elementCount) {
        return;
    }

    uint row = index / cols;
    uint col = index - row * cols;
    out[col * rows + row] = input[index];
}

template <typename Storage>
static inline void upsample_nearest2d_kernel(
    device const Storage* input,
    device Storage* out,
    constant uint& channels,
    constant uint& inHeight,
    constant uint& inWidth,
    constant uint& outHeight,
    constant uint& outWidth,
    constant uint& outElements,
    uint index
) {
    if (index >= outElements) {
        return;
    }

    uint outCol = index % outWidth;
    uint outRow = (index / outWidth) % outHeight;
    uint channel = (index / (outWidth * outHeight)) % channels;
    uint batch = index / (outWidth * outHeight * channels);
    uint inRow = outRow * inHeight / outHeight;
    uint inCol = outCol * inWidth / outWidth;
    uint inputElement = ((batch * channels + channel) * inHeight + inRow) * inWidth + inCol;
    out[index] = input[inputElement];
}

template <typename Storage>
static inline void merge_heads_kernel(
    device const Storage* input,
    device Storage* out,
    constant uint& batch,
    constant uint& seq,
    constant uint& heads,
    constant uint& headDim,
    uint index
) {
    uint headSpan = heads * headDim;

    if (index >= batch * seq * headSpan) {
        return;
    }

    uint headDimIndex = index % headDim;
    uint remainder = index / headDim;
    uint headIndex = remainder % heads;
    remainder /= heads;
    uint seqIndex = remainder % seq;
    uint batchIndex = remainder / seq;
    uint inIndex = ((batchIndex * seq + seqIndex) * heads + headIndex) * headDim + headDimIndex;
    out[index] = input[inIndex];
}

#define MERGE_HEADS_KERNEL(name, storage) \
kernel void name( \
    device const storage* input [[buffer(0)]], \
    device storage* out [[buffer(1)]], \
    constant uint& batch [[buffer(2)]], \
    constant uint& seq [[buffer(3)]], \
    constant uint& heads [[buffer(4)]], \
    constant uint& headDim [[buffer(5)]], \
    uint index [[thread_position_in_grid]] \
) { \
    merge_heads_kernel<storage>(input, out, batch, seq, heads, headDim, index); \
}

#define COPY_KERNEL(name) \
kernel void name( \
    device const uint4* inputVector [[buffer(0)]], \
    device uint4* outVector [[buffer(1)]], \
    constant uint& byteCount [[buffer(2)]], \
    uint index [[thread_position_in_grid]] \
) { \
    copy_bytes_kernel(inputVector, outVector, byteCount, index); \
}

#define CONCAT_KERNEL(name) \
kernel void name( \
    device const uint4* leftVector [[buffer(0)]], \
    device const uint4* rightVector [[buffer(1)]], \
    device uint4* outVector [[buffer(2)]], \
    constant uint& leftBytes [[buffer(3)]], \
    constant uint& totalBytes [[buffer(4)]], \
    uint index [[thread_position_in_grid]] \
) { \
    concat_bytes_kernel(leftVector, rightVector, outVector, leftBytes, totalBytes, index); \
}

#define CONCAT_LAST_DIM_KERNEL(name) \
kernel void name( \
    device const uchar* left [[buffer(0)]], \
    device const uchar* right [[buffer(1)]], \
    device uchar* out [[buffer(2)]], \
    constant uint& leftRowBytes [[buffer(3)]], \
    constant uint& rightRowBytes [[buffer(4)]], \
    constant uint& rowBytes [[buffer(5)]], \
    constant uint& totalBytes [[buffer(6)]], \
    uint index [[thread_position_in_grid]] \
) { \
    concat_last_dim_bytes_kernel( \
        left, right, out, leftRowBytes, rightRowBytes, rowBytes, totalBytes, index \
    ); \
}

#define SPLIT2_KERNEL(name) \
kernel void name( \
    device const uint4* inputVector [[buffer(0)]], \
    device uint4* leftVector [[buffer(1)]], \
    device uint4* rightVector [[buffer(2)]], \
    constant uint& leftBytes [[buffer(3)]], \
    constant uint& totalBytes [[buffer(4)]], \
    uint index [[thread_position_in_grid]] \
) { \
    split2_bytes_kernel(inputVector, leftVector, rightVector, leftBytes, totalBytes, index); \
}

#define LAST_TOKEN_KERNEL(name) \
kernel void name( \
    device const uint4* inputVector [[buffer(0)]], \
    device uint4* outVector [[buffer(1)]], \
    constant uint& seq [[buffer(2)]], \
    constant uint& hiddenBytes [[buffer(3)]], \
    constant uint& outBytes [[buffer(4)]], \
    uint index [[thread_position_in_grid]] \
) { \
    last_token_bytes_kernel(inputVector, outVector, seq, hiddenBytes, outBytes, index); \
}

#define SLICE_KERNEL(name) \
kernel void name( \
    device const uint4* inputVector [[buffer(0)]], \
    device uint4* outVector [[buffer(1)]], \
    constant uint& sliceLen [[buffer(2)]], \
    constant uint& inputDimSize [[buffer(3)]], \
    constant uint& innerBytes [[buffer(4)]], \
    constant uint& start [[buffer(5)]], \
    constant uint& outBytes [[buffer(6)]], \
    uint index [[thread_position_in_grid]] \
) { \
    slice_bytes_kernel( \
        inputVector, outVector, sliceLen, inputDimSize, innerBytes, start, outBytes, index \
    ); \
}

#define TRANSPOSE2D_KERNEL(name, storage) \
kernel void name( \
    device const storage* input [[buffer(0)]], \
    device storage* out [[buffer(1)]], \
    constant uint& rows [[buffer(2)]], \
    constant uint& cols [[buffer(3)]], \
    uint index [[thread_position_in_grid]] \
) { \
    transpose2d_kernel<storage>(input, out, rows, cols, index); \
}

#define UPSAMPLE_NEAREST2D_KERNEL(name, storage) \
kernel void name( \
    device const storage* input [[buffer(0)]], \
    device storage* out [[buffer(1)]], \
    constant uint& channels [[buffer(2)]], \
    constant uint& inHeight [[buffer(3)]], \
    constant uint& inWidth [[buffer(4)]], \
    constant uint& outHeight [[buffer(5)]], \
    constant uint& outWidth [[buffer(6)]], \
    constant uint& outElements [[buffer(7)]], \
    uint index [[thread_position_in_grid]] \
) { \
    upsample_nearest2d_kernel<storage>( \
        input, out, channels, inHeight, inWidth, outHeight, outWidth, outElements, index \
    ); \
}

#define GATHER_KERNEL(name, storage) \
kernel void name( \
    device const storage* source [[buffer(0)]], \
    device const int* indices [[buffer(1)]], \
    device storage* out [[buffer(2)]], \
    device atomic_uint* errorFlag [[buffer(3)]], \
    constant uint& sourceRows [[buffer(4)]], \
    constant uint& inner [[buffer(5)]], \
    constant uint& outRows [[buffer(6)]], \
    uint index [[thread_position_in_grid]] \
) { \
    gather_kernel<storage>(source, indices, out, errorFlag, sourceRows, inner, outRows, index); \
}

#define SCATTER_KERNEL(name, storage) \
kernel void name( \
    device const storage* target [[buffer(0)]], \
    device const int* indices [[buffer(1)]], \
    device const storage* updates [[buffer(2)]], \
    device storage* out [[buffer(3)]], \
    device atomic_uint* errorFlag [[buffer(4)]], \
    constant uint& targetRows [[buffer(5)]], \
    constant uint& inner [[buffer(6)]], \
    constant uint& updateRows [[buffer(7)]], \
    uint index [[thread_position_in_grid]] \
) { \
    scatter_kernel<storage>(target, indices, updates, out, errorFlag, targetRows, inner, updateRows, index); \
}

#define PAGE_WRITE_KERNEL(name, storage) \
kernel void name( \
    device const storage* storageRef [[buffer(0)]], \
    device const storage* values [[buffer(1)]], \
    device const int* pageIDs [[buffer(2)]], \
    device const int* offsets [[buffer(3)]], \
    device storage* out [[buffer(4)]], \
    device atomic_uint* errorFlag [[buffer(5)]], \
    constant uint& pageCount [[buffer(6)]], \
    constant uint& pageSize [[buffer(7)]], \
    constant uint& inner [[buffer(8)]], \
    constant uint& valueRows [[buffer(9)]], \
    constant uint& storageOffset [[buffer(10)]], \
    constant uint& outOffset [[buffer(11)]], \
    uint index [[thread_position_in_grid]] \
) { \
    page_write_kernel<storage>( \
        storageRef, values, pageIDs, offsets, out, errorFlag, \
        pageCount, pageSize, inner, valueRows, storageOffset, outOffset, index \
    ); \
}

#define PAGE_GATHER_KERNEL(name, storage) \
kernel void name( \
    device const storage* storageRef [[buffer(0)]], \
    device const int* pageTable [[buffer(1)]], \
    device storage* out [[buffer(2)]], \
    device atomic_uint* errorFlag [[buffer(3)]], \
    constant uint& pageCount [[buffer(4)]], \
    constant uint& pageSize [[buffer(5)]], \
    constant uint& inner [[buffer(6)]], \
    constant uint& outRows [[buffer(7)]], \
    constant uint& storageOffset [[buffer(8)]], \
    constant uint& outOffset [[buffer(9)]], \
    uint index [[thread_position_in_grid]] \
) { \
    page_gather_kernel<storage>( \
        storageRef, pageTable, out, errorFlag, \
        pageCount, pageSize, inner, outRows, storageOffset, outOffset, index \
    ); \
}

#define WHERE_KERNEL(name, scalar, vec) \
kernel void name( \
    device const uchar* mask [[buffer(0)]], \
    device const vec* positive [[buffer(1)]], \
    device const vec* negative [[buffer(2)]], \
    device vec* out [[buffer(3)]], \
    constant uint& count [[buffer(4)]], \
    uint index [[thread_position_in_grid]] \
) { \
    where_kernel<scalar, vec>(mask, positive, negative, out, count, index); \
}

#define MASKED_FILL_KERNEL(name, scalar, vec) \
kernel void name( \
    device const vec* input [[buffer(0)]], \
    device const uchar* mask [[buffer(1)]], \
    device const scalar* fill [[buffer(2)]], \
    device vec* out [[buffer(3)]], \
    constant uint& count [[buffer(4)]], \
    uint index [[thread_position_in_grid]] \
) { \
    masked_fill_kernel<scalar, vec>(input, mask, fill, out, count, index); \
}

#define TRANSPOSE_KERNEL(name, storage) \
kernel void name( \
    device const storage* input [[buffer(0)]], \
    device storage* out [[buffer(1)]], \
    device atomic_uint* errorFlag [[buffer(2)]], \
    constant uint& rank [[buffer(3)]], \
    constant uint& count [[buffer(4)]], \
    constant uint* permutation [[buffer(5)]], \
    constant uint* inputStrides [[buffer(6)]], \
    constant uint* outStrides [[buffer(7)]], \
    uint index [[thread_position_in_grid]] \
) { \
    transpose_kernel<storage>( \
        input, out, errorFlag, rank, count, permutation, inputStrides, outStrides, index \
    ); \
}

GATHER_KERNEL(gather_float32, uint)
GATHER_KERNEL(gather_float16, ushort)
GATHER_KERNEL(gather_bfloat16, ushort)

SCATTER_KERNEL(scatter_float32, uint)
SCATTER_KERNEL(scatter_float16, ushort)
SCATTER_KERNEL(scatter_bfloat16, ushort)

PAGE_WRITE_KERNEL(page_write_float32, uint)
PAGE_WRITE_KERNEL(page_write_float16, ushort)
PAGE_WRITE_KERNEL(page_write_bfloat16, ushort)

PAGE_GATHER_KERNEL(page_gather_float32, uint)
PAGE_GATHER_KERNEL(page_gather_float16, ushort)
PAGE_GATHER_KERNEL(page_gather_bfloat16, ushort)

WHERE_KERNEL(where_float32, uint, uint4)
WHERE_KERNEL(where_float16, ushort, ushort4)
WHERE_KERNEL(where_bfloat16, ushort, ushort4)

MASKED_FILL_KERNEL(masked_fill_float32, uint, uint4)
MASKED_FILL_KERNEL(masked_fill_float16, ushort, ushort4)
MASKED_FILL_KERNEL(masked_fill_bfloat16, ushort, ushort4)

TRANSPOSE_KERNEL(transpose_float32, uint)
TRANSPOSE_KERNEL(transpose_float16, ushort)
TRANSPOSE_KERNEL(transpose_bfloat16, ushort)

SLICE_KERNEL(slice_float32)
SLICE_KERNEL(slice_float16)
SLICE_KERNEL(slice_bfloat16)

COPY_KERNEL(copy_float32)
COPY_KERNEL(copy_float16)
COPY_KERNEL(copy_bfloat16)

CONCAT_KERNEL(concat_float32)
CONCAT_KERNEL(concat_float16)
CONCAT_KERNEL(concat_bfloat16)

CONCAT_LAST_DIM_KERNEL(concat_last_dim_float32)
CONCAT_LAST_DIM_KERNEL(concat_last_dim_float16)
CONCAT_LAST_DIM_KERNEL(concat_last_dim_bfloat16)

SPLIT2_KERNEL(split2_float32)
SPLIT2_KERNEL(split2_float16)
SPLIT2_KERNEL(split2_bfloat16)

LAST_TOKEN_KERNEL(last_token_float32)
LAST_TOKEN_KERNEL(last_token_float16)
LAST_TOKEN_KERNEL(last_token_bfloat16)

TRANSPOSE2D_KERNEL(transpose2d_float32, uint)
TRANSPOSE2D_KERNEL(transpose2d_float16, ushort)
TRANSPOSE2D_KERNEL(transpose2d_bfloat16, ushort)

UPSAMPLE_NEAREST2D_KERNEL(upsample_nearest2d_float32, uint)
UPSAMPLE_NEAREST2D_KERNEL(upsample_nearest2d_float16, ushort)
UPSAMPLE_NEAREST2D_KERNEL(upsample_nearest2d_bfloat16, ushort)

MERGE_HEADS_KERNEL(merge_heads_float32, uint)
MERGE_HEADS_KERNEL(merge_heads_float16, ushort)
MERGE_HEADS_KERNEL(merge_heads_bfloat16, ushort)
