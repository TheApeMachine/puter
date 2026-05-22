//go:build amd64

package shape

//go:noescape
func PageWriteFloat32AVX512Asm(
	storage *float32,
	values *float32,
	pageIDs *int32,
	offsets *int32,
	out *float32,
	pageCount int,
	pageSize int,
	inner int,
	valueRows int,
)

//go:noescape
func PageGatherFloat32AVX512Asm(
	storage *float32,
	pageTable *int32,
	out *float32,
	pageCount int,
	pageSize int,
	inner int,
	outRows int,
)

//go:noescape
func PageWriteUint16AVX512Asm(
	storage *uint16,
	values *uint16,
	pageIDs *int32,
	offsets *int32,
	out *uint16,
	pageCount int,
	pageSize int,
	inner int,
	valueRows int,
)

//go:noescape
func PageGatherUint16AVX512Asm(
	storage *uint16,
	pageTable *int32,
	out *uint16,
	pageCount int,
	pageSize int,
	inner int,
	outRows int,
)

//go:noescape
func PageWriteFloat32AVX2Asm(
	storage *float32,
	values *float32,
	pageIDs *int32,
	offsets *int32,
	out *float32,
	pageCount int,
	pageSize int,
	inner int,
	valueRows int,
)

//go:noescape
func PageGatherFloat32AVX2Asm(
	storage *float32,
	pageTable *int32,
	out *float32,
	pageCount int,
	pageSize int,
	inner int,
	outRows int,
)

//go:noescape
func PageWriteUint16AVX2Asm(
	storage *uint16,
	values *uint16,
	pageIDs *int32,
	offsets *int32,
	out *uint16,
	pageCount int,
	pageSize int,
	inner int,
	valueRows int,
)

//go:noescape
func PageGatherUint16AVX2Asm(
	storage *uint16,
	pageTable *int32,
	out *uint16,
	pageCount int,
	pageSize int,
	inner int,
	outRows int,
)

//go:noescape
func PageWriteFloat32SSE2Asm(
	storage *float32,
	values *float32,
	pageIDs *int32,
	offsets *int32,
	out *float32,
	pageCount int,
	pageSize int,
	inner int,
	valueRows int,
)

//go:noescape
func PageGatherFloat32SSE2Asm(
	storage *float32,
	pageTable *int32,
	out *float32,
	pageCount int,
	pageSize int,
	inner int,
	outRows int,
)

//go:noescape
func PageWriteUint16SSE2Asm(
	storage *uint16,
	values *uint16,
	pageIDs *int32,
	offsets *int32,
	out *uint16,
	pageCount int,
	pageSize int,
	inner int,
	valueRows int,
)

//go:noescape
func PageGatherUint16SSE2Asm(
	storage *uint16,
	pageTable *int32,
	out *uint16,
	pageCount int,
	pageSize int,
	inner int,
	outRows int,
)

func PageWriteFloat32AVX512(
	storage *float32,
	values *float32,
	pageIDs *int32,
	offsets *int32,
	out *float32,
	pageCount int,
	pageSize int,
	inner int,
	valueRows int,
) {
	PageWriteFloat32AVX512Asm(storage, values, pageIDs, offsets, out, pageCount, pageSize, inner, valueRows)
}

func PageGatherFloat32AVX512(
	storage *float32,
	pageTable *int32,
	out *float32,
	pageCount int,
	pageSize int,
	inner int,
	outRows int,
) {
	PageGatherFloat32AVX512Asm(storage, pageTable, out, pageCount, pageSize, inner, outRows)
}

func PageWriteUint16AVX512(
	storage *uint16,
	values *uint16,
	pageIDs *int32,
	offsets *int32,
	out *uint16,
	pageCount int,
	pageSize int,
	inner int,
	valueRows int,
) {
	PageWriteUint16AVX512Asm(storage, values, pageIDs, offsets, out, pageCount, pageSize, inner, valueRows)
}

func PageGatherUint16AVX512(
	storage *uint16,
	pageTable *int32,
	out *uint16,
	pageCount int,
	pageSize int,
	inner int,
	outRows int,
) {
	PageGatherUint16AVX512Asm(storage, pageTable, out, pageCount, pageSize, inner, outRows)
}

func PageWriteFloat32AVX2(
	storage *float32,
	values *float32,
	pageIDs *int32,
	offsets *int32,
	out *float32,
	pageCount int,
	pageSize int,
	inner int,
	valueRows int,
) {
	PageWriteFloat32AVX2Asm(storage, values, pageIDs, offsets, out, pageCount, pageSize, inner, valueRows)
}

func PageGatherFloat32AVX2(
	storage *float32,
	pageTable *int32,
	out *float32,
	pageCount int,
	pageSize int,
	inner int,
	outRows int,
) {
	PageGatherFloat32AVX2Asm(storage, pageTable, out, pageCount, pageSize, inner, outRows)
}

func PageWriteUint16AVX2(
	storage *uint16,
	values *uint16,
	pageIDs *int32,
	offsets *int32,
	out *uint16,
	pageCount int,
	pageSize int,
	inner int,
	valueRows int,
) {
	PageWriteUint16AVX2Asm(storage, values, pageIDs, offsets, out, pageCount, pageSize, inner, valueRows)
}

func PageGatherUint16AVX2(
	storage *uint16,
	pageTable *int32,
	out *uint16,
	pageCount int,
	pageSize int,
	inner int,
	outRows int,
) {
	PageGatherUint16AVX2Asm(storage, pageTable, out, pageCount, pageSize, inner, outRows)
}

func PageWriteFloat32SSE2(
	storage *float32,
	values *float32,
	pageIDs *int32,
	offsets *int32,
	out *float32,
	pageCount int,
	pageSize int,
	inner int,
	valueRows int,
) {
	PageWriteFloat32SSE2Asm(storage, values, pageIDs, offsets, out, pageCount, pageSize, inner, valueRows)
}

func PageGatherFloat32SSE2(
	storage *float32,
	pageTable *int32,
	out *float32,
	pageCount int,
	pageSize int,
	inner int,
	outRows int,
) {
	PageGatherFloat32SSE2Asm(storage, pageTable, out, pageCount, pageSize, inner, outRows)
}

func PageWriteUint16SSE2(
	storage *uint16,
	values *uint16,
	pageIDs *int32,
	offsets *int32,
	out *uint16,
	pageCount int,
	pageSize int,
	inner int,
	valueRows int,
) {
	PageWriteUint16SSE2Asm(storage, values, pageIDs, offsets, out, pageCount, pageSize, inner, valueRows)
}

func PageGatherUint16SSE2(
	storage *uint16,
	pageTable *int32,
	out *uint16,
	pageCount int,
	pageSize int,
	inner int,
	outRows int,
) {
	PageGatherUint16SSE2Asm(storage, pageTable, out, pageCount, pageSize, inner, outRows)
}
