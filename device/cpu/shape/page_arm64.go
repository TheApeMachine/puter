//go:build arm64

package shape

//go:noescape
func PageWriteFloat32NEONAsm(
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
func PageGatherFloat32NEONAsm(
	storage *float32,
	pageTable *int32,
	out *float32,
	pageCount int,
	pageSize int,
	inner int,
	outRows int,
)

//go:noescape
func PageWriteUint16NEONAsm(
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
func PageGatherUint16NEONAsm(
	storage *uint16,
	pageTable *int32,
	out *uint16,
	pageCount int,
	pageSize int,
	inner int,
	outRows int,
)

func PageWriteFloat32NEON(
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
	PageWriteFloat32NEONAsm(storage, values, pageIDs, offsets, out, pageCount, pageSize, inner, valueRows)
}

func PageGatherFloat32NEON(
	storage *float32,
	pageTable *int32,
	out *float32,
	pageCount int,
	pageSize int,
	inner int,
	outRows int,
) {
	PageGatherFloat32NEONAsm(storage, pageTable, out, pageCount, pageSize, inner, outRows)
}

func PageWriteUint16NEON(
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
	PageWriteUint16NEONAsm(storage, values, pageIDs, offsets, out, pageCount, pageSize, inner, valueRows)
}

func PageGatherUint16NEON(
	storage *uint16,
	pageTable *int32,
	out *uint16,
	pageCount int,
	pageSize int,
	inner int,
	outRows int,
) {
	PageGatherUint16NEONAsm(storage, pageTable, out, pageCount, pageSize, inner, outRows)
}
