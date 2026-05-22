package shape

import "unsafe"

func pageWriteF32Generic(
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
	storageView := unsafe.Slice(storage, pageCount*pageSize*inner)
	valueView := unsafe.Slice(values, valueRows*inner)
	pageIDView := unsafe.Slice(pageIDs, valueRows)
	offsetView := unsafe.Slice(offsets, valueRows)
	outView := unsafe.Slice(out, pageCount*pageSize*inner)

	copy(outView, storageView)

	for rowIndex := range valueRows {
		storageRow := (int(pageIDView[rowIndex])*pageSize + int(offsetView[rowIndex])) * inner
		copy(outView[storageRow:storageRow+inner], valueView[rowIndex*inner:(rowIndex+1)*inner])
	}
}

func pageGatherF32Generic(
	storage *float32,
	pageTable *int32,
	out *float32,
	pageCount int,
	pageSize int,
	inner int,
	outRows int,
) {
	storageView := unsafe.Slice(storage, pageCount*pageSize*inner)
	pageTableView := unsafe.Slice(pageTable, (outRows+pageSize-1)/pageSize)
	outView := unsafe.Slice(out, outRows*inner)

	for rowIndex := range outRows {
		tableIndex := rowIndex / pageSize
		pageOffset := rowIndex % pageSize
		storageRow := (int(pageTableView[tableIndex])*pageSize + pageOffset) * inner
		copy(outView[rowIndex*inner:(rowIndex+1)*inner], storageView[storageRow:storageRow+inner])
	}
}

func pageWriteU16Generic(
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
	storageView := unsafe.Slice(storage, pageCount*pageSize*inner)
	valueView := unsafe.Slice(values, valueRows*inner)
	pageIDView := unsafe.Slice(pageIDs, valueRows)
	offsetView := unsafe.Slice(offsets, valueRows)
	outView := unsafe.Slice(out, pageCount*pageSize*inner)

	copy(outView, storageView)

	for rowIndex := range valueRows {
		storageRow := (int(pageIDView[rowIndex])*pageSize + int(offsetView[rowIndex])) * inner
		copy(outView[storageRow:storageRow+inner], valueView[rowIndex*inner:(rowIndex+1)*inner])
	}
}

func pageGatherU16Generic(
	storage *uint16,
	pageTable *int32,
	out *uint16,
	pageCount int,
	pageSize int,
	inner int,
	outRows int,
) {
	storageView := unsafe.Slice(storage, pageCount*pageSize*inner)
	pageTableView := unsafe.Slice(pageTable, (outRows+pageSize-1)/pageSize)
	outView := unsafe.Slice(out, outRows*inner)

	for rowIndex := range outRows {
		tableIndex := rowIndex / pageSize
		pageOffset := rowIndex % pageSize
		storageRow := (int(pageTableView[tableIndex])*pageSize + pageOffset) * inner
		copy(outView[rowIndex*inner:(rowIndex+1)*inner], storageView[storageRow:storageRow+inner])
	}
}
