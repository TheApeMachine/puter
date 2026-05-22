package shape

type f32CopyContiguousKernelImpl struct {
	kernel    func(dst, src *float32, count int)
	name      string
	available bool
}

type f32WhereKernelImpl struct {
	kernel    func(dst, positive, negative *float32, mask []byte, count int)
	name      string
	available bool
}

type f32MaskedFillKernelImpl struct {
	kernel    func(dst, input *float32, fill float32, mask []byte, count int)
	name      string
	available bool
}

type f32PageWriteKernelImpl struct {
	kernel func(
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
	name      string
	available bool
}

type f32PageGatherKernelImpl struct {
	kernel func(
		storage *float32,
		pageTable *int32,
		out *float32,
		pageCount int,
		pageSize int,
		inner int,
		outRows int,
	)
	name      string
	available bool
}

type u16PageWriteKernelImpl struct {
	kernel func(
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
	name      string
	available bool
}

type u16PageGatherKernelImpl struct {
	kernel func(
		storage *uint16,
		pageTable *int32,
		out *uint16,
		pageCount int,
		pageSize int,
		inner int,
		outRows int,
	)
	name      string
	available bool
}

func pickF32CopyContiguousKernel(
	candidates []f32CopyContiguousKernelImpl,
) func(dst, src *float32, count int) {
	for _, candidate := range candidates {
		if candidate.available {
			return candidate.kernel
		}
	}

	panic("shape: no float32 copy-contiguous kernel available")
}

func pickF32WhereKernel(
	candidates []f32WhereKernelImpl,
) func(dst, positive, negative *float32, mask []byte, count int) {
	for _, candidate := range candidates {
		if candidate.available {
			return candidate.kernel
		}
	}

	panic("shape: no float32 where kernel available")
}

func pickF32MaskedFillKernel(
	candidates []f32MaskedFillKernelImpl,
) func(dst, input *float32, fill float32, mask []byte, count int) {
	for _, candidate := range candidates {
		if candidate.available {
			return candidate.kernel
		}
	}

	panic("shape: no float32 masked-fill kernel available")
}

func pickF32PageWriteKernel(candidates []f32PageWriteKernelImpl) f32PageWriteKernelImpl {
	for _, candidate := range candidates {
		if candidate.available {
			return candidate
		}
	}

	panic("shape: no float32 page-write kernel available")
}

func pickF32PageGatherKernel(candidates []f32PageGatherKernelImpl) f32PageGatherKernelImpl {
	for _, candidate := range candidates {
		if candidate.available {
			return candidate
		}
	}

	panic("shape: no float32 page-gather kernel available")
}

func pickU16PageWriteKernel(candidates []u16PageWriteKernelImpl) u16PageWriteKernelImpl {
	for _, candidate := range candidates {
		if candidate.available {
			return candidate
		}
	}

	panic("shape: no uint16 page-write kernel available")
}

func pickU16PageGatherKernel(candidates []u16PageGatherKernelImpl) u16PageGatherKernelImpl {
	for _, candidate := range candidates {
		if candidate.available {
			return candidate
		}
	}

	panic("shape: no uint16 page-gather kernel available")
}
