package pool

func requirePoolExtents(
	batch, channels, inHeight, inWidth, outHeight, outWidth int,
) {
	if batch*channels*inHeight*inWidth == 0 ||
		batch*channels*outHeight*outWidth == 0 {
		panic("pool: zero tensor extent")
	}
}
