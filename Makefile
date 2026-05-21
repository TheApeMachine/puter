.PHONY: test bench

# qpool uses go:linkname for ScheduleFast goroutine parking.
LDFLAGS := -ldflags='-checklinkname=0'

test:
	go test $(LDFLAGS) ./...

bench:
	go test $(LDFLAGS) -bench=. ./...
