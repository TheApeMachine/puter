.PHONY: test bench

# qpool uses go:linkname for ScheduleFast goroutine parking.
LDFLAGS := -ldflags='-checklinkname=0'

DUMP ?= puter.txt

test:
	go test $(LDFLAGS) ./...

bench:
	go test $(LDFLAGS) -bench=. ./...

dump:
	python3 "$(CURDIR)/scripts/dump-repo.py" "$(DUMP)"
