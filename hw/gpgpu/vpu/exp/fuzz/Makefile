CC=gcc
CFLAGS=-Wall -Wextra -Og -g -I.. -I../core -I../proto -I/usr/include/lua5.4
LDFLAGS=-lm -lpthread -ldl -llua5.4
SRCS=main.c fuzzer.c ../core/engine.c ../core/scheduler.c \
     ../core/decode_trie.c ../core/lpfp.c ../core/memory.c ../core/proto.c ../config.c
all: fuzz_test
	./fuzz_test -n 1000
fuzz_test: $(SRCS)
	$(CC) $(CFLAGS) -o $@ $(SRCS) $(LDFLAGS)
clean:
	rm -f fuzz_test
.PHONY: all clean
