# VPU standalone build (host)
TARGET  = libvpu.a
SRCS    = core/gpgpu_core.c core/memory.c core/proto.c core/lpfp.c ring/ring.c
OBJS    = $(SRCS:.c=.o)
CFLAGS  = -Wall -Wextra -O2 -g -std=c11 -I. -Icore -Iring -Iproto

$(TARGET): $(OBJS)
	ar rcs $@ $^

%.o: %.c
	$(CC) $(CFLAGS) -c -o $@ $<

gen:
	python3 proto/main.py proto/

clean:
	rm -f $(OBJS) $(TARGET)
