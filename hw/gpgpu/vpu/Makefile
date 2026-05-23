# VPU standalone build — fully self-contained (zero external dependencies)
# Can be moved out of the QEMU tree as a whole directory.

TARGET   = build/libvpu.a
BUILDDIR = build
SRCS     = core/gpgpu_core.c core/memory.c core/proto.c core/lpfp.c ring/ring.c \
           softfloat/softfloat.c
OBJS     = $(SRCS:%.c=$(BUILDDIR)/%.o)
CFLAGS   = -Wall -Wextra -Wno-unused-parameter -O2 -g -std=gnu11 -I. -Icore -Iring -Iproto -Isoftfloat

$(TARGET): $(OBJS)
	ar rcs $@ $^

$(BUILDDIR)/%.o: %.c
	@mkdir -p $(dir $@)
	$(CC) $(CFLAGS) -c -o $@ $<

gen:
	python3 proto/main.py proto/

clean:
	rm -rf $(BUILDDIR) $(TARGET)

.PHONY: clean gen
