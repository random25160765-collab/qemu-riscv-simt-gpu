# VPU standalone build — fully self-contained (zero external dependencies)
# Can be moved out of the QEMU tree as a whole directory.

TARGET   = build/libvpu.a
VPUEXE   = build/vpu
BUILDDIR = build
SRCS     = core/gpgpu_core.c core/memory.c core/proto.c core/lpfp.c ring/ring.c \
           socket/probe.c softfloat/softfloat.c
OBJS     = $(SRCS:%.c=$(BUILDDIR)/%.o)
CFLAGS   = -Wall -Wextra -Wno-unused-parameter -O2 -g -std=gnu11 -pthread -I. -Icore -Iring -Iproto -Isoftfloat
LDFLAGS  = -pthread -lm -lrt

all: $(TARGET) $(VPUEXE)

$(TARGET): $(OBJS)
	ar rcs $@ $^

$(VPUEXE): $(TARGET) $(BUILDDIR)/main.o
	$(CC) -o $@ $(BUILDDIR)/main.o -L$(BUILDDIR) -lvpu -lm -lrt

$(BUILDDIR)/%.o: %.c
	@mkdir -p $(dir $@)
	$(CC) $(CFLAGS) -c -o $@ $<

gen:
	python3 proto/main.py proto/

clean:
	rm -rf $(BUILDDIR) $(TARGET)

.PHONY: all clean gen
