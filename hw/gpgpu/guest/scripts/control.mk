# GPGPU Project Control Makefile
# ============================================================
# !! Only use in guest

install: $(DRIVER_KO)
	@sudo rmmod gpgpu 2>/dev/null || true
	@sudo insmod $(DRIVER_KO)
	@sudo chmod a+rw /dev/gpgpu0 2>/dev/null || true
	@echo "Driver loaded."

remove:
	@sudo rmmod gpgpu 2>/dev/null || true
	@echo "Driver removed."

dmesg:
	@sudo dmesg -w | grep -E "gpgpu|KERNEL|DEVICE"

dmesg-clear:
	@sudo dmesg -c > /dev/null

status:
	@echo "=== GPGPU Status ==="
	@ls -la /dev/gpgpu* 2>/dev/null || echo "No device nodes"
	@lsmod | grep gpgpu || echo "Driver not loaded"
	@lspci -d 1234:1337 2>/dev/null || echo "Device not found"

.PHONY: install remove dmesg dmesg-clear status