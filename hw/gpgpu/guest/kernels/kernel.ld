OUTPUT_FORMAT("elf32-littleriscv", "elf32-littleriscv", "elf32-littleriscv")
OUTPUT_ARCH(riscv)
ENTRY(_start)

SECTIONS
{
    . = 0x0;
    .text : {
        *(.text)
        *(.text.*)
    }
    .data : {
        *(.data)
        *(.rodata)
    }
    .bss : {
        *(.bss)
    }
    
    /DISCARD/ : {
        *(.comment)
        *(.note)
        *(.eh_frame)
    }
}
