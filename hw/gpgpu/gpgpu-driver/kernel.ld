OUTPUT_FORMAT("elf64-littleriscv", "elf64-littleriscv", "elf64-littleriscv")
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
    
    /* 丢弃不需要的节 */
    /DISCARD/ : {
        *(.comment)
        *(.note)
        *(.eh_frame)
    }
}