OUTPUT_FORMAT("elf32-littleriscv", "elf32-littleriscv", "elf32-littleriscv")
OUTPUT_ARCH(riscv)
ENTRY(_start)

PHDRS
{
    text   PT_LOAD FLAGS(5);   /* R-X */
    data   PT_LOAD FLAGS(6);   /* RW- */
}

SECTIONS
{
    . = 0x0;
    .text : {
        *(.text)
        *(.text.*)
    } :text

    . = ALIGN(4);
    .data : {
        *(.rodata)
        *(.data)
    } :data

    .bss : {
        *(.bss)
    } :data

    /DISCARD/ : {
        *(.comment)
        *(.note)
        *(.eh_frame)
    }
}
