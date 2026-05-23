"""parser.py — binary ring frame parser (level=0 instruction trace + level=1 control events)."""

import struct

EVENT_TABLE = {
    1: "REG_WRITE",
    2: "REG_READ",
    3: "DMA_START",
    4: "DMA_COMPLETE",
    5: "KERNEL_DISPATCH",
    6: "KERNEL_COMPLETE",
    7: "IRQ_FIRE",
    8: "ERROR_EVENT",
    9: "STATE_CHANGE",
}


def _extract_fields(word, table):
    result = {}
    for name, start, width in table:
        mask = (1 << width) - 1
        result[name] = (word >> start) & mask
    return result


_LEVEL0_TABLE = [
    ("level",  28, 4),
    ("nargs",  24, 4),
    ("opcode", 16, 8),
    ("branch",  0, 1),
]

_LEVEL1_TABLE = [
    ("level",  28, 4),
    ("nargs",  24, 4),
    ("evt_num", 16, 8),
]


def _parse_one(header_word, operands):
    level = (header_word >> 28) & 0xF

    if level == 0:
        fields = _extract_fields(header_word, _LEVEL0_TABLE)
        record = {
            "level":    fields["level"],
            "nargs":    fields["nargs"],
            "opcode":   fields["opcode"],
            "branch":   bool(fields["branch"]),
            "operands": operands,
        }
    else:
        fields = _extract_fields(header_word, _LEVEL1_TABLE)
        evt_num = fields["evt_num"]
        record = {
            "level":    fields["level"],
            "nargs":    fields["nargs"],
            "opcode":   evt_num,
            "event":    EVENT_TABLE.get(evt_num),
            "operands": operands,
        }

    return record


def parse_frame(data):
    """Parse a binary frame into a list of structured record dicts.

    Frame format: consecutive records of [header_word(4B) + nargs*4B operands].

    Returns list of dicts, each with keys:
      level, nargs, opcode, operands[, branch][, event]
    """
    records = []
    offset = 0

    while offset + 4 <= len(data):
        header_word = struct.unpack_from("<I", data, offset)[0]
        offset += 4

        nargs = (header_word >> 24) & 0xF

        operands = []
        for _ in range(nargs):
            if offset + 4 > len(data):
                break
            operands.append(struct.unpack_from("<I", data, offset)[0])
            offset += 4

        records.append(_parse_one(header_word, operands))

    return records
