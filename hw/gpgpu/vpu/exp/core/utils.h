/* Tools */
#ifndef UTILS_H
#define UTILS_H

#define BITMASK(bits) ((1ull << (bits)) - 1)
#define BITS(x, hi, lo) (((x) >> (lo)) & BITMASK((hi) - (lo) + 1)) // similar to x[hi:lo] in verilog
#define SEXT(x, len) ({ struct { int64_t n : len; } __x = { .n = x }; (uint64_t)__x.n; })
#ifndef ARRAY_SIZE
#define ARRAY_SIZE(arr) (sizeof(arr) / sizeof((arr)[0]))
#endif

/* ========== ANSI Color Codes ========== */

/* 前景色 (文字颜色) */
#define COLOR_RESET     "\x1b[0m"
#define COLOR_BLACK     "\x1b[30m"
#define COLOR_RED       "\x1b[31m"
#define COLOR_GREEN     "\x1b[32m"
#define COLOR_YELLOW    "\x1b[33m"
#define COLOR_BLUE      "\x1b[34m"
#define COLOR_MAGENTA   "\x1b[35m"
#define COLOR_CYAN      "\x1b[36m"
#define COLOR_WHITE     "\x1b[37m"

/* 亮色前景 */
#define COLOR_BRIGHT_BLACK   "\x1b[90m"
#define COLOR_BRIGHT_RED     "\x1b[91m"
#define COLOR_BRIGHT_GREEN   "\x1b[92m"
#define COLOR_BRIGHT_YELLOW  "\x1b[93m"
#define COLOR_BRIGHT_BLUE    "\x1b[94m"
#define COLOR_BRIGHT_MAGENTA "\x1b[95m"
#define COLOR_BRIGHT_CYAN    "\x1b[96m"
#define COLOR_BRIGHT_WHITE   "\x1b[97m"

/* 背景色 */
#define BG_BLACK     "\x1b[40m"
#define BG_RED       "\x1b[41m"
#define BG_GREEN     "\x1b[42m"
#define BG_YELLOW    "\x1b[43m"
#define BG_BLUE      "\x1b[44m"
#define BG_MAGENTA   "\x1b[45m"
#define BG_CYAN      "\x1b[46m"
#define BG_WHITE     "\x1b[47m"

/* 亮色背景 */
#define BG_BRIGHT_BLACK   "\x1b[100m"
#define BG_BRIGHT_RED     "\x1b[101m"
#define BG_BRIGHT_GREEN   "\x1b[102m"
#define BG_BRIGHT_YELLOW  "\x1b[103m"
#define BG_BRIGHT_BLUE    "\x1b[104m"
#define BG_BRIGHT_MAGENTA "\x1b[105m"
#define BG_BRIGHT_CYAN    "\x1b[106m"
#define BG_BRIGHT_WHITE   "\x1b[107m"

/* 样式 */
#define STYLE_BOLD      "\x1b[1m"
#define STYLE_DIM       "\x1b[2m"
#define STYLE_ITALIC    "\x1b[3m"
#define STYLE_UNDERLINE "\x1b[4m"
#define STYLE_BLINK     "\x1b[5m"
#define STYLE_REVERSE   "\x1b[7m"
#define STYLE_HIDDEN    "\x1b[8m"
#define STYLE_STRIKE    "\x1b[9m"

#endif /* UTILS_H */