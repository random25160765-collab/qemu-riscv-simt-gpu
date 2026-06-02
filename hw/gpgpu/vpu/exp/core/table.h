/*
 * table.h — simple auto-width table printer
 */
#ifndef TABLE_H
#define TABLE_H

#include <stdint.h>
#include <stdbool.h>

#define TABLE_MAX_COLS 16
#define TABLE_MAX_ROWS 256

typedef struct {
    const char *header;
    int width;  /* 0 = auto */
    bool right; /* right-align numbers */
    int color;  /* -1=none, 0-7=ANSI fg for header */
} ColDef;

typedef struct {
    ColDef cols[TABLE_MAX_COLS];
    char *data[TABLE_MAX_ROWS][TABLE_MAX_COLS];
    int n_cols, n_rows;
} Table;

void table_init(Table *t, const ColDef *defs, int n);
void table_row(Table *t, const char **vals);
void table_render(const Table *t);

#endif
