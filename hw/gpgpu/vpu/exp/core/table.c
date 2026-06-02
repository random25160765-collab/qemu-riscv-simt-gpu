#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "table.h"
#include "utils.h" /* STYLE_BOLD, STYLE_DIM, COLOR_* */

void table_init(Table *t, const ColDef *defs, int n)
{
    t->n_cols = n;
    t->n_rows = 0;
    memcpy(t->cols, defs, n * sizeof(ColDef));
}

void table_row(Table *t, const char **vals)
{
    if (t->n_rows >= TABLE_MAX_ROWS) return;
    for (int c = 0; c < t->n_cols; c++)
        t->data[t->n_rows][c] = vals[c] ? strdup(vals[c]) : strdup("");
    t->n_rows++;
}

void table_render(const Table *t)
{
    if (t->n_rows == 0) return;

    /* calculate column widths */
    int w[TABLE_MAX_COLS];
    for (int c = 0; c < t->n_cols; c++) {
        w[c] = t->cols[c].width > 0 ? t->cols[c].width : (int)strlen(t->cols[c].header);
        for (int r = 0; r < t->n_rows; r++)
            if (t->data[r][c]) {
                int l = (int)strlen(t->data[r][c]);
                if (l > w[c]) w[c] = l;
            }
    }

    /* header */
    printf("  ");
    for (int c = 0; c < t->n_cols; c++) {
        if (c) printf("  ");
        printf(STYLE_BOLD "%-*s" COLOR_RESET, w[c] + (t->cols[c].right ? 0 : 0), t->cols[c].header);
    }
    printf("\n");

    /* separator */
    printf("  " STYLE_DIM);
    for (int c = 0; c < t->n_cols; c++) {
        if (c) printf("  ");
        for (int i = 0; i < w[c]; i++)
            putchar('-');
    }
    printf(COLOR_RESET "\n");

    /* rows */
    for (int r = 0; r < t->n_rows; r++) {
        printf("  ");
        for (int c = 0; c < t->n_cols; c++) {
            if (c) printf("  ");
            const char *v = t->data[r][c] ? t->data[r][c] : "";
            if (t->cols[c].right)
                printf("%*s", w[c], v);
            else
                printf("%-*s", w[c], v);
        }
        printf("\n");
    }
}
