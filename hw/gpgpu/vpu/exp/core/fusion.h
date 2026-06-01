/*
 * fusion.h — DFG pattern-matching instruction fusion
 *
 * Copyright (c) 2024-2025
 * Licensed under GPL v2 or later.
 */

#ifndef FUSION_H
#define FUSION_H

#include "predecode.h"

ThOp *fusion_pass(ThOp *code_in, int tcount_in, int *tcount_out);

#endif /* FUSION_H */
