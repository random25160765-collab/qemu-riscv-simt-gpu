/*
 * config.c — Lua-based VPU configuration loader
 *
 * Copyright (c) 2024-2025
 * Licensed under GPL v2 or later.
 */

#include <stdio.h>
#include <stdlib.h>
#include <lua.h>
#include <lualib.h>
#include <lauxlib.h>
#include "config.h"

/* helper: read int from table, return default if missing */
static int lint(lua_State *L, const char *key, int def)
{
    lua_getfield(L, -1, key);
    int v = lua_isinteger(L, -1) ? (int)lua_tointeger(L, -1) : def;
    lua_pop(L, 1);
    return v;
}

static bool lbool(lua_State *L, const char *key, bool def)
{
    lua_getfield(L, -1, key);
    bool v = lua_isboolean(L, -1) ? lua_toboolean(L, -1) : def;
    lua_pop(L, 1);
    return v;
}

void vpu_config_default(vpu_config_t *c)
{
    *c = (vpu_config_t){
            .num_cus = 1,
            .warps_per_cu = 1,
            .warp_size = 32,
            .vram_mb = 64,
            .features =
                    {.vpu = true, .tcu = false, .sfu = true, .lp = true, .debug = false, .trace = false, .perf = true},
    };
}

int vpu_config_load(vpu_config_t *c, const char *path)
{
    lua_State *L = luaL_newstate();
    if (!L) return -1;
    luaL_openlibs(L);

    if (luaL_loadfile(L, path) || lua_pcall(L, 0, 1, 0)) {
        fprintf(stderr, "Lua: %s — using defaults\n", lua_tostring(L, -1));
        lua_close(L);
        vpu_config_default(c);
        return 1;
    }
    if (!lua_istable(L, -1)) {
        lua_close(L);
        return -1;
    }

    /* device */
    lua_getfield(L, -1, "device");
    if (lua_istable(L, -1)) {
        c->num_cus = (uint32_t)lint(L, "num_cus", 1);
        c->warps_per_cu = (uint32_t)lint(L, "warps_per_cu", 1);
        c->warp_size = (uint32_t)lint(L, "warp_size", 32);
        c->vram_mb = (uint64_t)lint(L, "vram_mb", 64);
    }
    lua_pop(L, 1);

    /* features */
    lua_getfield(L, -1, "features");
    if (lua_istable(L, -1)) {
        c->features.vpu = lbool(L, "vpu", true);
        c->features.tcu = lbool(L, "tcu", false);
        c->features.sfu = lbool(L, "sfu", true);
        c->features.lp = lbool(L, "lp", true);
        c->features.debug = lbool(L, "debug", false);
        c->features.trace = lbool(L, "trace", false);
        c->features.perf = lbool(L, "perf", true);
    }
    lua_pop(L, 1);

    lua_close(L);
    return 0;
}
