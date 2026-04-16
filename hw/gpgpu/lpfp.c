#include "lpfp.h"

/* ========== BF16 ========== */

uint16_t f32_to_bf16(uint32_t f32_bits) {
    uint32_t sign = f32_bits & 0x80000000;
    uint32_t exp = (f32_bits >> 23) & 0xFF;
    uint32_t mant = f32_bits & 0x7FFFFF;
    
    if (exp == 0xFF) {
        if (mant == 0) {
            return sign ? 0xFF80 : 0x7F80;
        }
        return 0x7FFF;
    }
    
    if (exp == 0) {
        if (mant == 0) {
            return sign ? 0x8000 : 0x0000;
        }
        int clz = __builtin_clz(mant) - 8;
        mant <<= clz;
        exp = 1 - clz;
    }
    
    uint32_t bf16_mant = mant >> 16;
    uint32_t rounding_bit = (mant >> 15) & 1;
    uint32_t sticky_bit = (mant & 0x7FFF) ? 1 : 0;
    
    if (rounding_bit && (sticky_bit || (bf16_mant & 1))) {
        bf16_mant++;
        if (bf16_mant > 0x7F) {
            bf16_mant = 0;
            exp++;
        }
    }
    
    if (exp >= 0xFF) {
        return sign ? 0xFF80 : 0x7F80;
    }
    
    return (sign >> 16) | (exp << 7) | bf16_mant;
}

uint32_t bf16_to_f32(uint16_t bf) {
    return ((uint32_t)bf << 16);
}

/* ========== E4M3 ========== */

// 辅助函数：uint32_t 位模式 → C float
static inline float f32_bits_to_float(uint32_t bits) {
    float f;
    memcpy(&f, &bits, sizeof(float));
    return f;
}

// 辅助函数：C float → uint32_t 位模式
static inline uint32_t float_to_f32_bits(float f) {
    uint32_t bits;
    memcpy(&bits, &f, sizeof(float));
    return bits;
}

uint8_t f32_to_e4m3(uint32_t f32_bits) {
    uint32_t sign = (f32_bits >> 31) & 1;
    uint32_t exp = (f32_bits >> 23) & 0xFF;
    uint32_t mant = f32_bits & 0x7FFFFF;
    
    if (exp == 0xFF) {
        return sign ? 0xFF : 0x7F;
    }
    
    if (exp == 0 && mant == 0) {
        return sign ? 0x80 : 0x00;
    }
    
    float abs_f = fabsf(f32_bits_to_float(f32_bits));
    
    const float max_val = 448.0f;
    if (abs_f >= max_val) {
        return sign ? 0xFF : 0x7F;
    }
    
    const float min_normal = 0.001953125f;
    
    if (abs_f < min_normal) {
        if (abs_f < min_normal / 16.0f) {
            return sign ? 0x80 : 0x00;
        }
        uint8_t best_idx = 0;
        float best_diff = abs_f;
        for (int i = 1; i < 8; i++) {
            float val = (i / 512.0f);
            float diff = fabsf(val - abs_f);
            if (diff < best_diff) {
                best_diff = diff;
                best_idx = i;
            }
        }
        return sign ? (0x80 | best_idx) : best_idx;
    }
    
    uint8_t best_encoding = 0;
    float best_diff = INFINITY;
    
    for (int e = 0; e < 16; e++) {
        int actual_exp = e - 7;
        if (actual_exp < -7 || actual_exp > 8) continue;
        
        float exp_val = powf(2.0f, actual_exp);
        
        for (int m = 0; m < 8; m++) {
            float val;
            if (e == 0) {
                if (m == 0) val = 0.0f;
                else val = (m / 8.0f) * powf(2.0f, -6);
            } else {
                val = (1.0f + m / 8.0f) * exp_val;
            }
            
            if (val > 448.0f) continue;
            
            float diff = fabsf(val - abs_f);
            if (diff < best_diff) {
                best_diff = diff;
                best_encoding = (e << 3) | m;
            }
        }
    }
    
    return sign ? (0x80 | best_encoding) : best_encoding;
}

uint32_t e4m3_to_f32(uint8_t e4m3) {
    uint8_t sign = (e4m3 >> 7) & 1;
    uint8_t exp = (e4m3 >> 3) & 0xF;
    uint8_t mant = e4m3 & 0x7;
    
    float result;
    
    if (exp == 0) {
        if (mant == 0) {
            result = 0.0f;
        } else {
            result = (mant / 8.0f) * powf(2.0f, -6);
        }
    } else if (exp == 0xF) {
        if (mant == 0x7) {
            result = 448.0f;
        } else {
            result = NAN;
        }
    } else {
        result = (1.0f + mant / 8.0f) * powf(2.0f, (int)exp - 7);
    }
    
    if (sign) result = -result;
    return float_to_f32_bits(result);
}

/* ========== E5M2 ========== */

uint8_t f32_to_e5m2(uint32_t f32_bits) {
    uint32_t sign = (f32_bits >> 31) & 1;
    uint32_t exp = (f32_bits >> 23) & 0xFF;
    uint32_t mant = f32_bits & 0x7FFFFF;
    
    if (exp == 0xFF) {
        if (mant == 0) {
            return sign ? 0xFC : 0x7C;
        }
        return 0x7E;
    }
    
    if (exp == 0 && mant == 0) {
        return sign ? 0x80 : 0x00;
    }
    
    float abs_f = fabsf(f32_bits_to_float(f32_bits));
    
    const float max_val = 57344.0f;
    if (abs_f >= max_val) {
        return sign ? 0xFC : 0x7C;
    }
    
    uint8_t best_encoding = 0;
    float best_diff = INFINITY;
    
    for (int e = 0; e < 32; e++) {
        int actual_exp = e - 15;
        if (actual_exp < -14 || actual_exp > 15) continue;
        
        float exp_val = powf(2.0f, actual_exp);
        
        for (int m = 0; m < 4; m++) {
            float val;
            if (e == 0) {
                val = (m / 4.0f) * powf(2.0f, -14);
            } else {
                val = (1.0f + m / 4.0f) * exp_val;
            }
            
            if (val > 57344.0f) continue;
            
            float diff = fabsf(val - abs_f);
            if (diff < best_diff) {
                best_diff = diff;
                best_encoding = (e << 2) | m;
            }
        }
    }
    
    return sign ? (0x80 | best_encoding) : best_encoding;
}

uint32_t e5m2_to_f32(uint8_t e5m2) {
    uint8_t sign = (e5m2 >> 7) & 1;
    uint8_t exp = (e5m2 >> 2) & 0x1F;
    uint8_t mant = e5m2 & 0x3;
    
    float result;
    
    if (exp == 0) {
        result = (mant / 4.0f) * powf(2.0f, -14);
    } else if (exp == 0x1F) {
        if (mant == 0) {
            result = INFINITY;
        } else {
            result = NAN;
        }
    } else {
        result = (1.0f + mant / 4.0f) * powf(2.0f, (int)exp - 15);
    }
    
    if (sign) result = -result;
    return float_to_f32_bits(result);
}

/* ========== E2M1 ========== */

uint8_t f32_to_e2m1(uint32_t f32_bits) {
    uint32_t sign = (f32_bits >> 31) & 1;
    uint32_t exp = (f32_bits >> 23) & 0xFF;
    
    if (exp == 0xFF) {
        return sign ? 0x8F : 0x07;
    }
    
    float abs_f = fabsf(f32_bits_to_float(f32_bits));
    
    const float max_val = 6.0f;
    if (abs_f >= max_val) {
        return sign ? 0x8F : 0x07;
    }
    
    uint8_t best_encoding = 0;
    float best_diff = INFINITY;
    
    for (int e = 0; e < 4; e++) {
        int actual_exp = e - 1;
        if (actual_exp < -1 || actual_exp > 2) continue;
        
        float exp_val = powf(2.0f, actual_exp);
        
        for (int m = 0; m < 2; m++) {
            float val;
            if (e == 0) {
                if (m == 0) val = 0.0f;
                else continue;
            } else {
                val = (1.0f + m / 2.0f) * exp_val;
            }
            
            if (val > 6.0f) continue;
            
            float diff = fabsf(val - abs_f);
            if (diff < best_diff) {
                best_diff = diff;
                best_encoding = (e << 1) | m;
            }
        }
    }
    
    if (fabsf(6.0f - abs_f) < best_diff) {
        best_encoding = 0x7;
    }
    
    return sign ? (0x80 | best_encoding) : best_encoding;
}

uint32_t e2m1_to_f32(uint8_t e2m1) {
    uint8_t sign = (e2m1 >> 3) & 1;
    uint8_t exp = (e2m1 >> 1) & 0x3;
    uint8_t mant = e2m1 & 0x1;
    
    float result;
    
    if (exp == 0) {
        result = 0.0f;
    } else {
        if (exp == 3 && mant == 1) {
            result = 6.0f;
        } else {
            result = (1.0f + mant / 2.0f) * powf(2.0f, (int)exp - 1);
        }
    }
    
    if (sign) result = -result;
    return float_to_f32_bits(result);
}