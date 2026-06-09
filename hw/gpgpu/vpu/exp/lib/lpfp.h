#ifndef LPFP_H
#define LPFP_H

#include <math.h>
#include <string.h>
#include <stdint.h>
#include <stdbool.h>

uint16_t f32_to_bf16(uint32_t f32_bits);
uint32_t bf16_to_f32(uint16_t bf);
uint8_t f32_to_e4m3(uint32_t f32_bits);
uint32_t e4m3_to_f32(uint8_t e4m3);
uint8_t f32_to_e5m2(uint32_t f32_bits);
uint32_t e5m2_to_f32(uint8_t e5m2);
uint8_t f32_to_e2m1(uint32_t f32_bits);
uint32_t e2m1_to_f32(uint8_t e2m1);

#endif /* LPFP_H */