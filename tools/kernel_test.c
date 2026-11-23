/* kernel_test.c — the SIMD integer kernels must equal the plain loops, exactly.
 *
 *   make kernel_test && ./kernel_test
 *
 * The int8 and int4 dot products have NEON and wasm bodies, and int4 has an
 * SSE2 one, picked at compile time. They are integer arithmetic, so each must
 * return exactly what the scalar definition returns, on every input, so the
 * choice of body can never change what the model says. This file
 * includes quant.c to reach the static kernels, checks them against the
 * definitions on the extremes (every weight and activation at +-127, every
 * nibble from -8 to 7) and on random groups of several sizes, including sizes
 * that leave a scalar tail. Speed is measured end to end with the CLI, not here.
 *
 * Build it the way the engine will be built for the target (for QNX, with
 * qcc) so the body under test is the one that ships.
 */
#include "../src/quant.c"

#include <stdio.h>
#include <string.h>

static int32_t ref_i8(const int8_t *a, const int8_t *b, int gs)
{
    int32_t s = 0;
    for (int k = 0; k < gs; k++)
        s += (int32_t)a[k] * (int32_t)b[k];
    return s;
}

static int32_t ref_q4(const int8_t *a, const uint8_t *wb, int gs)
{
    int half = gs / 2;
    int32_t s = 0;
    for (int k = 0; k < half; k++)
        s += (int32_t)a[k] * nib_lo(wb[k]) + (int32_t)a[k + half] * nib_hi(wb[k]);
    return s;
}

static uint32_t rng = 2463534242u;
static uint32_t next(void) { rng ^= rng << 13; rng ^= rng >> 17; rng ^= rng << 5; return rng; }

static const char *body(void)
{
#if defined(__ARM_NEON) && defined(__ARM_FEATURE_DOTPROD)
    return "neon+dotprod";
#elif defined(__wasm_simd128__)
    return "wasm simd128";
#elif defined(__SSE2__)
    return "sse2 (int4); int8 left to the compiler";
#else
    return "scalar";
#endif
}

int main(void)
{
    enum { MAXG = 256 };
    static const int sizes[] = { 16, 32, 48, 64, 128, 256, 20, 40, 72 };
    int8_t a[MAXG], b[MAXG];
    uint8_t w[MAXG / 2];
    long checks = 0, bad = 0;

    printf("kernel body: %s\n", body());

    for (size_t si = 0; si < sizeof sizes / sizeof sizes[0]; si++) {
        int gs = sizes[si];

        /* extremes: all four sign combinations at full scale, and every
         * nibble value against both activation extremes */
        for (int sa = -1; sa <= 1; sa += 2)
            for (int sb = -1; sb <= 1; sb += 2) {
                memset(a, sa * 127, (size_t)gs);
                memset(b, sb * 127, (size_t)gs);
                checks++; bad += dot_group_i8(a, b, gs) != ref_i8(a, b, gs);
            }
        for (int lo = -8; lo <= 7; lo++)
            for (int hi = -8; hi <= 7; hi++)
                for (int sa = -1; sa <= 1; sa += 2) {
                    memset(a, sa * 127, (size_t)gs);
                    memset(w, (uint8_t)(((hi & 15) << 4) | (lo & 15)), (size_t)gs / 2);
                    checks++; bad += dot_group_q4(a, w, gs) != ref_q4(a, w, gs);
                }

        for (int t = 0; t < 20000; t++) {
            for (int k = 0; k < gs; k++) {
                a[k] = (int8_t)((int)(next() % 255) - 127);
                b[k] = (int8_t)((int)(next() % 255) - 127);
            }
            for (int k = 0; k < gs / 2; k++)
                w[k] = (uint8_t)next();
            int32_t x = dot_group_i8(a, b, gs), rx = ref_i8(a, b, gs);
            int32_t y = dot_group_q4(a, w, gs), ry = ref_q4(a, w, gs);
            checks += 2;
            if (x != rx || y != ry) {
                if (bad < 5)
                    fprintf(stderr, "gs=%d: i8 %d vs %d, q4 %d vs %d\n", gs, x, rx, y, ry);
                bad += (x != rx) + (y != ry);
            }
        }
    }

    if (bad) {
        printf("FAIL: %ld of %ld checks differ from the scalar definition\n", bad, checks);
        return 1;
    }
    printf("ok: %ld checks, every SIMD result equals the scalar definition\n", checks);
    return 0;
}
