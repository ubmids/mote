/* tools/intent_build.c — compile a command registry into vectors, once.
 *
 *   intent_build [--raw] <model.mq> <tokenizer> <registry.txt> <out-basename>
 *
 * Reads the hand-written registry (see registry/apartment.txt for the format),
 * embeds every phrasing with the model that will later do the matching, and
 * writes the result twice:
 *
 *   <out-basename>.mint   the binary blob, for desktop and iOS, which mmap it
 *   <out-basename>.h      the same bytes as a const C array, for firmware,
 *                         where there is no filesystem to mmap
 *
 * The header is the blob rather than a set of float literals on purpose. One
 * layout, one parser, no second code path to drift out of step with the first —
 * and the array is _Alignas(4) so the reader can use its floats in place, which
 * is what keeps the registry in flash instead of copied into RAM.
 *
 * This is the only place a phrasing is ever embedded. Doing it at startup
 * instead would mean eighty forward passes before the device could answer
 * anything, on hardware chosen because it can barely afford one.
 *
 * --raw skips mean-centering, which exists so tools/intent_test.c can measure
 * what centering is worth rather than assert it.
 */
#include "mote_lib.h"
#include "intent.h"

#include <ctype.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define MAX_COMMANDS  128
#define MAX_PHRASINGS 2048
#define MAX_LINE      512

static double now(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

/* the parsed registry source: ids in file order, phrasings grouped behind them */
typedef struct {
    char  ids[MAX_COMMANDS][INTENT_ID_MAX];
    int   first[MAX_COMMANDS], count[MAX_COMMANDS];
    int   n_commands;
    char *phrasings[MAX_PHRASINGS];
    int   n_phrasings;
} Source;

static char *trim(char *s)
{
    while (*s == ' ' || *s == '\t') s++;
    char *e = s + strlen(s);
    while (e > s && (e[-1] == '\n' || e[-1] == '\r' || e[-1] == ' ' || e[-1] == '\t'))
        *--e = '\0';
    return s;
}

/* Parse the text registry. Deliberately forgiving about layout (indentation,
 * blank lines, comments) and unforgiving about the two things that would make a
 * silent mistake: a phrasing before any [id], and a duplicate id. Both would
 * compile into something plausible and wrong. */
static int parse(const char *path, Source *src)
{
    FILE *f = fopen(path, "r");
    if (!f) { perror(path); return 0; }

    char line[MAX_LINE];
    int lineno = 0;
    src->n_commands = 0;
    src->n_phrasings = 0;

    while (fgets(line, sizeof(line), f)) {
        lineno++;
        char *s = trim(line);
        if (*s == '\0' || *s == '#')
            continue;

        if (*s == '[') {
            char *close = strchr(s, ']');
            if (!close) {
                fprintf(stderr, "%s:%d: unterminated [id]\n", path, lineno);
                fclose(f); return 0;
            }
            *close = '\0';
            char *id = trim(s + 1);
            if (*id == '\0' || strlen(id) >= INTENT_ID_MAX) {
                fprintf(stderr, "%s:%d: id must be 1..%d characters\n",
                        path, lineno, INTENT_ID_MAX - 1);
                fclose(f); return 0;
            }
            for (int i = 0; i < src->n_commands; i++)
                if (strcmp(src->ids[i], id) == 0) {
                    fprintf(stderr, "%s:%d: duplicate id '%s'\n", path, lineno, id);
                    fclose(f); return 0;
                }
            if (src->n_commands >= MAX_COMMANDS) {
                fprintf(stderr, "%s:%d: more than %d commands\n", path, lineno, MAX_COMMANDS);
                fclose(f); return 0;
            }
            snprintf(src->ids[src->n_commands], INTENT_ID_MAX, "%s", id);
            src->first[src->n_commands] = src->n_phrasings;
            src->count[src->n_commands] = 0;
            src->n_commands++;
            continue;
        }

        if (src->n_commands == 0) {
            fprintf(stderr, "%s:%d: phrasing before any [id]\n", path, lineno);
            fclose(f); return 0;
        }
        if (src->n_phrasings >= MAX_PHRASINGS) {
            fprintf(stderr, "%s:%d: more than %d phrasings\n", path, lineno, MAX_PHRASINGS);
            fclose(f); return 0;
        }
        src->phrasings[src->n_phrasings] = strdup(s);
        src->n_phrasings++;
        src->count[src->n_commands - 1]++;
    }
    fclose(f);

    for (int i = 0; i < src->n_commands; i++)
        if (src->count[i] == 0) {
            fprintf(stderr, "%s: command '%s' has no phrasings\n", path, src->ids[i]);
            return 0;
        }
    return src->n_commands > 0;
}

static void l2_normalize(float *v, int n)
{
    float ss = 0.0f;
    for (int i = 0; i < n; i++) ss += v[i] * v[i];
    float inv = ss > 0.0f ? 1.0f / sqrtf(ss) : 0.0f;
    for (int i = 0; i < n; i++) v[i] *= inv;
}

/* Emit the blob as a C array. The id list goes in as a comment and as index
 * macros, so firmware can switch on a name instead of on a number nobody can
 * check by eye. */
static int write_header(const char *path, const char *symbol, const char *from,
                        const Source *src, const IntentRegistry *r,
                        const unsigned char *blob, size_t len)
{
    FILE *f = fopen(path, "w");
    if (!f) { perror(path); return 0; }

    char guard[128];
    snprintf(guard, sizeof(guard), "MOTE_INTENT_%s_H", symbol);
    for (char *p = guard; *p; p++) *p = (char)toupper((unsigned char)*p);

    fprintf(f, "/* generated by tools/intent_build from %s. do not edit.\n", from);
    fprintf(f, " *\n * %d commands, %d phrasings, dim %d, %s. %zu bytes.\n",
            r->n_commands, r->n_phrasings, r->dim,
            r->centered ? "mean-centered" : "raw (not centered)", len);
    fprintf(f, " *\n * The blob, byte for byte, so the reader in src/intent.c is the only parser\n"
               " * of this format that exists. const, therefore in flash; _Alignas(4) so its\n"
               " * float arrays can be read in place rather than copied out.\n */\n");
    fprintf(f, "#ifndef %s\n#define %s\n\n#include <stddef.h>\n\n", guard, guard);

    /* index macros, in the registry's own order */
    for (int i = 0; i < src->n_commands; i++) {
        char up[INTENT_ID_MAX + 32];
        snprintf(up, sizeof(up), "INTENT_%s_%s", symbol, src->ids[i]);
        for (char *p = up; *p; p++) *p = (char)toupper((unsigned char)*p);
        fprintf(f, "#define %-40s %d\n", up, i);
    }
    fprintf(f, "\n_Alignas(4) static const unsigned char intent_%s[] = {\n", symbol);
    for (size_t i = 0; i < len; i++) {
        fprintf(f, "0x%02x,", blob[i]);
        if ((i + 1) % 16 == 0) fputc('\n', f);
    }
    fprintf(f, "\n};\nstatic const size_t intent_%s_len = sizeof(intent_%s);\n\n",
            symbol, symbol);
    fprintf(f, "#endif /* %s */\n", guard);
    fclose(f);
    return 1;
}

int main(int argc, char **argv)
{
    int centered = 1, a = 1;
    if (a < argc && strcmp(argv[a], "--raw") == 0) { centered = 0; a++; }
    if (argc - a != 4) {
        fprintf(stderr, "usage: %s [--raw] <model> <tokenizer> <registry.txt> <out-basename>\n",
                argv[0]);
        return 1;
    }
    const char *model_path = argv[a];
    const char *tok_path   = argv[a + 1];
    const char *reg_path   = argv[a + 2];
    const char *out_base   = argv[a + 3];

    Source src;
    if (!parse(reg_path, &src))
        return 1;

    mote *m = mote_create_from_files(model_path, tok_path);
    if (!m) { fprintf(stderr, "could not load %s / %s\n", model_path, tok_path); return 1; }
    mote_info_t info;
    mote_get_info(m, &info);
    int dim = info.dim;
    printf("model %s  dim %d  vocab %d  %s%s\n", model_path, dim, info.vocab_size,
           info.quantized ? "int" : "fp32", info.quantized ? (info.qbits == 4 ? "4" : "8") : "");
    printf("registry %s  %d commands  %d phrasings\n",
           reg_path, src.n_commands, src.n_phrasings);

    float *P = calloc((size_t)src.n_phrasings * dim, sizeof(float));
    float *C = calloc((size_t)src.n_commands * dim, sizeof(float));
    float *mu = calloc((size_t)dim, sizeof(float));
    if (!P || !C || !mu) { fprintf(stderr, "out of memory\n"); return 1; }

    double t0 = now();
    for (int i = 0; i < src.n_phrasings; i++) {
        if (mote_embed(m, src.phrasings[i], P + (size_t)i * dim) != dim) {
            fprintf(stderr, "embed failed: %s\n", src.phrasings[i]);
            return 1;
        }
    }
    double embed_secs = now() - t0;

    /* The mean of every phrasing in the registry. Mean-pooled hidden states from
     * a decoder-only model share a large component that says nothing about
     * meaning — unrelated sentences still score around 0.75 against each other —
     * and that component is what a threshold has to see past. Subtracting the
     * registry's own mean is the cheapest way to remove it, and it is stored so
     * the device can do the same to the one vector it computes at run time. */
    for (int i = 0; i < src.n_phrasings; i++)
        for (int d = 0; d < dim; d++)
            mu[d] += P[(size_t)i * dim + d];
    for (int d = 0; d < dim; d++)
        mu[d] /= (float)src.n_phrasings;

    if (centered) {
        for (int i = 0; i < src.n_phrasings; i++) {
            float *v = P + (size_t)i * dim;
            for (int d = 0; d < dim; d++) v[d] -= mu[d];
            l2_normalize(v, dim);
        }
    } else {
        for (int d = 0; d < dim; d++) mu[d] = 0.0f;
    }

    /* One centroid per command, from whatever the phrasings now are. Stored even
     * though the matcher does not use it: it is what makes the choice of
     * max-over-phrasings a measurement in intent_test rather than an opinion,
     * and on a much larger registry it is the obvious prefilter. */
    for (int i = 0; i < src.n_commands; i++) {
        float *c = C + (size_t)i * dim;
        for (int k = 0; k < src.count[i]; k++) {
            const float *v = P + (size_t)(src.first[i] + k) * dim;
            for (int d = 0; d < dim; d++) c[d] += v[d];
        }
        l2_normalize(c, dim);
    }

    const char *ids[MAX_COMMANDS];
    for (int i = 0; i < src.n_commands; i++) ids[i] = src.ids[i];

    IntentBuild b = {
        .dim = dim, .n_commands = src.n_commands, .n_phrasings = src.n_phrasings,
        .centered = centered, .mean = centered ? mu : NULL,
        .ids = ids, .first = src.first, .count = src.count,
        .centroids = C, .phrasings = P,
    };
    size_t need = intent_blob_size(&b);
    unsigned char *blob = malloc(need);
    if (!blob || intent_pack(&b, blob, need) != need) {
        fprintf(stderr, "pack failed\n"); return 1;
    }

    /* read it straight back with the runtime's own reader: a blob that does not
     * open is not worth shipping, and this is the cheapest place to find out */
    IntentRegistry check;
    if (!intent_open(blob, need, &check) || check.dim != dim ||
        check.n_commands != src.n_commands || check.n_phrasings != src.n_phrasings) {
        fprintf(stderr, "the blob just written does not open\n"); return 1;
    }

    char path[1024];
    snprintf(path, sizeof(path), "%s.mint", out_base);
    FILE *f = fopen(path, "wb");
    if (!f) { perror(path); return 1; }
    if (fwrite(blob, 1, need, f) != need) { perror(path); return 1; }
    fclose(f);
    printf("wrote %s  (%zu bytes)\n", path, need);

    /* symbol name from the basename, so two registries do not collide */
    const char *slash = strrchr(out_base, '/');
    char symbol[64];
    snprintf(symbol, sizeof(symbol), "%s", slash ? slash + 1 : out_base);
    for (char *p = symbol; *p; p++)
        if (!((*p >= 'a' && *p <= 'z') || (*p >= 'A' && *p <= 'Z') ||
              (*p >= '0' && *p <= '9'))) *p = '_';

    snprintf(path, sizeof(path), "%s.h", out_base);
    if (!write_header(path, symbol, reg_path, &src, &check, blob, need))
        return 1;
    printf("wrote %s  (symbol intent_%s)\n", path, symbol);
    printf("%d phrasings embedded in %.2fs (%.0f ms each); run time embeds 1\n",
           src.n_phrasings, embed_secs, embed_secs * 1000.0 / src.n_phrasings);

    for (int i = 0; i < src.n_phrasings; i++) free(src.phrasings[i]);
    free(P); free(C); free(mu); free(blob);
    mote_free(m);
    return 0;
}
