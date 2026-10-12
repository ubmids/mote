/* tools/intent_test.c — does embedding-based intent matching actually work?
 *
 *   intent_test <model> <tokenizer> <registry.mint> [<registry_raw.mint>]
 *
 * A matcher like this passes the easy test by construction: embed the registry's
 * own phrasings, match them back, print 100%. That measures nothing. What
 * decides whether this can be trusted to switch a light is the four cases below,
 * and only the second of them is the one people usually check.
 *
 *   1. registry phrasings, matched back. A floor, not a result: if this fails
 *      the format or the arithmetic is broken, and nothing after it means
 *      anything.
 *   2. held-out paraphrases nobody wrote down. This is generalisation, and it
 *      is the case a keyword matcher cannot do at all.
 *   3. negation and antonym pairs. on/off, warmer/cooler, up/down. Two
 *      utterances differing by one short word that inverts the meaning, in a
 *      representation built by averaging token states — which is to say a
 *      representation with no obvious place to put the inversion. This is the
 *      case that decides whether the whole approach is usable, because getting
 *      it wrong does not mean a worse answer, it means the opposite action.
 *   4. out-of-domain speech. Small talk, a general-knowledge question, and
 *      keyboard mash. Cosine similarity always returns a nearest neighbour;
 *      "nearest" and "right" are different claims, and a device that acts on
 *      the former will act on nothing said to it.
 *
 * Every number printed is measured. Nothing here asserts a threshold: the
 * thresholds are swept at the end and the best one the data admits is reported,
 * including the case where there isn't one.
 *
 * ---- what it measured, so nobody has to run it to find out ----
 *
 * On models/default.mq (Qwen2.5-0.5B, int4) with registry/apartment.txt:
 *
 *   - unconstrained argmax gets 32 of 44 in-domain utterances right, 73%.
 *     Reported alone that number reads like a working system. It is not one:
 *     nothing was refused, so the other 12 are wrong actions.
 *   - out-of-domain top scores run 0.411 to 0.752. In-domain correct top scores
 *     run 0.506 to 1.000. Those ranges overlap over most of their length, so no
 *     absolute floor separates "a command" from "a sentence about taxes".
 *   - the lowest floors that produce no wrong action at all are score >= 0.88
 *     and margin >= 0.04, and they accept 19 of 44 — essentially only the
 *     utterances already written in the registry, which score 1.000 against
 *     themselves. Every paraphrase a lower floor would catch is also the reason
 *     a lower floor lets something wrong through.
 *   - 3 of 18 negation/antonym pairs rank the OPPOSITE command above the right
 *     one. The median gap between a command and its own opposite is +0.085,
 *     while correct matches routinely win by as little as 0.013. There is no
 *     margin floor that keeps polarity and admits paraphrase.
 *   - it is not a capacity problem. The same registry and cases on the 1.5B
 *     (models/aimm15b_q4_qat.mq, dim 1536) give 34 of 44, one inverted pair
 *     instead of three, and the same zero-wrong operating point of score >= 0.90
 *     accepting the same 19. Three times the model buys two utterances.
 *
 * The verdict: mean-pooled hidden states from a small decoder do not encode
 * "on" versus "off" strongly enough to switch a light on. What this path is
 * good for is refusing confidently and cheaply — 25 of 44 refusals at the safe
 * floor, each one costing one forward pass instead of a network round trip —
 * and for exact recall of phrasings someone actually wrote down. Treated as an
 * 80%-of-traffic classifier it would take the wrong physical action several
 * times a day.
 */
#include "mote_lib.h"
#include "intent.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

/* The registry baked in as a C array, if it has been built. Included so the
 * firmware path is exercised by the same test rather than assumed to agree with
 * the file path: same bytes, same reader, and pass 0 checks they open the same. */
#if defined(__has_include)
#  if __has_include("../registry/apartment.h")
#    include "../registry/apartment.h"
#    define HAVE_BAKED 1
#  endif
#endif

static double now(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

/* ---- the cases ----
 *
 * `want` is the command id that should win, or NULL for "nothing should". Case
 * text is written the way it would arrive from a speech recogniser: lowercase,
 * no punctuation to speak of, sometimes barely a sentence.
 */
typedef struct {
    const char *text;
    const char *want;      /* NULL = must fall through to no-match */
    int         group;
} Case;

enum { G_EXACT, G_HELDOUT, G_POLARITY, G_OOD, G_NEARMISS, N_GROUPS };
static const char *const GROUP_NAME[N_GROUPS] = {
    "registry phrasings", "held-out paraphrases", "negation / antonym",
    "out of domain", "near-miss between commands"
};

static const Case CASES[] = {
    /* 1. straight out of the registry — the floor */
    { "turn on the lights",        "lights_on",        G_EXACT },
    { "kill the lights",           "lights_off",       G_EXACT },
    { "make it warmer",            "temperature_up",   G_EXACT },
    { "pause the music",           "music_pause",      G_EXACT },
    { "lock the door",             "door_lock",        G_EXACT },
    { "what time is it",           "time_query",       G_EXACT },

    /* 2. held out: none of these appear in registry/apartment.txt */
    { "could you switch the lights on please",  "lights_on",       G_HELDOUT },
    { "hit the lights",                         "lights_off",      G_HELDOUT },
    { "bring the lights down a bit",            "lights_dim",      G_HELDOUT },
    { "I'm cold",                               "temperature_up",  G_HELDOUT },
    { "I'm sweating in here",                   "temperature_down", G_HELDOUT },
    { "put on some jazz",                       "music_play",      G_HELDOUT },
    { "shut the music off",                     "music_pause",     G_HELDOUT },
    { "bump the volume",                        "volume_up",       G_HELDOUT },
    { "bring the sound down",                   "volume_down",     G_HELDOUT },
    { "do you know what time it is",             "time_query",      G_HELDOUT },
    { "should I take an umbrella",              "weather_query",   G_HELDOUT },
    { "make sure the door is locked",           "door_lock",       G_HELDOUT },

    /* 3. the pairs. Each is one short word away from its opposite. */
    { "turn the lights on",        "lights_on",        G_POLARITY },
    { "turn the lights off",       "lights_off",       G_POLARITY },
    { "lights on",                 "lights_on",        G_POLARITY },
    { "lights off",                "lights_off",       G_POLARITY },
    { "switch the lights on",      "lights_on",        G_POLARITY },
    { "switch the lights off",     "lights_off",       G_POLARITY },
    { "warmer",                    "temperature_up",   G_POLARITY },
    { "cooler",                    "temperature_down", G_POLARITY },
    { "make the room warmer",      "temperature_up",   G_POLARITY },
    { "make the room cooler",      "temperature_down", G_POLARITY },
    { "volume up",                 "volume_up",        G_POLARITY },
    { "volume down",               "volume_down",      G_POLARITY },
    { "turn the volume up",        "volume_up",        G_POLARITY },
    { "turn the volume down",      "volume_down",      G_POLARITY },
    { "turn it up",                "volume_up",        G_POLARITY },
    { "turn it down",              "volume_down",      G_POLARITY },
    { "play the music",            "music_play",       G_POLARITY },
    { "stop the music",            "music_pause",      G_POLARITY },

    /* 4. nothing in the registry covers these */
    { "what's the capital of France",           NULL, G_OOD },
    { "I love you",                             NULL, G_OOD },
    { "asdfghjkl",                              NULL, G_OOD },
    { "write me a poem about the sea",          NULL, G_OOD },
    { "how do I file my taxes",                 NULL, G_OOD },
    { "call my mother",                         NULL, G_OOD },
    { "order me a pizza",                       NULL, G_OOD },
    { "what's the square root of 144",          NULL, G_OOD },
    { "book a flight to Berlin",                NULL, G_OOD },
    { "qwertyuiop zxcvbnm",                     NULL, G_OOD },
    { "tell me a joke",                         NULL, G_OOD },
    { "who won the match last night",           NULL, G_OOD },

    /* 5. near misses: shared vocabulary, different command. "turn the lights
     * down" is dimming, not switching off; "how hot is it outside" is weather,
     * not the thermostat; "is the door locked" is the door, not the lights. */
    { "turn the lights down",                   "lights_dim",      G_NEARMISS },
    { "turn the music down",                    "volume_down",     G_NEARMISS },
    { "turn the heat down",                     "temperature_down", G_NEARMISS },
    { "how hot is it outside",                  "weather_query",   G_NEARMISS },
    { "is it cold outside",                     "weather_query",   G_NEARMISS },
    { "is the front door locked",               "door_lock",       G_NEARMISS },
    { "turn the lights off in here",            "lights_off",      G_NEARMISS },
    { "put the music on",                       "music_play",      G_NEARMISS },
};
static const int N_CASES = (int)(sizeof(CASES) / sizeof(CASES[0]));

/* The pairs again, as pairs, so the gap between a command and its own opposite
 * can be measured directly rather than inferred from which one happened to win.
 * That gap is the quantity the whole approach lives or dies on. */
typedef struct { const char *text, *want, *opposite; } Pair;
static const Pair PAIRS[] = {
    { "turn the lights on",   "lights_on",        "lights_off"       },
    { "turn the lights off",  "lights_off",       "lights_on"        },
    { "lights on",            "lights_on",        "lights_off"       },
    { "lights off",           "lights_off",       "lights_on"        },
    { "switch the lights on", "lights_on",        "lights_off"       },
    { "switch the lights off","lights_off",       "lights_on"        },
    { "warmer",               "temperature_up",   "temperature_down" },
    { "cooler",               "temperature_down", "temperature_up"   },
    { "make the room warmer", "temperature_up",   "temperature_down" },
    { "make the room cooler", "temperature_down", "temperature_up"   },
    { "volume up",            "volume_up",        "volume_down"      },
    { "volume down",          "volume_down",      "volume_up"        },
    { "turn the volume up",   "volume_up",        "volume_down"      },
    { "turn the volume down", "volume_down",      "volume_up"        },
    { "turn it up",           "volume_up",        "volume_down"      },
    { "turn it down",         "volume_down",      "volume_up"        },
    { "play the music",       "music_play",       "music_pause"      },
    { "stop the music",       "music_pause",      "music_play"       },
};
static const int N_PAIRS = (int)(sizeof(PAIRS) / sizeof(PAIRS[0]));

/* ---- helpers ---- */

static void *read_file(const char *path, size_t *len_out)
{
    FILE *f = fopen(path, "rb");
    if (!f) { perror(path); return NULL; }
    fseek(f, 0, SEEK_END);
    long n = ftell(f);
    fseek(f, 0, SEEK_SET);
    if (n <= 0) { fclose(f); return NULL; }
    /* malloc is suitably aligned for any type, which the reader requires */
    void *buf = malloc((size_t)n);
    if (!buf || fread(buf, 1, (size_t)n, f) != (size_t)n) {
        fclose(f); free(buf); return NULL;
    }
    fclose(f);
    *len_out = (size_t)n;
    return buf;
}

/* command id -> index, by asking the registry. Linear because a registry has a
 * few dozen commands and this is a test, not the run-time path. */
static int id_index(const void *reg, const char *id)
{
    for (int i = 0; ; i++) {
        const char *s = mote_intent_id(reg, i);
        if (!s) return -1;
        if (strcmp(s, id) == 0) return i;
    }
}

static double median(double *v, int n)
{
    for (int i = 0; i < n; i++)
        for (int j = i + 1; j < n; j++)
            if (v[j] < v[i]) { double t = v[i]; v[i] = v[j]; v[j] = t; }
    if (n == 0) return 0.0;
    return n % 2 ? v[n / 2] : 0.5 * (v[n / 2 - 1] + v[n / 2]);
}

/* One measured case: the full ranking, so the runner-up and the margin are
 * available afterwards without re-running the model. */
typedef struct {
    int   top, second;
    float top_score, second_score;
    float want_score;        /* score of the command that should have won */
    int   correct;           /* top == want, or want == NULL (unused then) */
} Result;

/* ---- pass 1: score every case against a registry ---- */

static void measure(mote *m, const void *reg, const Case *cases, int n,
                    Result *out, double *secs_out)
{
    double t0 = now();
    for (int i = 0; i < n; i++) {
        mote_intent_hit hits[8];
        int k = mote_intent_rank(m, reg, cases[i].text, hits, 8);
        if (k < 2) {
            out[i].top = out[i].second = -1;
            out[i].top_score = out[i].second_score = out[i].want_score = 0.0f;
            out[i].correct = 0;
            continue;
        }
        out[i].top = hits[0].command;
        out[i].second = hits[1].command;
        out[i].top_score = hits[0].score;
        out[i].second_score = hits[1].score;
        out[i].want_score = 0.0f;
        int want = cases[i].want ? id_index(reg, cases[i].want) : -1;
        for (int j = 0; j < k; j++)
            if (hits[j].command == want) out[i].want_score = hits[j].score;
        out[i].correct = cases[i].want ? (out[i].top == want) : 1;
    }
    *secs_out = now() - t0;
}

static void print_table(const void *reg, const Case *cases, int n, const Result *r)
{
    int group = -1;
    for (int i = 0; i < n; i++) {
        if (cases[i].group != group) {
            group = cases[i].group;
            printf("\n  %s\n", GROUP_NAME[group]);
            printf("    %-34s %-18s %6s  %-18s %6s  %s\n",
                   "utterance", "top match", "score", "runner-up", "score", "margin");
        }
        const char *top = mote_intent_id(reg, r[i].top);
        const char *snd = mote_intent_id(reg, r[i].second);
        float margin = r[i].top_score - r[i].second_score;
        const char *verdict;
        if (!cases[i].want)
            verdict = "";                       /* judged by threshold, below */
        else
            verdict = r[i].correct ? "" : "  <-- WRONG COMMAND";
        printf("    %-34.34s %-18.18s %6.3f  %-18.18s %6.3f  %6.3f%s\n",
               cases[i].text, top ? top : "?", r[i].top_score,
               snd ? snd : "?", r[i].second_score, margin, verdict);
    }
}

/* ---- pass 2: the polarity gap ----
 *
 * For each pair, the score of the command that should have won minus the score
 * of its own opposite. Positive means the representation encodes the inversion
 * at all. How positive says whether a threshold could ever tell them apart. */
static void print_polarity(mote *m, const void *reg)
{
    printf("\n    %-26s %-18s %6s  %-18s %6s  %s\n",
           "utterance", "should be", "score", "its opposite", "score", "gap");
    double gaps[64];
    int n = 0, inverted = 0;
    for (int i = 0; i < N_PAIRS && n < 64; i++) {
        mote_intent_hit hits[16];
        int k = mote_intent_rank(m, reg, PAIRS[i].text, hits, 16);
        if (k < 2) continue;
        int wi = id_index(reg, PAIRS[i].want);
        int oi = id_index(reg, PAIRS[i].opposite);
        float ws = 0.0f, os = 0.0f;
        for (int j = 0; j < k; j++) {
            if (hits[j].command == wi) ws = hits[j].score;
            if (hits[j].command == oi) os = hits[j].score;
        }
        double gap = ws - os;
        gaps[n++] = gap;
        if (gap <= 0.0) inverted++;
        printf("    %-26.26s %-18.18s %6.3f  %-18.18s %6.3f  %+6.3f%s\n",
               PAIRS[i].text, PAIRS[i].want, ws, PAIRS[i].opposite, os, gap,
               gap <= 0.0 ? "  <-- INVERTED" : "");
    }
    double lo = 9.0, hi = -9.0, sum = 0.0;
    for (int i = 0; i < n; i++) {
        if (gaps[i] < lo) lo = gaps[i];
        if (gaps[i] > hi) hi = gaps[i];
        sum += gaps[i];
    }
    printf("\n    %d of %d pairs put the opposite command ahead of the right one\n",
           inverted, n);
    printf("    gap: min %+.3f  median %+.3f  mean %+.3f  max %+.3f\n",
           lo, median(gaps, n), sum / (n ? n : 1), hi);
}

/* ---- pass 3: what a threshold can and cannot do ----
 *
 * A grid of accuracies is not the question. The question is whether any pair of
 * floors admits useful utterances while admitting no harmful one, so what gets
 * printed is the frontier: for each margin floor, the lowest score floor at
 * which nothing wrong gets through, and how much still gets through at all.
 *
 * "Wrong" is deliberately harsh. An out-of-domain utterance accepted and an
 * in-domain utterance matched to the wrong command are counted together, because
 * both end with the device doing something nobody asked for, and that is the
 * only failure mode a light switch cannot absorb. Refusing is not in that
 * column: a refusal escalates to the big model and costs a round trip.
 */
typedef struct { int accepted, wrong, rejected, polarity; } Tally;

static Tally tally_at(const Case *cases, int n, const Result *r, float ts, float tm)
{
    Tally t = { 0, 0, 0, 0 };
    for (int i = 0; i < n; i++) {
        int accept = r[i].top_score >= ts &&
                     (r[i].top_score - r[i].second_score) >= tm;
        if (!cases[i].want) {
            if (accept) t.wrong++;
            continue;
        }
        if (!accept) { t.rejected++; continue; }
        if (r[i].correct) {
            t.accepted++;
            if (cases[i].group == G_POLARITY) t.polarity++;
        } else {
            t.wrong++;
        }
    }
    return t;
}

static void sweep(const Case *cases, int n, const Result *r,
                  float *rec_score, float *rec_margin)
{
    int in_domain = 0, polarity = 0;
    for (int i = 0; i < n; i++) {
        if (cases[i].want) in_domain++;
        if (cases[i].group == G_POLARITY) polarity++;
    }

    printf("\n    the frontier: lowest score floor with no wrong action, per margin floor\n");
    printf("    %6s %7s | %8s %8s | %s\n",
           "margin", "score", "accepted", "refused", "of which polarity");

    float best_s = 1.01f, best_m = 0.0f;
    int   best_ok = -1, feasible = 0;
    for (int mi = 0; mi <= 20; mi += 2) {
        float tm = 0.01f * mi;
        int found = 0;
        for (int si = 0; si <= 100; si++) {
            float ts = 0.01f * si;
            Tally t = tally_at(cases, n, r, ts, tm);
            if (t.wrong != 0)
                continue;
            printf("    %6.2f %7.2f | %8d %8d | %d of %d\n",
                   tm, ts, t.accepted, t.rejected, t.polarity, polarity);
            found = 1;
            feasible = 1;
            /* prefer more correct actions; on a tie prefer the wider margin,
             * which is the floor that generalises past this particular set */
            if (t.accepted > best_ok || (t.accepted == best_ok && tm > best_m)) {
                best_ok = t.accepted; best_s = ts; best_m = tm;
            }
            break;
        }
        if (!found)
            printf("    %6.2f       - | no score floor makes this margin safe\n", tm);
    }

    if (!feasible) {
        printf("\n    NO THRESHOLD SEPARATES: every operating point that acts at all\n"
               "    also acts wrongly on this set. Nothing here should drive a switch.\n");
        *rec_score = 1.01f; *rec_margin = 0.0f;
        return;
    }

    Tally t = tally_at(cases, n, r, best_s, best_m);
    printf("\n    recommended: score >= %.2f, margin >= %.2f\n", best_s, best_m);
    printf("      %d of %d in-domain utterances acted on correctly (%.0f%%)\n",
           t.accepted, in_domain, 100.0 * t.accepted / (in_domain ? in_domain : 1));
    printf("      %d of %d negation/antonym cases survive it\n", t.polarity, polarity);
    printf("      %d refused, all of which have to go to the big model\n", t.rejected);
    /* A floor this high is not a confidence threshold, it is an exact-match
     * gate: a registry phrasing scores 1.000 against itself, so anything close
     * to 1 is a lookup that happened to be spelled in vectors. Whether that is
     * a result or a restatement of the input is the whole finding. */
    if (best_s >= 0.85f)
        printf("      note: at this floor the only utterances admitted are the ones\n"
               "      already close to verbatim in the registry. That is a lookup\n"
               "      table, not generalisation.\n");
    *rec_score = best_s;
    *rec_margin = best_m;
}

/* score distributions, which is what actually decides whether a threshold is
 * findable: if the in-domain-correct scores and the out-of-domain scores
 * overlap, no single number can sit between them */
static void distributions(const Case *cases, int n, const Result *r)
{
    double corr[128], wrong[128], ood[128];
    double cm[128], wm[128], om[128];
    int nc = 0, nw = 0, no = 0;
    for (int i = 0; i < n; i++) {
        double s = r[i].top_score, mg = r[i].top_score - r[i].second_score;
        if (!cases[i].want)      { ood[no] = s;  om[no] = mg;  no++; }
        else if (r[i].correct)   { corr[nc] = s; cm[nc] = mg;  nc++; }
        else                     { wrong[nw] = s; wm[nw] = mg; nw++; }
    }
    struct { const char *label; double *s, *m; int n; } sets[3] = {
        { "in-domain, right command", corr, cm, nc },
        { "in-domain, wrong command", wrong, wm, nw },
        { "out of domain",            ood,  om, no },
    };
    printf("\n    %-26s %4s | %-22s | %s\n", "", "n", "top score", "margin to runner-up");
    printf("    %-26s %4s | %6s %6s %6s | %6s %6s %6s\n",
           "", "", "min", "med", "max", "min", "med", "max");
    for (int k = 0; k < 3; k++) {
        if (sets[k].n == 0) { printf("    %-26s %4d |\n", sets[k].label, 0); continue; }
        double smin = 9, smax = -9, mmin = 9, mmax = -9;
        for (int i = 0; i < sets[k].n; i++) {
            if (sets[k].s[i] < smin) smin = sets[k].s[i];
            if (sets[k].s[i] > smax) smax = sets[k].s[i];
            if (sets[k].m[i] < mmin) mmin = sets[k].m[i];
            if (sets[k].m[i] > mmax) mmax = sets[k].m[i];
        }
        double smed = median(sets[k].s, sets[k].n);
        double mmed = median(sets[k].m, sets[k].n);
        printf("    %-26s %4d | %6.3f %6.3f %6.3f | %6.3f %6.3f %6.3f\n",
               sets[k].label, sets[k].n, smin, smed, smax, mmin, mmed, mmax);
    }
}

/* max-over-phrasings versus one centroid per command, on the same vectors, so
 * the matcher's choice is a measurement */
static void compare_pooling(mote *m, const void *reg, const void *blob, size_t len)
{
    IntentRegistry r;
    if (!intent_open(blob, len, &r)) { printf("    registry did not open\n"); return; }
    mote_info_t info;
    mote_get_info(m, &info);
    float *q = malloc((size_t)r.dim * sizeof(float));
    float *sp = malloc((size_t)r.n_commands * sizeof(float));
    float *sc = malloc((size_t)r.n_commands * sizeof(float));
    if (!q || !sp || !sc) { free(q); free(sp); free(sc); return; }

    int ph_ok = 0, ce_ok = 0, in_domain = 0;
    for (int i = 0; i < N_CASES; i++) {
        if (!CASES[i].want) continue;
        in_domain++;
        mote_embed(m, CASES[i].text, q);
        intent_prepare_query(&r, q);
        intent_score(&r, q, sp, NULL);
        intent_score_centroids(&r, q, sc);
        int want = id_index(reg, CASES[i].want);
        int bp = 0, bc = 0;
        for (int c = 1; c < r.n_commands; c++) {
            if (sp[c] > sp[bp]) bp = c;
            if (sc[c] > sc[bc]) bc = c;
        }
        if (bp == want) ph_ok++;
        if (bc == want) ce_ok++;
    }
    printf("    max over phrasings: %d of %d correct\n", ph_ok, in_domain);
    printf("    one centroid each:  %d of %d correct\n", ce_ok, in_domain);
    free(q); free(sp); free(sc);
}

int main(int argc, char **argv)
{
    if (argc < 4) {
        fprintf(stderr, "usage: %s <model> <tokenizer> <registry.mint> [<registry_raw.mint>]\n"
                        "  build the registries first:  make registry\n", argv[0]);
        return 1;
    }
    size_t len = 0;
    void *blob = read_file(argv[3], &len);
    if (!blob) return 1;

    int ncmd = mote_intent_check(blob, len);
    if (ncmd < 0) { fprintf(stderr, "%s is not a registry\n", argv[3]); return 1; }
    printf("\nregistry %s: %d commands, dim %d, %zu bytes\n",
           argv[3], ncmd, mote_intent_dim(blob), len);

    printf("\n0. the same registry baked into firmware\n");
#ifdef HAVE_BAKED
    {
        int n2 = mote_intent_check(intent_apartment, intent_apartment_len);
        int same = (n2 == ncmd && intent_apartment_len == len &&
                    memcmp(intent_apartment, blob, len) == 0);
        printf("   registry/apartment.h: %d commands, %zu bytes — %s\n",
               n2, intent_apartment_len,
               same ? "byte-identical to the blob" : "DIFFERS from the blob");
        if (!same && intent_apartment_len != len)
            printf("   (expected when the .h and the .mint were built from different runs)\n");
    }
#else
    printf("   skipped: registry/apartment.h not built (make registry)\n");
#endif

    mote *m = mote_create_from_files(argv[1], argv[2]);
    if (!m) { fprintf(stderr, "could not load the model\n"); return 1; }
    mote_info_t info;
    mote_get_info(m, &info);
    if (info.dim != mote_intent_dim(blob)) {
        fprintf(stderr, "model dim %d, registry dim %d — different models\n",
                info.dim, mote_intent_dim(blob));
        return 1;
    }
    printf("model %s: dim %d, vocab %d, int%d\n", argv[1], info.dim, info.vocab_size,
           info.quantized ? info.qbits : 32);

    Result *res = calloc((size_t)N_CASES, sizeof(Result));
    double secs = 0.0;
    printf("\n1. every case, with its runner-up and margin\n");
    measure(m, blob, CASES, N_CASES, res, &secs);
    print_table(blob, CASES, N_CASES, res);
    printf("\n    %d utterances matched in %.2fs (%.0f ms each, one forward pass plus %d dots)\n",
           N_CASES, secs, secs * 1000.0 / N_CASES, ncmd);

    printf("\n2. negation and antonym pairs: the gap to the opposite command\n");
    print_polarity(m, blob);

    printf("\n3. score distributions — can any threshold sit between them?\n");
    distributions(CASES, N_CASES, res);

    printf("\n4. sweeping both floors over the measured scores\n");
    float rec_s = 0.0f, rec_m = 0.0f;
    sweep(CASES, N_CASES, res, &rec_s, &rec_m);

    printf("\n5. max over phrasings versus one centroid per command\n");
    compare_pooling(m, blob, blob, len);

    /* the same cases against a registry compiled without mean-centering, which
     * is the one preprocessing choice in the whole pipeline */
    if (argc >= 5) {
        size_t rlen = 0;
        void *raw = read_file(argv[4], &rlen);
        if (raw && mote_intent_check(raw, rlen) == ncmd) {
            printf("\n6. the same cases with mean-centering off\n");
            Result *rr = calloc((size_t)N_CASES, sizeof(Result));
            double rsecs;
            measure(m, raw, CASES, N_CASES, rr, &rsecs);
            int ok = 0, in_domain = 0;
            for (int i = 0; i < N_CASES; i++) {
                if (!CASES[i].want) continue;
                in_domain++;
                if (rr[i].correct) ok++;
            }
            int cok = 0;
            for (int i = 0; i < N_CASES; i++)
                if (CASES[i].want && res[i].correct) cok++;
            printf("    centered: %d of %d in-domain correct\n", cok, in_domain);
            printf("    raw:      %d of %d in-domain correct\n", ok, in_domain);
            distributions(CASES, N_CASES, rr);
            printf("\n    the raw frontier:\n");
            float rs = 0.0f, rm = 0.0f;
            sweep(CASES, N_CASES, rr, &rs, &rm);
            free(rr);
        }
        free(raw);
    }

    /* what the shipped thresholds actually do, since those are the numbers a
     * caller of mote_intent_match gets rather than anything swept above */
    printf("\n7. mote_intent_match as shipped (score >= %.2f, margin >= %.2f)\n",
           MOTE_INTENT_MIN_SCORE, MOTE_INTENT_MIN_MARGIN);
    int acted = 0, wrong = 0, refused = 0, ood_acted = 0;
    for (int i = 0; i < N_CASES; i++) {
        float sc = 0.0f;
        int cmd = mote_intent_match(m, blob, CASES[i].text, &sc);
        const char *got = cmd >= 0 ? mote_intent_id(blob, cmd) : NULL;
        if (!CASES[i].want) {
            if (cmd >= 0) { ood_acted++; printf("    ACTED ON NONSENSE  %-30.30s -> %s (%.3f)\n",
                                                CASES[i].text, got, sc); }
            continue;
        }
        if (cmd < 0) { refused++; continue; }
        if (got && strcmp(got, CASES[i].want) == 0) acted++;
        else { wrong++; printf("    WRONG ACTION       %-30.30s -> %s, wanted %s (%.3f)\n",
                               CASES[i].text, got ? got : "?", CASES[i].want, sc); }
    }
    printf("    %d acted on correctly, %d wrong actions, %d refused, %d out-of-domain accepted\n",
           acted, wrong, refused, ood_acted);

    free(res);
    free(blob);
    mote_free(m);
    /* This tool reports; it does not pass or fail. The numbers above are the
     * result, and a threshold that makes them all green would be a threshold
     * chosen after seeing them. */
    return 0;
}
