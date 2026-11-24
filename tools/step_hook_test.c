/* tools/step_hook_test.c: the step hook observes and never changes the text.
 *
 *   step_hook_test <model> <tokenizer>
 *
 * Works with either tokenizer (Qwen .mtok or the TinyStories tokenizer.bin).
 * Checks, in order:
 *
 *   1. with a hook installed, a greedy turn prints exactly what it prints
 *      without one, and the hook sees every position once, prompt first;
 *   2. the same under a grammar, and the choice statistics are sane;
 *   3. a hook that stops in the middle of the prompt ends the turn there,
 *      and the next turn on that context still prints the reference text (the
 *      half-replayed prompt left in the KV cache is correct to reuse);
 *   4. two contexts on one mapped copy of the weights, a lock handed over at
 *      every step including the prompt replay, each print their solo text.
 *
 * Exit status 0 only if all of them hold.
 */
#include "platform.h"
#include "mote_lib.h"

#include <pthread.h>
#include <sched.h>
#include <stdio.h>
#include <string.h>

typedef struct { char buf[8192]; size_t n; } Out;

static int collect(const char *piece, void *user)
{
    Out *o = user;
    size_t l = strlen(piece);
    if (o->n + l < sizeof o->buf) {
        memcpy(o->buf + o->n, piece, l);
        o->n += l;
        o->buf[o->n] = 0;
    }
    return 1;
}

typedef struct {
    int calls, prefill_calls, last_pos, order_ok, stop_at;
} Trace;

static int trace_hook(int pos, int prefill, void *user)
{
    Trace *t = user;
    if (t->calls > 0 && pos != t->last_pos + 1)
        t->order_ok = 0;
    if (prefill && t->prefill_calls != t->calls)
        t->order_ok = 0;              /* a prompt step after a generated one */
    t->calls++;
    t->prefill_calls += prefill;
    t->last_pos = pos;
    return !(t->stop_at && t->calls >= t->stop_at);
}

static int fails;
static void check(int ok, const char *what)
{
    printf("%-4s %s\n", ok ? "ok" : "FAIL", what);
    fails += !ok;
}

/* ---- 4: two contexts, the engine handed over at every step ---- */
static pthread_mutex_t engine = PTHREAD_MUTEX_INITIALIZER;
static int yield_hook(int pos, int prefill, void *user)
{
    (void)pos; (void)prefill; (void)user;
    pthread_mutex_unlock(&engine);
    sched_yield();
    pthread_mutex_lock(&engine);
    return 1;
}
typedef struct { mote *m; const char *prompt; Out out; } Job;
static const mote_params P48 = { 0.0f, 0.0f, 1.1f, 0, 48 };
static void *run_job(void *arg)
{
    Job *j = arg;
    pthread_mutex_lock(&engine);
    mote_generate(j->m, j->prompt, &P48, collect, &j->out);
    pthread_mutex_unlock(&engine);
    return NULL;
}

int main(int argc, char **argv)
{
    if (argc < 3) {
        fprintf(stderr, "usage: %s <model> <tokenizer>\n", argv[0]);
        return 2;
    }
    size_t ml, tl;
    int fd1, fd2;
    void *model = mote_map_file(argv[1], &ml, &fd1);
    void *tok = mote_map_file(argv[2], &tl, &fd2);
    if (!model || !tok) { perror("step_hook_test: map"); return 1; }

    const char *prompts[2] = {
        "<|im_start|>user\nName three planets and say one thing about each.<|im_end|>\n<|im_start|>assistant\n",
        "<|im_start|>user\nWhat does a resource manager do in an operating system?<|im_end|>\n<|im_start|>assistant\n",
    };

    /* 1. plain text, hook or not */
    Out ref = {{0}, 0}, hooked = {{0}, 0};
    mote *a = mote_create(model, ml, tok, tl);
    mote *b = mote_create(model, ml, tok, tl);
    if (!a || !b) return 1;
    mote_generate(a, prompts[0], &P48, collect, &ref);
    Trace tr = { 0, 0, 0, 1, 0 };
    mote_set_step_hook(b, trace_hook, &tr);
    mote_generate(b, prompts[0], &P48, collect, &hooked);
    check(ref.n > 0 && strcmp(ref.buf, hooked.buf) == 0, "1. text with a step hook == text without");
    check(tr.order_ok && tr.prefill_calls > 0 && tr.calls > tr.prefill_calls,
          "1. hook saw each position once, prompt steps first");
    printf("     %d steps: %d replaying the prompt, %d generating\n",
           tr.calls, tr.prefill_calls, tr.calls - tr.prefill_calls);
    mote_free(a);
    mote_free(b);

    /* 2. under a grammar */
    a = mote_create(model, ml, tok, tl);
    b = mote_create(model, ml, tok, tl);
    char err[256];
    const char *schema = "{\"type\":\"object\",\"properties\":{\"planet\":{\"type\":\"string\","
                         "\"enum\":[\"mercury\",\"venus\",\"earth\",\"mars\",\"jupiter\"]}},"
                         "\"required\":[\"planet\"]}";
    mote_grammar *ga = mote_grammar_compile(a, schema, err, sizeof err);
    mote_grammar *gb = mote_grammar_compile(b, schema, err, sizeof err);
    if (!ga || !gb) { fprintf(stderr, "schema: %s\n", err); return 1; }
    const char *q = "<|im_start|>user\nWhich planet do people live on? Answer as JSON.<|im_end|>\n"
                    "<|im_start|>assistant\n";
    mote_params pg = { 0.0f, 0.0f, 0.0f, 0, 24 };
    Out ca = {{0}, 0}, cb = {{0}, 0};
    int na = mote_generate_constrained(a, q, &pg, ga, collect, &ca);
    Trace tg = { 0, 0, 0, 1, 0 };
    mote_set_step_hook(b, trace_hook, &tg);
    int nb = mote_generate_constrained(b, q, &pg, gb, collect, &cb);
    mote_choice ch;
    mote_last_choice(b, &ch);
    check(na == nb && strcmp(ca.buf, cb.buf) == 0, "2. constrained text with a hook == without");
    check(ch.steps > 0 && ch.prob > 0.0f && ch.prob <= 1.0f && ch.min_margin >= 0.0f
          && ch.min_margin <= ch.prob + 1e-6f, "2. choice statistics in range");
    printf("     %s  (p %.3f, closest call %.3f, %d chosen steps)\n", cb.buf, ch.prob, ch.min_margin,
           ch.steps);
    mote_grammar_free(ga);
    mote_grammar_free(gb);
    mote_free(a);
    mote_free(b);

    /* 3. stop in the middle of the prompt, then run the turn again */
    a = mote_create(model, ml, tok, tl);
    Trace ts = { 0, 0, 0, 1, 0 };
    ts.stop_at = tr.prefill_calls / 2 > 0 ? tr.prefill_calls / 2 : 1;
    mote_set_step_hook(a, trace_hook, &ts);
    Out cut = {{0}, 0};
    int n_cut = mote_generate(a, prompts[0], &P48, collect, &cut);
    /* The Qwen path emits nothing while it replays; the legacy path has always
     * echoed the prompt as it goes, so there the cut text is a prefix of it. */
    check(ts.calls == ts.stop_at && cut.n < ref.n && strncmp(ref.buf, cut.buf, cut.n) == 0,
          "3. a hook that stops mid-prompt ends the turn there, with no generated text");
    mote_set_step_hook(a, NULL, NULL);
    Out again = {{0}, 0};
    mote_generate(a, prompts[0], &P48, collect, &again);
    check(strcmp(ref.buf, again.buf) == 0, "3. the next turn on that context == reference text");
    (void)n_cut;
    mote_free(a);

    /* 4. two contexts, interleaved at every step */
    Job jobs[2];
    Out solo[2];
    for (int i = 0; i < 2; i++) {
        mote *m = mote_create(model, ml, tok, tl);
        memset(&solo[i], 0, sizeof solo[i]);
        mote_generate(m, prompts[i], &P48, collect, &solo[i]);
        mote_free(m);
        memset(&jobs[i], 0, sizeof jobs[i]);
        jobs[i].m = mote_create(model, ml, tok, tl);
        jobs[i].prompt = prompts[i];
        mote_set_step_hook(jobs[i].m, yield_hook, NULL);
    }
    pthread_t th[2];
    for (int i = 0; i < 2; i++)
        pthread_create(&th[i], NULL, run_job, &jobs[i]);
    for (int i = 0; i < 2; i++)
        pthread_join(th[i], NULL);
    int same = 1;
    for (int i = 0; i < 2; i++) {
        same &= jobs[i].out.n > 0 && strcmp(solo[i].buf, jobs[i].out.buf) == 0;
        mote_free(jobs[i].m);
    }
    check(same, "4. two contexts handing over at every step (prompt too) == solo");

    mote_unmap_file(model, ml, fd1);
    mote_unmap_file(tok, tl, fd2);
    puts(fails ? "FAIL" : "ok: the step hook changes when work runs, never what it computes");
    return fails != 0;
}
