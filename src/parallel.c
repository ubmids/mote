/* parallel.c — platform backends for mote_parallel_for. See parallel.h.
 *
 * Which backend a build gets, first match wins:
 *   -DMOTE_SERIAL                 one plain call (any host; a reference build)
 *   -DMOTE_PTHREADS               the POSIX worker pool (any POSIX host)
 *   Apple                         Grand Central Dispatch
 *   -fopenmp                      OpenMP
 *   QNX, or emscripten -pthread   the POSIX worker pool
 *   anything else                 one plain call
 * MOTE_PTHREADS exists so the pool can be built and checked on a laptop before
 * it ever meets the target that needs it. MOTE_FOLLOW_PRIORITY makes the pool's
 * workers track the caller's scheduling priority job by job (always on for QNX;
 * see the pool below). */
#include "platform.h"   /* first: it sets the feature macros */

#include "parallel.h"

#if defined(MOTE_SERIAL)

void mote_parallel_for(int count, void (*fn)(int, int, void *), void *ctx)
{
    if (count > 0)
        fn(0, count, ctx);
}

#elif defined(__APPLE__) && !defined(MOTE_PTHREADS)

#include <dispatch/dispatch.h>
#include <stdlib.h>

void mote_parallel_for(int count, void (*fn)(int, int, void *), void *ctx)
{
    if (count <= 0)
        return;

    /* MOTE_RESERVE_CORES leaves that many cores free. It exists for hosts with a
     * UI: on iOS the matmul runs on a background queue, and pegging every core
     * starves the main thread so the send button freezes during prefill. A CLI
     * has no such problem and wants the throughput, so the default is 0 and the
     * app build opts in with -DMOTE_RESERVE_CORES=1. */
#ifndef MOTE_RESERVE_CORES
#define MOTE_RESERVE_CORES 0
#endif
    int chunks = mote_cpu_count();
    if (MOTE_RESERVE_CORES > 0 && chunks > MOTE_RESERVE_CORES + 1)
        chunks -= MOTE_RESERVE_CORES;
    /* MOTE_THREADS=n splits the work into exactly n chunks, so a benchmark can
     * compare against another engine at the same thread count. Unset = all cores. */
    int forced = mote_env_threads();
    if (forced > 0)
        chunks = forced;
    if (chunks > count)
        chunks = count;
    if (chunks <= 1) {
        fn(0, count, ctx);
        return;
    }

    dispatch_apply((size_t)chunks, DISPATCH_APPLY_AUTO, ^(size_t c) {
        int begin = (int)((long)c * count / chunks);
        int end   = (int)(((long)c + 1) * count / chunks);
        fn(begin, end, ctx);
    });
}

#elif defined(_OPENMP) && !defined(MOTE_PTHREADS)

#include <omp.h>

void mote_parallel_for(int count, void (*fn)(int, int, void *), void *ctx)
{
    if (count <= 0)
        return;
#pragma omp parallel
    {
        int nt = omp_get_num_threads();
        int t = omp_get_thread_num();
        int begin = (int)((long)t * count / nt);
        int end   = (int)(((long)t + 1) * count / nt);
        if (begin < end)
            fn(begin, end, ctx);
    }
}

#elif defined(__EMSCRIPTEN_PTHREADS__) || defined(MOTE_PTHREADS) || defined(__QNX__)

/* A small persistent pool of POSIX threads, built on first use and then fed
 * jobs. Spawning a thread per matmul would cost more than the matmul. The
 * calling thread works too (chunk 0) and waits on a condvar for the rest.
 *
 * WebAssembly: pthreads map onto web workers. Link with -sPTHREAD_POOL_SIZE so
 * the workers exist up front; creating one lazily needs the browser's main
 * thread, which may be busy, and that way lies deadlock. The engine already
 * lives in a worker, where blocking is allowed.
 *
 * QNX: workers are created with default attributes, so they inherit the
 * creating thread's priority and scheduling policy. Run the engine at a
 * real-time priority (`on -p`, or pthread_setschedparam before the first
 * forward pass) and the whole pool runs there with it.
 *
 * Inheriting once is not enough when the caller's priority changes later, which
 * on QNX it does whenever a server thread picks up a message: it runs at the
 * priority of the client that sent it. Workers stuck at the first caller's
 * priority would then compute a low-priority client's tokens at a high priority
 * (stealing time from whatever real-time work sits between the two), or a
 * high-priority client's tokens at a low one. So on QNX, or with
 * -DMOTE_FOLLOW_PRIORITY anywhere else, each job first moves the workers to the
 * calling thread's policy and priority, when those have changed. Callers must
 * already serialize jobs (there is one job slot), so this needs no extra lock. */
#include <pthread.h>
#include <sched.h>

#if defined(__QNX__) || defined(MOTE_FOLLOW_PRIORITY)
#define FOLLOW_PRIORITY 1
#endif

#if defined(__EMSCRIPTEN_PTHREADS__)
#define POOL_MAX 8     /* must not exceed web/build.sh's PTHREAD_POOL_SIZE */
#else
#define POOL_MAX 64
#endif

typedef struct {
    void (*fn)(int, int, void *);
    void *ctx;
    int count, chunks;
} Job;

static pthread_mutex_t mu = PTHREAD_MUTEX_INITIALIZER;
static pthread_cond_t cv_work = PTHREAD_COND_INITIALIZER;
static pthread_cond_t cv_done = PTHREAD_COND_INITIALIZER;
static Job job;
static int generation, pending, pool_n;
static pthread_t pool_tid[POOL_MAX];

#ifdef FOLLOW_PRIORITY
/* Move every worker to the calling thread's scheduling policy and priority.
 * Two kernel calls per worker, and only when the caller's priority differs
 * from the last job's, so steady-state cost is one getschedparam per job. A
 * worker that cannot be moved (no privilege for that policy) keeps running
 * where it was; the job still completes, it just is not re-prioritized. */
static void follow_caller_priority(void)
{
    static int last_policy = -1, last_prio;
    int policy;
    struct sched_param sp;
    if (pthread_getschedparam(pthread_self(), &policy, &sp) != 0)
        return;
    if (policy == last_policy && sp.sched_priority == last_prio)
        return;
    for (int i = 0; i + 1 < pool_n; i++)
        pthread_setschedparam(pool_tid[i], policy, &sp);
    last_policy = policy;
    last_prio = sp.sched_priority;
}
#endif

static void *worker(void *arg)
{
    int self = (int)(long)arg;   /* worker i handles chunk i+1 */
    int seen = 0;
    for (;;) {
        pthread_mutex_lock(&mu);
        while (generation == seen)
            pthread_cond_wait(&cv_work, &mu);
        seen = generation;
        Job j = job;
        pthread_mutex_unlock(&mu);

        int c = self + 1;
        if (c < j.chunks) {
            int begin = (int)((long)c * j.count / j.chunks);
            int end   = (int)((long)(c + 1) * j.count / j.chunks);
            j.fn(begin, end, j.ctx);
        }
        pthread_mutex_lock(&mu);
        if (--pending == 0)
            pthread_cond_signal(&cv_done);
        pthread_mutex_unlock(&mu);
    }
    return NULL;
}

void mote_parallel_for(int count, void (*fn)(int, int, void *), void *ctx)
{
    if (count <= 0)
        return;

    if (pool_n == 0) {
        int want = mote_env_threads();
        if (want == 0)
            want = mote_cpu_count();
        pool_n = want > POOL_MAX ? POOL_MAX : want;
        for (int i = 0; i + 1 < pool_n; i++) {
            if (pthread_create(&pool_tid[i], NULL, worker, (void *)(long)i) != 0) {
                pool_n = i + 1;   /* run with what we got */
                break;
            }
            pthread_detach(pool_tid[i]);
        }
    }

    int chunks = pool_n < count ? pool_n : count;
    if (chunks <= 1) {
        fn(0, count, ctx);
        return;
    }

#ifdef FOLLOW_PRIORITY
    follow_caller_priority();
#endif

    pthread_mutex_lock(&mu);
    job.fn = fn; job.ctx = ctx; job.count = count; job.chunks = chunks;
    pending = pool_n - 1;
    generation++;
    pthread_cond_broadcast(&cv_work);
    pthread_mutex_unlock(&mu);

    fn(0, (int)((long)count / chunks), ctx);   /* chunk 0, on this thread */

    pthread_mutex_lock(&mu);
    while (pending > 0)
        pthread_cond_wait(&cv_done, &mu);
    pthread_mutex_unlock(&mu);
}

#else

void mote_parallel_for(int count, void (*fn)(int, int, void *), void *ctx)
{
    if (count > 0)
        fn(0, count, ctx);
}

#endif
