/* platform.h — the few calls mote makes into the operating system.
 *
 * The engine core (forward pass, quantized matmuls, tokenizers, sampler) is
 * plain C11 and touches no OS at all. What is left is a short list: map a file,
 * read a monotonic clock, count the cores, and ask the user how many threads
 * to use. They live here so a new host (QNX, a bare-metal board, anything
 * POSIX-ish) is one file to read, not a grep across the tree.
 *
 * Header-only on purpose: the apps that vendor src/ pick this up without a new
 * compile unit in their build files.
 *
 * INCLUDE THIS FIRST, before any system header. It sets the feature macros the
 * POSIX calls below need under -std=c11, and a feature macro only counts if it
 * is defined before the first system header is seen.
 */
#ifndef MOTE_PLATFORM_H
#define MOTE_PLATFORM_H

/* Strict -std=c11 hides POSIX from <time.h> and friends on glibc, musl and
 * QNX. Apple shows everything by default and hides some of it (sysconf's core
 * count) when _POSIX_C_SOURCE is set, so Apple gets nothing. QNX's own switch
 * is _QNX_SOURCE, which is POSIX plus the QNX extensions. */
#if defined(__QNX__)
#  ifndef _QNX_SOURCE
#    define _QNX_SOURCE 1
#  endif
#elif !defined(__APPLE__) && !defined(_POSIX_C_SOURCE) && !defined(_GNU_SOURCE) \
      && !defined(_DEFAULT_SOURCE) && !defined(_XOPEN_SOURCE)
#  define _POSIX_C_SOURCE 200809L
#endif

#include <errno.h>
#include <fcntl.h>
#include <stddef.h>
#include <stdlib.h>
#include <sys/types.h>
#include <time.h>
#include <unistd.h>
#ifndef MOTE_NO_MMAP
#include <sys/mman.h>
#endif

/* .mq and llama2.c checkpoints are little-endian, and the loader points
 * straight into the mapped file instead of decoding it. Every target mote runs
 * on today is little-endian (x86-64, arm64, QNX aarch64le, wasm); a big-endian
 * build would load garbage, so it fails here instead. */
#if defined(__BYTE_ORDER__) && defined(__ORDER_BIG_ENDIAN__) \
    && __BYTE_ORDER__ == __ORDER_BIG_ENDIAN__
#  error "mote reads little-endian model files in place; big-endian hosts are not supported"
#endif

/* Seconds on a clock that never jumps. Wall time (gettimeofday) moves when NTP
 * or an RTC sync steps it, and a step in the middle of a run is a wrong tok/s.
 * CLOCK_MONOTONIC is POSIX: Linux, macOS 10.12+, QNX, and emscripten. */
static inline double mote_now_s(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

/* Cores the OS says are online, at least 1. _SC_NPROCESSORS_ONLN is not POSIX
 * but Linux, Apple, QNX, BSD and emscripten all provide it; a host without it
 * runs on one core rather than failing to build. */
static inline int mote_cpu_count(void)
{
#ifdef _SC_NPROCESSORS_ONLN
    long n = sysconf(_SC_NPROCESSORS_ONLN);
    return n > 1 ? (int)n : 1;
#else
    return 1;
#endif
}

/* MOTE_THREADS=n pins the thread count so a benchmark can compare engines, or
 * thread counts, at a fixed width. 0 means unset: use every core. Read once,
 * because the parallel-for that asks runs once per matmul. */
static inline int mote_env_threads(void)
{
    static int forced = -1;
    if (forced < 0) {
        const char *mt = getenv("MOTE_THREADS");
        int n = mt ? atoi(mt) : 0;
        forced = n > 0 ? n : 0;
    }
    return forced;
}

/* Map a file read-only. Returns its base and size, plus a handle to give back
 * to mote_unmap_file, or NULL with errno set.
 *
 * Built with -DMOTE_NO_MMAP it reads the file into the heap instead, for a
 * filesystem or RTOS image where mmap of a file is unavailable or unwanted. The
 * engine cannot tell the difference: it only ever sees a pointer and a size. */
static inline void *mote_map_file(const char *path, size_t *size, int *fd_out)
{
    *fd_out = -1;
    int fd = open(path, O_RDONLY);
    if (fd == -1)
        return NULL;
    off_t sz = lseek(fd, 0, SEEK_END);
    if (sz < 0) {
        int e = errno; close(fd); errno = e;
        return NULL;
    }
    *size = (size_t)sz;
#ifndef MOTE_NO_MMAP
    void *p = mmap(NULL, (size_t)sz, PROT_READ, MAP_PRIVATE, fd, 0);
    if (p == MAP_FAILED) {
        int e = errno; close(fd); errno = e;
        return NULL;
    }
    *fd_out = fd;
    return p;
#else
    char *p = malloc(sz > 0 ? (size_t)sz : 1);
    if (p == NULL || lseek(fd, 0, SEEK_SET) != 0) {
        int e = p ? errno : ENOMEM; free(p); close(fd); errno = e;
        return NULL;
    }
    size_t got = 0;
    while (got < (size_t)sz) {
        ssize_t r = read(fd, p + got, (size_t)sz - got);
        if (r <= 0) {
            int e = r < 0 ? errno : EIO; free(p); close(fd); errno = e;
            return NULL;
        }
        got += (size_t)r;
    }
    close(fd);
    return p;
#endif
}

static inline void mote_unmap_file(void *base, size_t size, int fd)
{
    if (base == NULL)
        return;
#ifndef MOTE_NO_MMAP
    munmap(base, size);
    if (fd != -1)
        close(fd);
#else
    (void)size; (void)fd;
    free(base);
#endif
}

#endif /* MOTE_PLATFORM_H */
