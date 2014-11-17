/*
 * fmem.c : fmemopen() on top of BSD's funopen()
 * 20081017 AF
 */

#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif 

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <Python.h>

#ifndef linux
struct fmem {
    size_t pos;
    size_t size;
    char *buffer;
};
typedef struct fmem fmem_t;

static int readfn(void *handler, char *buf, int size)
{
    int count = 0;
    fmem_t *mem = (fmem_t *)handler;
    size_t available = mem->size - mem->pos;

    if (size > available) { size = available; }
    for (count=0; count < size; mem->pos++, count++) {
        buf[count] = mem->buffer[mem->pos];
    }

    return count;
}

static int writefn(void *handler, const char *buf, int size)
{
    int count = 0;
    fmem_t *mem = (fmem_t *)handler;
    size_t available = mem->size - mem->pos;

    if (size > available) { size = available; }
    for (count=0; count < size; mem->pos++, count++) {
        mem->buffer[mem->pos] = buf[count];
    }

    return count; // ? count : size;
}

static fpos_t seekfn(void *handler, fpos_t offset, int whence)
{
    size_t pos;
    fmem_t *mem = (fmem_t *)handler;

    switch (whence) {
        case SEEK_SET: pos = offset; break;
        case SEEK_CUR: pos = mem->pos + offset; break;
        case SEEK_END: pos = mem->size + offset; break;
        default: return -1;
    }

    if (pos > mem->size) { return -1; }

    mem->pos = pos;
    return (fpos_t) pos;
}

static int closefn(void *handler)
{
    PyMem_Free(handler);
    return 0;
}

/* simple, but portable version of fmemopen for OS X / BSD */
FILE *fmemopen(void *buf, size_t size, const char *mode)
{
    fmem_t *mem = (fmem_t *) PyMem_Malloc(sizeof(fmem_t));

    memset(mem, 0, sizeof(fmem_t));
    mem->size = size, mem->buffer = (char *)buf;
    return funopen(mem, readfn, writefn, seekfn, closefn);
}
#endif