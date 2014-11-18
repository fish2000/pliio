
#ifndef _MVPMALLOC_H
#define _MVPMALLOC_H

#include <Python.h>

void *PyMem_Calloc(size_t num, size_t size);

// #define MVP_MALLOC  malloc
// #define MVP_CALLOC  calloc
// #define MVP_FREE    free

#define MVP_MALLOC  PyMem_Malloc
#define MVP_CALLOC  PyMem_Calloc
#define MVP_FREE    PyMem_Free

#endif /// _MVPMALLOC_H