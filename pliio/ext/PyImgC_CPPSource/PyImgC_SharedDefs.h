
#ifndef PY_ARRAY_UNIQUE_SYMBOL
#define PY_ARRAY_UNIQUE_SYMBOL PyImgC_PyArray_API_Symbol
#endif /// PY_ARRAY_UNIQUE_SYMBOL

#include <Python.h>
#include <structmember.h>
#include <iostream>
#include <string>

#ifndef NPY_NO_DEPRECATED_API
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#endif

#include <numpy/arrayobject.h>
#include <numpy/ndarraytypes.h>

#define IMGC_DEFAULT_TYPECODE 2 /// NPY_TYPE_UINT8
#define IMGC_DEFAULT_T unsigned char /// same as above


//////////////// TYPEDEFS
#ifndef RAWBUFFER_T_
#define RAWBUFFER_T_ 
typedef struct {
    Py_ssize_t len;
    void *buf;
} rawbuffer_t;
#endif

//////////////// MACROS
#ifndef IMGC_IO_MACROS
#define IMGC_IO_MACROS

void IMGC_OUT(FILE *stream, const char *format, ...) {
    va_list args;
    va_start(args, format);
    vfprintf(stream, format, args);
    va_end(args);
}

#define IMGC_STDOUT(format, ...) IMGC_OUT(stdout, format, ##__VA_ARGS__)
#define IMGC_STDERR(format, ...) IMGC_OUT(stderr, format, ##__VA_ARGS__)

#if IMGC_DEBUG > 0
    #define IMGC_TRACE(format, ...) IMGC_OUT(stderr, format, ##__VA_ARGS__)
#else
    #define IMGC_TRACE(format, ...) ((void)0)
#endif

#endif