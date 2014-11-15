
#ifndef PyHashTree_CONSTANTS_H
#define PyHashTree_CONSTANTS_H

#ifndef PY_ARRAY_UNIQUE_SYMBOL
#define PY_ARRAY_UNIQUE_SYMBOL PyHashTree_PyArray_API_Symbol
#endif
#ifndef NPY_NO_DEPRECATED_API
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#endif

#include <Python.h>

/// lil' bit pythonic
#ifndef False
#define False 0
#endif
#ifndef True
#define True 1
#endif
#ifndef None
#define None NULL
#endif

#ifndef IMGC_DEBUG
#define IMGC_DEBUG False
#endif

/// UGH
#ifndef SENTINEL
#define SENTINEL {NULL}
#endif
#ifndef PyMODINIT_FUNC
#define PyMODINIT_FUNC void
#endif

#ifndef BAIL_WITHOUT
#define BAIL_WITHOUT(thing) if (!thing) return None
#endif

//////////////// CONSTANTS
#if PY_VERSION_HEX <= 0x03000000
#define IMGC_PY3 False
#define IMGC_PY2 True
#else
#define IMGC_PY3 True
#define IMGC_PY2 False
#endif

#ifndef PyGetNone
#define PyGetNone Py_BuildValue("")
#endif

#endif /// PyHashTree_CONSTANTS_H