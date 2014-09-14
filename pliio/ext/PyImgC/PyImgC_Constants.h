
#ifndef PyImgC_CONSTANTS_H
#define PyImgC_CONSTANTS_H

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

#define IMGC_DEFAULT_TYPECODE 2
#define IMGC_DEFAULT_T unsigned char
#define IMGT typename T = IMGC_DEFAULT_T

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

/*
#ifndef PyObject_TypeName
#define PyObject_TypeName(object) static_cast<const char *>((PyTypeObject *)PyObject_Type(reinterpret_cast<PyObject *>(object)))->tp_name)
#endif
*/

#endif