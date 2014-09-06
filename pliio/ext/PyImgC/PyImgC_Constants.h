
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

#endif