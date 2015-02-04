
#include "PyImgC_IMP_SequenceProtocol.h"

/// __len__ implementation
Py_ssize_t PyCImage_Len(PyCImage *pyim) {
    if (pyim->cimage && pyim->dtype) {
#define HANDLE(type) { \
        auto cim = pyim->recast<type>(); \
        return static_cast<Py_ssize_t>(cim->size()); \
    }
    SAFE_SWITCH_ON_DTYPE(pyim->dtype, -1);
#undef HANDLE
    }
    return static_cast<Py_ssize_t>(0);
}

/// __getitem__ implementation
PyObject *PyCImage_GetItem(PyCImage *pyim, register Py_ssize_t idx) {
    if (pyim->cimage && pyim->dtype) {
        int tc = static_cast<int>(pyim->dtype->type_num);
        long op;
        Py_ssize_t siz;
#define HANDLE(type) { \
        auto cim = pyim->recast<type>(); \
        op = static_cast<long>(cim->operator()(idx)); \
        siz = static_cast<Py_ssize_t>(cim->size()); \
    }
    SAFE_SWITCH_ON_TYPECODE(tc, NULL);
#undef HANDLE
        if (idx < 0 || idx >= siz) {
            PyErr_SetString(PyExc_IndexError,
                "index out of range");
            return NULL;
        }
        switch (tc) {
            case NPY_FLOAT:
            case NPY_DOUBLE:
            case NPY_LONGDOUBLE: {
                return Py_BuildValue("f", static_cast<long double>(op));
            }
            break;
            case NPY_USHORT:
            case NPY_UBYTE:
            case NPY_UINT:
            case NPY_ULONG:
            case NPY_ULONGLONG: {
                return Py_BuildValue("k", static_cast<unsigned long>(op));
            }
            break;
        }
        return Py_BuildValue("l", op);
    }
    PyErr_SetString(PyExc_IndexError,
        "image index not initialized");
    return NULL;
}
