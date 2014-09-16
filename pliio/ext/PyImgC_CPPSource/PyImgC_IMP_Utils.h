
#ifndef PyImgC_IMP_UTILS_H
#define PyImgC_IMP_UTILS_H

#include <Python.h>
using namespace std;

static bool PyImgC_PathExists(PyObject *path) {
    PyStringObject *putative = reinterpret_cast<PyStringObject *>(path);
    if (!PyString_Check(putative)) {
        PyErr_SetString(PyExc_ValueError, "Bad path string");
        return false;
     }
     PyObject *ospath = PyImport_ImportModuleNoBlock("os.path");
     PyObject *exists = PyObject_GetAttrString(ospath, "exists");
     return (bool)PyObject_IsTrue(
         PyObject_CallFunctionObjArgs(exists, putative, NULL));
}

#endif /// PyImgC_IMP_UTILS_H