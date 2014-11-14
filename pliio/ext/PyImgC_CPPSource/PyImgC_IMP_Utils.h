
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

static PyObject *PyImgC_TemporaryPath(PyObject *self, PyObject *args) {
    /// call to cimg::temporary_path()
    return PyString_FromString(cimg::temporary_path());
}

static PyObject *PyImgC_GuessType(PyObject *self, PyObject *args) {
    /// call to cimg::file_type()
    PyObject *path;
    if (!PyArg_ParseTuple(args, "S", &path)) {
        PyErr_SetString(PyExc_ValueError,
            "bad arguments to guess_type");
        return NULL;
    }
    if (!PyImgC_PathExists(path)) {
        PyErr_SetString(PyExc_NameError,
            "path does not exist");
        return NULL;
    }
    return PyString_FromString(
        cimg::file_type(NULL,
            PyString_AS_STRING(path)));
}

#endif /// PyImgC_IMP_UTILS_H