
#ifndef PyImgC_IMP_UTILS_H
#define PyImgC_IMP_UTILS_H

#include <Python.h>
using namespace std;

void *PyMem_Calloc(size_t num, size_t size) {
    void *ptr; size_t total = num * size;
    ptr = PyMem_Malloc(total);
    if (ptr != NULL) { memset(ptr, 0, total); }
    return ptr;
}

static bool PyImgC_PathExists(PyObject *path) {
    PyStringObject *putative = reinterpret_cast<PyStringObject *>(path);
    if (!PyString_Check(putative)) {
        PyErr_SetString(PyExc_ValueError, "Bad path string");
        return false;
     }
     PyObject *ospath = PyImport_ImportModuleNoBlock("os.path");
     PyObject *exists = PyObject_GetAttrString(ospath, "exists");
     bool out = (bool)PyObject_IsTrue(
         PyObject_CallFunctionObjArgs(exists, putative, NULL));
     Py_DECREF(exists);
     Py_DECREF(ospath);
     return out;
}

static PyObject *PyImgC_TemporaryPath(PyObject *self, PyObject *args) {
    /// call to cimg::temporary_path()
    gil_release NOGIL;
    const char *cpath = cimg::temporary_path();
    NOGIL.~gil_release();
    return Py_BuildValue("s", cpath);
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
    const char *cpath = PyString_AS_STRING(path);
    Py_DECREF(path);
    gil_release NOGIL;
    const char *filetype = cimg::file_type(NULL, cpath);
    NOGIL.~gil_release();
    return Py_BuildValue("s", filetype);
}

#endif /// PyImgC_IMP_UTILS_H