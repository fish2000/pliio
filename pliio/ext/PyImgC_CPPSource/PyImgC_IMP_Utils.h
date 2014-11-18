
#ifndef PyImgC_IMP_UTILS_H
#define PyImgC_IMP_UTILS_H

#include <Python.h>
#include <string>
using namespace std;

void *PyMem_Calloc(size_t num, size_t size) {
    void *ptr; size_t total = num * size;
    ptr = PyMem_Malloc(total);
    if (ptr != NULL) { memset(ptr, 0, total); }
    return ptr;
}

static bool PyImgC_PathExists(char *path) {
    struct stat buffer;
    stat(path, &buffer);
    return S_ISREG(buffer.st_mode);
}
static bool PyImgC_PathExists(const char *path) {
    return PyImgC_PathExists(const_cast<char *>(path));
}
static bool PyImgC_PathExists(PyObject *path) {
    return PyImgC_PathExists(PyString_AS_STRING(path));
}
static bool PyImgC_PathExists(string path) {
    return PyImgC_PathExists(path.c_str());
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