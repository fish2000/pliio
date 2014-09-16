
#ifndef PyImgC_IMP_CIMAGETEST_H
#define PyImgC_IMP_CIMAGETEST_H

#include <Python.h>
#include <structmember.h>
#include <numpy/ndarrayobject.h>
#include "numpypp/dispatch.hpp"
#include "numpypp/utils.hpp"

#include "PyImgC_Constants.h"
#include "PyImgC_SharedDefs.h"

using namespace cimg_library;
using namespace std;

static PyObject *PyImgC_CImageTest(PyObject *self, PyObject *args, PyObject *kwargs) {
    PyObject *buffer = NULL;
    Py_ssize_t nin = -1, offset = 0;
    static char *kwlist[] = { "buffer", "dtype", "count", "offset", NULL };
    PyArray_Descr *dtype = NULL;

    if (!PyArg_ParseTupleAndKeywords(args, kwargs,
                "O|O&" NPY_SSIZE_T_PYFMT NPY_SSIZE_T_PYFMT, kwlist,
                &buffer, PyArray_DescrConverter, &dtype, &nin, &offset)) {
        Py_XDECREF(dtype);
        return NULL;
    }

    if (dtype == NULL) {
        if (PyArray_Check(buffer)) {
            dtype = PyArray_DESCR(
                reinterpret_cast<PyArrayObject *>(buffer));
        } else {
            dtype = PyArray_DescrFromType(NPY_DEFAULT_TYPE);
        }
    }

    if (PyArray_Check(buffer)) {
        int tc = (int)dtype->type_num;
#define HANDLE(type) { \
        CImg<type> cim(buffer); \
        return Py_BuildValue("iiiii", tc, cim.typecode(), \
                             cim.width(), cim.height(), cim.spectrum()); \
        }
        SAFE_SWITCH_ON_TYPECODE(tc, Py_BuildValue(""));
#undef HANDLE
    }
    
    return Py_BuildValue("");
}

#endif /// PyImgC_IMP_CIMAGETEST_H
