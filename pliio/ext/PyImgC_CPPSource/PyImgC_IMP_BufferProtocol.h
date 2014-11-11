
#ifndef PyImgC_PYIMGC_IMP_BUFFERPROTOCOL_H
#define PyImgC_PYIMGC_IMP_BUFFERPROTOCOL_H

#include <Python.h>
#include <numpy/ndarrayobject.h>
#include "numpypp/numpy.hpp"
#include "numpypp/dispatch.hpp"
#include "numpypp/utils.hpp"

using namespace cimg_library;
using namespace std;

static int PyCImage_GetBuffer(PyObject *self, Py_buffer *view, int flags) {
    gil_release NOGIL;
    PyCImage *pyim = reinterpret_cast<PyCImage *>(self);
    if (pyim->cimage && pyim->dtype) {
#define HANDLE(type) { \
            auto cim = pyim->recast<type>(); \
            cim->get_pybuffer(view); \
        }
        SAFE_SWITCH_ON_DTYPE(pyim->dtype, -1);
#undef HANDLE
        Py_INCREF(self);
        view->obj = self;
    }
    return 0;
}

static void PyCImage_ReleaseBuffer(PyObject *self, Py_buffer *view) {
    if (view->shape) { free(view->shape); }
    if (view->strides) { free(view->strides); }
    if (view->obj) {
        Py_DECREF(view->obj);
        view->obj = NULL;
    }
    PyBuffer_Release(view);
}

static PyBufferProcs PyCImage_Buffer3000Methods = {
    0, /*(readbufferproc)*/
    0, /*(writebufferproc)*/
    0, /*(segcountproc)*/
    0, /*(charbufferproc)*/
    (getbufferproc)PyCImage_GetBuffer,
    (releasebufferproc)PyCImage_ReleaseBuffer,
};

#endif /// PyImgC_PYIMGC_IMP_BUFFERPROTOCOL_H