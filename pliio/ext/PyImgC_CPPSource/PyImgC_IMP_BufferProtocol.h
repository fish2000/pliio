
#ifndef PyImgC_PYIMGC_IMP_BUFFERPROTOCOL_H
#define PyImgC_PYIMGC_IMP_BUFFERPROTOCOL_H

#include <Python.h>
#include "numpypp/dispatch.hpp"
#include "PyImgC_Constants.h"
#include "PyImgC_PyCImage.h"

int PyCImage_GetBuffer(PyObject *self, Py_buffer *view, int flags);
void PyCImage_ReleaseBuffer(PyObject *self, Py_buffer *view);

#endif /// PyImgC_PYIMGC_IMP_BUFFERPROTOCOL_H