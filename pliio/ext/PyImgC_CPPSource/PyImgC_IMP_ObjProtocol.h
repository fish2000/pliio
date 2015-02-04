
#ifndef PyImgC_PYIMGC_IMP_OBJPROTOCOL_H
#define PyImgC_PYIMGC_IMP_OBJPROTOCOL_H

#include <Python.h>
#include <numpy/ndarrayobject.h>
#include "numpypp/numpy.hpp"
#include "numpypp/dispatch.hpp"
#include "numpypp/utils.hpp"
#include "PyImgC_Constants.h"
#include "PyImgC_PyCImage.h"
#include "PyImgC_IMP_Utils.h"

using namespace cimg_library;
using namespace std;

PyObject *PyCImage_LoadFromFileViaCImg(PyObject *smelf, PyObject *args, PyObject *kwargs);
PyObject *PyCImage_SaveToFileViaCImg(PyObject *smelf, PyObject *args, PyObject *kwargs);

/// ALLOCATE / __new__
PyObject *PyCImage_new(PyTypeObject *type, PyObject *args, PyObject *kwargs);

/// __repr__ / __str__
PyObject *PyCImage_Repr(PyCImage *pyim);
const char *PyCImage_ReprCString(PyCImage *pyim);
string PyCImage_ReprString(PyCImage *pyim);
PyObject *PyCImage_Str(PyCImage *pyim);

/// __init__
int PyCImage_init(PyCImage *self, PyObject *args, PyObject *kwargs);

/// DEALLOCATE
void PyCImage_dealloc(PyCImage *self);

#endif /// PyImgC_PYIMGC_IMP_OBJPROTOCOL_H