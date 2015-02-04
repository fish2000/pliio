
#ifndef PyImgC_PYIMGC_IMP_SEQUENCEPROTOCOL_H
#define PyImgC_PYIMGC_IMP_SEQUENCEPROTOCOL_H

#include <Python.h>
#include <numpy/ndarrayobject.h>
#include "numpypp/numpy.hpp"
#include "numpypp/dispatch.hpp"
#include "numpypp/utils.hpp"
#include "PyImgC_Constants.h"
#include "PyImgC_PyCImage.h"

using namespace cimg_library;
using namespace std;

/// __len__ implementation
Py_ssize_t PyCImage_Len(PyCImage *pyim);

/// __getitem__ implementation
PyObject *PyCImage_GetItem(PyCImage *pyim, register Py_ssize_t idx);

#endif /// PyImgC_PYIMGC_IMP_SEQUENCEPROTOCOL_H