
#ifndef PyImgC_PYIMGC_IMP_GETSET_H
#define PyImgC_PYIMGC_IMP_GETSET_H

#include <Python.h>
#include <numpy/ndarrayobject.h>
#include "numpypp/numpy.hpp"
#include "numpypp/dispatch.hpp"
#include "numpypp/utils.hpp"
#include "PyImgC_Constants.h"
#include "PyImgC_PyCImage.h"
#include "PyImgC_IMP_PHash.h"

using namespace cimg_library;
using namespace std;

/// pycimage.dtype getter/setter
PyObject     *PyCImage_GET_dtype(PyCImage *self, void *closure);
int           PyCImage_SET_dtype(PyCImage *self, PyObject *value, void *closure);

/// pycimage.length getter
PyObject     *PyCImage_GET_height(PyCImage *self, void *closure);

/// pycimage.width getter
PyObject     *PyCImage_GET_width(PyCImage *self, void *closure);

/// pycimage.spectrum getter
PyObject     *PyCImage_GET_spectrum(PyCImage *self, void *closure);

/// pycimage.size getter -- NB This is NOT the same as len(pycimage)
PyObject     *PyCImage_GET_size(PyCImage *self, void *closure);

/// pycimage.shape getter
PyObject     *PyCImage_GET_shape(PyCImage *self, void *closure);

/// pycimage.itemsize getter
PyObject     *PyCImage_GET_itemsize(PyCImage *self, void *closure);

/// pycimage.strides getter
PyObject     *PyCImage_GET_strides(PyCImage *self, void *closure);

/// pycimage.ndarray getter
PyObject     *PyCImage_GET_ndarray(PyCImage *self, void *closure);

/// pycimage.dct_phash getter
PyObject     *PyCImage_GET_dct_phash(PyCImage *self, void *closure);

/// pycimage.mh_phash getter
PyObject     *PyCImage_GET_mh_phash(PyCImage *self, void *closure);

#endif /// PyImgC_PYIMGC_IMP_GETSET_H