
#ifndef PyImgC_PYIMGC_IMP_GETSET_H
#define PyImgC_PYIMGC_IMP_GETSET_H

#include <Python.h>
#include <numpy/ndarrayobject.h>
#include "numpypp/numpy.hpp"
#include "numpypp/dispatch.hpp"
#include "numpypp/utils.hpp"
#include "PyImgC_IMP_PHash.h"

using namespace cimg_library;
using namespace std;

/// pycimage.dtype getter/setter
static PyObject     *PyCImage_GET_dtype(PyCImage *self, void *closure) {
    BAIL_WITHOUT(self->dtype);
    Py_INCREF(self->dtype);
    return reinterpret_cast<PyObject *>(self->dtype);
}
static int           PyCImage_SET_dtype(PyCImage *self, PyObject *value, void *closure) {
    PyArray_Descr *dtype;
    PyArray_DescrConverter(value, &dtype);
    Py_DECREF(value);
    if (self->dtype) { Py_DECREF(self->dtype); }
    self->dtype = dtype;
    Py_INCREF(dtype);
    return 0;
}

/// pycimage.length getter
static PyObject     *PyCImage_GET_height(PyCImage *self, void *closure) {
    BAIL_WITHOUT(self->dtype);
#define HANDLE(type) { \
        auto cim = self->recast<type>(); \
        return PyInt_FromLong(cim->height()); \
    }
    SAFE_SWITCH_ON_DTYPE(self->dtype, Py_BuildValue(""));
#undef HANDLE
    return Py_BuildValue("");
}

/// pycimage.width getter
static PyObject     *PyCImage_GET_width(PyCImage *self, void *closure) {
    BAIL_WITHOUT(self->dtype);
#define HANDLE(type) { \
        auto cim = self->recast<type>(); \
        return PyInt_FromLong(cim->width()); \
    }
    SAFE_SWITCH_ON_DTYPE(self->dtype, Py_BuildValue(""));
#undef HANDLE
    return Py_BuildValue("");
}

/// pycimage.spectrum getter
static PyObject     *PyCImage_GET_spectrum(PyCImage *self, void *closure) {
    BAIL_WITHOUT(self->dtype);
#define HANDLE(type) { \
        auto cim = self->recast<type>(); \
        return PyInt_FromLong(cim->spectrum()); \
    }
    SAFE_SWITCH_ON_DTYPE(self->dtype, Py_BuildValue(""));
#undef HANDLE
    return Py_BuildValue("");
}

/// pycimage.size getter -- NB This is NOT the same as len(pycimage)
static PyObject     *PyCImage_GET_size(PyCImage *self, void *closure) {
    BAIL_WITHOUT(self->dtype);
#define HANDLE(type) { \
        auto cim = self->recast<type>(); \
        return Py_BuildValue("ii", cim->width(), cim->height()); \
    }
    SAFE_SWITCH_ON_DTYPE(self->dtype, Py_BuildValue(""));
#undef HANDLE
    return Py_BuildValue("");
}

/// pycimage.shape getter
static PyObject     *PyCImage_GET_shape(PyCImage *self, void *closure) {
    BAIL_WITHOUT(self->dtype);
#define HANDLE(type) { \
        auto cim = self->recast<type>(); \
        return Py_BuildValue("iii", cim->height(), cim->width(), cim->spectrum()); \
    }
    SAFE_SWITCH_ON_DTYPE(self->dtype, Py_BuildValue(""));
#undef HANDLE
    return Py_BuildValue("");
}

/// pycimage.itemsize getter
static PyObject     *PyCImage_GET_itemsize(PyCImage *self, void *closure) {
    BAIL_WITHOUT(self->dtype);
#define HANDLE(type) return PyInt_FromLong(sizeof(type));
    SAFE_SWITCH_ON_DTYPE(self->dtype, Py_BuildValue(""));
#undef HANDLE
    return Py_BuildValue("");
}

/// pycimage.strides getter
static PyObject     *PyCImage_GET_strides(PyCImage *self, void *closure) {
    BAIL_WITHOUT(self->dtype);
#define HANDLE(type) { \
        auto cim = self->recast<type>(); \
        return Py_BuildValue("iii", \
            cim->width() * cim->spectrum() * sizeof(type), \
            cim->spectrum() * sizeof(type), \
            sizeof(type)); \
    }
    SAFE_SWITCH_ON_DTYPE(self->dtype, Py_BuildValue(""));
#undef HANDLE
    return Py_BuildValue("");
}

/// pycimage.ndarray getter
static PyObject     *PyCImage_GET_ndarray(PyCImage *self, void *closure) {
    BAIL_WITHOUT(self->dtype);
#define HANDLE(type) { \
        auto cim = self->recast<type>(); \
        return cim->get_pyarray(); \
    }
    SAFE_SWITCH_ON_DTYPE(self->dtype, Py_BuildValue(""));
#undef HANDLE
    return Py_BuildValue("");
}

/// pycimage.phash getter
static PyObject     *PyCImage_GET_phash(PyCImage *self, void *closure) {
    gil_release NOGIL;
    return PyLong_FromUnsignedLongLong(
        ph_dct_imagehash(*self->recast<uint8_t>()));
}

static PyGetSetDef PyCImage_getset[] = {
    {
        "dtype",
            (getter)PyCImage_GET_dtype,
            (setter)PyCImage_SET_dtype,
            "Data Type (numpy.dtype)", None },
    {
        "height",
            (getter)PyCImage_GET_height,
            None,
            "Image Height", None },
    {
        "width",
            (getter)PyCImage_GET_width,
            None,
            "Image Width", None },
    {
        "spectrum",
            (getter)PyCImage_GET_spectrum,
            None,
            "Image Spectrum (Color Depth)", None },
    {
        "size",
            (getter)PyCImage_GET_size,
            None,
            "Image Size (PIL-style size tuple)", None },
    {
        "shape",
            (getter)PyCImage_GET_shape,
            None,
            "Image Shape (NumPy-style shape tuple)", None },
    {
        "itemsize",
            (getter)PyCImage_GET_itemsize,
            None,
            "Item Size", None },
    {
        "strides",
            (getter)PyCImage_GET_strides,
            None,
            "Image Stride Offsets (NumPy-style strides tuple)", None },
    {
        "ndarray",
            (getter)PyCImage_GET_ndarray,
            None,
            "Numpy Array Object", None },
    {
        "phash",
            (getter)PyCImage_GET_phash,
            None,
            "Perceptual Image Hash (phash)", None },
    SENTINEL
};

#endif /// PyImgC_PYIMGC_IMP_GETSET_H