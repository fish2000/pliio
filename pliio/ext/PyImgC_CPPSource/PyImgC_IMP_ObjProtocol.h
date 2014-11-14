
#ifndef PyImgC_PYIMGC_IMP_OBJPROTOCOL_H
#define PyImgC_PYIMGC_IMP_OBJPROTOCOL_H

#include <stdio.h>
#include <Python.h>
#include <numpy/ndarrayobject.h>
#include "numpypp/numpy.hpp"
#include "numpypp/dispatch.hpp"
#include "numpypp/utils.hpp"
#include "PyImgC_IMP_Utils.h"

using namespace cimg_library;
using namespace std;

/// SMELF ALERT!!!
static PyObject *PyCImage_LoadFromFileViaCImg(PyObject *smelf, PyObject *args, PyObject *kwargs) {
    PyCImage *self = reinterpret_cast<PyCImage *>(smelf);
    PyObject *path;
    PyArray_Descr *dtype=NULL;
    static char *keywords[] = { "path", "dtype", NULL };

    if (!PyArg_ParseTupleAndKeywords(args, kwargs,
                "O|O&", keywords,
                &path, PyArray_DescrConverter, &dtype)) {
        PyErr_SetString(PyExc_ValueError,
            "cannot load image (bad argument tuple passed to PyCImage_LoadFromFileViaCImg)");
        return NULL;
    }
    
    /// LOAD THAT SHIT
    if (!PyImgC_PathExists(path)) {
        Py_XDECREF(path);
        PyErr_SetString(PyExc_ValueError,
            "path does not exist");
        return NULL;
    }
    
    /// deal with dtype
    if (!self->dtype) {
        if (!dtype) {
            self->dtype = dtype = numpy::dtype_struct<IMGC_DEFAULT_T>();
        } else {
            self->dtype = dtype;
        }
    }
    
    /// load that shit, dogg
    if (self->dtype) {
        gil_ensure GIL;
        /// Base the loaded CImg struct type and ancilliaries
        /// on whatever is in the dtype we already have
#define HANDLE(type) { \
        try { \
            CImg<IMGC_DEFAULT_T> cim(PyString_AS_STRING(path)); \
            self->assign<type>(cim); \
        } catch (CImgArgumentException &err) { \
            Py_XDECREF(dtype); \
            Py_XDECREF(path); \
            PyErr_Format(PyExc_ValueError, \
                "CImg argument error: %.200s", err.what()); \
            return NULL; \
        } catch (CImgIOException &err) { \
            Py_XDECREF(dtype); \
            Py_XDECREF(path); \
            PyErr_Format(PyExc_IOError, \
                "CImg IO error: %.200s", err.what()); \
            return NULL; \
        } \
    }
    SAFE_SWITCH_ON_DTYPE(self->dtype, NULL);
#undef HANDLE
        GIL.~gil_ensure();
    } else if (!self->dtype) {
        /// We don't have a valid dtype - let's make one!
        /// We'll create a CImg<unsigned char> from the file path
        gil_ensure GIL;
        try {
            CImg<IMGC_DEFAULT_T> cim(PyString_AS_STRING(path));
            /// populate our dtype fields and ensconce the new CImg
            self->assign<IMGC_DEFAULT_T>(cim);
            self->dtype = numpy::dtype_struct<IMGC_DEFAULT_T>();
        } catch (CImgArgumentException &err) {
            Py_XDECREF(dtype);
            Py_XDECREF(path);
            PyErr_Format(PyExc_ValueError,
                "CImg argument error: %.200s", err.what());
            return NULL;
        } catch (CImgIOException &err) {
            Py_XDECREF(dtype);
            Py_XDECREF(path);
            PyErr_Format(PyExc_IOError,
                "CImg IO error: %.200s", err.what());
            return NULL;
        }
        GIL.~gil_ensure();
    }
    Py_INCREF(self);
    return reinterpret_cast<PyObject *>(self); /// all is well, return self
}

static PyObject *PyCImage_SaveToFileViaCImg(PyObject *smelf, PyObject *args, PyObject *kwargs) {
    PyCImage *self = reinterpret_cast<PyCImage *>(smelf);
    PyObject *path, *pyoverwrite;
    bool overwrite = false;
    bool exists = false;
    static char *keywords[] = { "path", "overwrite", NULL };

    if (!PyArg_ParseTupleAndKeywords(args, kwargs,
                "S|O", keywords,
                &path, &pyoverwrite)) {
        PyErr_SetString(PyExc_ValueError,
            "cannot save image (bad argument tuple passed to PyCImage_SaveToFileViaCImg)");
        return NULL;
    }
    
    overwrite = PyObject_IsTrue(pyoverwrite);
    exists = PyImgC_PathExists(path);
    
    /// SAVE THAT SHIT
    if (exists && !overwrite) {
        /// DON'T OVERWRITE
        Py_XDECREF(path);
        PyErr_SetString(PyExc_NameError,
            "path already exists");
        return NULL;
    }
    if (exists && overwrite) {
        /// PLEASE DO OVERWRITE
        if (remove(PyString_AS_STRING(path))) {
            if (self->save(PyString_AS_STRING(path))) {
                return reinterpret_cast<PyObject *>(self); /// all is well, return self
            } else {
                Py_XDECREF(path);
                PyErr_SetString(PyExc_OSError,
                    "could not save file");
                return NULL;
            }
        } else {
            Py_XDECREF(path);
            PyErr_SetString(PyExc_SystemError,
                "could not overwrite existing file");
            return NULL;
        }
    }
    
    if (self->save(PyString_AS_STRING(path))) {
        return reinterpret_cast<PyObject *>(self); /// all is well, return self
    }
    
    /// we got here, something must be wrong by now
    Py_XDECREF(path);
    PyErr_SetString(PyExc_OSError,
        "could not save file (out of options)");
    return NULL;
}

/// ALLOCATE / __new__ implementation
static PyObject *PyCImage_new(PyTypeObject *type, PyObject *args, PyObject *kwargs) {
    PyCImage *self;
    self = reinterpret_cast<PyCImage *>(type->tp_alloc(type, 0));
    if (self != None) {
        self->cimage = shared_ptr<CImg_Base>(nullptr);
        self->dtype = None;
    }
    return reinterpret_cast<PyObject *>(self); /// all is well, return self
}

/// __repr__ implementations
static PyObject *PyCImage_Repr(PyCImage *pyim) {
    if (!pyim->cimage) { return PyString_FromString("<PyCImage (empty backing store)>"); }
    int tc = static_cast<int>(pyim->typecode());
    if (pyim->dtype) {
        tc = static_cast<int>(pyim->dtype->type_num);
    }
#define HANDLE(type) { \
    CImg<type> cim = *pyim->recast<type>(); \
    return PyString_FromFormat("<PyCImage (%s|%s) [%ix%i, %ix%lubpp] @ %p>", \
        cim.pixel_type(), typeid(*cim.data()).name(), \
        cim.width(), cim.height(), cim.spectrum(), sizeof(type), \
        pyim); \
    }
    SAFE_SWITCH_ON_TYPECODE(tc, PyString_FromString("<PyCImage (bad backing store)>"));
#undef HANDLE
    return PyString_FromString("<PyCImage (unmatched typecode)>");
}
static const char *PyCImage_ReprCString(PyCImage *pyim) {
    return PyString_AS_STRING(PyCImage_Repr(pyim));
}
static string PyCImage_ReprString(PyCImage *pyim) {
    return string(PyCImage_ReprCString(pyim));
}

/// __str__ implementation
static PyObject *PyCImage_Str(PyCImage *pyim) {
    if (pyim->cimage && pyim->dtype) {
#define HANDLE(type) { \
        auto cim = pyim->recast<type>(); \
        auto value_string = cim->value_string(); \
        return PyString_FromString((const char *)value_string.data()); \
    }
    SAFE_SWITCH_ON_DTYPE(pyim->dtype, NULL);
#undef HANDLE
    }
    return PyString_FromString("");
}

/// __init__ implementation
static int PyCImage_init(PyCImage *self, PyObject *args, PyObject *kwargs) {
    PyObject *buffer = NULL;
    Py_ssize_t nin = -1, offset = 0, raise_errors = 1;
    static char *kwlist[] = { "buffer", "dtype", "count", "offset", "raise_errors", NULL };
    PyArray_Descr *dtype = PyArray_DescrFromType(IMGC_DEFAULT_TYPECODE);

    if (!PyArg_ParseTupleAndKeywords(args, kwargs,
                "|OO&nnn:dtype", kwlist,
                &buffer, PyArray_DescrConverter, &dtype,
                &nin, &offset, &raise_errors)) {
            PyErr_SetString(PyExc_ValueError,
                "cannot initialize PyCImage (bad argument tuple)");
        return -1;
    }

    if (!buffer && !dtype) {
        self->dtype = numpy::dtype_struct<IMGC_DEFAULT_T>();
        return 0; /// ITS CO0
    }

    if (self->dtype == NULL) {
        if (buffer != NULL) {
            if (PyArray_Check(buffer)) {
                dtype = PyArray_DESCR(
                    reinterpret_cast<PyArrayObject *>(buffer));
                self->dtype = dtype;
                Py_INCREF(self->dtype);
            } else {
                dtype = PyArray_DescrFromType(
                    IMGC_DEFAULT_TYPECODE);
                self->dtype = dtype;
                Py_INCREF(self->dtype);
            }
        } else if (dtype != NULL) {
            if (PyArray_DescrCheck(dtype)) {
                self->dtype = dtype;
                Py_INCREF(self->dtype);
            } else {
                /// SHOULD NEVER HAPPEN
                PyErr_Format(PyExc_ValueError,
                    "PyCImage __init__ TOTAL FREAKOUT: %s",
                        PyString_AsString(buffer));
                return -1;
            }
        }
    } else {
        if (dtype != NULL) {
            if (PyArray_DescrCheck(dtype)) {
                Py_XDECREF(self->dtype);
                delete self->dtype;
                self->dtype = dtype;
                Py_INCREF(self->dtype);
            }
        } else {
            /// WE DONT BELIEVE IN NOSING LEBAWSKIIIII
            self->dtype = numpy::dtype_struct<IMGC_DEFAULT_T>();
            Py_INCREF(self->dtype);
        }
    }
    
    if (!buffer) {
        /// GET OUT NOW BEFORE... THE BUFFERINGING
        return 0;
    }
    
    if (PyUnicode_Check(buffer)) {
        /// Fuck, it's unicode... let's de-UTF8 it
        buffer = PyUnicode_AsUTF8String(buffer);
    }
    
    if (PyString_Check(buffer)) {
        /// it's a path string, load (with CImg.h) -- DISPATCH!!!!
        PyObject *out = PyObject_CallMethodObjArgs(
            reinterpret_cast<PyObject *>(self),
            PyString_FromString("load"),
            buffer, self->dtype, NULL);
        if (out == NULL) {
            if (raise_errors) {
                PyErr_Format(PyExc_ValueError,
                    "CImg failed to load from path: %s",
                    PyString_AS_STRING(buffer));
                return -1;
            } else { /* warnings? dunno */ }
        }
        return 0;
    }
    
    if (PyArray_Check(buffer)) {
        /// it's a numpy array
        gil_release NOGIL;
#define HANDLE(type) { \
        self->assign(CImg<type>(buffer)); \
    }
    SAFE_SWITCH_ON_DTYPE(self->dtype, -1);
#undef HANDLE
        return 0;
    }
    
    /// FALLING OFF (DANGER WILL ROBINSON)
    PyErr_SetString(PyExc_ValueError,
        "NOTHING HAPPENED! Buffer: NOPE.");
    return -1;
}

/// DEALLOCATE
static void PyCImage_dealloc(PyCImage *self) {
    Py_XDECREF(self->dtype);
    self->ob_type->tp_free((PyObject *)self);
    self->cleanup();
}

#endif /// PyImgC_PYIMGC_IMP_OBJPROTOCOL_H