
#include <iostream>
#include <vector>
#include <string>
#include <typeinfo>

#include <Python.h>
#include <structmember.h>
#include <numpy/ndarrayobject.h>
#include "numpypp/numpy.hpp"
#include "numpypp/structcode.hpp"
#include "numpypp/dispatch.hpp"
#include "numpypp/utils.hpp"

#include "PyImgC_Constants.h"
#include "PyImgC_SharedDefs.h"
#include "PyImgC_Types.h"
#include "PyImgC_PyCImage.h"
#include "PyImgC_IMP_CImageTest.h"
#include "PyImgC_IMP_StructCodeParse.h"
#include "PyImgC_IMP_PyBufferDict.h"
#include "PyImgC_IMP_Utils.h"

using namespace cimg_library;
using namespace std;

/// SMELF ALERT!!!
static PyObject *PyCImage_LoadFromFileViaCImg(PyObject *smelf, PyObject *args, PyObject *kwargs) {
    PyCImage *self = reinterpret_cast<PyCImage *>(smelf);
    PyObject *path;
    PyArray_Descr *dtype=None;
    unsigned int tc = 0;
    static char *keywords[] = { "path", "dtype", None };

    if (!PyArg_ParseTupleAndKeywords(args, kwargs,
                "O|O&", keywords,
                &path, PyArray_DescrConverter, &dtype)) {
        PyErr_SetString(PyExc_ValueError,
            "cannot load image (bad argument tuple passed to PyCImage_LoadFromFileViaCImg)");
        return NULL;
    }
    
    /// LOAD THAT SHIT
    if (!PyImgC_PathExists(path)) {
        Py_XDECREF(dtype);
        Py_XDECREF(path);
        PyErr_SetString(PyExc_ValueError,
            "path does not exist");
        return NULL;
    }
    
    /// deal with dtype
    if (dtype != NULL) {
        Py_DECREF(self->dtype);
        self->dtype = NULL;
        Py_INCREF(dtype);
        self->dtype = dtype;
    } else {
        self->dtype = dtype = numpy::dtype_struct<IMGC_DEFAULT_T>();
    }
    tc = self->typecode();
    
    /// load that shit, dogg
    if (tc) {
        gil_ensure NOGIL;
        /// Base the loaded CImg struct type and ancilliaries
        /// on whatever is in the dtype we already have
#define HANDLE(type) \
        try { \
            CImg<type> cim(PyString_AS_STRING(path)); \
            self->cimage = CImage_TypePointer<type>(cim); \
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
        }
            SAFE_SWITCH_ON_TYPECODE(tc, NULL);
#undef HANDLE
        NOGIL.~gil_ensure();
    } else if (!tc) {
        /// We don't have a valid dtype - let's make one!
        /// We'll create a CImg<unsigned char> from the file path
        gil_ensure NOGIL;
        try {
            CImg<IMGC_DEFAULT_T> cim(PyString_AS_STRING(path));
            /// populate our dtype fields and ensconce the new CImg
            /// in a new instance of CImage_Type<unsigned char>
            self->cimage = CImage_TypePointer<IMGC_DEFAULT_T>(cim);
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
        NOGIL.~gil_ensure();
    }
    return reinterpret_cast<PyObject *>(self); /// all is well, return self
}

static PyObject *PyCImage_new(PyTypeObject *type, PyObject *args, PyObject *kwargs) {
    PyCImage *self;
    self = (PyCImage *)type->tp_alloc(type, 0);
    if (self != None) {
        self->cimage = unique_ptr<CImage_SubBase>(nullptr);
        self->dtype = None;
    }
    return (PyObject *)self;
}

/// forward declaration (for debugging)
static PyObject *PyCImage_Repr(PyCImage *pyim);

static int PyCImage_init(PyCImage *self, PyObject *args, PyObject *kwargs) {
    PyObject *buffer = NULL;
    Py_ssize_t nin = -1, offset = 0;
    static char *kwlist[] = { "buffer", "dtype", "count", "offset", NULL };
    PyArray_Descr *dtype = NULL;

    if (!PyArg_ParseTupleAndKeywords(args, kwargs,
                "|OO&nn:dtype", kwlist,
                &buffer, PyArray_DescrConverter, &dtype, &nin, &offset)) {
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
                dtype = PyArray_DESCR(reinterpret_cast<PyArrayObject *>(buffer));
                self->dtype = dtype;
                Py_INCREF(self->dtype);
            } else {
                dtype = PyArray_DescrFromType(IMGC_DEFAULT_TYPECODE);
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
    
    if (PyString_Check(buffer)) {
        /// it's a path string, load (with CImg.h)
        /// DISPATCH!!!!
        PyObject *out = PyObject_CallMethodObjArgs(
            reinterpret_cast<PyObject *>(self),
            PyString_FromString("cimg_load"),
            buffer, self->dtype, NULL);
        if (out == NULL) {
            PyErr_Format(PyExc_ValueError,
                "CImg failed to load from path: %s",
                PyString_AS_STRING(buffer));
            return -1;
        }
        return 0;
    }
    
    if (PyArray_Check(buffer)) {
        /// it's a numpy array
        int tc = (int)self->dtype->type_num;
#define HANDLE(type) \
        self->cimage = CImage_TypePointer<type>(buffer); \
        self->viewptr = (void *)self->force<type>()->get(false);
        SAFE_SWITCH_ON_TYPECODE(tc, -1);
#undef HANDLE
        IMGC_CERR("> REPR:" << PyString_AS_STRING(PyCImage_Repr(self)));
        return 0;
    }
    
    /// FALLING OFF
    PyErr_SetString(PyExc_ValueError,
        "NOTHING HAPPENED! Buffer: NOPE.");
    return -1;
}

static void PyCImage_dealloc(PyCImage *self) {
    Py_XDECREF(self->dtype);
    self->ob_type->tp_free((PyObject *)self);
}

static PyObject     *PyCImage_GET_dtype(PyCImage *self, void *closure) {
    BAIL_WITHOUT(self->dtype);
    Py_INCREF(self->dtype);
    return reinterpret_cast<PyObject *>(self->dtype);
}
static int           PyCImage_SET_dtype(PyCImage *self, PyObject *value, void *closure) {
    if (self->dtype) { Py_DECREF(self->dtype); }
    Py_INCREF(value);
    self->dtype = reinterpret_cast<PyArray_Descr *>(value);
    return 0;
}

static PyGetSetDef PyCImage_getset[] = {
    {
        "dtype",
            (getter)PyCImage_GET_dtype,
            (setter)PyCImage_SET_dtype,
            "Data Type (numpy.dtype)", None},
    SENTINEL
};

static int PyCImage_GetBuffer(PyCImage *pyim, Py_buffer &buf, int flags=0) {
    if (pyim->cimage && pyim->dtype) {
        auto cim = pyim->view();
        buf = cim->get_pybuffer();
        if (buf.len) { return 0; } /// success
    }
    return -1;
}

static PyObject *PyCImage_Repr(PyCImage *pyim) {
    if (!pyim->cimage) { PyString_FromString("<PyCImage (empty backing stores)>"); }
    int tc = (int)pyim->typecode();
    if (pyim->dtype) {
        tc = (int)pyim->dtype->type_num;
    }
#define HANDLE(type) \
    CImg<type> cim = pyim->recast<type>()->get(false); \
    return PyString_FromFormat("<PyCImage (%s|%s) [%ix%i, %ix%lubpp] @ %p>", \
        cim.pixel_type(), typeid(*cim.data()).name(), \
        cim.width(), cim.height(), cim.spectrum(), sizeof(type), \
        pyim);
    SAFE_SWITCH_ON_TYPECODE(tc, PyString_FromString("<PyCImage (unknown typecode)>"));
#undef HANDLE
    return PyString_FromString("<PyCImage (unmatched type)>");
}

static PyObject *PyCImage_Str(PyCImage *pyim) {
    if (pyim->cimage && pyim->dtype) {
        auto cim = pyim->view();
        return PyString_FromString((const char *)cim->value_string()->data());
    }
    return PyString_FromString("");
}

static Py_ssize_t PyCImage_Len(PyCImage *pyim) {
    if (pyim->cimage && pyim->dtype) {
        auto cim = pyim->view();
        return (Py_ssize_t)cim->size();
    }
    return (Py_ssize_t)0;
}

static PyObject *PyCImage_GetItem(PyCImage *pyim, register Py_ssize_t idx) {
    if (pyim->cimage && pyim->dtype) {
        void *cim = (void *)pyim->view();
        int tc = (int)pyim->dtype->type_num;
        Py_ssize_t siz = (Py_ssize_t)cim->size();
        if (idx < 0 || idx >= siz) {
            PyErr_SetString(PyExc_IndexError,
                "index out of range");
            return NULL;
        }
        switch (tc) {
            case NPY_FLOAT:
            case NPY_DOUBLE:
            case NPY_LONGDOUBLE: {
                return Py_BuildValue("f", static_cast<long double>(cim->operator()(idx)));
            }
            break;
            case NPY_USHORT:
            case NPY_UBYTE:
            case NPY_UINT:
            case NPY_ULONG:
            case NPY_ULONGLONG: {
                return Py_BuildValue("k", static_cast<unsigned long>(cim->operator()(idx)));
            }
            break;
        }
        return Py_BuildValue("l", static_cast<long>(cim->operator()(idx)));
    }
    PyErr_SetString(PyExc_IndexError,
        "image index not initialized");
    return NULL;
}

static PySequenceMethods PyCImage_SequenceMethods = {
    (lenfunc)PyCImage_Len,                      /*sq_length*/
    0,                                          /*sq_concat*/
    0,                                          /*sq_repeat*/
    (ssizeargfunc)PyCImage_GetItem,             /*sq_item*/
    0,                                          /*sq_slice*/
    0,                                          /*sq_ass_item HAHAHAHA*/
    0,                                          /*sq_ass_slice HEHEHE ASS <snort> HA*/
    0                                           /*sq_contains*/
};

static PyMethodDef PyCImage_methods[] = {
    {
        "cimg_load",
            (PyCFunction)PyCImage_LoadFromFileViaCImg,
            METH_VARARGS | METH_KEYWORDS,
            "Load image data (using CImg.h load methods)"},
    {
        "buffer_info",
            (PyCFunction)PyImgC_PyBufferDict,
            METH_VARARGS | METH_KEYWORDS,
            "Get buffer info dict"},
    SENTINEL
};

static Py_ssize_t PyCImage_TypeFlags = Py_TPFLAGS_DEFAULT |
    Py_TPFLAGS_BASETYPE |
    Py_TPFLAGS_HAVE_GETCHARBUFFER;

static PyTypeObject PyCImage_Type = {
    PyObject_HEAD_INIT(NULL)
    0,                                                          /* ob_size */
    "PyImgC.PyCImage",                                          /* tp_name */
    sizeof(PyCImage),                                           /* tp_basicsize */
    0,                                                          /* tp_itemsize */
    (destructor)PyCImage_dealloc,                               /* tp_dealloc */
    0,                                                          /* tp_print */
    0,                                                          /* tp_getattr */
    0,                                                          /* tp_setattr */
    0,                                                          /* tp_compare */
    (reprfunc)PyCImage_Repr,                                    /* tp_repr */
    0,                                                          /* tp_as_number */
    &PyCImage_SequenceMethods,                                  /* tp_as_sequence */
    0,                                                          /* tp_as_mapping */
    0,                                                          /* tp_hash */
    0,                                                          /* tp_call */
    (reprfunc)PyCImage_Str,                                     /* tp_str */
    0,                                                          /* tp_getattro */
    0,                                                          /* tp_setattro */
    0,                                                          /* tp_as_buffer */
    PyCImage_TypeFlags,                                         /* tp_flags*/
    "PyImgC object wrapper for CImg instances",                 /* tp_doc */
    0,                                                          /* tp_traverse */
    0,                                                          /* tp_clear */
    0,                                                          /* tp_richcompare */
    0,                                                          /* tp_weaklistoffset */
    0,                                                          /* tp_iter */
    0,                                                          /* tp_iternext */
    PyCImage_methods,                                           /* tp_methods */
    0,                                                          /* tp_members */
    PyCImage_getset,                                            /* tp_getset */
    0,                                                          /* tp_base */
    0,                                                          /* tp_dict */
    0,                                                          /* tp_descr_get */
    0,                                                          /* tp_descr_set */
    0,                                                          /* tp_dictoffset */
    (initproc)PyCImage_init,                                    /* tp_init */
    0,                                                          /* tp_alloc */
    PyCImage_new,                                               /* tp_new */
};

#define PyCImage_Check(op) PyObject_TypeCheck(op, &PyCImage_Type)

static PyMethodDef PyImgC_methods[] = {
    {
        "buffer_info",
            (PyCFunction)PyImgC_PyBufferDict,
            METH_VARARGS | METH_KEYWORDS,
            "Get Py_buffer info dict for an object"},
    {
        "cimage_test",
            (PyCFunction)PyImgC_CImageTest,
            METH_VARARGS | METH_KEYWORDS,
            "<<<<< TEST CIMG CALLS >>>>>"},
    {
        "structcode_parse",
            (PyCFunction)PyImgC_ParseStructCode,
            METH_VARARGS,
            "Parse struct code into list of dtype-string tuples"},
    {
        "structcode_parse_one",
            (PyCFunction)PyImgC_ParseSingleStructAtom,
            METH_VARARGS,
            "Parse unary struct code into a singular dtype string"},
    {
        "structcode_to_numpy_typenum",
            (PyCFunction)PyImgC_NumpyCodeFromStructAtom,
            METH_VARARGS,
            "Parse unary struct code into a NumPy typenum"},
    SENTINEL
};

PyMODINIT_FUNC initPyImgC(void) {
    PyObject* module;
    
    PyEval_InitThreads();
    if (PyType_Ready(&PyCImage_Type) < 0) { return; }

    module = Py_InitModule3(
        "pliio.PyImgC", PyImgC_methods,
        "PyImgC buffer interface module");
    if (module == None) { return; }

    /// Bring in NumPy
    import_array();

    /// Set up PyCImage object
    Py_INCREF(&PyCImage_Type);
    PyModule_AddObject(module,
        "PyCImage",
        (PyObject *)&PyCImage_Type);
}


