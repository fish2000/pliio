
#include <iostream>
#include <vector>
#include <string>
#include <typeinfo>

#include <Python.h>
#include <structmember.h>
#include <numpy/ndarrayobject.h>
#include "numpypp/numpy.hpp"
#include "numpypp/structcode.hpp"
#include "numpypp/typecode.hpp"
#include "numpypp/dispatch.hpp"
#include "numpypp/utils.hpp"

#include "PyImgC_Constants.h"
#include "PyImgC_SharedDefs.h"
#include "PyImgC_PyCImage.h"
#include "PyImgC_IMP_Operators.h"
#include "PyImgC_IMP_StructCodeParse.h"
#include "PyImgC_IMP_PyBufferDict.h"
#include "PyImgC_IMP_CImageTest.h"
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
        Py_XDECREF(dtype);
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
    if (!pyim->cimage) { PyString_FromString("<PyCImage (empty backing stores)>"); }
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
    SAFE_SWITCH_ON_TYPECODE(tc, PyString_FromString("<PyCImage (unknown typecode)>"));
#undef HANDLE
    return PyString_FromString("<PyCImage (unmatched type)>");
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
            PyString_FromString("cimg_load"),
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

/// pycimage.dtype getter/setter
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
            "Data Type (numpy.dtype)", None },
    SENTINEL
};

/// __len__ implementation
static Py_ssize_t PyCImage_Len(PyCImage *pyim) {
    if (pyim->cimage && pyim->dtype) {
#define HANDLE(type) { \
        auto cim = pyim->recast<type>(); \
        return static_cast<Py_ssize_t>(cim->size()); \
    }
    SAFE_SWITCH_ON_DTYPE(pyim->dtype, -1);
#undef HANDLE
    }
    return static_cast<Py_ssize_t>(0);
}

/// __getitem__ implementation
static PyObject *PyCImage_GetItem(PyCImage *pyim, register Py_ssize_t idx) {
    if (pyim->cimage && pyim->dtype) {
        int tc = static_cast<int>(pyim->dtype->type_num);
        long op;
        Py_ssize_t siz;
#define HANDLE(type) { \
        auto cim = pyim->recast<type>(); \
        op = static_cast<long>(cim->operator()(idx)); \
        siz = (Py_ssize_t)cim->size(); \
    }
    SAFE_SWITCH_ON_TYPECODE(tc, NULL);
#undef HANDLE
        if (idx < 0 || idx >= siz) {
            PyErr_SetString(PyExc_IndexError,
                "index out of range");
            return NULL;
        }
        switch (tc) {
            case NPY_FLOAT:
            case NPY_DOUBLE:
            case NPY_LONGDOUBLE: {
                return Py_BuildValue("f", static_cast<long double>(op));
            }
            break;
            case NPY_USHORT:
            case NPY_UBYTE:
            case NPY_UINT:
            case NPY_ULONG:
            case NPY_ULONGLONG: {
                return Py_BuildValue("k", static_cast<unsigned long>(op));
            }
            break;
        }
        return Py_BuildValue("l", op);
    }
    PyErr_SetString(PyExc_IndexError,
        "image index not initialized");
    return NULL;
}

static int PyCImage_Compare(PyObject *smelf, PyObject *smother) {
    if (!smelf || !smother) {
        PyErr_SetString(PyExc_ValueError,
            "Bad comparison arguments");
        return -1;
    }
    
    PyCImage *self = reinterpret_cast<PyCImage *>(smelf);
    PyCImage *other = reinterpret_cast<PyCImage *>(smother);
    auto result = self->compare(other);
    
    switch (result) {
        case -2: {
            PyErr_SetString(PyExc_ValueError,
                "Comparator object typecode mismatch (-2)");
            return -1;
        }
        case -3: {
            PyErr_SetString(PyExc_ValueError,
                "Comparison object typecode mismatch (-3)");
            return -1;
        }
    }
    return result;
}


static int PyCImage_NonZero(PyObject *smelf) {
    if (!smelf) {
        PyErr_SetString(PyExc_ValueError,
            "Bad nonzero argument");
        return -1;
    }
    
    PyCImage *self = reinterpret_cast<PyCImage *>(smelf);
    return self->is_empty() ? 0 : 1; /// NON-zero, so this is backwards-looking
}

/// BINARY OP MACROS -- Here's how these work:
/// PyCImage_BINARY_OP(OP_NAME) is a macro.
/// Invoking this macro will declare an in-place function,
/// named after its argument, e.g.:
///     PyCImage_BINARY_OP(ADD)
/// ... will wind up declaring something named:
///     PyCImage_ADD(PyCObject *self, PyCObject *other);
/// ... This newly-declared function is, itself,
/// a relatively simplistic wrapper around another macro, which
/// uses the OP_NAME from PyCImage_BINARY_OP(OP_NAME) to reference
/// an opcode in the BinaryOp enum:
///     SAFE_SWITCH_ON_DTYPE_FOR_BINARY_OP(
///         dtype, error_return_value, ADD, BinaryOp::ADD)
/// ... which goes through the Rube Goldberg-esque type-switch when invoked,
/// as defined in numpypp/dispatch.hpp -- eventually handing off
/// to a handler macro:
///     HANDLE_BINARY_OP(T, ADD, BinaryOp::ADD)
/// ... which FINALLY has enough type information to call the real function:
///     binary_op<T>(self, other, BinaryOp::ADD);
/// ... but wait, THERE IS MORE: binary_op<T>() has to call binary_op_LHS<T>()
/// (it's another simple wrapper) which then has to do one last type-switch
/// to obtain the RHS type before it is able to call binary_op_RHS<otherT>().
/// This then invokes the final step in the process: a call to the BINARY_OP() macro,
/// which executes a switch on the actual opcode (in this case, BinaryOp::ADD)
/// and does the actual fucking operation math. In a nutshell.

PyCImage_BINARY_OP(ADD);
PyCImage_BINARY_OP(SUBTRACT);
PyCImage_BINARY_OP(MULTIPLY);
PyCImage_BINARY_OP(DIVIDE);
PyCImage_BINARY_OP(REMAINDER);
/*PyCImage_BINARY_OP(DIVMOD);*/
PyCImage_BINARY_OP(POWER);
PyCImage_BINARY_OP(LSHIFT);
PyCImage_BINARY_OP(RSHIFT);
PyCImage_BINARY_OP(AND);
PyCImage_BINARY_OP(XOR);
PyCImage_BINARY_OP(OR);

PyCImage_BINARY_OP(INPLACE_ADD);
PyCImage_BINARY_OP(INPLACE_SUBTRACT);
PyCImage_BINARY_OP(INPLACE_MULTIPLY);
PyCImage_BINARY_OP(INPLACE_DIVIDE);
PyCImage_BINARY_OP(INPLACE_REMAINDER);
PyCImage_BINARY_OP(INPLACE_POWER);
PyCImage_BINARY_OP(INPLACE_LSHIFT);
PyCImage_BINARY_OP(INPLACE_RSHIFT);
PyCImage_BINARY_OP(INPLACE_AND);
PyCImage_BINARY_OP(INPLACE_XOR);
PyCImage_BINARY_OP(INPLACE_OR);

PyCImage_BINARY_OP(FLOOR_DIVIDE);
PyCImage_BINARY_OP(TRUE_DIVIDE);
PyCImage_BINARY_OP(INPLACE_FLOOR_DIVIDE);
PyCImage_BINARY_OP(INPLACE_TRUE_DIVIDE);

/// Unary Op Macros
PyCImage_UNARY_OP(NEGATIVE);
PyCImage_UNARY_OP(POSITIVE);
PyCImage_UNARY_OP(INVERT);
PyCImage_UNARY_OP(ABSOLUTE);
PyCImage_UNARY_OP(INT);
PyCImage_UNARY_OP(LONG);
PyCImage_UNARY_OP(FLOAT);
PyCImage_UNARY_OP(INDEX);

static PyNumberMethods PyCImage_NumberMethods = {
    (binaryfunc)PyCImage_ADD,                   /* nb_add */
    (binaryfunc)PyCImage_SUBTRACT,              /* nb_subtract */
    (binaryfunc)PyCImage_MULTIPLY,              /* nb_multiply */
    (binaryfunc)PyCImage_DIVIDE,                /* nb_divide */
    (binaryfunc)PyCImage_REMAINDER,             /* nb_remainder */
    0,                                          /* nb_divmod */
    0, /*(ternaryfunc)PyCImage_POWER,*/         /* nb_power */
    (unaryfunc)PyCImage_NEGATIVE,               /* nb_negative */
    (unaryfunc)PyCImage_POSITIVE,               /* nb_positive */
    (unaryfunc)PyCImage_ABSOLUTE,               /* nb_absolute */
    
    (inquiry)PyCImage_NonZero,                  /* nb_nonzero */
    
    (unaryfunc)PyCImage_INVERT,                 /* nb_invert */
    (binaryfunc)PyCImage_LSHIFT,                /* nb_lshift */
    (binaryfunc)PyCImage_RSHIFT,                /* nb_rshift */
    (binaryfunc)PyCImage_AND,                   /* nb_and */
    (binaryfunc)PyCImage_XOR,                   /* nb_xor */
    (binaryfunc)PyCImage_OR,                    /* nb_or */
    
    0, /*(coercion)PyCImage_COERCE,*/           /* nb_coerce */
    
    (unaryfunc)PyCImage_INT,                    /* nb_int */
    (unaryfunc)PyCImage_LONG,                   /* nb_long */
    (unaryfunc)PyCImage_FLOAT,                  /* nb_float */
    0,                                          /* nb_oct */
    0,                                          /* nb_hex */
    
    (binaryfunc)PyCImage_INPLACE_ADD,           /* nb_inplace_add */
    (binaryfunc)PyCImage_INPLACE_SUBTRACT,      /* nb_inplace_subtract */
    (binaryfunc)PyCImage_INPLACE_MULTIPLY,      /* nb_inplace_multiply */
    (binaryfunc)PyCImage_INPLACE_DIVIDE,        /* nb_inplace_divide */
    (binaryfunc)PyCImage_INPLACE_REMAINDER,     /* nb_inplace_remainder */
    0, /*(ternaryfunc)PyCImage_INPLACE_POWER,*/ /* nb_inplace_power */
    (binaryfunc)PyCImage_INPLACE_LSHIFT,        /* nb_inplace_lshift */
    (binaryfunc)PyCImage_INPLACE_RSHIFT,        /* nb_inplace_rshift */
    (binaryfunc)PyCImage_INPLACE_AND,           /* nb_inplace_and */
    (binaryfunc)PyCImage_INPLACE_XOR,           /* nb_inplace_xor */
    (binaryfunc)PyCImage_INPLACE_OR,            /* nb_inplace_or */
    
    (binaryfunc)PyCImage_FLOOR_DIVIDE,          /* nb_floor_divide */
    (binaryfunc)PyCImage_TRUE_DIVIDE,           /* nb_true_divide */
    (binaryfunc)PyCImage_INPLACE_FLOOR_DIVIDE,  /* nb_inplace_floor_divide */
    (binaryfunc)PyCImage_INPLACE_TRUE_DIVIDE,   /* nb_inplace_true_divide */

    0,                                          /* nb_index */
};


static PySequenceMethods PyCImage_SequenceMethods = {
    (lenfunc)PyCImage_Len,                      /* sq_length */
    0,                                          /* sq_concat */
    0,                                          /* sq_repeat */
    (ssizeargfunc)PyCImage_GetItem,             /* sq_item */
    0,                                          /* sq_slice */
    0,                                          /* sq_ass_item HAHAHAHA */
    0,                                          /* sq_ass_slice HEHEHE ASS <snort> HA */
    0                                           /* sq_contains*/
};

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
    }
    return 0;
}

static void PyCImage_ReleaseBuffer(PyObject *self, Py_buffer *view) {
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

static PyMethodDef PyCImage_methods[] = {
    {
        "cimg_load",
            (PyCFunction)PyCImage_LoadFromFileViaCImg,
            METH_VARARGS | METH_KEYWORDS,
            "Load image data (using CImg.h load methods)"},
    {
        "buffer_info",
            (PyCFunction)PyCImage_PyBufferDict,
            METH_VARARGS | METH_KEYWORDS,
            "Get buffer info dict"},
    SENTINEL
};

// static Py_ssize_t PyCImage_TypeFlags_Buffer = Py_TPFLAGS_DEFAULT |
//     Py_TPFLAGS_BASETYPE |
//     Py_TPFLAGS_HAVE_GETCHARBUFFER |
//     Py_TPFLAGS_HAVE_NEWBUFFER |
//     Py_TPFLAGS_HAVE_INPLACEOPS;

static Py_ssize_t PyCImage_TypeFlags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE | Py_TPFLAGS_HAVE_INPLACEOPS | Py_TPFLAGS_HAVE_NEWBUFFER;

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
    (cmpfunc)PyCImage_Compare,                                  /* tp_compare */
    (reprfunc)PyCImage_Repr,                                    /* tp_repr */
    &PyCImage_NumberMethods,                                    /* tp_as_number */
    &PyCImage_SequenceMethods,                                  /* tp_as_sequence */
    0,                                                          /* tp_as_mapping */
    0,                                                          /* tp_hash */
    0,                                                          /* tp_call */
    (reprfunc)PyCImage_Str,                                     /* tp_str */
    0,                                                          /* tp_getattro */
    0,                                                          /* tp_setattro */
    &PyCImage_Buffer3000Methods,                                /* tp_as_buffer */
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
    PyObject *module;
    
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
    
    /// Add dtype object references
    PyModule_AddObject(module, "bool_", numpy::dtype_object<bool>());
    PyModule_AddObject(module, "character", numpy::dtype_object<char>());
    
    PyModule_AddObject(module, "int8", numpy::dtype_object<char>());
    PyModule_AddObject(module, "short", numpy::dtype_object<short>());
    PyModule_AddObject(module, "int16", numpy::dtype_object<short>());
    PyModule_AddObject(module, "int32", numpy::dtype_object<int>());
    PyModule_AddObject(module, "long", numpy::dtype_object<long>());
    PyModule_AddObject(module, "int64", numpy::dtype_object<long>());
    PyModule_AddObject(module, "int_", numpy::dtype_object<long>());
    
    PyModule_AddObject(module, "uint8", numpy::dtype_object<unsigned char>());
    PyModule_AddObject(module, "ushort", numpy::dtype_object<unsigned short>());
    PyModule_AddObject(module, "uint16", numpy::dtype_object<unsigned short>());
    PyModule_AddObject(module, "uint32", numpy::dtype_object<unsigned int>());
    PyModule_AddObject(module, "uint", numpy::dtype_object<unsigned long>());
    PyModule_AddObject(module, "ulonglong", numpy::dtype_object<unsigned long>());
    PyModule_AddObject(module, "uint64", numpy::dtype_object<unsigned long>());
    
    PyModule_AddObject(module, "float", numpy::dtype_object<float>());
    PyModule_AddObject(module, "float32", numpy::dtype_object<float>());
    PyModule_AddObject(module, "double", numpy::dtype_object<double>());
    PyModule_AddObject(module, "float64", numpy::dtype_object<double>());
    PyModule_AddObject(module, "longdouble", numpy::dtype_object<long double>());
    PyModule_AddObject(module, "longfloat", numpy::dtype_object<long double>());
    
}


