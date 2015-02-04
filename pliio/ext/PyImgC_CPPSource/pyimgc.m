
#import <Cocoa/Cocoa.h>
#import <AppKit/AppKit.h>
#import <Quartz/Quartz.h>

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
#include "PyImgC_PyCImage.h"
#include "PyImgC_IMP_StructCodeParse.h"
#include "PyImgC_IMP_PyBufferDict.h"
#include "PyImgC_IMP_ObjProtocol.h"
#include "PyImgC_IMP_GetSet.h"
#include "PyImgC_IMP_SequenceProtocol.h"
#include "PyImgC_IMP_NumberProtocol.h"
#include "PyImgC_IMP_BufferProtocol.h"
#include "PyImgC_IMP_Utils.h"

using namespace cimg_library;
using namespace std;

static PyBufferProcs PyCImage_Buffer3000Methods = {
    0, /* (readbufferproc) */
    0, /* (writebufferproc) */
    0, /* (segcountproc) */
    0, /* (charbufferproc) */
    (getbufferproc)PyCImage_GetBuffer,
    (releasebufferproc)PyCImage_ReleaseBuffer,
};

static PySequenceMethods PyCImage_SequenceMethods = {
    (lenfunc)PyCImage_Len,                      /* sq_length */
    0,                                          /* sq_concat */
    0,                                          /* sq_repeat */
    (ssizeargfunc)PyCImage_GetItem,             /* sq_item */
    0,                                          /* sq_slice */
    0,                                          /* sq_ass_item HAHAHAHA */
    0,                                          /* sq_ass_slice HEHEHE ASS <snort> HA */
    0                                           /* sq_contains */
};

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
        "dct_phash",
            (getter)PyCImage_GET_dct_phash,
            None,
            "Perceptual Image DCT Hash", None },
    {
        "mh_phash",
            (getter)PyCImage_GET_mh_phash,
            None,
            "Perceptual Image Mexican-Hat Hash", None },
    SENTINEL
};

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

static PyMethodDef PyCImage_methods[] = {
    {
        "load",
            (PyCFunction)PyCImage_LoadFromFileViaCImg,
            METH_VARARGS | METH_KEYWORDS,
            "Load image data from file" },
    {
        "save",
            (PyCFunction)PyCImage_SaveToFileViaCImg,
            METH_VARARGS | METH_KEYWORDS,
            "Save image data to file" },
    {
        "buffer_info",
            (PyCFunction)PyCImage_PyBufferDict,
            METH_VARARGS | METH_KEYWORDS,
            "Get buffer info dict" },
    SENTINEL
};

static Py_ssize_t PyCImage_TypeFlags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE | Py_TPFLAGS_HAVE_INPLACEOPS | Py_TPFLAGS_HAVE_NEWBUFFER;

static PyTypeObject PyCImage_Type = {
    PyObject_HEAD_INIT(NULL)
    0,                                                          /* ob_size */
    "imgc.PyCImage",                                            /* tp_name */
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
    (getattrofunc)PyObject_GenericGetAttr,                      /* tp_getattro */
    (setattrofunc)PyObject_GenericSetAttr,                      /* tp_setattro */
    &PyCImage_Buffer3000Methods,                                /* tp_as_buffer */
    PyCImage_TypeFlags,                                         /* tp_flags */
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

static bool PyCImage_Check(PyObject *putative) {
    return PyObject_TypeCheck(putative, &PyCImage_Type);
}

static PyObject *PyCImage_DCTMatrix(PyObject *m, PyObject *args, PyObject *kwargs) {
    int N = 32;
    static char *kwlist[] = { "N", NULL };

    if (!PyArg_ParseTupleAndKeywords(args, kwargs,
                "|i", kwlist, &N)) {
            PyErr_SetString(PyExc_ValueError,
                "bad arguments to PyCImage_DCTMatrix()");
        return NULL;
    }
    
    PyCImage *self = reinterpret_cast<PyCImage *>(PyCImage_Type.tp_alloc(&PyCImage_Type, 0));
    if (self == NULL) {
        PyErr_SetString(PyExc_ValueError,
            "Couldn't allocate new PyCImage");
        return NULL;
    }
    self->assign<float>(*ph_dct_matrix(N));
    return reinterpret_cast<PyObject *>(self); /// all is well, return self
}

static PyMethodDef PyImgC_module_functions[] = {
    {
        "buffer_info",
            (PyCFunction)PyImgC_PyBufferDict,
            METH_VARARGS | METH_KEYWORDS,
            "Get Py_buffer info dict for an object" },
    {
        "temporary_path",
            (PyCFunction)PyImgC_TemporaryPath,
            METH_NOARGS,
            "Generate a path for a temporary file" },
    {
        "guess_type",
            (PyCFunction)PyImgC_GuessType,
            METH_VARARGS,
            "Guess the file type based on header data" },
    {
        "structcode_parse",
            (PyCFunction)PyImgC_ParseStructCode,
            METH_VARARGS,
            "Parse struct code into list of dtype-string tuples" },
    {
        "dct_matrix",
            (PyCFunction)PyCImage_DCTMatrix,
            METH_VARARGS | METH_KEYWORDS,
            "Get a DCT matrix of size N (default N = 32)" },
    SENTINEL
};

PyMODINIT_FUNC initimgc(void) {
    PyObject *module;
    //PyObject *pycmx;
    
    PyEval_InitThreads();
    if (PyType_Ready(&PyCImage_Type) < 0) { return; }

    module = Py_InitModule3(
        "pliio.imgc", PyImgC_module_functions,
        "PyImgC buffer interface module");
    if (module == None) { return; }
    
    /// Set up cleanup handler on interpreter exit
    //Py_AtExit(PyImgC_AtExit);
    
    /// Set up color manager capsule
    // pycmx = PyImgC_CMS_Startup(NULL);
    // if (pycmx != NULL) {
    //     Py_INCREF(pycmx);
    //     PyModule_AddObject(module, "_pycmx", pycmx);
    // }
    
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
    
    PyModule_AddObject(module, "complex64", numpy::dtype_object<complex<float>>());
    PyModule_AddObject(module, "complex_", numpy::dtype_object<complex<double>>());
    PyModule_AddObject(module, "complex128", numpy::dtype_object<complex<double>>());
    PyModule_AddObject(module, "complex256", numpy::dtype_object<complex<long double>>());
}


