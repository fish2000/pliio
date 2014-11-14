
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
#include "PyImgC_SharedDefs.h"
#include "PyImgC_PyCImage.h"
#include "PyImgC_IMP_StructCodeParse.h"
#include "PyImgC_IMP_PyBufferDict.h"
#include "PyImgC_IMP_CImageTest.h"
#include "PyImgC_IMP_ObjProtocol.h"
#include "PyImgC_IMP_GetSet.h"
#include "PyImgC_IMP_Imaging.h"
#include "PyImgC_IMP_SequenceProtocol.h"
#include "PyImgC_IMP_NumberProtocol.h"
#include "PyImgC_IMP_BufferProtocol.h"
#include "PyImgC_IMP_LittleCMSContext.h"
#include "PyImgC_IMP_Utils.h"

using namespace cimg_library;
using namespace std;

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
        "cimage_test",
            (PyCFunction)PyImgC_CImageTest,
            METH_VARARGS | METH_KEYWORDS,
            "<<<<< TEST CIMG CALLS >>>>>" },
    {
        "structcode_parse",
            (PyCFunction)PyImgC_ParseStructCode,
            METH_VARARGS,
            "Parse struct code into list of dtype-string tuples" },
    {
        "structcode_parse_one",
            (PyCFunction)PyImgC_ParseSingleStructAtom,
            METH_VARARGS,
            "Parse unary struct code into a singular dtype string" },
    {
        "structcode_to_numpy_typenum",
            (PyCFunction)PyImgC_NumpyCodeFromStructAtom,
            METH_VARARGS,
            "Parse unary struct code into a NumPy typenum" },
    SENTINEL
};

PyMODINIT_FUNC initPyImgC(void) {
    PyObject *module;
    PyObject *pycmx;
    
    PyEval_InitThreads();
    if (PyType_Ready(&PyCImage_Type) < 0) { return; }

    module = Py_InitModule3(
        "pliio.PyImgC", PyImgC_module_functions,
        "PyImgC buffer interface module");
    if (module == None) { return; }
    
    /// Set up cleanup handler on interpreter exit
    Py_AtExit(PyImgC_AtExit);
    
    /// Set up color manager capsule
    pycmx = PyImgC_CMS_Startup(NULL);
    if (pycmx != NULL) {
        Py_INCREF(pycmx);
        PyModule_AddObject(module, "_pycmx", pycmx);
    }
    
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


