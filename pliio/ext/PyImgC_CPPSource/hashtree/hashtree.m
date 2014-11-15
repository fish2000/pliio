

#include <Python.h>
#include <structmember.h>

#include "PyHashTree_Constants.h"
#include "PyHashTree_PyHashTree.h"
#include "PyHashTree_ObjProtocol.h"
#include "PyHashTree_PrintFunctions.h"
#include "PyHashTree_GetSet.h"

static PyMethodDef PyHashTree_methods[] = {
    {
        "load",
            (PyCFunction)PyHashTree_LoadFromMVPFile,
            METH_VARARGS | METH_KEYWORDS,
            "Load hash tree from MVP file" },
    {
        "save",
            (PyCFunction)PyHashTree_SaveToMVPFile,
            METH_VARARGS | METH_KEYWORDS,
            "Save hash tree to MVP file" },
    SENTINEL
};

static Py_ssize_t PyHashTree_TypeFlags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE;

static PyTypeObject PyHashTree_Type = {
    PyObject_HEAD_INIT(NULL)
    0,                                                          /* ob_size */
    "hashtree.PyHashTree",                                      /* tp_name */
    sizeof(PyHashTree),                                         /* tp_basicsize */
    0,                                                          /* tp_itemsize */
    (destructor)PyHashTree_dealloc,                             /* tp_dealloc */
    (printfunc)PyHashTree_Print,                                /* tp_print */
    0,                                                          /* tp_getattr */
    0,                                                          /* tp_setattr */
    0,                                                          /* tp_compare */
    (reprfunc)PyHashTree_Repr,                                  /* tp_repr */
    0,                                                          /* tp_as_number */
    0,                                                          /* tp_as_sequence */
    0,                                                          /* tp_as_mapping */
    0,                                                          /* tp_hash */
    0,                                                          /* tp_call */
    (reprfunc)PyHashTree_Str,                                   /* tp_str */
    (getattrofunc)PyObject_GenericGetAttr,                      /* tp_getattro */
    (setattrofunc)PyObject_GenericSetAttr,                      /* tp_setattro */
    0,                                                          /* tp_as_buffer */
    PyHashTree_TypeFlags,                                       /* tp_flags */
    "Python bindings for MVP hash trees",                       /* tp_doc */
    0,                                                          /* tp_traverse */
    0,                                                          /* tp_clear */
    0,                                                          /* tp_richcompare */
    0,                                                          /* tp_weaklistoffset */
    0,                                                          /* tp_iter */
    0,                                                          /* tp_iternext */
    PyHashTree_methods,                                         /* tp_methods */
    0,                                                          /* tp_members */
    PyHashTree_getset,                                          /* tp_getset */
    0,                                                          /* tp_base */
    0,                                                          /* tp_dict */
    0,                                                          /* tp_descr_get */
    0,                                                          /* tp_descr_set */
    0,                                                          /* tp_dictoffset */
    (initproc)PyHashTree_init,                                  /* tp_init */
    0,                                                          /* tp_alloc */
    PyHashTree_new,                                             /* tp_new */
};

static PyMethodDef PyHashTree_module_functions[] = {
    SENTINEL
};

PyMODINIT_FUNC inithashtree(void) {
    PyObject *module;
    
    PyEval_InitThreads();
    if (PyType_Ready(&PyHashTree_Type) < 0) { return; }

    module = Py_InitModule3(
        "pliio.hashtree", PyHashTree_module_functions,
        "PyHashTree interface module");
    if (module == None) { return; }

    /// Set up PyCImage object
    Py_INCREF(&PyHashTree_Type);
    PyModule_AddObject(module,
        "PyHashTree",
        (PyObject *)&PyHashTree_Type);
}


