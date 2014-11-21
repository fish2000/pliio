
#include <Python.h>
#include <structmember.h>

#include "PyHashTree_Constants.h"
#include "PyHashTree_PyHashTree.h"
#include "DataPoint_Type.h"
#include "PyHashTree_ObjProtocol.h"
#include "PyHashTree_SequenceProtocol.h"
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
    {
        "add",
            (PyCFunction)PyHashTree_AddPoint,
            METH_VARARGS | METH_KEYWORDS,
            "Add data point to hash tree" },
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
    &PyHashTree_SequenceMethods,                                /* tp_as_sequence */
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

static bool PyHashTree_Check(PyObject *putative) {
    return PyObject_TypeCheck(putative, &PyHashTree_Type);
}

#define PYHASHTREE_MODULE 1
#include "PyHashTree_API.h"

static PyMethodDef PyHashTree_module_functions[] = {
    { NULL, NULL }
};

PyMODINIT_FUNC inithashtree(void) {
    PyObject *module, *api;
    static void *PyHashTree_API[PyHashTree_API_pointers];
    
    /// Initialize threads
    PyEval_InitThreads();
    
    /// Initialize PyHashTree module types
    if (PyType_Ready(&PyHashTree_Type) < 0) { return; }
    if (PyType_Ready(&DataPoint_Type) < 0) { return; }
    
    /// Initialize module object
    module = Py_InitModule3(
        "pliio.hashtree", PyHashTree_module_functions,
        "PyHashTree interface module");
    if (module == NULL) { return; }
    
    /// Set up PyHashTree module C-API function table
    PyHashTree_API[PyHashTree_DF_HammingDistance_NUM] =     (void *)PyHashTree_DF_HammingDistance;
    PyHashTree_API[PyHashTree_Check_NUM] =                  (void *)PyHashTree_Check;
    PyHashTree_API[DataPoint_AddToTree_NUM] =               (void *)DataPoint_AddToTree;
    PyHashTree_API[DataPoint_FromDatum_NUM] =               (void *)DataPoint_FromDatum;
    
    /// Set up C-API module reference
    api = PyCapsule_New((void *)PyHashTree_API, "hashtree._API", NULL);
    if (api != NULL) { PyModule_AddObject(module, "_API", api); }
    
    /// Set up PyHashTree type
    Py_INCREF(&PyHashTree_Type);
    PyModule_AddObject(module,
        "PyHashTree",
        (PyObject *)&PyHashTree_Type);
    
    /// Set up DataPoint type
    Py_INCREF(&DataPoint_Type);
    PyModule_AddObject(module,
        "DataPoint",
        (PyObject *)&DataPoint_Type);
}


