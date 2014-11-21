
#ifndef PyHashTree_TYPESTRUCT_DataPoint_H
#define PyHashTree_TYPESTRUCT_DataPoint_H

#include <Python.h>
#include <structmember.h>
#include "mvptree/mvptree.h"
#include "PyHashTree_GIL.h"

#include <string>
#include <utility>
#include <vector>
#include <map>
using namespace std;

struct datatype {
    
    static map<MVPDataType, string> init_name_map() {
        map<MVPDataType, string> _name_map = {
            { MVP_NOTHING,      "NOTHING" },        /// 0
            { MVP_BYTEARRAY,    "BYTEARRAY" },      /// 1
            { MVP_UINT16ARRAY,  "UINT16ARRAY" },    /// 2
            { MVP_UINT32ARRAY,  "UINT32ARRAY" },    /// 4
            { MVP_UINT64ARRAY,  "UINT64ARRAY" },    /// 8
        };
        return _name_map;
    }
    
    static const map<MVPDataType, string> names;
    
};

const map<MVPDataType, string> datatype::names = datatype::init_name_map();

struct DataPoint {
    PyObject_HEAD
    PyObject *tree;
    MVPDP *dp;
    
    uint64_t data() {
        /// '0ULL' is not a typo, it's an unsigned long long literal
        if (!dp) { return 0ULL; }
        if (!dp->data) { return 0ULL; }
        return *static_cast<uint64_t *>(dp->data);
    }
    
    MVPDataType datatype() {
        if (!dp) { return MVP_NOTHING; }
        if (!dp->type) { return MVP_NOTHING; }
        return dp->type;
    }
    
    const char *datatypestring() {
        if (!dp) { return datatype::names.at(MVP_NOTHING).c_str(); }
        if (!dp->type) { return datatype::names.at(MVP_NOTHING).c_str(); }
        return datatype::names.at(dp->type).c_str();
    }
    
    const char *name() {
        if (!dp) { return "NULL"; }
        if (!dp->id && dp->data) { 
            return to_string(
                *static_cast<uint64_t *>(dp->data)).c_str();
        }
        return (const char *)dp->id;
    }
    
    void cleanup() {}
    
    ~DataPoint() {
        if (dp) { cleanup(); }
        Py_XDECREF(tree);
    }
};

/// ALLOCATE / __new__ implementation
static PyObject *DataPoint_new(PyTypeObject *type, PyObject *args, PyObject *kwargs) {
    DataPoint *self;
    self = reinterpret_cast<DataPoint *>(type->tp_alloc(type, 0));
    /// initialize with defaults
    if (self != NULL) {
        self->tree = NULL;
        self->dp = NULL;
    }
    return reinterpret_cast<PyObject *>(self); /// all is well, return self
}

static int DataPoint_AddToTree(DataPoint *self, PyHashTree *tree) {
    if (!tree->tree) {
        /// I DONT KNOW WHAT THE FUCK MAN
        PyErr_SetString(PyExc_ValueError,
            "Can't add a datapoint to an uninitialized tree");
        return -1;
    }
    if (!self->dp) {
        /// I DONT KNOW WHAT THE FUCK MAN
        PyErr_SetString(PyExc_ValueError,
            "Can't add an uninitialized datapoint to a tree");
        return -1;
    }
    
    //gil_release NOGIL;
    MVPDP **points = (MVPDP**)PyMem_Malloc(sizeof(MVPDP*));
    points[0] = self->dp;
    MVPError error = mvptree_add(tree->tree, points, 1);
    //NOGIL.~gil_release();
    
    if (error != MVP_SUCCESS) {
        PyErr_Format(PyExc_SystemError,
            "Adding point to tree raised an MVP error: %s",
            mvp_errstr(error));
        return -1;
    }
    return 0;
}

/// __init__ implementation
static int DataPoint_init(DataPoint *self, PyObject *args, PyObject *kwargs) {
    unsigned int datalen = 1;
    unsigned long long data;
    MVPDataType datatype = MVP_UINT64ARRAY;
    PyObject *tree = NULL;
    char *name = NULL;
    MVPDP *dp;
    
    static char *keywords[] = { "data", "datatype", "tree", "name", None };
    
    if (!PyArg_ParseTupleAndKeywords(
        args, kwargs, "K|IOs:dp_alloc",
        keywords,
        &data, &datatype, &tree, &name)) {
            PyErr_SetString(PyExc_ValueError,
                "bad arguments to DataPoint_init");
            return -1;
    }
    
    if (!name) { name = const_cast<char *>(to_string(data).c_str()); }
    if (!datatype) { datatype = MVP_UINT64ARRAY; }
    
    dp = dp_alloc(datatype);
    dp->id = name;
    dp->data = PyMem_Malloc(datalen*datatype);
    dp->datalen = datalen;
    memcpy(dp->data, &data, datalen*datatype);
    
    Py_INCREF(tree);
    self->tree = tree;
    self->dp = dp;
    
    if (self->tree && self->dp) {
        return DataPoint_AddToTree(self, reinterpret_cast<PyHashTree *>(self->tree));
    }
    return 0;
}

/// __repr__ implementations
static PyObject *DataPoint_Repr(DataPoint *dp) {
    if (!dp->dp) {
        return PyString_FromFormat("<DataPoint[NULL] @ %p>", dp);
    }
    if (!dp->tree) {
        return PyString_FromFormat("<DataPoint[%s] (%llu) @ %p>",
            dp->datatypestring(), dp->data(), dp);
    }
    return PyString_FromFormat("<DataPoint[%s] (%llu)->(<tree[%u]>) @ %p>",
        dp->datatypestring(), dp->data(),
        reinterpret_cast<PyHashTree *>(dp->tree)->length(), dp);
}

static const char *DataPoint_ReprCString(DataPoint *dp) {
    PyObject *out = DataPoint_Repr(dp);
    const char *outstr = PyString_AS_STRING(out);
    Py_DECREF(out);
    return outstr;
}

static string DataPoint_ReprString(DataPoint *dp) {
    return string(DataPoint_ReprCString(dp));
}

static long DataPoint_Hash(DataPoint *dp) {
    return static_cast<long>(dp->data());
}

/// DEALLOCATE
static void DataPoint_dealloc(DataPoint *self) {
    self->cleanup();
    self->ob_type->tp_free((PyObject *)self);
}

/// datapoint.data getter
static PyObject     *DataPoint_GET_data(DataPoint *self, void *closure) {
    return Py_BuildValue("K", self->data());
}

/// datapoint.datatype getter
static PyObject     *DataPoint_GET_datatype(DataPoint *self, void *closure) {
    return Py_BuildValue("s", self->datatypestring());
}

/// datapoint.tree reference getter
static PyObject     *DataPoint_GET_tree(DataPoint *self, void *closure) {
    if (!self->tree) {
        return Py_BuildValue("");
    }
    return Py_BuildValue("O", self->tree);
}

static PyGetSetDef DataPoint_getset[] = {
    {
        "data",
            (getter)DataPoint_GET_data,
            NULL,
            "Datapoint data value", NULL },
    {
        "datatype",
            (getter)DataPoint_GET_datatype,
            NULL,
            "Datapoint data type string", NULL },
    {
        "tree",
            (getter)DataPoint_GET_tree,
            NULL,
            "Reference to parent PyHashTree object", NULL },
    SENTINEL
};

static PyObject *DataPoint_FromDatum(DataPoint *self, MVPDP *datum) {
    DataPoint *instance;
    if (!datum) {
        PyErr_SetString(PyExc_ValueError,
            "DataPoint_FromDatum() received a NULL datum");
        return NULL;
    }
    
    instance = reinterpret_cast<DataPoint *>(
        self->ob_type->tp_alloc(self->ob_type, 0));
    
    if (!instance) {
        PyErr_SetString(PyExc_ValueError,
            "DataPoint allocated to NULL value");
        return NULL;
    }
    
    Py_INCREF(self->tree);
    instance->tree = self->tree;
    instance->dp = datum;
    return reinterpret_cast<PyObject *>(instance);
}

static PyObject *DataPoint_Nearest(PyObject *smelf, PyObject *args, PyObject *kwargs) {
    DataPoint *self = reinterpret_cast<DataPoint *>(smelf);
    PyHashTree *tree;
    PyObject *out;
    MVPDP **results;
    MVPError error;
    unsigned int nbresults;
    unsigned int nearest = 5;
    float radius = 21.0f;
    static char *keywords[] = { "nearest", "radius", None };

    if (!PyArg_ParseTupleAndKeywords(
        args, kwargs, "|If",
        keywords,
        &nearest, &radius)) {
            PyErr_SetString(PyExc_ValueError,
                "cannot get nearest values (bad arguments)");
            return NULL;
    }
    
    if (!self->tree) {
        PyErr_SetString(PyExc_ValueError,
            "DataPoint does not belong to an active tree");
        return NULL;
    }
    
    Py_INCREF(self->tree);
    tree = reinterpret_cast<PyHashTree *>(self->tree);
    
    if (!tree->tree) {
        PyErr_SetString(PyExc_ValueError,
            "DataPoint belongs to an uninitialized PyHashTree");
        Py_XDECREF(self->tree);
        return NULL;
    }
    
    //gil_release NOGIL;
    results = mvptree_retrieve(tree->tree, self->dp, nearest, radius, &nbresults, &error);
    //NOGIL.~gil_release();
    
    Py_DECREF(self->tree);
    tree = NULL;
    
    if (!results) {
        PyErr_SetString(PyExc_ValueError,
            "No results retrieved!");
        return NULL;
    }
    if (error != MVP_SUCCESS) {
        PyErr_Format(PyExc_SystemError,
            "MVP error when getting results: %s",
            mvp_errstr(error));
        PyMem_Free(results);
        return NULL;
    }
    
    out = PyTuple_New(static_cast<Py_ssize_t>(nbresults));
    for (Py_ssize_t idx = 0; idx < nbresults; idx++) {
        PyObject *result = DataPoint_FromDatum(self, results[idx]);
        if (!result) {
            PyErr_SetString(PyExc_ValueError,
                "DataPoint_FromDatum() returned NULL");
            PyMem_Free(results);
            return NULL;
        }
        PyTuple_SetItem(out, idx, result);
    }
    
    PyMem_Free(results);
    return out;
}
    
static PyMethodDef DataPoint_methods[] = {
    {
        "nearest",
            (PyCFunction)DataPoint_Nearest,
            METH_VARARGS | METH_KEYWORDS,
            "Get a tuple of nearby DataPoints" },
    SENTINEL
};

static Py_ssize_t DataPoint_TypeFlags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE;

static PyTypeObject DataPoint_Type = {
    PyObject_HEAD_INIT(NULL)
    0,                                                          /* ob_size */
    "hashtree.DataPoint",                                       /* tp_name */
    sizeof(DataPoint),                                          /* tp_basicsize */
    0,                                                          /* tp_itemsize */
    (destructor)DataPoint_dealloc,                              /* tp_dealloc */
    0,                                                          /* tp_print */
    0,                                                          /* tp_getattr */
    0,                                                          /* tp_setattr */
    0,                                                          /* tp_compare */
    (reprfunc)DataPoint_Repr,                                   /* tp_repr */
    0,                                                          /* tp_as_number */
    0,                                                          /* tp_as_sequence */
    0,                                                          /* tp_as_mapping */
    0, /*(hashfunc)DataPoint_Hash,*/                            /* tp_hash */
    0,                                                          /* tp_call */
    0,                                                          /* tp_str */
    (getattrofunc)PyObject_GenericGetAttr,                      /* tp_getattro */
    (setattrofunc)PyObject_GenericSetAttr,                      /* tp_setattro */
    0,                                                          /* tp_as_buffer */
    DataPoint_TypeFlags,                                        /* tp_flags */
    "Python bindings for MVP hash trees",                       /* tp_doc */
    0,                                                          /* tp_traverse */
    0,                                                          /* tp_clear */
    0,                                                          /* tp_richcompare */
    0,                                                          /* tp_weaklistoffset */
    0,                                                          /* tp_iter */
    0,                                                          /* tp_iternext */
    DataPoint_methods,                                          /* tp_methods */
    0,                                                          /* tp_members */
    DataPoint_getset,                                           /* tp_getset */
    0,                                                          /* tp_base */
    0,                                                          /* tp_dict */
    0,                                                          /* tp_descr_get */
    0,                                                          /* tp_descr_set */
    0,                                                          /* tp_dictoffset */
    (initproc)DataPoint_init,                                   /* tp_init */
    0,                                                          /* tp_alloc */
    DataPoint_new,                                              /* tp_new */
};

#endif /// PyHashTree_TYPESTRUCT_DataPoint_H