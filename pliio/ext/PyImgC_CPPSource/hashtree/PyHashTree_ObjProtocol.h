
#ifndef PyHashTree_PYHASHTREE_IMP_OBJPROTOCOL_H
#define PyHashTree_PYHASHTREE_IMP_OBJPROTOCOL_H

#include <Python.h>
#include "mvptree/mvptree.h"
#include "PyHashTree_Constants.h"
#include "PyHashTree_DistanceFunctions.h"

#define DEFAULT_BRANCH_FACTOR       2
#define DEFAULT_PATH_LENGTH         5 
#define DEFAULT_LEAFNODE_CAPACITY   25

//extern PyTypeObject PyHashTree_Type;

#define PyHashTree_Check(object) \
    PyObject_IsInstance(object, reinterpret_cast<PyObject *>(&PyHashTree_Type))

/// path check
static bool PyHashTree_PathExists(PyObject *path) {
    PyStringObject *putative = reinterpret_cast<PyStringObject *>(path);
    if (!PyString_Check(putative)) {
        PyErr_SetString(PyExc_ValueError, "Bad path string");
        return false;
    }
    PyObject *ospath = PyImport_ImportModuleNoBlock("os.path");
    PyObject *exists = PyObject_GetAttrString(ospath, "exists");
    return (bool)PyObject_IsTrue(
        PyObject_CallFunctionObjArgs(exists, putative, NULL));
}

/// SMELF ALERT!!!
static PyObject *PyHashTree_LoadFromMVPFile(PyObject *smelf, PyObject *args, PyObject *kwargs) {
    PyHashTree *self = reinterpret_cast<PyHashTree *>(smelf);
    CmpFunc comparator = PyHashTree_DF_HammingDistance;
    static char *keywords[] = { "path", NULL };
    PyObject *path;

    if (!PyArg_ParseTupleAndKeywords(args, kwargs,
                "O", keywords, &path)) {
        PyErr_SetString(PyExc_ValueError,
            "cannot load hash tree (bad argument tuple passed to PyHashTree_LoadFromMVPFile)");
        return NULL;
    }
    
    if (!PyHashTree_PathExists(path)) {
        Py_XDECREF(path);
        PyErr_SetString(PyExc_ValueError,
            "path does not exist");
        return NULL;
    }
    
    /// LOAD THAT SHIT
    MVPError error;
    MVPTree *tree = mvptree_read(PyString_AS_STRING(path),
        comparator, self->branch_factor, self->path_length,
        self->leafnode_capacity, &error);
    
    if (error != MVP_SUCCESS) {
        PyErr_Format(PyExc_ValueError,
            "Error loading MVP file: %s",
            mvp_errstr(error));
        return NULL;
    }
    
    /// clear out the old tree
    if (self->tree) { self->cleanup(); }
    
    /// set up the new tree
    self->tree = tree;
    
    return reinterpret_cast<PyObject *>(self); /// all is well, return self
}

static PyObject *PyHashTree_SaveToMVPFile(PyObject *smelf, PyObject *args, PyObject *kwargs) {
    PyHashTree *self = reinterpret_cast<PyHashTree *>(smelf);
    PyObject *path, *pyoverwrite;
    MVPError error;
    bool overwrite = true;
    bool exists = false;
    static char *keywords[] = { "path", "overwrite", NULL };

    if (!PyArg_ParseTupleAndKeywords(args, kwargs,
                "S|O", keywords,
                &path, &pyoverwrite)) {
        PyErr_SetString(PyExc_ValueError,
            "cannot save image (bad argument tuple passed to PyHashTree_SaveToMVPFile)");
        return NULL;
    }
    
    overwrite = PyObject_IsTrue(pyoverwrite);
    exists = PyHashTree_PathExists(path);
    
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
            error = mvptree_write(self->tree,
                PyString_AS_STRING(path), 00755);
            if (error == MVP_SUCCESS) {
                /// all is well, return self
                return reinterpret_cast<PyObject *>(self);
            } else {
                Py_XDECREF(path);
                PyErr_Format(PyExc_OSError,
                    "could not save file: %s",
                    mvp_errstr(error));
                return NULL;
            }
        } else {
            Py_XDECREF(path);
            PyErr_SetString(PyExc_SystemError,
                "could not overwrite existing file");
            return NULL;
        }
    }
    
    error = mvptree_write(self->tree,
        PyString_AS_STRING(path), 00755);
    if (error == MVP_SUCCESS) {
        /// all is well, return self
        return reinterpret_cast<PyObject *>(self);
    }
    
    /// we got here, something must be wrong by now
    Py_XDECREF(path);
    PyErr_Format(PyExc_OSError,
        "could not save file (out of options): %s",
        mvp_errstr(error));
    return NULL;
}

/// ALLOCATE / __new__ implementation
static PyObject *PyHashTree_new(PyTypeObject *type, PyObject *args, PyObject *kwargs) {
    PyHashTree *self;
    self = reinterpret_cast<PyHashTree *>(type->tp_alloc(type, 0));
    /// initialize with defaults
    if (self != NULL) {
        self->tree = NULL;
        self->branch_factor = DEFAULT_BRANCH_FACTOR;
        self->path_length = DEFAULT_PATH_LENGTH;
        self->leafnode_capacity = DEFAULT_LEAFNODE_CAPACITY;
    }
    return reinterpret_cast<PyObject *>(self); /// all is well, return self
}

/// __init__ implementation
static int PyHashTree_init(PyHashTree *self, PyObject *args, PyObject *kwargs) {
    unsigned int branch_factor = DEFAULT_BRANCH_FACTOR,
                 path_length = DEFAULT_PATH_LENGTH,
                 leafnode_capacity = DEFAULT_LEAFNODE_CAPACITY;
    static char *kwlist[] = { "tree",
        "branch_factor", "path_length", "leafnode_capacity", NULL };
    CmpFunc comparator = PyHashTree_DF_HammingDistance;
    PyObject *tree = NULL;

    if (!PyArg_ParseTupleAndKeywords(args, kwargs,
                "|OIII:PyHashTree.__init__", kwlist, &tree,
                &branch_factor, &path_length, &leafnode_capacity)) {
            PyErr_SetString(PyExc_ValueError,
                "cannot initialize PyHashTree (bad argument tuple)");
        return -1;
    }
    
    /// reset MVP tree parameters
    self->branch_factor = branch_factor;
    self->path_length = path_length;
    self->leafnode_capacity = leafnode_capacity;
    
    if (!tree) {
        /// nothing was passed in for a tree,
        /// allocate a new MVP tree and return
        self->tree = mvptree_alloc(NULL,
            comparator,
            self->branch_factor,
            self->path_length,
            self->leafnode_capacity);
        if (!self->tree) {
            PyErr_SetString(PyExc_SystemError,
                "Error allocating new MVPTree");
            return -1;
        }
        return 0;
    }
    
    if (PyUnicode_Check(tree)) {
        /// Fuck, it's unicode... let's de-UTF8 it
        tree = PyUnicode_AsUTF8String(tree);
    }
    
    if (PyString_Check(tree)) {
        /// tree is a file path
        PyObject *out = PyObject_CallMethodObjArgs(
            reinterpret_cast<PyObject *>(self),
            PyString_FromString("load"), tree, NULL);
        if (out == NULL) {
            PyErr_Format(PyExc_ValueError,
                "MVP tree failed to load from path: %s",
                PyString_AS_STRING(tree));
            return -1;
        }
        return 0;
    }
    
    //if (PyHashTree_Check(tree)) {
    if (true) {
        /// tree is a PyHashTree instance
        PyHashTree *pyhashtree = reinterpret_cast<PyHashTree *>(tree);
        if (!pyhashtree->tree) {
            PyErr_SetString(PyExc_ValueError,
                "Invalid PyHashTree: can't construct new instance");
            return -1;
        }
        self->tree = pyhashtree->tree;
        self->branch_factor = pyhashtree->branch_factor;
        self->path_length = pyhashtree->path_length;
        self->leafnode_capacity = pyhashtree->leafnode_capacity;
        return 0;
    }
    
    /// I DONT KNOW WHAT THE FUCK MAN
    PyErr_SetString(PyExc_ValueError,
        "NOTHING HAPPENED! Tree: NOPE.");
    return -1;
}

/// DEALLOCATE
static void PyHashTree_dealloc(PyHashTree *self) {
    self->cleanup();
    self->ob_type->tp_free((PyObject *)self);
}


#endif /// PyHashTree_PYHASHTREE_IMP_OBJPROTOCOL_H