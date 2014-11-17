
#ifndef PyHashTree_PYHASHTREE_IMP_OBJPROTOCOL_H
#define PyHashTree_PYHASHTREE_IMP_OBJPROTOCOL_H

#include <Python.h>
#include "mvptree/mvptree.h"
#include "PyHashTree_Constants.h"
//#include "PyHashTree_GIL.h"
#include "PyHashTree_DistanceFunctions.h"

#define DEFAULT_BRANCH_FACTOR       2
#define DEFAULT_PATH_LENGTH         5 
#define DEFAULT_LEAFNODE_CAPACITY   25

/// path check
static bool PyHashTree_PathExists(char *path) {
    struct stat buffer;
    stat(path, &buffer);
    return S_ISREG(buffer.st_mode);
}
static bool PyHashTree_PathExists(const char *path) {
    return PyHashTree_PathExists(const_cast<char *>(path));
}
static bool PyHashTree_PathExists(PyObject *path) {
    return PyHashTree_PathExists(PyString_AS_STRING(path));
}
static bool PyHashTree_PathExists(string path) {
    return PyHashTree_PathExists(path.c_str());
}

/// SMELF ALERT!!!
static void PyHashTree_LoadFromMVPFile(PyObject *smelf, PyObject *args, PyObject *kwargs) {
    PyHashTree *self = reinterpret_cast<PyHashTree *>(smelf);
    CmpFunc comparator = PyHashTree_DF_HammingDistance;
    static char *keywords[] = { "path", NULL };
    char *cpath;

    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "|s", keywords, &cpath)) {
        PyErr_SetString(PyExc_ValueError,
            "cannot load hash tree");
        return;
    }
    
    if (!PyHashTree_PathExists(cpath)) {
        PyErr_SetString(PyExc_ValueError,
            "path does not exist");
        return;
    }
    
    /// LOAD THAT SHIT
    //int fd;
    //char *cpath = PyString_AsString(pth);
    MVPError error;
    // PyFileObject *mvpfile = reinterpret_cast<PyFileObject *>(PyFile_FromString(cpath, "r+x"));
    // if (!mvpfile) {
    //     PyErr_Format(PyExc_ValueError,
    //         "Error opening PyFileObject %s", cpath);
    //     return -1;
    // }
    
    //FILE *fh = PyFile_AsFile(reinterpret_cast<PyObject *>(mvpfile));
    //PyFile_IncUseCount(mvpfile);
    //gil_release NOGIL;
    //fd = fileno(fh);
    // MVPTree *tree = mvptree_read_fd(fd,
    //     comparator, self->branch_factor, self->path_length,
    //     self->leafnode_capacity, &error);
    MVPTree *tree = mvptree_read(cpath,
        comparator, self->branch_factor, self->path_length,
        self->leafnode_capacity, &error);
    //NOGIL.~gil_release();
    //PyFile_DecUseCount(mvpfile);
    //fclose(fh);
    //fd = 0;
    
    // if (error != MVP_SUCCESS) {
    //     PyErr_Format(PyExc_ValueError,
    //         "Error loading MVP file %s", cpath);
    //     return NULL;
    // }
    
    /// clear out the old tree
    //if (self->tree) { self->cleanup(); }
    
    /// set up the new tree
    self->tree = tree;
    return;
    //return reinterpret_cast<PyObject *>(self); /// all is well, return self
}

static PyObject *PyHashTree_SaveToMVPFile(PyObject *smelf, PyObject *args, PyObject *kwargs) {
    PyHashTree *self = reinterpret_cast<PyHashTree *>(smelf);
    PyObject *path, *pyoverwrite = NULL;
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
    
    if (pyoverwrite != NULL) { overwrite = PyObject_IsTrue(pyoverwrite); }
    //exists = PyHashTree_PathExists(path);
    const char *cpath = PyString_AsString(path);
    Py_XDECREF(path);
    
    /// SAVE THAT SHIT
    if (exists && !overwrite) {
        /// DON'T OVERWRITE
        PyErr_SetString(PyExc_NameError,
            "path already exists");
        return NULL;
    }
    
    if (exists && overwrite) {
        /// PLEASE DO OVERWRITE
        if (remove(cpath)) {
            //gil_release NOGIL;
            error = mvptree_write(self->tree, cpath, 00644);
            //NOGIL.~gil_release();
            if (error == MVP_SUCCESS) {
                /// all is well, return self
                return reinterpret_cast<PyObject *>(self);
            } else {
                PyErr_Format(PyExc_OSError,
                    "could not save file: %s",
                    mvp_errstr(error));
                return NULL;
            }
        } else {
            PyErr_SetString(PyExc_SystemError,
                "could not overwrite existing file");
            return NULL;
        }
    }
    
    if (!self->tree->node) {
        PyErr_SetString(PyExc_ValueError,
            "Can't save tree with no 'node' attribute");
        return NULL;
    }
    if (!self->tree->dist) {
        PyErr_SetString(PyExc_ValueError,
            "Can't save tree with no 'dist' attribute");
        return NULL;
    }
    
    //gil_release NOGIL;
    error = mvptree_write(self->tree, cpath, 00644);
    //NOGIL.~gil_release();
    
    if (error == MVP_SUCCESS) {
        /// all is well, return self
        return reinterpret_cast<PyObject *>(self);
    }
    
    /// we got here, something must be wrong by now
    PyErr_Format(PyExc_OSError,
        "could not save file (%s): %s",
        cpath, mvp_errstr(error));
    
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
                "|OIII", kwlist, &tree,
                &branch_factor, &path_length, &leafnode_capacity)) {
        PyErr_SetString(PyExc_ValueError,
            "cannot initialize PyHashTree (bad argument tuple)");
        return -1;
    }
    
    /// reset MVP tree parameters
    self->branch_factor = branch_factor;
    self->path_length = path_length;
    self->leafnode_capacity = leafnode_capacity;
    
    // if (PyUnicode_Check(tree)) {
    //     /// Fuck, it's unicode... let's de-UTF8 it
    //     tree = PyUnicode_AsUTF8String(tree);
    // }
    
    if (!tree) {
        /// nothing was passed in for a tree,
        /// allocate a new MVP tree and return
        //gil_release NOGIL;
        self->tree = mvptree_alloc(NULL,
            comparator,
            self->branch_factor,
            self->path_length,
            self->leafnode_capacity);
        //NOGIL.~gil_release();
        if (!self->tree) {
            PyErr_SetString(PyExc_SystemError,
                "Error allocating new MVPTree");
            return -1;
        }
        return 0;
    }
    
    
    if (PyString_Check(tree)) {
        /// tree is a file path
        PyObject_CallMethodObjArgs(
            reinterpret_cast<PyObject *>(self),
            PyString_FromString("load"), tree, NULL);
        // if (out == NULL) {
        //     PyErr_Format(PyExc_ValueError,
        //         "MVP tree failed to load from path: %s",
        //         PyString_AsString(tree));
        //     return -1;
        // }
        // Py_DECREF(out);
        return 0;
    }
    
    return 0;
    
    //if (PyHashTree_Check(tree)) {
    // if (true) {
    //     /// tree is a PyHashTree instance
    //     PyHashTree *pyhashtree = reinterpret_cast<PyHashTree *>(tree);
    //     if (!pyhashtree->tree) {
    //         PyErr_SetString(PyExc_ValueError,
    //             "Invalid PyHashTree: can't construct new instance");
    //         return -1;
    //     }
    //     self->tree = pyhashtree->tree;
    //     self->branch_factor = pyhashtree->branch_factor;
    //     self->path_length = pyhashtree->path_length;
    //     self->leafnode_capacity = pyhashtree->leafnode_capacity;
    //     return 0;
    // }
    
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