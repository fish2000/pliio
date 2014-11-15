
#ifndef PyHashTree_PYHASHTREE_IMP_PRINTFUNCTIONS_H
#define PyHashTree_PYHASHTREE_IMP_PRINTFUNCTIONS_H

#include <Python.h>
#include <stdio.h>
#include "fmemopen/open_memstream.h"
#include "mvptree/mvptree.h"

#include <string>
using namespace std;

/// __repr__ implementations
static PyObject *PyHashTree_Repr(PyHashTree *tree) {
    if (!tree->tree) {
        return PyString_FromFormat("<PyHashTree (NULL) @ %p>", tree);
    }
    return PyString_FromFormat("<PyHashTree (%i,%i,%i) @ %p>",
        tree->branch_factor,
        tree->path_length, tree->leafnode_capacity,
        tree);
}

static const char *PyHashTree_ReprCString(PyHashTree *tree) {
    return PyString_AS_STRING(PyHashTree_Repr(tree));
}

static string PyHashTree_ReprString(PyHashTree *tree) {
    return string(PyHashTree_ReprCString(tree));
}

/// __str__ implementation
static PyObject *PyHashTree_Str(PyHashTree *tree) {
    if (!tree->tree) {
        return PyString_FromFormat("<PyHashTree (NULL) @ %p>", tree);
    }
    char *out;
    size_t siz;
    FILE *outstream;
    MVPError error;
    outstream = open_memstream(&out, &siz);
    error = mvptree_print(outstream, tree->tree);
    fflush(outstream);
    fclose(outstream);
    if (error != MVP_SUCCESS) {
        PyErr_Format(PyExc_ValueError,
            "Error stringifying hash tree: %s",
            mvp_errstr(error));
        return NULL;
    }
    return PyString_FromStringAndSize(
        const_cast<char *>(out),
        static_cast<Py_ssize_t>(siz));
}

static int PyHashTree_Print(PyObject *self, FILE *outstream, int flags) {
    PyHashTree *tree = reinterpret_cast<PyHashTree *>(self);
    if (!tree->tree) {
        PyErr_SetString(PyExc_ValueError,
            "Error printing hash tree: no hashtree instance");
        return -1;
    }
    if (flags == Py_PRINT_RAW) {
        MVPError error = mvptree_print(outstream, tree->tree);
        if (error != MVP_SUCCESS) {
            PyErr_Format(PyExc_ValueError,
                "Error printing hash tree: %s",
                mvp_errstr(error));
            return 0;
        }
        return 0;
    }
    return fprintf(outstream, "%s", PyHashTree_ReprCString(tree));
}


#endif /// PyHashTree_PYHASHTREE_IMP_PRINTFUNCTIONS_H