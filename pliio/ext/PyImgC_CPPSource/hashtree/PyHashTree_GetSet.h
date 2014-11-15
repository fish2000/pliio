
#ifndef PyHashTree_PYHASHTREE_IMP_GETSET_H
#define PyHashTree_PYHASHTREE_IMP_GETSET_H

#include <Python.h>
#include "PyHashTree_Constants.h"

/// pyhashtree.branch_factor getter/setter
static PyObject     *PyHashTree_GET_branch_factor(PyHashTree *self, void *closure) {
    return PyLong_FromUnsignedLong(
        static_cast<unsigned long>(
            self->branch_factor));
}
static int           PyHashTree_SET_branch_factor(PyHashTree *self, PyObject *value, void *closure) {
    if (!PyLong_Check(value) || !PyInt_Check(value)) {
        PyErr_SetString(PyExc_ValueError,
            "branch_factor must be an integer");
        return -1;
    }
    self->branch_factor = static_cast<unsigned int>(
        PyInt_AsUnsignedLongMask(value));
    return 0;
}

/// pyhashtree.path_length getter/setter
static PyObject     *PyHashTree_GET_path_length(PyHashTree *self, void *closure) {
    return PyLong_FromUnsignedLong(
        static_cast<unsigned long>(
            self->path_length));
}
static int           PyHashTree_SET_path_length(PyHashTree *self, PyObject *value, void *closure) {
    if (!PyLong_Check(value) || !PyInt_Check(value)) {
        PyErr_SetString(PyExc_ValueError,
            "path_length must be an integer");
        return -1;
    }
    self->path_length = static_cast<unsigned int>(
        PyInt_AsUnsignedLongMask(value));
    return 0;
}

/// pyhashtree.leafnode_capacity getter/setter
static PyObject     *PyHashTree_GET_leafnode_capacity(PyHashTree *self, void *closure) {
    return PyLong_FromUnsignedLong(
        static_cast<unsigned long>(
            self->leafnode_capacity));
}
static int           PyHashTree_SET_leafnode_capacity(PyHashTree *self, PyObject *value, void *closure) {
    if (!PyLong_Check(value) || !PyInt_Check(value)) {
        PyErr_SetString(PyExc_ValueError,
            "leafnode_capacity must be an integer");
        return -1;
    }
    self->leafnode_capacity = static_cast<unsigned int>(
        PyInt_AsUnsignedLongMask(value));
    return 0;
}


static PyGetSetDef PyHashTree_getset[] = {
    {
        "branch_factor",
            (getter)PyHashTree_GET_branch_factor,
            (setter)PyHashTree_SET_branch_factor,
            "Branch Factor", None },
    {
        "path_length",
            (getter)PyHashTree_GET_path_length,
            (setter)PyHashTree_SET_path_length,
            "Path Length", None },
    {
        "leafnode_capacity",
            (getter)PyHashTree_GET_leafnode_capacity,
            (setter)PyHashTree_SET_leafnode_capacity,
            "Leafnode Capacity", None },
    SENTINEL
};

#endif /// PyHashTree_PYHASHTREE_IMP_GETSET_H