
#ifndef PyImgC_PYIMGC_IMP_SEQUENCEPROTOCOL_H
#define PyImgC_PYIMGC_IMP_SEQUENCEPROTOCOL_H

#include <Python.h>
#include <numpy/ndarrayobject.h>
#include "mvptree/mvpvector.hpp"
#include "DataPoint_Type.h"

using namespace std;

/// __len__ implementation
static Py_ssize_t PyHashTree_Len(PyHashTree *tree) {
    return static_cast<Py_ssize_t>(tree->size());
}

/// __getitem__ implementation
static PyObject *PyHashTree_GetItem(PyHashTree *tree, register Py_ssize_t idx) {
    MVPDP *dp = tree->datapoint(static_cast<size_t>(idx));
    if (dp == NULL) {
        PyErr_SetNone(PyExc_StopIteration);
        return NULL;
    }
    DataPoint *instance = reinterpret_cast<DataPoint *>(
        DataPoint_Type.tp_alloc(&DataPoint_Type, 0));
    Py_INCREF(tree);
    instance->tree = reinterpret_cast<PyObject *>(tree);
    instance->dp = dp;
    Py_INCREF(instance);
    return reinterpret_cast<PyObject *>(instance);
}

static PySequenceMethods PyHashTree_SequenceMethods = {
    (lenfunc)PyHashTree_Len,                    /* sq_length */
    0,                                          /* sq_concat */
    0,                                          /* sq_repeat */
    (ssizeargfunc)PyHashTree_GetItem,           /* sq_item */
    0,                                          /* sq_slice */
    0,                                          /* sq_ass_item HAHAHAHA */
    0,                                          /* sq_ass_slice HEHEHE ASS <snort> HA */
    0                                           /* sq_contains */
};

/*
static PyObject *PyHashTree_GetIter(PyObject *tree) {
}

static PyObject *PyHashTree_IterNext(PyObject *iterator) {
}
*/

#endif /// PyImgC_PYIMGC_IMP_SEQUENCEPROTOCOL_H
