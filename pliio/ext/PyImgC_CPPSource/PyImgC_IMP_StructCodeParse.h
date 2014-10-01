
#ifndef PyImgC_IMP_STRUCTCODEPARSE_H
#define PyImgC_IMP_STRUCTCODEPARSE_H

#include <vector>
#include <string>

#include <Python.h>
#include <numpy/ndarrayobject.h> /// for PyArray_Descr,
                                 /// for converting structcodes to NPY_TYPES

#include "numpypp/numpy.hpp"
#include "numpypp/structcode.hpp"
#include "numpypp/dispatch.hpp"
#include "numpypp/utils.hpp"


static PyObject *structcode_to_dtype_code(const char *code) {
    vector<pair<string, string>> pairvec = structcode::parse(string(code));
    string byteorder = "";

    if (!pairvec.size()) {
        PyErr_Format(PyExc_ValueError,
            "Struct typecode string %.200s parsed to zero-length pair vector",
            code);
        return NULL;
    }

    /// get special values
    for (size_t idx = 0; idx < pairvec.size(); idx++) {
        if (pairvec[idx].first == "__byteorder__") {
            byteorder = string(pairvec[idx].second);
            pairvec.erase(pairvec.begin()+idx);
        }
    }

    /// Make python list of tuples
    PyObject *list = PyList_New(static_cast<Py_ssize_t>(0));
    for (size_t idx = 0; idx < pairvec.size(); idx++) {
        PyList_Append(list,
            PyTuple_Pack(static_cast<Py_ssize_t>(2),
                PyString_FromString(string(pairvec[idx].first).c_str()),
                PyString_FromString(string(byteorder + pairvec[idx].second).c_str())));
    }
    
    return list;
}

static string structcode_atom_to_dtype_atom(const char *code) {
    vector<pair<string, string>> pairvec = structcode::parse(string(code));
    string byteorder = "=";

    if (!pairvec.size()) {
        PyErr_Format(PyExc_ValueError,
            "Structcode string %.200s parsed to zero-length pair vector",
            code);
        return NULL;
    }

    /// get special values
    for (size_t idx = 0; idx < pairvec.size(); idx++) {
        if (pairvec[idx].first == "__byteorder__") {
            byteorder = string(pairvec[idx].second);
            pairvec.erase(pairvec.begin()+idx);
        }
    }

    /// Get singular value
    return string(byteorder + pairvec[0].second);
}

static PyObject *PyImgC_ParseStructCode(PyObject *self, PyObject *args) {
    const char *code;

    if (!PyArg_ParseTuple(args, "s", &code)) {
        PyErr_SetString(PyExc_ValueError,
            "cannot get structcode string (bad argument)");
        return NULL;
    }

    return Py_BuildValue("O",
        structcode_to_dtype_code(code));
}

static PyObject *PyImgC_ParseSingleStructAtom(PyObject *self, PyObject *args) {
    char *code = None;

    if (!PyArg_ParseTuple(args, "s", &code)) {
        PyErr_SetString(PyExc_ValueError,
            "cannot get structcode string (bad argument)");
        return NULL;
    }

    return Py_BuildValue("O", PyString_FromString(
        structcode_atom_to_dtype_atom(code).c_str()));
}

static int PyImgC_NPYCodeFromStructAtom(PyObject *self, PyObject *args) {
    PyObject *dtypecode = PyImgC_ParseSingleStructAtom(self, args);
    PyArray_Descr *descr;
    int npy_type_num = 0;

    if (!dtypecode) {
        PyErr_SetString(PyExc_ValueError,
            "cannot get structcode string (bad argument)");
        return -1;
    }

    if (!PyArray_DescrConverter(dtypecode, &descr)) {
        PyErr_SetString(PyExc_ValueError,
            "cannot convert string to PyArray_Descr");
        return -1;
    }

    npy_type_num = static_cast<int>(descr->type_num);
    Py_XDECREF(dtypecode);
    Py_XDECREF(descr);

    return npy_type_num;
}

static PyObject *PyImgC_NumpyCodeFromStructAtom(PyObject *self, PyObject *args) {
    return Py_BuildValue("i", PyImgC_NPYCodeFromStructAtom(self, args));
}

#endif /// PyImgC_IMP_STRUCTCODEPARSE_H
