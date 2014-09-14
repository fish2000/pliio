
#ifndef PyImgC_TYPESTRUCT_PYCIMAGE_H
#define PyImgC_TYPESTRUCT_PYCIMAGE_H

#include <vector>
#include <string>

#include <Python.h>
#include <numpy/ndarrayobject.h> /// for PyArray_Descr,
                                 /// for converting structcodes to NPY_TYPES

#include "numpypp/numpy.hpp"
#include "numpypp/structcode.hpp"
#include "numpypp/dispatch.hpp"
#include "numpypp/utils.hpp"

/// ... THESE ARE USED IN CIMG.H PLUGINS

/// structcode parser invocation (from pyimgc.cpp)
const char *structcode_to_dtype(const char *structcode, bool include_byteorder=true) {
    vector<pair<string, string>> pairvec = structcode::parse(string(structcode));
    string byteorder = "=";

    if (!pairvec.size()) {
        throw CImgInstanceException(_cimg_instance
                                    "Structcode string parsed to zero-length pair vector",
                                    cimg_instance);
    }

    /// get special values
    for (size_t idx = 0; idx < pairvec.size(); idx++) {
        if (pairvec[idx].first == "__byteorder__") {
            byteorder = string(pairvec[idx].second);
            pairvec.erase(pairvec.begin()+idx);
        }
    }

    /// Get singular value
    if (include_byteorder) {
        return string(byteorder + pairvec[0].second).c_str();
    }
    return string(pairvec[0].second).c_str();
}

unsigned int structcode_to_typecode(const char *structcode) {
    const char *dtypecode = structcode_to_dtype(structcode);
    PyArray_Descr *descr;
    int npy_type_num = 0;

    if (!dtypecode) {
        throw CImgInstanceException(_cimg_instance
                                    "Cannot get structcode string (bad argument)",
                                    cimg_instance);
    }

    if (!PyArray_DescrConverter(PyString_FromString(dtypecode), &descr)) {
        throw CImgInstanceException(_cimg_instance
                                    "cannot convert string to PyArray_Descr",
                                    cimg_instance);
    }

    npy_type_num = (unsigned int)descr->type_num;
    Py_XDECREF(descr);

    return npy_type_num;
}


/// ... whereas THESE ARE USED IN THE MODULE
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
    PyObject *list = PyList_New((Py_ssize_t)0);
    for (size_t idx = 0; idx < pairvec.size(); idx++) {
        PyList_Append(list,
            PyTuple_Pack((Py_ssize_t)2,
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

    npy_type_num = (int)descr->type_num;
    Py_XDECREF(dtypecode);
    Py_XDECREF(descr);

    return npy_type_num;
}

static PyObject *PyImgC_NumpyCodeFromStructAtom(PyObject *self, PyObject *args) {
    return Py_BuildValue("i", PyImgC_NPYCodeFromStructAtom(self, args));
}

#endif /// PyImgC_TYPESTRUCT_PYCIMAGE_H
