
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
#include "PyImgC_Constants.h"
#include "PyImgC_PyCImage.h"

PyObject *structcode_to_dtype_code(const char *code);
string structcode_atom_to_dtype_atom(const char *code);

PyObject *PyImgC_ParseStructCode(PyObject *self, PyObject *args);
PyObject *PyImgC_ParseSingleStructAtom(PyObject *self, PyObject *args);
int PyImgC_NPYCodeFromStructAtom(PyObject *self, PyObject *args);
PyObject *PyImgC_NumpyCodeFromStructAtom(PyObject *self, PyObject *args);

#endif /// PyImgC_IMP_STRUCTCODEPARSE_H
