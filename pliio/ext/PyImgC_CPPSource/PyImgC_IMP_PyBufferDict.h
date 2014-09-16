
#ifndef PyImgC_IMP_PYBUFFERDICT_H
#define PyImgC_IMP_PYBUFFERDICT_H

#include <Python.h>
//#include "PyImgC_IMP_StructCodeParse.h"

static PyObject *structcode_to_dtype_code(const char *code); /// FOREWARD!

static PyObject *PyImgC_PyBufferDict(PyObject *self, PyObject *args, PyObject *kwargs) {
    PyObject *buffer_dict = PyDict_New();
    PyObject *buffer = self, *parse_format_arg = PyInt_FromLong((long)1);
    static char *keywords[] = { "buffer", "parse_format", None };

    if (!PyArg_ParseTupleAndKeywords(
        args, kwargs, "|OO",
        keywords,
        &buffer, &parse_format_arg)) {
            PyErr_SetString(PyExc_ValueError,
                "cannot get Py_buffer (bad argument)");
            return NULL;
    }

    if (PyObject_CheckBuffer(buffer)) {
        /// buffer3000
        Py_buffer buf;
        if (PyObject_GetBuffer(buffer, &buf, PyBUF_FORMAT) == -1) {
            if (PyObject_GetBuffer(buffer, &buf, PyBUF_SIMPLE) == -1) {
                PyErr_Format(PyExc_ValueError,
                    "cannot get Py_buffer from %.200s instance (tried PyBUF_FORMAT and PyBUF_SIMPLE)",
                    buffer->ob_type->tp_name);
                return NULL;
            }
        }

        if (buf.len) {
            PyDict_SetItemString(buffer_dict, "len", PyInt_FromSsize_t(buf.len));
        } else {
            PyDict_SetItemString(buffer_dict, "len", PyGetNone);
        }

        if (buf.readonly) {
            PyDict_SetItemString(buffer_dict, "readonly", PyBool_FromLong((long)buf.readonly));
        } else {
            PyDict_SetItemString(buffer_dict, "readonly", PyGetNone);
        }

        if (buf.format) {
            if (PyObject_IsTrue(parse_format_arg)) {
                PyDict_SetItemString(buffer_dict, "format", structcode_to_dtype_code(buf.format));
            } else {
                PyDict_SetItemString(buffer_dict, "format", PyString_FromString(buf.format));
            }
        } else {
            PyDict_SetItemString(buffer_dict, "format", PyGetNone);
        }

        if (buf.ndim) {
            PyDict_SetItemString(buffer_dict, "ndim", PyInt_FromLong((long)buf.ndim));

            if (buf.shape) {
                PyObject *shape = PyTuple_New((Py_ssize_t)buf.ndim);
                for (int idx = 0; idx < (int)buf.ndim; idx++) {
                    PyTuple_SET_ITEM(shape, (Py_ssize_t)idx, PyInt_FromSsize_t(buf.shape[idx]));
                }
                PyDict_SetItemString(buffer_dict, "shape", shape);
            } else {
                PyDict_SetItemString(buffer_dict, "shape", PyGetNone);
            }

            if (buf.strides) {
                PyObject *strides = PyTuple_New((Py_ssize_t)buf.ndim);
                for (int idx = 0; idx < (int)buf.ndim; idx++) {
                    PyTuple_SET_ITEM(strides, (Py_ssize_t)idx, PyInt_FromSsize_t(buf.strides[idx]));
                }
                PyDict_SetItemString(buffer_dict, "strides", strides);
            } else {
                PyDict_SetItemString(buffer_dict, "strides", PyGetNone);
            }

            if (buf.suboffsets) {
                PyObject *suboffsets = PyTuple_New((Py_ssize_t)buf.ndim);
                for (int idx = 0; idx < (int)buf.ndim; idx++) {
                    PyTuple_SET_ITEM(suboffsets, (Py_ssize_t)idx, PyInt_FromSsize_t(buf.suboffsets[idx]));
                }
                PyDict_SetItemString(buffer_dict, "suboffsets", suboffsets);
            } else {
                PyDict_SetItemString(buffer_dict, "suboffsets", PyGetNone);
            }

        } else {
            PyDict_SetItemString(buffer_dict, "ndim", PyGetNone);
            PyDict_SetItemString(buffer_dict, "shape", PyGetNone);
            PyDict_SetItemString(buffer_dict, "strides", PyGetNone);
            PyDict_SetItemString(buffer_dict, "suboffsets", PyGetNone);
        }

        if (buf.itemsize) {
            PyDict_SetItemString(buffer_dict, "itemsize", PyInt_FromSsize_t(buf.itemsize));
        } else {
            PyDict_SetItemString(buffer_dict, "itemsize", PyGetNone);
        }

        PyBuffer_Release(&buf);
        return Py_BuildValue("O", buffer_dict);
        
    } else if (PyBuffer_Check(buffer)) {
        /// legacybuf
        PyErr_Format(PyExc_ValueError,
            "cannot get a buffer from %.200s instance (only supports legacy PyBufferObject)",
            buffer->ob_type->tp_name);
        return NULL;
    }

    PyErr_Format(PyExc_ValueError,
        "no buffer info for %.200s instance (no buffer API supported)",
        buffer->ob_type->tp_name);
    return NULL;
}

#endif /// PyImgC_IMP_PYBUFFERDICT_H
