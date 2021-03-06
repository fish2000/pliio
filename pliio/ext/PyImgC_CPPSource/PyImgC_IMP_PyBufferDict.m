
#include "PyImgC_IMP_PyBufferDict.h"

PyObject *PyImgC_PyBufferDict(PyObject *self, PyObject *args, PyObject *kwargs) {
    PyObject *buffer_dict = PyDict_New();
    PyObject *buffer = NULL,
             *parse_format_arg = PyInt_FromLong(1L);
    static char *keywords[] = { "buffer", "parse_format", NULL };

    if (!PyArg_ParseTupleAndKeywords(
        args, kwargs, "O|O",
        keywords,
        &buffer, &parse_format_arg)) {
            PyErr_SetString(PyExc_ValueError,
                "cannot get Py_buffer (bad argument)");
            return NULL;
    }

    if (PyObject_CheckBuffer(buffer)) {
        /// buffer3000
        Py_buffer buf;
        int idx;
        if (PyObject_GetBuffer(buffer, &buf, PyBUF_FORMAT) == -1) {
            if (PyObject_GetBuffer(buffer, &buf, PyBUF_SIMPLE) == -1) {
                PyErr_Format(PyExc_ValueError,
                    "cannot get Py_buffer from %.200s instance (tried PyBUF_FORMAT and PyBUF_SIMPLE)",
                    buffer->ob_type->tp_name);
                return NULL;
            }
        }
        
        if (buf.len) {
            PyDict_SetItemString(buffer_dict, "len",
                PyInt_FromSsize_t(buf.len));
        } else {
            PyDict_SetItemString(buffer_dict, "len", Py_BuildValue(""));
        }
        

        if (buf.readonly) {
            PyDict_SetItemString(buffer_dict, "readonly",
                PyBool_FromLong(buf.readonly));
        } else {
            PyDict_SetItemString(buffer_dict, "readonly", Py_BuildValue(""));
        }

        if (buf.format) {
            if (PyObject_IsTrue(parse_format_arg)) {
                PyDict_SetItemString(buffer_dict, "format",
                    structcode_to_dtype_code(buf.format));
            } else {
                PyDict_SetItemString(buffer_dict, "format",
                    PyString_FromString(buf.format));
            }
        } else {
            PyDict_SetItemString(buffer_dict, "format", Py_BuildValue(""));
        }

        if (buf.ndim) {
            PyDict_SetItemString(buffer_dict, "ndim",
                PyInt_FromLong(buf.ndim));

            if (buf.shape) {
                PyObject *shape = PyTuple_New(buf.ndim);
                for (idx = 0; idx < buf.ndim; ++idx) {
                    PyTuple_SET_ITEM(shape, idx,
                        PyInt_FromLong(buf.shape[idx]));
                }
                PyDict_SetItemString(buffer_dict, "shape", shape);
            } else {
                PyDict_SetItemString(buffer_dict, "shape", Py_BuildValue(""));
            }

            if (buf.strides) {
                PyObject *strides = PyTuple_New(buf.ndim);
                for (idx = 0; idx < buf.ndim; ++idx) {
                    PyTuple_SET_ITEM(strides, idx,
                        PyInt_FromLong(buf.strides[idx]));
                }
                PyDict_SetItemString(buffer_dict, "strides", strides);
            } else {
                PyDict_SetItemString(buffer_dict, "strides", Py_BuildValue(""));
            }

            if (buf.suboffsets) {
                PyObject *suboffsets = PyTuple_New(buf.ndim);
                for (idx = 0; idx < buf.ndim; ++idx) {
                    PyTuple_SET_ITEM(suboffsets, idx,
                        PyInt_FromLong(buf.suboffsets[idx]));
                }
                PyDict_SetItemString(buffer_dict, "suboffsets", suboffsets);
            } else {
                PyDict_SetItemString(buffer_dict, "suboffsets", Py_BuildValue(""));
            }

        } else {
            PyDict_SetItemString(buffer_dict, "ndim", Py_BuildValue(""));
            PyDict_SetItemString(buffer_dict, "shape", Py_BuildValue(""));
            PyDict_SetItemString(buffer_dict, "strides", Py_BuildValue(""));
            PyDict_SetItemString(buffer_dict, "suboffsets", Py_BuildValue(""));
        }

        if (buf.itemsize) {
            PyDict_SetItemString(buffer_dict, "itemsize",
                PyInt_FromSsize_t(buf.itemsize));
        } else {
            PyDict_SetItemString(buffer_dict, "itemsize", Py_BuildValue(""));
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

PyObject *PyCImage_PyBufferDict(PyObject *self, PyObject *args, PyObject *kwargs) {
    PyObject *parse_format_arg = PyInt_FromLong(1L);
    static char *keywords[] = { "parse_format", NULL };

    if (!PyArg_ParseTupleAndKeywords(
        args, kwargs, "|O",
        keywords,
        &parse_format_arg)) {
            PyErr_SetString(PyExc_ValueError,
                "bad arguments");
            return NULL;
    }
    
    PyObject *imgc = PyImport_ImportModuleNoBlock("pliio.imgc");
    PyObject *buffer_info = PyObject_GetAttrString(imgc, "buffer_info");
    return PyObject_CallFunction(buffer_info, "(OO)",
                                 self, parse_format_arg, NULL);
}
