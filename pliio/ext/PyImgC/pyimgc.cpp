
#include "pyimgc.h"
#include "numpypp/numpy.hpp"
#include "numpypp/structcode.hpp"
#include "numpypp/dispatch.hpp"
#include "numpypp/utils.hpp"
#include "PyImgC_CImage.h"

#include <iostream>
#include <vector>
#include <string>
#include <typeinfo>

using namespace std;

typedef struct {
    PyObject_HEAD
    PyObject *buffer;
    PyObject *source;
    PyObject *dtype;
} Image;

struct PyCImage {
    PyObject_HEAD
    PyArray_Descr *dtype = NULL;
    unique_ptr<CImage_SubBase> cimage = unique_ptr<CImage_SubBase>(nullptr);
    template <typename T>
    CImage_Type<T> *recast() { return dynamic_cast<CImage_Type<T>*>(cimage.get()); }
};

static PyObject *PyImgC_CImageTest(PyObject *self, PyObject *args, PyObject *kwargs) {
    PyObject *buffer = NULL;
    Py_ssize_t nin = -1, offset = 0;
    static char *kwlist[] = { "buffer", "dtype", "count", "offset", NULL };
    PyArray_Descr *dtype = NULL;

    if (!PyArg_ParseTupleAndKeywords(args, kwargs,
                "O|O&" NPY_SSIZE_T_PYFMT NPY_SSIZE_T_PYFMT, kwlist,
                &buffer, PyArray_DescrConverter, &dtype, &nin, &offset)) {
        Py_XDECREF(dtype);
        return NULL;
    }

    if (dtype == NULL) {
        if (PyArray_Check(buffer)) {
            dtype = PyArray_DESCR(
                reinterpret_cast<PyArrayObject *>(buffer));
        } else {
            dtype = PyArray_DescrFromType(NPY_DEFAULT_TYPE);
        }
    }

    if (PyArray_Check(buffer)) {
        switch ((int)dtype->type_num) {
            case NPY_BOOL: {
                auto converter = CImage_NumpyConverter<npy_uint8>(buffer);
                CImg<npy_uint8> cimage = converter->from_pyarray();
                return Py_BuildValue("iiiii", dtype->type_num, (int)converter->typecode(),
                                            cimage.width(), cimage.height(), cimage.spectrum());
            }
            case NPY_FLOAT: {
                auto converter = CImage_NumpyConverter<npy_float>(buffer);
                CImg<npy_float> cimage = converter->from_pyarray();
                return Py_BuildValue("iiiii", dtype->type_num, (int)converter->typecode(),
                                            cimage.width(), cimage.height(), cimage.spectrum());
            }
            case NPY_DOUBLE: {
                auto converter = CImage_NumpyConverter<npy_double>(buffer);
                CImg<npy_double> cimage = converter->from_pyarray();
                return Py_BuildValue("iiiii", dtype->type_num, (int)converter->typecode(),
                                            cimage.width(), cimage.height(), cimage.spectrum());
            }
            case NPY_LONGDOUBLE: {
                auto converter = CImage_NumpyConverter<npy_longdouble>(buffer);
                CImg<npy_longdouble> cimage = converter->from_pyarray();
                return Py_BuildValue("iiiii", dtype->type_num, (int)converter->typecode(),
                                            cimage.width(), cimage.height(), cimage.spectrum());
            }
            case NPY_BYTE: {
                auto converter = CImage_NumpyConverter<npy_byte>(buffer);
                CImg<npy_byte> cimage = converter->from_pyarray();
                return Py_BuildValue("iiiii", dtype->type_num, (int)converter->typecode(),
                                            cimage.width(), cimage.height(), cimage.spectrum());
            }
            case NPY_CHAR: {
                auto converter = CImage_NumpyConverter<npy_char>(buffer);
                CImg<npy_char> cimage = converter->from_pyarray();
                return Py_BuildValue("iiiii", dtype->type_num, (int)converter->typecode(),
                                            cimage.width(), cimage.height(), cimage.spectrum());
            }
            case NPY_SHORT: {
                auto converter = CImage_NumpyConverter<npy_short>(buffer);
                CImg<npy_short> cimage = converter->from_pyarray();
                return Py_BuildValue("iiiii", dtype->type_num, (int)converter->typecode(),
                                            cimage.width(), cimage.height(), cimage.spectrum());
            }
            case NPY_INT: {
                auto converter = CImage_NumpyConverter<npy_int>(buffer);
                CImg<npy_int> cimage = converter->from_pyarray();
                return Py_BuildValue("iiiii", dtype->type_num, (int)converter->typecode(),
                                            cimage.width(), cimage.height(), cimage.spectrum());
            }
            case NPY_LONG: {
                auto converter = CImage_NumpyConverter<npy_long>(buffer);
                CImg<npy_long> cimage = converter->from_pyarray();
                return Py_BuildValue("iiiii", dtype->type_num, (int)converter->typecode(),
                                            cimage.width(), cimage.height(), cimage.spectrum());
            }
            case NPY_LONGLONG: {
                auto converter = CImage_NumpyConverter<npy_longlong>(buffer);
                CImg<npy_longlong> cimage = converter->from_pyarray();
                return Py_BuildValue("iiiii", dtype->type_num, (int)converter->typecode(),
                                            cimage.width(), cimage.height(), cimage.spectrum());
            }
            case NPY_UBYTE: {
                auto converter = CImage_NumpyConverter<npy_ubyte>(buffer);
                CImg<npy_ubyte> cimage = converter->from_pyarray();
                return Py_BuildValue("iiiii", dtype->type_num, (int)converter->typecode(),
                                            cimage.width(), cimage.height(), cimage.spectrum());
            }
            case NPY_USHORT: {
                auto converter = CImage_NumpyConverter<npy_ushort>(buffer);
                CImg<npy_ushort> cimage = converter->from_pyarray();
                return Py_BuildValue("iiiii", dtype->type_num, (int)converter->typecode(),
                                            cimage.width(), cimage.height(), cimage.spectrum());
            }
            case NPY_UINT: {
                auto converter = CImage_NumpyConverter<npy_uint>(buffer);
                CImg<npy_uint> cimage = converter->from_pyarray();
                return Py_BuildValue("iiiii", dtype->type_num, (int)converter->typecode(),
                                            cimage.width(), cimage.height(), cimage.spectrum());
            }
            case NPY_ULONG: {
                auto converter = CImage_NumpyConverter<npy_ulong>(buffer);
                CImg<npy_ulong> cimage = converter->from_pyarray();
                return Py_BuildValue("iiiii", dtype->type_num, (int)converter->typecode(),
                                            cimage.width(), cimage.height(), cimage.spectrum());
            }
            case NPY_ULONGLONG: {
                auto converter = CImage_NumpyConverter<npy_ulonglong>(buffer);
                CImg<npy_ulonglong> cimage = converter->from_pyarray();
                return Py_BuildValue("iiiii", dtype->type_num, (int)converter->typecode(),
                                            cimage.width(), cimage.height(), cimage.spectrum());
            }
            
        }
    }
    
    return Py_BuildValue("");
}

static PyObject *structcode_parse(const char *code) {
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
                PyString_InternFromString(string(pairvec[idx].first).c_str()),
                PyString_InternFromString(string(byteorder + pairvec[idx].second).c_str())));
    }
    
    return list;
}

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

        /// len, readonly, format, ndim, Py_ssize_t *shape, Py_ssize_t *strides, Py_ssize_t *suboffsets, Py_ssize_t itemsize,  
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
                PyDict_SetItemString(buffer_dict, "format", structcode_parse(buf.format));
            } else {
                PyDict_SetItemString(buffer_dict, "format", PyString_InternFromString(buf.format));
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

static PyObject *PyImgC_ParseStructCode(PyObject *self, PyObject *args) {
    const char *code;

    if (!PyArg_ParseTuple(args, "s", &code)) {
        PyErr_SetString(PyExc_ValueError,
            "cannot get structcode string (bad argument)");
        return NULL;
    }

    return Py_BuildValue("O", structcode_parse(code));
}

static PyObject *PyImgC_ParseSingleStructAtom(PyObject *self, PyObject *args) {
    char *code = None;

    if (!PyArg_ParseTuple(args, "s", &code)) {
        PyErr_SetString(PyExc_ValueError,
            "cannot get structcode string (bad argument)");
        return NULL;
    }

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
    PyObject *dtypecode = PyString_InternFromString(
        string(byteorder + pairvec[0].second).c_str());

    return Py_BuildValue("O", dtypecode);
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


static PyObject *PyCImage_new(PyTypeObject *type, PyObject *args, PyObject *kwargs) {
    PyCImage *self;
    self = (PyCImage *)type->tp_alloc(type, 0);
    if (self != None) {
        self->cimage = unique_ptr<CImage_SubBase>(nullptr);
        self->dtype = None;
    }
    return (PyObject *)self;
}

static int PyCImage_init(PyCImage *self, PyObject *args, PyObject *kwargs) {
    PyObject *buffer = NULL;
    Py_ssize_t nin = -1, offset = 0;
    static char *kwlist[] = { "buffer", "dtype", "count", "offset", NULL };
    PyArray_Descr *dtype = NULL;

    if (!PyArg_ParseTupleAndKeywords(args, kwargs,
                "O|O&" NPY_SSIZE_T_PYFMT NPY_SSIZE_T_PYFMT, kwlist,
                &buffer, PyArray_DescrConverter, &dtype, &nin, &offset)) {
        Py_XDECREF(dtype);
        return -1;
    }

    if (self->dtype == None) {
        if (PyArray_Check(buffer)) {
            dtype = PyArray_DESCR(
                reinterpret_cast<PyArrayObject *>(buffer));
        } else {
            dtype = PyArray_DescrFromType(NPY_DEFAULT_TYPE);
        }
    } else {
        //Py_XDECREF(self->dtype);
    }
    Py_INCREF(dtype);
    self->dtype = dtype;

    if (PyArray_Check(buffer)) {
        int tc = (int)self->dtype->type_num;
#define HANDLE(type) \
        self->cimage = CImage_TypePointer<type>(buffer);
        SAFE_SWITCH_ON_TYPECODE(tc, -1);
#undef HANDLE
    }
    
    return 0;
}


static void PyCImage_dealloc(PyCImage *self) {
    Py_XDECREF(self->dtype);
    self->ob_type->tp_free((PyObject *)self);
}


static PyObject     *PyCImage_GET_dtype(PyCImage *self, void *closure) {
    BAIL_WITHOUT(self->dtype);
    Py_INCREF(self->dtype);
    return reinterpret_cast<PyObject *>(self->dtype);
}
static int           PyCImage_SET_dtype(PyCImage *self, PyObject *value, void *closure) {
    if (self->dtype) { Py_DECREF(self->dtype); }
    Py_INCREF(value);
    self->dtype = reinterpret_cast<PyArray_Descr *>(value);
    return 0;
}

static PyGetSetDef PyCImage_getset[] = {
    {
        "dtype",
            (getter)PyCImage_GET_dtype,
            (setter)PyCImage_SET_dtype,
            "Data Type (numpy.dtype)", None},
    SENTINEL
};

static PyObject *PyCImage_Repr(PyCImage *pyim) {
    if (pyim->cimage && pyim->dtype) {
        int tc = (int)pyim->dtype->type_num;
#define HANDLE(type) \
        CImg<type> cim = pyim->recast<type>()->from_pyarray(); \
        return PyString_FromFormat("<PyCImage(%s)[%ix%i, %ix%lubpp] @ %p>", \
            cim.pixel_type(), \
            cim.width(), cim.height(), cim.spectrum(), sizeof(type),\
            pyim);
        SAFE_SWITCH_ON_TYPECODE(tc, PyString_InternFromString("<PyCImage(type out-of-bounds)>"));
#undef HANDLE
    }
    return PyString_InternFromString("<PyCImage(type unknown)>");
}


static Py_ssize_t PyCImage_TypeFlags = Py_TPFLAGS_DEFAULT |
    Py_TPFLAGS_BASETYPE |
    Py_TPFLAGS_HAVE_GETCHARBUFFER;

static PyTypeObject PyCImage_Type = {
    PyObject_HEAD_INIT(NULL)
    0,                                                          /* ob_size */
    "PyImgC.PyCImage",                                          /* tp_name */
    sizeof(PyCImage),                                           /* tp_basicsize */
    0,                                                          /* tp_itemsize */
    (destructor)PyCImage_dealloc,                               /* tp_dealloc */
    0,                                                          /* tp_print */
    0,                                                          /* tp_getattr */
    0,                                                          /* tp_setattr */
    0,                                                          /* tp_compare */
    (reprfunc)PyCImage_Repr,                                    /* tp_repr */
    0,                                                          /* tp_as_number */
    0,                                                          /* tp_as_sequence */
    0,                                                          /* tp_as_mapping */
    0,                                                          /* tp_hash */
    0,                                                          /* tp_call */
    0,                                                          /* tp_str */
    0,                                                          /* tp_getattro */
    0,                                                          /* tp_setattro */
    0,                                                          /* tp_as_buffer */
    PyCImage_TypeFlags,                                         /* tp_flags*/
    "PyImgC object wrapper for CImg instances",                 /* tp_doc */
    0,                                                          /* tp_traverse */
    0,                                                          /* tp_clear */
    0,                                                          /* tp_richcompare */
    0,                                                          /* tp_weaklistoffset */
    0,                                                          /* tp_iter */
    0,                                                          /* tp_iternext */
    0,                                                          /* tp_methods */
    0,                                                          /* tp_members */
    PyCImage_getset,                                            /* tp_getset */
    0,                                                          /* tp_base */
    0,                                                          /* tp_dict */
    0,                                                          /* tp_descr_get */
    0,                                                          /* tp_descr_set */
    0,                                                          /* tp_dictoffset */
    (initproc)PyCImage_init,                                    /* tp_init */
    0,                                                          /* tp_alloc */
    PyCImage_new,                                               /* tp_new */
};

#define PyCImage_Check(op) PyObject_TypeCheck(op, &PyCImage_Type)

static PyObject *Image_new(PyTypeObject *type, PyObject *args, PyObject *kwargs) {
    Image *self;
    self = (Image *)type->tp_alloc(type, 0);
    if (self != None) {
        self->buffer = None;
        self->source = None;
        self->dtype = None;
    }
    return (PyObject *)self;
}

static int Image_init(Image *self, PyObject *args, PyObject *kwargs) {
    PyObject *source=None, *dtype=None, *fake;
    static char *keywords[] = { "source", "dtype", None };

    if (!PyArg_ParseTupleAndKeywords(
        args, kwargs, "|OO",
        keywords,
        &source, &dtype)) { return -1; }

    /// ok to pass nothing
    if (!source && !dtype) { return 0; }

    if (IMGC_PY2) {
        /// Try the legacy buffer interface while it's here
        if (PyObject_CheckReadBuffer(source)) {
            self->buffer = PyBuffer_FromObject(source,
                (Py_ssize_t)0,
                Py_END_OF_BUFFER);
            goto through;
        } else {
            IMGC_TRACE("YO DOGG: legacy buffer check failed");
        }
    }

    /// In the year 3000 the old ways are long gone
    if (PyObject_CheckBuffer(source)) {
        self->buffer = PyMemoryView_FromObject(source);
        goto through;
    } else {
        IMGC_TRACE("YO DOGG: buffer3000 check failed");
    }
    
    /// return before 'through' cuz IT DIDNT WORK DAMNIT
    return 0;
    
through:
    IMGC_TRACE("YO DOGG WERE THROUGH");
    fake = self->source;        Py_INCREF(source);
    self->source = source;      Py_XDECREF(fake);

    if ((source && !self->source) || source != self->source) {
        static PyArray_Descr *descr;

        if (!dtype && PyArray_Check(source)) {
            descr = PyArray_DESCR((PyArrayObject *)source);
        } else if (dtype && !self->dtype) {
            if (!PyArray_DescrConverter(dtype, &descr)) {
                IMGC_TRACE("Couldn't convert dtype arg");
            }
            Py_DECREF(dtype);
        }
    }

    if ((dtype && !self->dtype) || dtype != self->dtype) {
        fake = self->dtype;         Py_INCREF(dtype);
        self->dtype = dtype;        Py_XDECREF(fake);
    }

    return 0;
}

static void Image_dealloc(Image *self) {
    Py_XDECREF(self->buffer);
    Py_XDECREF(self->source);
    Py_XDECREF(self->dtype);
    self->ob_type->tp_free((PyObject *)self);
}

#define Image_members 0

static PyObject     *Image_GET_buffer(Image *self, void *closure) {
    BAIL_WITHOUT(self->buffer);
    Py_INCREF(self->buffer);
    return self->buffer;
}
static int           Image_SET_buffer(Image *self, PyObject *value, void *closure) {
    if (self->buffer) { Py_DECREF(self->buffer); }
    Py_INCREF(value);
    self->buffer = value;
    return 0;
}

static PyObject     *Image_GET_source(Image *self, void *closure) {
    BAIL_WITHOUT(self->source);
    Py_INCREF(self->source);
    return self->source;
}
static int           Image_SET_source(Image *self, PyObject *value, void *closure) {
    if (self->source) { Py_DECREF(self->source); }
    Py_INCREF(value);
    self->source = value;
    return 0;
}

static PyObject     *Image_GET_dtype(Image *self, void *closure) {
    BAIL_WITHOUT(self->dtype);
    Py_INCREF(self->dtype);
    return self->dtype;
}
static int           Image_SET_dtype(Image *self, PyObject *value, void *closure) {
    if (self->dtype) { Py_DECREF(self->dtype); }
    Py_INCREF(value);
    self->dtype = value;
    return 0;
}

static PyGetSetDef Image_getset[] = {
    {
        "buffer",
            (getter)Image_GET_buffer,
            (setter)Image_SET_buffer,
            "Buffer or MemoryView", None},
    {
        "source",
            (getter)Image_GET_source,
            (setter)Image_SET_source,
            "Buffer Source Object", None},
    {
        "dtype",
            (getter)Image_GET_dtype,
            (setter)Image_SET_dtype,
            "Data Type (numpy.dtype)", None},
    SENTINEL
};

static rawbuffer_t *PyImgC_rawbuffer(PyObject *buffer) {

    rawbuffer_t *raw = (rawbuffer_t *)malloc(sizeof(rawbuffer_t));

    if (PyObject_CheckBuffer(buffer)) {
        /// buffer3000
        Py_buffer *buf = 0;
        PyObject_GetBuffer(buffer, buf, PyBUF_SIMPLE); BAIL_WITHOUT(buf);

        raw->len = buf->len;
        raw->buf = buf->buf;
        PyBuffer_Release(buf);

        return raw;
    } else if (PyBuffer_Check(buffer)) {
        /// legacybuf
        PyObject *bufferobj = PyBuffer_FromObject(buffer, (Py_ssize_t)0, Py_END_OF_BUFFER);
        const void *buf = 0;
        Py_ssize_t len;
        PyObject_AsReadBuffer(bufferobj, &buf, &len); BAIL_WITHOUT(buf);

        raw->buf = (void *)buf;
        raw->len = len;
        Py_XDECREF(bufferobj);

        return raw;
    }

    return None;
}

static PyObject *Image_as_ndarray(Image *self) {

    if (self->source && self->dtype) {
        rawbuffer_t *raw = PyImgC_rawbuffer(self->buffer);

        npy_intp *shape = &raw->len;
        PyArray_Descr *descr = 0;
        PyArray_DescrConverter(self->dtype, &descr);
        BAIL_WITHOUT(descr);

        int ndims = 1;
        int typenum = (int)descr->type_num;

        PyObject *ndarray = PyArray_SimpleNewFromData(
            ndims, shape, typenum, raw->buf);
        Py_INCREF(ndarray);

        return (PyObject *)ndarray;
    }

    return None;
}

static PyMethodDef Image_methods[] = {
    {
        "as_ndarray",
            (PyCFunction)Image_as_ndarray,
            METH_NOARGS,
            "Cast to NumPy array"},
    {
        "buffer_info",
            (PyCFunction)PyImgC_PyBufferDict,
            METH_VARARGS,
            "Get buffer info dict"},
    SENTINEL
};

static Py_ssize_t Image_TypeFlags = Py_TPFLAGS_DEFAULT |
    Py_TPFLAGS_BASETYPE |
    Py_TPFLAGS_HAVE_GETCHARBUFFER;

static PyTypeObject Image_Type = {
    PyObject_HEAD_INIT(NULL)
    0,                                                          /* ob_size */
    "PyImgC.Image",                                             /* tp_name */
    sizeof(Image),                                              /* tp_basicsize */
    0,                                                          /* tp_itemsize */
    (destructor)Image_dealloc,                                  /* tp_dealloc */
    0,                                                          /* tp_print */
    0,                                                          /* tp_getattr */
    0,                                                          /* tp_setattr */
    0,                                                          /* tp_compare */
    0,                                                          /* tp_repr */
    0,                                                          /* tp_as_number */
    0,                                                          /* tp_as_sequence */
    0,                                                          /* tp_as_mapping */
    0,                                                          /* tp_hash */
    0,                                                          /* tp_call */
    0,                                                          /* tp_str */
    0,                                                          /* tp_getattro */
    0,                                                          /* tp_setattro */
    0,                                                          /* tp_as_buffer */
    Image_TypeFlags,                                            /* tp_flags*/
    "PyImgC image data container",                              /* tp_doc */
    0,                                                          /* tp_traverse */
    0,                                                          /* tp_clear */
    0,                                                          /* tp_richcompare */
    0,                                                          /* tp_weaklistoffset */
    0,                                                          /* tp_iter */
    0,                                                          /* tp_iternext */
    Image_methods,                                              /* tp_methods */
    Image_members,                                              /* tp_members */
    Image_getset,                                               /* tp_getset */
    0,                                                          /* tp_base */
    0,                                                          /* tp_dict */
    0,                                                          /* tp_descr_get */
    0,                                                          /* tp_descr_set */
    0,                                                          /* tp_dictoffset */
    (initproc)Image_init,                                       /* tp_init */
    0,                                                          /* tp_alloc */
    Image_new,                                                  /* tp_new */
};

#define Image_Check(op) PyObject_TypeCheck(op, &Image_Type)

static PyMethodDef _PyImgC_methods[] = {
    {
        "buffer_info",
            (PyCFunction)PyImgC_PyBufferDict,
            METH_VARARGS | METH_KEYWORDS,
            "Get Py_buffer info dict for an object"},
    {
        "cimage_test",
            (PyCFunction)PyImgC_CImageTest,
            METH_VARARGS | METH_KEYWORDS,
            "<<<<< TEST CIMG CALLS >>>>>"},
    {
        "structcode_parse",
            (PyCFunction)PyImgC_ParseStructCode,
            METH_VARARGS,
            "Parse struct code into list of dtype-string tuples"},
    {
        "structcode_parse_one",
            (PyCFunction)PyImgC_ParseSingleStructAtom,
            METH_VARARGS,
            "Parse unary struct code into a singular dtype string"},
    {
        "structcode_to_numpy_typenum",
            (PyCFunction)PyImgC_NumpyCodeFromStructAtom,
            METH_VARARGS,
            "Parse unary struct code into a NumPy typenum"},
    SENTINEL
};

PyMODINIT_FUNC init_PyImgC(void) {
    PyObject* module;

    if (PyType_Ready(&Image_Type) < 0) { return; }
    if (PyType_Ready(&PyCImage_Type) < 0) { return; }

    /// initialize CImage internals
    //CImage_Register();

    module = Py_InitModule3(
        "pliio._PyImgC",
        _PyImgC_methods,
        "PyImgC buffer interface module");
    if (module == None) { return; }

    /// Bring in NumPy
    import_array();

    /// Set up Image object
    Py_INCREF(&Image_Type);
    Py_INCREF(&PyCImage_Type);
    PyModule_AddObject(
        module, "Image", (PyObject *)&Image_Type);
    PyModule_AddObject(
        module, "PyCImage", (PyObject *)&PyCImage_Type);
}


