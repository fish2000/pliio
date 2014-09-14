
typedef struct {
    PyObject_HEAD
    PyObject *buffer;
    PyObject *source;
    PyObject *dtype;
} Image;

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
            METH_VARARGS | METH_KEYWORDS,
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
