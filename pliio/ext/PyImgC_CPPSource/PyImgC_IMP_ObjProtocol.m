
#include "PyImgC_IMP_ObjProtocol.h"

/// SMELF ALERT!!!
PyObject *PyCImage_LoadFromFileViaCImg(PyObject *smelf, PyObject *args, PyObject *kwargs) {
    PyCImage *self = reinterpret_cast<PyCImage *>(smelf);
    PyObject *path;
    PyArray_Descr *dtype = NULL;
    static char *keywords[] = { "path", "dtype", NULL };

    if (!PyArg_ParseTupleAndKeywords(args, kwargs,
                "O|O&", keywords,
                &path, PyArray_DescrConverter, &dtype)) {
        PyErr_SetString(PyExc_ValueError,
            "cannot load image (bad argument tuple passed to PyCImage_LoadFromFileViaCImg)");
        return NULL;
    }
    
    /// LOAD THAT SHIT
    if (!PyImgC_PathExists(path)) {
        Py_XDECREF(path);
        PyErr_SetString(PyExc_ValueError,
            "path does not exist");
        return NULL;
    }
    
    /// deal with dtype
    if (!self->dtype) {
        if (!dtype) {
            self->dtype = dtype = numpy::dtype_struct<IMGC_DEFAULT_T>();
        } else {
            self->dtype = dtype;
        }
    }
    
    const char *cpath = PyString_AS_STRING(path);
    
    /// load that shit, dogg
    if (self->dtype) {
        gil_ensure GIL;
        /// Base the loaded CImg struct type and ancilliaries
        /// on whatever is in the dtype we already have
#define HANDLE(type) { \
        try { \
            CImg<IMGC_DEFAULT_T> cim(cpath); \
            self->assign<type>(cim); \
        } catch (CImgArgumentException &err) { \
            Py_XDECREF(dtype); \
            Py_XDECREF(path); \
            PyErr_Format(PyExc_ValueError, \
                "CImg argument error: %.200s", err.what()); \
            return NULL; \
        } catch (CImgIOException &err) { \
            Py_XDECREF(dtype); \
            Py_XDECREF(path); \
            PyErr_Format(PyExc_IOError, \
                "CImg IO error: %.200s", err.what()); \
            return NULL; \
        } \
    }
    SAFE_SWITCH_ON_DTYPE(self->dtype, NULL);
#undef HANDLE
        GIL.~gil_ensure();
    } else if (!self->dtype) {
        /// We don't have a valid dtype - let's make one!
        /// We'll create a CImg<unsigned char> from the file path
        gil_ensure GIL;
        try {
            CImg<IMGC_DEFAULT_T> cim(cpath);
            /// populate our dtype fields and ensconce the new CImg
            self->assign<IMGC_DEFAULT_T>(cim);
            if (!self->dtype) { self->dtype = numpy::dtype_struct<IMGC_DEFAULT_T>(); }
        } catch (CImgArgumentException &err) {
            Py_XDECREF(dtype);
            Py_XDECREF(path);
            PyErr_Format(PyExc_ValueError,
                "CImg argument error: %.200s", err.what());
            return NULL;
        } catch (CImgIOException &err) {
            Py_XDECREF(dtype);
            Py_XDECREF(path);
            PyErr_Format(PyExc_IOError,
                "CImg IO error: %.200s", err.what());
            return NULL;
        }
        GIL.~gil_ensure();
    }
    Py_INCREF(self);
    return reinterpret_cast<PyObject *>(self); /// all is well, return self
}

PyObject *PyCImage_SaveToFileViaCImg(PyObject *smelf, PyObject *args, PyObject *kwargs) {
    PyCImage *self = reinterpret_cast<PyCImage *>(smelf);
    PyObject *path, *pyoverwrite;
    bool overwrite = true;
    bool exists = false;
    static char *keywords[] = { "path", "overwrite", NULL };

    if (!PyArg_ParseTupleAndKeywords(args, kwargs,
                "S|O", keywords,
                &path, &pyoverwrite)) {
        PyErr_SetString(PyExc_ValueError,
            "cannot save image (bad argument tuple passed to PyCImage_SaveToFileViaCImg)");
        return NULL;
    }
    
    //if (pyoverwrite != NULL) { overwrite = PyObject_IsTrue(pyoverwrite); }
    exists = PyImgC_PathExists(path);
    
    /// SAVE THAT SHIT
    if (exists && !overwrite) {
        /// DON'T OVERWRITE
        Py_XDECREF(path);
        PyErr_SetString(PyExc_NameError,
            "path already exists");
        return NULL;
    }
    
    const char *cpath = PyString_AS_STRING(path);
    bool saved = false;
    
    if (exists && overwrite) {
        /// PLEASE DO OVERWRITE
        if (remove(cpath)) {
            gil_release NOGIL;
            saved = self->save(cpath);
            NOGIL.~gil_release();
            if (saved) {
                return reinterpret_cast<PyObject *>(self); /// all is well, return self
            } else {
                Py_XDECREF(path);
                PyErr_SetString(PyExc_OSError,
                    "could not save file");
                return NULL;
            }
        } else {
            Py_XDECREF(path);
            PyErr_SetString(PyExc_SystemError,
                "could not overwrite existing file");
            return NULL;
        }
    }
    
    gil_release NOGIL;
    saved = self->save(cpath);
    NOGIL.~gil_release();
    
    if (saved) {
        return reinterpret_cast<PyObject *>(self); /// all is well, return self
    }
    
    /// we got here, something must be wrong by now
    Py_XDECREF(path);
    PyErr_SetString(PyExc_OSError,
        "could not save file (out of options)");
    return NULL;
}

/// ALLOCATE / __new__ implementation
PyObject *PyCImage_new(PyTypeObject *type, PyObject *args, PyObject *kwargs) {
    PyCImage *self = reinterpret_cast<PyCImage *>(type->tp_alloc(type, 0));
    if (self != None) {
        self->cimage = shared_ptr<CImg_Base>(nullptr);
        self->dtype = NULL;
    }
    return reinterpret_cast<PyObject *>(self); /// all is well, return self
}

/// __repr__ implementations
PyObject *PyCImage_Repr(PyCImage *pyim) {
    if (!pyim->cimage) { return PyString_InternFromString("<PyCImage (empty backing store)>"); }
    int tc = static_cast<int>(pyim->typecode());
    if (pyim->dtype) {
        tc = static_cast<int>(pyim->dtype->type_num);
    }
#define HANDLE(type) { \
    CImg<type> cim = *pyim->recast<type>(); \
    return PyString_FromFormat("<PyCImage (%s|%s) [%ix%i, %ix%lubpp] @ %p>", \
        cim.pixel_type(), typeid(*cim.data()).name(), \
        cim.width(), cim.height(), cim.spectrum(), sizeof(type), \
        pyim); \
    }
    SAFE_SWITCH_ON_TYPECODE(tc, PyString_InternFromString("<PyCImage (bad backing store)>"));
#undef HANDLE
    return PyString_InternFromString("<PyCImage (unmatched typecode)>");
}
const char *PyCImage_ReprCString(PyCImage *pyim) {
    return PyString_AS_STRING(PyCImage_Repr(pyim));
}
string PyCImage_ReprString(PyCImage *pyim) {
    return string(PyCImage_ReprCString(pyim));
}

/// __str__ implementation
PyObject *PyCImage_Str(PyCImage *pyim) {
    if (pyim->cimage && pyim->dtype) {
#define HANDLE(type) { \
        auto cim = pyim->recast<type>(); \
        auto value_string = cim->value_string(); \
        return PyString_FromString((const char *)value_string.data()); \
    }
    SAFE_SWITCH_ON_DTYPE(pyim->dtype, NULL);
#undef HANDLE
    }
    return PyString_FromString("");
}

/// __init__ implementation
int PyCImage_init(PyCImage *self, PyObject *args, PyObject *kwargs) {
    PyObject *buffer = NULL;
    PyArray_Descr *dtype = NULL;
    static char *kwlist[] = { "buffer", "dtype", NULL };
    
    NSLog(@"About to parse tuple and keywords");
    
    if (!PyArg_ParseTupleAndKeywords(args, kwargs,
                "|OO&:dtype", kwlist,
                &buffer, PyArray_DescrConverter, &dtype)) {
            PyErr_SetString(PyExc_ValueError,
                "cannot initialize PyCImage (bad argument tuple)");
        return -1;
    }
    
    NSLog(@"Evaluating dtype");
    
    if (!buffer && !dtype) {
        self->dtype = numpy::dtype_struct<IMGC_DEFAULT_T>();
        Py_INCREF(self->dtype);
        return 0; /// ITS CO0
    }

    if (self->dtype == NULL) {
        if (buffer != NULL) {
            NSLog(@"Non-null buffer, null dtype");
            if (PyArray_Check(buffer)) {
                dtype = PyArray_DESCR(
                    reinterpret_cast<PyArrayObject *>(buffer));
                self->dtype = dtype;
                Py_INCREF(dtype);
            } else {
                NSLog(@"Null buffer, null dtype");
                dtype = PyArray_DescrFromType(
                    IMGC_DEFAULT_TYPECODE);
                self->dtype = dtype;
                Py_INCREF(dtype);
            }
        } else if (dtype != NULL) {
            NSLog(@"Null buffer, non-null passed dtype");
            if (PyArray_DescrCheck(dtype)) {
                self->dtype = dtype;
                Py_INCREF(dtype);
            } else {
                /// SHOULD NEVER HAPPEN
                PyErr_SetString(PyExc_ValueError,
                    "PyCImage __init__ TOTAL FREAKOUT");
                return -1;
            }
        }
    } else {
        if (dtype != NULL) {
            if (PyArray_DescrCheck(dtype)) {
                Py_XDECREF(self->dtype);
                self->dtype = dtype;
                Py_INCREF(dtype);
            }
        } else {
            /// WE DONT BELIEVE IN NOSING LEBAWSKIIIII
            self->dtype = numpy::dtype_struct<IMGC_DEFAULT_T>();
            Py_INCREF(self->dtype);
        }
    }
    
    NSLog(@"Evaluating buffer");
    
    if (!buffer) {
        /// GET OUT NOW BEFORE... THE BUFFERINGING
        return 0;
    }
    
    // if (PyUnicode_Check(buffer)) {
    //     /// Fuck, it's unicode... let's de-UTF8 it
    //     buffer = PyUnicode_AsUTF8String(buffer);
    // }
    
    if (PyString_Check(buffer)) {
        /// it's a path string, load (with CImg.h) -- DISPATCH!!!!
        NSLog(@"Dispatching to PyCImage_LoadFromFileViaCImg");
        PyObject *out = PyObject_CallMethodObjArgs(
            reinterpret_cast<PyObject *>(self),
            PyString_InternFromString("load"),
            buffer, self->dtype, NULL);
        if (!out) {
            PyErr_Format(PyExc_ValueError,
                "CImg failed to load from path: %s",
                PyString_AS_STRING(buffer));
            return -1;
        }
        return 0;
    }
    
    NSLog(@"About to check/assign with a NumPy buffer");
    
    if (PyArray_Check(buffer)) {
        /// it's a numpy array
#define HANDLE(type) self->assign(CImg<type>(buffer));
        SAFE_SWITCH_ON_DTYPE(self->dtype, -1);
#undef HANDLE
        return 0;
    }
    
    /// FALLING OFF (DANGER WILL ROBINSON)
    PyErr_SetString(PyExc_ValueError,
        "NOTHING HAPPENED! Buffer: NOPE.");
    return -1;
}

/// DEALLOCATE
void PyCImage_dealloc(PyCImage *self) {
    Py_XDECREF(self->dtype);
    self->cleanup();
    self->ob_type->tp_free((PyObject *)self);
}
