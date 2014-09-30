
#ifndef PyImgC_CIMG_NUMPY_PLUGIN_H
#define PyImgC_CIMG_NUMPY_PLUGIN_H

// Check if this CImg<T> instance and a given PyObject* have identical pixel types.
bool not_typecode_of(const PyObject *const pyobject) const {
    const PyArrayObject *pyarray = reinterpret_cast<const PyArrayObject *>(pyobject);
    unsigned int typecode = static_cast<unsigned int>(PyArray_TYPE(pyarray));
    return TYPECODE_NOT(typecode);
}
bool not_typecode_of(const PyArrayObject *const pyarray) const {
    unsigned int typecode = static_cast<unsigned int>(PyArray_TYPE(pyarray));
    return TYPECODE_NOT(typecode);
}

//----------------------------
// NumPy-to-CImg conversion
//----------------------------
// Copy constructor
CImg(const PyObject *const pyobject, const int W = 0, const int H = 0):_depth(0),_spectrum(0),_is_shared(false),_data(0) {
    assign(pyobject, W, H);
}

/// In-place (PyObject *) dispatch constructor
CImg<T> &assign(const PyObject *const pyobject, const int W, const int H) {
    if (PyArray_Check(pyobject)) {
        /// It's a numpy array
        return assign(
            reinterpret_cast<const PyArrayObject *const>(pyobject));
    }
    if (PyBuffer_Check(pyobject)) {
        /// It's a legacy buffer object
        if (W == 0 && H == 0) {
            throw CImgInstanceException(_cimg_instance
                                        "assign(const PyObject*) : Legacy buffer constructor requires specifying width and height",
                                        cimg_instance);
        }
        return assign_legacybuf_data(pyobject, W, H);
    }
    return assign_legacybuf_data(pyobject, W, H);
}

/// In-place custom constructor for (PyBufferObject *)
/// ONLY ITS NOT THAT ITS JUST A PYOBJECT DURRRRR
CImg<T> &assign_legacybuf_data(const PyObject *const pylegacybuf, const int W, const int H) {
    if (!pylegacybuf) { return assign(); }
    if (typeid(T) != typeid(char)) {
        /// Legacy buffers only support const char* 
        throw CImgInstanceException(_cimg_instance
                                    "assign(const PyBufferObject*) : Legacy buffers convert only to CImg<char> or similar",
                                    cimg_instance);
    }
    if (!PyBuffer_Check(pylegacybuf)) {
        throw CImgInstanceException(_cimg_instance
                                    "assign(const PyBufferObject*) : Bad buffer object (legacy PyBufferObject interface)",
                                    cimg_instance);
    }
    return *this;
}

/// In-place constructor overload for (PyArrayObject *)
CImg<T> &assign(const PyArrayObject *const pyarray, const int width = 0, const int height = 0) {
    if (!pyarray) { return assign(); }
    if (!PyArray_Check(pyarray)) {
        throw CImgInstanceException(_cimg_instance
                                    "assign(const PyArrayObject*) : Invalid NumPy array",
                                    cimg_instance);
    }
    if (not_typecode_of(pyarray)) {
        throw CImgInstanceException(_cimg_instance
                                    "assign(const PyArrayObject*) : NumPy array has no corresponding pixel type",
                                    cimg_instance);
    }
    
    if (width > 0 && height > 0) {
        /// INSERT RESHAPERY HERE
    }
    const int W = (int)PyArray_DIM(pyarray, 1), H = (int)PyArray_DIM(pyarray, 0);
    
    unsigned char *dataPtrI = (unsigned char *)PyArray_DATA(const_cast<PyArrayObject *>(pyarray));
    const int nChannels = (int)PyArray_DIM(pyarray, 2);

    assign(const_cast<unsigned char *>(dataPtrI), W, H, 1, nChannels, true);
    Py_INCREF(pyarray);
    return *this;
}

//----------------------------
// CImg-to-NumPy conversion
//----------------------------
// z is the z-coordinate of the CImg slice that one wants to copy.
PyObject* get_pyarray(const unsigned z=0) const {
    const int typecode_int = typecode();
    if (!typecode_int) {
        throw CImgInstanceException(_cimg_instance
                                  "get_pyarray() : No NPY_TYPES definition for pixel type: %s",
                                  cimg_instance,
                                  pixel_type());
    }
    if (is_empty()) {
        throw CImgArgumentException(_cimg_instance
                                    "get_pyarray() : Called with empty CImg instance",
                                    cimg_instance);
    }
    
    if (z >= _depth) {
        throw CImgInstanceException(_cimg_instance
                                    "get_pyarray() : Instance lacks Z-dimension: %u",
                                    cimg_instance,
                                    z);
    }
    if (_spectrum > 4) {
        cimg::warn(_cimg_instance
                   "get_pyarray() : Most NumPy image schemas support up to 4 channels -- higher-order channels won't be copied",
                   cimg_instance);
    }
  
    npy_intp dims[3] = {
        (npy_intp)_height,
        (npy_intp)_width,
        (npy_intp)_spectrum
    };
    PyObject *pyarray = PyArray_SimpleNewFromData(3, dims, typecode_int, (void *)_data);
    Py_INCREF(pyarray);
    return pyarray;
}

//----------------------------
// CImg-to-LegacyBuf conversion
//----------------------------
// z is the z-coordinate of the CImg slice that one wants to copy.
PyObject* get_legacybuffer(const unsigned z=0, const bool readonly=true) const {
    if (is_empty()) {
        throw CImgArgumentException(_cimg_instance
                                    "get_pyarray() : Called with empty CImg instance",
                                    cimg_instance);
    }
    
    if (z >= _depth) {
        throw CImgInstanceException(_cimg_instance
                                    "get_pyarray() : Instance lacks Z-dimension: %u",
                                    cimg_instance,
                                    z);
    }
    
    PyObject *pylegacybuf;
    if (readonly) {
        pylegacybuf = PyBuffer_FromMemory(
            (void *)_data, datasize());
    } else {
        *pylegacybuf = PyBuffer_FromReadWriteMemory(
            (void *)_data, datasize());
    }
    Py_INCREF(pylegacybuf);
    return pylegacybuf;
}

#endif /// PyImgC_CIMG_NUMPY_PLUGIN_H
