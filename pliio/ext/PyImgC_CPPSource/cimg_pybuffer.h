
#ifndef PyImgC_CIMG_PYBUFFER_PLUGIN_H
#define PyImgC_CIMG_PYBUFFER_PLUGIN_H

#include <Python.h>

// Check if this CImg<T> instance and a given Py_buffer* have identical pixel types.
bool not_structcode_of(const Py_buffer *const pybuffer) const {
    if (pybuffer->format) {
        unsigned int typecode = structcode_to_typecode(pybuffer->format);
        return ((typecode == NPY_UBYTE      && typeid(T) != typeid(unsigned char)) ||
              (typecode == NPY_CHAR         && typeid(T) != typeid(char)) ||
              (typecode == NPY_USHORT       && typeid(T) != typeid(unsigned short)) ||
              (typecode == NPY_UINT         && typeid(T) != typeid(unsigned int)) ||
              (typecode == NPY_INT          && typeid(T) != typeid(int)) ||
              (typecode == NPY_FLOAT        && typeid(T) != typeid(float)) ||
              (typecode == NPY_DOUBLE       && typeid(T) != typeid(double)));
    }
    return false;
}

// Given this CImg<T> instance, return the corresponding structcode character
const char *pybuffer_structcode() const {
    if (typeid(T) == typeid(unsigned char))  return "B";
    if (typeid(T) == typeid(signed char))    return "b";
    if (typeid(T) == typeid(char))           return "c";
    if (typeid(T) == typeid(unsigned short)) return "H";
    if (typeid(T) == typeid(short))          return "h";
    if (typeid(T) == typeid(int))            return "i";
    if (typeid(T) == typeid(float))          return "f";
    if (typeid(T) == typeid(double))         return "d";
    return 0;
}

//----------------------------
// Py_buffer-to-CImg conversion
//----------------------------
// Copy constructor
CImg(const Py_buffer *const pybuffer):_width(0),_height(0),_depth(0),_spectrum(0),_is_shared(false),_data(0) {
    assign(pybuffer);
}

// In-place constructor
CImg<T> &assign(const Py_buffer *const pybuffer) {
    if (!pybuffer) return assign();
    if (not_structcode_of(pybuffer)) {
        throw CImgInstanceException(_cimg_instance
                                    "assign(const Py_buffer*) : Buffer structcode has no corresponding pixel type.",
                                    cimg_instance);
    }
    
    //pybuffer->len;
    //pybuffer->itemsize;
    
    const char *const dataPtrI = const_cast<const char *const>(static_cast<char *>(pybuffer->buf));
    int nChannels = 1, W, H, WH;
    
    if (!pybuffer->ndim) {
        return assign();
    }
    
    // for (int idx = 0; idx < (int)buf->ndim; idx++) {
    //     pybuffer->shape[idx];
    //     pybuffer->strides[idx];
    //     pybuffer->suboffsets[idx];
    // }
    
    if (pybuffer->ndim > 2) { nChannels = (int)pybuffer->shape[2]; }
    if (pybuffer->ndim > 1) {
        W = (int)pybuffer->shape[1];
        H = (int)pybuffer->shape[0];
    } else {
        /// fuck
        WH = (int)lrint(sqrt(pybuffer->len / pybuffer->itemsize));
        W = WH;
        H = WH;
    }

    assign(dataPtrI, const_cast<int&>(W), const_cast<int&>(H), 1, const_cast<int&>(nChannels));
    PyBuffer_Release(const_cast<Py_buffer *>(pybuffer));
    return *this;
}

//----------------------------
// CImg-to-NumPy conversion
//----------------------------
// z is the z-coordinate of the CImg slice that one wants to copy.
Py_buffer get_pybuffer(const unsigned z=0) const {
    const char *structcode = pybuffer_structcode();
    if (!structcode) {
        throw CImgInstanceException(_cimg_instance
                                  "get_pybuffer() : no corresponding structcode for CImg type.",
                                  cimg_instance);
    }
    if (is_empty()) {
        throw CImgArgumentException(_cimg_instance
                                    "get_pybuffer() : Empty CImg instance.",
                                    cimg_instance);
    }
    
    if (z >= _depth) {
        throw CImgInstanceException(_cimg_instance
                                    "get_pybuffer() : Instance has not Z-dimension %u.",
                                    cimg_instance,
                                    z);
    }
    if (_spectrum > 4) {
        cimg::warn(_cimg_instance
                   "get_pybuffer() : Most image libraries don't support >4 channels -- higher-order dimensions will be ignored.",
                   cimg_instance);
    }
    
    Py_buffer pybuffer;
    Py_ssize_t raw_buffer_size = (Py_ssize_t)(_width * _height * _spectrum * sizeof(T));
    int was_buffer_filled = PyBuffer_FillInfo(
        &pybuffer, NULL,            /// Output struct ref, and null PyObject ptr
        static_cast<T*>(_data),     /// Input raw-data ptr
        raw_buffer_size,            /// Size of *_data in bytes 
        1,                          /// Buffer is read-only
        PyBUF_F_CONTIGUOUS);        /// I *think* CImg instances are fortran-style 
    
    if (was_buffer_filled < 0) {
        throw CImgArgumentException(_cimg_instance
                                    "get_pybuffer() : PyBuffer_FillInfo() returned an error",
                                    cimg_instance);
    }
    
    pybuffer.format = const_cast<char *>(structcode);
    pybuffer.ndim = 3;              /// for now
    Py_ssize_t shape[3] = {
        (Py_ssize_t)_height,
        (Py_ssize_t)_width,
        (Py_ssize_t)_spectrum
    };
    pybuffer.shape = shape;
    return pybuffer;
}

#endif /// PyImgC_CIMG_PYBUFFER_PLUGIN_H
