
#ifndef PyImgC_CIMG_PYBUFFER_PLUGIN_H
#define PyImgC_CIMG_PYBUFFER_PLUGIN_H

// Check if this CImg<T> instance and a given Py_buffer* have identical pixel types.
bool not_structcode_of(const Py_buffer *const pybuffer) const {
    if (pybuffer->format) {
        unsigned int typecode = structcode_to_typecode(pybuffer->format);
        return TYPECODE_NOT(typecode);
    }
    return false;
}

//----------------------------
// Py_buffer-to-CImg conversion
//----------------------------
/// Copy constructor
CImg(const Py_buffer *const pybuffer):_width(0),_height(0),_depth(0),_spectrum(0),_is_shared(true),_data(0) {
    assign(pybuffer);
}
/// Copy constructor with width and height specified
CImg(const Py_buffer *const pybuffer, const int width = 0, const int height = 0):_depth(0),_spectrum(0),_is_shared(true),_data(0) {
    assign(pybuffer, width, height);
}

CImg(Py_buffer *pybuffer):_width(0),_height(0),_depth(0),_spectrum(0),_is_shared(true),_data(0) {
    assign(const_cast<Py_buffer *>(pybuffer));
}
/// Copy constructor with width and height specified
CImg(Py_buffer *pybuffer, int width=0, int height=0):_depth(0),_spectrum(0),_is_shared(true),_data(0) {
    assign(
        const_cast<Py_buffer *>(pybuffer),
        const_cast<int&>(width),
        const_cast<int&>(height));
}

// In-place constructor
CImg<T> &assign(const Py_buffer *const pybuffer, const int width=0, const int height=0) {
    if (!pybuffer) return assign();
    if (not_structcode_of(pybuffer)) {
        throw CImgInstanceException(_cimg_instance
                                    "assign(const Py_buffer*) : Buffer structcode has no corresponding pixel type.",
                                    cimg_instance);
    }
    if (!pybuffer->ndim) { return assign(); }
    
    const char *const dataPtrI = const_cast<const char *const>(
        static_cast<char *>(pybuffer->buf));
    int nChannels = 1, W, H, WH;
    
    if (pybuffer->ndim > 2) { nChannels = static_cast<int>(pybuffer->shape[2]); }
    if (pybuffer->ndim > 1) {
        W = width ? width : static_cast<int>(pybuffer->shape[1]);
        H = height ? height : static_cast<int>(pybuffer->shape[0]);
    } else {
        /// fuck
        WH = static_cast<int>(lrint(sqrt(pybuffer->len / pybuffer->itemsize)));
        W = width ? width : WH;
        H = height ? height : WH;
    }

    assign(dataPtrI,
        const_cast<int&>(W),
        const_cast<int&>(H), 1,
        const_cast<int&>(nChannels), true);
    
    return *this;
}

//----------------------------
// CImg-to-NumPy conversion
//----------------------------
// z is the z-coordinate of the CImg slice that one wants to copy.
void get_pybuffer(Py_buffer *pybuffer, const unsigned z=0, const bool readonly=true) const {
    const char *structcode_char = structcode();
    if (!structcode_char) {
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
    
    Py_ssize_t raw_buffer_size = static_cast<Py_ssize_t>(datasize());
    
    pybuffer->buf = static_cast<T*>(_data);
    pybuffer->format = const_cast<char *>(structcode_char);  /// for now (do we give fucks re:byte order?)
    pybuffer->ndim = 3;                                      /// for now
    pybuffer->len = raw_buffer_size;
    
    pybuffer->shape = (Py_ssize_t *)PyMem_Malloc(sizeof(Py_ssize_t) * 3);
    pybuffer->shape[0] = (Py_ssize_t)height();
    pybuffer->shape[1] = (Py_ssize_t)width();
    pybuffer->shape[2] = (Py_ssize_t)spectrum();
    
    pybuffer->strides = (Py_ssize_t *)PyMem_Malloc(sizeof(Py_ssize_t) * 3);
    pybuffer->strides[0] = (Py_ssize_t)(width() * spectrum() * sizeof(T));
    pybuffer->strides[1] = (Py_ssize_t)(spectrum() * sizeof(T));
    pybuffer->strides[2] = (Py_ssize_t)sizeof(T);
    
    pybuffer->itemsize = sizeof(T);
    
    pybuffer->readonly = readonly;
    pybuffer->internal = IMGC_PYBUFFER_MALLOC;
    pybuffer->suboffsets = NULL;
    pybuffer->obj = NULL;
}

#endif /// PyImgC_CIMG_PYBUFFER_PLUGIN_H
