
#ifdef __OBJC__
#import <Cocoa/Cocoa.h>
#import <Foundation/Foundation.h>
#import <Accelerate/Accelerate.h>

#ifndef PyImgC_CIMG_VIMAGEBUFFER_PLUGIN_H
#define PyImgC_CIMG_VIMAGEBUFFER_PLUGIN_H

//----------------------------
// vImage_Buffer-to-CImg conversion
//----------------------------
/// Copy constructor
CImg(const vImage_Buffer, *const vbuf):_width(0),_height(0),_depth(0),_spectrum(0),_is_shared(true),_data(0) {
    assign(vbuf);
}
/// Copy constructor with width and height specified
CImg(const vImage_Buffer *const vbuf, const int width=0, const int height=0):_depth(0),_spectrum(0),_is_shared(true),_data(0) {
    assign(vbuf, width, height);
}

CImg(vImage_Buffer *vbuf):_width(0),_height(0),_depth(0),_spectrum(0),_is_shared(true),_data(0) {
    assign(const_cast<vImage_Buffer *>(vbuf));
}
/// Copy constructor with width and height specified
CImg(vImage_Buffer *vbuf, int width=0, int height=0):_depth(0),_spectrum(0),_is_shared(true),_data(0) {
    assign(
        const_cast<vImage_Buffer *>(vbuf),
        const_cast<vImagePixelCount &>(width),
        const_cast<vImagePixelCount &>(height));
}

// In-place constructor
CImg<T> &assign(const vImage_Buffer *const vbuf, const vImagePixelCount width=0, const vImagePixelCount height=0) {
    if (!vbuf) { return assign(); }
    
    const char *const dataPtrI = const_cast<const char *const>(
        static_cast<char *>(vbuf->data));
    
    vImagePixelCount nChannels = 4L,
        W = width ? width : vbuf->width,
        H = height ? height : vbuf->height;

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
void get_pybuffer(Py_buffer *vbuf, const unsigned z=0, const bool readonly=true) const {
    //const char *structcode_char = structcode();
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
    
    pybuffer->shape = (Py_ssize_t *)malloc(sizeof(Py_ssize_t) * 3);
    pybuffer->shape[0] = (Py_ssize_t)height();
    pybuffer->shape[1] = (Py_ssize_t)width();
    pybuffer->shape[2] = (Py_ssize_t)spectrum();
    
    pybuffer->strides = (Py_ssize_t *)malloc(sizeof(Py_ssize_t) * 3);
    pybuffer->strides[0] = (Py_ssize_t)(width() * spectrum() * sizeof(T));
    pybuffer->strides[1] = (Py_ssize_t)(spectrum() * sizeof(T));
    pybuffer->strides[2] = (Py_ssize_t)sizeof(T);
    
    pybuffer->itemsize = sizeof(T);
    
    pybuffer->readonly = readonly;
    pybuffer->internal = NULL;                               /// for now
    pybuffer->suboffsets = NULL;
    pybuffer->obj = NULL;
}

#endif /// PyImgC_CIMG_VIMAGEBUFFER_PLUGIN_H

#endif