
#ifndef PyImgC_OPTIONS_H
#define PyImgC_OPTIONS_H

#define cimg_OS 1                       /// unix-like
#define cimg_verbosity 1                /// log to the console
#define cimg_display 0                  /// don't need this

#define cimg_use_jpeg 1                 /// jpeg
#define cimg_use_zlib 1                 /// compressed output

#ifndef cimg_imagepath
#define cimg_imagepath "cimg/img/"
#endif

//#define cimg_use_png 1                /// png (via setup.py)
//#define cimg_use_tiff 1               /// tiff (via setup.py)
//#define cimg_use_magick 1             /// ImageMagick++ I/O (via setup.py)
//#define cimg_use_fftw3 1              /// libFFTW3 (via setup.py)
//#define cimg_use_openexr 1            /// OpenEXR (via setup.py)
//#define cimg_use_lapack 1             /// LAPACK

#define cimg_plugin1 "../cimg_common.h"
#define cimg_plugin2 "../cimg_numpy.h"
#define cimg_plugin3 "../cimg_pybuffer.h"

#endif /// PyImgC_OPTIONS_H