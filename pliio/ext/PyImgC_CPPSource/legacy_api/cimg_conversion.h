
#ifndef PyImgC_CIMG_CONVERSION_PLUGIN_H
#define PyImgC_CIMG_CONVERSION_PLUGIN_H

#include "../numpypp/utils.hpp"
#include "../numpypp/dispatch.hpp"



//CImg<T>(const struct CImage_Type<T>&)
//struct CImage_Type<T>;

/// THE IDEA HERE IS:
/// When you have a CImage_Type<T> e.g.
/// CImage_Type<unsigned char> cmtype;
/// ... you can then be all like:
/// CImg<T> = CImage_Type<T>; specifically --
/// cimage = CImage_Type<unsigned char>; or (maybe)

//CImg<T>(const struct CImage_Type<T>&) {}


#endif /// PyImgC_CIMG_CONVERSION_PLUGIN_H