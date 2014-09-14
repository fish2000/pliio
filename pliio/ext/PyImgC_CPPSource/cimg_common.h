
#ifndef PyImgC_CIMG_COMMON_PLUGIN_H
#define PyImgC_CIMG_COMMON_PLUGIN_H

#define TYPECODE_NOT(typecode) \
       ((typecode == NPY_BOOL         && typeid(T) != typeid(bool)) || \
        (typecode == NPY_BYTE         && typeid(T) != typeid(char)) || \
        (typecode == NPY_HALF         && typeid(T) != typeid(npy_half)) || \
        (typecode == NPY_SHORT        && typeid(T) != typeid(short)) || \
        (typecode == NPY_INT          && typeid(T) != typeid(int)) || \
        (typecode == NPY_LONG         && typeid(T) != typeid(long)) || \
        (typecode == NPY_LONGLONG     && typeid(T) != typeid(long long)) || \
        (typecode == NPY_UBYTE        && typeid(T) != typeid(unsigned char)) || \
        (typecode == NPY_USHORT       && typeid(T) != typeid(unsigned short)) || \
        (typecode == NPY_UINT         && typeid(T) != typeid(unsigned int)) || \
        (typecode == NPY_ULONG        && typeid(T) != typeid(unsigned long)) || \
        (typecode == NPY_ULONGLONG    && typeid(T) != typeid(unsigned long long)) || \
        (typecode == NPY_CFLOAT       && typeid(T) != typeid(std::complex<float>)) || \
        (typecode == NPY_CLONGDOUBLE  && typeid(T) != typeid(std::complex<long double>)) || \
        (typecode == NPY_CDOUBLE      && typeid(T) != typeid(std::complex<double>)) || \
        (typecode == NPY_FLOAT        && typeid(T) != typeid(float)) || \
        (typecode == NPY_LONGDOUBLE   && typeid(T) != typeid(long double)) || \
        (typecode == NPY_DOUBLE       && typeid(T) != typeid(double)))

/// Given this CImg<T> instance, return the corresponding structcode character --
/// from inside the CImg<T> struct this is way faster than using structcode::parse()
const char *structcode() const {
    if (typeid(T) == typeid(bool))                      return "?";
    if (typeid(T) == typeid(char))                      return "b";
    if (typeid(T) == typeid(signed char))               return "b";
    if (typeid(T) == typeid(npy_half))                  return "e";
    if (typeid(T) == typeid(short))                     return "h";
    if (typeid(T) == typeid(int))                       return "i";
    if (typeid(T) == typeid(long))                      return "l";
    if (typeid(T) == typeid(long long))                 return "q";
    if (typeid(T) == typeid(unsigned char))             return "B";
    if (typeid(T) == typeid(unsigned short))            return "H";
    if (typeid(T) == typeid(unsigned int))              return "I";
    if (typeid(T) == typeid(unsigned long))             return "L";
    if (typeid(T) == typeid(unsigned long long))        return "Q";
    if (typeid(T) == typeid(std::complex<float>))       return "f";
    if (typeid(T) == typeid(std::complex<double>))      return "d";
    if (typeid(T) == typeid(std::complex<long double>)) return "g";
    if (typeid(T) == typeid(float))                     return "f";
    if (typeid(T) == typeid(double))                    return "d";
    if (typeid(T) == typeid(long double))               return "g";
    return 0;
}

int typecode() const {
    return numpy::dtype_code<T>();
}

PyArray_Descr *typestruct() const {
    return numpy::dtype_struct<T>();
}

int datasize() const {
    return static_cast<int>(size()) * sizeof(T);
}

Py_ssize_t *shape2D() const {
    Py_ssize_t shape[3] = {
        (Py_ssize_t)_height,
        (Py_ssize_t)_width,
        (Py_ssize_t)_spectrum
    };
    return shape;
}

Py_ssize_t *shape3D() const {
    Py_ssize_t shape[4] = {
        (Py_ssize_t)_height,
        (Py_ssize_t)_width,
        (Py_ssize_t)_depth,
        (Py_ssize_t)_spectrum
    };
    return shape;
}

#ifndef shape
#define shape() shape2D()
#endif





//CImg<T>(const struct CImage_Type<T>&)
//struct CImage_Type<T>;

/// THE IDEA HERE IS:
/// When you have a CImage_Type<T> e.g.
/// CImage_Type<unsigned char> cmtype;
/// ... you can then be all like:
/// CImg<T> = CImage_Type<T>; specifically --
/// cimage = CImage_Type<unsigned char>; or (maybe)

//CImg<T>(const struct CImage_Type<T>&) {}


#endif /// PyImgC_CIMG_COMMON_PLUGIN_H