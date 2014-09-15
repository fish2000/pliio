
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

std::array<Py_ssize_t, 3> shape2D() const {
    return {{
        static_cast<Py_ssize_t>(_height),
        static_cast<Py_ssize_t>(_width),
        static_cast<Py_ssize_t>(_spectrum)
    }};
}

std::array<Py_ssize_t, 4> shape3D() const {
    return {{
        static_cast<Py_ssize_t>(_height),
        static_cast<Py_ssize_t>(_width),
        static_cast<Py_ssize_t>(_depth),
        static_cast<Py_ssize_t>(_spectrum)
    }};
}

#ifndef shape
#define shape() shape2D()
#endif

/// structcode parser invocation (from pyimgc.cpp)
const char *structcode_to_dtype(const char *structcode, bool include_byteorder=true) {
    std::vector<pair<std::string, std::string>> pairvec = structcode::parse(std::string(structcode));
    std::string byteorder = "=";

    if (!pairvec.size()) {
        throw CImgInstanceException(_cimg_instance
                                    "Structcode std::string parsed to zero-length pair vector",
                                    cimg_instance);
    }

    /// get special values
    for (size_t idx = 0; idx < pairvec.size(); idx++) {
        if (pairvec[idx].first == "__byteorder__") {
            byteorder = std::string(pairvec[idx].second);
            pairvec.erase(pairvec.begin()+idx);
        }
    }

    /// Get singular value
    if (include_byteorder) {
        return std::string(byteorder + pairvec[0].second).c_str();
    }
    return std::string(pairvec[0].second).c_str();
}

unsigned int structcode_to_typecode(const char *structcode) {
    const char *dtypecode = structcode_to_dtype(structcode);
    PyArray_Descr *descr;
    int npy_type_num = 0;

    if (!dtypecode) {
        throw CImgInstanceException(_cimg_instance
                                    "Cannot get structcode std::string (bad argument)",
                                    cimg_instance);
    }

    if (!PyArray_DescrConverter(PyString_FromString(dtypecode), &descr)) {
        throw CImgInstanceException(_cimg_instance
                                    "cannot convert std::string to PyArray_Descr",
                                    cimg_instance);
    }

    npy_type_num = (unsigned int)descr->type_num;
    Py_XDECREF(descr);

    return npy_type_num;
}




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