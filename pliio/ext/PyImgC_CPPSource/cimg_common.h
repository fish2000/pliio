
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

#define STRUCTCODE_CHAR_NOT(structcode_char) \
       ((structcode_char == NPY_BOOLLTR         && typeid(T) != typeid(bool)) || \
        (structcode_char == NPY_BYTELTR         && typeid(T) != typeid(char)) || \
        (structcode_char == NPY_HALFLTR         && typeid(T) != typeid(npy_half)) || \
        (structcode_char == NPY_SHORTLTR        && typeid(T) != typeid(short)) || \
        (structcode_char == NPY_INTLTR          && typeid(T) != typeid(int)) || \
        (structcode_char == NPY_LONGLTR         && typeid(T) != typeid(long)) || \
        (structcode_char == NPY_LONGLONGLTR     && typeid(T) != typeid(long long)) || \
        (structcode_char == NPY_UBYTELTR        && typeid(T) != typeid(unsigned char)) || \
        (structcode_char == NPY_USHORTLTR       && typeid(T) != typeid(unsigned short)) || \
        (structcode_char == NPY_UINTLTR         && typeid(T) != typeid(unsigned int)) || \
        (structcode_char == NPY_ULONGLTR        && typeid(T) != typeid(unsigned long)) || \
        (structcode_char == NPY_ULONGLONGLTR    && typeid(T) != typeid(unsigned long long)) || \
        (structcode_char == NPY_CFLOATLTR       && typeid(T) != typeid(std::complex<float>)) || \
        (structcode_char == NPY_CLONGDOUBLELTR  && typeid(T) != typeid(std::complex<long double>)) || \
        (structcode_char == NPY_CDOUBLELTR      && typeid(T) != typeid(std::complex<double>)) || \
        (structcode_char == NPY_FLOATLTR        && typeid(T) != typeid(float)) || \
        (structcode_char == NPY_LONGDOUBLELTR   && typeid(T) != typeid(long double)) || \
        (structcode_char == NPY_DOUBLELTR       && typeid(T) != typeid(double)))

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

int rowbytes() const {
    return static_cast<int>(width() * spectrum()) * sizeof(T);
}


std::array<long, 3> shape() const {
    return {{
        (long)height(),
        (long)width(),
        (long)spectrum()
    }};
}

std::array<long, 4> shape3D() const {
    return {{
        (long)height(),
        (long)width(),
        (long)depth(),
        (long)spectrum()
    }};
}

#define shape2D() shape()

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

#endif /// PyImgC_CIMG_COMMON_PLUGIN_H