#ifndef PyImgC_CIMAGE_H
#define PyImgC_CIMAGE_H

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

#define cimg_plugin1 "../cimg_numpy.h"

#include <map>
#include <cstdlib>
#include <typeinfo>
#include <type_traits>
#include <Python.h>
#include <structmember.h>

#include <numpy/ndarrayobject.h>
#include <numpy/ndarraytypes.h>
#include "numpypp/numpy.hpp"

#include "cimg/CImg.h"
using namespace cimg_library;
using namespace std;

template <typename T>
CImg<T> cimage_from_pybuffer(Py_buffer *pybuffer, int sW, int sH,
                    int channels, bool is_shared=true) {
    CImg<T> view(pybuffer->buf,
        sW, sH, 1,
        channels, is_shared);
    return view;
}

template <typename T>
CImg<T> cimage_from_pybuffer(Py_buffer *pybuffer, bool is_shared=true) {
    CImg<T> view(
        pybuffer->buf,
        pybuffer->shape[1],
        pybuffer->shape[0],
        1, 3, is_shared);
    return view;
}

template <typename T>
CImg<T> cimage_from_pyarray(PyArrayObject *pyarray, bool is_shared=true) {
    int sW = 0;
    int sH = 0;
    int channels = 0;
    switch (PyArray_NDIM(pyarray)) {
        case 3:
        {
            channels = PyArray_DIM(pyarray, 2); /// starts from zero, right?...
            sW = PyArray_DIM(pyarray, 1);
            sH = PyArray_DIM(pyarray, 0);
        }
        break;
        case 2:
        {
            channels = 1;
            sW = PyArray_DIM(pyarray, 1);
            sH = PyArray_DIM(pyarray, 0);
        }
        break;
        default:
        {
            return CImg<T>();
        }
        break;
    }
    CImg<T> view(
        numpy::ndarray_cast<T*>(pyarray),
        sW, sH,
        1, channels, is_shared);
    return view;
}

template <typename T>
CImg<T> cimage_from_pyarray(PyObject *pyobj, bool is_shared=true) {
    if (!PyArray_Check(pyobj)) {
        return CImg<T>();
    }
    PyArrayObject *pyarray = reinterpret_cast<PyArrayObject *>(pyobj);
    return cimage_from_pyarray<T>(pyarray, is_shared);
}

template <typename T>
CImg<T> cimage_from_pyobject(PyObject *datasource, int sW, int sH,
                    int channels, bool is_shared=true) {
    CImg<T> view(sW, sH, 1, channels, is_shared);
    return view;
}

template <typename T>
CImg<T> cimage_from_pyobject(PyObject *datasource, bool is_shared=true) {
    CImg<T> view(640, 480, 1, 3, is_shared);
    return view;
}

#define NILCODE '~'

struct CImage_SubBase {
    virtual ~CImage_SubBase() {};
};

template <typename dT>
struct CImage_Traits;

template <typename dT>
struct CImage_Base : public CImage_SubBase {
    typedef typename CImage_Traits<dT>::value_type value_type;

    CImg<value_type> from_pybuffer(Py_buffer *pybuffer, bool is_shared=true) {
        return cimage_from_pybuffer<value_type>(pybuffer, is_shared);
    }

    CImg<value_type> from_pybuffer_with_dims(Py_buffer *pybuffer,
        int sW, int sH, int channels=3,
        bool is_shared=true) {
        return cimage_from_pybuffer<value_type>(pybuffer, sW, sH, channels, is_shared);
    }

    CImg<value_type> from_pyarray(PyArrayObject *pyarray, bool is_shared=true) {
        return cimage_from_pyarray<value_type>(pyarray, is_shared);
    }

    CImg<value_type> from_pyarray(PyObject *pyarray, bool is_shared=true) {
        return cimage_from_pyarray<value_type>(
            reinterpret_cast<PyArrayObject *>(pyarray), is_shared);
    }

    CImg<value_type> from_datasource(PyObject *datasource, bool is_shared=true) {
        return cimage_from_pyobject<value_type>(datasource, is_shared);
    }

    CImg<value_type> from_datasource_with_dims(PyObject *datasource,
        int sW, int sH, int channels=3,
        bool is_shared=true) {
        return cimage_from_pyobject<value_type>(datasource, sW, sH, channels, is_shared);
    }

    inline bool operator()(const char sc) {
        dT self = static_cast<dT*>(this);
        for (int idx = 0; self->structcode[idx] != NILCODE; ++idx) {
            if (self->structcode[idx] == sc) { return true; }
        }
        return false;
    }

    inline bool operator[](const unsigned int tc) {
        dT self = static_cast<dT*>(this);
        return tc == self->typecode();
    }
};

template <typename T>
struct CImage_Type : public CImage_Base<CImage_Type<T>> {
    typedef typename CImage_Traits<CImage_Type<T>>::value_type value_type;
    const unsigned int value_typecode = CImage_Traits<CImage_Type<T>>::value_typecode();
    Py_buffer *pybuffer = 0;
    PyObject *datasource = 0;
    CImage_Type() {}
    CImage_Type(Py_buffer *pb) : pybuffer(pb) {}
    CImage_Type(PyArrayObject *pyarray) : datasource(reinterpret_cast<PyObject *>(pyarray)) { Py_INCREF(pyarray); }
    CImage_Type(PyObject *ds) : datasource(ds) { Py_INCREF(datasource); }

    ~CImage_Type() {
        if (datasource) { Py_DECREF(datasource); }
        if (pybuffer) { PyBuffer_Release(pybuffer); }
    }

    CImg<value_type> from_pybuffer(bool is_shared=true) {
        return cimage_from_pybuffer<value_type>(pybuffer, is_shared);
    }

    CImg<value_type> from_pybuffer_with_dims(
        int sW, int sH, int channels=3,
        bool is_shared=true) {
        return cimage_from_pybuffer<value_type>(pybuffer, sW, sH, channels, is_shared);
    }

    CImg<value_type> from_pyobject(bool is_shared=true) {
        return cimage_from_pyobject<value_type>(
            reinterpret_cast<PyObject *>(datasource), is_shared);
    }

    CImg<value_type> from_pyarray(bool is_shared=true) {
        return cimage_from_pyarray<value_type>(datasource, is_shared);
    }

    CImg<value_type> from_datasource(bool is_shared=true) {
        return cimage_from_pyobject<value_type>(datasource, is_shared);
    }

    CImg<value_type> from_datasource_with_dims(
        int sW, int sH, int channels=3,
        bool is_shared=true) {
        return cimage_from_pyobject<value_type>(datasource, sW, sH, channels, is_shared);
    }
    
    inline const unsigned int typecode() const {
        return value_typecode;
    }
    
    inline PyArray_Descr *typestruct() {
        return PyArray_DescrFromType(value_typecode);
    }
    
};

template <typename T>
struct CImage_Traits<CImage_Type<T>> {
    typedef T value_type;
    static const unsigned int value_typecode() {
        return static_cast<unsigned int>(
            numpy::dtype_code<T>());
    }
    static const PyArray_Descr *value_typestruct() {
        return numpy::dtype_struct<T>();
    }
};

template <typename T, typename dT>
unique_ptr<CImage_SubBase> create() {
    return unique_ptr<CImage_Type<dT>>(new T());
}

typedef std::map<unsigned int, unique_ptr<CImage_SubBase>(*)()> CImage_TypeMap;
static CImage_TypeMap *tmap;

struct CImage_FunctorType {
    static inline CImage_TypeMap *get_map() {
        if (!tmap) { tmap = new CImage_TypeMap(); }
        return tmap;
    }
};

template <typename dT>
static inline CImage_Type<dT> *CImage_NumpyConverter(unsigned int key) {
    CImage_TypeMap::iterator it = CImage_FunctorType::get_map()->find(key);
    if (it == CImage_FunctorType::get_map()->end()) {
        return new CImage_Type<dT>();
    }
    return dynamic_cast<CImage_Type<dT>*>(it->second());
}

template <typename dT>
static inline CImage_Type<dT> *CImage_NumpyConverter(PyObject *pyarray) {
    return new CImage_Type<dT>(pyarray);
}

template <typename dT>
static inline unique_ptr<CImage_Type<dT>> CImage_TypePointer(PyObject *pyarray) {
    return unique_ptr<CImage_Type<dT>>(new CImage_Type<dT>(pyarray));
}

template <NPY_TYPES, typename T>
struct CImage_Functor : public CImage_FunctorType {};

/////////////////////////////////// AUTOGENERATED ///////////////////////////////////////
/////////////////////////////////// AUTOGENERATED ///////////////////////////////////////
/////////////////////////////////// AUTOGENERATED ///////////////////////////////////////


struct CImage_NPY_BOOL : public CImage_Type<bool> {
    const char structcode[2] = { '?', NILCODE };
    const unsigned int structcode_length = 2;
    const bool native = true;
    const bool complex = false;
    CImage_NPY_BOOL() {}
    CImage_Functor<NPY_BOOL, bool> reg();
};

struct CImage_NPY_BYTE : public CImage_Type<char> {
    const char structcode[2] = { 'b', NILCODE };
    const unsigned int structcode_length = 2;
    const bool native = true;
    const bool complex = false;
    CImage_NPY_BYTE() {}
    CImage_Functor<NPY_BYTE, char> reg();
};

struct CImage_NPY_HALF : public CImage_Type<npy_half> {
    const char structcode[2] = { 'e', NILCODE };
    const unsigned int structcode_length = 2;
    const bool native = false;
    const bool complex = false;
    CImage_NPY_HALF() {}
    CImage_Functor<NPY_HALF, npy_half> reg();
};

struct CImage_NPY_SHORT : public CImage_Type<short> {
    const char structcode[2] = { 'h', NILCODE };
    const unsigned int structcode_length = 2;
    const bool native = true;
    const bool complex = false;
    CImage_NPY_SHORT() {}
    CImage_Functor<NPY_SHORT, short> reg();
};

struct CImage_NPY_INT : public CImage_Type<int> {
    const char structcode[2] = { 'i', NILCODE };
    const unsigned int structcode_length = 2;
    const bool native = true;
    const bool complex = false;
    CImage_NPY_INT() {}
    CImage_Functor<NPY_INT, int> reg();
};

struct CImage_NPY_LONG : public CImage_Type<long> {
    const char structcode[2] = { 'l', NILCODE };
    const unsigned int structcode_length = 2;
    const bool native = true;
    const bool complex = false;
    CImage_NPY_LONG() {}
    CImage_Functor<NPY_LONG, long> reg();
};

struct CImage_NPY_LONGLONG : public CImage_Type<long long> {
    const char structcode[2] = { 'q', NILCODE };
    const unsigned int structcode_length = 2;
    const bool native = true;
    const bool complex = false;
    CImage_NPY_LONGLONG() {}
    CImage_Functor<NPY_LONGLONG, long long> reg();
};

struct CImage_NPY_UBYTE : public CImage_Type<unsigned char> {
    const char structcode[2] = { 'B', NILCODE };
    const unsigned int structcode_length = 2;
    const bool native = true;
    const bool complex = false;
    CImage_NPY_UBYTE() {}
    CImage_Functor<NPY_UBYTE, unsigned char> reg();
};

struct CImage_NPY_USHORT : public CImage_Type<unsigned short> {
    const char structcode[2] = { 'H', NILCODE };
    const unsigned int structcode_length = 2;
    const bool native = true;
    const bool complex = false;
    CImage_NPY_USHORT() {}
    CImage_Functor<NPY_USHORT, unsigned short> reg();
};

struct CImage_NPY_UINT : public CImage_Type<unsigned int> {
    const char structcode[2] = { 'I', NILCODE };
    const unsigned int structcode_length = 2;
    const bool native = true;
    const bool complex = false;
    CImage_NPY_UINT() {}
    CImage_Functor<NPY_UINT, unsigned int> reg();
};

struct CImage_NPY_ULONG : public CImage_Type<unsigned long> {
    const char structcode[2] = { 'L', NILCODE };
    const unsigned int structcode_length = 2;
    const bool native = true;
    const bool complex = false;
    CImage_NPY_ULONG() {}
    CImage_Functor<NPY_ULONG, unsigned long> reg();
};

struct CImage_NPY_ULONGLONG : public CImage_Type<unsigned long long> {
    const char structcode[2] = { 'Q', NILCODE };
    const unsigned int structcode_length = 2;
    const bool native = true;
    const bool complex = false;
    CImage_NPY_ULONGLONG() {}
    CImage_Functor<NPY_ULONGLONG, unsigned long long> reg();
};

struct CImage_NPY_CFLOAT : public CImage_Type<std::complex<float>> {
    const char structcode[2] = { 'f', NILCODE };
    const unsigned int structcode_length = 2;
    const bool native = false;
    const bool complex = true;
    CImage_NPY_CFLOAT() {}
    CImage_Functor<NPY_CFLOAT, std::complex<float>> reg();
};

struct CImage_NPY_CDOUBLE : public CImage_Type<std::complex<double>> {
    const char structcode[2] = { 'd', NILCODE };
    const unsigned int structcode_length = 2;
    const bool native = false;
    const bool complex = true;
    CImage_NPY_CDOUBLE() {}
    CImage_Functor<NPY_CDOUBLE, std::complex<double>> reg();
};

struct CImage_NPY_FLOAT : public CImage_Type<float> {
    const char structcode[2] = { 'f', NILCODE };
    const unsigned int structcode_length = 2;
    const bool native = false;
    const bool complex = false;
    CImage_NPY_FLOAT() {}
    CImage_Functor<NPY_FLOAT, float> reg();
};

struct CImage_NPY_DOUBLE : public CImage_Type<double> {
    const char structcode[2] = { 'd', NILCODE };
    const unsigned int structcode_length = 2;
    const bool native = false;
    const bool complex = false;
    CImage_NPY_DOUBLE() {}
    CImage_Functor<NPY_DOUBLE, double> reg();
};

struct CImage_NPY_CLONGDOUBLE : public CImage_Type<std::complex<long double>> {
    const char structcode[2] = { 'g', NILCODE };
    const unsigned int structcode_length = 2;
    const bool native = true;
    const bool complex = false;
    CImage_NPY_CLONGDOUBLE() {}
    CImage_Functor<NPY_CLONGDOUBLE, std::complex<long double>> reg();
};

struct CImage_NPY_LONGDOUBLE : public CImage_Type<std::complex<long double>> {
    const char structcode[2] = { 'g', NILCODE };
    const unsigned int structcode_length = 2;
    const bool native = true;
    const bool complex = true;
    CImage_NPY_LONGDOUBLE() {}
    CImage_Functor<NPY_LONGDOUBLE, std::complex<long double>> reg();
};

template <>
struct CImage_Functor<NPY_BOOL, bool> : public CImage_FunctorType {
    CImage_Functor<NPY_BOOL, bool>(unsigned int const& key) {
        get_map()->insert(make_pair(key, &create<CImage_NPY_BOOL, bool>));
    }
};

template <>
struct CImage_Functor<NPY_BYTE, char> : public CImage_FunctorType {
    CImage_Functor<NPY_BYTE, char>(unsigned int const& key) {
        get_map()->insert(make_pair(key, &create<CImage_NPY_BYTE, char>));
    }
};

template <>
struct CImage_Functor<NPY_HALF, npy_half> : public CImage_FunctorType {
    CImage_Functor<NPY_HALF, npy_half>(unsigned int const& key) {
        get_map()->insert(make_pair(key, &create<CImage_NPY_HALF, npy_half>));
    }
};

template <>
struct CImage_Functor<NPY_SHORT, short> : public CImage_FunctorType {
    CImage_Functor<NPY_SHORT, short>(unsigned int const& key) {
        get_map()->insert(make_pair(key, &create<CImage_NPY_SHORT, short>));
    }
};

template <>
struct CImage_Functor<NPY_INT, int> : public CImage_FunctorType {
    CImage_Functor<NPY_INT, int>(unsigned int const& key) {
        get_map()->insert(make_pair(key, &create<CImage_NPY_INT, int>));
    }
};

template <>
struct CImage_Functor<NPY_LONG, long> : public CImage_FunctorType {
    CImage_Functor<NPY_LONG, long>(unsigned int const& key) {
        get_map()->insert(make_pair(key, &create<CImage_NPY_LONG, long>));
    }
};

template <>
struct CImage_Functor<NPY_LONGLONG, long long> : public CImage_FunctorType {
    CImage_Functor<NPY_LONGLONG, long long>(unsigned int const& key) {
        get_map()->insert(make_pair(key, &create<CImage_NPY_LONGLONG, long long>));
    }
};

template <>
struct CImage_Functor<NPY_UBYTE, unsigned char> : public CImage_FunctorType {
    CImage_Functor<NPY_UBYTE, unsigned char>(unsigned int const& key) {
        get_map()->insert(make_pair(key, &create<CImage_NPY_UBYTE, unsigned char>));
    }
};

template <>
struct CImage_Functor<NPY_USHORT, unsigned short> : public CImage_FunctorType {
    CImage_Functor<NPY_USHORT, unsigned short>(unsigned int const& key) {
        get_map()->insert(make_pair(key, &create<CImage_NPY_USHORT, unsigned short>));
    }
};

template <>
struct CImage_Functor<NPY_UINT, unsigned int> : public CImage_FunctorType {
    CImage_Functor<NPY_UINT, unsigned int>(unsigned int const& key) {
        get_map()->insert(make_pair(key, &create<CImage_NPY_UINT, unsigned int>));
    }
};

template <>
struct CImage_Functor<NPY_ULONG, unsigned long> : public CImage_FunctorType {
    CImage_Functor<NPY_ULONG, unsigned long>(unsigned int const& key) {
        get_map()->insert(make_pair(key, &create<CImage_NPY_ULONG, unsigned long>));
    }
};

template <>
struct CImage_Functor<NPY_ULONGLONG, unsigned long long> : public CImage_FunctorType {
    CImage_Functor<NPY_ULONGLONG, unsigned long long>(unsigned int const& key) {
        get_map()->insert(make_pair(key, &create<CImage_NPY_ULONGLONG, unsigned long long>));
    }
};

template <>
struct CImage_Functor<NPY_CFLOAT, std::complex<float>> : public CImage_FunctorType {
    CImage_Functor<NPY_CFLOAT, std::complex<float>>(unsigned int const& key) {
        get_map()->insert(make_pair(key, &create<CImage_NPY_CFLOAT, std::complex<float>>));
    }
};

template <>
struct CImage_Functor<NPY_CDOUBLE, std::complex<double>> : public CImage_FunctorType {
    CImage_Functor<NPY_CDOUBLE, std::complex<double>>(unsigned int const& key) {
        get_map()->insert(make_pair(key, &create<CImage_NPY_CDOUBLE, std::complex<double>>));
    }
};

template <>
struct CImage_Functor<NPY_FLOAT, float> : public CImage_FunctorType {
    CImage_Functor<NPY_FLOAT, float>(unsigned int const& key) {
        get_map()->insert(make_pair(key, &create<CImage_NPY_FLOAT, float>));
    }
};

template <>
struct CImage_Functor<NPY_DOUBLE, double> : public CImage_FunctorType {
    CImage_Functor<NPY_DOUBLE, double>(unsigned int const& key) {
        get_map()->insert(make_pair(key, &create<CImage_NPY_DOUBLE, double>));
    }
};

template <>
struct CImage_Functor<NPY_CLONGDOUBLE, std::complex<long double>> : public CImage_FunctorType {
    CImage_Functor<NPY_CLONGDOUBLE, std::complex<long double>>(unsigned int const& key) {
        get_map()->insert(make_pair(key, &create<CImage_NPY_CLONGDOUBLE, std::complex<long double>>));
    }
};

template <>
struct CImage_Functor<NPY_LONGDOUBLE, std::complex<long double>> : public CImage_FunctorType {
    CImage_Functor<NPY_LONGDOUBLE, std::complex<long double>>(unsigned int const& key) {
        get_map()->insert(make_pair(key, &create<CImage_NPY_LONGDOUBLE, std::complex<long double>>));
    }
};

/////////////////////////////////// !AUTOGENERATED ///////////////////////////////////////
/////////////////////////////////// !AUTOGENERATED ///////////////////////////////////////
/////////////////////////////////// !AUTOGENERATED ///////////////////////////////////////

extern "C" void CImage_Register() {}

#endif /// PyImgC_CIMAGE_H