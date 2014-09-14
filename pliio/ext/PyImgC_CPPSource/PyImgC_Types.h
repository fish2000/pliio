#ifndef PyImgC_CIMAGE_H
#define PyImgC_CIMAGE_H

#include <map>
#include <cmath>
#include <cstdlib>
#include <type_traits>
#if IMGC_DEBUG > 0
#include <typeinfo>
#endif

#include <Python.h>
#include <structmember.h>
#include <numpy/ndarrayobject.h>
#include <numpy/ndarraytypes.h>
#include "numpypp/numpy.hpp"

#include "PyImgC_Options.h"
#include "PyImgC_SharedDefs.h"
#include "cimg/CImg.h"

using namespace cimg_library;
using namespace std;

template <IMGT>
CImg<T> cimage_from_pybuffer(Py_buffer *pybuffer, int sW, int sH,
                    int channels, bool is_shared=true) {
    CImg<T> view(pybuffer->buf,
        sW, sH, 1,
        channels, is_shared);
    return view;
}

template <IMGT>
CImg<T> cimage_from_pybuffer(Py_buffer *pybuffer, bool is_shared=true) {
    CImg<T> view(pybuffer->buf,
        pybuffer->shape[1],
        pybuffer->shape[0],
        1, 3, is_shared);
    return view;
}

/// ***FIX THE FUCK OUT OF ME***
template <IMGT>
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

template <IMGT>
CImg<T> cimage_from_pyarray(PyObject *pyobj, bool is_shared=true) {
    if (!PyArray_Check(pyobj)) { return CImg<T>(); }
    PyArrayObject *pyarray = reinterpret_cast<PyArrayObject *>(pyobj);
    return cimage_from_pyarray<T>(pyarray, is_shared);
}

/// ***FIX THE FUCK OUT OF ME***
template <IMGT>
CImg<T> cimage_from_pyobject(PyObject *datasource, int sW, int sH,
                    int channels, bool is_shared=true) {
    CImg<T> view(sW, sH, 1, channels, is_shared);
    return view;
}

/// ***FIX THE FUCK OUT OF ME***
template <IMGT>
CImg<T> cimage_from_pyobject(PyObject *datasource, bool is_shared=true) {
    CImg<T> view(640, 480, 1, 3, is_shared);
    return view;
}

struct CImage_SubBase {
    virtual ~CImage_SubBase() {};
};

template <typename dT>
struct CImage_Traits;

template <typename dT>
struct CImage_Base : public CImage_SubBase {
    typedef typename CImage_Traits<dT>::value_type value_type;
    const unsigned int value_typecode = CImage_Traits<CImage_Type<T>>::value_typecode();
    
    inline CImg<value_type> from_pybuffer(Py_buffer *pybuffer, bool is_shared=true) {
        return cimage_from_pybuffer<value_type>(pybuffer, is_shared);
    }

    inline CImg<value_type> from_pybuffer_with_dims(Py_buffer *pybuffer,
        int sW, int sH, int channels=3,
        bool is_shared=true) {
        return cimage_from_pybuffer<value_type>(pybuffer, sW, sH, channels, is_shared);
    }

    inline CImg<value_type> from_pyarray(PyArrayObject *pyarray, bool is_shared=true) {
        return cimage_from_pyarray<value_type>(pyarray, is_shared);
    }

    inline CImg<value_type> from_pyarray(PyObject *pyarray, bool is_shared=true) {
        return cimage_from_pyarray<value_type>(
            reinterpret_cast<PyArrayObject *>(pyarray), is_shared);
    }

    inline CImg<value_type> from_datasource(PyObject *datasource, bool is_shared=true) {
        return cimage_from_pyobject<value_type>(datasource, is_shared);
    }

    inline CImg<value_type> from_datasource_with_dims(PyObject *datasource,
        int sW, int sH, int channels=3,
        bool is_shared=true) {
        return cimage_from_pyobject<value_type>(datasource, sW, sH, channels, is_shared);
    }
    
    inline const unsigned int typecode() { return value_typecode; }
    inline const PyArray_Descr *const typestruct() {
        /// calling this directly is a few references' worth of speed faster
        /// than using numpy::dtype_code<value_type>()
        return PyArray_DescrFromType(value_typecode);
    }

    /// QUESTIONABLE
    inline bool operator()(const char sc) {
        dT self = static_cast<dT*>(this);
        for (int idx = 0; self->structcode[idx] != NILCODE; ++idx) {
            if (self->structcode[idx] == sc) { return true; }
        }
        return false;
    }

    /// QUESTIONABLE
    inline bool operator[](const unsigned int tc) {
        dT self = static_cast<dT*>(this);
        return tc == self->typecode();
    }
};

template <IMGT>
struct CImage_Type : public CImage_Base<CImage_Type<T>> {
    typedef typename CImage_Traits<CImage_Type<T>>::value_type value_type;
    const unsigned int value_typecode = CImage_Traits<CImage_Type<T>>::value_typecode();
    
    Py_buffer *pybuffer = 0;
    PyObject *datasource = 0;
    CImg<value_type> cinstance;
    
    CImage_Type() {}
    CImage_Type(Py_buffer *pb) : pybuffer(pb) {}
    CImage_Type(PyArrayObject *pyarray) : datasource(reinterpret_cast<PyObject *>(pyarray)) { Py_INCREF(pyarray); }
    CImage_Type(PyObject *ds) : datasource(ds) { Py_INCREF(datasource); }
    CImage_Type(CImg<value_type> cim) : cinstance() {
        IMGC_COUT("> constructing CImage_Type<" << typeid(T).name() << "> with CImg<"
                << cim.pixel_type() << ">");
    }
    CImage_Type(CImg<value_type> *cim) : cinstance() {
        IMGC_COUT("> constructing CImage_Type<" << typeid(T).name() << "> with *CImg<"
                << cim->pixel_type() << ">");
    }

    virtual ~CImage_Type() {
        if (check_datasource()) { Py_DECREF(datasource); }
        if (check_pybuffer()) { PyBuffer_Release(pybuffer); }
        if (check_instance()) {}
    }

    virtual bool check_instance() { return static_cast<bool>(cinstance.size()); }
    virtual bool check_datasource() { return datasource != 0; }
    virtual bool check_pyarray() { return check_datasource() && PyArray_Check(datasource); }
    virtual bool check_pybuffer() { return pybuffer != 0; }

    CImg<value_type> get(bool is_shared=true) {
        if (check_instance()) { return this->cinstance; }
        if (check_pyarray()) { return this->from_pyarray(is_shared); }
        if (check_datasource()) { return this->from_pyobject(is_shared); }
        if (check_pybuffer()) { return this->from_pybuffer(is_shared); }
        return CImg<value_type>(this->datasource); /// ugh we can do better
    }

    operator CImg<value_type>() { return this->get(false); }
    //operator CImg<value_type>(void) const;

    /*
     * THIS FUCKING PILE OF METHODS IS A DIRTY MESS AND NEEDS TO GET CLEANED UP
     */


    void set(PyArrayObject *pyarray) {
        datasource = reinterpret_cast<PyObject *>(pyarray);
        Py_INCREF(pyarray);
        cinstance = cimage_from_pyarray<value_type>(datasource, true);
    }
    void set(PyObject *ds) {
        datasource = ds;
        Py_INCREF(datasource);
        cinstance = cimage_from_pyobject<value_type>(datasource, true);
    }
    void set(Py_buffer *pb) {
        pybuffer = pb;
        cinstance = cimage_from_pybuffer<value_type>(pybuffer, true);
    }

    inline CImg<value_type> from_pybuffer(bool is_shared=true) {
        return cimage_from_pybuffer<value_type>(pybuffer, is_shared);
    }

    inline CImg<value_type> from_pybuffer_with_dims(
        int sW, int sH, int channels=3,
        bool is_shared=true) {
        return cimage_from_pybuffer<value_type>(pybuffer, sW, sH, channels, is_shared);
    }

    inline CImg<value_type> from_pyobject(bool is_shared=true) {
        return cimage_from_pyobject<value_type>(
            reinterpret_cast<PyObject *>(datasource), is_shared);
    }

    inline CImg<value_type> from_pyarray(bool is_shared=true) {
        return cimage_from_pyarray<value_type>(datasource, is_shared);
    }

    inline CImg<value_type> from_datasource(bool is_shared=true) {
        return cimage_from_pyobject<value_type>(datasource, is_shared);
    }

    inline CImg<value_type> from_datasource_with_dims(
        int sW, int sH, int channels=3,
        bool is_shared=true) {
        return cimage_from_pyobject<value_type>(datasource, sW, sH, channels, is_shared);
    }
    
};

template <typename T>
struct CImage_Traits<CImage_Type<T>> {
    typedef T value_type;
    static inline const unsigned int value_typecode() {
        return static_cast<unsigned int>(
            numpy::dtype_code<T>());
    }
    static inline PyArray_Descr *value_typestruct() {
        return numpy::dtype_struct<T>();
    }
};

}

#endif /// PyImgC_CIMAGE_H