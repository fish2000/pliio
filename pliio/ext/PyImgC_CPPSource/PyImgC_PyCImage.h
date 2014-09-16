
#ifndef PyImgC_TYPESTRUCT_PYCIMAGE_H
#define PyImgC_TYPESTRUCT_PYCIMAGE_H

#include <map>
#include <array>
#include <cmath>
#include <string>
#include <vector>
#include <memory>
#include <cstdlib>
#include <type_traits>

#include "PyImgC_Options.h"
#include "PyImgC_SharedDefs.h"
#include <Python.h>
#include <structmember.h>
#include <numpy/ndarrayobject.h>
#include <numpy/ndarraytypes.h>
#include "numpypp/numpy.hpp"
#include "numpypp/dispatch.hpp"
#include "numpypp/utils.hpp"
using namespace std;

#include "cimg/CImg.h"
using namespace cimg_library;

struct PyCImage {
    PyObject_HEAD
    
public:
    PyArray_Descr *dtype = NULL;
    shared_ptr<CImg_Base> cimage = shared_ptr<CImg_Base>(nullptr);
    
    PyCImage() = default;
    
    template <typename T>
    PyCImage(CImg<T> const &ci) {
        cimage(make_shared<CImg<T>>(ci));
        dtype(recast<T>()->typestruct());
    }
    template <typename T>
    PyCImage &operator=(CImg<T> const &ci) {
        if (checkptr()) { cimage.reset(); }
        cimage = make_shared<CImg<T>>(ci);
        dtype = recast<T>()->typestruct();
    }
    
    template <typename T>
    PyCImage(shared_ptr<CImg<T>> &ptr) {
        cimage(ptr);
        dtype(recast<T>()->typestruct());
    }
    template <typename T>
    PyCImage &operator=(shared_ptr<CImg<T>> const &ptr) {
        if (checkptr()) { cimage.reset(); }
        if (checkdtype()) { delete dtype; }
        cimage = ptr;
        dtype = recast<T>()->typestruct();
    }
    
    template <typename T>
    void assign(CImg<T> &ci) {
        if (checkptr()) { cimage.reset(); }
        cimage = make_shared<CImg<T>>(ci);
        dtype = recast<T>()->typestruct();
    }
    template <typename T>
    void assign(CImg<T> const &ci) {
        if (checkptr()) { cimage.reset(); }
        CImg<T> cim(ci);
        cimage = make_shared<CImg<T>>(cim);
        dtype = recast<T>()->typestruct();
    }
    
    inline bool checkptr() { return cimage.get() != nullptr; }
    inline bool checkdtype() { return dtype != NULL && PyArray_DescrCheck(dtype); }
    inline unsigned int typecode() {
        if (checkdtype()) { return (unsigned int)dtype->type_num; }
        return 0;
    }
    
    inline unsigned short compare(PyCImage *other) {
        if (!dtype) {
            PyErr_SetString(PyExc_ValueError,
                "Comparator object has no dtype");
                return -1;
        }
#define HANDLE(type) return compare_with<type>(other);
    SAFE_SWITCH_ON_DTYPE(dtype, -2);
#undef HANDLE
    PyErr_SetString(PyExc_ValueError,
        "Comparison failure in PyCImage.compare()");
        return -1;
    }
    
    template <typename selfT>
    inline unsigned short compare_with(PyCImage *other) {
        if (!other->dtype) {
            PyErr_SetString(PyExc_ValueError,
                "Object to compare has no dtype");
            return -1;
        }
#define HANDLE(otherT) { \
        auto self = *recast<selfT>(); \
        auto another = *other->recast<otherT>(); \
        if (self == another) { return 0; } \
        return self.size() > another.size() ? 1 : -1; \
    }
    SAFE_SWITCH_ON_DTYPE(other->dtype, -3);
#undef HANDLE
    PyErr_SetString(PyExc_ValueError,
        "Comparison failure in PyCImage.compare_with<T>();");
        return -1;
    }
    
    template <typename T>
    CImg<T> *recast() {
        if (!checkptr()) { return nullptr; }
        return dynamic_cast<CImg<T>*>(cimage.get());
    }
    
    template <typename T>
    operator shared_ptr<CImg<T>>() const { return dynamic_pointer_cast<CImg<T>>(cimage); }
    template <typename T>
    operator CImg<T>() const { return cimage.get(); }
    template <typename T>
    operator CImg<T>&() { return *cimage.get(); }
    
    operator PyArray_Descr*() {
        if (checkdtype()) { Py_INCREF(dtype); return dtype; }
        return 0;
    }
    
};

#endif /// PyImgC_TYPESTRUCT_PYCIMAGE_H