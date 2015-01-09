
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
#include "numpypp/typecode.hpp"
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
    
    ~PyCImage() {
        cleanup();
        Py_XDECREF(dtype);
    }
    
    bool save(const char *path) {
        if (!path) { return false; }
#define HANDLE(type) { \
            recast<type>()->save(path); \
            return true; \
        }
        SAFE_SWITCH_ON_DTYPE(dtype, false);
#undef HANDLE
        return false;
    }
    
    void cgRelease() {
        if (checkcontext()) {
#define HANDLE(type) { \
            CGContextRef ctx = recast<type>()->_context; \
            CFIndex count = CFGetRetainCount(ctx); \
            for (CFIndex idx = 0; idx < count; ++idx) { \
                CGContextRelease(ctx); \
            } \
            ctx = NULL; \
        }
        VOID_SWITCH_ON_DTYPE(dtype);
#undef HANDLE
        }
    }
    
    void cleanup() {
        if (checkptr()) { cimage.reset(); }
        cgRelease();
    }
    
    inline bool is_empty() {
#define HANDLE(type) return recast<type>()->is_empty();
        SAFE_SWITCH_ON_DTYPE(dtype, true);
#undef HANDLE
        return true;
    }
    
    inline bool checkptr() { return cimage.get() != nullptr; }
    inline bool checkdtype() { return dtype != NULL && PyArray_DescrCheck(dtype); }
    inline bool checkcontext() {
        if (!checkptr()) { return false; }
#ifdef __OBJC__
#define HANDLE(type) return recast<type>()->_context != NULL;
        SAFE_SWITCH_ON_DTYPE(dtype, false);
        return false;
#undef HANDLE
#else
        return false;
#endif
    }
    
    inline unsigned int typecode() {
        if (checkdtype()) {
            return static_cast<unsigned int>(dtype->type_num);
        }
        return 0;
    }
    inline char typechar() {
        if (checkdtype()) { return typecode::typechar(typecode()); }
        return '?';
    }
    inline string typecode_name() {
        if (checkdtype()) { return typecode::name(typecode()); }
        return string("?");
    }
    
    inline CImg_Base *__base__() { return cimage.get(); }
    
    inline signed short compare(PyCImage *other) {
        if (!dtype) {
            PyErr_SetString(PyExc_ValueError,
                "Comparator object has no dtype");
                return -1;
        }
        gil_release NOGIL;
#define HANDLE(type) return compare_with<type>(other);
        SAFE_SWITCH_ON_DTYPE(dtype, -2);
#undef HANDLE
        NOGIL.~gil_release();
        PyErr_SetString(PyExc_ValueError,
            "Comparison failure in PyCImage.compare()");
        return -1;
    }
    
    template <typename selfT>
    inline signed short compare_with(PyCImage *other) {
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
    
    operator PyArray_Descr*() {
        if (checkdtype()) { Py_INCREF(dtype); return dtype; }
        return 0;
    }
    
};

#endif /// PyImgC_TYPESTRUCT_PYCIMAGE_H