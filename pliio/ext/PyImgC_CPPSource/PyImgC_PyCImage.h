
#ifndef PyImgC_TYPESTRUCT_PYCIMAGE_H
#define PyImgC_TYPESTRUCT_PYCIMAGE_H

#ifdef __OBJC__
#import <Cocoa/Cocoa.h>
#import <Foundation/Foundation.h>
#import <CoreFoundation/CoreFoundation.h>
#endif /// __OBJC__

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
#ifdef __OBJC__
        CFIndex count, idx;
        if (checkcontext()) {
            count = CFGetRetainCount(_context);
            for (idx = 0; idx < count; ++idx) {
                CGContextRelease(_context);
            }
            _context = NULL;
        }
        if (checkcolorspace()) {
            count = CFGetRetainCount(_colorspace);
            for (idx = 0; idx < count; ++idx) {
                CGColorSpaceRelease(_colorspace);
            }
            _colorspace = NULL;
        }
#endif /// __OBJC__
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
#ifdef __OBJC__
        return _context != NULL;
#else
        return false;
#endif /// __OBJC__
    }
    inline bool checkcolorspace() {
#ifdef __OBJC__
        return _colorspace != NULL;
#else
        return false;
#endif /// __OBJC__
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
    
#ifdef __OBJC__
    
    CGColorSpaceRef &cgColorSpace() {
        if (!checkcolorspace()) {
            _colorspace = CGColorSpaceCreateDeviceRGB();
        }
        CGColorSpaceRetain(_colorspace);
        return _colorspace;
    }
    
    CGContextRef &cgContext() {
        if (!checkcontext()) {
#define HANDLE(type) _context = recast<type>()->cgContext(cgColorSpace());
        VOID_SWITCH_ON_DTYPE(dtype);
#undef HANDLE
        }
        CGContextRetain(_context);
        return _context;
    }
    
    CGImageRef cgImageRef() {
        return CGBitmapContextCreateImage(cgContext());
    }
    
private:
    CGColorSpaceRef _colorspace = NULL;
    CGContextRef _context = NULL;
    
#endif /// __OBJC__

};

#endif /// PyImgC_TYPESTRUCT_PYCIMAGE_H