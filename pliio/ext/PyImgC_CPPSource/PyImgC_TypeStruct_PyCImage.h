
#ifndef PyImgC_TYPESTRUCT_PYCIMAGE_H
#define PyImgC_TYPESTRUCT_PYCIMAGE_H

#include <Python.h>
#include <numpy/ndarrayobject.h>
#include "PyImgC_Constants.h"
#include "PyImgC_SharedDefs.h"
#include "PyImgC_Types.h"
#include "PyImgC_IMP_PyBufferDict.h"

using namespace cimg_library;
using namespace std;

struct PyCImage {
    PyObject_HEAD
    
    void **view();
    void *viewptr;
    PyArray_Descr *dtype = NULL;
    unique_ptr<CImage_SubBase> cimage = unique_ptr<CImage_SubBase>(nullptr);
    
    inline bool checkptr() { return cimage.get() != nullptr; }
    inline bool checkdtype() { return dtype != NULL; }
    inline unsigned int typecode() {
        if (checkdtype()) { return (unsigned int)dtype->type_num; }
        return 0;
    }
    
    template <typename T>
    CImage_Type<T> *recast() {
        if (!checkptr()) { return NULL; }
        return dynamic_cast<CImage_Type<T>*>(cimage.get());
    }
    
    template <typename T>
    CImage_Type<T> *force() {
        if (!checkptr()) { return NULL; }
        return reinterpret_cast<CImage_Type<T>*>(cimage.get());
    }
};


void **PyCImage::view() {
    if (PyArray_DescrCheck(dtype)) {
        int tc = (int)dtype->type_num;
        if (!tc) { tc = IMGC_DEFAULT_TYPECODE; }
        if (!viewptr) {
#define HANDLE(type) \
            viewptr = (void *)recast<type>()->get(false);
            SAFE_SWITCH_ON_TYPECODE(tc, CImg<IMGC_DEFAULT_T>());
#undef HANDLE
        }
        return *viewptr;
    }
    return (void **)CImg<IMGC_DEFAULT_T>();
}

#endif /// PyImgC_TYPESTRUCT_PYCIMAGE_H