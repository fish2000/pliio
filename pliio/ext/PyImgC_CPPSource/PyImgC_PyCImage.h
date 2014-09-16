
#ifndef PyImgC_TYPESTRUCT_PYCIMAGE_H
#define PyImgC_TYPESTRUCT_PYCIMAGE_H

#include <Python.h>
#include <numpy/ndarrayobject.h>
#include "PyImgC_Constants.h"
#include "PyImgC_SharedDefs.h"
#include "PyImgC_Types.h"
//#include "PyImgC_IMP_PyBufferDict.h"

using namespace cimg_library;
using namespace std;

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