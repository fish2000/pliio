
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

private:
    PyArray_Descr *dtype = NULL;
    shared_ptr<void> cimage = shared_ptr<void>(nullptr);
    
public:
    PyCImage() = default;
    
    // template <typename T>
    // PyCImage(const struct CImg<T> &ci) {
    //     cimage(shared_ptr<CImg<T>>(ci));
    //     dtype((cimage.get())->typestruct());
    // }
    template <typename T>
    PyCImage(CImg<T> const &ci) {
        cimage(shared_ptr<CImg<T>>(ci));
        dtype((cimage.get())->typestruct());
    }
    template <typename T>
    PyCImage &operator=(CImg<T> const &ci) {
        if (checkptr()) { cimage.reset(); }
        cimage = shared_ptr<T>(ci);
        dtype = (cimage.get())->typestruct();
    }
    
    template <typename T>
    PyCImage(shared_ptr<CImg<T>> &ptr) {
        cimage(ptr);
        dtype((cimage.get())->typestruct());
    }
    template <typename T>
    PyCImage &operator=(shared_ptr<CImg<T>> const &ptr) {
        if (checkptr()) { cimage.reset(); }
        if (checkdtype()) { delete dtype; }
        cimage = ptr;
        dtype = (cimage.get())->typestruct();
    }
    
    auto view() -> decltype(remove_pointer<cimage.get()>::type>);
    inline bool checkptr() { return cimage.get() != nullptr; }
    inline bool checkdtype() { return dtype != NULL && PyArray_DescrCheck(dtype); }
    inline unsigned int typecode() {
        if (checkdtype()) { return (unsigned int)dtype->type_num; }
        if (checkptr()) { dtype((cimage.get())->typestruct()); return dtype; }
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
        if (checkptr()) { dtype = (cimage.get())->typestruct(); Py_INCREF(dtype); return dtype; }
        return 0;
    }
    
    
};

auto PyCImage::view() -> decltype(remove_pointer<cimage.get()>::type>) {
    if (PyArray_DescrCheck(dtype)) {
        int tc = (int)dtype->type_num;
        if (!tc) { tc = IMGC_DEFAULT_TYPECODE; }
#define HANDLE(type) \
        return CImg<type>(*recast<type>());
        SAFE_SWITCH_ON_TYPECODE(tc, CImg<IMGC_DEFAULT_T>());
#undef HANDLE
    }
    return CImg<IMGC_DEFAULT_T>();
}



#endif /// PyImgC_TYPESTRUCT_PYCIMAGE_H