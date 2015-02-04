
#include "PyImgC_IMP_BufferProtocol.h"

using namespace cimg_library;
using namespace std;

int PyCImage_GetBuffer(PyObject *self, Py_buffer *view, int flags) {
    PyCImage *pyim = reinterpret_cast<PyCImage *>(self);
    if (pyim->cimage && pyim->dtype) {
#define HANDLE(type) { \
            auto cim = pyim->recast<type>(); \
            cim->get_pybuffer(view); \
        }
        SAFE_SWITCH_ON_DTYPE(pyim->dtype, -1);
#undef HANDLE
        Py_INCREF(self);
        view->obj = self;
    }
    return 0;
}

void PyCImage_ReleaseBuffer(PyObject *self, Py_buffer *view) {
    if (view->internal) {
        const char *internal = (const char *)view->internal;
        if (internal == IMGC_PYBUFFER_PYMEM_MALLOC) {
            if (view->shape) { PyMem_Free(view->shape); }
            if (view->strides) { PyMem_Free(view->strides); }
        } else if (internal == IMGC_PYBUFFER_GLIBC_MALLOC) {
            if (view->shape) { free(view->shape); }
            if (view->strides) { free(view->strides); }
        }
    }
    if (view->obj) {
        Py_DECREF(view->obj);
        view->obj = NULL;
    }
    PyBuffer_Release(view);
}
