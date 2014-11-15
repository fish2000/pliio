
#ifndef PyImgC_PYIMGC_IMP_LITTLECMSCONTEXT_H
#define PyImgC_PYIMGC_IMP_LITTLECMSCONTEXT_H

#include <Python.h>
#include <numpy/ndarrayobject.h>
#include "numpypp/numpy.hpp"
#include "numpypp/dispatch.hpp"
#include "numpypp/utils.hpp"

#include <lcms2.h>

using namespace cimg_library;
using namespace std;

#define IMGC_CMS_NAME "imgc._pycmx"
#define IMGC_CMS_CONTEXT(po) (cmsContext)PyCapsule_GetPointer(po, IMGC_CMS_NAME)

static void PyImgC_CMS_Shutdown(PyObject *pycmx) {
    cmsContext cmx = IMGC_CMS_CONTEXT(pycmx);
    cmsDeleteContext(cmx);
}

static PyObject *PyImgC_CMS_Startup(PyObject *cmxdata) {
    cmsContext cmx;
    if (cmxdata) {
        cmx = cmsCreateContext(NULL, (void *)cmxdata);
    } else {
        cmx = cmsCreateContext(NULL, NULL);
    }
    return PyCapsule_New(
        (void *)cmx, IMGC_CMS_NAME,
        (PyCapsule_Destructor)PyImgC_CMS_Shutdown);
}

static void PyImgC_AtExit(void) {
    /// clean up module-level resources
}

/* re-redefine T_FLOAT from python/includes/structmember.h */
#define T_FLOAT 3

#endif /// PyImgC_PYIMGC_IMP_LITTLECMSCONTEXT_H
