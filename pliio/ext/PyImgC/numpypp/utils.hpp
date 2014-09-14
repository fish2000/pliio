/* Copyright 2010-2012 (C)
 * Luis Pedro Coelho <luis@luispedro.org>
 * License: MIT
 * Annotated and rearranged by FI$H 2000
 */

#include <Python.h>
#include <numpy/ndarrayobject.h>

// holdref is a RAII object for decreasing a reference at scope exit
struct holdref {
    holdref(PyObject* obj, bool incref=true)
        :obj(obj) {
        if (incref) { Py_XINCREF(obj); }
    }
    holdref(PyArrayObject* obj, bool incref=true)
        :obj((PyObject*)obj) {
        if (incref) { Py_XINCREF(obj); }
    }
    ~holdref() { Py_XDECREF(obj); }

private:
    PyObject* const obj;
};

// gil_release is a sort of reverse RAII object: it acquires the GIL on scope exit
/// [... would that not then make this a RAID, since the resource is allocated on destruction? -fish]
struct gil_release {
    PyThreadState *thread_state;
    bool gil_active;
    
    gil_release() {
        //IMGC_CERR("> GIL: releasing");
        thread_state = PyEval_SaveThread();
        gil_active = true;
    }
    
    ~gil_release() {
        //IMGC_CERR("> GIL: restoring from release");
        if (gil_active) { restore(); }
    }
    
    void restore() {
        PyEval_RestoreThread(thread_state);
        gil_active = false;
    }
};

/// This one does the opposite of gil_release -- it uses C++ scoping rules to 
/// acquire the GIL for threaded C/C++ calls, resetting the threading state
/// when blown away
struct gil_ensure {
    PyGILState_STATE gil_state;
    bool gil_ensured;
    
    gil_ensure() {
        //IMGC_CERR("> GIL: ensuring state");
        gil_state = PyGILState_Ensure();
        gil_ensured = true;
    }
    
    ~gil_ensure() {
        //IMGC_CERR("> GIL: THE UNINSURED");
        if (gil_ensured) { restore(); }
    }
    
    void restore() {
        PyGILState_Release(gil_state);
        gil_ensured = false;
    }
};



// This encapsulates the arguments to PyErr_SetString
// The reason that it doesn't call PyErr_SetString directly is that we wish
// that this be throw-able in an environment where the thread might not own the
// GIL as long as it is caught when the GIL is held.
struct PythonException {
    PythonException(PyObject *type, const char *message)
        :type_(type)
        ,message_(message)
        { }

    PyObject* type() const { return type_; }
    const char* message() const { return message_; }

    PyObject* const type_;
    const char* const message_;
};



// DECLARE_MODULE is slightly ugly, but it encapsulates the differences in
// initializing a module between Python 2.x & Python 3.x

#if PY_MAJOR_VERSION < 3
#define DECLARE_MODULE(name) \
extern "C" \
void init##name () { \
    import_array(); \
    (void)Py_InitModule(#name, methods); \
}

#else

#define DECLARE_MODULE(name) \
namespace { \
    struct PyModuleDef moduledef = { \
        PyModuleDef_HEAD_INIT, \
        #name, \
        NULL, \
        -1, \
        methods, \
        NULL, \
        NULL, \
        NULL, \
        NULL \
    }; \
} \
PyMODINIT_FUNC \
PyInit_##name () { \
    import_array(); \
    return PyModule_Create(&moduledef); \
}


#endif


