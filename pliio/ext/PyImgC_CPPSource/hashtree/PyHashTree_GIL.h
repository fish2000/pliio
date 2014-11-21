

#ifndef PyHashTree_PYHASHTREE_IMP_GIL_H
#define PyHashTree_PYHASHTREE_IMP_GIL_H

#include <Python.h>

struct gil_release {
    PyThreadState *thread_state;
    bool gil_active;
    
    gil_release() {
        thread_state = PyEval_SaveThread();
        gil_active = true;
    }
    
    ~gil_release() {
        if (gil_active) { restore(); }
    }
    
    void restore() {
        PyEval_RestoreThread(thread_state);
        gil_active = false;
    }
};

struct gil_ensure {
    PyGILState_STATE gil_state;
    bool gil_ensured;
    
    gil_ensure() {
        gil_state = PyGILState_Ensure();
        gil_ensured = true;
    }
    
    ~gil_ensure() {
        if (gil_ensured) { restore(); }
    }
    
    void restore() {
        PyGILState_Release(gil_state);
        gil_ensured = false;
    }
};

#endif /// PyHashTree_PYHASHTREE_IMP_GIL_H