
#ifndef PyHashTree_TYPESTRUCT_PYHASHTREE_H
#define PyHashTree_TYPESTRUCT_PYHASHTREE_H

#include <Python.h>
#include <structmember.h>
#include "mvptree/mvptree.h"
#include "mvptree/mvpvector.hpp"

struct PyHashTree {
    PyObject_HEAD
    MVPTree *tree;
    unsigned int branch_factor;
    unsigned int path_length;
    unsigned int leafnode_capacity;
    
    vector<MVPDP, DataPointAllocator<MVPDP>> treevector() {
        return MVP::mvpvector(tree);
    }
    
    void cleanup() {
        mvptree_clear(tree, PyMem_Free);
        PyMem_Free(tree);
    }
    
    ~PyHashTree() {
        if (tree) { cleanup(); }
    }
};

#endif /// PyHashTree_TYPESTRUCT_PYHASHTREE_H