
#ifndef PyHashTree_TYPESTRUCT_PYHASHTREE_H
#define PyHashTree_TYPESTRUCT_PYHASHTREE_H

#include <Python.h>
#include <structmember.h>
#include "mvptree/mvptree.h"

struct PyHashTree {
    PyObject_HEAD
    MVPTree *tree;
    unsigned int branch_factor;
    unsigned int path_length;
    unsigned int leafnode_capacity;
    
    void cleanup() {
        mvptree_clear(tree, free);
        free(tree);
    }
};

#endif /// PyHashTree_TYPESTRUCT_PYHASHTREE_H