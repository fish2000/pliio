
#ifndef PyHashTree_TYPESTRUCT_PYHASHTREE_H
#define PyHashTree_TYPESTRUCT_PYHASHTREE_H

#include <Python.h>
#include <structmember.h>
#include <iostream>
#include "mvptree/mvptree.h"
#include "mvptree/mvpvector.hpp"
using namespace std;

struct PyHashTree {
    PyObject_HEAD
    MVPTree *tree = nullptr;
    shared_ptr<MVP::MVPVector> treevector = shared_ptr<MVP::MVPVector>(nullptr);
    unsigned int branch_factor;
    unsigned int path_length;
    unsigned int leafnode_capacity;
    
    inline bool checkptr() { return treevector.get() != nullptr; }
    inline bool checktree() { return (tree != nullptr); }
    inline void update() {
        //cerr << "PyHashTree::update()" << "\n";
        if (!checktree()) { return; }
        if (checkptr()) { treevector.reset(); }
        MVP::MVPVector tv = MVP::mvpvector(tree);
        treevector = make_shared<MVP::MVPVector>(tv);
    }
    
    inline MVP::MVPVector *vector() {
        //cerr << "PyHashTree::vector()" << "\n";
        if (!checkptr()) { update(); }
        return treevector.get();
    }
    
    inline size_t size() {
        //cerr << "PyHashTree::size()" << "\n";
        if (!checktree()) { return 0; }
        return vector()->size();
    }
    
    inline unsigned int length() {
        //cerr << "PyHashTree::length()" << "\n";
        return static_cast<unsigned int>(size());
    }
    
    inline MVPDP *datapoint(size_t idx) {
        //cerr << "PyHashTree::datapoint()" << "\n";
        if (!checkptr() || !checktree()) { return NULL; }
        try {
            return &(vector()->at(idx));
        } catch (out_of_range& exc) {
            return NULL;
        }
        return NULL;
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