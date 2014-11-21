
#include <Python.h>
#include <iostream>
#include "mvpvector.hpp"
using namespace std;

namespace MVP {
    
    MVPError _mvpvector(MVPVector &treevec, MVPTree *tree, Node *node, int lvl) {
        MVPError error = MVP_SUCCESS;
        Node *next_node = node;
        int bf = tree->branchfactor,
            fanout = bf*bf,
            idx = 0;
        
        if (next_node) {
            if (next_node->leaf.type == LEAF_NODE) {
                if (next_node->leaf.sv1) {
                    treevec.push_back(*next_node->leaf.sv1);
                }
                if (next_node->leaf.sv2) {
                    treevec.push_back(*next_node->leaf.sv2);
                }
                for (idx = 0; idx < next_node->leaf.nbpoints; idx++) {
                    treevec.push_back(*next_node->leaf.points[idx]);
                }
            } else if (next_node->internal.type == INTERNAL_NODE) {
                if (next_node->internal.sv1) {
                    treevec.push_back(*next_node->internal.sv1);
                }
                if (next_node->internal.sv2) {
                    treevec.push_back(*next_node->internal.sv2);
                }
                for (idx = 0; idx < fanout; idx++) {
                    error = _mvpvector(
                        treevec, tree,
                        (Node *)node->internal.child_nodes[idx],
                        lvl+2);
                    if (error != MVP_SUCCESS) { break; }
                }
            } else {
                error = MVP_UNRECOGNIZED;
            }
        }
        return error;
    }
    
    MVPVector mvpvector(MVPTree *tree, MVPFreeFunc f, MVPDataType datatype) {
        DataPointAllocator<MVPDP> alloc(f, datatype);
        MVPVector treevec(alloc);
        if (tree) {
            MVPError error = _mvpvector(treevec, tree, tree->node, 0);
            if (error != MVP_SUCCESS) {
                //fprintf(stream,"malformed tree: %s\n", mvp_errstr(err));
            }
        }
        treevec.shrink_to_fit();
        return treevec;
    }

} /// namespace MVP
