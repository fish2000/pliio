
#include "mvpvector.hpp"
using namespace std;

namespace MVP {
    
    MVPError _mvpvector(vector<MVPDP> &treevec, MVPTree *tree, Node *node, int lvl) {
        MVPError error = MVP_SUCCESS;
        Node *next_node = node;
        int bf = tree->branchfactor,
            lengthM1 = bf-1,
            lengthM2 = bf,
            fanout = bf*bf,
            idx;
    
        if (next_node) {
            if (next_node->leaf.type == LEAF_NODE) {
                if (next_node->leaf.sv1) {
                    treevec.push_back(next_node->leaf.sv1);
                }
                if (next_node->leaf.sv2) {
                    treevec.push_back(next_node->leaf.sv2);
                }
                for (idx = 0; idx < next_node->leaf.nbpoints; idx++) {
                    treevec.push_back(next_node->leaf.points[idx]);
                }
            } else if (next_node->internal.type == INTERNAL_NODE) {
                if (next_node->internal.sv1) {
                    treevec.push_back(next_node->internal.sv1);
                }
                if (next_node->internal.sv2) {
                    treevec.push_back(next_node->internal.sv2);
                }
                // for (idx = 0; idx < lengthM1; idx++) {
                //     treevec.push_back(next_node->internal.M1[idx]);
                // }
                // for (idx = 0; idx < lengthM2; idx++) {
                //     treevec.push_back(next_node->internal.M2[idx]);
                // }
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

    vector<MVPDP> mvpvector(MVPTree *tree) {
        DataPointAllocator<MVPDP> alloc(PyMem_Free, MVP_UINT64ARRAY);
        vector<MVPDP> treevec(alloc);
        if (tree) {
            MVPError error = _mvpvector(&treevec, tree, tree->node, 0);
            if (error != MVP_SUCCESS) {
                //fprintf(stream,"malformed tree: %s\n", mvp_errstr(err));
            }
        }
        return treevec;
    }

} /// namespace MVP
