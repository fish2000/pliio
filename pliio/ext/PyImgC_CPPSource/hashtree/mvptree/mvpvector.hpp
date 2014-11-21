
#ifndef _MVPVECTOR_H
#define _MVPVECTOR_H

#include <cstring>
#include <vector>
#include <functional>
#include <stdlib.h>
#include "mvptree.h"
using namespace std;

namespace MVP {
    
    template <typename T>
    struct DataPointAllocator : public allocator<T> {
        typedef T value_type;
        typedef T* pointer;
        typedef const T* const_pointer;
        typedef T& reference;
        typedef const T& const_reference;
        typedef size_t size_type;
        typedef ptrdiff_t difference_type;
        
        template <typename U>
        struct rebind {
            typedef DataPointAllocator<U> other;
        };
        
        inline pointer allocate(size_type n, const void *hint=0) {
            pointer ptr = (pointer)PyMem_Calloc(n, sizeof(T));
            return ptr;
        }
        
        inline void deallocate(pointer ptr, size_type n) {
            freefunc(ptr);
        }
        
        inline void construct(pointer p, const T& val) {
            p->id = (char *)PyMem_Calloc(strlen(val.id), sizeof(char));
            strcpy(p->id, val.id);
            p->data = (void *)PyMem_Calloc(val.datalen, datatype);
            memcpy(&p->data, &val.data, val.datalen*datatype);
            p->datalen = val.datalen;
            p->type = datatype;
            p->path = (float *)PyMem_Malloc(sizeof(float));
            memcpy(&p->path, &val.path, sizeof(float));
        }
        
        inline void destroy(pointer p) {
            //if (p->id) { freefunc(p->id); }
            //if (p->data) { freefunc(p->data); }
        }
        
        DataPointAllocator():
            allocator<T>(),
            datatype(MVP_UINT64ARRAY),
            freefunc((MVPFreeFunc)PyMem_Free)
                { }
        
        DataPointAllocator(MVPFreeFunc f):
            allocator<T>(),
            datatype(MVP_UINT64ARRAY),
            freefunc((MVPFreeFunc)f)
                { }
        
        DataPointAllocator(MVPFreeFunc f, MVPDataType d):
            allocator<T>(),
            datatype(d),
            freefunc((MVPFreeFunc)f)
                { }
        
        DataPointAllocator(const allocator<T> &a):
            allocator<T>(a),
            datatype(MVP_UINT64ARRAY),
            freefunc((MVPFreeFunc)PyMem_Free)
                { }
        
        DataPointAllocator(const DataPointAllocator<T> &a):
            allocator<T>(a),
            datatype(a.datatype),
            freefunc(a.freefunc)
                { }
        
        ~DataPointAllocator() { }
        
    private:
        MVPDataType datatype;
        MVPFreeFunc freefunc;
    };
    
    typedef vector<MVPDP, DataPointAllocator<MVPDP>> MVPVector;
    
    MVPVector mvpvector(MVPTree *tree, MVPFreeFunc f=PyMem_Free, MVPDataType datatype=MVP_UINT64ARRAY);
}

#endif /// _MVPVECTOR_H