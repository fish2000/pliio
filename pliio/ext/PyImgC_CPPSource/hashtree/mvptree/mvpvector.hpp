
#ifndef _MVPVECTOR_H
#define _MVPVECTOR_H

#include <vector>
#include <functional>
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
            return dp_alloc(datatype);
        }
        
        inline void deallocate(pointer ptr, size_type n) {
            dp_free(ptr, freefunc);
        }
        
        DataPointAllocator():
            allocator<T>(),
            datatype(MVP_UINT64ARRAY),
            freefunc((MVPFreeFunc)free)
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
            freefunc((MVPFreeFunc)free)
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
    
    vector<MVPDP, DataPointAllocator<MVPDP>> mvpvector(MVPTree *tree);
}

#endif /// _MVPVECTOR_H