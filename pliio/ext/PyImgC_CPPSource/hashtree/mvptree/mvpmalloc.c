
#include <stdlib.h>
#include <Python.h>

void *PyMem_Calloc(size_t num, size_t size) {
    void *ptr; size_t total = num * size;
    ptr = PyMem_Malloc(total);
    if (ptr != NULL) { memset(ptr, 0, total); }
    return ptr;
}
