/* Copyright 2010-2012 (C)
 * Luis Pedro Coelho <luis@luispedro.org>
 * License: MIT
 * Annotated by FI$H 2000
 */

typedef unsigned char uchar;
typedef unsigned short ushort;

#define HANDLE_INTEGER_TYPES() \
    case NPY_BOOL:          HANDLE(bool); break; \
    case NPY_BYTE:          HANDLE(char); break; \
    case NPY_SHORT:         HANDLE(short); break; \
    case NPY_INT:           HANDLE(int); break; \
    case NPY_LONG:          HANDLE(long); break; \
    case NPY_LONGLONG:      HANDLE(long long); break; \
    case NPY_UBYTE:         HANDLE(unsigned char); break; \
    case NPY_USHORT:        HANDLE(unsigned short); break; \
    case NPY_UINT:          HANDLE(unsigned int); break; \
    case NPY_ULONG:         HANDLE(unsigned long); break; \
    case NPY_ULONGLONG:     HANDLE(unsigned long long); break;

#define HANDLE_FLOAT_TYPES() \
    case NPY_FLOAT:         HANDLE(float); break; \
    case NPY_DOUBLE:        HANDLE(double); break; \
    case NPY_LONGDOUBLE:    HANDLE(long double); break;

#define HANDLE_COMPLEX_TYPES() \
    case NPY_CFLOAT:        HANDLE(std::complex<float>); break; \
    case NPY_CDOUBLE:       HANDLE(std::complex<double>); break; \
    case NPY_CLONGDOUBLE:   HANDLE(std::complex<long double>); break;

#define HANDLE_TYPES() \
    HANDLE_INTEGER_TYPES() \
    HANDLE_FLOAT_TYPES()

#define HANDLE_ERROR(typecode) \
    PyErr_Format(PyExc_RuntimeError, "Dispatch on typecode %i failed!", typecode)

#define SAFE_SWITCH_ON_TYPECODE(typecode, error_value) \
    try { \
        switch(typecode) { \
            HANDLE_TYPES();\
            default: \
                            HANDLE_ERROR(typecode); \
                            return error_value; \
        } \
    } \
    CATCH_PYTHON_EXCEPTIONS(error_value)

#define SAFE_SWITCH_ON_DTYPE(dtype, error_value) \
    try { \
        switch(dtype->type_num) { \
            HANDLE_TYPES();\
            default: \
                            HANDLE_ERROR(dtype->type_num); \
                            return error_value; \
        } \
    } \
    CATCH_PYTHON_EXCEPTIONS(error_value)

#define SAFE_SWITCH_ON_TYPES_OF(array) \
    try { \
        switch(PyArray_TYPE(array)) { \
                HANDLE_TYPES();\
                default: \
                PyErr_SetString(PyExc_RuntimeError, "Dispatch on all real types failed!"); \
                return NULL; \
        } \
    } \
    CATCH_PYTHON_EXCEPTIONS(NULL)

#define SAFE_SWITCH_ON_INTEGER_TYPES_OF(array) \
    try { \
        switch(PyArray_TYPE(array)) { \
                HANDLE_INTEGER_TYPES();\
                default: \
                PyErr_SetString(PyExc_RuntimeError, "Dispatch on integer types failed!"); \
                return NULL; \
        } \
    } \
    CATCH_PYTHON_EXCEPTIONS(NULL)

#define SAFE_SWITCH_ON_FLOAT_TYPES_OF(array) \
    try { \
        switch(PyArray_TYPE(array)) { \
                HANDLE_FLOAT_TYPES();\
                default: \
                PyErr_SetString(PyExc_RuntimeError, "Dispatch on float types failed!"); \
                return NULL; \
        } \
    } \
    CATCH_PYTHON_EXCEPTIONS(NULL)

#define SAFE_SWITCH_ON_COMPLEX_TYPES_OF(array) \
    try { \
        switch(PyArray_TYPE(array)) { \
                HANDLE_COMPLEX_TYPES();\
                default: \
                PyErr_SetString(PyExc_RuntimeError, "Dispatch on float types failed!"); \
                return NULL; \
        } \
    } \
    CATCH_PYTHON_EXCEPTIONS(NULL)

#define CATCH_PYTHON_EXCEPTIONS(error_value) \
    catch (const PythonException& pe) { \
        PyErr_SetString(pe.type(), pe.message()); \
        return error_value; \
    } catch (const std::bad_alloc&) { \
        PyErr_NoMemory(); \
        return error_value; \
    }

