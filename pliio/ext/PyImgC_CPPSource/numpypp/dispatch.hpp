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

#define HANDLE_TYPES_FOR_BINARY_OP(opname) \
    case NPY_BOOL:          HANDLE_BINARY_OP(bool, opname); break; \
    case NPY_BYTE:          HANDLE_BINARY_OP(char, opname); break; \
    case NPY_SHORT:         HANDLE_BINARY_OP(short, opname); break; \
    case NPY_INT:           HANDLE_BINARY_OP(int, opname); break; \
    case NPY_LONG:          HANDLE_BINARY_OP(long, opname); break; \
    case NPY_LONGLONG:      HANDLE_BINARY_OP(long long, opname); break; \
    case NPY_UBYTE:         HANDLE_BINARY_OP(unsigned char, opname); break; \
    case NPY_USHORT:        HANDLE_BINARY_OP(unsigned short, opname); break; \
    case NPY_UINT:          HANDLE_BINARY_OP(unsigned int, opname); break; \
    case NPY_ULONG:         HANDLE_BINARY_OP(unsigned long, opname); break; \
    case NPY_ULONGLONG:     HANDLE_BINARY_OP(unsigned long long, opname); break; \
    case NPY_FLOAT:         HANDLE_BINARY_OP(float, opname); break; \
    case NPY_DOUBLE:        HANDLE_BINARY_OP(double, opname); break; \
    case NPY_LONGDOUBLE:    HANDLE_BINARY_OP(long double, opname); break;

#define HANDLE_TYPES_FOR_UNARY_OP(opname) \
    case NPY_BOOL:          HANDLE_UNARY_OP(bool, opname); break; \
    case NPY_BYTE:          HANDLE_UNARY_OP(char, opname); break; \
    case NPY_SHORT:         HANDLE_UNARY_OP(short, opname); break; \
    case NPY_INT:           HANDLE_UNARY_OP(int, opname); break; \
    case NPY_LONG:          HANDLE_UNARY_OP(long, opname); break; \
    case NPY_LONGLONG:      HANDLE_UNARY_OP(long long, opname); break; \
    case NPY_UBYTE:         HANDLE_UNARY_OP(unsigned char, opname); break; \
    case NPY_USHORT:        HANDLE_UNARY_OP(unsigned short, opname); break; \
    case NPY_UINT:          HANDLE_UNARY_OP(unsigned int, opname); break; \
    case NPY_ULONG:         HANDLE_UNARY_OP(unsigned long, opname); break; \
    case NPY_ULONGLONG:     HANDLE_UNARY_OP(unsigned long long, opname); break; \
    case NPY_FLOAT:         HANDLE_UNARY_OP(float, opname); break; \
    case NPY_DOUBLE:        HANDLE_UNARY_OP(double, opname); break; \
    case NPY_LONGDOUBLE:    HANDLE_UNARY_OP(long double, opname); break;


#define HANDLE_ERROR(typecode) \
    PyErr_Format(PyExc_RuntimeError, "Dispatch on typecode %i failed!", typecode)

#define SAFE_SWITCH_ON_TYPECODE(typecode, error_value) \
    try { \
        switch(typecode) { \
            HANDLE_TYPES(); \
            default: \
                            HANDLE_ERROR(typecode); \
                            return error_value; \
        } \
    } \
    CATCH_PYTHON_EXCEPTIONS(error_value)

#define SAFE_SWITCH_ON_DTYPE(dtype, error_value) \
    try { \
        switch(dtype->type_num) { \
            HANDLE_TYPES(); \
            default: \
                            HANDLE_ERROR(dtype->type_num); \
                            return error_value; \
        } \
    } \
    CATCH_PYTHON_EXCEPTIONS(error_value)

#define VOID_SWITCH_ON_DTYPE(dtype) \
    switch(dtype->type_num) { \
        HANDLE_TYPES(); \
    }

#define SAFE_SWITCH_ON_DTYPE_FOR_BINARY_OP(dtype, error_value, opname) \
    try { \
        switch(dtype->type_num) { \
            HANDLE_TYPES_FOR_BINARY_OP(opname); \
            default: \
                            HANDLE_ERROR(dtype->type_num); \
                            return error_value; \
        } \
    } \
    CATCH_PYTHON_EXCEPTIONS(error_value)

#define SAFE_SWITCH_ON_DTYPE_FOR_UNARY_OP(dtype, error_value, opname) \
    try { \
        switch(dtype->type_num) { \
            HANDLE_TYPES_FOR_UNARY_OP(opname); \
            default: \
                            HANDLE_ERROR(dtype->type_num); \
                            return error_value; \
        } \
    } \
    CATCH_PYTHON_EXCEPTIONS(error_value)

#define SAFE_SWITCH_ON_TYPES_OF(array) \
    try { \
        switch(PyArray_TYPE(array)) { \
                HANDLE_TYPES(); \
                default: \
                PyErr_SetString(PyExc_RuntimeError, "Dispatch on all real types failed!"); \
                return NULL; \
        } \
    } \
    CATCH_PYTHON_EXCEPTIONS(NULL)

#define SAFE_SWITCH_ON_INTEGER_TYPES_OF(array) \
    try { \
        switch(PyArray_TYPE(array)) { \
                HANDLE_INTEGER_TYPES(); \
                default: \
                PyErr_SetString(PyExc_RuntimeError, "Dispatch on integer types failed!"); \
                return NULL; \
        } \
    } \
    CATCH_PYTHON_EXCEPTIONS(NULL)

#define SAFE_SWITCH_ON_FLOAT_TYPES_OF(array) \
    try { \
        switch(PyArray_TYPE(array)) { \
                HANDLE_FLOAT_TYPES(); \
                default: \
                PyErr_SetString(PyExc_RuntimeError, "Dispatch on float types failed!"); \
                return NULL; \
        } \
    } \
    CATCH_PYTHON_EXCEPTIONS(NULL)

#define SAFE_SWITCH_ON_COMPLEX_TYPES_OF(array) \
    try { \
        switch(PyArray_TYPE(array)) { \
                HANDLE_COMPLEX_TYPES(); \
                default: \
                PyErr_SetString(PyExc_RuntimeError, "Dispatch on complex types failed!"); \
                return NULL; \
        } \
    } \
    CATCH_PYTHON_EXCEPTIONS(NULL)

#define CATCH_PYTHON_EXCEPTIONS(error_value) \
    catch (const PythonException& pe) { \
        PyErr_Format(PyExc_RuntimeError, \
            "Python Exception: %s (%s)", \
                static_cast<const char *>( \
                    reinterpret_cast<PyTypeObject *>(pe.type())->tp_name), \
                pe.message()); \
        return error_value; \
    } catch (CImgArgumentException& ca) { \
        PyErr_Format(PyExc_AttributeError, \
            "Bad arguments to the CImg library: %s", \
                ca.what()); \
        return error_value; \
    } catch (CImgInstanceException& ca) { \
        PyErr_Format(PyExc_ValueError, \
            "CImg Library Instance Freakout: %s", \
                ca.what()); \
        return error_value; \
    } catch (CImgIOException& ca) { \
        PyErr_Format(PyExc_IOError, \
            "CImg I/O shit the bed: %s", \
                ca.what()); \
        return error_value; \
    } catch (CImgWarningException& ca) { \
        PyErr_Format(PyExc_EnvironmentError, \
            "WARNING! DANGER! CAUTION! %s", ca.what()); \
        return error_value; \
    } catch (const std::bad_alloc& ba) { \
        PyErr_NoMemory(); \
        return error_value; \
    }

