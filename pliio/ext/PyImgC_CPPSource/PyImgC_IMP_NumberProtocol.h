
#ifndef PyImgC_PYIMGC_IMP_NUMBERPROTOCOL_H
#define PyImgC_PYIMGC_IMP_NUMBERPROTOCOL_H

#include <Python.h>
#include <numpy/ndarrayobject.h>
#include "numpypp/numpy.hpp"
#include "numpypp/dispatch.hpp"
#include "numpypp/utils.hpp"
#include "PyImgC_IMP_Operators.h"

using namespace cimg_library;
using namespace std;

static int PyCImage_Compare(PyObject *smelf, PyObject *smother) {
    if (!smelf || !smother) {
        PyErr_SetString(PyExc_ValueError,
            "Bad comparison arguments");
        return -1;
    }
    
    PyCImage *self = reinterpret_cast<PyCImage *>(smelf);
    PyCImage *other = reinterpret_cast<PyCImage *>(smother);
    auto result = self->compare(other);
    
    switch (result) {
        case -2: {
            PyErr_SetString(PyExc_ValueError,
                "Comparator object typecode mismatch (-2)");
            return -1;
        }
        case -3: {
            PyErr_SetString(PyExc_ValueError,
                "Comparison object typecode mismatch (-3)");
            return -1;
        }
    }
    return result;
}

static int PyCImage_NonZero(PyObject *smelf) {
    if (!smelf) {
        PyErr_SetString(PyExc_ValueError,
            "Bad nonzero argument");
        return -1;
    }
    
    PyCImage *self = reinterpret_cast<PyCImage *>(smelf);
    return self->is_empty() ? 0 : 1; /// NON-zero, so this is backwards-looking
}

/// BINARY OP MACROS -- Here's how these work:
/// PyCImage_BINARY_OP(OP_NAME) is a macro.
/// Invoking this macro will declare an in-place function,
/// named after its argument, e.g.:
///     PyCImage_BINARY_OP(ADD)
/// ... will wind up declaring something named:
///     PyCImage_ADD(PyCObject *self, PyCObject *other);
/// ... This newly-declared function is, itself,
/// a relatively simplistic wrapper around another macro, which
/// uses the OP_NAME from PyCImage_BINARY_OP(OP_NAME) to reference
/// an opcode in the BinaryOp enum:
///     SAFE_SWITCH_ON_DTYPE_FOR_BINARY_OP(
///         dtype, error_return_value, ADD, BinaryOp::ADD)
/// ... which goes through the Rube Goldberg-esque type-switch when invoked,
/// as defined in numpypp/dispatch.hpp -- eventually handing off
/// to a handler macro:
///     HANDLE_BINARY_OP(T, ADD, BinaryOp::ADD)
/// ... which FINALLY has enough type information to call the real function:
///     binary_op<T>(self, other, BinaryOp::ADD);
/// ... but wait, THERE IS MORE: binary_op<T>() has to call binary_op_LHS<T>()
/// (it's another simple wrapper) which then has to do one last type-switch
/// to obtain the RHS type before it is able to call binary_op_RHS<otherT>().
/// This then invokes the final step in the process: a call to the BINARY_OP() macro,
/// which executes a switch on the actual opcode (in this case, BinaryOp::ADD)
/// and does the actual fucking operation math. In a nutshell.

PyCImage_BINARY_OP(ADD);
PyCImage_BINARY_OP(SUBTRACT);
PyCImage_BINARY_OP(MULTIPLY);
PyCImage_BINARY_OP(DIVIDE);
PyCImage_BINARY_OP(REMAINDER);
/* PyCImage_BINARY_OP(DIVMOD); */
/* PyCImage_BINARY_OP(POWER); */
PyCImage_BINARY_OP(LSHIFT);
PyCImage_BINARY_OP(RSHIFT);
PyCImage_BINARY_OP(AND);
PyCImage_BINARY_OP(XOR);
PyCImage_BINARY_OP(OR);

PyCImage_BINARY_OP(INPLACE_ADD);
PyCImage_BINARY_OP(INPLACE_SUBTRACT);
PyCImage_BINARY_OP(INPLACE_MULTIPLY);
PyCImage_BINARY_OP(INPLACE_DIVIDE);
PyCImage_BINARY_OP(INPLACE_REMAINDER);
PyCImage_BINARY_OP(INPLACE_POWER);
PyCImage_BINARY_OP(INPLACE_LSHIFT);
PyCImage_BINARY_OP(INPLACE_RSHIFT);
PyCImage_BINARY_OP(INPLACE_AND);
PyCImage_BINARY_OP(INPLACE_XOR);
PyCImage_BINARY_OP(INPLACE_OR);

PyCImage_BINARY_OP(FLOOR_DIVIDE);
PyCImage_BINARY_OP(TRUE_DIVIDE);
PyCImage_BINARY_OP(INPLACE_FLOOR_DIVIDE);
PyCImage_BINARY_OP(INPLACE_TRUE_DIVIDE);

/// Unary Op Macros
PyCImage_UNARY_OP(NEGATIVE);
PyCImage_UNARY_OP(POSITIVE);
PyCImage_UNARY_OP(INVERT);
PyCImage_UNARY_OP(ABSOLUTE);
PyCImage_UNARY_OP(INT);
PyCImage_UNARY_OP(LONG);
PyCImage_UNARY_OP(FLOAT);
PyCImage_UNARY_OP(INDEX);

static PyNumberMethods PyCImage_NumberMethods = {
    (binaryfunc)PyCImage_ADD,                   /* nb_add */
    (binaryfunc)PyCImage_SUBTRACT,              /* nb_subtract */
    (binaryfunc)PyCImage_MULTIPLY,              /* nb_multiply */
    (binaryfunc)PyCImage_DIVIDE,                /* nb_divide */
    (binaryfunc)PyCImage_REMAINDER,             /* nb_remainder */
    0,                                          /* nb_divmod */
    0, /*(ternaryfunc)PyCImage_POWER,*/         /* nb_power */
    (unaryfunc)PyCImage_NEGATIVE,               /* nb_negative */
    (unaryfunc)PyCImage_POSITIVE,               /* nb_positive */
    (unaryfunc)PyCImage_ABSOLUTE,               /* nb_absolute */
    
    (inquiry)PyCImage_NonZero,                  /* nb_nonzero */
    
    (unaryfunc)PyCImage_INVERT,                 /* nb_invert */
    (binaryfunc)PyCImage_LSHIFT,                /* nb_lshift */
    (binaryfunc)PyCImage_RSHIFT,                /* nb_rshift */
    (binaryfunc)PyCImage_AND,                   /* nb_and */
    (binaryfunc)PyCImage_XOR,                   /* nb_xor */
    (binaryfunc)PyCImage_OR,                    /* nb_or */
    
    0, /*(coercion)PyCImage_COERCE,*/           /* nb_coerce */
    
    (unaryfunc)PyCImage_INT,                    /* nb_int */
    (unaryfunc)PyCImage_LONG,                   /* nb_long */
    (unaryfunc)PyCImage_FLOAT,                  /* nb_float */
    0,                                          /* nb_oct */
    0,                                          /* nb_hex */
    
    (binaryfunc)PyCImage_INPLACE_ADD,           /* nb_inplace_add */
    (binaryfunc)PyCImage_INPLACE_SUBTRACT,      /* nb_inplace_subtract */
    (binaryfunc)PyCImage_INPLACE_MULTIPLY,      /* nb_inplace_multiply */
    (binaryfunc)PyCImage_INPLACE_DIVIDE,        /* nb_inplace_divide */
    (binaryfunc)PyCImage_INPLACE_REMAINDER,     /* nb_inplace_remainder */
    0, /*(ternaryfunc)PyCImage_INPLACE_POWER,*/ /* nb_inplace_power */
    (binaryfunc)PyCImage_INPLACE_LSHIFT,        /* nb_inplace_lshift */
    (binaryfunc)PyCImage_INPLACE_RSHIFT,        /* nb_inplace_rshift */
    (binaryfunc)PyCImage_INPLACE_AND,           /* nb_inplace_and */
    (binaryfunc)PyCImage_INPLACE_XOR,           /* nb_inplace_xor */
    (binaryfunc)PyCImage_INPLACE_OR,            /* nb_inplace_or */
    
    (binaryfunc)PyCImage_FLOOR_DIVIDE,          /* nb_floor_divide */
    (binaryfunc)PyCImage_TRUE_DIVIDE,           /* nb_true_divide */
    (binaryfunc)PyCImage_INPLACE_FLOOR_DIVIDE,  /* nb_inplace_floor_divide */
    (binaryfunc)PyCImage_INPLACE_TRUE_DIVIDE,   /* nb_inplace_true_divide */

    0,                                          /* nb_index */
};

#endif /// PyImgC_PYIMGC_IMP_NUMBERPROTOCOL_H