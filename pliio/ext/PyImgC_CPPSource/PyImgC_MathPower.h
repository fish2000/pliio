
#ifndef PyImgC_MATHPOWER_H
#define PyImgC_MATHPOWER_H

#include <string>
using namespace std;

#define OP(nm) nm

enum class BinaryOp : unsigned int {
    OP(ADD), OP(SUBTRACT), OP(MULTIPLY), OP(DIVIDE), OP(REMAINDER),
    //OP(DIVMOD),
    OP(POWER), OP(LSHIFT), OP(RSHIFT),
    OP(AND), OP(XOR), OP(OR),
    
    OP(INPLACE_ADD), OP(INPLACE_SUBTRACT),
    OP(INPLACE_MULTIPLY), OP(INPLACE_DIVIDE), OP(INPLACE_REMAINDER),
    OP(INPLACE_POWER), OP(INPLACE_LSHIFT), OP(INPLACE_RSHIFT),
    OP(INPLACE_AND), OP(INPLACE_XOR), OP(INPLACE_OR),
    
    OP(FLOOR_DIVIDE), OP(TRUE_DIVIDE),
    OP(INPLACE_FLOOR_DIVIDE), OP(INPLACE_TRUE_DIVIDE)
};

enum class UnaryOp : unsigned int {
    OP(NEGATIVE), OP(POSITIVE), OP(INVERT),
    OP(ABSOLUTE),
    OP(INT), OP(LONG), OP(FLOAT),
    //OP(OCT), OP(HEX),
    OP(INDEX)
};

#define UNARY_OP(self, op) \
    switch (op) { \
        case UnaryOp::NEGATIVE:         { return -self; } \
        case UnaryOp::POSITIVE:         { return self; } \
        case UnaryOp::INVERT:           { return ~self; } \
        case UnaryOp::ABSOLUTE:         { return self.get_abs(); } \
        case UnaryOp::INT:              { return self.assign<int>(self); } \
        case UnaryOp::LONG:             { return self.assign<long>(self); } \
        case UnaryOp::FLOAT:            { return self.assign<float>(self); } \
        case UnaryOp::INDEX:            { return self.assign<long>(self); } \
    } \
    return self;

#define BINARY_OP(self, another, op) \
    switch (op) { \
        case BinaryOp::ADD:                 { return self + another; } \
        case BinaryOp::SUBTRACT:            { return self - another; } \
        case BinaryOp::MULTIPLY:            { return self * another; } \
        case BinaryOp::DIVIDE:              { return self / another; } \
        case BinaryOp::REMAINDER:           { return self % another; } \
        case BinaryOp::POWER:               { return self.get_pow(another); } \
        case BinaryOp::LSHIFT:              { return self << another; } \
        case BinaryOp::RSHIFT:              { return self >> another; } \
        case BinaryOp::AND:                 { return self & another; } \
        case BinaryOp::XOR:                 { return self ^ another; } \
        case BinaryOp::OR:                  { return self | another; } \
        case BinaryOp::INPLACE_ADD:         { return self += another; } \
        case BinaryOp::INPLACE_SUBTRACT:    { return self -= another; } \
        case BinaryOp::INPLACE_MULTIPLY:    { return self *= another; } \
        case BinaryOp::INPLACE_DIVIDE:      { return self /= another; } \
        case BinaryOp::INPLACE_REMAINDER:   { return self %= another; } \
        case BinaryOp::INPLACE_POWER:       { return self.pow(another); } \
        case BinaryOp::INPLACE_LSHIFT:      { return self <<= another; } \
        case BinaryOp::INPLACE_RSHIFT:      { return self >>= another; } \
        case BinaryOp::INPLACE_AND:         { return self &= another; } \
        case BinaryOp::INPLACE_XOR:         { return self ^= another; } \
        case BinaryOp::INPLACE_OR:          { return self |= another; } \
        case BinaryOp::FLOOR_DIVIDE:        { return (self / another).get_round(1, -1); } \
        case BinaryOp::TRUE_DIVIDE:         { return self / another; } \
        case BinaryOp::INPLACE_FLOOR_DIVIDE:{ return (self /= another).round(1, -1); } \
        case BinaryOp::INPLACE_TRUE_DIVIDE: { return self /= another; } \
    } \
    return self;

#define UNARY_OP_TRACE(self, op) \
    cout << "> UNARY_OP_TRACE:\n" \
         << " \tOP = " << static_cast<unsigned int>(op) << "\n" \
         << " \ttypeid(self) = " << typeid(self).name() << "\n"; \
    UNARY_OP(self, op)

#define BINARY_OP_TRACE(self, another, op) \
    cout << "> BINARY_OP_TRACE:\n" \
         << " \tOP = " << static_cast<unsigned int>(op) << "\n" \
         << " \ttypeid(self) = " << typeid(self).name() << "\n" \
         << " \ttypeid(another) = " << typeid(another).name() << "\n"; \
    BINARY_OP(self, another, op)

static PyObject *PyCImage_Repr(PyCImage *pyim);


template <typename selfT>
    selfT unary_op_LHS(PyCImage *self, UnaryOp op) {
        if (!self->dtype) {
            PyErr_Format(PyExc_ValueError,
                "Binary op %i LHS has no dtype", op);
            return selfT();
        }
    #ifdef IMGC_DEBUG
        #define HANDLE(selfT) { \
            cout << PyString_AS_STRING(PyCImage_Repr(self)) << "\n"; \
            auto cm_self = *dynamic_cast<CImg<unsigned char>*>((self->cimage).get()); \
            cout << "{CImg self} -> " \
                 << (const char *)cm_self.value_string('/', 100).data() << "\n"; \
            UNARY_OP_TRACE(cm_self, op); \
        }
        SAFE_SWITCH_ON_DTYPE(self->dtype, selfT());
        #undef HANDLE
    #else
        #define HANDLE(selfT) { \
            auto cm_self = *dynamic_cast<CImg<unsigned char>*>((self->cimage).get()); \
            return UNARY_OP(cm_self, op); \
        }
        SAFE_SWITCH_ON_DTYPE(self->dtype, selfT());
        #undef HANDLE
    #endif
        PyErr_Format(PyExc_ValueError,
            "Failure in unary_op_LHS<T>() with op: %i", op);
        return selfT();
    }

template <typename rT>
CImg<rT> unary_op(PyCImage *self, UnaryOp op) {
    return unary_op_LHS<CImg<rT>>(self, op);
}


template <typename selfT>
selfT binary_op_RHS(PyCImage *self, PyCImage *other, BinaryOp op) {
    if (!other->dtype) {
        PyErr_Format(PyExc_ValueError,
            "Binary op %i RHS has no dtype", op);
        return selfT();
    }
#ifdef IMGC_DEBUG
    #define HANDLE(otherT) { \
        cout << PyString_AS_STRING(PyCImage_Repr(self)) << "\n"; \
        cout << PyString_AS_STRING(PyCImage_Repr(other)) << "\n"; \
        auto cm_self = *dynamic_cast<CImg<unsigned char>*>((self->cimage).get()); \
        auto cm_other = *dynamic_cast<CImg<unsigned char>*>((other->cimage).get()); \
        cout << "{CImg self} -> " \
             << (const char *)cm_self.value_string('/', 100).data() << "\n"; \
        cout << "{CImg other} -> " \
             << (const char *)cm_other.value_string('/', 100).data() << "\n"; \
        BINARY_OP_TRACE(cm_self, cm_other, op); \
    }
    SAFE_SWITCH_ON_DTYPE(other->dtype, selfT());
    #undef HANDLE
#else
    #define HANDLE(otherT) { \
        auto cm_self = *dynamic_cast<CImg<unsigned char>*>((self->cimage).get()); \
        auto cm_other = *dynamic_cast<CImg<unsigned char>*>((other->cimage).get()); \
        return BINARY_OP(cm_self, cm_other, op); \
    }
    SAFE_SWITCH_ON_DTYPE(other->dtype, selfT());
    #undef HANDLE
#endif
    PyErr_Format(PyExc_ValueError,
        "Failure in binary_op_RHS<T>() with op: %i", op);
    return selfT();
}

template <typename otherT>
otherT binary_op_LHS(PyCImage *self, PyCImage *other, BinaryOp op) {
    if (!self->dtype) {
        PyErr_Format(PyExc_ValueError,
            "Binary op %i LHS object has no dtype", op);
        return otherT();
    }
#define HANDLE(type) return binary_op_RHS<CImg<type>>(self, other, op);
    SAFE_SWITCH_ON_DTYPE(self->dtype, otherT());
#undef HANDLE
    PyErr_Format(PyExc_ValueError,
        "Failure in binary_op_LHS() with op: %i", op);
    return otherT();
}

template <typename rT>
CImg<rT> binary_op(PyCImage *self, PyCImage *other, BinaryOp op) {
    return binary_op_LHS<CImg<rT>>(self, other, op);
}

#endif /// PyImgC_MATHPOWER_H