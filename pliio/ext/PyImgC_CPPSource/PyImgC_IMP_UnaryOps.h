
#ifndef PyImgC_IMP_UNARY_OPS_H
#define PyImgC_IMP_UNARY_OPS_H

#define HANDLE_UNARY_OP(type, opname, op) { \
        auto out_img = unary_op<type>(self, op); \
        self->assign(out_img); \
        return reinterpret_cast<PyObject *>(self); \
    }

#define PyCImage_UNARY_OP(opname, op) \
static PyObject *PyCImage_##opname(PyObject *smelf) { \
    if (!smelf) { \
        PyErr_SetString(PyExc_ValueError, \
            "Bad argument to unary operation '##opname##'"); \
        return NULL; \
    } \
    PyCImage *self = reinterpret_cast<PyCImage *>(smelf); \
    Py_INCREF(self); \
    SAFE_SWITCH_ON_DTYPE_FOR_UNARY_OP(self->dtype, NULL, opname, op); \
    return NULL; \
}

#endif /// PyImgC_IMP_UNARY_OPS_H