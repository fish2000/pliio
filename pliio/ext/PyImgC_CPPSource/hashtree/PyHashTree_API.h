#ifndef Py_PYHASHTREE_H
#define Py_PYHASHTREE_H

#include "mvptree/mvptree.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Header file for PyHashTree */
/* C API functions:
    PyHashTree_DF_HammingDistance(MVPDP *ptA, MVPDP *ptB) -> float
    PyHashTree_Check(PyObject *putative) -> bool
    DataPoint_AddToTree(DataPoint *self, PyHashTree *tree) -> int
    DataPoint_FromDatum(DataPoint *self, MVPDP *datum) -> PyObject *
*/
#define PyHashTree_DF_HammingDistance_NUM 0
#define PyHashTree_DF_HammingDistance_RETURN float
#define PyHashTree_DF_HammingDistance_PROTO (MVPDP *ptA, MVPDP *ptB)

#define PyHashTree_Check_NUM 1
#define PyHashTree_Check_RETURN bool
#define PyHashTree_Check_PROTO (PyObject *putative)

#define DataPoint_AddToTree_NUM 2
#define DataPoint_AddToTree_RETURN int
#define DataPoint_AddToTree_PROTO (DataPoint *self, PyHashTree *tree)

#define DataPoint_FromDatum_NUM 3
#define DataPoint_FromDatum_RETURN PyObject *
#define DataPoint_FromDatum_PROTO (DataPoint *self, MVPDP *datum)

/* Total number of C API pointers */
#define PyHashTree_API_pointers 4


#ifdef PYHASHTREE_MODULE
/* This section is used when compiling hashtree.m */

static \
    PyHashTree_DF_HammingDistance_RETURN \
    PyHashTree_DF_HammingDistance \
    PyHashTree_DF_HammingDistance_PROTO;

static \
    PyHashTree_Check_RETURN \
    PyHashTree_Check \
    PyHashTree_Check_PROTO;

static \
    DataPoint_AddToTree_RETURN \
    DataPoint_AddToTree \
    DataPoint_AddToTree_PROTO;

static \
    DataPoint_FromDatum_RETURN \
    DataPoint_FromDatum \
    DataPoint_FromDatum_PROTO;

#else
/* This section is used in modules that use hashtree's API */

static void **PyHashTree_API;

#define PyHashTree_DF_HammingDistance \
    (*(PyHashTree_DF_HammingDistance_RETURN \
    (*)PyHashTree_DF_HammingDistance_PROTO) \
    PyHashTree_API[PyHashTree_DF_HammingDistance_NUM])

#define PyHashTree_Check \
    (*(PyHashTree_Check_RETURN \
    (*)PyHashTree_Check_PROTO) \
    PyHashTree_API[PyHashTree_Check_NUM])

#define DataPoint_AddToTree \
    (*(DataPoint_AddToTree_RETURN \
    (*)DataPoint_AddToTree_PROTO) \
    PyHashTree_API[DataPoint_AddToTree_NUM])

#define DataPoint_FromDatum \
    (*(DataPoint_FromDatum_RETURN \
    (*)DataPoint_FromDatum_PROTO) \
    PyHashTree_API[DataPoint_FromDatum_NUM])

/* Return -1 on error, 0 on success.
 * PyCapsule_Import will set an exception if there's an error.
 */
static int PyHashTree_Import(void) {
    PyHashTree_API = (void **)PyCapsule_Import("hashtree._API", 0);
    return (PyHashTree_API != NULL) ? 0 : -1;
}

#endif

#ifdef __cplusplus
}
#endif

#endif /* !defined(Py_PYHASHTREE_H) */