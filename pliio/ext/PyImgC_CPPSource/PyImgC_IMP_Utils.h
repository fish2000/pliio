
#ifndef PyImgC_IMP_UTILS_H
#define PyImgC_IMP_UTILS_H

#include <Python.h>
#include <unistd.h>
#include <string>
#include "PyImgC_Constants.h"
#include "PyImgC_PyCImage.h"
using namespace std;

void *PyMem_Calloc(size_t num, size_t size);

/// path check
bool PyImgC_PathExists(char *path);
bool PyImgC_PathExists(const char *path);
bool PyImgC_PathExists(PyObject *path);
bool PyImgC_PathExists(string path);

PyObject *PyImgC_TemporaryPath(PyObject *self, PyObject *args);
PyObject *PyImgC_GuessType(PyObject *self, PyObject *args);

#endif /// PyImgC_IMP_UTILS_H