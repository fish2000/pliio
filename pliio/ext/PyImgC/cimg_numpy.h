
#ifndef PyImgC_CIMG_NUMPY_PLUGIN_H
#define PyImgC_CIMG_NUMPY_PLUGIN_H

// Check if this CImg<T> instance and a given PyObject* have identical pixel types.
bool not_typecode_of(const PyObject *const pyobject) const {
    if (PyArray_Check(pyobject)) {
        const PyArrayObject *pyarray = reinterpret_cast<const PyArrayObject *>(pyobject);
        unsigned int typecode = static_cast<unsigned int>(PyArray_TYPE(pyarray));
        return ((typecode == NPY_UBYTE      && typeid(T) != typeid(unsigned char)) ||
              (typecode == NPY_CHAR         && typeid(T) != typeid(char)) ||
              (typecode == NPY_USHORT       && typeid(T) != typeid(unsigned short)) ||
              (typecode == NPY_UINT         && typeid(T) != typeid(unsigned int)) ||
              (typecode == NPY_INT          && typeid(T) != typeid(int)) ||
              (typecode == NPY_FLOAT        && typeid(T) != typeid(float)) ||
              (typecode == NPY_DOUBLE       && typeid(T) != typeid(double)));
    }
}

// Given this CImg<T> instance, return the corresponding bit-depth flag for use in the PyArray_Descr struct
int get_npy_typecode() const {
    if (typeid(T) == typeid(unsigned char))  return NPY_UBYTE;
    if (typeid(T) == typeid(char))           return NPY_CHAR;
    if (typeid(T) == typeid(unsigned short)) return NPY_USHORT;
    if (typeid(T) == typeid(short))          return NPY_SHORT;
    if (typeid(T) == typeid(int))            return NPY_INT;
    if (typeid(T) == typeid(float))          return NPY_FLOAT;
    if (typeid(T) == typeid(double))         return NPY_DOUBLE;
    return 0;
}

//----------------------------
// NumPy-to-CImg conversion
//----------------------------
// Copy constructor
CImg(const PyObject *const pyobject):_width(0),_height(0),_depth(0),_spectrum(0),_is_shared(true),_data(0) {
  assign(pyobject);
}

// In-place constructor; the optional flag will be ignored when the number of color channels is less than 3.
CImg<T> &assign(const PyObject *const pyobject) {
  if (!pyobject) return assign();
  if (not_typecode_of(pyobject))
    throw CImgInstanceException(_cimg_instance
                                "assign(const PyObject*) : NumPy array has no corresponding pixel type.",
                                cimg_instance);
  const PyArrayObject *pyarray = reinterpret_cast<const PyArrayObject *>(pyobject);
  const int W = (int)PyArray_DIM(pyarray, 1), H = (int)PyArray_DIM(pyarray, 0);
  const char *const dataPtrI = const_cast<char *>(
      PyArray_BYTES(const_cast<PyArrayObject *>(pyarray)));
  const int nChannels = (int)PyArray_DIM(pyarray, 2);
  //char *const dataPtrC = (char *)_data;
  assign(dataPtrI, W, H, 1, nChannels);
  //
  // const int
  //   byte_depth = (sizeof(T) & 255) >> 3,        // number of bytes per color
  //   widthStepI = PyArray_STRIDE(pyarray, 1),    // to do: handle the case img->origin==1
  //                                               // (currently assumption: img->origin==0)
  //   widthStepC = W * byte_depth,
  //   channelStepC = H * widthStepC;
  //
  Py_INCREF(pyobject);
  return *this;
}

//----------------------------
// CImg-to-NumPy conversion
//----------------------------
// z is the z-coordinate of the CImg slice that one wants to copy.
PyObject* get_pyobject(const unsigned z=0) const {
  const int typecode = get_npy_typecode();
  if (!typecode)
    throw CImgInstanceException(_cimg_instance
                                "get_pyobject() : NPY_TYPES has no corresponding typecode.",
                                cimg_instance);
    if (is_empty())
    throw CImgArgumentException(_cimg_instance
                                "get_pyobject() : Empty CImg instance.",
                                cimg_instance);
  if (z >= _depth)
    throw CImgInstanceException(_cimg_instance
                                "get_pyobject() : Instance has not Z-dimension %u.",
                                cimg_instance,
                                z);
  if (_spectrum > 4)
    cimg::warn(_cimg_instance
               "get_pyobject() : Most NumPy image libraries only support 4 channels -- only the first four will be copied.",
               cimg_instance);

  
  npy_intp dims[3] = {
      (npy_intp)_height,
      (npy_intp)_width,
      (npy_intp)_spectrum
  };
  PyObject *pyobject = PyArray_SimpleNewFromData(3, dims, typecode, (void *)_data);
  Py_INCREF(pyobject);
  return pyobject;
}

#endif
