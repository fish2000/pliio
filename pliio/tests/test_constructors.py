
from __future__ import print_function

import numpy
from basecase import FilePathCase
from pliio import PyImgC as imgc

dtypes = (
    numpy.uint8,
    numpy.uint32,
    numpy.uint64,
    numpy.float32,
    numpy.float64)

class ConstructorTests(FilePathCase):
    
    def test_empty_constructor(self):
        self.assertIsNotNone(imgc.PyCImage())
    
    def test_constructor_dtypes_only(self):
        for dtype in dtypes:
            self.assertIsNotNone(imgc.PyCImage(dtype=dtype))
    
    def test_empty_constructor_subsequent_file_load(self):
        for pth in self.image_paths:
            im = imgc.PyCImage()
            im.cimg_load(pth)
            self.assertIsNotNone(im)
    
    def test_constructor_with_dtype_subsequent_file_load(self):
        for pth in self.image_paths:
            for dtype in dtypes:
                im = imgc.PyCImage(dtype=dtype)
                im.cimg_load(pth)
                self.assertIsNotNone(im)
    
    def test_constructor_file_path(self):
        for pth in self.image_paths:
            self.assertIsNotNone(
                imgc.PyCImage(pth))
    
    def test_constructor_file_path_dtype_uint8(self):
        for pth in self.image_paths:
            self.assertIsNotNone(
                imgc.PyCImage(pth, dtype=numpy.uint8))
    
    def test_constructor_file_path_dtype_uint32(self):
        for pth in self.image_paths:
            self.assertIsNotNone(
                imgc.PyCImage(pth, dtype=numpy.uint32))
    
    def test_constructor_file_path_dtype_float32(self):
        for pth in self.image_paths:
            self.assertIsNotNone(
                imgc.PyCImage(pth, dtype=numpy.float32))
    
    def test_constructor_file_path_dtype_uint64(self):
        for pth in self.image_paths:
            self.assertIsNotNone(
                imgc.PyCImage(pth, dtype=numpy.uint64))
    
    def test_constructor_file_path_dtype_float64(self):
        for pth in self.image_paths:
            self.assertIsNotNone(
                imgc.PyCImage(pth, dtype=numpy.float64))
    

