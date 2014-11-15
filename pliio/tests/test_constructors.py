
from __future__ import print_function

import sys
from basecase import FilePathCase
from pliio import imgc

dtypes = (
    imgc.uint8,
    imgc.uint32,
    imgc.uint64,
    imgc.float32,
    imgc.float64)

class ConstructorTests(FilePathCase):
    
    def test_empty_constructor(self):
        self.assertIsNotNone(imgc.PyCImage())
    
    def test_constructor_dtypes_only(self):
        for dtype in dtypes:
            self.assertIsNotNone(imgc.PyCImage(dtype=dtype))
    
    def test_empty_constructor_subsequent_file_load(self):
        for pth in self.image_paths:
            im = imgc.PyCImage()
            im.load(pth)
            self.assertIsNotNone(im)
    
    def test_constructor_with_dtype_subsequent_file_load(self):
        for pth in self.image_paths:
            for dtype in dtypes:
                im = imgc.PyCImage(dtype=dtype)
                im.load(pth)
                self.assertIsNotNone(im)
    
    def test_constructor_file_path(self):
        i = 0
        for pth in self.image_paths:
            im = imgc.PyCImage(pth)
            self.assertIsNotNone(im)
            #nupth = "%s.jpg" % imgc.temporary_path()
            nupth = "/tmp/000%s.jpg" % i
            print(nupth, file=sys.stderr)
            im.save(nupth, overwrite=True)
            i += 1
    
    def test_constructor_file_path_dtype_uint8(self):
        for pth in self.image_paths:
            self.assertIsNotNone(
                imgc.PyCImage(pth, dtype=imgc.uint8))
    
    def test_constructor_file_path_dtype_uint32(self):
        for pth in self.image_paths:
            self.assertIsNotNone(
                imgc.PyCImage(pth, dtype=imgc.uint32))
    
    def test_constructor_file_path_dtype_float32(self):
        for pth in self.image_paths:
            self.assertIsNotNone(
                imgc.PyCImage(pth, dtype=imgc.float32))
    
    def test_constructor_file_path_dtype_uint64(self):
        for pth in self.image_paths:
            self.assertIsNotNone(
                imgc.PyCImage(pth, dtype=imgc.uint64))
    
    def test_constructor_file_path_dtype_float64(self):
        for pth in self.image_paths:
            self.assertIsNotNone(
                imgc.PyCImage(pth, dtype=imgc.float64))
    

