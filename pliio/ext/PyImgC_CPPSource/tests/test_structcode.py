
from __future__ import print_function

from unittest import TestCase

import numpy
from imread import imread
from pliio import PyImgC as imgc

from os import listdir
from os.path import join, abspath
#root = join(dirname(__file__), '..', 'data')
root = join('/', 'Users', 'fish')
listfiles = lambda *pth: listdir(join(root, *pth))
filesoftype = lambda ext, *pth: filter(
    lambda filepth: filepth.lower().endswith(ext),
        listdir(join(root, *pth)))
path = lambda *pth: abspath(join(root, *pth))

class BaseCase(TestCase):
    
    def setUp(self):
        self.image_paths = map(
            lambda image_file: path('.', image_file),
                filesoftype('jpg', 'Downloads'))
        self.imgc = map(
            lambda image_path: imgc.PyCImage(image_path,
                dtype=numpy.uint8), self.image_paths)
        self.imread = map(
            lambda image_path: imread(image_path),
                self.image_paths)
    
    def test_cimage_test_method(self):
        for im in self.imread:
            print(imgc.cimage_test(im))
    
    def test_simple_structcodes(self):
        print(imgc.structcode_parse('B'))
        print(imgc.structcode_parse('b'))
        print(imgc.structcode_parse('Q'))
        print(imgc.structcode_parse('O'))
        print(imgc.structcode_parse('x'))
        print(imgc.structcode_parse('d'))
        print(imgc.structcode_parse('f'))
    
    def test_less_simple_structcodes(self):
        print(imgc.structcode_parse('>BBBB'))
        print(imgc.structcode_parse('=bb'))
        print(imgc.structcode_parse('@QBQB'))
        print(imgc.structcode_parse('OxOxO'))
        print(imgc.structcode_parse('>??i'))
        print(imgc.structcode_parse('efZfZd'))
        print(imgc.structcode_parse('!IIIIiiii'))
    
    def test_memoryview_from_pycimage(self):
        for ci in self.imgc:
            m = memoryview(ci)
            print(m)
        