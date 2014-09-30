
from __future__ import print_function

from basecase import BaseCase

#import sys
#from imread import imread
from pliio import PyImgC as imgc
import numpy

class StructCodeTests(BaseCase):
    
    def test_cimage_test_method(self):
        for im in self.imread:
            imgc.cimage_test(im, dtype=numpy.uint8)
    
    def test_simple_structcodes(self):
        imgc.structcode_parse('B')
        imgc.structcode_parse('b')
        imgc.structcode_parse('Q')
        imgc.structcode_parse('O')
        imgc.structcode_parse('x')
        imgc.structcode_parse('d')
        imgc.structcode_parse('f')
    
    def test_less_simple_structcodes(self):
        imgc.structcode_parse('>BBBB')
        imgc.structcode_parse('=bb')
        imgc.structcode_parse('@QBQB')
        imgc.structcode_parse('OxOxO')
        imgc.structcode_parse('>??i')
        imgc.structcode_parse('efZfZd')
        imgc.structcode_parse('!IIIIiiii')
    
    def test_memoryview_from_pycimage(self):
        for ci in self.imgc:
            m = memoryview(ci)
            self.assertIsNotNone(m)