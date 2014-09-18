
from __future__ import print_function

from basecase import BaseCase

import sys
#from imread import imread
from pliio import PyImgC as imgc

class StructCodeTests(BaseCase):
    
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
