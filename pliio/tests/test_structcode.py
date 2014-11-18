
from __future__ import print_function

from basecase import BaseCase

from pliio import imgc

class StructCodeTests(BaseCase):
    
    def test_cimage_test_method(self):
        for im in self.imgc:
            imgc.cimage_test(im)
    
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
    
    def test_structcode_labels(self):
        imgc.structcode_parse('B:r: B:g: B:b:')             # RGB 888
        imgc.structcode_parse('d:X: d:Y: d:Z:')             # XYZ triple-dub
        imgc.structcode_parse('4f')                         # CMYK (unlabled)
        imgc.structcode_parse('xfxfxfxf')                   # CMYK (padded)
        imgc.structcode_parse('xf:C: xf:M: xf:Y: xf:K:')    # CMYK (everything)
    
    def test_memoryview_from_pycimage(self):
        for ci in self.imgc:
            m = memoryview(ci)
            self.assertIsNotNone(m)