
from __future__ import print_function

from basecase import BaseCase

#import sys
#from pprint import pformat
#from pliio import PyImgC as imgc

class StringifyTests(BaseCase):
    
    def test_str(self):
        for im in self.imgc:
            self.assertIsNotNone(str(im))
    
    def test_repr(self):
        for im in self.imgc:
            self.assertIsNotNone(repr(im))
