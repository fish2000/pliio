
from __future__ import print_function

from basecase import BaseCase

class StringifyTests(BaseCase):
    
    def _test_str(self):
        for im in self.imgc:
            self.assertIsNotNone(str(im))
    
    def test_repr(self):
        for im in self.imgc:
            self.assertIsNotNone(repr(im))
