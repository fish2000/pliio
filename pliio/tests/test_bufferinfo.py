
from __future__ import print_function

from basecase import BaseCase

import sys
#from imread import imread
from pprint import pformat
from pliio import PyImgC as imgc

class StructCodeTests(BaseCase):
    
    def _test_bufferinfo_PyCImage(self):
        for im in self.imgc:
            print(pformat(
                im.buffer_info(im), indent=4), file=sys.stderr)
    
    def test_bufferinfo_modulefunc_imread(self):
        for im in self.imread:
            print(pformat(
                imgc.buffer_info(im),
                indent=4), file=sys.stderr)
