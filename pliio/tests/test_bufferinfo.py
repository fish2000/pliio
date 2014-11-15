
from __future__ import print_function

from basecase import BaseCase

from pliio import imgc

class BufferInfoTests(BaseCase):
    
    def test_bufferinfo_PyCImage(self):
        for im in self.imgc:
            im.buffer_info()
    
    def test_bufferinfo_modulefunc_imread(self):
        for im in self.imread:
            imgc.buffer_info(im)

