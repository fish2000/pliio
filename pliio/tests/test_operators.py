
from __future__ import print_function

from basecase import BaseCase

#import sys
#from pprint import pformat
#from pliio import PyImgC as imgc

class StructCodeTests(BaseCase):
    
    def test_unary_ops(self):
        for im in self.imgc:
            negative_im = -im
            self.assertIsNotNone(negative_im)
            
            positive_im = +im
            self.assertIsNotNone(positive_im)
            
            ''' These next ops have to get internally refactored;
                apparently, int(something) has to return
                and ACTUAL INT. Crazy, rite??!
            '''
            #int_im = int(im)
            #self.assertIsNotNone(int_im)
            
            #long_im = long(im)
            #self.assertIsNotNone(long_im)
            
            #float_im = float(im)
            #self.assertIsNotNone(float_im)
    
    def test_binary_ops(self):
        if len(self.imgc) < 2:
            raise ValueError(
                "Need more than %s images to test binary ops!" % len(self.imgc))
        im, im2 = self.imgc[0:2]
        self.assertIsNotNone(im + im2)
        self.assertIsNotNone(im - im2)
        self.assertIsNotNone(im >> im2)
        self.assertIsNotNone(im << im2)
        self.assertIsNotNone(im & im2)
        self.assertIsNotNone(im ^ im2)
        self.assertIsNotNone(im | im2)
    
    def test_comparison(self):
        if len(self.imgc) < 2:
            raise ValueError(
                "Need more than %s images to test binary ops!" % len(self.imgc))
        im, im2 = self.imgc[0:2]
        self.assertFalse(im == im2)
    
    def test_subscript(self):
        for im in self.imgc:
            self.assertIsNotNone(im[66])
    
    def test_len(self):
        for im in self.imgc:
            self.assertIsNotNone(len(im))
