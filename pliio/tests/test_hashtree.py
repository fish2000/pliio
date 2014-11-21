
from __future__ import print_function

import tempfile, sys
from os.path import basename, exists
from basecase import FilePathCase
from pliio import imgc, hashtree
from pprint import pformat

dtypes = (
    imgc.uint8,
    imgc.uint32,
    imgc.uint64,
    imgc.float32,
    imgc.float64)

err = lambda *thing: print(pformat(*thing), file=sys.stderr)

class HashTreeTests(FilePathCase):
    
    def _test_hashtree_make_datapoints(self):
        for pth in self.image_paths:
            im = imgc.PyCImage(pth)
            dp = hashtree.DataPoint(
                data=im.dct_phash, name=basename(pth))
            self.assertIsNotNone(dp)
            err(dp)
    
    def test_hashtree_build_tree(self):
        tree = hashtree.PyHashTree()
        
        for pth in self.image_paths:
            im = imgc.PyCImage(pth)
            dp = hashtree.DataPoint(
                data=im.dct_phash,
                name=basename(pth), tree=tree)
            self.assertIsNotNone(dp)
            #err(dp)
        
        err(len(tree))
        tree.save("/Users/fish/Desktop/hash-tree.mvp")
        
        for dp in tree:
            err(dp)
    
    def test_hashtree_read_tree(self):
        pth = "/Users/fish/Desktop/hash-tree.mvp"
        if exists(pth):
            newtree = hashtree.PyHashTree()
            newtree.load(pth)
    
    def test_hashtree_nearest(self):
        tree = hashtree.PyHashTree()
        for pth in self.image_paths:
            im = imgc.PyCImage(pth)
            dp = hashtree.DataPoint(
                data=im.dct_phash,
                name=basename(pth), tree=tree)
        
        tree.save("/Users/fish/Desktop/hash-tree-nearest.mvp")
        err("TREE LENGTH: %s" % len(tree))
        for dp in tree:
            err(dp.nearest(10, radius=100.0))

