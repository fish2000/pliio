
from __future__ import print_function

import tempfile, sys
from os.path import join, basename, exists
from basecase import FilePathCase
from pliio import imgc, hashtree

dtypes = (
    imgc.uint8,
    imgc.uint32,
    imgc.uint64,
    imgc.float32,
    imgc.float64)

err = lambda *thing: print(*thing, file=sys.stderr)

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
        tree.save(join(self.trees, "hash-tree.mvp"))
        
        for dp in tree:
            err(dp)
    
    def test_hashtree_read_tree(self):
        pth = join(self.trees, "hash-tree.mvp")
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
        
        tree.save(join(self.trees, "hash-tree-nearest.mvp"))
        err("TREE LENGTH: %s" % len(tree))
        for dp in tree:
            err(dp.nearest(5, radius=95.0))

