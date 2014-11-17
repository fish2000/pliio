
from __future__ import print_function

import tempfile, random, sys
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
    
    def test_hashtree_make_datapoints(self):
        for pth in self.image_paths:
            im = imgc.PyCImage(pth)
            dp = hashtree.DataPoint(
                data=im.dct_phash, name=pth)
            self.assertIsNotNone(dp)
            err(dp)
    
    def _test_hashtree_make_hashtree(self):
        tf = tempfile.mktemp(
            suffix='.mvp',
            prefix='pliio-hashtree-test-')
        tree = hashtree.PyHashTree()
        tree.save(tf)
    
    def test_hashtree_build_tree(self):
        tree = hashtree.PyHashTree()
        
        for pth in self.image_paths:
            im = imgc.PyCImage(pth)
            dp = hashtree.DataPoint(
                data=im.dct_phash,
                name=pth, tree=tree)
            self.assertIsNotNone(dp)
            err(dp)
        #err(tree)
        
        tree.save(path="/Users/fish/Desktop/hash-tree.mvp")
        #newtree = hashtree.PyHashTree()
        #newtree.load("/Users/fish/Desktop/hash-tree.mvp")
        #err(str(newtree))
    
    def test_hashtree_read_tree(self):
        newtree = hashtree.PyHashTree()
        #newtree = hashtree.PyHashTree(tree="/Users/fish/Desktop/hash-tree.mvp")
        newtree.load("/Users/fish/Desktop/hash-tree.mvp")
        #newtree.load()
        err(newtree)
        self.assertIsNotNone(repr(newtree))
    
    def test_hashtree_nearest(self):
        newtree = hashtree.PyHashTree()
        newtree.load("/Users/fish/Desktop/hash-tree.mvp")
        
        newtree.nearest()

