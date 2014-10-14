
from unittest2 import TestCase

import numpy, sys
from pprint import pprint
from imread import imread
from pliio import PyImgC as imgc

from os import listdir
from os.path import join, abspath, expanduser

class BaseCase(TestCase):
    
    def setUp(self):
        self.image_paths = map(
            lambda nm: join(expanduser('~fish'), 'Downloads', nm), filter(
                lambda nm: nm.lower().endswith('jpg'),
                    listdir(abspath(expanduser(join('~fish', 'Downloads'))))))[:10]
        self.imgc = map(
            lambda image_path: imgc.PyCImage(image_path,
                dtype=numpy.uint8), self.image_paths)
        self.imread = map(
            lambda image_path: imread(image_path),
                self.image_paths)
        pprint(self.image_paths, indent=4)

def main():
    import nose
    return nose.main()

if __name__ == '__main__':
    sys.exit(main())