
from unittest2 import TestCase

import sys
#from pprint import pprint
from imread import imread
from pliio import imgc

from os import listdir
from os.path import join, abspath, expanduser

class FilePathCase(TestCase):
    
    def setUp(self):
        self._image_paths = map(
            lambda nm: join(expanduser('~fish'), 'Downloads', nm), filter(
                lambda nm: nm.lower().endswith('jpg'),
                    listdir(abspath(expanduser(join('~fish', 'Downloads'))))))[:15]
        self.image_paths = set(self._image_paths)

class BaseCase(TestCase):
    
    def setUp(self):
        self._image_paths = map(
            lambda nm: join(expanduser('~fish'), 'Downloads', nm), filter(
                lambda nm: nm.lower().endswith('jpg'),
                    listdir(abspath(expanduser(join('~fish', 'Downloads'))))))[:15]
        self.image_paths = set(self._image_paths)
        self.imgc = map(
            lambda image_path: imgc.PyCImage(image_path,
                dtype=imgc.uint8), self.image_paths)
        # self.imread = map(
        #     lambda image_path: imread(image_path),
        #         self.image_paths)

def main():
    import nose
    return nose.main()

if __name__ == '__main__':
    sys.exit(main())