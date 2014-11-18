
from unittest2 import TestCase

import sys
#from pprint import pprint
#from imread import imread
from pliio import imgc

from os import listdir, getcwd
from os.path import join, abspath, expanduser

class FilePathCase(TestCase):
    
    def setUp(self):
        self.test_data = join(getcwd(), 'pliio', 'tests', 'testdata')
        self.images = join(self.test_data, 'img')
        self.trees = join(self.test_data, 'tree')
        self.image_paths = map(
            lambda nm: join(expanduser('~fish'), 'Downloads', nm), filter(
                lambda nm: nm.lower().endswith('jpg'),
                    listdir(abspath(self.images))))
        self._image_paths = map(
            lambda nm: join(expanduser('~fish'), 'Downloads', nm), filter(
                lambda nm: nm.lower().endswith('jpg'),
                    listdir(abspath(expanduser(join('~fish', 'Downloads'))))))[:20]

class BaseCase(TestCase):
    
    def setUp(self):
        self.test_data = join(getcwd(), 'pliio', 'tests', 'testdata')
        self.images = join(self.test_data, 'img')
        self.trees = join(self.test_data, 'tree')
        self.image_paths = map(
            lambda nm: join(expanduser('~fish'), 'Downloads', nm), filter(
                lambda nm: nm.lower().endswith('jpg'),
                    listdir(abspath(self.images))))
        self._image_paths = map(
            lambda nm: join(expanduser('~fish'), 'Downloads', nm), filter(
                lambda nm: nm.lower().endswith('jpg'),
                    listdir(abspath(expanduser(join('~fish', 'Downloads'))))))[:20]
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