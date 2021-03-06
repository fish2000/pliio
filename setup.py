
from __future__ import division, print_function

import sys, os
from pprint import pformat
from clint.textui.colored import red, cyan, white

# SETUPTOOLS
try:
    import setuptools
except:
    print('''
setuptools not found.

On linux, the package is often called python-setuptools''')
    sys.exit(1)

# GOSUB: basicaly `backticks` (cribbed from plotdevice)
def gosub(cmd, on_err=True):
    """ Run a shell command and return the output """
    from subprocess import Popen, PIPE
    shell = isinstance(cmd, basestring)
    proc = Popen(cmd, stdout=PIPE, stderr=PIPE, shell=shell)
    out, err = proc.communicate()
    ret = proc.returncode
    if on_err:
        msg = '%s:\n' % on_err if isinstance(on_err, basestring) else ''
        assert ret==0, msg + (err or out)
    return out, err, ret


# PYTHON & NUMPY INCLUDES
from distutils.sysconfig import get_python_inc
from distutils.spawn import find_executable as which
try:
    import numpy
except ImportError:
    class FakeNumpy(object):
        def get_include(self):
            return "."
    numpy = FakeNumpy()

# VERSION & METADATA
__version__ = "<undefined>"
exec(compile(open('pliio-version.py').read(),
             'pliio-version.py', 'exec'))

long_description = open('README.md').read()

# COMPILATION
DEBUG = os.environ.get('DEBUG', '1')

# LIBS: ENABLED BY DEFAULT
USE_PNG = os.environ.get('USE_PNG', '16')
USE_TIFF = os.environ.get('USE_TIFF', '1')
USE_MAGICKPP = os.environ.get('USE_MAGICKPP', '0')
USE_FFTW3 = os.environ.get('USE_FFTW3', '1')
USE_OPENEXR = os.environ.get('USE_OPENEXR', '0')
USE_LCMS2 = os.environ.get('USE_LCMS2', '0')

# LIBS: disabled
USE_OPENCV = os.environ.get('USE_OPENCV', '0') # libtbb won't link

# 'other, misc'
USE_MINC2 = os.environ.get('USE_MINC2', '0')
USE_FFMPEG = os.environ.get('USE_FFMPEG', '0') # won't even work
USE_LAPACK = os.environ.get('USE_LAPACK', '0') # HOW U MAEK LINKED

undef_macros = []
auxilliary_macros = []
define_macros = []
define_macros.append(
    ('NPY_NO_DEPRECATED_API', 'NPY_1_7_API_VERSION'))
define_macros.append(
    ('__OBJC__', '1'))
define_macros.append(
    ('__OBJC2__', '1'))

if DEBUG:
    undef_macros = ['NDEBUG']
    if int(DEBUG) > 2:
        define_macros.append(
            ('IMGC_DEBUG', DEBUG))
        define_macros.append(
            ('_GLIBCXX_DEBUG', '1'))
        auxilliary_macros.append(
            ('IMGC_DEBUG', DEBUG))
        auxilliary_macros.append(
            ('_GLIBCXX_DEBUG', '1'))

print(red(""" %(s)s DEBUG: %(lv)s %(s)s """ % dict(s='*' * 65, lv=DEBUG)))

include_dirs = [
    numpy.get_include(),
    get_python_inc(plat_specific=1),
    os.path.join(os.getcwd(), 'pliio', 'ext'),
    os.path.join(os.getcwd(), 'pliio', 'ext', 'PyImgC_CPPSource')]

library_dirs = []

for pth in (
    '/usr/local/include',
    '/usr/X11/include'):
    if os.path.isdir(pth):
        include_dirs.append(pth)

for pth in (
    '/usr/local/lib',
    '/usr/X11/lib'):
    if os.path.isdir(pth):
        library_dirs.append(pth)

extensions = {
    'imgc': [
        "pliio/ext/PyImgC_CPPSource/pyimgc.m",
        "pliio/ext/PyImgC_CPPSource/PyImgC_IMP_ObjProtocol.m",
        "pliio/ext/PyImgC_CPPSource/PyImgC_IMP_NumberProtocol.m",
        "pliio/ext/PyImgC_CPPSource/PyImgC_IMP_BufferProtocol.m",
        "pliio/ext/PyImgC_CPPSource/PyImgC_IMP_SequenceProtocol.m",
        "pliio/ext/PyImgC_CPPSource/PyImgC_IMP_StructCodeParse.m",
        "pliio/ext/PyImgC_CPPSource/PyImgC_IMP_PyBufferDict.m",
        "pliio/ext/PyImgC_CPPSource/PyImgC_IMP_GetSet.m",
        "pliio/ext/PyImgC_CPPSource/PyImgC_IMP_PHash.m",
        "pliio/ext/PyImgC_CPPSource/PyImgC_IMP_Utils.m",
        "pliio/ext/PyImgC_CPPSource/numpypp/structcode.cpp",
        "pliio/ext/PyImgC_CPPSource/numpypp/typecode.cpp",
        "pliio/ext/PyImgC_CPPSource/UTI/UTI.m",
        "pliio/ext/PyImgC_CPPSource/ICC/Profile.m",
    ],
    'hashtree': [
        "pliio/ext/PyImgC_CPPSource/hashtree/hashtree.m",
        "pliio/ext/PyImgC_CPPSource/hashtree/fmemopen/fmemopen.c",
        "pliio/ext/PyImgC_CPPSource/hashtree/fmemopen/open_memstream.c",
        "pliio/ext/PyImgC_CPPSource/hashtree/mvptree/mvptree.c",
        "pliio/ext/PyImgC_CPPSource/hashtree/mvptree/mvpvector.cpp",
    ],
}

# the basics
libraries = ['png', 'jpeg', 'z', 'm', 'pthread']
PKG_CONFIG = which('pkg-config')

# the addenda
def parse_config_flags(config, config_flags=None):
    """ Get compiler/linker flags from pkg-config and similar CLI tools """
    if config_flags is None: # need something in there
        config_flags = ['']
    for config_flag in config_flags:
        out, err, ret = gosub(' '.join([config, config_flag]))
        if len(out):
            for flag in out.split():
                if flag.startswith('-L'): # link path
                    if os.path.exists(flag[2:]) and flag[2:] not in library_dirs:
                        library_dirs.append(flag[2:])
                    continue
                if flag.startswith('-l'): # library link name
                    if flag[2:] not in libraries:
                        libraries.append(flag[2:])
                    continue
                if flag.startswith('-D'): # preprocessor define
                    macro = flag[2:].split('=')
                    if macro[0] not in dict(define_macros).keys():
                        if len(macro) < 2:
                            macro.append('1')
                        define_macros.append(tuple(macro))
                    continue
                if flag.startswith('-I'):
                    if os.path.exists(flag[2:]) and flag[2:] not in include_dirs:
                        include_dirs.append(flag[2:])
                    continue

# if we're using it, ask it how to fucking work it
if int(USE_TIFF):
    print(white(""" CImg: TIFF support enabled """))
    parse_config_flags(
        PKG_CONFIG,
        ('libtiff-4 --libs', 'libtiff-4 --cflags'))
    define_macros.append(
        ('cimg_use_tiff', '1'))

if int(USE_PNG):
    print(white(""" CImg: PNG support enabled """))
    libpng_pkg = 'libpng'
    if USE_PNG.strip().endswith('6'):
        libpng_pkg += '16' # use 1.6
    elif USE_PNG.strip().endswith('5'):
        libpng_pkg += '15' # use 1.5
    parse_config_flags(
        PKG_CONFIG, (
            '%s --libs' % libpng_pkg,
            '%s --cflags' % libpng_pkg))
    define_macros.append(
        ('cimg_use_png', '1'))

if int(USE_MAGICKPP):
    print(white(""" CImg: Magick++ support enabled """))
    # Linking to ImageMagick++ calls for a bunch of libraries and paths,
    # all with crazy names that change depending on compile options
    parse_config_flags(
        which('Magick++-config'),
        ('--ldflags', '--cppflags'))
    define_macros.append(
        ('cimg_use_magick', '1'))

if int(USE_MINC2):
    print(white(""" CImg: MINC2 support enabled """))
    # I have no idea what this library does (off by default)
    parse_config_flags(
        PKG_CONFIG,
        ('minc2 --libs', 'minc2 --cflags'))
    define_macros.append(
        ('cimg_use_minc2', '1'))

if int(USE_FFTW3):
    print(white(""" CImg: FFTW3 support enabled """))
    # FFTW3 has been config'd for three pkgs:
    # fftw3 orig, fftwl (long? like long integers?),
    # and fftw3f (floats? fuckery? fiber-rich?) --
    # hence this deceptively non-repetitive flag list:
    parse_config_flags(
        PKG_CONFIG, (
        'fftw3f --libs-only-l',
        'fftw3l --libs-only-l',
        'fftw3 --libs', 'fftw3 --cflags'))
    define_macros.append(
        ('cimg_use_fftw3', '1'))

if int(USE_OPENEXR):
    print(white(""" CImg: OpenEXR support enabled """))
    # Linking OpenEXR pulls in ilmBase, which includes its own
    # math and threading libraries... WATCH OUT!!
    parse_config_flags(
        PKG_CONFIG,
        ('OpenEXR --libs', 'OpenEXR --cflags'))
    define_macros.append(
        ('cimg_use_openexr', '1'))

if int(USE_LCMS2):
    print(white(""" PyImgC: LittleCMS2 support enabled """))
    # Not used by CImg directly -- LittleCMS2 provides color
    # management at the PyCImage level
    parse_config_flags(
        PKG_CONFIG,
        ('lcms2 --libs', 'lcms2 --cflags'))
    define_macros.append(
        ('IMGC_LCMS2', '1'))

if int(USE_OPENCV):
    print(white(""" CImg: OpenCV support enabled """))
    # Linking OpenCV gets you lots more including TBB and IPL,
    # and also maybe ffmpeg, I think somehow
    parse_config_flags(
        PKG_CONFIG,
        ('opencv --libs', 'opencv --cflags'))
    out, err, ret = gosub('brew --prefix tbb')
    if out:
        library_dirs.append(os.path.join(out.strip(), 'lib'))
        include_dirs.append(os.path.join(out.strip(), 'include'))
    define_macros.append(
        ('cimg_use_opencv', '1'))


print(red(""" %(s)s CONFIGURATION: %(s)s """ % dict(s='*' * 65)))
print(cyan(" EXTENSION MODULES:"))
print(cyan(pformat(extensions)))
print(cyan(" DEFINED MACROS:"))
print(cyan(pformat(define_macros)))
print(cyan(" LINKED LIBRARIES:"))
print(cyan(" " + ", ".join(libraries)))

from distutils.extension import Extension
from distutils.core import setup

ext_modules = []
for key, sources in extensions.iteritems():
    ext_modules.append(Extension("pliio.%s" % key,
        libraries=map(
            lambda lib: lib.endswith('.dylib') and lib.split('.')[0] or lib,
                libraries),
        library_dirs=library_dirs,
        include_dirs=include_dirs,
        sources=sources,
        language="objc++",
        undef_macros=undef_macros,
        define_macros=define_macros,
        extra_link_args=[
            '-framework', 'AppKit',
            '-framework', 'Accelerate',
            '-framework', 'Quartz',
            '-framework', 'CoreFoundation',
            '-framework', 'Foundation'],
        extra_compile_args=[
            '-O2',
            '-ObjC++',
            '-std=c++11',
            '-stdlib=libc++',
            '-Werror=unused-command-line-argument',
            '-Wno-unused-function',
            '-Wno-delete-non-virtual-dtor',
            '-Wno-overloaded-virtual', # WARNING WARNING WARNING
            '-Wno-dynamic-class-memaccess', # WARNING WARNING etc
            '-Wno-deprecated-register', # CImg w/OpenEXR throws these
            '-Wno-deprecated-writable-strings',
            '-Qunused-arguments',
        ]))

packages = setuptools.find_packages()
package_dir = { 
    'pliio.tools': 'pliio/tools',
    'pliio.ext': 'pliio/ext',
}
#package_data = { 'pliio/tools': ['pliio/tools/*.*', 'data/pvrsamples/*'] }
package_data = dict()

classifiers = [
    'Development Status :: 4 - Beta',
    'Intended Audience :: Developers',
    'Intended Audience :: Science/Research',
    'Topic :: Multimedia',
    'Topic :: Scientific/Engineering :: Image Recognition',
    'Topic :: Software Development :: Libraries',
    'Programming Language :: Python',
    'Programming Language :: Python :: 3',
    'Programming Language :: C++',
    'License :: OSI Approved :: MIT License']

setup(name='imread',
    version=__version__,
    description='PyImgC: CImg bridge library',
    long_description=long_description,
    author='Alexander Bohn',
    author_email='fish2000@gmail.com',
    license='MIT',
    platforms=['Any'],
    classifiers=classifiers,
    url='http://github.com/fish2000/pliio',
    packages=packages,
    ext_modules=ext_modules,
    package_dir=package_dir,
    package_data=package_data,
    test_suite='nose.collector')
