
clean: clean-pyc clean-cython

distclean: clean-all-pyc clean-so clean-build-artifacts

ext: distclean buildext test checkext

prepare: distclean build test

dist: ext upload

clean-pyc:
	find . -name \*.pyc -print -delete

clean-all-pyc:
	find . -name \*.pyc -print -delete

clean-so:
	rm -f pliio/*.so

clean-build-artifacts:
	rm -rf build dist pliio.egg-info

buildext:
	DEBUG=3 python setup.py build_ext --inplace

build:
	DEBUG=0 python setup.py build_ext --inplace

test:
	DEBUG=3 NOSE_REDNOSE=1 python runtests.py
	#DEBUG=3 NOSE_REDNOSE=1 nosetests --verbose

IMG = "/Users/fish/Downloads/tumblr_mgq73sTl6z1qb9r7fo1_r1_500.jpg"
#IMG = "/Users/fish/Downloads/___17ZYXH2.jpg"
#IMG = "/Users/fish/Downloads/__1n68Pmd.jpg"

checkext:
	py 'print(clint.textui.colored.red("%(s)s TESTS: %(s)s" % dict(s="*"*65)))'
	
	# Checking structcode parser
	py 'pliio.imgc.structcode_parse(">BBBB")'
	
	# Checking type constructor and PyCImage.__repr__
	bpython -c "$(shell echo "'';\
		from imread import imread; \
		from pliio import imgc; \
		print repr(imgc.PyCImage(imread('$(IMG)')))" | gsed -e "s/[\\s]+/ /g")"
	
	# Checking PyCImage.__str__
	bpython -c "$(shell echo "'';\
		from imread import imread; \
		from pliio import imgc; \
		print str(imgc.PyCImage(imread('$(IMG)')))[:500]" | gsed -e "s/[\\s]+/ /g")"
	
	# Checking PyCImage.__len__
	bpython -c "$(shell echo "'';\
		from imread import imread; \
		from pliio import imgc; \
		print 'len(PyCImage) = %s' % len(imgc.PyCImage(imread('$(IMG)')))" | gsed -e "s/[\\s]+/ /g")"
	
	# Checking PyCImage[idx]
	bpython -c "$(shell echo "'';\
		from imread import imread; \
		from pliio import imgc; \
		print 'PyCImage[66] = %s' % imgc.PyCImage(imread('$(IMG)'))[66]" | gsed -e "s/[\\s]+/ /g")"
	
	# Checking buffer_info
	bpython -c "$(shell echo "'';\
		from imread import imread; \
		from pliio import imgc; \
		from pprint import pformat; \
		print pformat(imgc.buffer_info(imread('$(IMG)')))" | gsed -e "s/[\\s]+/ /g")"
	
	# Checking PyCImage file loading...
	bpython -c "$(shell echo "'';\
		from pliio import imgc; \
		im = imgc.PyCImage('$(IMG)', dtype=imgc.uint8); \
		print repr(im)" | gsed -e "s/[\\s]+/ /g")"
	
	bpython -c "$(shell echo "'';\
		from pliio import imgc; \
		im = imgc.PyCImage(dtype=imgc.uint8); \
		im.load('${IMG}'); \
		print repr(im)" | gsed -e "s/[\\s]+/ /g")"
		
	bpython -c "$(shell echo "'';\
		from pliio import imgc; \
		im = imgc.PyCImage(); \
		im.load('${IMG}'); \
		print repr(im)" | gsed -e "s/[\\s]+/ /g")"
	
	# Checking comparison function...
	bpython -c "$(shell echo "'';\
		from pliio import imgc; \
		im = imgc.PyCImage(dtype=imgc.uint8); \
		im2 = imgc.PyCImage(dtype=imgc.int32); \
		im.load('${IMG}'); \
		im2.load('${IMG}'); \
		print 'im : %s' % repr(im); \
		print 'im2: %s' % repr(im2); \
		print 'im == im2 -> %s' % str(im == im2)" | gsed -e "s/[\\s]+/ /g")"
	
	# Checking BINARY_OP() implementations... WORKS:
	bpython -c "$(shell echo "'';\
		from pliio import imgc; \
		im = imgc.PyCImage('$(IMG)', dtype=imgc.int32); \
		im2 = imgc.PyCImage('$(IMG)', dtype=imgc.int32); \
		print 'im : %s' % repr(im); \
		print 'im2: %s' % repr(im2); \
		print 'im + im2 = %s' % repr(im + im2); \
		print 'im - im2 = %s' % repr(im - im2); \
		print 'im >> im2 = %s' % repr(im >> im2); \
		print 'im << im2 = %s' % repr(im << im2); \
		print 'im & im2 = %s' % repr(im & im2); \
		print 'im ^ im2 = %s' % repr(im ^ im2); \
		print 'im | im2 = %s' % repr(im | im2)" | gsed -e "s/[\\s]+/ /g")"
		
	bpython -c "$(shell echo "'';\
		from pliio import imgc; \
		im = imgc.PyCImage(dtype=imgc.uint8); \
		im.load('${IMG}'); \
		print im.buffer_info()" | gsed -e "s/[\\s]+/ /g")"
		
	bpython -c "$(shell echo "'';\
		from pliio import imgc; \
		im = imgc.PyCImage(dtype=imgc.uint8); \
		im.load('${IMG}'); \
		print imgc.buffer_info(im)" | gsed -e "s/[\\s]+/ /g")"
		
	bpython -c "$(shell echo "'';\
		from pliio import imgc; \
		im = imgc.PyCImage('${IMG}', dtype=imgc.uint8); \
		print im.shape; \
		print im.strides; \
		print im.ndarray.shape; \
		print im.ndarray.strides; \
		print im.dct_phash; \
		print im.mh_phash; \
		print im.buffer_info()" | gsed -e "s/[\\s]+/ /g")"
		
	bpython -c "$(shell echo "'';\
		from pliio import imgc; \
		im = imgc.PyCImage('${IMG}', dtype=imgc.uint8); \
		print im.shape; \
		print im.strides; \
		print im.ndarray.shape; \
		print im.ndarray.strides; \
		print im.dct_phash; \
		print im.mh_phash; \
		print imgc.buffer_info(im)" | gsed -e "s/[\\s]+/ /g")"

upload:
	python setup.py sdist upload

.PHONY: clean distclean dist buildext upload
