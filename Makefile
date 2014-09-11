
clean: clean-pyc clean-cython

distclean: clean-all-pyc clean-so clean-build-artifacts

ext: distclean buildext checkext

dist: ext upload

clean-pyc:
	find . -name \*.pyc -print -delete

clean-all-pyc:
	find . -name \*.pyc -print -delete

clean-so:
	rm -f pliio/_*.so

clean-build-artifacts:
	rm -rf build dist pliio.egg-info

buildext:
	DEBUG=3 python setup.py build_ext --inplace

#IMG = "/Users/fish/Downloads/tumblr_mgq73sTl6z1qb9r7fo1_r1_500.jpg"
#IMG = "/Users/fish/Downloads/___17ZYXH2.jpg"
IMG = "/Users/fish/Downloads/__1n68Pmd.jpg"

checkext:
	py 'print(clint.textui.colored.red("%(s)s TESTS: %(s)s" % dict(s="*"*65)))'
	
	# Checking _PyImgC
	bpython -c "$(shell echo "'';\
		from imread import imread; \
		from pliio import _PyImgC; \
		print _PyImgC.cimage_test(imread('$(IMG)'))" | gsed -e "s/[\\s]+/ /g")"
	
	# Checking type constructor and PyCImage.__repr__
	bpython -c "$(shell echo "'';\
		from imread import imread; \
		from pliio import _PyImgC; \
		print repr(_PyImgC.PyCImage(imread('$(IMG)')))" | gsed -e "s/[\\s]+/ /g")"
	
	# Checking PyCImage.__str__
	bpython -c "$(shell echo "'';\
		from imread import imread; \
		from pliio import _PyImgC; \
		print str(_PyImgC.PyCImage(imread('$(IMG)')))[:500]" | gsed -e "s/[\\s]+/ /g")"
	
	# Checking PyCImage.__len__
	bpython -c "$(shell echo "'';\
		from imread import imread; \
		from pliio import _PyImgC; \
		print 'len(PyCImage) = %s' % len(_PyImgC.PyCImage(imread('$(IMG)')))" | gsed -e "s/[\\s]+/ /g")"
	
	# Checking PyCImage[idx]
	bpython -c "$(shell echo "'';\
		from imread import imread; \
		from pliio import _PyImgC; \
		print 'PyCImage[66] = %s' % _PyImgC.PyCImage(imread('$(IMG)'))[66]" | gsed -e "s/[\\s]+/ /g")"
	
	# Checking buffer_info
	bpython -c "$(shell echo "'';\
		from imread import imread; \
		from pliio import _PyImgC as imgc; \
		from pprint import pformat; \
		print pformat(imgc.buffer_info(imread('$(IMG)')))" | gsed -e "s/[\\s]+/ /g")"
	
	# Checking structcode parser
	py 'pliio._PyImgC.structcode_parse(">BBBB")'

upload:
	python setup.py sdist upload

.PHONY: clean distclean dist buildext upload
