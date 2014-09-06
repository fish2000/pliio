
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
	#py 'pliio._PyImgC.cimage_test(imread.imread(${IMG}))'
	bpython -c "$(shell echo "'';\
		from imread import imread; \
		from pliio import _PyImgC; \
		_PyImgC.cimage_test(imread('$(IMG)'))" | gsed -e "s/[\\s]+/ /g")"

	
	# Checking _structcode
	py 'pliio._structcode.parse(">BBBB")'

upload:
	python setup.py sdist upload

.PHONY: clean distclean dist buildext upload
