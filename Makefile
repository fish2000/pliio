
clean: clean-pyc clean-cython

distclean: clean-all-pyc clean-so clean-build-artifacts

ext: distclean buildext

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

upload:
	python setup.py sdist upload

.PHONY: clean distclean dist buildext upload
