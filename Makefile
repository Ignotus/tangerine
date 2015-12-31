all:
	pushd src
	python setup.py build_ext --inplace
	popd
clean:
	rm src/*.c src/*.so
