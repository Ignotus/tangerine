all:
	cd src
	python setup.py build_ext --inplace
	cd ..
clean:
	rm src/*.c src/*.so
