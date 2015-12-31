all:
	cd src && python setup.py build_ext --inplace
clean:
	rm src/*.c src/*.so
