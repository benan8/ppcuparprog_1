.PHONY: build run clean tbuild test bbuild benchmark valgrind testAll

build:
	cmake -E make_directory $(CURDIR)/release
	cmake -S $(CURDIR) -B $(CURDIR)/release -DCMAKE_BUILD_TYPE=Release -DUSE_WARNINGS=OFF
	cmake --build $(CURDIR)/release --parallel --verbose

run: build
	mpirun -np 9 ./release/mpilings