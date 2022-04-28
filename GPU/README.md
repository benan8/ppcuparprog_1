# Parallel programming example project
## Project structure
* **[CMakeLists.txt](CMakeLists.txt):** CMake will handle how to build this project
* **[Dockerfile.development](Dockerfile.development):** It creates a Docker container which contains everything which is necessary do develop for OpenMP. It will be updated as we use new technologies
* **[Makefile](Makefile):** It contains all the predefined commands that you'll need for development. If there is another target name after colon, that means it will also run that target. For example the `run` target will also `build` it before. To use it run `make target`, instead of target write one of the followings:
  * `build`: builds your code in release mode to release directory (you can turn off the warning here if some HPC idioms require "dirty" code)
  * `run`: runs the code with some predefined parameters (feel free to change those)
  * `clean`: removes all build directory, use it if something "strange" happens during build
  * `tbuild`: builds the code in debug mode and also build the test codes from the test directory
  * `test`: runs all tests in the test directory
  * `bbuild`: build the code in release and also build the code in the benchmark directory
  * `benchmark`: run the benchmark binary, use the filter with any regex to run a subset of all benchmarks
  * `valgrind`: build the code in debug mode and runs it under valgrind. Valgrind can check for memory leaks and wrong memory access (like out of range access)
  * `testAll`: runs both the `test` and the `valgrind` target
* **[include](include):** contains the header file(s) of the simulation code
* **[src](src):** contains the source files
  *  [CMakeLists.txt](src/CMakeLists.txt): add the local source file(s) to the binary, and adds the [include](include) to the path (makes you able to import)
*  **[benchmark](benchmark):** Contains the code for the benchmark
   *  [CMakeLists.txt](benchmark/CMakeLists.txt): downloads the benchmark library and links it to the benchmark binary
*  **[test](test):** Contains the code for testing
   *  [CMakeLists.txt](test/CMakeLists.txt): downloads the Google test library and links it to the test binary
*  **[.devcontainer](.devcontainer/devcontainer.json):** this describes who to setup the container environment
*  **[.vscode](.vscode/settings.json):** It can contain many settings for different extensions and the vanilla VSCode too, but at the moment the only settings are about how to use the CMake extension. The first part sets the cmake flags, by defult it will build all targets (normal, test and benchmark). The second part is something you might want to change. It describes what programs argument should it pass when you run it in debug mode (from the bottom blue bar)