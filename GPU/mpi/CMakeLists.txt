cmake_minimum_required(VERSION 3.15)

# Set the project name to your project name, my project isn't very descriptive
project(mpilings CXX)
include(cmake/StandardProjectSettings.cmake)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(SRC_LIBRARY "${PROJECT_NAME}Library")
set(MAIN_BINARY "${PROJECT_NAME}")

# Project warning section, this library will be empty if USE_WARNINGS is OFF
add_library(project_warnings INTERFACE)
option(USE_WARNINGS "To turn on standard warnings" OFF)
if(USE_WARNINGS)
    # standard compiler warnings
    include(cmake/CompilerWarnings.cmake)
    set_project_warnings(project_warnings)
endif()

# Adding fmt library for modern printing
include(FetchContent)

FetchContent_Declare(
    fmtfetch
    GIT_REPOSITORY https://github.com/fmtlib/fmt.git
    GIT_TAG 8.1.1
)

FetchContent_MakeAvailable(fmtfetch)

find_package(OpenMP REQUIRED)
find_package(MPI REQUIRED)

# creating the main executable and include the CMakeList file from src to include and link all the files
add_executable(${MAIN_BINARY} src/main.cpp)
add_library(${SRC_LIBRARY})
target_link_libraries(${SRC_LIBRARY} PUBLIC project_warnings fmt::fmt OpenMP::OpenMP_CXX MPI::MPI_CXX)
include(src/CMakeLists.txt)
