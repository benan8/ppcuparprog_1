cmake_minimum_required(VERSION 3.15)

# Set the project name to your project name, my project isn't very descriptive
project(acm CXX)
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

add_executable(${MAIN_BINARY} src/main.cpp)
add_library(${SRC_LIBRARY})

# GPU section
find_package(OpenACC REQUIRED)
# include(cmake/mycudautils.cmake)
# set(ARCHITECTURE
#     "AUTO"
#     CACHE STRING "Your target CUDA architecture")
# select_cuda_architectures("${ARCHITECTURE}" ARCH_LIST)
# message(STATUS "CUDA_ARCH: ${ARCH_LIST}")
# set(CMAKE_CUDA_ARCHITECTURES
#     "${ARCH_LIST}"
#     CACHE STRING "CUDA architectures" FORCE)

# function(CUDA_CONVERT_FLAGS EXISTING_TARGET)
#     get_property(
#         old_flags
#         TARGET ${EXISTING_TARGET}
#         PROPERTY INTERFACE_COMPILE_OPTIONS)
#     if(NOT "${old_flags}" STREQUAL "")
#         string(REPLACE ";" "," CUDA_flags "${old_flags}")
#         set_property(
#             TARGET ${EXISTING_TARGET}
#             PROPERTY
#                 INTERFACE_COMPILE_OPTIONS
#                 "$<$<BUILD_INTERFACE:$<COMPILE_LANGUAGE:CXX>>:${old_flags}>$<$<BUILD_INTERFACE:$<COMPILE_LANGUAGE:CUDA>>:-Xcompiler=${CUDA_flags}>"
#         )
#     endif()
# endfunction()

# enable_language(CUDA)
# cuda_convert_flags(project_warnings)
# set_target_properties(${SRC_LIBRARY} PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
# set(OpenACC_CXX_FLAGS "${OpenACC_CXX_FLAGS} -acc -fast -Minfo=acce")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -acc -fast -Minfo=accel -gpu=managed")

# creating the main executable and include the CMakeList file from src to include and link all the files
include_directories(SYSTEM ${MPI_INCLUDE_PATH})
target_link_libraries(${SRC_LIBRARY} PUBLIC project_warnings fmt::fmt OpenMP::OpenMP_CXX ${MPI_CXX_LIBRARIES} OpenACC::OpenACC_CXX)
# set_target_properties(${SRC_LIBRARY} PROPERTIES LINKER_LANGUAGE CUDA)
# set_target_properties(${SRC_LIBRARY} PROPERTIES LANGUAGE CUDA)
include(src/CMakeLists.txt)

# Testing section
option(ENABLE_TESTING "Enable Test Builds" OFF)
if(ENABLE_TESTING)
    set(TEST_BINARY "test${PROJECT_NAME}")
    enable_testing()
    add_subdirectory(test)
endif()

# Benchmark section
option(ENABLE_BENCHMARK "Enable Benchmark Builds" OFF)
if(ENABLE_BENCHMARK)
    set(BENCH_BINARY "bench${PROJECT_NAME}")
    add_subdirectory(benchmark)
endif()
