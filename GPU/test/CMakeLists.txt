include(FetchContent)
FetchContent_Declare(
    googletest
    GIT_REPOSITORY https://github.com/google/googletest.git
    GIT_TAG e2239ee6043f73722e7aa812a459f54a28552929
)

FetchContent_MakeAvailable(googletest)

add_executable(${TEST_BINARY} ${CMAKE_CURRENT_LIST_DIR}/main.cpp)
target_sources(${TEST_BINARY} PRIVATE ${CMAKE_CURRENT_LIST_DIR}/tSimulator.cpp)
target_link_libraries(${TEST_BINARY} gtest gtest_main ${SRC_LIBRARY})
add_test(NAME allGoogleTest COMMAND $<TARGET_FILE:${TEST_BINARY}>)