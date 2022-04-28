target_sources(${SRC_LIBRARY} PRIVATE ${CMAKE_CURRENT_LIST_DIR}/simulator.cpp)
# set_source_files_properties(${CMAKE_CURRENT_LIST_DIR}/simulator.cpp PROPERTIES LANGUAGE CUDA)
target_include_directories(${SRC_LIBRARY} PUBLIC include)

target_link_libraries(${MAIN_BINARY} PRIVATE ${SRC_LIBRARY})