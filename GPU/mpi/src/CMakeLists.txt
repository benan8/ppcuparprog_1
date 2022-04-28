target_sources(${SRC_LIBRARY} PRIVATE ${CMAKE_CURRENT_LIST_DIR}/bigArray.cpp
                                        ${CMAKE_CURRENT_LIST_DIR}/pi.cpp
                                        ${CMAKE_CURRENT_LIST_DIR}/pingPong.cpp
                                        ${CMAKE_CURRENT_LIST_DIR}/cannon.cpp
                                        ${CMAKE_CURRENT_LIST_DIR}/mpiCart.cpp)
target_include_directories(${SRC_LIBRARY} PUBLIC include)

target_link_libraries(${MAIN_BINARY} PRIVATE ${SRC_LIBRARY})