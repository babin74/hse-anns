cmake_minimum_required(VERSION 3.31)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED True)
project(hse-anns)

find_package(HDF5 REQUIRED COMPONENTS C CXX)
find_package(Catch2 3 REQUIRED)

add_executable(tann src/main.cpp src/drivers/hdf5.cpp src/task/build_and_test.cpp) # Teraqqq's ANN
target_include_directories(tann PRIVATE src)
target_link_libraries(tann hdf5::hdf5_cpp hdf5::hdf5)

add_subdirectory(test)