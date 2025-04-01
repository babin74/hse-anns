cmake_minimum_required(VERSION 3.31)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -march=native")

project(hse-anns)

find_package(HDF5 REQUIRED COMPONENTS C CXX)
find_package(Catch2 3 REQUIRED)

set(THREADS_PREFER_PTHREAD_FLAG ON)
find_package(Threads REQUIRED)

add_executable(tann src/main.cpp src/drivers/hdf5.cpp src/task/build_and_test.cpp src/index/greedy-net.cpp) # Teraqqq's ANN
target_include_directories(tann PRIVATE src)
target_link_libraries(tann PRIVATE hdf5::hdf5_cpp hdf5::hdf5)
target_link_libraries(tann PRIVATE Threads::Threads)

add_subdirectory(test)
