add_executable(space space.cpp)
target_link_libraries(space PRIVATE Catch2::Catch2WithMain)
target_include_directories(space PRIVATE ${PROJECT_SOURCE_DIR}/src)
