cmake_minimum_required(VERSION 3.5)
project(Test)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_subdirectory(KelmanFilter/Cpp)

add_executable(Testfile test.cpp)
target_link_libraries(Testfile KelmanCpp)