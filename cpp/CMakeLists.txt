cmake_minimum_required(VERSION 3.15)

if(CMAKE_HOST_SYSTEM_NAME STREQUAL "Darwin")
    message(STATUS "Using compiler for MacOS")
    set(CMAKE_CXX_COMPILER "/opt/homebrew/Cellar/gcc/13.2.0/bin/g++-13")
else()
    message(STATUS "Using default compilers because ${CMAKE_HOST_SYSTEM_NAME}")    
endif()

project(cpp_a_star CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(pybind11 REQUIRED)
find_package(OpenMP REQUIRED)

if(CMAKE_HOST_SYSTEM_NAME STREQUAL "Darwin")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS} -stdlib=libstdc++")
else()
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()

pybind11_add_module(cpp_a_star src/main.cpp)

target_include_directories(
    cpp_a_star 
    PUBLIC 
    ${Python3_INCLUDE_DIRS}
)

target_link_libraries(
    cpp_a_star PUBLIC 
    ${Python3_LIBRARIES}
    OpenMP::OpenMP_CXX
    pybind11::module
)
