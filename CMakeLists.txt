cmake_minimum_required(VERSION 3.15)
project(pinokin LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(Python REQUIRED COMPONENTS Interpreter Development.Module)
find_package(nanobind CONFIG REQUIRED)
find_package(pinocchio REQUIRED)

nanobind_add_module(_core
    cpp/robot.cpp
    cpp/ik_solver.cpp
    cpp/bindings.cpp
)

target_include_directories(_core PRIVATE cpp)
target_link_libraries(_core PRIVATE pinocchio::pinocchio)

install(TARGETS _core LIBRARY DESTINATION pinokin)
