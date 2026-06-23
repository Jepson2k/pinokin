cmake_minimum_required(VERSION 3.15)
project(pinokin LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(Python REQUIRED COMPONENTS Interpreter Development.Module)
find_package(nanobind CONFIG REQUIRED)
find_package(pinocchio REQUIRED)

# Pinocchio 3.x ships against coal (renamed hpp-fcl). Find one of them
# for collision checking. coal is preferred (modern); hpp-fcl is the
# legacy fallback. Pinocchio's installed CMake config usually pulls one
# of these in transitively, but linking explicitly makes the symbol
# requirement obvious to the auditwheel / delocate repair step.
find_package(coal QUIET)
if (NOT coal_FOUND)
    find_package(hpp-fcl QUIET)
endif()

nanobind_add_module(_core
    cpp/robot.cpp
    cpp/ik_solver.cpp
    cpp/collision_checker.cpp
    cpp/bindings.cpp
)

target_include_directories(_core PRIVATE cpp)
target_link_libraries(_core PRIVATE pinocchio::pinocchio)

# Prefer the explicit collision-enabled Pinocchio target when available.
if (TARGET pinocchio::pinocchio_collision)
    target_link_libraries(_core PRIVATE pinocchio::pinocchio_collision)
endif()
if (coal_FOUND)
    target_link_libraries(_core PRIVATE coal::coal)
elseif (hpp-fcl_FOUND)
    target_link_libraries(_core PRIVATE hpp-fcl::hpp-fcl)
endif()

# Gate the computeCollisions / computeDistances overloads in Pinocchio's
# headers. Without this define, the symbols won't be visible.
target_compile_definitions(_core PRIVATE PINOCCHIO_WITH_HPP_FCL=1)

install(TARGETS _core LIBRARY DESTINATION pinokin)
