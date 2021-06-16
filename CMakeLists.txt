# CMakeList.txt : CMake project for VITAMIN-E, include source and define
# project specific logic here.
#
cmake_minimum_required (VERSION 3.8)

project ("VITAMIN-E")

find_package(Eigen3 REQUIRED)
# find_package(Sophus REQUIRED)
find_package(OpenCV REQUIRED)
# find_package(Ceres REQUIRED)

set(CMAKE_CXX_FLAGS_RELEASE "-O2")
set(CMAKE_CXX_STANDARD 14)

# Add source to this project's executable.
add_executable (VITAMIN-E "VITAMIN-E.cpp" "VITAMIN-E.h" "Dataset.h")

if (MSVC)
	find_path(SOPHUS_INCLUDE_DIRS "sophus/common.hpp")
	target_include_directories(VITAMIN-E PUBLIC ${SOPHUS_INCLUDE_DIRS})
	add_compile_options("$<$<C_COMPILER_ID:MSVC>:/utf-8>")
	add_compile_options("$<$<CXX_COMPILER_ID:MSVC>:/utf-8>")
	add_definitions(/arch:SSE2)
endif (MSVC)

# TODO: Add tests and install targets if needed.
target_link_libraries(VITAMIN-E PUBLIC Eigen3::Eigen)
target_link_libraries(VITAMIN-E PUBLIC ${OpenCV_LIBS})

