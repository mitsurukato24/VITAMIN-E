# CMakeList.txt : CMake project for VITAMIN-E, include source and define
# project specific logic here.
#
cmake_minimum_required (VERSION 3.8)

project ("VITAMIN-E")

find_package(Eigen3 REQUIRED)
include_directories(${Eigen_INCLUDE_DIRS})
link_libraries(Eigen3::Eigen)
find_package(OpenCV REQUIRED)
link_libraries(${OpenCV_LIBS})
find_package(opengv REQUIRED)
include_directories(${opengv_INCLUDE_DIRS})
link_libraries(opengv)

set(CMAKE_CXX_FLAGS_RELEASE "-O3 -s -DNDEBUG -march=native")
set(CMAKE_CXX_STANDARD 17)

# Add source to this project's executable.
include_directories(VITAMIN-E "include/")
add_executable (VITAMIN-E "vitamin-e.cpp")

if (MSVC)
	find_path(SOPHUS_INCLUDE_DIRS "sophus/common.hpp")  # vcpkg
	target_include_directories(VITAMIN-E PUBLIC ${SOPHUS_INCLUDE_DIRS})
	add_compile_options("$<$<C_COMPILER_ID:MSVC>:/utf-8>")
	add_compile_options("$<$<CXX_COMPILER_ID:MSVC>:/utf-8>")
	add_definitions(/arch:SSE2)
else()
	include_directories(VITAMIN-E "thirdparty/Sophus")
endif (MSVC)

# TODO: Add tests and install targets if needed.
# target_link_libraries(VITAMIN-E PUBLIC Eigen3::Eigen)
# target_link_libraries(VITAMIN-E PUBLIC ${OpenCV_LIBS})

