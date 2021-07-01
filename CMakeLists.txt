# CMakeList.txt : CMake project for VITAMIN-E, include source and define
# project specific logic here.
#
cmake_minimum_required (VERSION 3.10)

project ("VITAMIN-E")

# set(CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} -Wall -Wextra")
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -g")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3")

set(OpenGL_GL_PREFERENCE GLVND)

find_package(OpenMP REQUIRED)
if(OPENMP_FOUND)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set (CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()
find_package(Eigen3 REQUIRED)
include_directories(${Eigen_INCLUDE_DIRS})
link_libraries(Eigen3::Eigen)
find_package(OpenCV REQUIRED)
link_libraries(${OpenCV_LIBS})
find_package(opengv REQUIRED)
include_directories(${opengv_INCLUDE_DIRS})
link_libraries(opengv)
find_package(g2o REQUIRED)
include_directories(${g2o_INCLUDE_DIR})
link_libraries(${g2o_LIBRARIES})
find_package(Pangolin REQUIRED)
include_directories(${Pangolin_INCLUDE_DIRS})
link_libraries(${Pangolin_LIBRARIES})

set(CMAKE_CXX_FLAGS_RELEASE "-O3 -s -DNDEBUG -march=native")
set(CMAKE_CXX_STANDARD 17)

# Add source to this project's executable.
include_directories("include/")
if (MSVC)
	find_path(SOPHUS_INCLUDE_DIRS "sophus/common.hpp")  # vcpkg
	include_directories(${SOPHUS_INCLUDE_DIRS})
	add_compile_options("$<$<C_COMPILER_ID:MSVC>:/utf-8>")
	add_compile_options("$<$<CXX_COMPILER_ID:MSVC>:/utf-8>")
	add_definitions(/arch:SSE2)
else()
	include_directories("thirdparty/Sophus")
endif (MSVC)


add_executable (VITAMIN-E "vitamin-e.cpp")
# add_executable(test_ba "test/test_ba.cpp")