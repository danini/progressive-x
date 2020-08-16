cmake_minimum_required(VERSION 3.1)

project(ProgressiveX LANGUAGES CXX)

# indicate if OPENMP should be enabled
option(USE_OPENMP "Use OPENMP" ON)

# indicate if the sample project should be included
option(CREATE_SAMPLE_PROJECT "Create the Sample Project" ON)

SET (CMAKE_BUILD_TYPE "RELEASE")

IF(CMAKE_COMPILER_IS_GNUCXX)
  SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wno-strict-aliasing")
  SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wextra -Wno-write-strings")
  SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-deprecated -ansi")
  SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -ftree-vectorize -funroll-loops")
ENDIF(CMAKE_COMPILER_IS_GNUCXX)

# ==============================================================================
# Check C++17 support
# ==============================================================================
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# ==============================================================================
# Find OpenCV
# ==============================================================================
find_package(OpenCV)
if (OPENCV_FOUND)
  SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenCV_CXX_FLAGS}")
  	add_definitions(-DUSE_OPENMP)
	set(TRGT_LNK_LBS_ADDITIONAL OpenMP::OpenMP_CXX)
endif (OPENCV_FOUND)

# ==============================================================================
# Find Eigen
# ==============================================================================
find_package(Eigen3 REQUIRED)
set(EIGEN_INCLUDE_DIRS ${EIGEN3_INCLUDE_DIR})

# ==============================================================================
# Find OpenMP
# ==============================================================================
find_package(OpenMP)
if (USE_OPENMP)
	if(NOT OPENMP_FOUND)
		message(FATAL_ERROR "OPENMP not found.")
	endif()
	add_definitions(-DUSE_OPENMP)
	set(TRGT_LNK_LBS_ADDITIONAL OpenMP::OpenMP_CXX)
endif (USE_OPENMP)

# ==============================================================================
# Includes
# ==============================================================================
SET(SOURCE_DIR "src/pyprogressivex")

include_directories (
	${PROJECT_SOURCE_DIR}/graph-cut-ransac/src/pygcransac/include
)

# ==============================================================================
# Structure: Graph-Cut RANSAC Library
# ==============================================================================
# Set header files for the library
file(GLOB_RECURSE HDRS_GraphCutRANSAC
	"graph-cut-ransac/src/pygcransac/include/*.h"
) 

# Set source files to be added to the library
file(GLOB_RECURSE SRCS_GraphCutRANSAC 
	"graph-cut-ransac/src/pygcransac/include/*.cpp"
)

add_library(GraphCutRANSAC SHARED
	${HDRS_GraphCutRANSAC}
	${SRCS_GraphCutRANSAC}
)

target_link_libraries(GraphCutRANSAC
	${OpenCV_LIBS}
	Eigen3::Eigen)

# ==============================================================================
# Structure: Progressive-X Library
# ==============================================================================

# Tell cmake that headers are in alse in source_dir
include_directories(${SOURCE_DIR}/include)

# Set header files for the library
file(GLOB_RECURSE HDRS_PROGRESSIVEX
	"${SOURCE_DIR}/include/*.h"
) 

# Set source files to be added to the library
file(GLOB_RECURSE SRCS_PROGRESSIVEX 
	"${SOURCE_DIR}/include/*.cpp"
)

# ==============================================================================
# Structure: Python binding
# ==============================================================================
# Generate python module
add_subdirectory(lib/pybind11)

pybind11_add_module(pyprogressivex ${SOURCE_DIR}/src/bindings.cpp ${SOURCE_DIR}/src/progressivex_python.cpp  ${HDRS_PROGRESSIVEX} ${SRCS_PROGRESSIVEX} )
target_link_libraries(pyprogressivex PRIVATE  
	${OpenCV_LIBS} 
	Eigen3::Eigen 
	GraphCutRANSAC)

add_library(${PROJECT_NAME} STATIC
	${HDRS_PROGRESSIVEX}
	${SRCS_PROGRESSIVEX}
)

target_link_libraries (${PROJECT_NAME}
	${OpenCV_LIBS}
	Eigen3::Eigen
	GraphCutRANSAC)
	
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/lib)
	
# ==============================================================================
# Structure: Applications
# ==============================================================================
if (CREATE_SAMPLE_PROJECT)
	set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/build)

	add_executable(SampleProject
		examples/cpp_example.cpp)
		
	target_link_libraries(SampleProject PUBLIC
		${OpenCV_LIBS}
		${PROJECT_NAME}
		Eigen3::Eigen
		GraphCutRANSAC
	)

	add_dependencies(SampleProject ${PROJECT_NAME})
endif (CREATE_SAMPLE_PROJECT)