cmake_minimum_required(VERSION 3.8)

project(ProgressiveX LANGUAGES CXX)

# indicate if OPENMP should be enabled
option(USE_OPENMP "Use OPENMP" ON)

# indicate if OPENMP should be enabled
option(CREATE_SAMPLE_PROJECT "Create the Sample Project" ON)

# ==============================================================================
# Check C++17 support
# ==============================================================================
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# ==============================================================================
# Find OpenCV
# ==============================================================================
find_package(OpenCV 3.0 REQUIRED)

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
include_directories (
	${PROJECT_SOURCE_DIR}/include
	${PROJECT_SOURCE_DIR}/graph-cut-ransac/include
)

# ==============================================================================
# Structure: Graph-Cut RANSAC Library
# ==============================================================================
# Set header files for the library
file(GLOB_RECURSE HDRS_GraphCutRANSAC
	"graph-cut-ransac/include/*.h"
) 

# Set source files to be added to the library
file(GLOB_RECURSE SRCS_GraphCutRANSAC 
	"graph-cut-ransac/include/*.cpp"
)

source_group(third_party\\GCOptimization FILES
	graph-cut-ransac/include/GCoptimization.cpp
	graph-cut-ransac/include/LinkedBlockList.cpp
	graph-cut-ransac/include/graph.cpp
	graph-cut-ransac/include/maxflow.cpp
	graph-cut-ransac/include/block.h
	graph-cut-ransac/include/energy.h
	graph-cut-ransac/include/GCoptimization.h
	graph-cut-ransac/include/graph.h
	graph-cut-ransac/include/LinkedBlockList.h
)

source_group(third_party\\Theia FILES
	graph-cut-ransac/include/prosac_sampler.h
	graph-cut-ransac/include/sampler.h
	graph-cut-ransac/include/estimator.h
)

add_library(GraphCutRANSAC STATIC
	${HDRS_GraphCutRANSAC}
	${SRCS_GraphCutRANSAC}
)

target_link_libraries(GraphCutRANSAC
	${OpenCV_LIBS}
)

target_link_libraries(GraphCutRANSAC
	Eigen3::Eigen)

# ==============================================================================
# Structure: Progressive-X Library
# ==============================================================================
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/lib)

# Set header files for the library
file(GLOB_RECURSE HDRS_PROGRESSIVEX
	"include/*.h"
) 

# Set source files to be added to the library
file(GLOB_RECURSE SRCS_PROGRESSIVEX 
	"include/*.cpp"
)

add_library(${PROJECT_NAME} STATIC
	${HDRS_PROGRESSIVEX}
	${SRCS_PROGRESSIVEX}
)

target_link_libraries(${PROJECT_NAME} 
	${OpenCV_LIBS}
)

target_link_libraries (${PROJECT_NAME}
	Eigen3::Eigen)
	
# ==============================================================================
# Structure: Applications
# ==============================================================================
if (CREATE_SAMPLE_PROJECT)
	set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/bin)

	add_executable(SampleProject
		src/main.cpp)
		
	target_link_libraries (SampleProject 
		Eigen3::Eigen)
		
	target_link_libraries(SampleProject 
		${OpenCV_LIBS}
		${PROJECT_NAME}
	)

	add_dependencies(SampleProject ${PROJECT_NAME})
endif (CREATE_SAMPLE_PROJECT)
