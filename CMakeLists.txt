cmake_minimum_required(VERSION 3.8)

project(ProgressiveX LANGUAGES CXX)

# indicate if OPENMP should be enabled
option(USE_OPENMP "Use OPENMP" ON)

# indicate if the sample project should be included
option(CREATE_SAMPLE_PROJECT "Create the Sample Project" ON)

# indicate if GFLags should be enabled
option(USE_GFLAGS "Enable Google Flags." ON)

# indicate if GLog should be enabled
option(USE_GLOG "Enable Google Log." ON)

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
# Find GFlags
# ==============================================================================
# if (USE_GFLAGS)
# 	find_package(gflags REQUIRED)
# 	if(NOT gflags_FOUND)
# 		message(FATAL_ERROR "GFlags not found.")
# 	endif()
# endif (USE_GFLAGS)

if (USE_GLOG)
	find_package(glog REQUIRED)
	if(NOT glog_FOUND)
		message(FATAL_ERROR "Glog not found.")
	endif()
endif (USE_GLOG)

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

add_library(GraphCutRANSAC STATIC
	${HDRS_GraphCutRANSAC}
	${SRCS_GraphCutRANSAC}
)

target_link_libraries(GraphCutRANSAC
	${OpenCV_LIBS}
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

SET_TARGET_PROPERTIES(${PROJECT_NAME} PROPERTIES LINKER_LANGUAGE CXX)

target_link_libraries (${PROJECT_NAME}
	${OpenCV_LIBS}
	Eigen3::Eigen
	glog::glog
	gflags
	GraphCutRANSAC)
	
# ==============================================================================
# Structure: Applications
# ==============================================================================
if (CREATE_SAMPLE_PROJECT)
	set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/bin)

	add_executable(SampleProject
		src/main.cpp)
		
	target_link_libraries(SampleProject 
		${PROJECT_NAME}
		${OpenCV_LIBS}
		Eigen3::Eigen
		glog::glog
		gflags
		GraphCutRANSAC
	)

	add_dependencies(SampleProject ${PROJECT_NAME})
endif (CREATE_SAMPLE_PROJECT)
