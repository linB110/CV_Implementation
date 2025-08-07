cmake_minimum_required(VERSION 3.0)
project(DBoW2)
include(ExternalProject)

option(BUILD_DBoW2 "Build DBoW2" ON)
option(BUILD_Demo  "Build demo application" ON)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build." FORCE)
  set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
endif()

if(MSVC)
  add_compile_options(/W4)
else()
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# Header files
set(HDRS
  DBoW2/include/DBoW2/BowVector.h
  DBoW2/include/DBoW2/FBrief.h
  DBoW2/include/DBoW2/QueryResults.h
  DBoW2/include/DBoW2/TemplatedDatabase.h
  DBoW2/include/DBoW2/FORB.h
  DBoW2/include/DBoW2/DBoW2.h
  DBoW2/include/DBoW2/FClass.h
  DBoW2/include/DBoW2/FeatureVector.h
  DBoW2/include/DBoW2/ScoringObject.h
  DBoW2/include/DBoW2/TemplatedVocabulary.h)

set(HDRS_DUTILS
  DUtils/Random.h
  DUtils/TimeStamp.h)

# Source files
set(SRCS 
  DBoW2/src/BowVector.cpp
  DBoW2/src/FBrief.cpp
  DBoW2/src/FORB.cpp
  DBoW2/src/FeatureVector.cpp
  DBoW2/src/QueryResults.cpp
  DBoW2/src/ScoringObject.cpp)

set(SRCS_DUTILS
  DUtils/Random.cpp
  DUtils/TimeStamp.cpp)

set(DEPENDENCY_DIR ${CMAKE_CURRENT_BINARY_DIR}/dependencies)
set(DEPENDENCY_INSTALL_DIR ${DEPENDENCY_DIR}/install)

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

if(BUILD_DBoW2)
  set(LIB_SHARED "SHARED")
  if(WIN32)
    set(LIB_SHARED "STATIC")
  endif()

  add_library(${PROJECT_NAME} ${LIB_SHARED} ${SRCS} ${SRCS_DUTILS})

  target_include_directories(${PROJECT_NAME} PUBLIC
    DBoW2/include/DBoW2
    DUtils)

  target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS})
  set_target_properties(${PROJECT_NAME} PROPERTIES CXX_STANDARD 11)
endif()

if(BUILD_Demo)
  add_executable(demo DBoW2/demo/demo.cpp)
  target_link_libraries(demo ${PROJECT_NAME} ${OpenCV_LIBS})
  set_target_properties(demo PROPERTIES CXX_STANDARD 11)
  file(COPY DBoW2/demo/images DESTINATION ${CMAKE_BINARY_DIR}/)
endif()

configure_file(DBoW2/src/DBoW2.cmake.in
  "${PROJECT_BINARY_DIR}/DBoW2Config.cmake" @ONLY)

install(TARGETS ${PROJECT_NAME} DESTINATION ${CMAKE_INSTALL_PREFIX}/lib)

if(BUILD_DBoW2)
  install(DIRECTORY DBoW2/include/DBoW2 DESTINATION ${CMAKE_INSTALL_PREFIX}/include)
  install(DIRECTORY DUtils/include/DUtils DESTINATION ${CMAKE_INSTALL_PREFIX}/include)
endif()

install(FILES "${CMAKE_CURRENT_BINARY_DIR}/DBoW2Config.cmake"
  DESTINATION ${CMAKE_INSTALL_PREFIX}/include/${PROJECT_NAME})

install(FILES "${PROJECT_BINARY_DIR}/DBoW2Config.cmake" 
  DESTINATION ${CMAKE_INSTALL_PREFIX}/lib/cmake/DBoW2/)

install(DIRECTORY ${DEPENDENCY_INSTALL_DIR}/ DESTINATION ${CMAKE_INSTALL_PREFIX} OPTIONAL)

