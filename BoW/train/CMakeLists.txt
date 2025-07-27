cmake_minimum_required(VERSION 2.8)
project(train_bow)

find_package(OpenCV REQUIRED)

include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(../include)
include_directories(../src)

add_executable(train_bow train_bow.cpp)

target_link_libraries(train_bow 
    ${OpenCV_LIBS} 
    /home/lab605/DBoW2/build/libDBoW2.so
)

add_executable(evaluate_bow evaluate_bow.cpp)

target_link_libraries(evaluate_bow 
    ${OpenCV_LIBS} 
    /home/lab605/DBoW2/build/libDBoW2.so
)

