cmake_minimum_required(VERSION 3.10)
project(train_bow)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_BUILD_TYPE Release)

# === OpenCV ===
find_package(OpenCV REQUIRED)

# === Include paths ===
include_directories(
    ${OpenCV_INCLUDE_DIRS}
    ${PROJECT_SOURCE_DIR}/../include        # DBoW2 public headers
    ${PROJECT_SOURCE_DIR}/../../DUtils      # Random.h, TimeStamp.h
    /usr/local/include                      # cnpy headers
)

# === Link paths ===
link_directories(
    /usr/local/lib                          # cnpy
    ${PROJECT_SOURCE_DIR}/../../build       # libDBoW2.so
)

# === Executable ===
add_executable(train_bow train_bow.cpp)

# === Link libraries ===
target_link_libraries(train_bow
    DBoW2
    cnpy
    ${OpenCV_LIBS}
)

