cmake_minimum_required(VERSION 3.19)
project(usbcam)
set(CMAKE_CXX_STANDARD 14)

add_executable(detect main.cpp)
find_package(OpenCV)
include_directories(${OpenCV_INCLUDE_DIRS})

set(CMAKE_CXX_STANDARD 11)

target_link_libraries(detect ${OpenCV_LIBS})

