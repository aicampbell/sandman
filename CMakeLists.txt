cmake_minimum_required(VERSION 3.8)
project(sandman)

set(CMAKE_C_STANDARD 99)

set(SOURCE_FILES main.cu)
add_executable(sandman ${SOURCE_FILES})