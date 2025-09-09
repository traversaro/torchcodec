cmake_minimum_required(VERSION 3.18)

project(TorchCodec)

message(FATAL_ERROR "Fake error")

add_subdirectory(src/torchcodec/_core)


option(BUILD_TESTS "Build tests" OFF)
if(BUILD_TESTS)
    enable_testing()
    add_subdirectory(test)
endif()
