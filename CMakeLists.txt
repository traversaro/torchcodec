cmake_minimum_required(VERSION 3.18)
project(TorchCodec)

# Workaround for https://github.com/conda-forge/conda-forge.github.io/issues/1880
set(PKG_CONFIG_EXECUTABLE $ENV{CONDA_PREFIX}/Library/bin/pkgconf.exe)

add_subdirectory(src/torchcodec/_core)


option(BUILD_TESTS "Build tests" OFF)
if(BUILD_TESTS)
    enable_testing()
    add_subdirectory(test)
endif()
