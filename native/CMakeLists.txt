cmake_minimum_required(VERSION 3.15...3.31)
project(${SKBUILD_PROJECT_NAME} LANGUAGES CXX)

find_package(Python 3.12 COMPONENTS Interpreter Development.Module REQUIRED)
execute_process(
  COMMAND "${Python_EXECUTABLE}" -m nanobind --cmake_dir
  OUTPUT_STRIP_TRAILING_WHITESPACE
  OUTPUT_VARIABLE nanobind_ROOT)
find_package(nanobind CONFIG REQUIRED)

# RSB_SANITIZE=address,undefined (or thread) builds an instrumented extension for the
# sanitizer qualification runs (qualification/native_sanitizers.sh).
set(RSB_SANITIZE "" CACHE STRING "Sanitizers for the native extension")

nanobind_add_module(_native NOMINSIZE src/module.cpp)
target_compile_features(_native PRIVATE cxx_std_17)
# No FMA contraction: kernel arithmetic must be reproducible across compilers and
# architectures and bit-identical to the numpy references (clang contracts on arm64,
# GCC contracts by default on x86-64).
target_compile_options(_native PRIVATE -Wall -Wextra -Wno-unused-parameter -ffp-contract=off)
if(RSB_SANITIZE)
  target_compile_options(_native PRIVATE -fsanitize=${RSB_SANITIZE} -fno-omit-frame-pointer -g)
  target_link_options(_native PRIVATE -fsanitize=${RSB_SANITIZE})
endif()
install(TARGETS _native DESTINATION raysimbatch)
