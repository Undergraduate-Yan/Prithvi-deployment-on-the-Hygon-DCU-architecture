cmake_minimum_required(VERSION 3.20)
project(k100_pipeline_runner LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(K100_DTK_ROOT "/opt/dtk-25.04.2" CACHE PATH "Frozen DTK/ONNX Runtime root")
find_path(ONNXRUNTIME_INCLUDE_DIR onnxruntime_cxx_api.h
  PATHS "${K100_DTK_ROOT}/include" REQUIRED NO_DEFAULT_PATH)
find_library(ONNXRUNTIME_LIBRARY onnxruntime
  PATHS "${K100_DTK_ROOT}/lib" REQUIRED NO_DEFAULT_PATH)
find_path(HIP_INCLUDE_DIR hip/hip_runtime.h
  PATHS "${K100_DTK_ROOT}/hip/include" REQUIRED NO_DEFAULT_PATH)
find_library(HIP_RUNTIME_LIBRARY amdhip64
  PATHS "${K100_DTK_ROOT}/hip/lib" "${K100_DTK_ROOT}/lib" REQUIRED NO_DEFAULT_PATH)

# The frozen vendor SDK ships libonnxruntime.so.1.19.2 with SONAME
# libonnxruntime.so.1 but omits that compatibility symlink. Keep the workaround
# inside the build artifact; never mutate the frozen image.
set(K100_RUNTIME_LIB_DIR "${CMAKE_BINARY_DIR}/runtime_lib")
file(MAKE_DIRECTORY "${K100_RUNTIME_LIB_DIR}")
file(CREATE_LINK "${K100_DTK_ROOT}/lib/libonnxruntime.so.1.19.2"
  "${K100_RUNTIME_LIB_DIR}/libonnxruntime.so.1" SYMBOLIC)
foreach(provider shared migraphx rocm)
  file(CREATE_LINK "${K100_DTK_ROOT}/lib/libonnxruntime_providers_${provider}.so"
    "${K100_RUNTIME_LIB_DIR}/libonnxruntime_providers_${provider}.so" SYMBOLIC)
endforeach()
set(CMAKE_BUILD_RPATH
  "\$ORIGIN/runtime_lib;${K100_DTK_ROOT}/lib;${K100_DTK_ROOT}/hip/lib")

add_library(k100_pipeline_runner STATIC
  src/k100_pipeline_runner.cpp)
target_include_directories(k100_pipeline_runner
  PUBLIC include)
target_include_directories(k100_pipeline_runner SYSTEM PRIVATE
  "${ONNXRUNTIME_INCLUDE_DIR}" "${HIP_INCLUDE_DIR}")
target_link_libraries(k100_pipeline_runner
  PUBLIC "${ONNXRUNTIME_LIBRARY}" "${HIP_RUNTIME_LIBRARY}")
target_compile_definitions(k100_pipeline_runner PRIVATE __HIP_PLATFORM_AMD__=1)
target_compile_options(k100_pipeline_runner PRIVATE -Wall -Wextra -Wpedantic -Werror)

add_executable(k100_pipeline_benchmark src/benchmark_main.cpp)
target_include_directories(k100_pipeline_benchmark
  SYSTEM PRIVATE "${ONNXRUNTIME_INCLUDE_DIR}" "${HIP_INCLUDE_DIR}")
target_link_libraries(k100_pipeline_benchmark PRIVATE k100_pipeline_runner)
target_compile_options(k100_pipeline_benchmark PRIVATE -Wall -Wextra -Wpedantic -Werror)

set_target_properties(k100_pipeline_benchmark PROPERTIES
  INSTALL_RPATH "\$ORIGIN/runtime_lib;${K100_DTK_ROOT}/lib;${K100_DTK_ROOT}/hip/lib")
