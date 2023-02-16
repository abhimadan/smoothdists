# Many pieces of this file came from libigl-example-project and gptoolbox for mex compilation.

cmake_minimum_required(VERSION 3.20)
project(smooth_distance LANGUAGES CXX)

set(CMAKE_BUILD_TYPE Release)

list(PREPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)

# libigl
option(LIBIGL_OPENGL              "Use OpenGL"         ON)
option(LIBIGL_GLFW                "Use GLFW"           ON)
option(LIBIGL_IMGUI               "Use ImGui"          ON)
option(LIBIGL_PNG                 "Use PNG"            ON)

find_package(Matlab COMPONENTS MEX_COMPILER MX_LIBRARY ENG_LIBRARY MAT_LIBRARY)
if(Matlab_FOUND)
  option(LIBIGL_RESTRICTED_MATLAB   "Use Matlab"         ON)
endif()

#find_package(LIBIGL REQUIRED QUIET)
include(libigl)

# Add your project files
file(GLOB SRCFILES "src/*.cpp")
file(GLOB INCFILES "include/*.h" "include/*.hpp")
add_library(smooth_dist_cpu STATIC ${SRCFILES} ${INCFILES})
target_link_libraries(smooth_dist_cpu igl::core igl::opengl igl::glfw igl::imgui igl::png)
target_include_directories(smooth_dist_cpu PUBLIC include)

add_executable(scene_builder app/scene_builder.cpp)
target_link_libraries(scene_builder smooth_dist_cpu igl::core igl::opengl igl::glfw igl::imgui igl::png)

add_executable(benchmark_cpu app/benchmark_cpu.cpp)
target_link_libraries(benchmark_cpu smooth_dist_cpu igl::core igl::opengl igl::glfw igl::imgui igl::png)

add_executable(beta_error_test app/beta_error_test.cpp)
target_link_libraries(beta_error_test smooth_dist_cpu igl::core igl::opengl igl::glfw igl::imgui igl::png)

add_executable(barnes_hut_smoothness_test app/barnes_hut_smoothness_test.cpp)
target_link_libraries(barnes_hut_smoothness_test smooth_dist_cpu igl::core igl::opengl igl::glfw igl::imgui igl::png)

add_executable(rigid_body_simulator app/rigid_body_simulator.cpp)
target_link_libraries(rigid_body_simulator smooth_dist_cpu igl::core igl::opengl igl::glfw igl::imgui igl::png)

# mex compilation
if(Matlab_FOUND)
  # Based on gptoolbox
  set( CMAKE_SHARED_LIBRARY_SUFFIX ".${Matlab_MEX_EXTENSION}" )  # set suffix to .mexa64
  set( CMAKE_SHARED_LIBRARY_PREFIX )      # remove the "lib" prefix
  set( CMAKE_SHARED_LIBRARY_CXX_FLAGS )       # remove the -fPIC option. mex does not accept the "-fPIC" option
  set( CMAKE_POSITION_INDEPENDENT_CODE ON)
  set( CMAKE_INSTALL_RPATH "\$ORIGIN" )

  function(compile_each I_MEX_SOURCES I_LIBS )
    foreach(MEX_SOURCE ${I_MEX_SOURCES})
      message(STATUS "MEX_SOURCE: ${MEX_SOURCE}")
      get_filename_component(MEX_OUTPUT ${MEX_SOURCE} NAME_WE)
      add_library(${MEX_OUTPUT} SHARED ${MEX_SOURCE})
      # Not sure if this should be (WIN32) or (MSVC OR MSYS OR MINGW)
      # https://stackoverflow.com/a/40152725/148668
      if(WIN32)
        # https://stackoverflow.com/a/11095194/148668
        set_target_properties(${MEX_OUTPUT} PROPERTIES LINK_FLAGS "/export:mexFunction")
      endif()
      target_link_libraries(${MEX_OUTPUT} ${I_LIBS})
      target_include_directories(${MEX_OUTPUT} PUBLIC ${Matlab_INCLUDE_DIRS})
      #list(APPEND MEX_OUTPUTS ${MEX_OUTPUT})
    endforeach()
  endfunction()

  #file( GLOB MEX_SOURCES *.cpp )
  list(APPEND CORE_LIBS ${Matlab_LIBRARIES})

  add_definitions("-DMEX")
  compile_each("mex/build_distance_bvh.cpp" "${CORE_LIBS};igl::core;igl_restricted::matlab;smooth_dist_cpu")
  compile_each("mex/smooth_min_distance.cpp" "${CORE_LIBS};igl::core;igl_restricted::matlab;smooth_dist_cpu")
endif()

# Cuda compilation
find_package(CUDA)
if(CUDA_FOUND)
  cmake_policy(SET CMP0104 NEW)
  enable_language(CUDA)
  file(GLOB SRCFILES_GPU "src/*.cu")
  set(SRCFILES_GPU_HOST_ONLY ${SRCFILES})
  foreach(F IN LISTS SRCFILES_GPU)
    string(REPLACE ".cu" ".cpp" F_CPP ${F})
    list(REMOVE_ITEM SRCFILES_GPU_HOST_ONLY ${F_CPP})
  endforeach()
  set(CUDA_USE_STATIC_CUDA_RUNTIME ON)
  include_directories(${CUDA_TOOLKIT_INCLUDE})

  add_library(smooth_dist_gpu STATIC ${SRCFILES_GPU} ${SRCFILES_GPU_HOST_ONLY} ${INCFILES})
  target_link_libraries(smooth_dist_gpu igl::core igl::opengl igl::glfw igl::imgui igl::png)
  target_include_directories(smooth_dist_gpu PUBLIC include)
  set_target_properties(smooth_dist_gpu PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

  add_executable(sphere_tracer_gpu app/sphere_tracer_gpu.cu)
  target_link_libraries(sphere_tracer_gpu smooth_dist_gpu igl::core igl::opengl igl::glfw igl::imgui igl::png)
  set_target_properties(sphere_tracer_gpu PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

  add_executable(benchmark_gpu app/benchmark_gpu.cu)
  target_link_libraries(benchmark_gpu smooth_dist_gpu igl::core igl::opengl igl::glfw igl::imgui igl::png)
  set_target_properties(benchmark_gpu PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
endif()
