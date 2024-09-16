cmake_minimum_required(VERSION 3.25)

project(neo)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(EXAMPLES_PATH ./examples)
set(SRC_PATH ./src)
set(CMAKE_CXX_STANDARD 23)

set(CMAKE_CXX_FLAGS "-O3 -Wall -Wextra")

include_directories(${PROJECT_SOURCE_DIR}/include)
find_package(OpenCL REQUIRED)

file(GLOB examples ${EXAMPLES_PATH}/*)
foreach(example ${examples})
  get_filename_component(example_name ${example} NAME_WE)
  file(GLOB sources ${example}/*.cpp)
  add_executable(${example_name} ${sources})
  target_link_libraries(${example_name} OpenCL::OpenCL)
endforeach()
