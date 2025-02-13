cmake_minimum_required( VERSION 3.22.1 )

project( autograd_nn )

set(CXX_STANDARD 17)
set(CXX_STANDARD_REQUIRED TRUE)

add_executable( autograd_nn 
	src/main.cpp
	src/neural_network.cpp src/neural_network.hpp
	src/img_data.cpp src/img_data.hpp
	src/dataset.cpp src/dataset.hpp
	src/utils.hpp src/utils.cpp
	src/optimizer.hpp src/optimizer.cpp
	src/compute_graph.cpp
)

if(MSVC)
  target_compile_options(autograd_nn PRIVATE /W4 /WX)
else()
  target_compile_options(autograd_nn PRIVATE -Wall -Wextra -Wpedantic -Werror)
endif()

add_subdirectory("eigen")

target_link_libraries(autograd_nn PUBLIC Eigen3::Eigen)