cmake_minimum_required( VERSION 3.22.1 )

project( ia_framework )

set(CXX_STANDARD 17)
set(CXX_STANDARD_REQUIRED TRUE)

add_executable( ia_framework 
	src/main.cpp
	src/ann.cpp src/ann.hpp
	src/img_data.cpp src/img_data.hpp
	src/knn.cpp src/knn.hpp
	src/utils.hpp src/utils.cpp
)