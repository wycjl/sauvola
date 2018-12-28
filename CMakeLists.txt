cmake_minimum_required(VERSION 2.8)
project(sauvola)

set(CMAKE_CXX_STANDARD 11)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake/Modules/")


find_package(PkgConfig)
########### opencv libs
find_package(OpenCV 3.4.3 REQUIRED)
#${OpenCV_LIBS}
include_directories(
        ${OpenCV_INCLUDE_DIRS}
)

AUX_SOURCE_DIRECTORY(. DIR_SRCS)
add_executable(${PROJECT_NAME} ${DIR_SRCS} src/BinaryMethod/hsvThrehold.cpp src/BinaryMethod/hsvThrehold.h src/BinaryMethod/calc.cpp src/BinaryMethod/calc.h)
#链接库
target_link_libraries(${PROJECT_NAME}
        ${OpenCV_LIBS}
        )