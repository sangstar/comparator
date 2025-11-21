cmake_minimum_required(VERSION 4.0)
project(comparator)

set(CMAKE_CXX_STANDARD 17)

find_package(CURL REQUIRED)
find_package(Arrow REQUIRED CONFIG)
find_package(Parquet REQUIRED CONFIG)


Include(FetchContent)

set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -DDEBUG")

FetchContent_Declare(
        Catch2
        GIT_REPOSITORY https://github.com/catchorg/Catch2.git
        GIT_TAG v3.8.1 # or a later release
)

FetchContent_MakeAvailable(Catch2)

FetchContent_Declare(
        nlohmann_json
        GIT_REPOSITORY https://github.com/nlohmann/json.git
        GIT_TAG        v3.11.3
)

FetchContent_MakeAvailable(nlohmann_json)

add_executable(tests tests/test_curl_handler.cpp
        tests/test_response_parser.cpp
        tests/test_parquet_handler.cpp
        tests/test_dataset.cpp
        tests/test_scoring.cpp)

target_link_libraries(tests PRIVATE Catch2::Catch2WithMain CURL::libcurl nlohmann_json::nlohmann_json Arrow::arrow_shared Parquet::parquet_shared comparator_lib)
include(CTest)
include(ParseAndAddCatchTests)
ParseAndAddCatchTests(tests)

file(GLOB SRC_FILES "src/*.cpp")
file(GLOB INCLUDE_FILES "include/*.h")
include_directories(include)

add_library(comparator_lib ${SRC_FILES} ${INCLUDE_FILES})

add_executable(comparator src/main.cpp)

target_link_libraries(comparator_lib PUBLIC CURL::libcurl nlohmann_json::nlohmann_json Arrow::arrow_shared Parquet::parquet_shared nlohmann_json::nlohmann_json)

target_link_libraries(comparator PRIVATE comparator_lib)

