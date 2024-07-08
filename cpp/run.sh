#!/bin/bash

set -e
set -x

BASEDIR=$(dirname "$0")
pushd "$BASEDIR"

rm -rf build

conan install . --output-folder=build --build=missing --settings=build_type=Release
cd build
cmake .. -DCMAKE_TOOLCHAIN_FILE=./build/Release/generators/conan_toolchain.cmake -DCMAKE_BUILD_TYPE=Release

cmake --build .
cd ..
python test_a_star.py
