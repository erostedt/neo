#!/bin/bash

# Check if CMakeLists.txt exists
if [ ! -f "CMakeLists.txt" ]; then
    echo "CMakeLists.txt not found. Exiting..."
    exit 1
fi

# Create a build directory if it doesn't exist
if [ ! -d "build" ]; then
    mkdir build
fi

# Navigate to the build directory
cd build

# Run CMake to configure the project
cmake ..

# Run make to build the project
make

# Check if make was successful
if [ $? -eq 0 ]; then
    echo "Build successful!"
else
    echo "Build failed!"
fi

rm ../compile_commands.json
ln compile_commands.json ..
