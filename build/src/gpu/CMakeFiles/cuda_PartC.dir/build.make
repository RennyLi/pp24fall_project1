# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.14

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list


# Suppress display of executed commands.
$(VERBOSE).SILENT:


# A target that is always out of date.
cmake_force:

.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/local/bin/cmake

# The command to remove a file.
RM = /usr/local/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /nfsmnt/119010148/CUHKSZ-CSC4005/project1

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /nfsmnt/119010148/CUHKSZ-CSC4005/project1/build

# Include any dependencies generated for this target.
include src/gpu/CMakeFiles/cuda_PartC.dir/depend.make

# Include the progress variables for this target.
include src/gpu/CMakeFiles/cuda_PartC.dir/progress.make

# Include the compile flags for this target's objects.
include src/gpu/CMakeFiles/cuda_PartC.dir/flags.make

src/gpu/CMakeFiles/cuda_PartC.dir/cuda_PartC_generated_cuda_PartC.cu.o: src/gpu/CMakeFiles/cuda_PartC.dir/cuda_PartC_generated_cuda_PartC.cu.o.depend
src/gpu/CMakeFiles/cuda_PartC.dir/cuda_PartC_generated_cuda_PartC.cu.o: src/gpu/CMakeFiles/cuda_PartC.dir/cuda_PartC_generated_cuda_PartC.cu.o.cmake
src/gpu/CMakeFiles/cuda_PartC.dir/cuda_PartC_generated_cuda_PartC.cu.o: ../src/gpu/cuda_PartC.cu
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --blue --bold --progress-dir=/nfsmnt/119010148/CUHKSZ-CSC4005/project1/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building NVCC (Device) object src/gpu/CMakeFiles/cuda_PartC.dir/cuda_PartC_generated_cuda_PartC.cu.o"
	cd /nfsmnt/119010148/CUHKSZ-CSC4005/project1/build/src/gpu/CMakeFiles/cuda_PartC.dir && /usr/local/bin/cmake -E make_directory /nfsmnt/119010148/CUHKSZ-CSC4005/project1/build/src/gpu/CMakeFiles/cuda_PartC.dir//.
	cd /nfsmnt/119010148/CUHKSZ-CSC4005/project1/build/src/gpu/CMakeFiles/cuda_PartC.dir && /usr/local/bin/cmake -D verbose:BOOL=$(VERBOSE) -D build_configuration:STRING= -D generated_file:STRING=/nfsmnt/119010148/CUHKSZ-CSC4005/project1/build/src/gpu/CMakeFiles/cuda_PartC.dir//./cuda_PartC_generated_cuda_PartC.cu.o -D generated_cubin_file:STRING=/nfsmnt/119010148/CUHKSZ-CSC4005/project1/build/src/gpu/CMakeFiles/cuda_PartC.dir//./cuda_PartC_generated_cuda_PartC.cu.o.cubin.txt -P /nfsmnt/119010148/CUHKSZ-CSC4005/project1/build/src/gpu/CMakeFiles/cuda_PartC.dir//cuda_PartC_generated_cuda_PartC.cu.o.cmake

src/gpu/CMakeFiles/cuda_PartC.dir/__/utils.cpp.o: src/gpu/CMakeFiles/cuda_PartC.dir/flags.make
src/gpu/CMakeFiles/cuda_PartC.dir/__/utils.cpp.o: ../src/utils.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/nfsmnt/119010148/CUHKSZ-CSC4005/project1/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building CXX object src/gpu/CMakeFiles/cuda_PartC.dir/__/utils.cpp.o"
	cd /nfsmnt/119010148/CUHKSZ-CSC4005/project1/build/src/gpu && pgc++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/cuda_PartC.dir/__/utils.cpp.o -c /nfsmnt/119010148/CUHKSZ-CSC4005/project1/src/utils.cpp

src/gpu/CMakeFiles/cuda_PartC.dir/__/utils.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/cuda_PartC.dir/__/utils.cpp.i"
	cd /nfsmnt/119010148/CUHKSZ-CSC4005/project1/build/src/gpu && pgc++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /nfsmnt/119010148/CUHKSZ-CSC4005/project1/src/utils.cpp > CMakeFiles/cuda_PartC.dir/__/utils.cpp.i

src/gpu/CMakeFiles/cuda_PartC.dir/__/utils.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/cuda_PartC.dir/__/utils.cpp.s"
	cd /nfsmnt/119010148/CUHKSZ-CSC4005/project1/build/src/gpu && pgc++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /nfsmnt/119010148/CUHKSZ-CSC4005/project1/src/utils.cpp -o CMakeFiles/cuda_PartC.dir/__/utils.cpp.s

# Object files for target cuda_PartC
cuda_PartC_OBJECTS = \
"CMakeFiles/cuda_PartC.dir/__/utils.cpp.o"

# External object files for target cuda_PartC
cuda_PartC_EXTERNAL_OBJECTS = \
"/nfsmnt/119010148/CUHKSZ-CSC4005/project1/build/src/gpu/CMakeFiles/cuda_PartC.dir/cuda_PartC_generated_cuda_PartC.cu.o"

src/gpu/CMakeFiles/cuda_PartC.dir/cmake_device_link.o: src/gpu/CMakeFiles/cuda_PartC.dir/__/utils.cpp.o
src/gpu/CMakeFiles/cuda_PartC.dir/cmake_device_link.o: src/gpu/CMakeFiles/cuda_PartC.dir/cuda_PartC_generated_cuda_PartC.cu.o
src/gpu/CMakeFiles/cuda_PartC.dir/cmake_device_link.o: src/gpu/CMakeFiles/cuda_PartC.dir/build.make
src/gpu/CMakeFiles/cuda_PartC.dir/cmake_device_link.o: /opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/lib64/libcudart_static.a
src/gpu/CMakeFiles/cuda_PartC.dir/cmake_device_link.o: /usr/lib64/librt.so
src/gpu/CMakeFiles/cuda_PartC.dir/cmake_device_link.o: src/gpu/CMakeFiles/cuda_PartC.dir/dlink.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/nfsmnt/119010148/CUHKSZ-CSC4005/project1/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Linking CUDA device code CMakeFiles/cuda_PartC.dir/cmake_device_link.o"
	cd /nfsmnt/119010148/CUHKSZ-CSC4005/project1/build/src/gpu && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/cuda_PartC.dir/dlink.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
src/gpu/CMakeFiles/cuda_PartC.dir/build: src/gpu/CMakeFiles/cuda_PartC.dir/cmake_device_link.o

.PHONY : src/gpu/CMakeFiles/cuda_PartC.dir/build

# Object files for target cuda_PartC
cuda_PartC_OBJECTS = \
"CMakeFiles/cuda_PartC.dir/__/utils.cpp.o"

# External object files for target cuda_PartC
cuda_PartC_EXTERNAL_OBJECTS = \
"/nfsmnt/119010148/CUHKSZ-CSC4005/project1/build/src/gpu/CMakeFiles/cuda_PartC.dir/cuda_PartC_generated_cuda_PartC.cu.o"

src/gpu/cuda_PartC: src/gpu/CMakeFiles/cuda_PartC.dir/__/utils.cpp.o
src/gpu/cuda_PartC: src/gpu/CMakeFiles/cuda_PartC.dir/cuda_PartC_generated_cuda_PartC.cu.o
src/gpu/cuda_PartC: src/gpu/CMakeFiles/cuda_PartC.dir/build.make
src/gpu/cuda_PartC: /opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/lib64/libcudart_static.a
src/gpu/cuda_PartC: /usr/lib64/librt.so
src/gpu/cuda_PartC: src/gpu/CMakeFiles/cuda_PartC.dir/cmake_device_link.o
src/gpu/cuda_PartC: src/gpu/CMakeFiles/cuda_PartC.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/nfsmnt/119010148/CUHKSZ-CSC4005/project1/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_4) "Linking CXX executable cuda_PartC"
	cd /nfsmnt/119010148/CUHKSZ-CSC4005/project1/build/src/gpu && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/cuda_PartC.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
src/gpu/CMakeFiles/cuda_PartC.dir/build: src/gpu/cuda_PartC

.PHONY : src/gpu/CMakeFiles/cuda_PartC.dir/build

src/gpu/CMakeFiles/cuda_PartC.dir/clean:
	cd /nfsmnt/119010148/CUHKSZ-CSC4005/project1/build/src/gpu && $(CMAKE_COMMAND) -P CMakeFiles/cuda_PartC.dir/cmake_clean.cmake
.PHONY : src/gpu/CMakeFiles/cuda_PartC.dir/clean

src/gpu/CMakeFiles/cuda_PartC.dir/depend: src/gpu/CMakeFiles/cuda_PartC.dir/cuda_PartC_generated_cuda_PartC.cu.o
	cd /nfsmnt/119010148/CUHKSZ-CSC4005/project1/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /nfsmnt/119010148/CUHKSZ-CSC4005/project1 /nfsmnt/119010148/CUHKSZ-CSC4005/project1/src/gpu /nfsmnt/119010148/CUHKSZ-CSC4005/project1/build /nfsmnt/119010148/CUHKSZ-CSC4005/project1/build/src/gpu /nfsmnt/119010148/CUHKSZ-CSC4005/project1/build/src/gpu/CMakeFiles/cuda_PartC.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : src/gpu/CMakeFiles/cuda_PartC.dir/depend

