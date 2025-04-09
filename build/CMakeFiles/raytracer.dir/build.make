# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.28

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:

#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:

# Disable VCS-based implicit rules.
% : %,v

# Disable VCS-based implicit rules.
% : RCS/%

# Disable VCS-based implicit rules.
% : RCS/%,v

# Disable VCS-based implicit rules.
% : SCCS/s.%

# Disable VCS-based implicit rules.
% : s.%

.SUFFIXES: .hpux_make_needs_suffix_list

# Command-line flag to silence nested $(MAKE).
$(VERBOSE)MAKESILENT = -s

#Suppress display of executed commands.
$(VERBOSE).SILENT:

# A target that is always out of date.
cmake_force:
.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/christianw/raytracer-cuda

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/christianw/raytracer-cuda/build

# Include any dependencies generated for this target.
include CMakeFiles/raytracer.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include CMakeFiles/raytracer.dir/compiler_depend.make

# Include the progress variables for this target.
include CMakeFiles/raytracer.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/raytracer.dir/flags.make

CMakeFiles/raytracer.dir/src/cpu/main.cpp.o: CMakeFiles/raytracer.dir/flags.make
CMakeFiles/raytracer.dir/src/cpu/main.cpp.o: /home/christianw/raytracer-cuda/src/cpu/main.cpp
CMakeFiles/raytracer.dir/src/cpu/main.cpp.o: CMakeFiles/raytracer.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir=/home/christianw/raytracer-cuda/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/raytracer.dir/src/cpu/main.cpp.o"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/raytracer.dir/src/cpu/main.cpp.o -MF CMakeFiles/raytracer.dir/src/cpu/main.cpp.o.d -o CMakeFiles/raytracer.dir/src/cpu/main.cpp.o -c /home/christianw/raytracer-cuda/src/cpu/main.cpp

CMakeFiles/raytracer.dir/src/cpu/main.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing CXX source to CMakeFiles/raytracer.dir/src/cpu/main.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/christianw/raytracer-cuda/src/cpu/main.cpp > CMakeFiles/raytracer.dir/src/cpu/main.cpp.i

CMakeFiles/raytracer.dir/src/cpu/main.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling CXX source to assembly CMakeFiles/raytracer.dir/src/cpu/main.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/christianw/raytracer-cuda/src/cpu/main.cpp -o CMakeFiles/raytracer.dir/src/cpu/main.cpp.s

CMakeFiles/raytracer.dir/src/gpu/render.cu.o: CMakeFiles/raytracer.dir/flags.make
CMakeFiles/raytracer.dir/src/gpu/render.cu.o: CMakeFiles/raytracer.dir/includes_CUDA.rsp
CMakeFiles/raytracer.dir/src/gpu/render.cu.o: /home/christianw/raytracer-cuda/src/gpu/render.cu
CMakeFiles/raytracer.dir/src/gpu/render.cu.o: CMakeFiles/raytracer.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir=/home/christianw/raytracer-cuda/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building CUDA object CMakeFiles/raytracer.dir/src/gpu/render.cu.o"
	/usr/bin/nvcc -forward-unknown-to-host-compiler $(CUDA_DEFINES) $(CUDA_INCLUDES) $(CUDA_FLAGS) -MD -MT CMakeFiles/raytracer.dir/src/gpu/render.cu.o -MF CMakeFiles/raytracer.dir/src/gpu/render.cu.o.d -x cu -rdc=true -c /home/christianw/raytracer-cuda/src/gpu/render.cu -o CMakeFiles/raytracer.dir/src/gpu/render.cu.o

CMakeFiles/raytracer.dir/src/gpu/render.cu.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing CUDA source to CMakeFiles/raytracer.dir/src/gpu/render.cu.i"
	$(CMAKE_COMMAND) -E cmake_unimplemented_variable CMAKE_CUDA_CREATE_PREPROCESSED_SOURCE

CMakeFiles/raytracer.dir/src/gpu/render.cu.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling CUDA source to assembly CMakeFiles/raytracer.dir/src/gpu/render.cu.s"
	$(CMAKE_COMMAND) -E cmake_unimplemented_variable CMAKE_CUDA_CREATE_ASSEMBLY_SOURCE

# Object files for target raytracer
raytracer_OBJECTS = \
"CMakeFiles/raytracer.dir/src/cpu/main.cpp.o" \
"CMakeFiles/raytracer.dir/src/gpu/render.cu.o"

# External object files for target raytracer
raytracer_EXTERNAL_OBJECTS =

CMakeFiles/raytracer.dir/cmake_device_link.o: CMakeFiles/raytracer.dir/src/cpu/main.cpp.o
CMakeFiles/raytracer.dir/cmake_device_link.o: CMakeFiles/raytracer.dir/src/gpu/render.cu.o
CMakeFiles/raytracer.dir/cmake_device_link.o: CMakeFiles/raytracer.dir/build.make
CMakeFiles/raytracer.dir/cmake_device_link.o: CMakeFiles/raytracer.dir/deviceLinkLibs.rsp
CMakeFiles/raytracer.dir/cmake_device_link.o: CMakeFiles/raytracer.dir/deviceObjects1.rsp
CMakeFiles/raytracer.dir/cmake_device_link.o: CMakeFiles/raytracer.dir/dlink.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --bold --progress-dir=/home/christianw/raytracer-cuda/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Linking CUDA device code CMakeFiles/raytracer.dir/cmake_device_link.o"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/raytracer.dir/dlink.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/raytracer.dir/build: CMakeFiles/raytracer.dir/cmake_device_link.o
.PHONY : CMakeFiles/raytracer.dir/build

# Object files for target raytracer
raytracer_OBJECTS = \
"CMakeFiles/raytracer.dir/src/cpu/main.cpp.o" \
"CMakeFiles/raytracer.dir/src/gpu/render.cu.o"

# External object files for target raytracer
raytracer_EXTERNAL_OBJECTS =

raytracer: CMakeFiles/raytracer.dir/src/cpu/main.cpp.o
raytracer: CMakeFiles/raytracer.dir/src/gpu/render.cu.o
raytracer: CMakeFiles/raytracer.dir/build.make
raytracer: CMakeFiles/raytracer.dir/cmake_device_link.o
raytracer: CMakeFiles/raytracer.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --bold --progress-dir=/home/christianw/raytracer-cuda/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_4) "Linking CXX executable raytracer"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/raytracer.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/raytracer.dir/build: raytracer
.PHONY : CMakeFiles/raytracer.dir/build

CMakeFiles/raytracer.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/raytracer.dir/cmake_clean.cmake
.PHONY : CMakeFiles/raytracer.dir/clean

CMakeFiles/raytracer.dir/depend:
	cd /home/christianw/raytracer-cuda/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/christianw/raytracer-cuda /home/christianw/raytracer-cuda /home/christianw/raytracer-cuda/build /home/christianw/raytracer-cuda/build /home/christianw/raytracer-cuda/build/CMakeFiles/raytracer.dir/DependInfo.cmake "--color=$(COLOR)"
.PHONY : CMakeFiles/raytracer.dir/depend

