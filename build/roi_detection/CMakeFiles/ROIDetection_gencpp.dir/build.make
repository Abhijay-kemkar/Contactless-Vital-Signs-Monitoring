# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.16

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
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/abhijay/Desktop/MIT-BWH/src

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/abhijay/Desktop/MIT-BWH/build

# Utility rule file for ROIDetection_gencpp.

# Include the progress variables for this target.
include roi_detection/CMakeFiles/ROIDetection_gencpp.dir/progress.make

ROIDetection_gencpp: roi_detection/CMakeFiles/ROIDetection_gencpp.dir/build.make

.PHONY : ROIDetection_gencpp

# Rule to build all files generated by this target.
roi_detection/CMakeFiles/ROIDetection_gencpp.dir/build: ROIDetection_gencpp

.PHONY : roi_detection/CMakeFiles/ROIDetection_gencpp.dir/build

roi_detection/CMakeFiles/ROIDetection_gencpp.dir/clean:
	cd /home/abhijay/Desktop/MIT-BWH/build/roi_detection && $(CMAKE_COMMAND) -P CMakeFiles/ROIDetection_gencpp.dir/cmake_clean.cmake
.PHONY : roi_detection/CMakeFiles/ROIDetection_gencpp.dir/clean

roi_detection/CMakeFiles/ROIDetection_gencpp.dir/depend:
	cd /home/abhijay/Desktop/MIT-BWH/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/abhijay/Desktop/MIT-BWH/src /home/abhijay/Desktop/MIT-BWH/src/roi_detection /home/abhijay/Desktop/MIT-BWH/build /home/abhijay/Desktop/MIT-BWH/build/roi_detection /home/abhijay/Desktop/MIT-BWH/build/roi_detection/CMakeFiles/ROIDetection_gencpp.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : roi_detection/CMakeFiles/ROIDetection_gencpp.dir/depend

