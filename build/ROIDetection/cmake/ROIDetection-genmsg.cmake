# generated from genmsg/cmake/pkg-genmsg.cmake.em

message(STATUS "ROIDetection: 1 messages, 0 services")

set(MSG_I_FLAGS "-IROIDetection:/home/abhijay/Desktop/MIT-BWH/src/ROIDetection/msg;-Istd_msgs:/opt/ros/noetic/share/std_msgs/cmake/../msg")

# Find all generators
find_package(gencpp REQUIRED)
find_package(geneus REQUIRED)
find_package(genlisp REQUIRED)
find_package(gennodejs REQUIRED)
find_package(genpy REQUIRED)

add_custom_target(ROIDetection_generate_messages ALL)

# verify that message/service dependencies have not changed since configure



get_filename_component(_filename "/home/abhijay/Desktop/MIT-BWH/src/ROIDetection/msg/RGB.msg" NAME_WE)
add_custom_target(_ROIDetection_generate_messages_check_deps_${_filename}
  COMMAND ${CATKIN_ENV} ${PYTHON_EXECUTABLE} ${GENMSG_CHECK_DEPS_SCRIPT} "ROIDetection" "/home/abhijay/Desktop/MIT-BWH/src/ROIDetection/msg/RGB.msg" ""
)

#
#  langs = gencpp;geneus;genlisp;gennodejs;genpy
#

### Section generating for lang: gencpp
### Generating Messages
_generate_msg_cpp(ROIDetection
  "/home/abhijay/Desktop/MIT-BWH/src/ROIDetection/msg/RGB.msg"
  "${MSG_I_FLAGS}"
  ""
  ${CATKIN_DEVEL_PREFIX}/${gencpp_INSTALL_DIR}/ROIDetection
)

### Generating Services

### Generating Module File
_generate_module_cpp(ROIDetection
  ${CATKIN_DEVEL_PREFIX}/${gencpp_INSTALL_DIR}/ROIDetection
  "${ALL_GEN_OUTPUT_FILES_cpp}"
)

add_custom_target(ROIDetection_generate_messages_cpp
  DEPENDS ${ALL_GEN_OUTPUT_FILES_cpp}
)
add_dependencies(ROIDetection_generate_messages ROIDetection_generate_messages_cpp)

# add dependencies to all check dependencies targets
get_filename_component(_filename "/home/abhijay/Desktop/MIT-BWH/src/ROIDetection/msg/RGB.msg" NAME_WE)
add_dependencies(ROIDetection_generate_messages_cpp _ROIDetection_generate_messages_check_deps_${_filename})

# target for backward compatibility
add_custom_target(ROIDetection_gencpp)
add_dependencies(ROIDetection_gencpp ROIDetection_generate_messages_cpp)

# register target for catkin_package(EXPORTED_TARGETS)
list(APPEND ${PROJECT_NAME}_EXPORTED_TARGETS ROIDetection_generate_messages_cpp)

### Section generating for lang: geneus
### Generating Messages
_generate_msg_eus(ROIDetection
  "/home/abhijay/Desktop/MIT-BWH/src/ROIDetection/msg/RGB.msg"
  "${MSG_I_FLAGS}"
  ""
  ${CATKIN_DEVEL_PREFIX}/${geneus_INSTALL_DIR}/ROIDetection
)

### Generating Services

### Generating Module File
_generate_module_eus(ROIDetection
  ${CATKIN_DEVEL_PREFIX}/${geneus_INSTALL_DIR}/ROIDetection
  "${ALL_GEN_OUTPUT_FILES_eus}"
)

add_custom_target(ROIDetection_generate_messages_eus
  DEPENDS ${ALL_GEN_OUTPUT_FILES_eus}
)
add_dependencies(ROIDetection_generate_messages ROIDetection_generate_messages_eus)

# add dependencies to all check dependencies targets
get_filename_component(_filename "/home/abhijay/Desktop/MIT-BWH/src/ROIDetection/msg/RGB.msg" NAME_WE)
add_dependencies(ROIDetection_generate_messages_eus _ROIDetection_generate_messages_check_deps_${_filename})

# target for backward compatibility
add_custom_target(ROIDetection_geneus)
add_dependencies(ROIDetection_geneus ROIDetection_generate_messages_eus)

# register target for catkin_package(EXPORTED_TARGETS)
list(APPEND ${PROJECT_NAME}_EXPORTED_TARGETS ROIDetection_generate_messages_eus)

### Section generating for lang: genlisp
### Generating Messages
_generate_msg_lisp(ROIDetection
  "/home/abhijay/Desktop/MIT-BWH/src/ROIDetection/msg/RGB.msg"
  "${MSG_I_FLAGS}"
  ""
  ${CATKIN_DEVEL_PREFIX}/${genlisp_INSTALL_DIR}/ROIDetection
)

### Generating Services

### Generating Module File
_generate_module_lisp(ROIDetection
  ${CATKIN_DEVEL_PREFIX}/${genlisp_INSTALL_DIR}/ROIDetection
  "${ALL_GEN_OUTPUT_FILES_lisp}"
)

add_custom_target(ROIDetection_generate_messages_lisp
  DEPENDS ${ALL_GEN_OUTPUT_FILES_lisp}
)
add_dependencies(ROIDetection_generate_messages ROIDetection_generate_messages_lisp)

# add dependencies to all check dependencies targets
get_filename_component(_filename "/home/abhijay/Desktop/MIT-BWH/src/ROIDetection/msg/RGB.msg" NAME_WE)
add_dependencies(ROIDetection_generate_messages_lisp _ROIDetection_generate_messages_check_deps_${_filename})

# target for backward compatibility
add_custom_target(ROIDetection_genlisp)
add_dependencies(ROIDetection_genlisp ROIDetection_generate_messages_lisp)

# register target for catkin_package(EXPORTED_TARGETS)
list(APPEND ${PROJECT_NAME}_EXPORTED_TARGETS ROIDetection_generate_messages_lisp)

### Section generating for lang: gennodejs
### Generating Messages
_generate_msg_nodejs(ROIDetection
  "/home/abhijay/Desktop/MIT-BWH/src/ROIDetection/msg/RGB.msg"
  "${MSG_I_FLAGS}"
  ""
  ${CATKIN_DEVEL_PREFIX}/${gennodejs_INSTALL_DIR}/ROIDetection
)

### Generating Services

### Generating Module File
_generate_module_nodejs(ROIDetection
  ${CATKIN_DEVEL_PREFIX}/${gennodejs_INSTALL_DIR}/ROIDetection
  "${ALL_GEN_OUTPUT_FILES_nodejs}"
)

add_custom_target(ROIDetection_generate_messages_nodejs
  DEPENDS ${ALL_GEN_OUTPUT_FILES_nodejs}
)
add_dependencies(ROIDetection_generate_messages ROIDetection_generate_messages_nodejs)

# add dependencies to all check dependencies targets
get_filename_component(_filename "/home/abhijay/Desktop/MIT-BWH/src/ROIDetection/msg/RGB.msg" NAME_WE)
add_dependencies(ROIDetection_generate_messages_nodejs _ROIDetection_generate_messages_check_deps_${_filename})

# target for backward compatibility
add_custom_target(ROIDetection_gennodejs)
add_dependencies(ROIDetection_gennodejs ROIDetection_generate_messages_nodejs)

# register target for catkin_package(EXPORTED_TARGETS)
list(APPEND ${PROJECT_NAME}_EXPORTED_TARGETS ROIDetection_generate_messages_nodejs)

### Section generating for lang: genpy
### Generating Messages
_generate_msg_py(ROIDetection
  "/home/abhijay/Desktop/MIT-BWH/src/ROIDetection/msg/RGB.msg"
  "${MSG_I_FLAGS}"
  ""
  ${CATKIN_DEVEL_PREFIX}/${genpy_INSTALL_DIR}/ROIDetection
)

### Generating Services

### Generating Module File
_generate_module_py(ROIDetection
  ${CATKIN_DEVEL_PREFIX}/${genpy_INSTALL_DIR}/ROIDetection
  "${ALL_GEN_OUTPUT_FILES_py}"
)

add_custom_target(ROIDetection_generate_messages_py
  DEPENDS ${ALL_GEN_OUTPUT_FILES_py}
)
add_dependencies(ROIDetection_generate_messages ROIDetection_generate_messages_py)

# add dependencies to all check dependencies targets
get_filename_component(_filename "/home/abhijay/Desktop/MIT-BWH/src/ROIDetection/msg/RGB.msg" NAME_WE)
add_dependencies(ROIDetection_generate_messages_py _ROIDetection_generate_messages_check_deps_${_filename})

# target for backward compatibility
add_custom_target(ROIDetection_genpy)
add_dependencies(ROIDetection_genpy ROIDetection_generate_messages_py)

# register target for catkin_package(EXPORTED_TARGETS)
list(APPEND ${PROJECT_NAME}_EXPORTED_TARGETS ROIDetection_generate_messages_py)



if(gencpp_INSTALL_DIR AND EXISTS ${CATKIN_DEVEL_PREFIX}/${gencpp_INSTALL_DIR}/ROIDetection)
  # install generated code
  install(
    DIRECTORY ${CATKIN_DEVEL_PREFIX}/${gencpp_INSTALL_DIR}/ROIDetection
    DESTINATION ${gencpp_INSTALL_DIR}
  )
endif()
if(TARGET std_msgs_generate_messages_cpp)
  add_dependencies(ROIDetection_generate_messages_cpp std_msgs_generate_messages_cpp)
endif()

if(geneus_INSTALL_DIR AND EXISTS ${CATKIN_DEVEL_PREFIX}/${geneus_INSTALL_DIR}/ROIDetection)
  # install generated code
  install(
    DIRECTORY ${CATKIN_DEVEL_PREFIX}/${geneus_INSTALL_DIR}/ROIDetection
    DESTINATION ${geneus_INSTALL_DIR}
  )
endif()
if(TARGET std_msgs_generate_messages_eus)
  add_dependencies(ROIDetection_generate_messages_eus std_msgs_generate_messages_eus)
endif()

if(genlisp_INSTALL_DIR AND EXISTS ${CATKIN_DEVEL_PREFIX}/${genlisp_INSTALL_DIR}/ROIDetection)
  # install generated code
  install(
    DIRECTORY ${CATKIN_DEVEL_PREFIX}/${genlisp_INSTALL_DIR}/ROIDetection
    DESTINATION ${genlisp_INSTALL_DIR}
  )
endif()
if(TARGET std_msgs_generate_messages_lisp)
  add_dependencies(ROIDetection_generate_messages_lisp std_msgs_generate_messages_lisp)
endif()

if(gennodejs_INSTALL_DIR AND EXISTS ${CATKIN_DEVEL_PREFIX}/${gennodejs_INSTALL_DIR}/ROIDetection)
  # install generated code
  install(
    DIRECTORY ${CATKIN_DEVEL_PREFIX}/${gennodejs_INSTALL_DIR}/ROIDetection
    DESTINATION ${gennodejs_INSTALL_DIR}
  )
endif()
if(TARGET std_msgs_generate_messages_nodejs)
  add_dependencies(ROIDetection_generate_messages_nodejs std_msgs_generate_messages_nodejs)
endif()

if(genpy_INSTALL_DIR AND EXISTS ${CATKIN_DEVEL_PREFIX}/${genpy_INSTALL_DIR}/ROIDetection)
  install(CODE "execute_process(COMMAND \"/usr/bin/python3\" -m compileall \"${CATKIN_DEVEL_PREFIX}/${genpy_INSTALL_DIR}/ROIDetection\")")
  # install generated code
  install(
    DIRECTORY ${CATKIN_DEVEL_PREFIX}/${genpy_INSTALL_DIR}/ROIDetection
    DESTINATION ${genpy_INSTALL_DIR}
  )
endif()
if(TARGET std_msgs_generate_messages_py)
  add_dependencies(ROIDetection_generate_messages_py std_msgs_generate_messages_py)
endif()
