cmake_minimum_required(VERSION 3.21)
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

project(FlowVk
  VERSION 0.1.0
  LANGUAGES CXX
)

option(FLOWVK_INSTALL "Enable install + package export" ON)

# ----------------------------
# Library target
# ----------------------------
add_library(FlowVk STATIC
	src/Instance.cpp
	src/Buffer.cpp
)
add_library(FlowVk::FlowVk ALIAS FlowVk)

target_compile_features(FlowVk PUBLIC cxx_std_23)
target_include_directories(FlowVk
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

find_package(Vulkan REQUIRED)
target_link_libraries(FlowVk PUBLIC Vulkan::Vulkan)

# If VMA header is in your repo (e.g. external/vma/vk_mem_alloc.h):
target_include_directories(FlowVk SYSTEM PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/external/)

# ----------------------------
# Shader pipeline CMake module
# ----------------------------
include(cmake/FlowVkShaderPipeline.cmake)

_flowvk_ensure_shaderpp_tool()

add_custom_target(FlowVk_Tools ALL DEPENDS FlowVk_ShaderPP)


# ----------------------------
# Install + export package 
# ----------------------------
if(FLOWVK_INSTALL)
  include(GNUInstallDirs)
  include(CMakePackageConfigHelpers)

  if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/include")
    install(DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/include/"
      DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}"
    )
  endif()

  install(TARGETS FlowVk
    EXPORT FlowVkTargets
  )

  # Install the shader pipeline module so consumers get the functions
  install(FILES
    "${CMAKE_CURRENT_SOURCE_DIR}/cmake/FlowVkShaderPipeline.cmake"
    DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/FlowVk"
  )

  install(EXPORT FlowVkTargets
    NAMESPACE FlowVk::
    DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/FlowVk"
  )

  configure_package_config_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/cmake/FlowVkConfig.cmake.in"
    "${CMAKE_CURRENT_BINARY_DIR}/FlowVkConfig.cmake"
    INSTALL_DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/FlowVk"
  )

  write_basic_package_version_file(
    "${CMAKE_CURRENT_BINARY_DIR}/FlowVkConfigVersion.cmake"
    VERSION "${PROJECT_VERSION}"
    COMPATIBILITY SameMajorVersion
  )

  install(FILES
    "${CMAKE_CURRENT_BINARY_DIR}/FlowVkConfig.cmake"
    "${CMAKE_CURRENT_BINARY_DIR}/FlowVkConfigVersion.cmake"
    DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/FlowVk"
  )
endif()

