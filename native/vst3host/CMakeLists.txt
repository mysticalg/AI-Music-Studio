cmake_minimum_required(VERSION 3.22)

project(AIMusicStudioVSTHost VERSION 0.3.4)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

option(AIMS_ENABLE_WASAPI_EXCLUSIVE "Enable JUCE WASAPI exclusive mode backend" ON)
option(AIMS_ENABLE_ASIO "Enable JUCE ASIO backend" ON)

set(AIMS_JUCE_WASAPI_EXCLUSIVE 0)
set(AIMS_JUCE_ASIO 0)

if(WIN32 AND AIMS_ENABLE_WASAPI_EXCLUSIVE)
  set(AIMS_JUCE_WASAPI_EXCLUSIVE 1)
endif()

if(WIN32 AND AIMS_ENABLE_ASIO)
  set(AIMS_JUCE_ASIO 1)
endif()

message(STATUS "AIMS native host WASAPI exclusive backend: ${AIMS_JUCE_WASAPI_EXCLUSIVE}")
message(STATUS "AIMS native host ASIO backend: ${AIMS_JUCE_ASIO}")

if(NOT DEFINED JUCE_DIR)
  message(FATAL_ERROR "Set -DJUCE_DIR=/path/to/JUCE")
endif()

add_subdirectory(${JUCE_DIR} JUCE)

juce_add_gui_app(AIMusicStudioVSTHost
    PRODUCT_NAME "AI Music Studio VST Host"
    COMPANY_NAME "AI Music Studio"
    BUNDLE_ID "com.aimusicstudio.vsthost"
)

target_sources(AIMusicStudioVSTHost
    PRIVATE
        Source/Main.cpp
)

target_compile_definitions(AIMusicStudioVSTHost
    PRIVATE
        AIMS_VST_HOST_BUILD_APP=1
        JUCE_DISABLE_CAUTIOUS_PARAMETER_ID_CHECKING=1
        JUCE_PLUGINHOST_VST=0
        JUCE_PLUGINHOST_VST3=1
        JUCE_USE_CURL=0
        JUCE_VST3_HOST_CROSS_PLATFORM_UID=1
        JUCE_WEB_BROWSER=0
)

if(WIN32)
  target_compile_definitions(AIMusicStudioVSTHost
      PRIVATE
          JUCE_DIRECTSOUND=1
          JUCE_PLUGINHOST_AU=0
          JUCE_PLUGINHOST_LADSPA=0
          JUCE_PLUGINHOST_LV2=0
          JUCE_ASIO=${AIMS_JUCE_ASIO}
          JUCE_ASIO_USE_EXTERNAL_SDK=0
          JUCE_WASAPI=1
          JUCE_WASAPI_EXCLUSIVE=${AIMS_JUCE_WASAPI_EXCLUSIVE}
  )
elseif(APPLE)
  target_compile_definitions(AIMusicStudioVSTHost
      PRIVATE
          JUCE_PLUGINHOST_AU=1
          JUCE_PLUGINHOST_LADSPA=0
          JUCE_PLUGINHOST_LV2=0
  )
elseif(UNIX)
  target_compile_definitions(AIMusicStudioVSTHost
      PRIVATE
          JUCE_PLUGINHOST_AU=0
          JUCE_PLUGINHOST_LADSPA=0
          JUCE_PLUGINHOST_LV2=0
          JUCE_ALSA=1
          JUCE_JACK=1
  )
endif()

target_link_libraries(AIMusicStudioVSTHost
    PRIVATE
        juce::juce_audio_devices
        juce::juce_audio_formats
        juce::juce_audio_processors
        juce::juce_audio_utils
        juce::juce_gui_extra
        juce::juce_recommended_config_flags
        juce::juce_recommended_lto_flags
        juce::juce_recommended_warning_flags
)

add_library(AIMusicStudioVSTHostLib SHARED)

target_sources(AIMusicStudioVSTHostLib
    PRIVATE
        Source/Main.cpp
)

target_compile_definitions(AIMusicStudioVSTHostLib
    PRIVATE
        AIMS_VST_HOST_BUILD_LIBRARY=1
        AIMS_VST_HOST_SHARED=1
        JUCE_DISABLE_CAUTIOUS_PARAMETER_ID_CHECKING=1
        JUCE_PLUGINHOST_VST=0
        JUCE_PLUGINHOST_VST3=1
        JUCE_USE_CURL=0
        JUCE_VST3_HOST_CROSS_PLATFORM_UID=1
        JUCE_WEB_BROWSER=0
)

if(WIN32)
  target_compile_definitions(AIMusicStudioVSTHostLib
      PRIVATE
          JUCE_DIRECTSOUND=1
          JUCE_PLUGINHOST_AU=0
          JUCE_PLUGINHOST_LADSPA=0
          JUCE_PLUGINHOST_LV2=0
          JUCE_ASIO=${AIMS_JUCE_ASIO}
          JUCE_ASIO_USE_EXTERNAL_SDK=0
          JUCE_WASAPI=1
          JUCE_WASAPI_EXCLUSIVE=${AIMS_JUCE_WASAPI_EXCLUSIVE}
  )
elseif(APPLE)
  target_compile_definitions(AIMusicStudioVSTHostLib
      PRIVATE
          JUCE_PLUGINHOST_AU=1
          JUCE_PLUGINHOST_LADSPA=0
          JUCE_PLUGINHOST_LV2=0
  )
elseif(UNIX)
  target_compile_definitions(AIMusicStudioVSTHostLib
      PRIVATE
          JUCE_PLUGINHOST_AU=0
          JUCE_PLUGINHOST_LADSPA=0
          JUCE_PLUGINHOST_LV2=0
          JUCE_ALSA=1
          JUCE_JACK=1
  )
endif()

target_link_libraries(AIMusicStudioVSTHostLib
    PRIVATE
        juce::juce_audio_devices
        juce::juce_audio_formats
        juce::juce_audio_processors
        juce::juce_audio_utils
        juce::juce_gui_extra
        juce::juce_recommended_config_flags
        juce::juce_recommended_lto_flags
        juce::juce_recommended_warning_flags
)

set_target_properties(AIMusicStudioVSTHostLib
    PROPERTIES
        OUTPUT_NAME "AIMusicStudioVSTHostLib"
)
