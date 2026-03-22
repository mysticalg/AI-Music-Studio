cmake_minimum_required(VERSION 3.22)

project(AIMusicStudioVSTHost VERSION 0.1.0)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT DEFINED JUCE_DIR)
  message(FATAL_ERROR "Set -DJUCE_DIR=/path/to/JUCE")
endif()

add_subdirectory(${JUCE_DIR} JUCE)

juce_add_gui_app(AIMusicStudioVSTHost
    PRODUCT_NAME "AI Music Studio VST Host"
    COMPANY_NAME "AI Music Studio"
    BUNDLE_ID "com.aimusicstudio.vsthost"
)

juce_generate_juce_header(AIMusicStudioVSTHost)

target_sources(AIMusicStudioVSTHost
    PRIVATE
        Source/Main.cpp
)

target_compile_definitions(AIMusicStudioVSTHost
    PRIVATE
        JUCE_DIRECTSOUND=1
        JUCE_DISABLE_CAUTIOUS_PARAMETER_ID_CHECKING=1
        JUCE_PLUGINHOST_AU=0
        JUCE_PLUGINHOST_LADSPA=0
        JUCE_PLUGINHOST_LV2=0
        JUCE_PLUGINHOST_VST=0
        JUCE_PLUGINHOST_VST3=1
        JUCE_USE_CURL=0
        JUCE_VST3_HOST_CROSS_PLATFORM_UID=1
        JUCE_WASAPI=1
        JUCE_WEB_BROWSER=0
)

target_link_libraries(AIMusicStudioVSTHost
    PRIVATE
        juce::juce_audio_devices
        juce::juce_audio_processors
        juce::juce_audio_utils
        juce::juce_gui_extra
        juce::juce_recommended_config_flags
        juce::juce_recommended_lto_flags
        juce::juce_recommended_warning_flags
)
