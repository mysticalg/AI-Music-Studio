cmake_minimum_required(VERSION 3.22)

project(AdvancedVSTi VERSION 0.1.0)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT DEFINED JUCE_DIR)
  message(FATAL_ERROR "Set -DJUCE_DIR=/path/to/JUCE")
endif()

add_subdirectory(${JUCE_DIR} JUCE)

juce_add_plugin(AdvancedVSTi
    COMPANY_NAME "AI Music Studio"
    IS_SYNTH TRUE
    NEEDS_MIDI_INPUT TRUE
    NEEDS_MIDI_OUTPUT FALSE
    IS_MIDI_EFFECT FALSE
    EDITOR_WANTS_KEYBOARD_FOCUS FALSE
    COPY_PLUGIN_AFTER_BUILD TRUE
    PLUGIN_MANUFACTURER_CODE AIms
    PLUGIN_CODE AdvS
    FORMATS VST3 Standalone
    PRODUCT_NAME "AdvancedVSTi")

target_sources(AdvancedVSTi
    PRIVATE
      Source/PluginProcessor.cpp
      Source/PluginProcessor.h
      Source/PluginEditor.cpp
      Source/PluginEditor.h)

target_compile_definitions(AdvancedVSTi
  PUBLIC
    JUCE_WEB_BROWSER=0
    JUCE_USE_CURL=0
    JUCE_VST3_CAN_REPLACE_VST2=0)

target_link_libraries(AdvancedVSTi
    PRIVATE
      juce::juce_audio_utils
      juce::juce_dsp
    PUBLIC
      juce::juce_recommended_config_flags
      juce::juce_recommended_lto_flags
      juce::juce_recommended_warning_flags)
