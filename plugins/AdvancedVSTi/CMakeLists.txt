cmake_minimum_required(VERSION 3.22)

project(AdvancedVSTi VERSION 0.2.0)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT DEFINED JUCE_DIR)
  message(FATAL_ERROR "Set -DJUCE_DIR=/path/to/JUCE")
endif()

add_subdirectory(${JUCE_DIR} JUCE)

function(add_ai_music_vst3 target_name product_name plugin_code flavor_code bundle_id)
  juce_add_plugin(${target_name}
      COMPANY_NAME "AI Music Studio"
      BUNDLE_ID ${bundle_id}
      IS_SYNTH TRUE
      NEEDS_MIDI_INPUT TRUE
      NEEDS_MIDI_OUTPUT FALSE
      IS_MIDI_EFFECT FALSE
      EDITOR_WANTS_KEYBOARD_FOCUS FALSE
      COPY_PLUGIN_AFTER_BUILD FALSE
      PLUGIN_MANUFACTURER_CODE AIms
      PLUGIN_CODE ${plugin_code}
      FORMATS VST3 Standalone
      PRODUCT_NAME "${product_name}")

  juce_generate_juce_header(${target_name})

  target_sources(${target_name}
      PRIVATE
        Source/PluginProcessor.cpp
        Source/PluginProcessor.h
        Source/PluginEditor.cpp
        Source/PluginEditor.h)

  target_compile_definitions(${target_name}
    PRIVATE
      AIMS_INSTRUMENT_FLAVOR=${flavor_code}
    PUBLIC
      JUCE_WEB_BROWSER=0
      JUCE_USE_CURL=0
      JUCE_VST3_CAN_REPLACE_VST2=0)

  target_link_libraries(${target_name}
      PRIVATE
        juce::juce_audio_utils
        juce::juce_dsp
      PUBLIC
        juce::juce_recommended_config_flags
        juce::juce_recommended_lto_flags
        juce::juce_recommended_warning_flags)
endfunction()

add_ai_music_vst3(AdvancedVSTi "AdvancedVSTi" AdvS 0 com.aimusicstudio.advancedvsti)
add_ai_music_vst3(AIDrumMachine "AI Drum Machine" ADrm 1 com.aimusicstudio.aidrummachine)
add_ai_music_vst3(AIBassSynth "AI Bass Synth" ABas 2 com.aimusicstudio.aibasssynth)
add_ai_music_vst3(AIStringSynth "AI String Synth" AStr 3 com.aimusicstudio.aistringsynth)
