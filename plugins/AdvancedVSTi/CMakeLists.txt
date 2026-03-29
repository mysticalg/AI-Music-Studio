cmake_minimum_required(VERSION 3.22)

project(AdvancedVSTi VERSION 0.2.0)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT DEFINED JUCE_DIR)
  message(FATAL_ERROR "Set -DJUCE_DIR=/path/to/JUCE")
endif()

add_subdirectory(${JUCE_DIR} JUCE)

set(AIMS_REPO_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../..")
get_filename_component(AIMS_REPO_ROOT "${AIMS_REPO_ROOT}" ABSOLUTE)

set(AIMS_VIRUS_IMAGE "${AIMS_REPO_ROOT}/virus.jpg")
if(NOT EXISTS "${AIMS_VIRUS_IMAGE}")
  message(FATAL_ERROR "Required Virus faceplate image not found: ${AIMS_VIRUS_IMAGE}")
endif()

set(AIMS_ASSET_SOURCES "${AIMS_VIRUS_IMAGE}")
set(AIMS_HAS_TR909_IMAGE 0)
set(AIMS_VEC1_LIBRARY_DIR "D:/OneDrive/Music/Sound Design/Sample Library/sample library/VEC1")
set(AIMS_PIANO_LIBRARY_DIR "${AIMS_REPO_ROOT}/.cache/SplendidGrandPiano")
set(AIMS_TR909_IMAGE "${AIMS_REPO_ROOT}/tr9092.jpg")
if(EXISTS "${AIMS_TR909_IMAGE}")
  list(APPEND AIMS_ASSET_SOURCES "${AIMS_TR909_IMAGE}")
  set(AIMS_HAS_TR909_IMAGE 1)
endif()
juce_add_binary_data(AIMusicStudioVirusAssets SOURCES ${AIMS_ASSET_SOURCES})

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
      AIMS_HAS_TR909_IMAGE=${AIMS_HAS_TR909_IMAGE}
      AIMS_REPO_ROOT_PATH="${AIMS_REPO_ROOT}"
    PUBLIC
      JUCE_WEB_BROWSER=0
      JUCE_USE_CURL=0
      JUCE_VST3_CAN_REPLACE_VST2=0)

  target_link_libraries(${target_name}
      PRIVATE
        AIMusicStudioVirusAssets
        juce::juce_audio_utils
        juce::juce_dsp
      PUBLIC
        juce::juce_recommended_config_flags
        juce::juce_recommended_lto_flags
        juce::juce_recommended_warning_flags)
endfunction()

add_ai_music_vst3(VirusSynth "Virus Synth" AdvS 0 com.aimusicstudio.virussynth)
add_ai_music_vst3(AIDrumMachine "AI Drum Machine" ADrm 1 com.aimusicstudio.aidrummachine)
add_ai_music_vst3(AIBassSynth "AI Bass Synth" ABas 2 com.aimusicstudio.aibasssynth)
add_ai_music_vst3(AIStringSynth "AI String Synth" AStr 3 com.aimusicstudio.aistringsynth)
add_ai_music_vst3(AILeadSynth "AI Lead Synth" ALed 4 com.aimusicstudio.aileadsynth)
add_ai_music_vst3(AIPadSynth "AI Pad Synth" APad 5 com.aimusicstudio.aipadsynth)
add_ai_music_vst3(AIPluckSynth "AI Pluck Synth" APlk 6 com.aimusicstudio.aiplucksynth)
add_ai_music_vst3(AISampler "AI Sampler" ASmp 7 com.aimusicstudio.aisampler)
add_ai_music_vst3(AI808Machine "AI 808 Machine" A808 8 com.aimusicstudio.ai808machine)
add_ai_music_vst3(AITB303 "AI TB303" ATB3 9 com.aimusicstudio.aitb303)
add_ai_music_vst3(AIVEC1DrumPads "AI VEC1 Drum Pads" AVPD 10 com.aimusicstudio.aivec1drumpads)
add_ai_music_vst3(AIPiano "AI Piano" AIPn 11 com.aimusicstudio.aipiano)
add_ai_music_vst3(AIStrings "AI Strings" AISe 12 com.aimusicstudio.aistrings)
add_ai_music_vst3(AIViolin "AI Violin" AIVl 13 com.aimusicstudio.aiviolin)
add_ai_music_vst3(AIFlute "AI Flute" AIFt 14 com.aimusicstudio.aiflute)
add_ai_music_vst3(AISaxophone "AI Saxophone" AISx 15 com.aimusicstudio.aisaxophone)
add_ai_music_vst3(AIBassGuitar "AI Bass Guitar" AIBG 16 com.aimusicstudio.aibassguitar)
add_ai_music_vst3(AIOrgan "AI Organ" AIOr 17 com.aimusicstudio.aiorgan)

if(EXISTS "${AIMS_VEC1_LIBRARY_DIR}")
  add_custom_command(TARGET AIVEC1DrumPads_VST3 POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E echo "Bundling VEC1 sample library into AI VEC1 Drum Pads.vst3"
    COMMAND ${CMAKE_COMMAND} -E rm -rf "$<TARGET_FILE_DIR:AIVEC1DrumPads_VST3>/../../Resources"
    COMMAND ${CMAKE_COMMAND} -E rm -rf "$<TARGET_FILE_DIR:AIVEC1DrumPads_VST3>/../Resources/VEC1"
    COMMAND ${CMAKE_COMMAND} -E copy_directory
            "${AIMS_VEC1_LIBRARY_DIR}"
            "$<TARGET_FILE_DIR:AIVEC1DrumPads_VST3>/../Resources/VEC1"
    VERBATIM)
endif()

if(EXISTS "${AIMS_PIANO_LIBRARY_DIR}")
  add_custom_command(TARGET AIPiano_VST3 POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E echo "Bundling Splendid Grand Piano into AI Piano.vst3"
    COMMAND ${CMAKE_COMMAND} -E rm -rf "$<TARGET_FILE_DIR:AIPiano_VST3>/../Resources/Piano"
    COMMAND ${CMAKE_COMMAND} -E copy_directory
            "${AIMS_PIANO_LIBRARY_DIR}"
            "$<TARGET_FILE_DIR:AIPiano_VST3>/../Resources/Piano"
    VERBATIM)
endif()
