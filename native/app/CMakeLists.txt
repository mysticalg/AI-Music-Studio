cmake_minimum_required(VERSION 3.22)

project(AIMusicStudioNative VERSION 0.3.3)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT DEFINED JUCE_DIR)
  message(FATAL_ERROR "Set -DJUCE_DIR=/path/to/JUCE")
endif()

add_subdirectory(${JUCE_DIR} JUCE)

juce_add_binary_data(MutagenAssets
    SOURCES
        Assets/mutagen-logo-source.png
        Assets/mutagen-logo-80.png
        Assets/demucs_separate.py
        Assets/audio_to_midi.py
)

set(AIMS_HOST_COMPILE_DEFINITIONS
    JUCE_DISABLE_CAUTIOUS_PARAMETER_ID_CHECKING=1
    JUCE_PLUGINHOST_VST=0
    JUCE_PLUGINHOST_VST3=1
    JUCE_USE_CURL=0
    JUCE_USE_MP3AUDIOFORMAT=1
    JUCE_VST3_HOST_CROSS_PLATFORM_UID=1
    JUCE_WEB_BROWSER=0
)

if(WIN32)
  list(APPEND AIMS_HOST_COMPILE_DEFINITIONS
      JUCE_DIRECTSOUND=1
      JUCE_PLUGINHOST_AU=0
      JUCE_PLUGINHOST_LADSPA=0
      JUCE_PLUGINHOST_LV2=0
      JUCE_ASIO=1
      JUCE_ASIO_USE_EXTERNAL_SDK=0
      JUCE_WASAPI=1
      JUCE_WASAPI_EXCLUSIVE=1
  )
elseif(APPLE)
  list(APPEND AIMS_HOST_COMPILE_DEFINITIONS
      JUCE_PLUGINHOST_AU=1
      JUCE_PLUGINHOST_LADSPA=0
      JUCE_PLUGINHOST_LV2=0
  )
elseif(UNIX)
  list(APPEND AIMS_HOST_COMPILE_DEFINITIONS
      JUCE_PLUGINHOST_AU=0
      JUCE_PLUGINHOST_LADSPA=0
      JUCE_PLUGINHOST_LV2=0
      JUCE_ALSA=1
      JUCE_JACK=1
  )
endif()

add_library(AIMusicStudioNativeHostLib STATIC
    ../vst3host/Source/Main.cpp
)

target_compile_definitions(AIMusicStudioNativeHostLib
    PRIVATE
        AIMS_VST_HOST_BUILD_LIBRARY=1
        ${AIMS_HOST_COMPILE_DEFINITIONS}
)

target_link_libraries(AIMusicStudioNativeHostLib
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

juce_add_gui_app(AIMusicStudioNative
    PRODUCT_NAME "Mutagen"
    COMPANY_NAME "Mutagen"
    BUNDLE_ID "com.mutagen.native"
    ICON_BIG "${CMAKE_CURRENT_SOURCE_DIR}/Assets/mutagen-logo-source.png"
    ICON_SMALL "${CMAKE_CURRENT_SOURCE_DIR}/Assets/mutagen-logo-80.png"
)

target_sources(AIMusicStudioNative
    PRIVATE
        Source/Main.cpp
        Source/MainWindow.cpp
        Source/MainWindow.h
        Source/AudioExport.cpp
        Source/AudioExport.h
        Source/AIClient.cpp
        Source/AIClient.h
        Source/AceStepClient.cpp
        Source/AceStepClient.h
        Source/AutomationEditorComponent.cpp
        Source/AutomationEditorComponent.h
        Source/ArrangementOverviewComponent.cpp
        Source/ArrangementOverviewComponent.h
        Source/MidiFileIO.cpp
        Source/MidiFileIO.h
        Source/MixerComponent.cpp
        Source/MixerComponent.h
        Source/NativeVstHostSession.cpp
        Source/NativeVstHostSession.h
        Source/PianoRollComponent.cpp
        Source/PianoRollComponent.h
        Source/ProjectModel.cpp
        Source/ProjectModel.h
        Source/SampleFileIO.cpp
        Source/SampleFileIO.h
        Source/SampleTimelineComponent.cpp
        Source/SampleTimelineComponent.h
)

target_compile_definitions(AIMusicStudioNative
    PRIVATE
        JUCE_USE_CURL=0
        JUCE_WEB_BROWSER=0
        ${AIMS_HOST_COMPILE_DEFINITIONS}
)

target_link_libraries(AIMusicStudioNative
    PRIVATE
        AIMusicStudioNativeHostLib
        MutagenAssets
        juce::juce_audio_basics
        juce::juce_audio_formats
        juce::juce_data_structures
        juce::juce_gui_extra
        juce::juce_recommended_config_flags
        juce::juce_recommended_lto_flags
        juce::juce_recommended_warning_flags
)
