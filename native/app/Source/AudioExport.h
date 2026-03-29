#pragma once

#include "NativeVstHostSession.h"
#include "ProjectModel.h"

#include <juce_core/juce_core.h>

namespace aims
{

struct AudioExportSummary
{
    int sampleRate = 44100;
    int frameCount = 0;
    int renderedInstrumentTrackCount = 0;
    int mixedAudioTrackCount = 0;
    juce::StringArray warnings;
};

struct AudioExportStemFile
{
    int trackIndex = -1;
    juce::String trackName;
    juce::String filePath;
};

struct AudioExportBatchSummary
{
    int sampleRate = 44100;
    int exportedTrackCount = 0;
    juce::StringArray stemPaths;
    juce::StringArray warnings;
    std::vector<AudioExportStemFile> exportedFiles;
};

juce::Result exportProjectRangeToWavFile(const juce::File& file,
                                         const ProjectState& project,
                                         NativeVstHostSession& nativeVstHost,
                                         AudioExportSummary& outSummary,
                                         int sampleRate = 44100);
juce::Result exportTrackRangeToWavFile(const juce::File& file,
                                       const ProjectState& project,
                                       int trackIndex,
                                       NativeVstHostSession& nativeVstHost,
                                       AudioExportSummary& outSummary,
                                       int sampleRate = 44100);
juce::Result exportProjectStemsToFolder(const juce::File& folder,
                                        const ProjectState& project,
                                        NativeVstHostSession& nativeVstHost,
                                        AudioExportBatchSummary& outSummary,
                                        int sampleRate = 44100);

} // namespace aims
