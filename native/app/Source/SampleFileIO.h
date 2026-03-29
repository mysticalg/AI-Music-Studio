#pragma once

#include "ProjectModel.h"

#include <juce_audio_basics/juce_audio_basics.h>
#include <juce_core/juce_core.h>

namespace aims
{

juce::Result loadSampleAssetFromFile(const juce::File& file, SampleAsset& outAsset);
juce::Result loadAudioFileToStereoBuffer(const juce::File& file,
                                         juce::AudioBuffer<float>& outBuffer,
                                         double targetSampleRate = 0.0);

} // namespace aims
