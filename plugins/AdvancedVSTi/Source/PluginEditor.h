#pragma once

#include <JuceHeader.h>
#include "PluginProcessor.h"

class AdvancedVSTiAudioProcessorEditor final : public juce::AudioProcessorEditor
{
public:
    explicit AdvancedVSTiAudioProcessorEditor (AdvancedVSTiAudioProcessor&);
    ~AdvancedVSTiAudioProcessorEditor() override = default;

    void paint (juce::Graphics&) override;
    void resized() override;

private:
    AdvancedVSTiAudioProcessor& audioProcessor;
    juce::GenericAudioProcessorEditor genericEditor;

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR (AdvancedVSTiAudioProcessorEditor)
};
