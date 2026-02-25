#include "PluginEditor.h"

AdvancedVSTiAudioProcessorEditor::AdvancedVSTiAudioProcessorEditor (AdvancedVSTiAudioProcessor& p)
    : AudioProcessorEditor (&p), audioProcessor (p), genericEditor (audioProcessor)
{
    addAndMakeVisible (genericEditor);
    setSize (860, 620);
}

void AdvancedVSTiAudioProcessorEditor::paint (juce::Graphics& g)
{
    g.fillAll (juce::Colour::fromRGB (16, 18, 24));
}

void AdvancedVSTiAudioProcessorEditor::resized()
{
    genericEditor.setBounds (getLocalBounds().reduced (8));
}
