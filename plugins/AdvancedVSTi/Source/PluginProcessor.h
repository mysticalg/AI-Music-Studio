#pragma once

#include <JuceHeader.h>

class AdvancedVSTiAudioProcessor final : public juce::AudioProcessor
{
public:
    AdvancedVSTiAudioProcessor();
    ~AdvancedVSTiAudioProcessor() override = default;

    void prepareToPlay (double sampleRate, int samplesPerBlock) override;
    void releaseResources() override;
    void processBlock (juce::AudioBuffer<float>&, juce::MidiBuffer&) override;

    [[nodiscard]] juce::AudioProcessorEditor* createEditor() override;
    [[nodiscard]] bool hasEditor() const override { return true; }

    [[nodiscard]] const juce::String getName() const override { return JucePlugin_Name; }
    [[nodiscard]] bool acceptsMidi() const override { return true; }
    [[nodiscard]] bool producesMidi() const override { return false; }
    [[nodiscard]] bool isMidiEffect() const override { return false; }
    [[nodiscard]] double getTailLengthSeconds() const override { return 0.0; }

    int getNumPrograms() override { return 1; }
    int getCurrentProgram() override { return 0; }
    void setCurrentProgram (int) override {}
    const juce::String getProgramName (int) override { return {}; }
    void changeProgramName (int, const juce::String&) override {}

    void getStateInformation (juce::MemoryBlock& destData) override;
    void setStateInformation (const void* data, int sizeInBytes) override;

    static juce::AudioProcessorValueTreeState::ParameterLayout createParameterLayout();

    juce::AudioProcessorValueTreeState apvts;

private:
    enum class OscType
    {
        sine = 0,
        saw,
        square,
        noise,
        sample
    };

    struct VoiceState
    {
        bool active = false;
        int midiNote = -1;
        float velocity = 0.0f;
        float phase = 0.0f;
        float fmPhase = 0.0f;
        float syncPhase = 0.0f;
        float samplePos = 0.0f;
        float noteAge = 0.0f;

        juce::ADSR ampEnv;
        juce::ADSR filterEnv;
    };

    static constexpr int maxVoices = 16;
    std::array<VoiceState, maxVoices> voices;

    double currentSampleRate = 44100.0;
    juce::Random random;
    juce::AudioBuffer<float> loadedSample;

    juce::dsp::StateVariableTPTFilter<float> leftFilter;
    juce::dsp::StateVariableTPTFilter<float> rightFilter;

    float lfo1Phase = 0.0f;
    float lfo2Phase = 0.0f;
    int arpStep = 0;
    juce::Array<int> heldNotes;

    void handleMidi (const juce::MidiBuffer& midiMessages, int numSamples);
    float renderVoiceSample (VoiceState& voice);
    float oscSample (VoiceState& voice, float baseFreq, OscType type, float syncAmount);
    float fmOperator (VoiceState& voice, float baseFreq, float amount);
    float lfoValue (int index, float phase) const;

    void applyEnvelopeSettings();
    void advanceArpIfNeeded();
    int getArpNote() const;

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR (AdvancedVSTiAudioProcessor)
};
