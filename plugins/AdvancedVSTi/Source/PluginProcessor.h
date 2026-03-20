#pragma once

#include <JuceHeader.h>

#ifndef AIMS_INSTRUMENT_FLAVOR
#define AIMS_INSTRUMENT_FLAVOR 0
#endif

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
    enum class InstrumentFlavor
    {
        advanced = 0,
        drumMachine,
        bassSynth,
        stringSynth,
        leadSynth,
        padSynth,
        pluckSynth,
        sampler,
        drum808,
        acid303
    };

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
        float auxPhase = 0.0f;
        float toneState = 0.0f;
        float colourState = 0.0f;

        juce::ADSR ampEnv;
        juce::ADSR filterEnv;
    };

    struct RenderParameters
    {
        OscType oscType = OscType::sine;
        int unisonVoices = 1;
        int lfo1Shape = 0;
        int lfo2Shape = 0;
        int arpMode = 0;
        int filterType = 0;

        float detune = 0.0f;
        float fmAmount = 0.0f;
        float syncAmount = 0.0f;
        float gateLength = 8.0f;
        float envCurve = 0.0f;
        float cutoff = 1200.0f;
        float resonance = 0.4f;
        float filterEnvAmount = 0.5f;
        float lfo1Rate = 2.0f;
        float lfo1Pitch = 0.0f;
        float lfo2Rate = 3.0f;
        float lfo2Filter = 0.0f;
        float arpRate = 4.0f;
        float rhythmGateRate = 8.0f;
        float rhythmGateDepth = 0.0f;
        int sampleBank = 0;
        float sampleStart = 0.0f;
        float sampleEnd = 1.0f;

        juce::ADSR::Parameters ampEnv;
        juce::ADSR::Parameters filterEnv;
    };

    static constexpr int maxVoices = 16;
    std::array<VoiceState, maxVoices> voices;
    RenderParameters renderParams;

    double currentSampleRate = 44100.0;
    juce::Random random;
    juce::AudioBuffer<float> loadedSample;
    int loadedSampleBank = -1;

    juce::dsp::StateVariableTPTFilter<float> leftFilter;
    juce::dsp::StateVariableTPTFilter<float> rightFilter;

    float lfo1Phase = 0.0f;
    float lfo2Phase = 0.0f;
    int arpStep = 0;
    juce::Array<int> heldNotes;

    [[nodiscard]] static constexpr InstrumentFlavor buildFlavor() noexcept
    {
#if AIMS_INSTRUMENT_FLAVOR == 1
        return InstrumentFlavor::drumMachine;
#elif AIMS_INSTRUMENT_FLAVOR == 2
        return InstrumentFlavor::bassSynth;
#elif AIMS_INSTRUMENT_FLAVOR == 3
        return InstrumentFlavor::stringSynth;
#elif AIMS_INSTRUMENT_FLAVOR == 4
        return InstrumentFlavor::leadSynth;
#elif AIMS_INSTRUMENT_FLAVOR == 5
        return InstrumentFlavor::padSynth;
#elif AIMS_INSTRUMENT_FLAVOR == 6
        return InstrumentFlavor::pluckSynth;
#elif AIMS_INSTRUMENT_FLAVOR == 7
        return InstrumentFlavor::sampler;
#elif AIMS_INSTRUMENT_FLAVOR == 8
        return InstrumentFlavor::drum808;
#elif AIMS_INSTRUMENT_FLAVOR == 9
        return InstrumentFlavor::acid303;
#else
        return InstrumentFlavor::advanced;
#endif
    }

    [[nodiscard]] static constexpr bool isDrumFlavor() noexcept
    {
        return buildFlavor() == InstrumentFlavor::drumMachine || buildFlavor() == InstrumentFlavor::drum808;
    }

    void handleMidi (const juce::MidiBuffer& midiMessages, int numSamples);
    float renderVoiceSample (VoiceState& voice);
    float renderDrumVoiceSample (VoiceState& voice);
    float oscSample (VoiceState& voice, float baseFreq, OscType type, float syncAmount);
    float fmOperator (VoiceState& voice, float baseFreq, float amount);
    float lfoValue (int shape, float phase) const;
    void refreshSampleBank();
    void buildGeneratedSampleBank (int bankIndex);
    [[nodiscard]] static juce::StringArray sampleBankChoices();

    void updateRenderParameters();
    void applyEnvelopeSettings();
    void advanceArpIfNeeded();
    int getArpNote() const;

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR (AdvancedVSTiAudioProcessor)
};
