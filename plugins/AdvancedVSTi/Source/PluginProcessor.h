#pragma once

#include <JuceHeader.h>

#ifndef AIMS_INSTRUMENT_FLAVOR
#define AIMS_INSTRUMENT_FLAVOR 0
#endif

class AdvancedVSTiAudioProcessor final : public juce::AudioProcessor,
                                         private juce::AudioProcessorValueTreeState::Listener,
                                         private juce::AsyncUpdater
{
public:
    AdvancedVSTiAudioProcessor();
    ~AdvancedVSTiAudioProcessor() override;

    void prepareToPlay (double sampleRate, int samplesPerBlock) override;
    void releaseResources() override;
    void reset() override;
    void processBlock (juce::AudioBuffer<float>&, juce::MidiBuffer&) override;

    [[nodiscard]] juce::AudioProcessorEditor* createEditor() override;
    [[nodiscard]] bool hasEditor() const override { return true; }

    [[nodiscard]] const juce::String getName() const override { return JucePlugin_Name; }
    [[nodiscard]] bool acceptsMidi() const override { return true; }
    [[nodiscard]] bool producesMidi() const override { return false; }
    [[nodiscard]] bool isMidiEffect() const override { return false; }
    [[nodiscard]] double getTailLengthSeconds() const override { return 0.0; }

    int getNumPrograms() override;
    int getCurrentProgram() override;
    void setCurrentProgram (int) override;
    const juce::String getProgramName (int) override;
    void changeProgramName (int, const juce::String&) override {}

    void getStateInformation (juce::MemoryBlock& destData) override;
    void setStateInformation (const void* data, int sizeInBytes) override;

    static juce::AudioProcessorValueTreeState::ParameterLayout createParameterLayout();
    [[nodiscard]] juce::StringArray presetNames() const;
    void previewDrumPad (int midiNote, float velocity = 0.9f);

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

    static constexpr int drumVoiceLevelCount = 15;

    enum class OscType
    {
        sine = 0,
        saw,
        square,
        noise,
        sample
    };

    static constexpr int maxVoices = 256;
    static constexpr int maxUnisonOscillators = 8;

    struct VoiceState
    {
        bool active = false;
        int midiNote = -1;
        float velocity = 0.0f;
        float phase = 0.0f;
        float osc2Phase = 0.0f;
        float subPhase = 0.0f;
        float fmPhase = 0.0f;
        float syncPhase = 0.0f;
        float samplePos = 0.0f;
        float noteAge = 0.0f;
        float auxPhase = 0.0f;
        float toneState = 0.0f;
        float colourState = 0.0f;
        std::array<float, maxUnisonOscillators> unisonPhases {};
        std::array<float, maxUnisonOscillators> unisonSyncPhases {};
        std::array<float, maxUnisonOscillators> unisonSamplePositions {};

        juce::ADSR ampEnv;
        juce::ADSR filterEnv;
    };

    struct RenderParameters
    {
        OscType oscType = OscType::sine;
        int polyphony = 16;
        int unisonVoices = 1;
        int lfo1Shape = 0;
        int lfo2Shape = 0;
        int arpMode = 0;
        int filterType = 0;
        int filterSlope = 0;
        int osc2Type = 0;
        int filter2Type = 0;
        int filter2Slope = 0;
        int fxType = 0;

        float masterLevel = 1.0f;
        float detune = 0.0f;
        float fmAmount = 0.0f;
        float syncAmount = 0.0f;
        float gateLength = 8.0f;
        float osc2Semitone = 0.0f;
        float osc2Detune = 0.0f;
        float osc2Mix = 0.0f;
        float subOscLevel = 0.0f;
        bool osc3Enabled = true;
        float noiseLevel = 0.0f;
        float ringModAmount = 0.0f;
        float envCurve = 0.0f;
        float cutoff = 1200.0f;
        float cutoff2 = 2200.0f;
        float resonance = 0.4f;
        float filterEnvAmount = 0.5f;
        float filterBalance = 0.0f;
        float lfo1Rate = 2.0f;
        float lfo1Pitch = 0.0f;
        float lfo2Rate = 3.0f;
        float lfo2Filter = 0.0f;
        float arpRate = 4.0f;
        float rhythmGateRate = 8.0f;
        float rhythmGateDepth = 0.0f;
        float fxMix = 0.0f;
        float fxIntensity = 0.0f;
        float delaySend = 0.0f;
        float delayTimeSec = 0.32f;
        float delayFeedback = 0.25f;
        float reverbMix = 0.0f;
        float drumMasterLevel = 1.0f;
        float drumKickAttack = 0.5f;
        float drumSnareTone = 0.5f;
        float drumSnareSnappy = 0.5f;
        int sampleBank = 0;
        float sampleStart = 0.0f;
        float sampleEnd = 1.0f;
        std::array<float, drumVoiceLevelCount> drumVoiceTunes {
            0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
            0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
            0.0f, 0.0f, 0.0f, 0.0f, 0.0f
        };
        std::array<float, drumVoiceLevelCount> drumVoiceDecays {
            0.5f, 0.5f, 0.5f, 0.5f, 0.5f,
            0.5f, 0.5f, 0.5f, 0.5f, 0.5f,
            0.5f, 0.5f, 0.5f, 0.5f, 0.5f
        };
        std::array<float, drumVoiceLevelCount> drumVoiceLevels {
            1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
            1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
            1.0f, 1.0f, 1.0f, 1.0f, 1.0f
        };

        juce::ADSR::Parameters ampEnv;
        juce::ADSR::Parameters filterEnv;
    };
    std::array<VoiceState, maxVoices> voices;
    RenderParameters renderParams;

    double currentSampleRate = 44100.0;
    juce::Random random;
    juce::AudioBuffer<float> loadedSample;
    int loadedSampleBank = -1;

    juce::dsp::StateVariableTPTFilter<float> leftFilter;
    juce::dsp::StateVariableTPTFilter<float> rightFilter;
    juce::dsp::StateVariableTPTFilter<float> leftFilterCascade;
    juce::dsp::StateVariableTPTFilter<float> rightFilterCascade;
    juce::dsp::StateVariableTPTFilter<float> leftFilter2;
    juce::dsp::StateVariableTPTFilter<float> rightFilter2;
    juce::dsp::StateVariableTPTFilter<float> leftFilter2Cascade;
    juce::dsp::StateVariableTPTFilter<float> rightFilter2Cascade;
    juce::dsp::Chorus<float> chorusLeft;
    juce::dsp::Chorus<float> chorusRight;
    juce::dsp::Chorus<float> flangerLeft;
    juce::dsp::Chorus<float> flangerRight;
    juce::dsp::Phaser<float> phaserLeft;
    juce::dsp::Phaser<float> phaserRight;
    juce::Reverb reverb;
    juce::AudioBuffer<float> delayBuffer;
    int delayWritePosition = 0;
    float currentFilterEnvPeak = 0.0f;

    float lfo1Phase = 0.0f;
    float lfo2Phase = 0.0f;
    int arpStep = 0;
    juce::Array<int> heldNotes;
    std::atomic<int> pendingPresetIndex { -1 };
    int currentProgramIndex = 0;
    bool suppressPresetCallback = false;
    juce::SpinLock pendingPreviewMidiLock;
    juce::MidiBuffer pendingPreviewMidi;

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

    [[nodiscard]] static constexpr bool isMonophonicFlavor() noexcept
    {
        return buildFlavor() == InstrumentFlavor::bassSynth;
    }

    [[nodiscard]] static constexpr int voiceLimitForFlavor() noexcept
    {
        if constexpr (buildFlavor() == InstrumentFlavor::bassSynth)
            return 1;
        if constexpr (buildFlavor() == InstrumentFlavor::drumMachine || buildFlavor() == InstrumentFlavor::drum808)
            return 8;
        if constexpr (buildFlavor() == InstrumentFlavor::stringSynth)
            return maxVoices;
        if constexpr (buildFlavor() == InstrumentFlavor::leadSynth)
            return 8;
        if constexpr (buildFlavor() == InstrumentFlavor::padSynth)
            return 8;
        if constexpr (buildFlavor() == InstrumentFlavor::pluckSynth)
            return 6;
        if constexpr (buildFlavor() == InstrumentFlavor::sampler)
            return 6;
        if constexpr (buildFlavor() == InstrumentFlavor::acid303)
            return 1;
        if constexpr (buildFlavor() == InstrumentFlavor::advanced)
            return 8;
        return maxVoices;
    }

    [[nodiscard]] static constexpr int maxUnisonForFlavor() noexcept
    {
        if constexpr (buildFlavor() == InstrumentFlavor::bassSynth
                      || buildFlavor() == InstrumentFlavor::drumMachine
                      || buildFlavor() == InstrumentFlavor::drum808
                      || buildFlavor() == InstrumentFlavor::sampler
                      || buildFlavor() == InstrumentFlavor::pluckSynth
                      || buildFlavor() == InstrumentFlavor::acid303)
            return 1;
        if constexpr (buildFlavor() == InstrumentFlavor::stringSynth)
            return maxUnisonOscillators;
        if constexpr (buildFlavor() == InstrumentFlavor::padSynth)
            return 2;
        if constexpr (buildFlavor() == InstrumentFlavor::leadSynth)
            return 1;
        if constexpr (buildFlavor() == InstrumentFlavor::advanced)
            return 2;
        return 2;
    }

    [[nodiscard]] int activeVoiceLimit() const noexcept;
    void handleMidiMessage (const juce::MidiMessage& message);
    float renderVoiceSample (VoiceState& voice);
    float renderDrumVoiceSample (VoiceState& voice);
    float oscSample (VoiceState& voice, float baseFreq, OscType type, float syncAmount);
    float oscSampleForState (float& phase, float& syncPhase, float& samplePos, float baseFreq, OscType type, float syncAmount);
    float basicOscSample (float& phase, float frequency, OscType type);
    float fmOperator (VoiceState& voice, float baseFreq, float amount);
    float lfoValue (int shape, float phase) const;
    void refreshSampleBank();
    void buildGeneratedSampleBank (int bankIndex);
    [[nodiscard]] static juce::StringArray sampleBankChoices();
    [[nodiscard]] static juce::StringArray presetChoicesForFlavor();
    void applyAdvancedEffects (juce::AudioBuffer<float>& buffer);

    void updateRenderParameters();
    void applyEnvelopeSettings();
    void advanceArpIfNeeded();
    int getArpNote() const;
    void applyPresetByIndex (int presetIndex);
    void setParameterActual (const char* paramId, float value);
    void parameterChanged (const juce::String& parameterID, float newValue) override;
    void handleAsyncUpdate() override;

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR (AdvancedVSTiAudioProcessor)
};
