#pragma once

#include <JuceHeader.h>
#include <memory>

#ifndef AIMS_INSTRUMENT_FLAVOR
#define AIMS_INSTRUMENT_FLAVOR 0
#endif

class AdvancedVSTiAudioProcessor final : public juce::AudioProcessor,
                                         private juce::AudioProcessorValueTreeState::Listener,
                                         private juce::AsyncUpdater
{
public:
    struct ExternalPadState
    {
        juce::String title;
        juce::String note;
        juce::String preset;
        juce::String sample;
        int midiNote = 36;
        bool canStepLeft = false;
        bool canStepRight = false;
    };

    struct VirusPresetMetadata
    {
        bool imported = false;
        juce::String bankLabel;
        juce::String slotLabel;
        juce::String categoryCode;
        juce::String descriptorA;
        juce::String descriptorB;
        juce::String descriptorC;
    };

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
    [[nodiscard]] bool isVec1DrumPadFlavor() const noexcept;
    [[nodiscard]] int externalPadCount() const noexcept;
    [[nodiscard]] ExternalPadState getExternalPadState (int padIndex) const;
    void stepExternalPadSample (int padIndex, int delta);
    [[nodiscard]] juce::String externalPadLevelParameterId (int padIndex) const;
    [[nodiscard]] juce::String externalPadSustainParameterId (int padIndex) const;
    [[nodiscard]] juce::String externalPadReleaseParameterId (int padIndex) const;
    [[nodiscard]] juce::MidiKeyboardState& getKeyboardState() noexcept { return keyboardState; }
    [[nodiscard]] bool toggleArpHold();
    [[nodiscard]] bool isArpHoldEnabled() const noexcept;
    void panicAllNotes();
    void auditionPresetNote (int midiNote = 60, float velocity = 0.9f, int durationMs = 420);
    [[nodiscard]] VirusPresetMetadata getVirusPresetMetadata (int presetIndex) const;

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
        acid303,
        vec1DrumPad,
        piano,
        stringEnsemble,
        violin,
        flute,
        saxophone,
        bassGuitar,
        organ
    };

    static constexpr int drumVoiceLevelCount = 15;
    static constexpr int vecPadCount = 23;

    enum class OscType
    {
        sine = 0,
        saw,
        square,
        noise,
        sample,
        hypersaw,
        wavetableFormant,
        wavetableComplex,
        wavetableMetal,
        wavetableVocal
    };

    enum class MatrixSource
    {
        off = 0,
        lfo1,
        lfo2,
        lfo3,
        filterEnv,
        ampEnv,
        velocity,
        note,
        random
    };

    enum class MatrixDestination
    {
        off = 0,
        osc1Pitch,
        osc23Pitch,
        pulseWidth,
        resonance,
        filterGain,
        cutoff1,
        cutoff2,
        shape,
        fmAmount,
        panorama,
        assign,
        ampLevel,
        filterBalance,
        oscVolume,
        subOscVolume,
        noiseVolume,
        fxMix,
        fxIntensity,
        delaySend,
        delayTime,
        delayFeedback,
        reverbMix,
        reverbTime,
        lowEqGain,
        midEqGain,
        highEqGain,
        reverbDamping,
        lowEqFreq,
        lowEqQ,
        midEqFreq,
        midEqQ,
        highEqFreq,
        highEqQ,
        masterLevel,
        detune,
        syncAmount,
        gateLength,
        filterEnvAmount,
        osc2Mix,
        ringModAmount
    };

    struct ExternalSampleData;
    struct PianoSampleLibrary;
    struct AcousticSampleLibrary;

    static constexpr int maxVoices = 256;
    static constexpr int maxUnisonOscillators = 8;

    struct VoiceState
    {
        bool active = false;
        bool arpControlled = false;
        int midiNote = -1;
        float currentMidiNote = -1.0f;
        float targetMidiNote = -1.0f;
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
        int externalPadIndex = -1;
        int externalSampleRootMidi = 60;
        double externalSamplePosition = 0.0;
        float externalSampleGain = 1.0f;
        float externalSampleTuneSemitones = 0.0f;
        int externalSampleLoopStart = 0;
        int externalSampleLoopEnd = 0;
        bool externalSampleLoopEnabled = false;
        std::shared_ptr<const ExternalSampleData> externalSample;
        std::array<float, maxUnisonOscillators> unisonPhases {};
        std::array<float, maxUnisonOscillators> unisonSyncPhases {};
        std::array<float, maxUnisonOscillators> unisonSamplePositions {};

        juce::ADSR ampEnv;
        juce::ADSR filterEnv;
    };

    struct ModMatrixTarget
    {
        int destination = static_cast<int> (MatrixDestination::off);
        float amount = 0.0f;
    };

    struct ModMatrixSlot
    {
        int source = static_cast<int> (MatrixSource::off);
        std::array<ModMatrixTarget, 3> targets {};
    };

    struct SampleModulationSums
    {
        float filterEnvPeak = 0.0f;
        float ampEnvPeak = 0.0f;
        float cutoff1 = 0.0f;
        float cutoff2 = 0.0f;
        float resonance = 0.0f;
        float filterBalance = 0.0f;
        float panorama = 0.0f;
        float fxMix = 0.0f;
        float fxIntensity = 0.0f;
        float delaySend = 0.0f;
        float delayTime = 0.0f;
        float delayFeedback = 0.0f;
        float reverbMix = 0.0f;
        float reverbTime = 0.0f;
        float reverbDamping = 0.0f;
        float lowEqGain = 0.0f;
        float lowEqFreq = 0.0f;
        float lowEqQ = 0.0f;
        float midEqGain = 0.0f;
        float midEqFreq = 0.0f;
        float midEqQ = 0.0f;
        float highEqGain = 0.0f;
        float highEqFreq = 0.0f;
        float highEqQ = 0.0f;
        float masterLevel = 0.0f;
        float filterEnvAmount = 0.0f;
        float notePitch = 0.0f;
        int activeVoices = 0;
    };

    struct RenderParameters
    {
        OscType oscType = OscType::sine;
        int polyphony = 16;
        int unisonVoices = 1;
        int lfo1Shape = 0;
        int lfo2Shape = 0;
        int lfo3Shape = 0;
        int arpMode = 0;
        bool arpEnabled = false;
        bool lfo1Enabled = true;
        bool lfo2Enabled = true;
        bool lfo3Enabled = true;
        bool lfo1EnvMode = false;
        bool lfo2EnvMode = false;
        bool lfo3EnvMode = true;
        bool filter1Enabled = true;
        bool filter2Enabled = true;
        int filterType = 0;
        int filterSlope = 0;
        int osc2Type = 0;
        int osc3Type = static_cast<int> (OscType::square);
        int filter2Type = 0;
        int filter2Slope = 0;
        int fxType = 0;

        float masterLevel = 1.0f;
        float detune = 0.0f;
        float portamentoTime = 0.0f;
        float fmAmount = 0.0f;
        float syncAmount = 0.0f;
        float gateLength = 8.0f;
        float osc1Semitone = 0.0f;
        float osc1Detune = 0.0f;
        float osc1PulseWidth = 0.5f;
        float osc2Semitone = 0.0f;
        float osc2Detune = 0.0f;
        float osc2PulseWidth = 0.5f;
        float osc3Semitone = -12.0f;
        float osc3Detune = 0.0f;
        float osc3PulseWidth = 0.5f;
        float osc2Mix = 0.0f;
        float subOscLevel = 0.0f;
        bool osc3Enabled = true;
        bool monoEnabled = false;
        float noiseLevel = 0.0f;
        float ringModAmount = 0.0f;
        float envCurve = 0.0f;
        float cutoff = 1200.0f;
        float cutoff2 = 2200.0f;
        float resonance = 0.4f;
        float resonance2 = 0.4f;
        float filterEnvAmount = 0.5f;
        float filterBalance = 0.0f;
        float panorama = 0.0f;
        float keyFollow = 0.0f;
        float lfo1Rate = 2.0f;
        float lfo1Amount = 0.0f;
        int lfo1Destination = 0;
        int lfo1AssignDestination = static_cast<int> (MatrixDestination::off);
        float lfo1Pitch = 0.0f;
        float lfo2Rate = 3.0f;
        float lfo2Amount = 0.0f;
        int lfo2Destination = 5;
        int lfo2AssignDestination = static_cast<int> (MatrixDestination::off);
        float lfo2Filter = 0.0f;
        int arpPattern = 0;
        int arpOctaves = 1;
        float arpRate = 4.0f;
        float arpSwing = 0.0f;
        float arpGate = 0.85f;
        float rhythmGateRate = 8.0f;
        float lfo3Amount = 0.0f;
        int lfo3Destination = 10;
        int lfo3AssignDestination = static_cast<int> (MatrixDestination::ampLevel);
        float rhythmGateDepth = 0.0f;
        float fxMix = 0.0f;
        float fxIntensity = 0.0f;
        float fxRate = 0.65f;
        float fxColour = 0.5f;
        float fxSpread = 0.4f;
        float delaySend = 0.0f;
        float delayTimeSec = 0.32f;
        float delayFeedback = 0.25f;
        float reverbMix = 0.0f;
        float reverbTime = 0.45f;
        float reverbDamping = 0.45f;
        float lowEqGainDb = 0.0f;
        float lowEqFreq = 220.0f;
        float lowEqQ = 0.8f;
        float midEqGainDb = 0.0f;
        float midEqFreq = 1400.0f;
        float midEqQ = 1.1f;
        float highEqGainDb = 0.0f;
        float highEqFreq = 5000.0f;
        float highEqQ = 0.8f;
        int saturationType = 0;
        std::array<ModMatrixSlot, 6> modulationMatrix {};
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
        std::array<float, vecPadCount> externalPadLevels = []
        {
            std::array<float, vecPadCount> values {};
            values.fill (1.0f);
            return values;
        }();
        std::array<float, vecPadCount> externalPadSustainTimes = []
        {
            std::array<float, vecPadCount> values {};
            values.fill (120.0f);
            return values;
        }();
        std::array<float, vecPadCount> externalPadReleaseTimes = []
        {
            std::array<float, vecPadCount> values {};
            values.fill (0.2f);
            return values;
        }();

        juce::ADSR::Parameters ampEnv;
        juce::ADSR::Parameters filterEnv;
    };

    struct ExternalSampleEntry
    {
        juce::String filePath;
        juce::String displayName;
        juce::String presetName;
    };

    struct ExternalPadDefinition
    {
        juce::String folderName;
        juce::String displayName;
        juce::String noteName;
        int midiNote = 36;
        std::vector<ExternalSampleEntry> samples;
    };

    struct ExternalSampleData
    {
        juce::AudioBuffer<float> audio;
        double sampleRate = 44100.0;
        juce::String displayName;
        juce::String presetName;
    };

    std::array<VoiceState, maxVoices> voices;
    RenderParameters renderParams;

    double currentSampleRate = 44100.0;
    juce::Random random;
    juce::AudioBuffer<float> loadedSample;
    int loadedSampleBank = -1;
    juce::AudioFormatManager audioFormatManager;
    std::vector<ExternalPadDefinition> externalPads;
    std::array<std::shared_ptr<const ExternalSampleData>, vecPadCount> externalPadSamples {};
    std::shared_ptr<const PianoSampleLibrary> pianoSampleLibrary;
    std::shared_ptr<const AcousticSampleLibrary> acousticSampleLibrary;
    int loadedAcousticSampleBank = -1;
    std::array<int, vecPadCount> loadedExternalPadIndices = []
    {
        std::array<int, vecPadCount> values {};
        values.fill (-1);
        return values;
    }();

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
    juce::dsp::IIR::Filter<float> lowEqLeft;
    juce::dsp::IIR::Filter<float> lowEqRight;
    juce::dsp::IIR::Filter<float> midEqLeft;
    juce::dsp::IIR::Filter<float> midEqRight;
    juce::dsp::IIR::Filter<float> highEqLeft;
    juce::dsp::IIR::Filter<float> highEqRight;
    juce::Reverb reverb;
    juce::AudioBuffer<float> delayBuffer;
    int delayWritePosition = 0;
    float currentFilterEnvPeak = 0.0f;

    float lfo1Phase = 0.0f;
    float lfo2Phase = 0.0f;
    float lfo3Phase = 0.0f;
    int arpPatternStep = -1;
    int arpNoteIndex = -1;
    int arpOctaveIndex = 0;
    int arpDirection = 1;
    int arpSamplesUntilNextStep = 0;
    int arpGateSamplesRemaining = 0;
    bool arpSwingPhase = false;
    bool arpWasEnabled = false;
    juce::Array<int> heldNotes;
    juce::MidiKeyboardState keyboardState;
    std::atomic<int> pendingPresetIndex { -1 };
    std::atomic<bool> pendingExternalPadReload { false };
    std::atomic<bool> arpHoldEnabled { false };
    std::atomic<bool> pendingReleaseHeldNotes { false };
    std::atomic<bool> pendingPanicAllNotes { false };
    std::atomic<int> pendingAuditionNote { -1 };
    std::atomic<int> pendingAuditionVelocity { 114 };
    std::atomic<int> pendingAuditionDurationMs { 420 };
    int currentProgramIndex = 0;
    bool suppressPresetCallback = false;
    int activeAuditionNote = -1;
    int auditionSamplesRemaining = 0;
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
#elif AIMS_INSTRUMENT_FLAVOR == 10
        return InstrumentFlavor::vec1DrumPad;
#elif AIMS_INSTRUMENT_FLAVOR == 11
        return InstrumentFlavor::piano;
#elif AIMS_INSTRUMENT_FLAVOR == 12
        return InstrumentFlavor::stringEnsemble;
#elif AIMS_INSTRUMENT_FLAVOR == 13
        return InstrumentFlavor::violin;
#elif AIMS_INSTRUMENT_FLAVOR == 14
        return InstrumentFlavor::flute;
#elif AIMS_INSTRUMENT_FLAVOR == 15
        return InstrumentFlavor::saxophone;
#elif AIMS_INSTRUMENT_FLAVOR == 16
        return InstrumentFlavor::bassGuitar;
#elif AIMS_INSTRUMENT_FLAVOR == 17
        return InstrumentFlavor::organ;
#else
        return InstrumentFlavor::advanced;
#endif
    }

    [[nodiscard]] static constexpr bool isDrumFlavor() noexcept
    {
        return buildFlavor() == InstrumentFlavor::drumMachine
               || buildFlavor() == InstrumentFlavor::drum808
               || buildFlavor() == InstrumentFlavor::vec1DrumPad;
    }

    [[nodiscard]] static constexpr bool isSynthDrumFlavor() noexcept
    {
        return buildFlavor() == InstrumentFlavor::drumMachine || buildFlavor() == InstrumentFlavor::drum808;
    }

    [[nodiscard]] static constexpr bool isAcousticFlavor() noexcept
    {
        return buildFlavor() == InstrumentFlavor::piano
               || buildFlavor() == InstrumentFlavor::stringEnsemble
               || buildFlavor() == InstrumentFlavor::violin
               || buildFlavor() == InstrumentFlavor::flute
               || buildFlavor() == InstrumentFlavor::saxophone
               || buildFlavor() == InstrumentFlavor::bassGuitar
               || buildFlavor() == InstrumentFlavor::organ;
    }

    [[nodiscard]] static constexpr bool supportsOffFilterChoice() noexcept
    {
        return isDrumFlavor() || isAcousticFlavor();
    }

    [[nodiscard]] static constexpr bool isMonophonicFlavor() noexcept
    {
        return buildFlavor() == InstrumentFlavor::bassSynth
               || buildFlavor() == InstrumentFlavor::flute
               || buildFlavor() == InstrumentFlavor::saxophone
               || buildFlavor() == InstrumentFlavor::bassGuitar;
    }

    [[nodiscard]] static constexpr int voiceLimitForFlavor() noexcept
    {
        if constexpr (buildFlavor() == InstrumentFlavor::bassSynth)
            return 1;
        else if constexpr (buildFlavor() == InstrumentFlavor::drumMachine || buildFlavor() == InstrumentFlavor::drum808)
            return 8;
        else if constexpr (buildFlavor() == InstrumentFlavor::vec1DrumPad)
            return 24;
        else if constexpr (buildFlavor() == InstrumentFlavor::stringSynth)
            return maxVoices;
        else if constexpr (buildFlavor() == InstrumentFlavor::leadSynth)
            return 8;
        else if constexpr (buildFlavor() == InstrumentFlavor::padSynth)
            return 8;
        else if constexpr (buildFlavor() == InstrumentFlavor::pluckSynth)
            return 6;
        else if constexpr (buildFlavor() == InstrumentFlavor::sampler)
            return 6;
        else if constexpr (buildFlavor() == InstrumentFlavor::acid303)
            return 1;
        else if constexpr (buildFlavor() == InstrumentFlavor::piano)
            return 16;
        else if constexpr (buildFlavor() == InstrumentFlavor::stringEnsemble)
            return 12;
        else if constexpr (buildFlavor() == InstrumentFlavor::violin)
            return 8;
        else if constexpr (buildFlavor() == InstrumentFlavor::flute)
            return 1;
        else if constexpr (buildFlavor() == InstrumentFlavor::saxophone)
            return 1;
        else if constexpr (buildFlavor() == InstrumentFlavor::bassGuitar)
            return 1;
        else if constexpr (buildFlavor() == InstrumentFlavor::organ)
            return 16;
        else if constexpr (buildFlavor() == InstrumentFlavor::advanced)
            return 8;
        else
            return maxVoices;
    }

    [[nodiscard]] static constexpr int maxUnisonForFlavor() noexcept
    {
        if constexpr (buildFlavor() == InstrumentFlavor::bassSynth
                      || buildFlavor() == InstrumentFlavor::drumMachine
                      || buildFlavor() == InstrumentFlavor::drum808
                      || buildFlavor() == InstrumentFlavor::vec1DrumPad
                      || buildFlavor() == InstrumentFlavor::sampler
                      || buildFlavor() == InstrumentFlavor::pluckSynth
                      || buildFlavor() == InstrumentFlavor::piano
                      || buildFlavor() == InstrumentFlavor::flute
                      || buildFlavor() == InstrumentFlavor::saxophone
                      || buildFlavor() == InstrumentFlavor::bassGuitar
                      || buildFlavor() == InstrumentFlavor::acid303)
            return 1;
        else if constexpr (buildFlavor() == InstrumentFlavor::stringSynth)
            return maxUnisonOscillators;
        else if constexpr (buildFlavor() == InstrumentFlavor::stringEnsemble)
            return 2;
        else if constexpr (buildFlavor() == InstrumentFlavor::violin)
            return 2;
        else if constexpr (buildFlavor() == InstrumentFlavor::padSynth)
            return 2;
        else if constexpr (buildFlavor() == InstrumentFlavor::leadSynth)
            return 1;
        else
            return 2;
    }

    [[nodiscard]] int activeVoiceLimit() const noexcept;
    void handleMidiMessage (const juce::MidiMessage& message);
    float renderVoiceSample (VoiceState& voice, SampleModulationSums& sampleModSums);
    float renderDrumVoiceSample (VoiceState& voice);
    float renderExternalPadVoiceSample (VoiceState& voice);
    float oscSample (VoiceState& voice, float baseFreq, OscType type, float syncAmount, float pulseWidth);
    float oscSampleForState (float& phase, float& syncPhase, float& samplePos, float baseFreq, OscType type, float syncAmount, float pulseWidth);
    float basicOscSample (float& phase, float frequency, OscType type, float pulseWidth);
    float wavetableOscSample (float& phase, float frequency, float shape, int variant);
    float hypersawSample (float phase, float shape) const;
    float fmOperator (VoiceState& voice, float baseFreq, float amount);
    float lfoValue (int shape, float phase) const;
    void refreshSampleBank();
    void buildGeneratedSampleBank (int bankIndex);
    [[nodiscard]] static juce::StringArray sampleBankChoices();
    [[nodiscard]] static juce::StringArray presetChoicesForFlavor();
    void initializeExternalPadLibrary();
    void initializePianoSampleLibrary();
    void initializeAcousticSampleLibrary (int bankIndex);
    void refreshExternalPadSamples();
    void assignPianoSampleToVoice (VoiceState& voice, int midiNote, float velocity);
    void assignAcousticSampleToVoice (VoiceState& voice, int midiNote, float velocity);
    void loadExternalPadSample (int padIndex);
    [[nodiscard]] std::shared_ptr<const ExternalSampleData> loadExternalSampleData (const juce::File& sourceFile,
                                                                                    const juce::String& displayName,
                                                                                    const juce::String& presetName);
    [[nodiscard]] int externalPadIndexForMidi (int midiNote) const noexcept;
    [[nodiscard]] static juce::String externalPadSampleParameterIdForIndex (int padIndex);
    [[nodiscard]] static juce::String externalPadLevelParameterIdForIndex (int padIndex);
    [[nodiscard]] static juce::String externalPadSustainParameterIdForIndex (int padIndex);
    [[nodiscard]] static juce::String externalPadReleaseParameterIdForIndex (int padIndex);
    void applyAdvancedEffects (juce::AudioBuffer<float>& buffer);

    void updateRenderParameters();
    void applyEnvelopeSettings();
    void applyPendingUiActions (juce::MidiBuffer& midiMessages, int blockSamples);
    void resetArpState();
    void startVoiceForMidiNote (int midiNote, float velocity, int externalPadIndex, bool arpControlled);
    void releaseArpVoices (bool immediate);
    void triggerArpStep();
    void applyPresetByIndex (int presetIndex);
    void setParameterActual (const char* paramId, float value);
    void parameterChanged (const juce::String& parameterID, float newValue) override;
    void handleAsyncUpdate() override;
    float renderInstrumentMultisampleVoice (VoiceState& voice, float soundingMidiNote);

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR (AdvancedVSTiAudioProcessor)
};
