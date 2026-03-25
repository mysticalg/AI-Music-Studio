#include "PluginProcessor.h"
#include "PluginEditor.h"

namespace
{
constexpr float twoPi = juce::MathConstants<float>::twoPi;
constexpr int pluginStateVersion = 2;
constexpr std::array<const char*, 15> drumVoiceLevelParamIds {
    "DRUMLEVEL_KICK",
    "DRUMLEVEL_SNARE",
    "DRUMLEVEL_RIM",
    "DRUMLEVEL_CLAP",
    "DRUMLEVEL_CLOSED_HAT",
    "DRUMLEVEL_OPEN_HAT",
    "DRUMLEVEL_LOW_TOM",
    "DRUMLEVEL_MID_TOM",
    "DRUMLEVEL_HIGH_TOM",
    "DRUMLEVEL_CRASH",
    "DRUMLEVEL_RIDE",
    "DRUMLEVEL_COWBELL",
    "DRUMLEVEL_CLAVE",
    "DRUMLEVEL_MARACA",
    "DRUMLEVEL_PERC"
};

constexpr std::array<const char*, 15> drumVoiceLevelNames {
    "Bass Drum Level",
    "Snare Level",
    "Rim Shot Level",
    "Clap Level",
    "Closed Hat Level",
    "Open Hat Level",
    "Low Tom Level",
    "Mid Tom Level",
    "High Tom Level",
    "Crash Level",
    "Ride Level",
    "Cowbell Level",
    "Clave Level",
    "Maraca Level",
    "Perc Level"
};

constexpr std::array<float, 15> drumMachineVoiceLevelDefaults {
    1.0f, 0.96f, 0.72f, 0.9f, 0.84f,
    0.88f, 0.84f, 0.82f, 0.8f, 0.78f,
    0.76f, 0.68f, 0.66f, 0.62f, 0.72f
};

constexpr std::array<float, 15> drum808VoiceLevelDefaults {
    1.08f, 0.94f, 0.7f, 0.8f, 0.72f,
    0.8f, 0.88f, 0.84f, 0.8f, 0.72f,
    0.74f, 0.68f, 0.64f, 0.66f, 0.72f
};

float paramValue (const juce::AudioProcessorValueTreeState& apvts, const char* paramId)
{
    if (const auto* value = apvts.getRawParameterValue (paramId))
        return value->load();

    jassertfalse;
    return 0.0f;
}

int paramIndex (const juce::AudioProcessorValueTreeState& apvts, const char* paramId)
{
    return juce::roundToInt (paramValue (apvts, paramId));
}

enum class DrumVoiceKind
{
    kick,
    snare,
    rim,
    clap,
    closedHat,
    openHat,
    lowTom,
    midTom,
    highTom,
    crash,
    ride,
    cowbell,
    clave,
    maraca,
    perc
};

float midiToHz (int note)
{
    return 440.0f * std::pow (2.0f, (note - 69) / 12.0f);
}

float shapedEnv (float value, float curve)
{
    if (curve >= 0.0f)
        return std::pow (juce::jlimit (0.0001f, 1.0f, value), 1.0f + 5.0f * curve);

    return 1.0f - std::pow (juce::jlimit (0.0f, 1.0f, 1.0f - value), 1.0f + 5.0f * -curve);
}

float softSaturate (float value)
{
    return std::tanh (value);
}

float smoothTowards (float target, float coeff, float& state)
{
    state += coeff * (target - state);
    return state;
}

DrumVoiceKind drumKindForMidi (int midiNote)
{
    // Bundled drum machines use a keyboard-friendly layout starting at C:
    // C kick, C# rim, D snare, D# clap, E closed hat, F open hat,
    // F# low tom, G mid tom, G# high tom, A crash, A# ride, B cowbell,
    // next C clave, C# maraca, D perc.
    if (midiNote >= 36)
    {
        switch (midiNote - 36)
        {
            case 0:  return DrumVoiceKind::kick;
            case 1:  return DrumVoiceKind::rim;
            case 2:  return DrumVoiceKind::snare;
            case 3:  return DrumVoiceKind::clap;
            case 4:  return DrumVoiceKind::closedHat;
            case 5:  return DrumVoiceKind::openHat;
            case 6:  return DrumVoiceKind::lowTom;
            case 7:  return DrumVoiceKind::midTom;
            case 8:  return DrumVoiceKind::highTom;
            case 9:  return DrumVoiceKind::crash;
            case 10: return DrumVoiceKind::ride;
            case 11: return DrumVoiceKind::cowbell;
            case 12: return DrumVoiceKind::clave;
            case 13: return DrumVoiceKind::maraca;
            case 14: return DrumVoiceKind::perc;
            default: break;
        }
    }

    switch (midiNote)
    {
        case 35: return DrumVoiceKind::kick;
        case 51:
        case 53:
        case 59: return DrumVoiceKind::ride;
        case 52:
        case 55:
        case 57: return DrumVoiceKind::crash;
        case 56: return DrumVoiceKind::cowbell;
        case 69:
        case 70:
        case 82: return DrumVoiceKind::maraca;
        case 75:
        case 76: return DrumVoiceKind::clave;
        default: return DrumVoiceKind::perc;
    }
}

bool isHatKind (DrumVoiceKind kind) noexcept
{
    return kind == DrumVoiceKind::closedHat || kind == DrumVoiceKind::openHat;
}

constexpr size_t drumVoiceIndex (DrumVoiceKind kind) noexcept
{
    return static_cast<size_t> (kind);
}

bool migrateLegacyDrumFilterChoice (juce::ValueTree state)
{
    if (! state.isValid())
        return false;

    if (state.hasProperty ("id") && state["id"].toString() == "FILTERTYPE")
    {
        const auto currentValue = static_cast<float> (state.getProperty ("value", 0.0f));
        state.setProperty ("value", juce::jlimit (1.0f, 4.0f, currentValue + 1.0f), nullptr);
        return true;
    }

    if (state.hasProperty ("FILTERTYPE"))
    {
        const auto currentValue = static_cast<float> (state.getProperty ("FILTERTYPE", 0.0f));
        state.setProperty ("FILTERTYPE", juce::jlimit (1.0f, 4.0f, currentValue + 1.0f), nullptr);
        return true;
    }

    for (int index = 0; index < state.getNumChildren(); ++index)
    {
        if (migrateLegacyDrumFilterChoice (state.getChild (index)))
            return true;
    }

    return false;
}
} // namespace

AdvancedVSTiAudioProcessor::AdvancedVSTiAudioProcessor()
    : AudioProcessor (BusesProperties().withOutput ("Output", juce::AudioChannelSet::stereo(), true)),
      apvts (*this, nullptr, "PARAMETERS", createParameterLayout())
{
    loadedSample.setSize (1, 1);
    loadedSample.clear();
    apvts.addParameterListener ("PRESET", this);
    currentProgramIndex = juce::jlimit (0, juce::jmax (0, presetChoicesForFlavor().size() - 1), paramIndex (apvts, "PRESET"));
}

AdvancedVSTiAudioProcessor::~AdvancedVSTiAudioProcessor()
{
    cancelPendingUpdate();
    apvts.removeParameterListener ("PRESET", this);
}

void AdvancedVSTiAudioProcessor::previewDrumPad (int midiNote, float velocity)
{
    if constexpr (! isDrumFlavor())
        return;

    const juce::SpinLock::ScopedLockType lock (pendingPreviewMidiLock);
    pendingPreviewMidi.addEvent (juce::MidiMessage::noteOn (1,
                                                            juce::jlimit (0, 127, midiNote),
                                                            juce::jlimit (0.0f, 1.0f, velocity)),
                                 0);
}

void AdvancedVSTiAudioProcessor::prepareToPlay (double sampleRate, int samplesPerBlock)
{
    currentSampleRate = sampleRate;
    loadedSampleBank = -1;

    juce::dsp::ProcessSpec spec { sampleRate, static_cast<juce::uint32> (samplesPerBlock), 1 };
    leftFilter.prepare (spec);
    rightFilter.prepare (spec);
    leftFilter.reset();
    rightFilter.reset();

    for (auto& voice : voices)
    {
        voice.ampEnv.setSampleRate (sampleRate);
        voice.filterEnv.setSampleRate (sampleRate);
        voice.toneState = 0.0f;
        voice.colourState = 0.0f;
    }

    updateRenderParameters();
    refreshSampleBank();
    applyEnvelopeSettings();
    reset();
}

void AdvancedVSTiAudioProcessor::releaseResources() {}

void AdvancedVSTiAudioProcessor::reset()
{
    heldNotes.clear();
    arpStep = 0;
    lfo1Phase = 0.0f;
    lfo2Phase = 0.0f;

    leftFilter.reset();
    rightFilter.reset();

    for (auto& voice : voices)
    {
        voice.active = false;
        voice.midiNote = -1;
        voice.velocity = 0.0f;
        voice.phase = 0.0f;
        voice.fmPhase = 0.0f;
        voice.syncPhase = 0.0f;
        voice.samplePos = 0.0f;
        voice.noteAge = 0.0f;
        voice.auxPhase = 0.0f;
        voice.toneState = 0.0f;
        voice.colourState = 0.0f;
        voice.ampEnv.reset();
        voice.filterEnv.reset();
    }
}

juce::StringArray AdvancedVSTiAudioProcessor::sampleBankChoices()
{
    return { "Dusty Keys", "Tape Choir", "Velvet Pluck", "Vox Chop", "Sub Stab", "Glass Bell" };
}

juce::StringArray AdvancedVSTiAudioProcessor::presetChoicesForFlavor()
{
    if constexpr (buildFlavor() == InstrumentFlavor::drumMachine)
        return { "Classic 909", "Tight Club", "Dusty Machine" };
    if constexpr (buildFlavor() == InstrumentFlavor::drum808)
        return { "Classic 808", "Deep 808", "Sharp Electro" };
    if constexpr (buildFlavor() == InstrumentFlavor::bassSynth)
        return { "Sub Bass", "Saw Bass", "Square Bass", "Picked Bass" };
    if constexpr (buildFlavor() == InstrumentFlavor::stringSynth)
        return { "Ensemble", "Soft Strings", "Synth Brass", "Warm Choir" };
    if constexpr (buildFlavor() == InstrumentFlavor::leadSynth)
        return { "Saw Lead", "Square Solo", "Soft Lead", "Brass Lead" };
    if constexpr (buildFlavor() == InstrumentFlavor::padSynth)
        return { "Warm Pad", "Analog Pad", "Choir Pad", "Slow Sweep" };
    if constexpr (buildFlavor() == InstrumentFlavor::pluckSynth)
        return { "Poly Pluck", "Bell Pluck", "Muted Pluck", "Glass Pluck" };
    if constexpr (buildFlavor() == InstrumentFlavor::sampler)
        return { "Dusty Keys", "Tape Choir", "Velvet Pluck", "Vox Chop", "Sub Stab", "Glass Bell" };
    if constexpr (buildFlavor() == InstrumentFlavor::acid303)
        return { "Acid Saw", "Acid Square", "Rounded 303", "Reso Line" };
    return { "Init Saw", "Init Square", "Wide Pad", "Mono Lead" };
}

juce::StringArray AdvancedVSTiAudioProcessor::presetNames() const
{
    return presetChoicesForFlavor();
}

int AdvancedVSTiAudioProcessor::getNumPrograms()
{
    return juce::jmax (1, presetChoicesForFlavor().size());
}

int AdvancedVSTiAudioProcessor::getCurrentProgram()
{
    return juce::jlimit (0, juce::jmax (0, getNumPrograms() - 1), currentProgramIndex);
}

void AdvancedVSTiAudioProcessor::setCurrentProgram (int index)
{
    auto* parameter = apvts.getParameter ("PRESET");
    if (parameter == nullptr)
        return;

    const auto clamped = juce::jlimit (0, juce::jmax (0, getNumPrograms() - 1), index);
    currentProgramIndex = clamped;
    parameter->setValueNotifyingHost (parameter->convertTo0to1 (static_cast<float> (clamped)));
}

const juce::String AdvancedVSTiAudioProcessor::getProgramName (int index)
{
    const auto names = presetChoicesForFlavor();
    if (juce::isPositiveAndBelow (index, names.size()))
        return names[index];
    return {};
}

void AdvancedVSTiAudioProcessor::refreshSampleBank()
{
    const auto banks = sampleBankChoices();
    const auto targetBank = juce::jlimit (0, juce::jmax (0, banks.size() - 1), renderParams.sampleBank);
    if (loadedSampleBank == targetBank && loadedSample.getNumSamples() > 1)
        return;

    buildGeneratedSampleBank (targetBank);
    loadedSampleBank = targetBank;
}

void AdvancedVSTiAudioProcessor::buildGeneratedSampleBank (int bankIndex)
{
    const auto sampleRateF = static_cast<float> (currentSampleRate);
    if (sampleRateF <= 1000.0f)
    {
        loadedSample.setSize (1, 1);
        loadedSample.clear();
        return;
    }

    float lengthSec = 1.8f;
    switch (bankIndex)
    {
        case 1: lengthSec = 2.6f; break;
        case 2: lengthSec = 1.4f; break;
        case 3: lengthSec = 1.1f; break;
        case 4: lengthSec = 1.2f; break;
        case 5: lengthSec = 2.8f; break;
        default: break;
    }

    const auto numSamples = juce::jmax (256, juce::roundToInt (lengthSec * sampleRateF));
    loadedSample.setSize (1, numSamples);
    loadedSample.clear();

    auto* dst = loadedSample.getWritePointer (0);
    float phaseA = 0.0f;
    float phaseB = 0.0f;
    float phaseC = 0.0f;
    float phaseD = 0.0f;
    float stateA = 0.0f;

    for (int i = 0; i < numSamples; ++i)
    {
        const auto t = static_cast<float> (i) / sampleRateF;
        const auto noise = random.nextFloat() * 2.0f - 1.0f;
        float sample = 0.0f;

        switch (bankIndex)
        {
            case 1:
            {
                phaseA = std::fmod (phaseA + (261.6256f / sampleRateF), 1.0f);
                phaseB = std::fmod (phaseB + (523.2511f / sampleRateF), 1.0f);
                phaseC = std::fmod (phaseC + (130.8128f / sampleRateF), 1.0f);
                phaseD = std::fmod (phaseD + (0.33f / sampleRateF), 1.0f);
                const auto swell = juce::jlimit (0.0f, 1.0f, t / 0.18f);
                const auto env = swell * std::exp (-t * 0.55f);
                const auto saw = (2.0f * phaseA) - 1.0f;
                const auto tri = 1.0f - (4.0f * std::abs (phaseB - 0.5f));
                sample = (saw * 0.46f) + (tri * 0.2f) + (std::sin (twoPi * phaseC) * 0.24f) + (noise * 0.01f);
                sample = smoothTowards (sample, 0.04f, stateA);
                sample += std::sin (twoPi * phaseD) * 0.07f * env;
                sample *= env;
                break;
            }
            case 2:
            {
                phaseA = std::fmod (phaseA + (261.6256f / sampleRateF), 1.0f);
                phaseB = std::fmod (phaseB + (527.0f / sampleRateF), 1.0f);
                const auto env = std::exp (-t * 5.4f);
                const auto click = std::exp (-t * 55.0f);
                sample = ((std::sin (twoPi * phaseA) * 0.7f) + (std::sin (twoPi * phaseB) * 0.28f)) * env;
                sample += noise * 0.17f * click;
                sample = softSaturate (sample * 1.22f);
                break;
            }
            case 3:
            {
                phaseA = std::fmod (phaseA + (220.0f / sampleRateF), 1.0f);
                phaseB = std::fmod (phaseB + (660.0f / sampleRateF), 1.0f);
                phaseC = std::fmod (phaseC + (1210.0f / sampleRateF), 1.0f);
                phaseD = std::fmod (phaseD + (330.0f / sampleRateF), 1.0f);
                const auto attack = juce::jlimit (0.0f, 1.0f, t / 0.03f);
                const auto env = attack * std::exp (-t * 1.9f);
                const auto vowel = (std::sin (twoPi * phaseA) * 0.4f) + (std::sin (twoPi * phaseB) * 0.22f) + (std::sin (twoPi * phaseC) * 0.13f);
                sample = smoothTowards (vowel + (noise * 0.05f), 0.09f, stateA);
                sample += std::sin (twoPi * phaseD) * 0.08f * std::exp (-std::pow ((t - 0.24f) * 4.2f, 2.0f));
                sample *= env;
                break;
            }
            case 4:
            {
                phaseA = std::fmod (phaseA + (65.4064f / sampleRateF), 1.0f);
                phaseB = std::fmod (phaseB + (130.8128f / sampleRateF), 1.0f);
                const auto attack = juce::jlimit (0.0f, 1.0f, t * 28.0f);
                const auto env = attack * std::exp (-t * 3.4f);
                const auto square = phaseB < 0.5f ? 1.0f : -1.0f;
                sample = ((std::sin (twoPi * phaseA) * 0.78f) + (square * 0.14f)) * env;
                sample += noise * 0.04f * std::exp (-t * 44.0f);
                sample = softSaturate (sample * 1.35f);
                break;
            }
            case 5:
            {
                phaseA = std::fmod (phaseA + (261.6256f / sampleRateF), 1.0f);
                phaseB = std::fmod (phaseB + ((261.6256f * 2.76f) / sampleRateF), 1.0f);
                phaseC = std::fmod (phaseC + ((261.6256f * 4.17f) / sampleRateF), 1.0f);
                const auto env = std::exp (-t * 1.55f);
                const auto mod = std::sin (twoPi * phaseB) * (1.8f * std::exp (-t * 2.4f));
                const auto bell = std::sin ((twoPi * phaseA) + mod);
                const auto sparkle = std::sin (twoPi * phaseC) * std::exp (-t * 3.0f);
                sample = (bell * env) + (sparkle * 0.32f) + (noise * 0.012f * std::exp (-t * 7.0f));
                break;
            }
            case 0:
            default:
            {
                phaseA = std::fmod (phaseA + (261.6256f / sampleRateF), 1.0f);
                phaseB = std::fmod (phaseB + (523.2511f / sampleRateF), 1.0f);
                phaseC = std::fmod (phaseC + (784.8768f / sampleRateF), 1.0f);
                const auto env = std::exp (-t * 2.9f);
                const auto hammer = std::exp (-t * 38.0f);
                sample = ((std::sin (twoPi * phaseA) * 0.72f) + (std::sin (twoPi * phaseB) * 0.22f) + (std::sin (twoPi * phaseC) * 0.08f)) * env;
                sample += noise * (0.026f * env + 0.11f * hammer);
                sample = smoothTowards (sample, 0.08f, stateA);
                sample += std::sin (twoPi * std::fmod (phaseA * 0.5f, 1.0f)) * 0.05f * env;
                break;
            }
        }

        dst[i] = sample;
    }

    float peak = 0.0f;
    for (int i = 0; i < numSamples; ++i)
        peak = juce::jmax (peak, std::abs (dst[i]));

    if (peak > 0.0001f)
        juce::FloatVectorOperations::multiply (dst, 0.92f / peak, numSamples);
}

void AdvancedVSTiAudioProcessor::handleMidiMessage (const juce::MidiMessage& msg)
{
    if (msg.isNoteOn())
    {
        constexpr int polyphonyLimit = voiceLimitForFlavor();
        if constexpr (isDrumFlavor())
        {
            const auto triggeredKind = drumKindForMidi (msg.getNoteNumber());
            if (isHatKind (triggeredKind))
            {
                for (auto& activeVoice : voices)
                {
                    if (! activeVoice.active)
                        continue;

                    if (isHatKind (drumKindForMidi (activeVoice.midiNote)))
                    {
                        activeVoice.active = false;
                        activeVoice.ampEnv.reset();
                        activeVoice.filterEnv.reset();
                    }
                }
            }
        }

        int voiceIndex = 0;
        bool foundVoice = false;
        float oldestVoiceAge = -1.0f;

        if constexpr (isMonophonicFlavor())
        {
            heldNotes.clear();
            for (auto& activeVoice : voices)
            {
                if (! activeVoice.active)
                    continue;

                activeVoice.active = false;
                activeVoice.ampEnv.reset();
                activeVoice.filterEnv.reset();
            }
        }

        for (int i = 0; i < polyphonyLimit; ++i)
        {
            auto& candidate = voices[static_cast<size_t> (i)];
            if (! candidate.active)
            {
                voiceIndex = i;
                foundVoice = true;
                break;
            }

            if (candidate.noteAge > oldestVoiceAge)
            {
                oldestVoiceAge = candidate.noteAge;
                voiceIndex = i;
            }
        }

        auto& voice = voices[static_cast<size_t> (voiceIndex)];
        if (! foundVoice && voice.active)
        {
            voice.ampEnv.reset();
            voice.filterEnv.reset();
        }
        voice.active = true;
        voice.midiNote = msg.getNoteNumber();
        voice.velocity = msg.getFloatVelocity();
        voice.phase = 0.0f;
        voice.fmPhase = 0.0f;
        voice.syncPhase = 0.0f;
        voice.samplePos = 0.0f;
        voice.noteAge = 0.0f;
        voice.auxPhase = 0.0f;
        voice.toneState = 0.0f;
        voice.colourState = 0.0f;
        voice.ampEnv.noteOn();
        voice.filterEnv.noteOn();

        if constexpr (! isDrumFlavor())
            heldNotes.addIfNotAlreadyThere (voice.midiNote);
        return;
    }

    if (msg.isNoteOff())
    {
        if constexpr (isDrumFlavor())
            return;

        const auto note = msg.getNoteNumber();
        heldNotes.removeAllInstancesOf (note);
        for (auto& voice : voices)
        {
            if (voice.active && voice.midiNote == note)
            {
                voice.ampEnv.noteOff();
                voice.filterEnv.noteOff();
            }
        }
    }
}

float AdvancedVSTiAudioProcessor::lfoValue (int shape, float phase) const
{
    if (shape == 1)
        return 2.0f * phase - 1.0f;
    if (shape == 2)
        return phase < 0.5f ? 1.0f : -1.0f;
    return std::sin (twoPi * phase);
}

float AdvancedVSTiAudioProcessor::fmOperator (VoiceState& voice, float baseFreq, float amount)
{
    const auto increment = baseFreq / static_cast<float> (currentSampleRate);
    voice.fmPhase = std::fmod (voice.fmPhase + increment, 1.0f);
    return std::sin (twoPi * voice.fmPhase) * amount;
}

float AdvancedVSTiAudioProcessor::oscSample (VoiceState& voice, float baseFreq, OscType type, float syncAmount)
{
    auto increment = baseFreq / static_cast<float> (currentSampleRate);

    voice.syncPhase = std::fmod (voice.syncPhase + increment * (1.0f + syncAmount), 1.0f);
    if (voice.syncPhase < increment)
        voice.phase = 0.0f;

    voice.phase = std::fmod (voice.phase + increment, 1.0f);

    switch (type)
    {
        case OscType::saw: return 2.0f * voice.phase - 1.0f;
        case OscType::square: return voice.phase < 0.5f ? 1.0f : -1.0f;
        case OscType::noise: return random.nextFloat() * 2.0f - 1.0f;
        case OscType::sample:
        {
            if (loadedSample.getNumSamples() <= 1)
                refreshSampleBank();
            if (loadedSample.getNumSamples() <= 1)
                return 0.0f;
            const auto totalSamples = loadedSample.getNumSamples();
            const auto loopStart = juce::jlimit (0, totalSamples - 1, juce::roundToInt (renderParams.sampleStart * static_cast<float> (totalSamples - 1)));
            const auto loopEnd = juce::jlimit (loopStart + 1, totalSamples, juce::roundToInt (renderParams.sampleEnd * static_cast<float> (totalSamples)));
            if (voice.samplePos < static_cast<float> (loopStart) || voice.samplePos >= static_cast<float> (loopEnd))
                voice.samplePos = static_cast<float> (loopStart);
            const auto idx = juce::jlimit (loopStart, loopEnd - 1, static_cast<int> (voice.samplePos));
            const auto sample = loadedSample.getSample (0, idx);
            const auto playbackRatio = juce::jlimit (0.125f, 8.0f, baseFreq / 261.6256f);
            voice.samplePos += playbackRatio;
            if (voice.samplePos >= static_cast<float> (loopEnd))
                voice.samplePos = static_cast<float> (loopStart);
            return sample;
        }
        case OscType::sine:
        default:
            return std::sin (twoPi * voice.phase);
    }
}

float AdvancedVSTiAudioProcessor::renderDrumVoiceSample (VoiceState& voice)
{
    if (! voice.active)
        return 0.0f;

    const auto age = voice.noteAge;
    const auto dt = 1.0f / static_cast<float> (currentSampleRate);
    const auto noise = random.nextFloat() * 2.0f - 1.0f;
    const auto kind = drumKindForMidi (voice.midiNote);
    const auto punch = juce::jlimit (0.0f, 1.0f, renderParams.fmAmount / 1000.0f);
    const auto snap = juce::jlimit (0.0f, 1.0f, renderParams.syncAmount / 4.0f);
    const auto brightness = juce::jlimit (0.0f, 1.0f, (renderParams.cutoff - 20.0f) / (20000.0f - 20.0f));
    const auto ring = juce::jlimit (0.0f, 1.0f, (renderParams.resonance - 0.1f) / 1.1f);

    float output = 0.0f;
    float duration = 0.4f;
    auto advanceSine = [dt] (float& phase, float hz) -> float
    {
        phase = std::fmod (phase + hz * dt, 1.0f);
        return std::sin (twoPi * phase);
    };
    auto advanceSquare = [dt] (float& phase, float hz) -> float
    {
        phase = std::fmod (phase + hz * dt, 1.0f);
        return phase < 0.5f ? 1.0f : -1.0f;
    };

    if constexpr (buildFlavor() == InstrumentFlavor::drum808)
    {
        switch (kind)
        {
            case DrumVoiceKind::kick:
            {
                duration = 1.45f + (renderParams.gateLength * 0.18f);
                const auto pitch = 34.0f + (82.0f + (punch * 26.0f)) * std::exp (-age * (7.5f + snap * 4.0f));
                voice.auxPhase = std::fmod (voice.auxPhase + pitch * dt, 1.0f);
                const auto body = std::sin (twoPi * voice.auxPhase) * std::exp (-age * (2.2f + ring * 0.6f));
                const auto sub = std::sin (twoPi * std::fmod (voice.auxPhase * 0.5f, 1.0f)) * std::exp (-age * 1.8f);
                const auto beater = noise * (0.03f + punch * 0.06f) * std::exp (-age * 95.0f);
                output = (body * 0.92f) + (sub * 0.42f) + beater;
                break;
            }
            case DrumVoiceKind::snare:
            {
                duration = 0.62f;
                const auto body = ((advanceSine (voice.phase, 182.0f) * 0.6f) + (advanceSine (voice.fmPhase, 331.0f) * 0.38f)) * std::exp (-age * 8.0f);
                const auto rattle = noise * std::exp (-age * (13.0f - brightness * 2.0f));
                output = (body * 0.58f) + (rattle * 0.82f);
                break;
            }
            case DrumVoiceKind::rim:
            {
                duration = 0.12f;
                const auto wood = (advanceSine (voice.phase, 1710.0f) * 0.54f) + (advanceSine (voice.fmPhase, 2440.0f) * 0.24f);
                const auto click = noise * std::exp (-age * 78.0f);
                output = (wood * std::exp (-age * 26.0f)) + (click * 0.12f);
                break;
            }
            case DrumVoiceKind::clap:
            {
                duration = 0.5f;
                const auto pulseA = std::exp (-std::pow ((age - 0.00f) * 38.0f, 2.0f));
                const auto pulseB = std::exp (-std::pow ((age - 0.034f) * 34.0f, 2.0f));
                const auto pulseC = std::exp (-std::pow ((age - 0.072f) * 30.0f, 2.0f));
                const auto tail = std::exp (-age * 7.6f);
                output = noise * ((0.82f * pulseA) + (0.78f * pulseB) + (0.66f * pulseC) + (0.4f * tail));
                break;
            }
            case DrumVoiceKind::closedHat:
            case DrumVoiceKind::openHat:
            {
                duration = kind == DrumVoiceKind::closedHat ? 0.16f : 0.72f;
                const auto metallic =
                    advanceSine (voice.phase, 4180.0f)
                    + advanceSine (voice.fmPhase, 6450.0f)
                    + advanceSine (voice.auxPhase, 8630.0f)
                    + advanceSine (voice.syncPhase, 10940.0f);
                const auto env = std::exp (-age * (kind == DrumVoiceKind::closedHat ? 34.0f : 10.5f));
                const auto airy = noise * std::exp (-age * (kind == DrumVoiceKind::closedHat ? 42.0f : 12.5f));
                output = (metallic * 0.13f + airy * 0.78f) * env;
                break;
            }
            case DrumVoiceKind::lowTom:
            case DrumVoiceKind::midTom:
            case DrumVoiceKind::highTom:
            {
                const auto basePitch = kind == DrumVoiceKind::lowTom ? 98.0f
                                       : kind == DrumVoiceKind::midTom ? 146.0f
                                                                       : 198.0f;
                duration = kind == DrumVoiceKind::lowTom ? 1.0f
                           : kind == DrumVoiceKind::midTom ? 0.86f
                                                           : 0.7f;
                const auto pitch = juce::jlimit (72.0f, 250.0f, basePitch + (voice.midiNote - 45) * 1.6f);
                const auto sweep = 1.0f + 0.42f * std::exp (-age * 5.4f);
                const auto fundamental = advanceSine (voice.auxPhase, pitch * sweep);
                const auto overtone = advanceSine (voice.syncPhase, (pitch * 1.49f) * (0.95f + std::exp (-age * 4.6f) * 0.1f));
                output = (fundamental * 0.94f * std::exp (-age * 3.4f)) + (overtone * 0.18f * std::exp (-age * 4.1f));
                break;
            }
            case DrumVoiceKind::crash:
            {
                duration = 1.28f;
                const auto cluster =
                    advanceSine (voice.phase, 2890.0f)
                    + advanceSine (voice.fmPhase, 4310.0f)
                    + advanceSine (voice.auxPhase, 6120.0f)
                    + advanceSine (voice.syncPhase, 9150.0f);
                const auto wash = noise * (0.86f + brightness * 0.08f);
                output = (cluster * 0.11f + wash * 0.82f) * std::exp (-age * 2.9f);
                output += noise * 0.08f * std::exp (-age * 10.0f);
                break;
            }
            case DrumVoiceKind::ride:
            {
                duration = 1.85f;
                const auto bell = advanceSine (voice.phase, 612.0f);
                const auto metallic =
                    advanceSine (voice.fmPhase, 3020.0f)
                    + advanceSine (voice.auxPhase, 4680.0f)
                    + advanceSine (voice.syncPhase, 7010.0f);
                output = (bell * 0.34f * std::exp (-age * 2.1f)) + ((metallic * 0.16f) + (noise * 0.22f)) * std::exp (-age * 2.7f);
                break;
            }
            case DrumVoiceKind::cowbell:
            {
                duration = 0.42f;
                const auto bellA = advanceSquare (voice.phase, 541.0f);
                const auto bellB = advanceSquare (voice.fmPhase, 811.0f);
                const auto env = std::exp (-age * 8.8f);
                output = ((bellA * 0.46f) + (bellB * 0.34f)) * env;
                break;
            }
            case DrumVoiceKind::clave:
            {
                duration = 0.14f;
                const auto body = advanceSine (voice.phase, 2470.0f) * std::exp (-age * 28.0f);
                const auto click = noise * std::exp (-age * 96.0f);
                output = body + (click * 0.05f);
                break;
            }
            case DrumVoiceKind::maraca:
            {
                duration = 0.18f;
                const auto shaker = noise * std::exp (-age * 20.0f);
                const auto grain = advanceSine (voice.phase, 5200.0f) * 0.08f * std::exp (-age * 26.0f);
                output = shaker * 0.82f + grain;
                break;
            }
            case DrumVoiceKind::perc:
            default:
            {
                duration = 0.38f;
                const auto squareA = advanceSquare (voice.phase, 540.0f);
                const auto squareB = advanceSquare (voice.fmPhase, 846.0f);
                output = (squareA * 0.42f + squareB * 0.34f) * std::exp (-age * 9.4f);
                output += noise * 0.08f * std::exp (-age * 20.0f);
                break;
            }
        }
        output = softSaturate (output * (1.16f + punch * 0.2f)) * 0.94f;
    }
    else
    {
        switch (kind)
        {
            case DrumVoiceKind::kick:
            {
                duration = 1.0f + (renderParams.gateLength * 0.08f);
                const auto pitch = 46.0f + (126.0f + (punch * 40.0f)) * std::exp (-age * (10.5f + snap * 4.2f));
                voice.auxPhase = std::fmod (voice.auxPhase + pitch * dt, 1.0f);
                const auto tone = std::sin (twoPi * voice.auxPhase) * std::exp (-age * (4.7f + ring * 1.3f));
                const auto thump = std::sin (twoPi * std::fmod (voice.auxPhase * 0.5f, 1.0f)) * 0.22f * std::exp (-age * 3.6f);
                const auto click = noise * (0.13f + punch * 0.11f) * std::exp (-age * 85.0f);
                output = tone + thump + click;
                break;
            }
            case DrumVoiceKind::snare:
            {
                duration = 0.48f;
                const auto body = ((advanceSine (voice.phase, 186.0f) * 0.56f) + (advanceSine (voice.fmPhase, 329.0f) * 0.31f)) * std::exp (-age * 10.6f);
                const auto sizzle = noise * std::exp (-age * (17.0f - brightness * 4.0f));
                output = (body * 0.58f) + (sizzle * 0.88f);
                break;
            }
            case DrumVoiceKind::rim:
            {
                duration = 0.11f;
                const auto crack = (advanceSine (voice.phase, 1825.0f) * 0.46f) + (advanceSine (voice.fmPhase, 2510.0f) * 0.26f);
                const auto click = noise * std::exp (-age * 92.0f);
                output = (crack * std::exp (-age * 30.0f)) + (click * 0.08f);
                break;
            }
            case DrumVoiceKind::clap:
            {
                duration = 0.36f;
                const auto pulseA = std::exp (-std::pow ((age - 0.00f) * 44.0f, 2.0f));
                const auto pulseB = std::exp (-std::pow ((age - 0.032f) * 42.0f, 2.0f));
                const auto pulseC = std::exp (-std::pow ((age - 0.068f) * 36.0f, 2.0f));
                const auto tail = std::exp (-age * 9.4f);
                output = noise * ((0.86f * pulseA) + (0.74f * pulseB) + (0.58f * pulseC) + (0.32f * tail));
                break;
            }
            case DrumVoiceKind::closedHat:
            case DrumVoiceKind::openHat:
            {
                duration = kind == DrumVoiceKind::closedHat ? 0.13f : 0.56f;
                const auto metallic =
                    advanceSine (voice.phase, 4620.0f)
                    + advanceSine (voice.fmPhase, 6840.0f)
                    + advanceSine (voice.auxPhase, 9470.0f)
                    + advanceSine (voice.syncPhase, 12030.0f);
                const auto body = metallic * 0.1f;
                const auto air = noise * (0.74f + brightness * 0.14f);
                const auto env = std::exp (-age * (kind == DrumVoiceKind::closedHat ? 44.0f : 12.8f));
                output = (body + air) * env;
                break;
            }
            case DrumVoiceKind::lowTom:
            case DrumVoiceKind::midTom:
            case DrumVoiceKind::highTom:
            {
                const auto basePitch = kind == DrumVoiceKind::lowTom ? 112.0f
                                       : kind == DrumVoiceKind::midTom ? 162.0f
                                                                       : 222.0f;
                duration = kind == DrumVoiceKind::lowTom ? 0.82f
                           : kind == DrumVoiceKind::midTom ? 0.66f
                                                           : 0.5f;
                const auto pitch = juce::jlimit (90.0f, 280.0f, basePitch + (voice.midiNote - 45) * 1.8f);
                const auto sweep = 0.84f + 0.31f * std::exp (-age * 6.2f);
                const auto ringTone = advanceSine (voice.auxPhase, pitch * sweep) * std::exp (-age * 5.1f);
                const auto overtone = advanceSine (voice.syncPhase, pitch * 1.52f) * 0.18f * std::exp (-age * 7.2f);
                output = ringTone + overtone + noise * 0.05f * std::exp (-age * 24.0f);
                break;
            }
            case DrumVoiceKind::crash:
            {
                duration = 1.14f;
                const auto metallic =
                    advanceSine (voice.phase, 3420.0f)
                    + advanceSine (voice.fmPhase, 5180.0f)
                    + advanceSine (voice.auxPhase, 7760.0f)
                    + advanceSine (voice.syncPhase, 10520.0f);
                const auto air = noise * (0.82f + brightness * 0.12f);
                output = (metallic * 0.12f + air * 0.78f) * std::exp (-age * 3.8f);
                break;
            }
            case DrumVoiceKind::ride:
            {
                duration = 1.72f;
                const auto bell = advanceSine (voice.phase, 680.0f);
                const auto metallic =
                    advanceSine (voice.fmPhase, 3510.0f)
                    + advanceSine (voice.auxPhase, 5630.0f)
                    + advanceSine (voice.syncPhase, 8240.0f);
                output = (bell * 0.32f * std::exp (-age * 2.4f)) + ((metallic * 0.14f) + (noise * 0.18f)) * std::exp (-age * 3.0f);
                break;
            }
            case DrumVoiceKind::cowbell:
            {
                duration = 0.34f;
                const auto bellA = advanceSquare (voice.phase, 587.0f);
                const auto bellB = advanceSquare (voice.fmPhase, 845.0f);
                const auto env = std::exp (-age * 10.5f);
                output = ((bellA * 0.4f) + (bellB * 0.28f)) * env;
                break;
            }
            case DrumVoiceKind::clave:
            {
                duration = 0.12f;
                const auto body = advanceSine (voice.phase, 2860.0f) * std::exp (-age * 36.0f);
                const auto click = noise * std::exp (-age * 120.0f);
                output = body + (click * 0.04f);
                break;
            }
            case DrumVoiceKind::maraca:
            {
                duration = 0.16f;
                const auto shaker = noise * std::exp (-age * 24.0f);
                const auto grit = advanceSine (voice.phase, 6100.0f) * 0.05f * std::exp (-age * 28.0f);
                output = shaker * 0.88f + grit;
                break;
            }
            case DrumVoiceKind::perc:
            default:
            {
                duration = 0.28f;
                const auto metallic = (advanceSine (voice.phase, 460.0f) * 0.52f) + (advanceSine (voice.fmPhase, 744.0f) * 0.31f);
                const auto chop = advanceSquare (voice.syncPhase, 1220.0f);
                output = (metallic + chop * 0.16f) * std::exp (-age * 13.0f);
                output += noise * 0.11f * std::exp (-age * 26.0f);
                break;
            }
        }
        output = softSaturate (output * (1.28f + punch * 0.28f)) * 0.92f;
    }

    voice.noteAge += dt;
    if (voice.noteAge >= duration)
        voice.active = false;

    return output * voice.velocity * renderParams.drumVoiceLevels[drumVoiceIndex (kind)];
}

float AdvancedVSTiAudioProcessor::renderVoiceSample (VoiceState& voice)
{
    if (! voice.active)
        return 0.0f;

    if constexpr (isDrumFlavor())
        return renderDrumVoiceSample (voice);

    const auto& params = renderParams;

    auto baseHz = midiToHz (voice.midiNote);
    baseHz *= std::pow (2.0f, (lfoValue (params.lfo1Shape, lfo1Phase) * params.lfo1Pitch) / 12.0f);

    const auto fm = fmOperator (voice, baseHz, params.fmAmount);

    float s = 0.0f;
    for (int i = 0; i < params.unisonVoices; ++i)
    {
        const auto spread = (static_cast<float> (i) - (params.unisonVoices - 1) * 0.5f) * params.detune;
        s += oscSample (voice, baseHz * std::pow (2.0f, spread / 12.0f) + fm, params.oscType, params.syncAmount);
    }
    s /= static_cast<float> (params.unisonVoices);

    if constexpr (buildFlavor() == InstrumentFlavor::bassSynth)
    {
        voice.auxPhase = std::fmod (voice.auxPhase + (baseHz * 0.5f) / static_cast<float> (currentSampleRate), 1.0f);
        const auto sub = std::sin (twoPi * voice.auxPhase);
        const auto squareSub = voice.auxPhase < 0.5f ? 1.0f : -1.0f;
        s = (s * 0.76f) + (sub * 0.18f) + (squareSub * 0.08f);
        s = smoothTowards (s, 0.11f, voice.toneState);
        s = softSaturate (s * 1.18f) * 0.9f;
    }
    else if constexpr (buildFlavor() == InstrumentFlavor::stringSynth)
    {
        voice.auxPhase = std::fmod (voice.auxPhase + 0.12f / static_cast<float> (currentSampleRate), 1.0f);
        const auto driftA = std::sin (twoPi * voice.auxPhase) * 0.006f;
        const auto driftB = std::sin (twoPi * std::fmod (voice.auxPhase + 0.31f, 1.0f)) * 0.008f;
        const auto ensembleA = std::sin (twoPi * std::fmod (voice.phase + driftA, 1.0f));
        const auto ensembleB = std::sin (twoPi * std::fmod ((voice.phase * 1.004f) + driftB + 0.17f, 1.0f));
        s = (s * 0.74f) + (ensembleA * 0.14f) + (ensembleB * 0.12f);
        s = smoothTowards (s, 0.04f, voice.toneState);
        s = softSaturate (s * 1.08f) * 0.9f;
    }
    else if constexpr (buildFlavor() == InstrumentFlavor::leadSynth)
    {
        voice.auxPhase = std::fmod (voice.auxPhase + 4.6f / static_cast<float> (currentSampleRate), 1.0f);
        const auto vibrato = std::sin (twoPi * voice.auxPhase) * 0.006f;
        const auto octave = std::sin (twoPi * std::fmod ((voice.phase * 2.0f) + vibrato, 1.0f));
        const auto bite = std::exp (-voice.noteAge * 4.0f);
        s = (s * 0.82f) + (octave * 0.12f * bite);
        s = softSaturate (s * 1.2f) * 0.92f;
    }
    else if constexpr (buildFlavor() == InstrumentFlavor::padSynth)
    {
        voice.auxPhase = std::fmod (voice.auxPhase + 0.08f / static_cast<float> (currentSampleRate), 1.0f);
        const auto driftA = std::sin (twoPi * voice.auxPhase) * 0.007f;
        const auto driftB = std::sin (twoPi * std::fmod (voice.auxPhase + 0.31f, 1.0f)) * 0.01f;
        const auto wideA = std::sin (twoPi * std::fmod (voice.phase + driftA, 1.0f));
        const auto wideB = std::sin (twoPi * std::fmod ((voice.phase * 0.5f) + driftB + 0.23f, 1.0f));
        s = (s * 0.72f) + (wideA * 0.12f) + (wideB * 0.1f);
        s = smoothTowards (s, 0.03f, voice.toneState);
        s = softSaturate (s * 1.04f) * 0.88f;
    }
    else if constexpr (buildFlavor() == InstrumentFlavor::pluckSynth)
    {
        const auto attackNoise = (random.nextFloat() * 2.0f - 1.0f) * 0.08f * std::exp (-voice.noteAge * 80.0f);
        const auto body = std::sin (twoPi * std::fmod ((voice.phase * 1.01f) + 0.13f, 1.0f)) * 0.12f * std::exp (-voice.noteAge * 2.8f);
        s = (s * 0.8f) + body + attackNoise;
        s = smoothTowards (s, 0.12f, voice.toneState);
        s = softSaturate (s * 1.16f) * 0.9f;
    }
    else if constexpr (buildFlavor() == InstrumentFlavor::sampler)
    {
        const auto glue = smoothTowards (s, 0.09f, voice.toneState);
        const auto body = std::sin (twoPi * std::fmod (voice.phase * 0.5f, 1.0f)) * 0.08f;
        s = (s * 0.88f) + (glue * 0.1f) + body;
        s = softSaturate (s * 1.08f) * 0.95f;
    }
    else if constexpr (buildFlavor() == InstrumentFlavor::acid303)
    {
        const auto saw = (2.0f * voice.phase) - 1.0f;
        const auto square = voice.phase < 0.5f ? 1.0f : -1.0f;
        const auto waveform = params.oscType == OscType::square ? square : saw;
        const auto squelch = std::sin (twoPi * std::fmod ((voice.phase * 2.05f) + (voice.fmPhase * 0.07f), 1.0f));
        const auto accent = juce::jlimit (0.75f, 1.35f, 0.75f + (voice.velocity * 0.85f));
        const auto notePush = std::exp (-voice.noteAge * 7.2f);
        const auto grip = juce::jmap (juce::jlimit (0.0f, 1.0f, params.fmAmount / 1000.0f), 0.14f, 0.62f);
        s = (waveform * 0.86f) + (squelch * grip * notePush);
        s += std::sin (twoPi * std::fmod (voice.phase * 0.5f, 1.0f)) * 0.05f;
        s = smoothTowards (s, 0.14f, voice.toneState);
        s = softSaturate (s * (1.45f + (params.resonance * 0.42f))) * 0.92f * accent;
    }
    else
    {
        s = softSaturate (s * 1.12f) * 0.94f;
    }

    const auto ampEnv = shapedEnv (voice.ampEnv.getNextSample(), params.envCurve);
    const auto filtEnv = shapedEnv (voice.filterEnv.getNextSample(), params.envCurve);

    voice.noteAge += 1.0f / static_cast<float> (currentSampleRate);
    const auto gatePass = voice.noteAge < params.gateLength ? 1.0f : 0.0f;

    const auto rg = 0.5f * (1.0f + std::sin (twoPi * params.rhythmGateRate * voice.noteAge));
    const auto rhythmGate = 1.0f - params.rhythmGateDepth + params.rhythmGateDepth * rg;

    auto cutoff = params.cutoff;
    cutoff += filtEnv * params.filterEnvAmount * 10000.0f;
    cutoff += lfoValue (params.lfo2Shape, lfo2Phase) * params.lfo2Filter * 5000.0f;

    leftFilter.setCutoffFrequency (juce::jlimit (20.0f, 20000.0f, cutoff));
    rightFilter.setCutoffFrequency (juce::jlimit (20.0f, 20000.0f, cutoff));

    if (! voice.ampEnv.isActive())
        voice.active = false;

    return s * ampEnv * gatePass * rhythmGate * voice.velocity;
}

void AdvancedVSTiAudioProcessor::advanceArpIfNeeded()
{
    if (heldNotes.isEmpty())
        return;

    const auto mode = renderParams.arpMode;
    if (mode == 3)
    {
        arpStep = random.nextInt (heldNotes.size());
        return;
    }

    ++arpStep;
    if (arpStep >= heldNotes.size())
        arpStep = 0;
}

int AdvancedVSTiAudioProcessor::getArpNote() const
{
    if (heldNotes.isEmpty())
        return -1;

    const auto mode = renderParams.arpMode;
    if (mode == 1)
        return heldNotes[heldNotes.size() - 1 - juce::jlimit (0, heldNotes.size() - 1, arpStep)];

    return heldNotes[juce::jlimit (0, heldNotes.size() - 1, arpStep)];
}

void AdvancedVSTiAudioProcessor::updateRenderParameters()
{
    renderParams.oscType = static_cast<OscType> (paramIndex (apvts, "OSCTYPE"));
    renderParams.unisonVoices = juce::jlimit (1, maxUnisonForFlavor(), paramIndex (apvts, "UNISON"));
    renderParams.lfo1Shape = paramIndex (apvts, "LFO1SHAPE");
    renderParams.lfo2Shape = paramIndex (apvts, "LFO2SHAPE");
    renderParams.arpMode = paramIndex (apvts, "ARPMODE");
    if constexpr (isDrumFlavor())
        renderParams.filterType = juce::jlimit (0, 4, paramIndex (apvts, "FILTERTYPE"));
    else
        renderParams.filterType = juce::jlimit (0, 3, paramIndex (apvts, "FILTERTYPE"));

    renderParams.detune = paramValue (apvts, "DETUNE");
    renderParams.fmAmount = paramValue (apvts, "FMAMOUNT");
    renderParams.syncAmount = paramValue (apvts, "SYNC");
    renderParams.gateLength = paramValue (apvts, "OSCGATE");
    renderParams.envCurve = paramValue (apvts, "ENVCURVE");
    renderParams.cutoff = paramValue (apvts, "CUTOFF");
    renderParams.resonance = paramValue (apvts, "RESONANCE");
    renderParams.filterEnvAmount = paramValue (apvts, "FILTERENVAMOUNT");
    renderParams.sampleBank = paramIndex (apvts, "SAMPLEBANK");
    renderParams.sampleStart = paramValue (apvts, "SAMPLESTART");
    renderParams.sampleEnd = juce::jlimit (0.02f, 1.0f, paramValue (apvts, "SAMPLEEND"));
    if (renderParams.sampleEnd <= renderParams.sampleStart)
        renderParams.sampleEnd = juce::jlimit (0.02f, 1.0f, renderParams.sampleStart + 0.02f);
    renderParams.lfo1Rate = paramValue (apvts, "LFO1RATE");
    renderParams.lfo1Pitch = paramValue (apvts, "LFO1PITCH");
    renderParams.lfo2Rate = paramValue (apvts, "LFO2RATE");
    renderParams.lfo2Filter = paramValue (apvts, "LFO2FILTER");
    renderParams.arpRate = paramValue (apvts, "ARPRATE");
    renderParams.rhythmGateRate = paramValue (apvts, "RHYTHMGATE_RATE");
    renderParams.rhythmGateDepth = paramValue (apvts, "RHYTHMGATE_DEPTH");

    if constexpr (isDrumFlavor())
    {
        for (size_t index = 0; index < drumVoiceLevelParamIds.size(); ++index)
            renderParams.drumVoiceLevels[index] = paramValue (apvts, drumVoiceLevelParamIds[index]);
    }
    else
    {
        renderParams.drumVoiceLevels.fill (1.0f);
    }

    renderParams.ampEnv.attack = paramValue (apvts, "AMPATTACK");
    renderParams.ampEnv.decay = paramValue (apvts, "AMPDECAY");
    renderParams.ampEnv.sustain = paramValue (apvts, "AMPSUSTAIN");
    renderParams.ampEnv.release = paramValue (apvts, "AMPRELEASE");

    renderParams.filterEnv.attack = paramValue (apvts, "FILTATTACK");
    renderParams.filterEnv.decay = paramValue (apvts, "FILTDECAY");
    renderParams.filterEnv.sustain = paramValue (apvts, "FILTSUSTAIN");
    renderParams.filterEnv.release = paramValue (apvts, "FILTRELEASE");

    refreshSampleBank();
}

void AdvancedVSTiAudioProcessor::processBlock (juce::AudioBuffer<float>& buffer, juce::MidiBuffer& midiMessages)
{
    juce::ScopedNoDenormals noDenormals;
    buffer.clear();

    updateRenderParameters();
    applyEnvelopeSettings();

    {
        const juce::SpinLock::ScopedLockType lock (pendingPreviewMidiLock);
        midiMessages.addEvents (pendingPreviewMidi, 0, -1, 0);
        pendingPreviewMidi.clear();
    }

    leftFilter.setResonance (renderParams.resonance);
    rightFilter.setResonance (renderParams.resonance);

    bool bypassFilter = false;
    if constexpr (isDrumFlavor())
    {
        bypassFilter = renderParams.filterType == 0;
        if (! bypassFilter)
        {
            const auto filterMode = juce::jlimit (0, 3, renderParams.filterType - 1);
            leftFilter.setType (static_cast<juce::dsp::StateVariableTPTFilterType> (filterMode));
            rightFilter.setType (static_cast<juce::dsp::StateVariableTPTFilterType> (filterMode));
        }
    }
    else
    {
        leftFilter.setType (static_cast<juce::dsp::StateVariableTPTFilterType> (renderParams.filterType));
        rightFilter.setType (static_cast<juce::dsp::StateVariableTPTFilterType> (renderParams.filterType));
    }

    int arpCounter = 0;
    const auto arpIntervalSamples = juce::jmax (1, static_cast<int> (currentSampleRate / juce::jmax (0.25f, renderParams.arpRate)));
    juce::MidiBuffer::Iterator midiIterator (midiMessages);
    juce::MidiMessage nextMidiMessage;
    int nextMidiSample = 0;
    auto hasMidi = midiIterator.getNextEvent (nextMidiMessage, nextMidiSample);

    for (int sample = 0; sample < buffer.getNumSamples(); ++sample)
    {
        while (hasMidi && nextMidiSample <= sample)
        {
            handleMidiMessage (nextMidiMessage);
            hasMidi = midiIterator.getNextEvent (nextMidiMessage, nextMidiSample);
        }

        if (++arpCounter >= arpIntervalSamples)
        {
            arpCounter = 0;
            advanceArpIfNeeded();
            const auto arpNote = getArpNote();
            if (arpNote >= 0)
            {
                for (auto& voice : voices)
                {
                    if (voice.active)
                    {
                        voice.midiNote = arpNote;
                        break;
                    }
                }
            }
        }

        float sum = 0.0f;
        for (auto& voice : voices)
            sum += renderVoiceSample (voice);

        const auto filteredL = bypassFilter ? sum : leftFilter.processSample (0, sum);
        const auto filteredR = bypassFilter ? sum : rightFilter.processSample (0, sum);

        buffer.setSample (0, sample, filteredL);
        buffer.setSample (1, sample, filteredR);

        lfo1Phase = std::fmod (lfo1Phase + renderParams.lfo1Rate / static_cast<float> (currentSampleRate), 1.0f);
        lfo2Phase = std::fmod (lfo2Phase + renderParams.lfo2Rate / static_cast<float> (currentSampleRate), 1.0f);
    }

    while (hasMidi)
    {
        handleMidiMessage (nextMidiMessage);
        hasMidi = midiIterator.getNextEvent (nextMidiMessage, nextMidiSample);
    }
}

void AdvancedVSTiAudioProcessor::applyEnvelopeSettings()
{
    for (auto& voice : voices)
    {
        voice.ampEnv.setParameters (renderParams.ampEnv);
        voice.filterEnv.setParameters (renderParams.filterEnv);
    }
}

juce::AudioProcessorEditor* AdvancedVSTiAudioProcessor::createEditor()
{
    return new AdvancedVSTiAudioProcessorEditor (*this);
}

void AdvancedVSTiAudioProcessor::getStateInformation (juce::MemoryBlock& destData)
{
    auto state = apvts.copyState();
    state.setProperty ("STATE_VERSION", pluginStateVersion, nullptr);
    std::unique_ptr<juce::XmlElement> xml (state.createXml());
    copyXmlToBinary (*xml, destData);
}

void AdvancedVSTiAudioProcessor::setStateInformation (const void* data, int sizeInBytes)
{
    std::unique_ptr<juce::XmlElement> xmlState (getXmlFromBinary (data, sizeInBytes));
    if (xmlState != nullptr && xmlState->hasTagName (apvts.state.getType()))
    {
        auto restoredState = juce::ValueTree::fromXml (*xmlState);
        if constexpr (isDrumFlavor())
        {
            const auto savedStateVersion = static_cast<int> (restoredState.getProperty ("STATE_VERSION", 1));
            if (savedStateVersion < pluginStateVersion)
                migrateLegacyDrumFilterChoice (restoredState);
        }

        const juce::ScopedValueSetter<bool> suppress (suppressPresetCallback, true);
        apvts.replaceState (restoredState);
    }
    pendingPresetIndex.store (-1);
    currentProgramIndex = juce::jlimit (0, juce::jmax (0, getNumPrograms() - 1), paramIndex (apvts, "PRESET"));
    updateRenderParameters();
    refreshSampleBank();
    applyEnvelopeSettings();
}

juce::AudioProcessorValueTreeState::ParameterLayout AdvancedVSTiAudioProcessor::createParameterLayout()
{
    std::vector<std::unique_ptr<juce::RangedAudioParameter>> params;

    int oscDefault = 0;
    int unisonDefault = 1;
    float detuneDefault = 0.1f;
    float fmDefault = 0.0f;
    float syncDefault = 0.0f;
    float gateDefault = 8.0f;

    float ampAttackDefault = 0.01f;
    float ampDecayDefault = 0.2f;
    float ampSustainDefault = 0.8f;
    float ampReleaseDefault = 0.4f;

    float filtAttackDefault = 0.02f;
    float filtDecayDefault = 0.3f;
    float filtSustainDefault = 0.6f;
    float filtReleaseDefault = 0.4f;
    float envCurveDefault = 0.0f;

    int filterTypeDefault = 0;
    float cutoffDefault = 1200.0f;
    float resonanceDefault = 0.4f;
    float filterEnvAmountDefault = 0.5f;
    int sampleBankDefault = 0;
    float sampleStartDefault = 0.0f;
    float sampleEndDefault = 1.0f;

    float lfo1RateDefault = 2.0f;
    int lfo1ShapeDefault = 0;
    float lfo1PitchDefault = 0.0f;

    float lfo2RateDefault = 3.0f;
    int lfo2ShapeDefault = 0;
    float lfo2FilterDefault = 0.0f;

    int presetDefault = 0;
    int arpModeDefault = 0;
    float arpRateDefault = 4.0f;
    float rhythmRateDefault = 8.0f;
    float rhythmDepthDefault = 0.0f;
    std::array<float, 15> drumVoiceLevelDefaults {
        1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
        1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
        1.0f, 1.0f, 1.0f, 1.0f, 1.0f
    };

    if constexpr (buildFlavor() == InstrumentFlavor::drumMachine)
    {
        oscDefault = 3;
        filterTypeDefault = 1;
        gateDefault = 0.42f;
        ampAttackDefault = 0.001f;
        ampDecayDefault = 0.18f;
        ampSustainDefault = 0.0f;
        ampReleaseDefault = 0.08f;
        filtAttackDefault = 0.001f;
        filtDecayDefault = 0.09f;
        filtSustainDefault = 0.0f;
        filtReleaseDefault = 0.08f;
        fmDefault = 150.0f;
        syncDefault = 0.28f;
        cutoffDefault = 14000.0f;
        resonanceDefault = 0.24f;
        filterEnvAmountDefault = 0.12f;
        rhythmRateDefault = 16.0f;
        rhythmDepthDefault = 0.0f;
        drumVoiceLevelDefaults = drumMachineVoiceLevelDefaults;
    }
    else if constexpr (buildFlavor() == InstrumentFlavor::drum808)
    {
        oscDefault = 0;
        filterTypeDefault = 1;
        gateDefault = 0.95f;
        ampAttackDefault = 0.001f;
        ampDecayDefault = 0.34f;
        ampSustainDefault = 0.0f;
        ampReleaseDefault = 0.12f;
        filtAttackDefault = 0.001f;
        filtDecayDefault = 0.16f;
        filtSustainDefault = 0.0f;
        filtReleaseDefault = 0.1f;
        fmDefault = 90.0f;
        syncDefault = 0.08f;
        cutoffDefault = 9200.0f;
        resonanceDefault = 0.16f;
        filterEnvAmountDefault = 0.05f;
        rhythmRateDefault = 16.0f;
        rhythmDepthDefault = 0.0f;
        drumVoiceLevelDefaults = drum808VoiceLevelDefaults;
    }
    else if constexpr (buildFlavor() == InstrumentFlavor::bassSynth)
    {
        oscDefault = 1;
        unisonDefault = 1;
        detuneDefault = 0.0f;
        fmDefault = 0.0f;
        ampAttackDefault = 0.005f;
        ampDecayDefault = 0.16f;
        ampSustainDefault = 0.72f;
        ampReleaseDefault = 0.12f;
        filtAttackDefault = 0.001f;
        filtDecayDefault = 0.18f;
        filtSustainDefault = 0.22f;
        filtReleaseDefault = 0.12f;
        envCurveDefault = 0.06f;
        cutoffDefault = 240.0f;
        resonanceDefault = 0.18f;
        filterEnvAmountDefault = 0.36f;
        lfo1RateDefault = 0.1f;
        lfo2RateDefault = 0.2f;
        lfo2FilterDefault = 0.0f;
    }
    else if constexpr (buildFlavor() == InstrumentFlavor::stringSynth)
    {
        oscDefault = 1;
        unisonDefault = 2;
        detuneDefault = 0.035f;
        gateDefault = 8.0f;
        ampAttackDefault = 0.3f;
        ampDecayDefault = 0.7f;
        ampSustainDefault = 0.88f;
        ampReleaseDefault = 1.1f;
        filtAttackDefault = 0.12f;
        filtDecayDefault = 0.52f;
        filtSustainDefault = 0.72f;
        filtReleaseDefault = 0.9f;
        envCurveDefault = -0.08f;
        cutoffDefault = 3200.0f;
        resonanceDefault = 0.12f;
        filterEnvAmountDefault = 0.08f;
        lfo1RateDefault = 0.08f;
        lfo1PitchDefault = 0.0f;
        lfo2RateDefault = 0.08f;
        lfo2FilterDefault = 0.0f;
        rhythmDepthDefault = 0.0f;
    }
    else if constexpr (buildFlavor() == InstrumentFlavor::leadSynth)
    {
        oscDefault = 1;
        unisonDefault = 1;
        detuneDefault = 0.01f;
        fmDefault = 0.0f;
        syncDefault = 0.0f;
        ampAttackDefault = 0.002f;
        ampDecayDefault = 0.12f;
        ampSustainDefault = 0.7f;
        ampReleaseDefault = 0.1f;
        filtAttackDefault = 0.001f;
        filtDecayDefault = 0.1f;
        filtSustainDefault = 0.34f;
        filtReleaseDefault = 0.08f;
        envCurveDefault = 0.12f;
        cutoffDefault = 2200.0f;
        resonanceDefault = 0.22f;
        filterEnvAmountDefault = 0.28f;
        lfo1RateDefault = 3.2f;
        lfo1PitchDefault = 0.0f;
        lfo2RateDefault = 0.25f;
        lfo2FilterDefault = 0.0f;
    }
    else if constexpr (buildFlavor() == InstrumentFlavor::padSynth)
    {
        oscDefault = 1;
        unisonDefault = 2;
        detuneDefault = 0.04f;
        gateDefault = 8.0f;
        ampAttackDefault = 0.5f;
        ampDecayDefault = 0.95f;
        ampSustainDefault = 0.9f;
        ampReleaseDefault = 1.5f;
        filtAttackDefault = 0.18f;
        filtDecayDefault = 0.72f;
        filtSustainDefault = 0.8f;
        filtReleaseDefault = 1.0f;
        envCurveDefault = -0.16f;
        cutoffDefault = 4200.0f;
        resonanceDefault = 0.12f;
        filterEnvAmountDefault = 0.08f;
        lfo1RateDefault = 0.08f;
        lfo1PitchDefault = 0.0f;
        lfo2RateDefault = 0.08f;
        lfo2FilterDefault = 0.0f;
        rhythmDepthDefault = 0.0f;
    }
    else if constexpr (buildFlavor() == InstrumentFlavor::pluckSynth)
    {
        oscDefault = 2;
        unisonDefault = 1;
        detuneDefault = 0.0f;
        fmDefault = 0.0f;
        syncDefault = 0.0f;
        gateDefault = 0.5f;
        ampAttackDefault = 0.001f;
        ampDecayDefault = 0.16f;
        ampSustainDefault = 0.12f;
        ampReleaseDefault = 0.12f;
        filtAttackDefault = 0.001f;
        filtDecayDefault = 0.14f;
        filtSustainDefault = 0.12f;
        filtReleaseDefault = 0.12f;
        envCurveDefault = 0.18f;
        cutoffDefault = 1800.0f;
        resonanceDefault = 0.18f;
        filterEnvAmountDefault = 0.44f;
        lfo1RateDefault = 0.1f;
        lfo2RateDefault = 0.14f;
        lfo2FilterDefault = 0.0f;
    }
    else if constexpr (buildFlavor() == InstrumentFlavor::sampler)
    {
        oscDefault = 4;
        unisonDefault = 1;
        detuneDefault = 0.0f;
        fmDefault = 0.0f;
        syncDefault = 0.0f;
        gateDefault = 8.0f;
        ampAttackDefault = 0.001f;
        ampDecayDefault = 0.55f;
        ampSustainDefault = 0.86f;
        ampReleaseDefault = 0.42f;
        filtAttackDefault = 0.001f;
        filtDecayDefault = 0.3f;
        filtSustainDefault = 0.76f;
        filtReleaseDefault = 0.32f;
        envCurveDefault = 0.04f;
        cutoffDefault = 6200.0f;
        resonanceDefault = 0.18f;
        filterEnvAmountDefault = 0.1f;
        sampleBankDefault = 0;
        sampleStartDefault = 0.0f;
        sampleEndDefault = 0.92f;
        lfo1RateDefault = 0.12f;
        lfo2RateDefault = 0.1f;
        lfo2FilterDefault = 0.0f;
    }
    else if constexpr (buildFlavor() == InstrumentFlavor::acid303)
    {
        oscDefault = 1;
        unisonDefault = 1;
        detuneDefault = 0.0f;
        fmDefault = 460.0f;
        syncDefault = 0.08f;
        gateDefault = 0.55f;
        ampAttackDefault = 0.001f;
        ampDecayDefault = 0.32f;
        ampSustainDefault = 0.0f;
        ampReleaseDefault = 0.14f;
        filtAttackDefault = 0.001f;
        filtDecayDefault = 0.36f;
        filtSustainDefault = 0.0f;
        filtReleaseDefault = 0.12f;
        envCurveDefault = 0.45f;
        cutoffDefault = 540.0f;
        resonanceDefault = 0.82f;
        filterEnvAmountDefault = 0.94f;
        lfo1RateDefault = 0.14f;
        lfo2RateDefault = 0.22f;
        lfo2FilterDefault = 0.06f;
    }

    params.push_back (std::make_unique<juce::AudioParameterChoice> ("PRESET", "Preset", presetChoicesForFlavor(), presetDefault));
    params.push_back (std::make_unique<juce::AudioParameterChoice> ("OSCTYPE", "Osc Type", juce::StringArray { "Sine", "Saw", "Square", "Noise", "Sample" }, oscDefault));
    params.push_back (std::make_unique<juce::AudioParameterChoice> ("SAMPLEBANK", "Sample Bank", sampleBankChoices(), sampleBankDefault));
    params.push_back (std::make_unique<juce::AudioParameterFloat> ("SAMPLESTART", "Sample Start", 0.0f, 0.95f, sampleStartDefault));
    params.push_back (std::make_unique<juce::AudioParameterFloat> ("SAMPLEEND", "Sample End", 0.05f, 1.0f, sampleEndDefault));
    params.push_back (std::make_unique<juce::AudioParameterInt> ("UNISON", "Unison", 1, maxUnisonForFlavor(), unisonDefault));
    params.push_back (std::make_unique<juce::AudioParameterFloat> ("DETUNE", "Detune", 0.0f, 1.0f, detuneDefault));
    params.push_back (std::make_unique<juce::AudioParameterFloat> ("FMAMOUNT", "FM Amount", 0.0f, 1000.0f, fmDefault));
    params.push_back (std::make_unique<juce::AudioParameterFloat> ("SYNC", "Sync", 0.0f, 4.0f, syncDefault));
    params.push_back (std::make_unique<juce::AudioParameterFloat> ("OSCGATE", "Osc Note Length Gate", 0.01f, 8.0f, gateDefault));

    params.push_back (std::make_unique<juce::AudioParameterFloat> ("AMPATTACK", "Amp Attack", 0.001f, 10.0f, ampAttackDefault));
    params.push_back (std::make_unique<juce::AudioParameterFloat> ("AMPDECAY", "Amp Decay", 0.001f, 10.0f, ampDecayDefault));
    params.push_back (std::make_unique<juce::AudioParameterFloat> ("AMPSUSTAIN", "Amp Sustain", 0.0f, 1.0f, ampSustainDefault));
    params.push_back (std::make_unique<juce::AudioParameterFloat> ("AMPRELEASE", "Amp Release", 0.001f, 10.0f, ampReleaseDefault));

    params.push_back (std::make_unique<juce::AudioParameterFloat> ("FILTATTACK", "Filter Attack", 0.001f, 10.0f, filtAttackDefault));
    params.push_back (std::make_unique<juce::AudioParameterFloat> ("FILTDECAY", "Filter Decay", 0.001f, 10.0f, filtDecayDefault));
    params.push_back (std::make_unique<juce::AudioParameterFloat> ("FILTSUSTAIN", "Filter Sustain", 0.0f, 1.0f, filtSustainDefault));
    params.push_back (std::make_unique<juce::AudioParameterFloat> ("FILTRELEASE", "Filter Release", 0.001f, 10.0f, filtReleaseDefault));
    params.push_back (std::make_unique<juce::AudioParameterFloat> ("ENVCURVE", "Envelope Curve", -1.0f, 1.0f, envCurveDefault));

    if constexpr (isDrumFlavor())
        params.push_back (std::make_unique<juce::AudioParameterChoice> ("FILTERTYPE", "Filter Type", juce::StringArray { "Off", "LP", "BP", "HP", "Notch" }, filterTypeDefault));
    else
        params.push_back (std::make_unique<juce::AudioParameterChoice> ("FILTERTYPE", "Filter Type", juce::StringArray { "LP", "BP", "HP", "Notch" }, filterTypeDefault));
    params.push_back (std::make_unique<juce::AudioParameterFloat> ("CUTOFF", "Cutoff", 20.0f, 20000.0f, cutoffDefault));
    params.push_back (std::make_unique<juce::AudioParameterFloat> ("RESONANCE", "Resonance", 0.1f, 1.2f, resonanceDefault));
    params.push_back (std::make_unique<juce::AudioParameterFloat> ("FILTERENVAMOUNT", "Filter Env Amount", 0.0f, 1.0f, filterEnvAmountDefault));

    params.push_back (std::make_unique<juce::AudioParameterFloat> ("LFO1RATE", "LFO1 Rate", 0.05f, 40.0f, lfo1RateDefault));
    params.push_back (std::make_unique<juce::AudioParameterChoice> ("LFO1SHAPE", "LFO1 Shape", juce::StringArray { "Sine", "Saw", "Square" }, lfo1ShapeDefault));
    params.push_back (std::make_unique<juce::AudioParameterFloat> ("LFO1PITCH", "LFO1 -> Pitch", 0.0f, 24.0f, lfo1PitchDefault));

    params.push_back (std::make_unique<juce::AudioParameterFloat> ("LFO2RATE", "LFO2 Rate", 0.05f, 40.0f, lfo2RateDefault));
    params.push_back (std::make_unique<juce::AudioParameterChoice> ("LFO2SHAPE", "LFO2 Shape", juce::StringArray { "Sine", "Saw", "Square" }, lfo2ShapeDefault));
    params.push_back (std::make_unique<juce::AudioParameterFloat> ("LFO2FILTER", "LFO2 -> Filter", 0.0f, 1.0f, lfo2FilterDefault));

    params.push_back (std::make_unique<juce::AudioParameterChoice> ("ARPMODE", "Arp Mode", juce::StringArray { "Up", "Down", "UpDown", "Random" }, arpModeDefault));
    params.push_back (std::make_unique<juce::AudioParameterFloat> ("ARPRATE", "Arp Rate", 0.25f, 16.0f, arpRateDefault));

    params.push_back (std::make_unique<juce::AudioParameterFloat> ("RHYTHMGATE_RATE", "Rhythm Gate Rate", 0.25f, 32.0f, rhythmRateDefault));
    params.push_back (std::make_unique<juce::AudioParameterFloat> ("RHYTHMGATE_DEPTH", "Rhythm Gate Depth", 0.0f, 1.0f, rhythmDepthDefault));

    if constexpr (isDrumFlavor())
    {
        for (size_t index = 0; index < drumVoiceLevelParamIds.size(); ++index)
        {
            params.push_back (std::make_unique<juce::AudioParameterFloat> (drumVoiceLevelParamIds[index],
                                                                            drumVoiceLevelNames[index],
                                                                            0.0f, 1.5f,
                                                                            drumVoiceLevelDefaults[index]));
        }
    }

    return { params.begin(), params.end() };
}

void AdvancedVSTiAudioProcessor::setParameterActual (const char* paramId, float value)
{
    auto* parameter = apvts.getParameter (paramId);
    if (parameter == nullptr)
        return;

    parameter->beginChangeGesture();
    parameter->setValueNotifyingHost (parameter->convertTo0to1 (value));
    parameter->endChangeGesture();
}

void AdvancedVSTiAudioProcessor::applyPresetByIndex (int presetIndex)
{
    const juce::ScopedValueSetter<bool> suppress (suppressPresetCallback, true);
    currentProgramIndex = juce::jlimit (0, juce::jmax (0, getNumPrograms() - 1), presetIndex);

    setParameterActual ("LFO1PITCH", 0.0f);
    setParameterActual ("LFO2FILTER", 0.0f);
    setParameterActual ("RHYTHMGATE_DEPTH", 0.0f);
    setParameterActual ("ARPMODE", 0.0f);
    setParameterActual ("ARPRATE", 4.0f);
    setParameterActual ("LFO1RATE", 0.1f);
    setParameterActual ("LFO2RATE", 0.1f);

    if constexpr (buildFlavor() == InstrumentFlavor::bassSynth)
    {
        switch (presetIndex)
        {
            case 1:
                setParameterActual ("OSCTYPE", 1.0f);
                setParameterActual ("CUTOFF", 320.0f);
                setParameterActual ("RESONANCE", 0.2f);
                setParameterActual ("FILTERENVAMOUNT", 0.34f);
                setParameterActual ("AMPDECAY", 0.16f);
                setParameterActual ("AMPSUSTAIN", 0.76f);
                break;
            case 2:
                setParameterActual ("OSCTYPE", 2.0f);
                setParameterActual ("CUTOFF", 210.0f);
                setParameterActual ("RESONANCE", 0.16f);
                setParameterActual ("FILTERENVAMOUNT", 0.28f);
                setParameterActual ("AMPDECAY", 0.14f);
                setParameterActual ("AMPSUSTAIN", 0.7f);
                break;
            case 3:
                setParameterActual ("OSCTYPE", 1.0f);
                setParameterActual ("CUTOFF", 430.0f);
                setParameterActual ("RESONANCE", 0.22f);
                setParameterActual ("FILTERENVAMOUNT", 0.46f);
                setParameterActual ("AMPDECAY", 0.11f);
                setParameterActual ("AMPSUSTAIN", 0.42f);
                setParameterActual ("AMPRELEASE", 0.09f);
                break;
            default:
                setParameterActual ("OSCTYPE", 0.0f);
                setParameterActual ("CUTOFF", 170.0f);
                setParameterActual ("RESONANCE", 0.14f);
                setParameterActual ("FILTERENVAMOUNT", 0.2f);
                setParameterActual ("AMPDECAY", 0.14f);
                setParameterActual ("AMPSUSTAIN", 0.82f);
                setParameterActual ("AMPRELEASE", 0.1f);
                break;
        }
    }
    else if constexpr (buildFlavor() == InstrumentFlavor::stringSynth)
    {
        switch (presetIndex)
        {
            case 1:
                setParameterActual ("OSCTYPE", 2.0f);
                setParameterActual ("DETUNE", 0.025f);
                setParameterActual ("CUTOFF", 2600.0f);
                setParameterActual ("AMPATTACK", 0.42f);
                setParameterActual ("AMPRELEASE", 1.4f);
                break;
            case 2:
                setParameterActual ("OSCTYPE", 1.0f);
                setParameterActual ("DETUNE", 0.02f);
                setParameterActual ("CUTOFF", 1800.0f);
                setParameterActual ("FILTERENVAMOUNT", 0.22f);
                setParameterActual ("AMPATTACK", 0.08f);
                setParameterActual ("AMPDECAY", 0.35f);
                setParameterActual ("AMPSUSTAIN", 0.62f);
                setParameterActual ("AMPRELEASE", 0.34f);
                break;
            case 3:
                setParameterActual ("OSCTYPE", 2.0f);
                setParameterActual ("DETUNE", 0.03f);
                setParameterActual ("CUTOFF", 2200.0f);
                setParameterActual ("AMPATTACK", 0.34f);
                setParameterActual ("AMPRELEASE", 1.6f);
                break;
            default:
                setParameterActual ("OSCTYPE", 1.0f);
                setParameterActual ("DETUNE", 0.035f);
                setParameterActual ("CUTOFF", 3200.0f);
                setParameterActual ("FILTERENVAMOUNT", 0.08f);
                setParameterActual ("AMPATTACK", 0.3f);
                setParameterActual ("AMPDECAY", 0.7f);
                setParameterActual ("AMPSUSTAIN", 0.88f);
                setParameterActual ("AMPRELEASE", 1.1f);
                break;
        }
    }
    else if constexpr (buildFlavor() == InstrumentFlavor::leadSynth)
    {
        switch (presetIndex)
        {
            case 1:
                setParameterActual ("OSCTYPE", 2.0f);
                setParameterActual ("CUTOFF", 2100.0f);
                setParameterActual ("RESONANCE", 0.24f);
                setParameterActual ("FILTERENVAMOUNT", 0.22f);
                break;
            case 2:
                setParameterActual ("OSCTYPE", 0.0f);
                setParameterActual ("CUTOFF", 1500.0f);
                setParameterActual ("RESONANCE", 0.18f);
                setParameterActual ("FILTERENVAMOUNT", 0.16f);
                setParameterActual ("AMPDECAY", 0.18f);
                setParameterActual ("AMPSUSTAIN", 0.78f);
                break;
            case 3:
                setParameterActual ("OSCTYPE", 1.0f);
                setParameterActual ("CUTOFF", 1700.0f);
                setParameterActual ("RESONANCE", 0.2f);
                setParameterActual ("FILTERENVAMOUNT", 0.34f);
                setParameterActual ("AMPATTACK", 0.01f);
                setParameterActual ("AMPDECAY", 0.22f);
                setParameterActual ("AMPSUSTAIN", 0.56f);
                break;
            default:
                setParameterActual ("OSCTYPE", 1.0f);
                setParameterActual ("CUTOFF", 2200.0f);
                setParameterActual ("RESONANCE", 0.22f);
                setParameterActual ("FILTERENVAMOUNT", 0.28f);
                setParameterActual ("AMPATTACK", 0.002f);
                setParameterActual ("AMPDECAY", 0.12f);
                setParameterActual ("AMPSUSTAIN", 0.7f);
                break;
        }
    }
    else if constexpr (buildFlavor() == InstrumentFlavor::padSynth)
    {
        switch (presetIndex)
        {
            case 1:
                setParameterActual ("OSCTYPE", 2.0f);
                setParameterActual ("DETUNE", 0.03f);
                setParameterActual ("CUTOFF", 3200.0f);
                setParameterActual ("AMPATTACK", 0.62f);
                setParameterActual ("AMPRELEASE", 1.9f);
                break;
            case 2:
                setParameterActual ("OSCTYPE", 2.0f);
                setParameterActual ("DETUNE", 0.02f);
                setParameterActual ("CUTOFF", 2500.0f);
                setParameterActual ("AMPATTACK", 0.72f);
                setParameterActual ("AMPRELEASE", 2.1f);
                break;
            case 3:
                setParameterActual ("OSCTYPE", 1.0f);
                setParameterActual ("DETUNE", 0.04f);
                setParameterActual ("CUTOFF", 5200.0f);
                setParameterActual ("FILTERENVAMOUNT", 0.14f);
                setParameterActual ("AMPATTACK", 1.1f);
                setParameterActual ("AMPRELEASE", 2.6f);
                break;
            default:
                setParameterActual ("OSCTYPE", 1.0f);
                setParameterActual ("DETUNE", 0.04f);
                setParameterActual ("CUTOFF", 4200.0f);
                setParameterActual ("FILTERENVAMOUNT", 0.08f);
                setParameterActual ("AMPATTACK", 0.5f);
                setParameterActual ("AMPRELEASE", 1.5f);
                break;
        }
    }
    else if constexpr (buildFlavor() == InstrumentFlavor::pluckSynth)
    {
        switch (presetIndex)
        {
            case 1:
                setParameterActual ("OSCTYPE", 0.0f);
                setParameterActual ("FMAMOUNT", 48.0f);
                setParameterActual ("CUTOFF", 2800.0f);
                setParameterActual ("AMPDECAY", 0.2f);
                break;
            case 2:
                setParameterActual ("OSCTYPE", 2.0f);
                setParameterActual ("CUTOFF", 1250.0f);
                setParameterActual ("AMPDECAY", 0.11f);
                setParameterActual ("FILTERENVAMOUNT", 0.36f);
                break;
            case 3:
                setParameterActual ("OSCTYPE", 0.0f);
                setParameterActual ("FMAMOUNT", 22.0f);
                setParameterActual ("CUTOFF", 4200.0f);
                setParameterActual ("AMPDECAY", 0.26f);
                break;
            default:
                setParameterActual ("OSCTYPE", 2.0f);
                setParameterActual ("FMAMOUNT", 0.0f);
                setParameterActual ("CUTOFF", 1800.0f);
                setParameterActual ("FILTERENVAMOUNT", 0.44f);
                setParameterActual ("AMPDECAY", 0.16f);
                break;
        }
    }
    else if constexpr (buildFlavor() == InstrumentFlavor::sampler)
    {
        setParameterActual ("SAMPLEBANK", static_cast<float> (juce::jlimit (0, juce::jmax (0, sampleBankChoices().size() - 1), presetIndex)));
        switch (presetIndex)
        {
            case 1:
                setParameterActual ("CUTOFF", 5400.0f);
                setParameterActual ("AMPRELEASE", 0.9f);
                break;
            case 2:
                setParameterActual ("CUTOFF", 3200.0f);
                setParameterActual ("AMPRELEASE", 0.22f);
                break;
            case 3:
                setParameterActual ("CUTOFF", 4800.0f);
                setParameterActual ("AMPRELEASE", 0.18f);
                break;
            case 4:
                setParameterActual ("CUTOFF", 1400.0f);
                setParameterActual ("AMPRELEASE", 0.2f);
                break;
            case 5:
                setParameterActual ("CUTOFF", 6200.0f);
                setParameterActual ("AMPRELEASE", 1.2f);
                break;
            default:
                setParameterActual ("CUTOFF", 4200.0f);
                setParameterActual ("AMPRELEASE", 0.36f);
                break;
        }
    }
    else if constexpr (buildFlavor() == InstrumentFlavor::acid303)
    {
        switch (presetIndex)
        {
            case 1:
                setParameterActual ("OSCTYPE", 2.0f);
                setParameterActual ("CUTOFF", 620.0f);
                setParameterActual ("RESONANCE", 0.74f);
                setParameterActual ("FILTERENVAMOUNT", 0.88f);
                setParameterActual ("FMAMOUNT", 360.0f);
                break;
            case 2:
                setParameterActual ("OSCTYPE", 1.0f);
                setParameterActual ("CUTOFF", 420.0f);
                setParameterActual ("RESONANCE", 0.58f);
                setParameterActual ("FILTERENVAMOUNT", 0.62f);
                setParameterActual ("FMAMOUNT", 180.0f);
                break;
            case 3:
                setParameterActual ("OSCTYPE", 1.0f);
                setParameterActual ("CUTOFF", 820.0f);
                setParameterActual ("RESONANCE", 0.92f);
                setParameterActual ("FILTERENVAMOUNT", 1.0f);
                setParameterActual ("FMAMOUNT", 520.0f);
                break;
            default:
                setParameterActual ("OSCTYPE", 1.0f);
                setParameterActual ("CUTOFF", 540.0f);
                setParameterActual ("RESONANCE", 0.82f);
                setParameterActual ("FILTERENVAMOUNT", 0.94f);
                setParameterActual ("FMAMOUNT", 460.0f);
                break;
        }
    }
    else if constexpr (buildFlavor() == InstrumentFlavor::drumMachine)
    {
        for (size_t index = 0; index < drumVoiceLevelParamIds.size(); ++index)
            setParameterActual (drumVoiceLevelParamIds[index], drumMachineVoiceLevelDefaults[index]);

        switch (presetIndex)
        {
            case 1:
                setParameterActual ("FMAMOUNT", 120.0f);
                setParameterActual ("OSCGATE", 0.34f);
                setParameterActual ("CUTOFF", 10000.0f);
                setParameterActual ("FILTERENVAMOUNT", 0.08f);
                break;
            case 2:
                setParameterActual ("FMAMOUNT", 90.0f);
                setParameterActual ("OSCGATE", 0.52f);
                setParameterActual ("CUTOFF", 7600.0f);
                setParameterActual ("FILTERENVAMOUNT", 0.18f);
                break;
            default:
                setParameterActual ("FMAMOUNT", 150.0f);
                setParameterActual ("OSCGATE", 0.42f);
                setParameterActual ("CUTOFF", 14000.0f);
                setParameterActual ("FILTERENVAMOUNT", 0.12f);
                break;
        }
    }
    else if constexpr (buildFlavor() == InstrumentFlavor::drum808)
    {
        for (size_t index = 0; index < drumVoiceLevelParamIds.size(); ++index)
            setParameterActual (drumVoiceLevelParamIds[index], drum808VoiceLevelDefaults[index]);

        switch (presetIndex)
        {
            case 1:
                setParameterActual ("FMAMOUNT", 70.0f);
                setParameterActual ("OSCGATE", 1.35f);
                setParameterActual ("CUTOFF", 8400.0f);
                setParameterActual ("FILTERENVAMOUNT", 0.04f);
                break;
            case 2:
                setParameterActual ("FMAMOUNT", 120.0f);
                setParameterActual ("OSCGATE", 0.75f);
                setParameterActual ("CUTOFF", 11200.0f);
                setParameterActual ("FILTERENVAMOUNT", 0.08f);
                break;
            default:
                setParameterActual ("FMAMOUNT", 90.0f);
                setParameterActual ("OSCGATE", 0.95f);
                setParameterActual ("CUTOFF", 9200.0f);
                setParameterActual ("FILTERENVAMOUNT", 0.05f);
                break;
        }
    }
    else
    {
        switch (presetIndex)
        {
            case 1:
                setParameterActual ("OSCTYPE", 2.0f);
                setParameterActual ("DETUNE", 0.0f);
                setParameterActual ("FMAMOUNT", 0.0f);
                setParameterActual ("SYNC", 0.0f);
                setParameterActual ("CUTOFF", 1800.0f);
                setParameterActual ("RESONANCE", 0.16f);
                setParameterActual ("FILTERENVAMOUNT", 0.22f);
                break;
            case 2:
                setParameterActual ("OSCTYPE", 1.0f);
                setParameterActual ("DETUNE", 0.05f);
                setParameterActual ("UNISON", 2.0f);
                setParameterActual ("CUTOFF", 3600.0f);
                setParameterActual ("AMPATTACK", 0.45f);
                setParameterActual ("AMPRELEASE", 1.2f);
                break;
            case 3:
                setParameterActual ("OSCTYPE", 1.0f);
                setParameterActual ("DETUNE", 0.0f);
                setParameterActual ("UNISON", 1.0f);
                setParameterActual ("CUTOFF", 2200.0f);
                setParameterActual ("AMPATTACK", 0.002f);
                setParameterActual ("AMPDECAY", 0.12f);
                setParameterActual ("AMPSUSTAIN", 0.7f);
                break;
            default:
                setParameterActual ("OSCTYPE", 1.0f);
                setParameterActual ("DETUNE", 0.0f);
                setParameterActual ("FMAMOUNT", 0.0f);
                setParameterActual ("SYNC", 0.0f);
                setParameterActual ("UNISON", 1.0f);
                setParameterActual ("CUTOFF", 2600.0f);
                setParameterActual ("RESONANCE", 0.18f);
                setParameterActual ("FILTERENVAMOUNT", 0.26f);
                break;
        }
    }

    updateRenderParameters();
    refreshSampleBank();
    applyEnvelopeSettings();
}

void AdvancedVSTiAudioProcessor::parameterChanged (const juce::String& parameterID, float newValue)
{
    if (parameterID != "PRESET" || suppressPresetCallback)
        return;

    currentProgramIndex = juce::jlimit (0, juce::jmax (0, getNumPrograms() - 1), juce::roundToInt (newValue));
    pendingPresetIndex.store (currentProgramIndex);
    triggerAsyncUpdate();
}

void AdvancedVSTiAudioProcessor::handleAsyncUpdate()
{
    const auto presetIndex = pendingPresetIndex.exchange (-1);
    if (presetIndex < 0)
        return;

    applyPresetByIndex (presetIndex);
}

juce::AudioProcessor* JUCE_CALLTYPE createPluginFilter()
{
    return new AdvancedVSTiAudioProcessor();
}
