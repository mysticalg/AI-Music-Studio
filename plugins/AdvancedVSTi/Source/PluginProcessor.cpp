#include "PluginProcessor.h"
#include "PluginEditor.h"

namespace
{
constexpr float twoPi = juce::MathConstants<float>::twoPi;

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
    closedHat,
    openHat,
    clap,
    tom,
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
    switch (midiNote)
    {
        case 36:
        case 35: return DrumVoiceKind::kick;
        case 38:
        case 40: return DrumVoiceKind::snare;
        case 42:
        case 44: return DrumVoiceKind::closedHat;
        case 46:
        case 26: return DrumVoiceKind::openHat;
        case 39:
        case 54: return DrumVoiceKind::clap;
        case 41:
        case 43:
        case 45:
        case 47:
        case 48:
        case 50: return DrumVoiceKind::tom;
        default: return DrumVoiceKind::perc;
    }
}
} // namespace

AdvancedVSTiAudioProcessor::AdvancedVSTiAudioProcessor()
    : AudioProcessor (BusesProperties().withOutput ("Output", juce::AudioChannelSet::stereo(), true)),
      apvts (*this, nullptr, "PARAMETERS", createParameterLayout())
{
    loadedSample.setSize (1, 1);
    loadedSample.clear();
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
}

void AdvancedVSTiAudioProcessor::releaseResources() {}

juce::StringArray AdvancedVSTiAudioProcessor::sampleBankChoices()
{
    return { "Dusty Keys", "Tape Choir", "Velvet Pluck", "Vox Chop", "Sub Stab", "Glass Bell" };
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

void AdvancedVSTiAudioProcessor::handleMidi (const juce::MidiBuffer& midiMessages, int)
{
    for (const auto metadata : midiMessages)
    {
        const auto msg = metadata.getMessage();

        if (msg.isNoteOn())
        {
            int voiceIndex = 0;
            for (int i = 0; i < maxVoices; ++i)
            {
                if (! voices[static_cast<size_t> (i)].active)
                {
                    voiceIndex = i;
                    break;
                }
            }

            auto& voice = voices[static_cast<size_t> (voiceIndex)];
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

            heldNotes.addIfNotAlreadyThere (voice.midiNote);
        }
        else if (msg.isNoteOff())
        {
            if constexpr (isDrumFlavor())
                continue;

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
                voice.phase = std::fmod (voice.phase + 182.0f * dt, 1.0f);
                voice.fmPhase = std::fmod (voice.fmPhase + 331.0f * dt, 1.0f);
                const auto body = ((std::sin (twoPi * voice.phase) * 0.6f) + (std::sin (twoPi * voice.fmPhase) * 0.38f)) * std::exp (-age * 8.0f);
                const auto rattle = noise * std::exp (-age * (13.0f - brightness * 2.0f));
                output = (body * 0.58f) + (rattle * 0.82f);
                break;
            }
            case DrumVoiceKind::closedHat:
            case DrumVoiceKind::openHat:
            {
                duration = kind == DrumVoiceKind::closedHat ? 0.16f : 0.72f;
                voice.phase = std::fmod (voice.phase + 412.0f * dt, 1.0f);
                const auto metallic =
                    std::sin (twoPi * voice.phase * 2.0f)
                    + std::sin (twoPi * voice.phase * 3.97f)
                    + std::sin (twoPi * voice.phase * 5.11f);
                const auto env = std::exp (-age * (kind == DrumVoiceKind::closedHat ? 34.0f : 10.5f));
                const auto airy = noise * std::exp (-age * (kind == DrumVoiceKind::closedHat ? 42.0f : 12.5f));
                output = (metallic * 0.16f + airy * 0.78f) * env;
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
            case DrumVoiceKind::tom:
            {
                duration = 0.95f;
                const auto pitch = juce::jlimit (72.0f, 215.0f, midiToHz (voice.midiNote) * 0.9f);
                const auto sweep = 1.0f + 0.42f * std::exp (-age * 5.4f);
                voice.auxPhase = std::fmod (voice.auxPhase + (pitch * sweep) * dt, 1.0f);
                output = std::sin (twoPi * voice.auxPhase) * std::exp (-age * 3.6f);
                output += std::sin (twoPi * std::fmod (voice.auxPhase * 0.5f, 1.0f)) * 0.22f * std::exp (-age * 2.7f);
                break;
            }
            case DrumVoiceKind::perc:
            default:
            {
                duration = 0.38f;
                voice.phase = std::fmod (voice.phase + 540.0f * dt, 1.0f);
                voice.fmPhase = std::fmod (voice.fmPhase + 846.0f * dt, 1.0f);
                const auto squareA = voice.phase < 0.5f ? 1.0f : -1.0f;
                const auto squareB = voice.fmPhase < 0.5f ? 1.0f : -1.0f;
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
                voice.phase = std::fmod (voice.phase + 186.0f * dt, 1.0f);
                voice.fmPhase = std::fmod (voice.fmPhase + 329.0f * dt, 1.0f);
                const auto body = ((std::sin (twoPi * voice.phase) * 0.56f) + (std::sin (twoPi * voice.fmPhase) * 0.31f)) * std::exp (-age * 10.6f);
                const auto sizzle = noise * std::exp (-age * (17.0f - brightness * 4.0f));
                output = (body * 0.58f) + (sizzle * 0.88f);
                break;
            }
            case DrumVoiceKind::closedHat:
            case DrumVoiceKind::openHat:
            {
                duration = kind == DrumVoiceKind::closedHat ? 0.13f : 0.56f;
                voice.phase = std::fmod (voice.phase + 442.0f * dt, 1.0f);
                const auto metallic =
                    std::sin (twoPi * voice.phase * 2.0f)
                    + std::sin (twoPi * voice.phase * 3.18f)
                    + std::sin (twoPi * voice.phase * 4.27f)
                    + std::sin (twoPi * voice.phase * 5.12f);
                const auto body = metallic * 0.11f;
                const auto air = noise * (0.74f + brightness * 0.14f);
                const auto env = std::exp (-age * (kind == DrumVoiceKind::closedHat ? 44.0f : 12.8f));
                output = (body + air) * env;
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
            case DrumVoiceKind::tom:
            {
                duration = 0.76f;
                const auto pitch = juce::jlimit (90.0f, 245.0f, midiToHz (voice.midiNote));
                const auto sweep = 0.84f + 0.31f * std::exp (-age * 6.2f);
                voice.auxPhase = std::fmod (voice.auxPhase + (pitch * sweep) * dt, 1.0f);
                const auto ringTone = std::sin (twoPi * voice.auxPhase) * std::exp (-age * 5.1f);
                output = ringTone + noise * 0.06f * std::exp (-age * 24.0f);
                break;
            }
            case DrumVoiceKind::perc:
            default:
            {
                duration = 0.28f;
                voice.phase = std::fmod (voice.phase + 460.0f * dt, 1.0f);
                voice.fmPhase = std::fmod (voice.fmPhase + 744.0f * dt, 1.0f);
                const auto metallic = (std::sin (twoPi * voice.phase) * 0.52f) + (std::sin (twoPi * voice.fmPhase) * 0.31f);
                const auto chop = voice.phase < 0.5f ? 1.0f : -1.0f;
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

    return output * voice.velocity;
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
        const auto subOctave = std::sin (twoPi * voice.auxPhase * 0.5f);
        const auto growl = std::sin (twoPi * std::fmod ((voice.phase * 1.98f) + (voice.fmPhase * 0.17f), 1.0f));
        const auto lowBloom = smoothTowards ((sub * 0.62f) + (subOctave * 0.38f), 0.09f, voice.colourState);
        s = (s * 0.56f) + (sub * 0.38f) + (subOctave * 0.12f) + (growl * 0.16f);
        s += lowBloom * 0.16f;
        s = smoothTowards (s, 0.18f, voice.toneState);
        s = softSaturate (s * 1.7f) * 0.92f;
    }
    else if constexpr (buildFlavor() == InstrumentFlavor::stringSynth)
    {
        voice.auxPhase = std::fmod (voice.auxPhase + 0.18f / static_cast<float> (currentSampleRate), 1.0f);
        const auto drift = std::sin (twoPi * voice.auxPhase) * 0.014f;
        const auto ensemble = std::sin (twoPi * std::fmod (voice.phase + drift, 1.0f));
        const auto octaveShimmer = std::sin (twoPi * std::fmod ((voice.phase * 1.99f) + 0.17f, 1.0f));
        const auto bowNoise = (random.nextFloat() * 2.0f - 1.0f) * 0.018f;
        const auto bloom = juce::jlimit (0.0f, 1.0f, voice.noteAge * 2.5f);
        const auto airy = smoothTowards (bowNoise * bloom, 0.05f, voice.colourState);
        s = (s * 0.72f) + (ensemble * 0.2f) + (octaveShimmer * 0.11f * bloom);
        s = smoothTowards (s + airy, 0.07f, voice.toneState);
        s = softSaturate (s * 1.22f) * 0.9f;
    }
    else if constexpr (buildFlavor() == InstrumentFlavor::leadSynth)
    {
        voice.auxPhase = std::fmod (voice.auxPhase + 5.8f / static_cast<float> (currentSampleRate), 1.0f);
        const auto vibrato = std::sin (twoPi * voice.auxPhase) * 0.018f;
        const auto octave = std::sin (twoPi * std::fmod ((voice.phase * 2.0f) + vibrato, 1.0f));
        const auto edge = std::sin (twoPi * std::fmod ((voice.phase * 3.01f) + (voice.fmPhase * 0.13f), 1.0f));
        const auto bite = std::exp (-voice.noteAge * 5.0f);
        s = (s * 0.68f) + (octave * 0.18f) + (edge * 0.15f * bite);
        s = softSaturate (s * 1.55f) * 0.92f;
    }
    else if constexpr (buildFlavor() == InstrumentFlavor::padSynth)
    {
        voice.auxPhase = std::fmod (voice.auxPhase + 0.11f / static_cast<float> (currentSampleRate), 1.0f);
        const auto driftA = std::sin (twoPi * voice.auxPhase) * 0.011f;
        const auto driftB = std::sin (twoPi * std::fmod (voice.auxPhase + 0.31f, 1.0f)) * 0.016f;
        const auto wideA = std::sin (twoPi * std::fmod (voice.phase + driftA, 1.0f));
        const auto wideB = std::sin (twoPi * std::fmod ((voice.phase * 0.5f) + driftB, 1.0f));
        const auto shimmer = (random.nextFloat() * 2.0f - 1.0f) * 0.012f;
        const auto bloom = juce::jlimit (0.0f, 1.0f, voice.noteAge * 1.4f);
        s = (s * 0.58f) + (wideA * 0.2f) + (wideB * 0.16f) + (shimmer * bloom);
        s = smoothTowards (s, 0.045f, voice.toneState);
        s = softSaturate (s * 1.12f) * 0.88f;
    }
    else if constexpr (buildFlavor() == InstrumentFlavor::pluckSynth)
    {
        const auto attackNoise = (random.nextFloat() * 2.0f - 1.0f) * std::exp (-voice.noteAge * 72.0f);
        const auto body = std::sin (twoPi * std::fmod ((voice.phase * 1.01f) + 0.13f, 1.0f)) * std::exp (-voice.noteAge * 2.2f);
        s = (s * 0.62f) + (body * 0.2f) + (attackNoise * 0.18f);
        s = smoothTowards (s, 0.18f, voice.toneState);
        s = softSaturate (s * 1.35f) * 0.9f;
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
    renderParams.unisonVoices = juce::jmax (1, paramIndex (apvts, "UNISON"));
    renderParams.lfo1Shape = paramIndex (apvts, "LFO1SHAPE");
    renderParams.lfo2Shape = paramIndex (apvts, "LFO2SHAPE");
    renderParams.arpMode = paramIndex (apvts, "ARPMODE");
    renderParams.filterType = juce::jlimit (0, 2, paramIndex (apvts, "FILTERTYPE"));

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
    handleMidi (midiMessages, buffer.getNumSamples());

    leftFilter.setResonance (renderParams.resonance);
    rightFilter.setResonance (renderParams.resonance);

    leftFilter.setType (static_cast<juce::dsp::StateVariableTPTFilterType> (renderParams.filterType));
    rightFilter.setType (static_cast<juce::dsp::StateVariableTPTFilterType> (renderParams.filterType));

    int arpCounter = 0;
    const auto arpIntervalSamples = juce::jmax (1, static_cast<int> (currentSampleRate / juce::jmax (0.25f, renderParams.arpRate)));

    for (int sample = 0; sample < buffer.getNumSamples(); ++sample)
    {
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

        auto filteredL = leftFilter.processSample (0, sum);
        auto filteredR = rightFilter.processSample (0, sum);

        buffer.setSample (0, sample, filteredL);
        buffer.setSample (1, sample, filteredR);

        lfo1Phase = std::fmod (lfo1Phase + renderParams.lfo1Rate / static_cast<float> (currentSampleRate), 1.0f);
        lfo2Phase = std::fmod (lfo2Phase + renderParams.lfo2Rate / static_cast<float> (currentSampleRate), 1.0f);
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
    std::unique_ptr<juce::XmlElement> xml (state.createXml());
    copyXmlToBinary (*xml, destData);
}

void AdvancedVSTiAudioProcessor::setStateInformation (const void* data, int sizeInBytes)
{
    std::unique_ptr<juce::XmlElement> xmlState (getXmlFromBinary (data, sizeInBytes));
    if (xmlState != nullptr && xmlState->hasTagName (apvts.state.getType()))
        apvts.replaceState (juce::ValueTree::fromXml (*xmlState));
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

    int arpModeDefault = 0;
    float arpRateDefault = 4.0f;
    float rhythmRateDefault = 8.0f;
    float rhythmDepthDefault = 0.0f;

    if constexpr (buildFlavor() == InstrumentFlavor::drumMachine)
    {
        oscDefault = 3;
        gateDefault = 0.42f;
        ampAttackDefault = 0.001f;
        ampDecayDefault = 0.18f;
        ampSustainDefault = 0.0f;
        ampReleaseDefault = 0.08f;
        filtAttackDefault = 0.001f;
        filtDecayDefault = 0.09f;
        filtSustainDefault = 0.0f;
        filtReleaseDefault = 0.08f;
        fmDefault = 180.0f;
        syncDefault = 0.38f;
        cutoffDefault = 14000.0f;
        resonanceDefault = 0.34f;
        filterEnvAmountDefault = 0.18f;
        rhythmRateDefault = 16.0f;
        rhythmDepthDefault = 0.08f;
    }
    else if constexpr (buildFlavor() == InstrumentFlavor::drum808)
    {
        oscDefault = 0;
        gateDefault = 0.95f;
        ampAttackDefault = 0.001f;
        ampDecayDefault = 0.34f;
        ampSustainDefault = 0.0f;
        ampReleaseDefault = 0.12f;
        filtAttackDefault = 0.001f;
        filtDecayDefault = 0.16f;
        filtSustainDefault = 0.0f;
        filtReleaseDefault = 0.1f;
        fmDefault = 120.0f;
        syncDefault = 0.12f;
        cutoffDefault = 9200.0f;
        resonanceDefault = 0.22f;
        filterEnvAmountDefault = 0.08f;
        rhythmRateDefault = 16.0f;
    }
    else if constexpr (buildFlavor() == InstrumentFlavor::bassSynth)
    {
        oscDefault = 1;
        unisonDefault = 2;
        detuneDefault = 0.05f;
        fmDefault = 8.0f;
        ampAttackDefault = 0.005f;
        ampDecayDefault = 0.24f;
        ampSustainDefault = 0.78f;
        ampReleaseDefault = 0.28f;
        filtAttackDefault = 0.001f;
        filtDecayDefault = 0.28f;
        filtSustainDefault = 0.24f;
        filtReleaseDefault = 0.22f;
        envCurveDefault = 0.2f;
        cutoffDefault = 280.0f;
        resonanceDefault = 0.38f;
        filterEnvAmountDefault = 0.82f;
        lfo1RateDefault = 0.18f;
        lfo2RateDefault = 0.42f;
        lfo2FilterDefault = 0.12f;
    }
    else if constexpr (buildFlavor() == InstrumentFlavor::stringSynth)
    {
        oscDefault = 1;
        unisonDefault = 5;
        detuneDefault = 0.22f;
        gateDefault = 8.0f;
        ampAttackDefault = 0.42f;
        ampDecayDefault = 1.1f;
        ampSustainDefault = 0.86f;
        ampReleaseDefault = 2.2f;
        filtAttackDefault = 0.18f;
        filtDecayDefault = 0.95f;
        filtSustainDefault = 0.72f;
        filtReleaseDefault = 1.8f;
        envCurveDefault = -0.15f;
        cutoffDefault = 4600.0f;
        resonanceDefault = 0.18f;
        filterEnvAmountDefault = 0.22f;
        lfo1RateDefault = 0.16f;
        lfo1PitchDefault = 0.12f;
        lfo2RateDefault = 0.14f;
        lfo2FilterDefault = 0.18f;
        rhythmDepthDefault = 0.05f;
    }
    else if constexpr (buildFlavor() == InstrumentFlavor::leadSynth)
    {
        oscDefault = 1;
        unisonDefault = 3;
        detuneDefault = 0.08f;
        fmDefault = 24.0f;
        syncDefault = 0.55f;
        ampAttackDefault = 0.002f;
        ampDecayDefault = 0.16f;
        ampSustainDefault = 0.68f;
        ampReleaseDefault = 0.22f;
        filtAttackDefault = 0.001f;
        filtDecayDefault = 0.18f;
        filtSustainDefault = 0.22f;
        filtReleaseDefault = 0.16f;
        envCurveDefault = 0.28f;
        cutoffDefault = 3200.0f;
        resonanceDefault = 0.48f;
        filterEnvAmountDefault = 0.54f;
        lfo1RateDefault = 5.8f;
        lfo1PitchDefault = 0.25f;
        lfo2RateDefault = 0.9f;
        lfo2FilterDefault = 0.14f;
    }
    else if constexpr (buildFlavor() == InstrumentFlavor::padSynth)
    {
        oscDefault = 1;
        unisonDefault = 6;
        detuneDefault = 0.3f;
        gateDefault = 8.0f;
        ampAttackDefault = 0.65f;
        ampDecayDefault = 1.5f;
        ampSustainDefault = 0.9f;
        ampReleaseDefault = 2.9f;
        filtAttackDefault = 0.28f;
        filtDecayDefault = 1.2f;
        filtSustainDefault = 0.84f;
        filtReleaseDefault = 2.1f;
        envCurveDefault = -0.28f;
        cutoffDefault = 6200.0f;
        resonanceDefault = 0.16f;
        filterEnvAmountDefault = 0.18f;
        lfo1RateDefault = 0.11f;
        lfo1PitchDefault = 0.08f;
        lfo2RateDefault = 0.19f;
        lfo2FilterDefault = 0.2f;
        rhythmDepthDefault = 0.03f;
    }
    else if constexpr (buildFlavor() == InstrumentFlavor::pluckSynth)
    {
        oscDefault = 2;
        unisonDefault = 2;
        detuneDefault = 0.03f;
        fmDefault = 10.0f;
        syncDefault = 0.18f;
        gateDefault = 0.75f;
        ampAttackDefault = 0.001f;
        ampDecayDefault = 0.19f;
        ampSustainDefault = 0.12f;
        ampReleaseDefault = 0.16f;
        filtAttackDefault = 0.001f;
        filtDecayDefault = 0.16f;
        filtSustainDefault = 0.08f;
        filtReleaseDefault = 0.14f;
        envCurveDefault = 0.38f;
        cutoffDefault = 2600.0f;
        resonanceDefault = 0.34f;
        filterEnvAmountDefault = 0.9f;
        lfo1RateDefault = 0.2f;
        lfo2RateDefault = 0.35f;
        lfo2FilterDefault = 0.08f;
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
        cutoffDefault = 6800.0f;
        resonanceDefault = 0.22f;
        filterEnvAmountDefault = 0.16f;
        sampleBankDefault = 0;
        sampleStartDefault = 0.0f;
        sampleEndDefault = 0.92f;
        lfo1RateDefault = 0.18f;
        lfo2RateDefault = 0.12f;
        lfo2FilterDefault = 0.06f;
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

    params.push_back (std::make_unique<juce::AudioParameterChoice> ("OSCTYPE", "Osc Type", juce::StringArray { "Sine", "Saw", "Square", "Noise", "Sample" }, oscDefault));
    params.push_back (std::make_unique<juce::AudioParameterChoice> ("SAMPLEBANK", "Sample Bank", sampleBankChoices(), sampleBankDefault));
    params.push_back (std::make_unique<juce::AudioParameterFloat> ("SAMPLESTART", "Sample Start", 0.0f, 0.95f, sampleStartDefault));
    params.push_back (std::make_unique<juce::AudioParameterFloat> ("SAMPLEEND", "Sample End", 0.05f, 1.0f, sampleEndDefault));
    params.push_back (std::make_unique<juce::AudioParameterInt> ("UNISON", "Unison", 1, 8, unisonDefault));
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

    return { params.begin(), params.end() };
}

juce::AudioProcessor* JUCE_CALLTYPE createPluginFilter()
{
    return new AdvancedVSTiAudioProcessor();
}
