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
    applyEnvelopeSettings();
}

void AdvancedVSTiAudioProcessor::releaseResources() {}

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
            if constexpr (buildFlavor() == InstrumentFlavor::drumMachine)
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
                return 0.0f;
            const auto idx = static_cast<int> (voice.samplePos);
            const auto sample = loadedSample.getSample (0, juce::jlimit (0, loadedSample.getNumSamples() - 1, idx));
            voice.samplePos += (baseFreq / 261.6256f);
            if (voice.samplePos >= static_cast<float> (loadedSample.getNumSamples()))
                voice.samplePos = 0.0f;
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

    float output = 0.0f;
    float duration = 0.4f;

    switch (kind)
    {
        case DrumVoiceKind::kick:
        {
            duration = 1.0f;
            const auto pitch = 42.0f + 120.0f * std::exp (-age * 10.0f);
            voice.auxPhase = std::fmod (voice.auxPhase + pitch * dt, 1.0f);
            const auto tone = std::sin (twoPi * voice.auxPhase) * std::exp (-age * 4.8f);
            const auto click = noise * 0.18f * std::exp (-age * 70.0f);
            output = tone + click;
            break;
        }
        case DrumVoiceKind::snare:
        {
            duration = 0.45f;
            voice.auxPhase = std::fmod (voice.auxPhase + 190.0f * dt, 1.0f);
            const auto body = std::sin (twoPi * voice.auxPhase) * std::exp (-age * 11.0f);
            const auto sizzle = noise * std::exp (-age * 16.0f);
            output = body * 0.35f + sizzle * 0.85f;
            break;
        }
        case DrumVoiceKind::closedHat:
        {
            duration = 0.14f;
            output = noise * std::exp (-age * 48.0f);
            break;
        }
        case DrumVoiceKind::openHat:
        {
            duration = 0.55f;
            output = noise * std::exp (-age * 12.5f);
            break;
        }
        case DrumVoiceKind::clap:
        {
            duration = 0.35f;
            const auto pulseA = std::exp (-std::pow ((age - 0.00f) * 45.0f, 2.0f));
            const auto pulseB = std::exp (-std::pow ((age - 0.04f) * 40.0f, 2.0f));
            const auto pulseC = std::exp (-std::pow ((age - 0.09f) * 35.0f, 2.0f));
            const auto tail = std::exp (-age * 9.0f);
            output = noise * (0.8f * pulseA + 0.7f * pulseB + 0.6f * pulseC + 0.35f * tail);
            break;
        }
        case DrumVoiceKind::tom:
        {
            duration = 0.7f;
            const auto pitch = juce::jlimit (85.0f, 240.0f, midiToHz (voice.midiNote));
            const auto sweep = 0.65f + 0.35f * std::exp (-age * 6.0f);
            voice.auxPhase = std::fmod (voice.auxPhase + (pitch * sweep) * dt, 1.0f);
            output = std::sin (twoPi * voice.auxPhase) * std::exp (-age * 5.6f);
            break;
        }
        case DrumVoiceKind::perc:
        default:
        {
            duration = 0.3f;
            voice.auxPhase = std::fmod (voice.auxPhase + 320.0f * dt, 1.0f);
            const auto tone = std::sin (twoPi * voice.auxPhase * 1.7f) * std::exp (-age * 14.0f);
            output = tone * 0.45f + noise * 0.45f * std::exp (-age * 24.0f);
            break;
        }
    }

    voice.noteAge += dt;
    if (voice.noteAge >= duration)
        voice.active = false;

    return softSaturate (output * 1.3f) * 0.88f * voice.velocity;
}

float AdvancedVSTiAudioProcessor::renderVoiceSample (VoiceState& voice)
{
    if (! voice.active)
        return 0.0f;

    if constexpr (buildFlavor() == InstrumentFlavor::drumMachine)
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
        gateDefault = 0.18f;
        ampAttackDefault = 0.001f;
        ampDecayDefault = 0.08f;
        ampSustainDefault = 0.0f;
        ampReleaseDefault = 0.05f;
        filtAttackDefault = 0.001f;
        filtDecayDefault = 0.05f;
        filtSustainDefault = 0.0f;
        filtReleaseDefault = 0.05f;
        cutoffDefault = 14000.0f;
        resonanceDefault = 0.2f;
        filterEnvAmountDefault = 0.0f;
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

    params.push_back (std::make_unique<juce::AudioParameterChoice> ("OSCTYPE", "Osc Type", juce::StringArray { "Sine", "Saw", "Square", "Noise", "Sample" }, oscDefault));
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
