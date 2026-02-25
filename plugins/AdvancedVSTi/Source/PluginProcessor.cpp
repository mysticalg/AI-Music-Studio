#include "PluginProcessor.h"
#include "PluginEditor.h"

namespace
{
constexpr float twoPi = juce::MathConstants<float>::twoPi;

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
    juce::ignoreUnused (samplesPerBlock);
    currentSampleRate = sampleRate;

    juce::dsp::ProcessSpec spec { sampleRate, static_cast<juce::uint32> (samplesPerBlock), 1 };
    leftFilter.prepare (spec);
    rightFilter.prepare (spec);

    for (auto& voice : voices)
    {
        voice.ampEnv.setSampleRate (sampleRate);
        voice.filterEnv.setSampleRate (sampleRate);
    }

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
            voice.ampEnv.noteOn();
            voice.filterEnv.noteOn();

            heldNotes.addIfNotAlreadyThere (voice.midiNote);
        }
        else if (msg.isNoteOff())
        {
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

float AdvancedVSTiAudioProcessor::lfoValue (int index, float phase) const
{
    const auto shape = static_cast<int> (*apvts.getRawParameterValue (index == 1 ? "LFO1SHAPE" : "LFO2SHAPE"));
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

float AdvancedVSTiAudioProcessor::renderVoiceSample (VoiceState& voice)
{
    if (! voice.active)
        return 0.0f;

    const auto oscType = static_cast<OscType> (static_cast<int> (*apvts.getRawParameterValue ("OSCTYPE")));
    const auto unisonVoices = juce::jmax (1, static_cast<int> (*apvts.getRawParameterValue ("UNISON")));
    const auto detune = *apvts.getRawParameterValue ("DETUNE");
    const auto fmAmount = *apvts.getRawParameterValue ("FMAMOUNT");
    const auto syncAmount = *apvts.getRawParameterValue ("SYNC");
    const auto gateLength = *apvts.getRawParameterValue ("OSCGATE");

    auto baseHz = midiToHz (voice.midiNote);
    const auto lfo1Amt = *apvts.getRawParameterValue ("LFO1PITCH");
    baseHz *= std::pow (2.0f, (lfoValue (1, lfo1Phase) * lfo1Amt) / 12.0f);

    const auto fm = fmOperator (voice, baseHz, fmAmount);

    float s = 0.0f;
    for (int i = 0; i < unisonVoices; ++i)
    {
        const auto spread = (static_cast<float> (i) - (unisonVoices - 1) * 0.5f) * detune;
        s += oscSample (voice, baseHz * std::pow (2.0f, spread / 12.0f) + fm, oscType, syncAmount);
    }
    s /= static_cast<float> (unisonVoices);

    const auto envCurve = *apvts.getRawParameterValue ("ENVCURVE");
    const auto ampEnv = shapedEnv (voice.ampEnv.getNextSample(), envCurve);
    const auto filtEnv = shapedEnv (voice.filterEnv.getNextSample(), envCurve);

    voice.noteAge += 1.0f / static_cast<float> (currentSampleRate);
    const auto gatePass = voice.noteAge < gateLength ? 1.0f : 0.0f;

    const auto rgRate = *apvts.getRawParameterValue ("RHYTHMGATE_RATE");
    const auto rgDepth = *apvts.getRawParameterValue ("RHYTHMGATE_DEPTH");
    const auto rg = 0.5f * (1.0f + std::sin (twoPi * rgRate * voice.noteAge));
    const auto rhythmGate = 1.0f - rgDepth + rgDepth * rg;

    auto cutoff = *apvts.getRawParameterValue ("CUTOFF");
    cutoff += filtEnv * *apvts.getRawParameterValue ("FILTERENVAMOUNT") * 10000.0f;
    cutoff += lfoValue (2, lfo2Phase) * *apvts.getRawParameterValue ("LFO2FILTER") * 5000.0f;

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

    const auto mode = static_cast<int> (*apvts.getRawParameterValue ("ARPMODE"));
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

    const auto mode = static_cast<int> (*apvts.getRawParameterValue ("ARPMODE"));
    if (mode == 1)
        return heldNotes[heldNotes.size() - 1 - juce::jlimit (0, heldNotes.size() - 1, arpStep)];

    return heldNotes[juce::jlimit (0, heldNotes.size() - 1, arpStep)];
}

void AdvancedVSTiAudioProcessor::processBlock (juce::AudioBuffer<float>& buffer, juce::MidiBuffer& midiMessages)
{
    juce::ScopedNoDenormals noDenormals;
    buffer.clear();

    applyEnvelopeSettings();
    handleMidi (midiMessages, buffer.getNumSamples());

    const auto resonance = *apvts.getRawParameterValue ("RESONANCE");
    leftFilter.setResonance (resonance);
    rightFilter.setResonance (resonance);

    const auto filterType = static_cast<int> (*apvts.getRawParameterValue ("FILTERTYPE"));
    leftFilter.setType (static_cast<juce::dsp::StateVariableTPTFilterType> (filterType));
    rightFilter.setType (static_cast<juce::dsp::StateVariableTPTFilterType> (filterType));

    const auto lfo1Rate = *apvts.getRawParameterValue ("LFO1RATE");
    const auto lfo2Rate = *apvts.getRawParameterValue ("LFO2RATE");

    const auto arpRate = *apvts.getRawParameterValue ("ARPRATE");
    int arpCounter = 0;
    const auto arpIntervalSamples = juce::jmax (1, static_cast<int> (currentSampleRate / juce::jmax (0.25f, arpRate)));

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

        auto filteredL = leftFilter.processSample (sum);
        auto filteredR = rightFilter.processSample (sum);

        buffer.setSample (0, sample, filteredL);
        buffer.setSample (1, sample, filteredR);

        lfo1Phase = std::fmod (lfo1Phase + lfo1Rate / static_cast<float> (currentSampleRate), 1.0f);
        lfo2Phase = std::fmod (lfo2Phase + lfo2Rate / static_cast<float> (currentSampleRate), 1.0f);
    }
}

void AdvancedVSTiAudioProcessor::applyEnvelopeSettings()
{
    juce::ADSR::Parameters amp;
    amp.attack = *apvts.getRawParameterValue ("AMPATTACK");
    amp.decay = *apvts.getRawParameterValue ("AMPDECAY");
    amp.sustain = *apvts.getRawParameterValue ("AMPSUSTAIN");
    amp.release = *apvts.getRawParameterValue ("AMPRELEASE");

    juce::ADSR::Parameters filt;
    filt.attack = *apvts.getRawParameterValue ("FILTATTACK");
    filt.decay = *apvts.getRawParameterValue ("FILTDECAY");
    filt.sustain = *apvts.getRawParameterValue ("FILTSUSTAIN");
    filt.release = *apvts.getRawParameterValue ("FILTRELEASE");

    for (auto& voice : voices)
    {
        voice.ampEnv.setParameters (amp);
        voice.filterEnv.setParameters (filt);
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

    params.push_back (std::make_unique<juce::AudioParameterChoice> ("OSCTYPE", "Osc Type", juce::StringArray { "Sine", "Saw", "Square", "Noise", "Sample" }, 0));
    params.push_back (std::make_unique<juce::AudioParameterInt> ("UNISON", "Unison", 1, 8, 1));
    params.push_back (std::make_unique<juce::AudioParameterFloat> ("DETUNE", "Detune", 0.0f, 1.0f, 0.1f));
    params.push_back (std::make_unique<juce::AudioParameterFloat> ("FMAMOUNT", "FM Amount", 0.0f, 1000.0f, 0.0f));
    params.push_back (std::make_unique<juce::AudioParameterFloat> ("SYNC", "Sync", 0.0f, 4.0f, 0.0f));
    params.push_back (std::make_unique<juce::AudioParameterFloat> ("OSCGATE", "Osc Note Length Gate", 0.01f, 8.0f, 8.0f));

    params.push_back (std::make_unique<juce::AudioParameterFloat> ("AMPATTACK", "Amp Attack", 0.001f, 10.0f, 0.01f));
    params.push_back (std::make_unique<juce::AudioParameterFloat> ("AMPDECAY", "Amp Decay", 0.001f, 10.0f, 0.2f));
    params.push_back (std::make_unique<juce::AudioParameterFloat> ("AMPSUSTAIN", "Amp Sustain", 0.0f, 1.0f, 0.8f));
    params.push_back (std::make_unique<juce::AudioParameterFloat> ("AMPRELEASE", "Amp Release", 0.001f, 10.0f, 0.4f));

    params.push_back (std::make_unique<juce::AudioParameterFloat> ("FILTATTACK", "Filter Attack", 0.001f, 10.0f, 0.02f));
    params.push_back (std::make_unique<juce::AudioParameterFloat> ("FILTDECAY", "Filter Decay", 0.001f, 10.0f, 0.3f));
    params.push_back (std::make_unique<juce::AudioParameterFloat> ("FILTSUSTAIN", "Filter Sustain", 0.0f, 1.0f, 0.6f));
    params.push_back (std::make_unique<juce::AudioParameterFloat> ("FILTRELEASE", "Filter Release", 0.001f, 10.0f, 0.4f));
    params.push_back (std::make_unique<juce::AudioParameterFloat> ("ENVCURVE", "Envelope Curve", -1.0f, 1.0f, 0.0f));

    params.push_back (std::make_unique<juce::AudioParameterChoice> ("FILTERTYPE", "Filter Type", juce::StringArray { "LP", "BP", "HP", "Notch" }, 0));
    params.push_back (std::make_unique<juce::AudioParameterFloat> ("CUTOFF", "Cutoff", 20.0f, 20000.0f, 1200.0f));
    params.push_back (std::make_unique<juce::AudioParameterFloat> ("RESONANCE", "Resonance", 0.1f, 1.2f, 0.4f));
    params.push_back (std::make_unique<juce::AudioParameterFloat> ("FILTERENVAMOUNT", "Filter Env Amount", 0.0f, 1.0f, 0.5f));

    params.push_back (std::make_unique<juce::AudioParameterFloat> ("LFO1RATE", "LFO1 Rate", 0.05f, 40.0f, 2.0f));
    params.push_back (std::make_unique<juce::AudioParameterChoice> ("LFO1SHAPE", "LFO1 Shape", juce::StringArray { "Sine", "Saw", "Square" }, 0));
    params.push_back (std::make_unique<juce::AudioParameterFloat> ("LFO1PITCH", "LFO1 -> Pitch", 0.0f, 24.0f, 0.0f));

    params.push_back (std::make_unique<juce::AudioParameterFloat> ("LFO2RATE", "LFO2 Rate", 0.05f, 40.0f, 3.0f));
    params.push_back (std::make_unique<juce::AudioParameterChoice> ("LFO2SHAPE", "LFO2 Shape", juce::StringArray { "Sine", "Saw", "Square" }, 0));
    params.push_back (std::make_unique<juce::AudioParameterFloat> ("LFO2FILTER", "LFO2 -> Filter", 0.0f, 1.0f, 0.0f));

    params.push_back (std::make_unique<juce::AudioParameterChoice> ("ARPMODE", "Arp Mode", juce::StringArray { "Up", "Down", "UpDown", "Random" }, 0));
    params.push_back (std::make_unique<juce::AudioParameterFloat> ("ARPRATE", "Arp Rate", 0.25f, 16.0f, 4.0f));

    params.push_back (std::make_unique<juce::AudioParameterFloat> ("RHYTHMGATE_RATE", "Rhythm Gate Rate", 0.25f, 32.0f, 8.0f));
    params.push_back (std::make_unique<juce::AudioParameterFloat> ("RHYTHMGATE_DEPTH", "Rhythm Gate Depth", 0.0f, 1.0f, 0.0f));

    return { params.begin(), params.end() };
}

juce::AudioProcessor* JUCE_CALLTYPE createPluginFilter()
{
    return new AdvancedVSTiAudioProcessor();
}
