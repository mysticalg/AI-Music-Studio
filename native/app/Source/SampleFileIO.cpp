#include "SampleFileIO.h"

#include <juce_audio_formats/juce_audio_formats.h>
#include <cmath>
#include <cstdint>

namespace aims
{
namespace
{
constexpr int kWaveformPreviewPoints = 256;
constexpr int kReadBufferFrames = 8192;

juce::AudioBuffer<float> ensureStereoBuffer(const juce::AudioBuffer<float>& source)
{
    juce::AudioBuffer<float> stereo(2, juce::jmax(1, source.getNumSamples()));
    stereo.clear();

    if (source.getNumChannels() <= 0)
        return stereo;

    stereo.copyFrom(0, 0, source, 0, 0, source.getNumSamples());
    stereo.copyFrom(1, 0, source, source.getNumChannels() > 1 ? 1 : 0, 0, source.getNumSamples());
    return stereo;
}

juce::AudioBuffer<float> resampleBuffer(const juce::AudioBuffer<float>& input,
                                        double sourceSampleRate,
                                        double targetSampleRate)
{
    if (input.getNumSamples() <= 0)
        return ensureStereoBuffer(input);

    if (sourceSampleRate <= 0.0
        || targetSampleRate <= 0.0
        || std::abs(sourceSampleRate - targetSampleRate) < 0.5)
    {
        return ensureStereoBuffer(input);
    }

    const auto ratio = targetSampleRate / sourceSampleRate;
    const auto targetSamples = juce::jmax(1, static_cast<int>(std::llround(input.getNumSamples() * ratio)));
    juce::AudioBuffer<float> resampled(input.getNumChannels(), targetSamples);
    resampled.clear();

    for (int channel = 0; channel < input.getNumChannels(); ++channel)
    {
        const auto* source = input.getReadPointer(channel);
        auto* destination = resampled.getWritePointer(channel);
        for (int sample = 0; sample < targetSamples; ++sample)
        {
            const auto sourcePosition = (static_cast<double>(sample) * sourceSampleRate) / targetSampleRate;
            const auto lowerIndex = juce::jlimit(0,
                                                 juce::jmax(0, input.getNumSamples() - 1),
                                                 static_cast<int>(std::floor(sourcePosition)));
            const auto upperIndex = juce::jlimit(0,
                                                 juce::jmax(0, input.getNumSamples() - 1),
                                                 lowerIndex + 1);
            const auto fraction = static_cast<float>(sourcePosition - static_cast<double>(lowerIndex));
            destination[sample] = source[lowerIndex] + ((source[upperIndex] - source[lowerIndex]) * fraction);
        }
    }

    return ensureStereoBuffer(resampled);
}
}

juce::Result loadSampleAssetFromFile(const juce::File& file, SampleAsset& outAsset)
{
    if (!file.existsAsFile())
        return juce::Result::fail("Sample file not found.");

    juce::AudioFormatManager formatManager;
    formatManager.registerBasicFormats();

    std::unique_ptr<juce::AudioFormatReader> reader(formatManager.createReaderFor(file));
    if (reader == nullptr)
        return juce::Result::fail("Could not open the sample file.");

    outAsset = {};
    outAsset.path = file.getFullPathName();
    outAsset.sampleRate = juce::jmax(1, juce::roundToInt(reader->sampleRate));
    outAsset.durationSec = reader->lengthInSamples / juce::jmax(1.0, reader->sampleRate);
    outAsset.waveformPreview.clear();
    outAsset.waveformPreview.reserve(kWaveformPreviewPoints);

    const auto totalSamples = juce::jmax<int64_t>(1, static_cast<int64_t>(reader->lengthInSamples));
    const auto channelCount = juce::jmax(1, static_cast<int>(reader->numChannels));
    juce::AudioBuffer<float> buffer(channelCount, kReadBufferFrames);

    for (int pointIndex = 0; pointIndex < kWaveformPreviewPoints; ++pointIndex)
    {
        const auto startSample = (totalSamples * static_cast<int64_t>(pointIndex)) / kWaveformPreviewPoints;
        auto endSample = (totalSamples * static_cast<int64_t>(pointIndex + 1)) / kWaveformPreviewPoints;
        if (endSample <= startSample)
            endSample = juce::jmin(totalSamples, startSample + 1);

        float peak = 0.0f;
        auto cursor = startSample;
        while (cursor < endSample)
        {
            const auto framesToRead = static_cast<int>(juce::jmin<int64_t>(kReadBufferFrames, endSample - cursor));
            buffer.clear();
            if (!reader->read(&buffer, 0, framesToRead, cursor, true, true))
                break;

            for (int channel = 0; channel < channelCount; ++channel)
            {
                const auto* channelData = buffer.getReadPointer(channel);
                for (int sample = 0; sample < framesToRead; ++sample)
                    peak = juce::jmax(peak, std::abs(channelData[sample]));
            }

            cursor += framesToRead;
        }

        outAsset.waveformPreview.push_back(juce::jlimit(0.0f, 1.0f, peak));
    }

    return juce::Result::ok();
}

juce::Result loadAudioFileToStereoBuffer(const juce::File& file,
                                         juce::AudioBuffer<float>& outBuffer,
                                         double targetSampleRate)
{
    if (!file.existsAsFile())
        return juce::Result::fail("Audio file not found.");

    juce::AudioFormatManager formatManager;
    formatManager.registerBasicFormats();

    std::unique_ptr<juce::AudioFormatReader> reader(formatManager.createReaderFor(file));
    if (reader == nullptr)
        return juce::Result::fail("Could not open the audio file.");

    const auto sourceChannels = juce::jmax(1, static_cast<int>(reader->numChannels));
    const auto sourceSamples = juce::jmax(1, static_cast<int>(reader->lengthInSamples));
    juce::AudioBuffer<float> source(sourceChannels, sourceSamples);
    source.clear();
    if (!reader->read(&source, 0, sourceSamples, 0, true, true))
        return juce::Result::fail("Could not read the audio file.");

    outBuffer = resampleBuffer(source, reader->sampleRate, targetSampleRate);
    return juce::Result::ok();
}

} // namespace aims
