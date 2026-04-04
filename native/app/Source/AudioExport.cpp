#include "AudioExport.h"

#include "SampleFileIO.h"

#include <juce_audio_formats/juce_audio_formats.h>

#include <algorithm>
#include <cmath>
#include <map>
#include <memory>
#include <unordered_map>

namespace aims
{
namespace
{
constexpr int kOfflineRenderChunkFrames = 2048;
constexpr int kNoteOffAdvanceFrameLimit = 64;

struct ClipPlacement
{
    juce::String path;
    double startSec = 0.0;
};

struct RenderMidiEvent
{
    int sampleOffset = 0;
    int priority = 0;
    juce::String type;
    int channel = 1;
    int note = 60;
    int controller = 0;
    int controllerValue = 0;
    double velocity = 0.0;
};

bool fxSlotBypassed(const std::vector<bool>& bypassStates, int index)
{
    return index >= 0
        && index < static_cast<int>(bypassStates.size())
        && bypassStates[static_cast<size_t>(index)];
}

bool trackIsAudible(const ProjectState& project, int trackIndex)
{
    bool anySolo = false;
    for (const auto& track : project.tracks)
    {
        if (track.solo)
        {
            anySolo = true;
            break;
        }
    }

    if (!juce::isPositiveAndBelow(trackIndex, static_cast<int>(project.tracks.size())))
        return false;

    const auto& track = project.tracks[static_cast<size_t>(trackIndex)];
    if (track.mute)
        return false;
    if (anySolo && !track.solo)
        return false;
    return true;
}

bool trackHasEnabledRoutes(const juce::Array<juce::var>& routes)
{
    for (const auto& routeVar : routes)
    {
        auto* routeObject = routeVar.getDynamicObject();
        if (routeObject == nullptr)
            continue;
        if (static_cast<bool>(routeObject->getProperty("enabled")))
            return true;
    }
    return false;
}

int64_t tickToFrame(int tick, int bpm, double sampleRate)
{
    return static_cast<int64_t>(
        (static_cast<double>(tick) * sampleRate * 60.0)
        / (static_cast<double>(juce::jmax(1, bpm)) * static_cast<double>(kTicksPerBeat)));
}

double frameToTickDouble(int64_t frame, int bpm, double sampleRate)
{
    return (
        (static_cast<double>(frame) * static_cast<double>(juce::jmax(1, bpm)) * static_cast<double>(kTicksPerBeat))
        / (sampleRate * 60.0)
    );
}

int64_t fileMtimeNs(const juce::File& file)
{
    if (!file.existsAsFile())
        return 0;
    return static_cast<int64_t>(file.getLastModificationTime().toMilliseconds()) * 1000000LL;
}

juce::String trackDisplayName(const TrackState& track, int trackIndex)
{
    const auto trimmedName = track.name.trim();
    return trimmedName.isNotEmpty() ? trimmedName : "Track " + juce::String(trackIndex + 1);
}

juce::String describeTrackWarningsPrefix(const TrackState& track, int trackIndex)
{
    return trackDisplayName(track, trackIndex) + ": ";
}

juce::String makeStemFileName(const TrackState& track, int trackIndex)
{
    const auto baseName = juce::File::createLegalFileName(trackDisplayName(track, trackIndex).replaceCharacter(' ', '_'));
    return "track_" + juce::String(trackIndex + 1).paddedLeft('0', 2)
        + "_" + (baseName.isNotEmpty() ? baseName : "Track")
        + ".wav";
}

juce::var buildParameterPayload(const TrackState& track)
{
    auto* params = new juce::DynamicObject();
    for (int index = 0; index < track.vstiParameters.size(); ++index)
        params->setProperty(track.vstiParameters.getName(index), track.vstiParameters.getValueAt(index));
    return juce::var(params);
}

const VstInstrument* findRackEntryByReference(const ProjectState& project,
                                              const juce::String& reference,
                                              bool wantEffect)
{
    const auto trimmed = reference.trim();
    if (trimmed.isEmpty())
        return nullptr;

    const auto normalizedPath = juce::File(trimmed).getFullPathName();
    for (const auto& entry : project.vstRack)
    {
        if (!entry.hostSupported)
            continue;
        if (wantEffect && !entry.isEffect)
            continue;
        if (!wantEffect && !entry.isInstrument)
            continue;

        if (entry.name.equalsIgnoreCase(trimmed)
            || entry.pluginName.equalsIgnoreCase(trimmed)
            || entry.path.equalsIgnoreCase(trimmed)
            || (!normalizedPath.isEmpty() && juce::File(entry.path).getFullPathName().equalsIgnoreCase(normalizedPath)))
        {
            return &entry;
        }
    }

    return nullptr;
}

juce::Array<juce::var> buildFxChainPayload(const ProjectState& project,
                                           const TrackState& track,
                                           int trackIndex,
                                           juce::StringArray& warnings)
{
    juce::Array<juce::var> payload;

    if (track.vstFxBypassed)
        return payload;

    for (int fxIndex = 0; fxIndex < track.vstFxChain.size(); ++fxIndex)
    {
        const auto& fxName = track.vstFxChain[fxIndex];
        const auto* entry = findRackEntryByReference(project, fxName, true);
        if (entry == nullptr || entry->path.isEmpty() || !juce::File(entry->path).exists())
        {
            warnings.addIfNotAlreadyThere(describeTrackWarningsPrefix(track, trackIndex)
                                          + "skipped unresolved FX slot \"" + fxName + "\" during native export.");
            continue;
        }

        auto* fxObject = new juce::DynamicObject();
        fxObject->setProperty("name", entry->name.isNotEmpty() ? entry->name : fxName);
        fxObject->setProperty("plugin_path", entry->path);
        fxObject->setProperty("state_path", "");
        fxObject->setProperty("parameters", juce::var(new juce::DynamicObject()));
        fxObject->setProperty("bypassed", fxSlotBypassed(track.vstFxSlotBypassed, fxIndex));
        payload.add(juce::var(fxObject));
    }

    return payload;
}

std::vector<ClipPlacement> collectTrackClipPlacements(const ProjectState& project,
                                                      int trackIndex,
                                                      bool includeRenderedFallback)
{
    std::vector<ClipPlacement> placements;

    for (const auto& clip : project.sampleClips)
    {
        if (clip.trackIndex != trackIndex)
            continue;
        if (clip.path.trim().isEmpty())
            continue;

        ClipPlacement placement;
        placement.path = clip.path.trim();
        placement.startSec = juce::jmax(0.0, clip.startSec);
        placements.push_back(std::move(placement));
    }

    if (includeRenderedFallback
        && juce::isPositiveAndBelow(trackIndex, static_cast<int>(project.tracks.size())))
    {
        const auto& track = project.tracks[static_cast<size_t>(trackIndex)];
        if (track.renderedAudioPath.isNotEmpty() && juce::File(track.renderedAudioPath).existsAsFile())
        {
            ClipPlacement placement;
            placement.path = track.renderedAudioPath;
            placement.startSec = 0.0;
            placements.push_back(std::move(placement));
        }
    }

    return placements;
}

const AutomationLane* findEnabledLane(const TrackState& track, const juce::String& target)
{
    for (const auto& lane : track.automationLanes)
    {
        if (!lane.enabled || lane.points.empty())
            continue;
        if (lane.target.equalsIgnoreCase(target))
            return &lane;
    }
    return nullptr;
}

float evaluateAutomationLane(const AutomationLane& lane, double tick, float defaultValue)
{
    if (lane.points.empty())
        return defaultValue;

    if (lane.points.size() == 1)
    {
        const auto& point = lane.points.front();
        return tick >= static_cast<double>(point.tick) ? static_cast<float>(point.value) : defaultValue;
    }

    auto previousTick = 0.0;
    auto previousValue = static_cast<double>(defaultValue);
    for (const auto& point : lane.points)
    {
        const auto pointTick = static_cast<double>(point.tick);
        const auto pointValue = static_cast<double>(point.value);
        if (pointTick <= 0.0)
        {
            previousTick = pointTick;
            previousValue = pointValue;
            if (tick <= 0.0)
                return static_cast<float>(point.value);
            continue;
        }

        if (tick < pointTick)
        {
            if (previousTick <= 0.0)
                return static_cast<float>(previousValue);

            const auto span = juce::jmax(1.0, pointTick - previousTick);
            const auto progress = juce::jlimit(0.0, 1.0, (tick - previousTick) / span);
            return static_cast<float>(previousValue + ((pointValue - previousValue) * progress));
        }

        previousTick = pointTick;
        previousValue = pointValue;
    }

    return static_cast<float>(previousValue);
}

void applyOutputMixToStereoPair(float& leftSample, float& rightSample, float gain, float pan)
{
    gain = juce::jmax(0.0f, gain);
    pan = juce::jlimit(-1.0f, 1.0f, pan);
    const auto angle = (pan + 1.0f) * juce::MathConstants<float>::pi * 0.25f;
    const auto leftGain = std::cos(angle) * gain;
    const auto rightGain = std::sin(angle) * gain;
    leftSample *= leftGain;
    rightSample *= rightGain;
}

void applyTrackMixAutomationToBuffer(const ProjectState& project,
                                     const TrackState& track,
                                     juce::AudioBuffer<float>& buffer,
                                     int64_t absoluteStartFrame,
                                     double sampleRate)
{
    jassert(buffer.getNumChannels() >= 2);

    const auto* volumeLane = findEnabledLane(track, kAutomationTargetVolume);
    const auto* panLane = findEnabledLane(track, kAutomationTargetPan);
    const auto* outputGainLane = findEnabledLane(track, kAutomationTargetVstOutputGainDb);

    if (volumeLane == nullptr && panLane == nullptr && outputGainLane == nullptr)
    {
        const auto gain = juce::jmax(0.0f,
                                     static_cast<float>(track.volume)
                                         * std::pow(10.0f, static_cast<float>(track.vstiOutputGainDb) / 20.0f));
        const auto pan = static_cast<float>(juce::jlimit(-1.0, 1.0, track.pan));
        auto* left = buffer.getWritePointer(0);
        auto* right = buffer.getWritePointer(1);
        for (int sample = 0; sample < buffer.getNumSamples(); ++sample)
            applyOutputMixToStereoPair(left[sample], right[sample], gain, pan);
        return;
    }

    auto* left = buffer.getWritePointer(0);
    auto* right = buffer.getWritePointer(1);
    for (int sample = 0; sample < buffer.getNumSamples(); ++sample)
    {
        const auto tick = frameToTickDouble(absoluteStartFrame + static_cast<int64_t>(sample),
                                            project.bpm,
                                            sampleRate);
        const auto volume = volumeLane != nullptr
            ? evaluateAutomationLane(*volumeLane, tick, static_cast<float>(track.volume))
            : static_cast<float>(track.volume);
        const auto pan = panLane != nullptr
            ? evaluateAutomationLane(*panLane, tick, static_cast<float>(track.pan))
            : static_cast<float>(track.pan);
        const auto outputGainDb = outputGainLane != nullptr
            ? evaluateAutomationLane(*outputGainLane, tick, static_cast<float>(track.vstiOutputGainDb))
            : static_cast<float>(track.vstiOutputGainDb);
        const auto gain = juce::jmax(0.0f, volume * std::pow(10.0f, outputGainDb / 20.0f));
        applyOutputMixToStereoPair(left[sample], right[sample], gain, pan);
    }
}

bool trackHasUnsupportedParameterAutomation(const TrackState& track)
{
    for (const auto& lane : track.automationLanes)
    {
        if (lane.enabled && !lane.points.empty() && automationTargetIsVstParameter(lane.target))
            return true;
    }
    return false;
}

juce::Result decodeRenderedStereoBuffer(const juce::var& response,
                                        int expectedFrames,
                                        juce::AudioBuffer<float>& outBuffer)
{
    auto* responseObject = response.getDynamicObject();
    if (responseObject == nullptr)
        return juce::Result::fail("Native export received an invalid render response.");

    const auto audioBase64 = responseObject->getProperty("audio_b64").toString().trim();
    if (audioBase64.isEmpty())
        return juce::Result::fail("Native export render response did not contain audio.");

    const auto providedChannels = juce::jmax(1, static_cast<int>(responseObject->getProperty("channels")));
    juce::MemoryOutputStream decoded;
    if (!juce::Base64::convertFromBase64(decoded, audioBase64))
        return juce::Result::fail("Native export could not decode rendered audio.");

    const auto totalFloatCount = static_cast<int>(decoded.getDataSize() / sizeof(float));
    const auto availableFrames = providedChannels > 0 ? totalFloatCount / providedChannels : 0;
    if (availableFrames <= 0)
        return juce::Result::fail("Native export render response was empty.");

    const auto frameCount = juce::jmax(1, juce::jmin(expectedFrames, availableFrames));
    outBuffer.setSize(2, frameCount, false, false, true);
    outBuffer.clear();

    const auto* samples = static_cast<const float*>(decoded.getData());
    auto* left = outBuffer.getWritePointer(0);
    auto* right = outBuffer.getWritePointer(1);
    for (int frame = 0; frame < frameCount; ++frame)
    {
        left[frame] = samples[(frame * providedChannels)];
        right[frame] = samples[(frame * providedChannels) + (providedChannels > 1 ? 1 : 0)];
    }

    return juce::Result::ok();
}

juce::String formatTrackErrors(const juce::var& response)
{
    juce::StringArray messages;

    auto* responseObject = response.getDynamicObject();
    if (responseObject == nullptr)
        return {};

    auto* errorsArray = responseObject->getProperty("track_errors").getArray();
    if (errorsArray == nullptr)
        return {};

    for (const auto& errorVar : *errorsArray)
    {
        auto* errorObject = errorVar.getDynamicObject();
        if (errorObject == nullptr)
            continue;

        auto message = errorObject->getProperty("message").toString().trim();
        if (message.isEmpty())
            message = "Unknown host render error";
        messages.add(message);
    }

    return messages.joinIntoString("; ");
}

juce::Array<juce::var> buildChunkEventPayload(const ProjectState& project,
                                              int trackIndex,
                                              const TrackState& track,
                                              int64_t absoluteStartFrame,
                                              int frameCount,
                                              double sampleRate,
                                              bool bootstrapActive)
{
    std::vector<RenderMidiEvent> events;
    events.reserve(track.notes.size() * 2);
    const auto absoluteEndFrame = absoluteStartFrame + static_cast<int64_t>(frameCount);
    const auto noteOffAdvanceFrames = juce::jmax(1,
                                                 juce::jmin(kNoteOffAdvanceFrameLimit,
                                                            static_cast<int>(sampleRate / 1000.0)));

    for (const auto& note : track.notes)
    {
        const auto noteStartFrame = tickToFrame(note.startTick, project.bpm, sampleRate);
        const auto noteEndFrame = juce::jmax(noteStartFrame + 1,
                                             tickToFrame(note.startTick + note.durationTick, project.bpm, sampleRate));
        const auto noteOffFrame = juce::jmax(noteStartFrame, noteEndFrame - noteOffAdvanceFrames);

        if (noteEndFrame <= absoluteStartFrame || noteStartFrame >= absoluteEndFrame)
            continue;

        if (bootstrapActive && noteStartFrame < absoluteStartFrame && absoluteStartFrame < noteEndFrame)
        {
            RenderMidiEvent event;
            event.sampleOffset = 0;
            event.priority = 2;
            event.type = "note_on";
            event.channel = juce::jlimit(1, 16, track.midiChannel + 1);
            event.note = juce::jlimit(0, 127, note.pitch);
            event.velocity = juce::jlimit(0.0, 1.0, static_cast<double>(juce::jlimit(1, 127, note.velocity)) / 127.0);
            events.push_back(event);
        }
        else if (absoluteStartFrame <= noteStartFrame && noteStartFrame < absoluteEndFrame)
        {
            RenderMidiEvent event;
            event.sampleOffset = static_cast<int>(noteStartFrame - absoluteStartFrame);
            event.priority = 2;
            event.type = "note_on";
            event.channel = juce::jlimit(1, 16, track.midiChannel + 1);
            event.note = juce::jlimit(0, 127, note.pitch);
            event.velocity = juce::jlimit(0.0, 1.0, static_cast<double>(juce::jlimit(1, 127, note.velocity)) / 127.0);
            events.push_back(event);
        }

        if (absoluteStartFrame <= noteOffFrame && noteOffFrame < absoluteEndFrame)
        {
            RenderMidiEvent event;
            event.sampleOffset = static_cast<int>(noteOffFrame - absoluteStartFrame);
            event.priority = 1;
            event.type = "note_off";
            event.channel = juce::jlimit(1, 16, track.midiChannel + 1);
            event.note = juce::jlimit(0, 127, note.pitch);
            event.velocity = 0.0;
            events.push_back(event);
        }
    }

    const auto controllerEvents = collectTrackControllerEvents(project, trackIndex);
    std::map<std::pair<int, int>, MidiControllerEvent> bootstrapControllers;
    for (const auto& controllerEvent : controllerEvents)
    {
        const auto eventFrame = tickToFrame(controllerEvent.tick, project.bpm, sampleRate);
        if (bootstrapActive && eventFrame < absoluteStartFrame)
        {
            bootstrapControllers[{ controllerEvent.channel, controllerEvent.controller }] = controllerEvent;
            continue;
        }

        if (eventFrame < absoluteStartFrame || eventFrame >= absoluteEndFrame)
            continue;

        RenderMidiEvent event;
        event.sampleOffset = static_cast<int>(eventFrame - absoluteStartFrame);
        event.priority = 0;
        event.type = "control_change";
        event.channel = juce::jlimit(1, 16, controllerEvent.channel);
        event.controller = juce::jlimit(0, 127, controllerEvent.controller);
        event.controllerValue = juce::jlimit(0, 127, controllerEvent.value);
        events.push_back(event);
    }

    if (bootstrapActive)
    {
        for (const auto& entry : bootstrapControllers)
        {
            const auto& controllerEvent = entry.second;
            RenderMidiEvent event;
            event.sampleOffset = 0;
            event.priority = 0;
            event.type = "control_change";
            event.channel = juce::jlimit(1, 16, controllerEvent.channel);
            event.controller = juce::jlimit(0, 127, controllerEvent.controller);
            event.controllerValue = juce::jlimit(0, 127, controllerEvent.value);
            events.push_back(event);
        }
    }

    std::sort(events.begin(), events.end(), [] (const RenderMidiEvent& lhs, const RenderMidiEvent& rhs)
    {
        if (lhs.sampleOffset != rhs.sampleOffset)
            return lhs.sampleOffset < rhs.sampleOffset;
        if (lhs.priority != rhs.priority)
            return lhs.priority < rhs.priority;
        if (lhs.channel != rhs.channel)
            return lhs.channel < rhs.channel;
        return lhs.note < rhs.note;
    });

    juce::Array<juce::var> payload;
    payload.ensureStorageAllocated(static_cast<int>(events.size()));
    for (const auto& event : events)
    {
        auto* eventObject = new juce::DynamicObject();
        eventObject->setProperty("sample_offset", event.sampleOffset);
        eventObject->setProperty("priority", event.priority);
        eventObject->setProperty("type", event.type);
        eventObject->setProperty("channel", event.channel);
        eventObject->setProperty("note", event.note);
        eventObject->setProperty("control", event.controller);
        eventObject->setProperty("value", event.controllerValue);
        eventObject->setProperty("velocity", event.velocity);
        payload.add(juce::var(eventObject));
    }

    return payload;
}

juce::Result renderNativeTrackToMix(const ProjectState& project,
                                    int trackIndex,
                                    const TrackState& track,
                                    const juce::String& pluginPath,
                                    int64_t exportStartFrame,
                                    int totalFrames,
                                    int sampleRate,
                                    NativeVstHostSession& nativeVstHost,
                                    juce::AudioBuffer<float>& finalMix,
                                    AudioExportSummary& summary)
{
    auto result = nativeVstHost.ensureCreated();
    if (result.failed())
        return result;

    result = nativeVstHost.clearPluginGraph();
    if (result.failed())
        return result;

    if (trackHasUnsupportedParameterAutomation(track))
    {
        summary.warnings.addIfNotAlreadyThere(describeTrackWarningsPrefix(track, trackIndex)
                                              + "native WAV export currently ignores VST parameter automation lanes.");
    }

    if (trackHasEnabledRoutes(track.vstiInputBusRoutes) || trackHasEnabledRoutes(track.vstiOutputBusRoutes))
    {
        summary.warnings.addIfNotAlreadyThere(describeTrackWarningsPrefix(track, trackIndex)
                                              + "native WAV export currently ignores custom VST bus routing.");
    }

    juce::Array<juce::var> fxChainPayload = buildFxChainPayload(project, track, trackIndex, summary.warnings);
    const auto stateFile = juce::File(track.vstiStatePath);
    const auto statePath = stateFile.existsAsFile() ? stateFile.getFullPathName() : juce::String();
    const auto stateMtimeNs = stateFile.existsAsFile() ? fileMtimeNs(stateFile) : int64_t(0);

    for (int chunkStart = 0; chunkStart < totalFrames; chunkStart += kOfflineRenderChunkFrames)
    {
        const auto chunkFrames = juce::jmin(kOfflineRenderChunkFrames, totalFrames - chunkStart);
        const auto absoluteChunkStartFrame = exportStartFrame + static_cast<int64_t>(chunkStart);

        auto* trackObject = new juce::DynamicObject();
        trackObject->setProperty("name", trackDisplayName(track, trackIndex));
        trackObject->setProperty("plugin_path", pluginPath);
        trackObject->setProperty("state_path", statePath);
        trackObject->setProperty("state_mtime_ns", static_cast<double>(stateMtimeNs));
        trackObject->setProperty("parameters", buildParameterPayload(track));
        trackObject->setProperty("gain", 1.0);
        trackObject->setProperty("pan", 0.0);
        trackObject->setProperty("events", juce::var(buildChunkEventPayload(project,
                                                                            trackIndex,
                                                                            track,
                                                                            absoluteChunkStartFrame,
                                                                            chunkFrames,
                                                                            sampleRate,
                                                                            chunkStart == 0)));
        if (!fxChainPayload.isEmpty())
            trackObject->setProperty("fx_chain", juce::var(fxChainPayload));

        juce::Array<juce::var> tracks;
        tracks.add(juce::var(trackObject));

        juce::var response;
        result = nativeVstHost.renderOfflinePluginGraph(juce::var(tracks), chunkFrames, response);
        if (result.failed())
            return result;

        const auto trackErrors = formatTrackErrors(response);
        if (trackErrors.isNotEmpty())
        {
            return juce::Result::fail("Native graph render failed for "
                                      + trackDisplayName(track, trackIndex)
                                      + ": "
                                      + trackErrors);
        }

        juce::AudioBuffer<float> renderedChunk;
        result = decodeRenderedStereoBuffer(response, chunkFrames, renderedChunk);
        if (result.failed())
            return result;

        applyTrackMixAutomationToBuffer(project, track, renderedChunk, absoluteChunkStartFrame, sampleRate);
        finalMix.addFrom(0, chunkStart, renderedChunk, 0, 0, renderedChunk.getNumSamples());
        finalMix.addFrom(1, chunkStart, renderedChunk, 1, 0, renderedChunk.getNumSamples());
    }

    nativeVstHost.clearPluginGraph();
    ++summary.renderedInstrumentTrackCount;
    return juce::Result::ok();
}

juce::Result writeStereoBufferToWavFile(const juce::File& file,
                                        const juce::AudioBuffer<float>& buffer,
                                        double sampleRate)
{
    if (!file.getParentDirectory().createDirectory())
        return juce::Result::fail("Could not create the export folder.");

    auto fileStream = file.createOutputStream();
    if (fileStream == nullptr)
        return juce::Result::fail("Could not create the export file.");
    std::unique_ptr<juce::OutputStream> stream(fileStream.release());

    juce::WavAudioFormat wavFormat;
    const auto options = juce::AudioFormatWriterOptions()
        .withSampleRate(sampleRate)
        .withNumChannels(2)
        .withBitsPerSample(24);
    auto writer = wavFormat.createWriterFor(stream, options);
    if (writer == nullptr)
        return juce::Result::fail("Could not create the WAV writer.");

    if (!writer->writeFromAudioSampleBuffer(buffer, 0, buffer.getNumSamples()))
        return juce::Result::fail("Could not write the WAV export.");

    return juce::Result::ok();
}

juce::Result writeStereoBufferToMp3File(const juce::File& file,
                                        const juce::AudioBuffer<float>& buffer,
                                        double sampleRate)
{
#if JUCE_USE_MP3AUDIOFORMAT
    if (!file.getParentDirectory().createDirectory())
        return juce::Result::fail("Could not create the export folder.");

    auto fileStream = file.createOutputStream();
    if (fileStream == nullptr)
        return juce::Result::fail("Could not create the export file.");
    std::unique_ptr<juce::OutputStream> stream(fileStream.release());

    juce::MP3AudioFormat mp3Format;
    const auto options = juce::AudioFormatWriterOptions()
        .withSampleRate(sampleRate)
        .withNumChannels(2)
        .withBitsPerSample(16);
    auto writer = mp3Format.createWriterFor(stream, options);
    if (writer == nullptr)
        return juce::Result::fail("Could not create the MP3 writer.");

    if (!writer->writeFromAudioSampleBuffer(buffer, 0, buffer.getNumSamples()))
        return juce::Result::fail("Could not write the MP3 export.");

    return juce::Result::ok();
#else
    juce::ignoreUnused(file, buffer, sampleRate);
    return juce::Result::fail("MP3 export is not enabled in this build.");
#endif
}

juce::Result writeStereoBufferToAudioFile(const juce::File& file,
                                          const juce::AudioBuffer<float>& buffer,
                                          double sampleRate)
{
    const auto extension = file.getFileExtension().trim().toLowerCase();
    if (extension == ".mp3")
        return writeStereoBufferToMp3File(file, buffer, sampleRate);

    if (extension == ".wav" || extension == ".bwf" || extension.isEmpty())
        return writeStereoBufferToWavFile(file, buffer, sampleRate);

    return juce::Result::fail("Unsupported export format: " + file.getFileExtension());
}

juce::Result renderTrackRangeToBuffer(const ProjectState& project,
                                      int trackIndex,
                                      NativeVstHostSession& nativeVstHost,
                                      juce::AudioBuffer<float>& outBuffer,
                                      AudioExportSummary& outSummary,
                                      int sampleRate)
{
    outSummary = {};
    outSummary.sampleRate = juce::jmax(1, sampleRate);

    if (!juce::isPositiveAndBelow(trackIndex, static_cast<int>(project.tracks.size())))
        return juce::Result::fail("Track index is out of range.");

    const auto& track = project.tracks[static_cast<size_t>(trackIndex)];
    const auto leftSec = juce::jmax(0.0, project.leftLocatorSec);
    const auto rightSec = juce::jmax(leftSec, project.rightLocatorSec);
    if (rightSec <= leftSec)
        return juce::Result::fail("Right locator must be greater than left locator for WAV export.");

    const auto exportStartFrame = static_cast<int64_t>(std::llround(leftSec * outSummary.sampleRate));
    const auto totalFrames = juce::jmax(1, static_cast<int>(std::llround((rightSec - leftSec) * outSummary.sampleRate)));
    outSummary.frameCount = totalFrames;

    outBuffer.setSize(2, totalFrames, false, false, true);
    outBuffer.clear();

    std::unordered_map<std::string, std::unique_ptr<juce::AudioBuffer<float>>> clipCache;
    bool anyRenderedSource = false;

    const auto pluginPath = resolveRackPluginPath(project, track);
    const auto canRenderNativeInstrument = track.trackType.trim().equalsIgnoreCase("instrument")
        && pluginPath.isNotEmpty()
        && juce::File(pluginPath).exists();
    bool renderFailed = false;

    if (canRenderNativeInstrument)
    {
        const auto result = renderNativeTrackToMix(project,
                                                   trackIndex,
                                                   track,
                                                   pluginPath,
                                                   exportStartFrame,
                                                   totalFrames,
                                                   outSummary.sampleRate,
                                                   nativeVstHost,
                                                   outBuffer,
                                                   outSummary);
        if (result.failed())
        {
            renderFailed = true;
            if (track.renderedAudioPath.isNotEmpty() && juce::File(track.renderedAudioPath).existsAsFile())
            {
                outSummary.warnings.addIfNotAlreadyThere(describeTrackWarningsPrefix(track, trackIndex)
                                                         + "native instrument render failed, so export fell back to the rendered audio stem.");
            }
            else
            {
                outSummary.warnings.addIfNotAlreadyThere(describeTrackWarningsPrefix(track, trackIndex)
                                                         + result.getErrorMessage());
            }
        }
        else
        {
            anyRenderedSource = true;
        }
    }

    const auto clipPlacements = collectTrackClipPlacements(project,
                                                           trackIndex,
                                                           !canRenderNativeInstrument || renderFailed);
    bool mixedTrackAudio = false;
    if (!clipPlacements.empty() && track.vstFxChain.size() > 0)
    {
        outSummary.warnings.addIfNotAlreadyThere(describeTrackWarningsPrefix(track, trackIndex)
                                                 + "native WAV export currently ignores audio-track FX-chain processing.");
    }

    for (const auto& placement : clipPlacements)
    {
        const juce::File clipFile(placement.path);
        if (!clipFile.existsAsFile())
        {
            outSummary.warnings.addIfNotAlreadyThere(describeTrackWarningsPrefix(track, trackIndex)
                                                     + "skipped missing audio clip " + placement.path + ".");
            continue;
        }

        const auto cacheKey = clipFile.getFullPathName().toStdString();
        auto cacheIt = clipCache.find(cacheKey);
        if (cacheIt == clipCache.end())
        {
            auto loadedBuffer = std::make_unique<juce::AudioBuffer<float>>();
            const auto loadResult = loadAudioFileToStereoBuffer(clipFile,
                                                                *loadedBuffer,
                                                                static_cast<double>(outSummary.sampleRate));
            if (loadResult.failed())
            {
                outSummary.warnings.addIfNotAlreadyThere(describeTrackWarningsPrefix(track, trackIndex)
                                                         + "could not read audio clip "
                                                         + clipFile.getFileName()
                                                         + ": "
                                                         + loadResult.getErrorMessage());
                continue;
            }

            cacheIt = clipCache.emplace(cacheKey, std::move(loadedBuffer)).first;
        }

        const auto& clipBuffer = *cacheIt->second;
        const auto clipStartFrame = static_cast<int64_t>(std::llround(placement.startSec * outSummary.sampleRate));
        const auto clipEndFrame = clipStartFrame + clipBuffer.getNumSamples();
        const auto exportEndFrame = exportStartFrame + static_cast<int64_t>(totalFrames);
        if (clipEndFrame <= exportStartFrame || clipStartFrame >= exportEndFrame)
            continue;

        const auto overlapStartFrame = juce::jmax(exportStartFrame, clipStartFrame);
        const auto overlapEndFrame = juce::jmin(exportEndFrame, clipEndFrame);
        const auto sourceStart = static_cast<int>(overlapStartFrame - clipStartFrame);
        const auto destinationStart = static_cast<int>(overlapStartFrame - exportStartFrame);
        const auto sampleCount = static_cast<int>(overlapEndFrame - overlapStartFrame);
        if (sampleCount <= 0)
            continue;

        juce::AudioBuffer<float> clipSlice(2, sampleCount);
        clipSlice.copyFrom(0, 0, clipBuffer, 0, sourceStart, sampleCount);
        clipSlice.copyFrom(1, 0, clipBuffer, 1, sourceStart, sampleCount);
        applyTrackMixAutomationToBuffer(project, track, clipSlice, overlapStartFrame, outSummary.sampleRate);
        outBuffer.addFrom(0, destinationStart, clipSlice, 0, 0, sampleCount);
        outBuffer.addFrom(1, destinationStart, clipSlice, 1, 0, sampleCount);
        mixedTrackAudio = true;
    }

    if (mixedTrackAudio)
    {
        ++outSummary.mixedAudioTrackCount;
        anyRenderedSource = true;
    }

    nativeVstHost.clearPluginGraph();

    if (!anyRenderedSource)
        return juce::Result::fail("No instrument render or audio clip content was available for this track inside the locator range.");

    return juce::Result::ok();
}
} // namespace

juce::Result exportProjectRangeToAudioFile(const juce::File& file,
                                           const ProjectState& project,
                                           NativeVstHostSession& nativeVstHost,
                                           AudioExportSummary& outSummary,
                                           int sampleRate)
{
    outSummary = {};
    outSummary.sampleRate = juce::jmax(1, sampleRate);

    const auto leftSec = juce::jmax(0.0, project.leftLocatorSec);
    const auto rightSec = juce::jmax(leftSec, project.rightLocatorSec);
    if (rightSec <= leftSec)
        return juce::Result::fail("Right locator must be greater than left locator for WAV export.");

    const auto exportStartFrame = static_cast<int64_t>(std::llround(leftSec * outSummary.sampleRate));
    const auto totalFrames = juce::jmax(1, static_cast<int>(std::llround((rightSec - leftSec) * outSummary.sampleRate)));
    outSummary.frameCount = totalFrames;

    juce::AudioBuffer<float> finalMix(2, totalFrames);
    finalMix.clear();

    std::unordered_map<std::string, std::unique_ptr<juce::AudioBuffer<float>>> clipCache;

    bool anyRenderedSource = false;
    for (int trackIndex = 0; trackIndex < static_cast<int>(project.tracks.size()); ++trackIndex)
    {
        if (!trackIsAudible(project, trackIndex))
            continue;

        const auto& track = project.tracks[static_cast<size_t>(trackIndex)];
        const auto pluginPath = resolveRackPluginPath(project, track);
        const auto canRenderNativeInstrument = track.trackType.trim().equalsIgnoreCase("instrument")
            && pluginPath.isNotEmpty()
            && juce::File(pluginPath).exists();
        bool renderFailed = false;

        if (canRenderNativeInstrument)
        {
            const auto result = renderNativeTrackToMix(project,
                                                       trackIndex,
                                                       track,
                                                       pluginPath,
                                                       exportStartFrame,
                                                       totalFrames,
                                                       outSummary.sampleRate,
                                                       nativeVstHost,
                                                       finalMix,
                                                       outSummary);
            if (result.failed())
            {
                renderFailed = true;
                if (track.renderedAudioPath.isNotEmpty() && juce::File(track.renderedAudioPath).existsAsFile())
                {
                    outSummary.warnings.addIfNotAlreadyThere(describeTrackWarningsPrefix(track, trackIndex)
                                                              + "native instrument render failed, so export fell back to the rendered audio stem.");
                }
                else
                {
                    outSummary.warnings.addIfNotAlreadyThere(describeTrackWarningsPrefix(track, trackIndex)
                                                              + result.getErrorMessage());
                }
            }
            else
            {
                anyRenderedSource = true;
            }
        }

        const auto clipPlacements = collectTrackClipPlacements(project,
                                                               trackIndex,
                                                               !canRenderNativeInstrument || renderFailed);
        bool mixedTrackAudio = false;
        if (!clipPlacements.empty() && track.vstFxChain.size() > 0)
        {
            outSummary.warnings.addIfNotAlreadyThere(describeTrackWarningsPrefix(track, trackIndex)
                                                      + "native WAV export currently ignores audio-track FX-chain processing.");
        }

        for (const auto& placement : clipPlacements)
        {
            const juce::File clipFile(placement.path);
            if (!clipFile.existsAsFile())
            {
                outSummary.warnings.addIfNotAlreadyThere(describeTrackWarningsPrefix(track, trackIndex)
                                                          + "skipped missing audio clip " + placement.path + ".");
                continue;
            }

            const auto cacheKey = clipFile.getFullPathName().toStdString();
            auto cacheIt = clipCache.find(cacheKey);
            if (cacheIt == clipCache.end())
            {
                auto loadedBuffer = std::make_unique<juce::AudioBuffer<float>>();
                const auto loadResult = loadAudioFileToStereoBuffer(clipFile,
                                                                    *loadedBuffer,
                                                                    static_cast<double>(outSummary.sampleRate));
                if (loadResult.failed())
                {
                    outSummary.warnings.addIfNotAlreadyThere(describeTrackWarningsPrefix(track, trackIndex)
                                                              + "could not read audio clip "
                                                              + clipFile.getFileName()
                                                              + ": "
                                                              + loadResult.getErrorMessage());
                    continue;
                }

                cacheIt = clipCache.emplace(cacheKey, std::move(loadedBuffer)).first;
            }

            const auto& clipBuffer = *cacheIt->second;
            const auto clipStartFrame = static_cast<int64_t>(std::llround(placement.startSec * outSummary.sampleRate));
            const auto clipEndFrame = clipStartFrame + clipBuffer.getNumSamples();
            const auto exportEndFrame = exportStartFrame + static_cast<int64_t>(totalFrames);
            if (clipEndFrame <= exportStartFrame || clipStartFrame >= exportEndFrame)
                continue;

            const auto overlapStartFrame = juce::jmax(exportStartFrame, clipStartFrame);
            const auto overlapEndFrame = juce::jmin(exportEndFrame, clipEndFrame);
            const auto sourceStart = static_cast<int>(overlapStartFrame - clipStartFrame);
            const auto destinationStart = static_cast<int>(overlapStartFrame - exportStartFrame);
            const auto sampleCount = static_cast<int>(overlapEndFrame - overlapStartFrame);
            if (sampleCount <= 0)
                continue;

            juce::AudioBuffer<float> clipSlice(2, sampleCount);
            clipSlice.copyFrom(0, 0, clipBuffer, 0, sourceStart, sampleCount);
            clipSlice.copyFrom(1, 0, clipBuffer, 1, sourceStart, sampleCount);
            applyTrackMixAutomationToBuffer(project, track, clipSlice, overlapStartFrame, outSummary.sampleRate);
            finalMix.addFrom(0, destinationStart, clipSlice, 0, 0, sampleCount);
            finalMix.addFrom(1, destinationStart, clipSlice, 1, 0, sampleCount);
            mixedTrackAudio = true;
        }

        if (mixedTrackAudio)
        {
            ++outSummary.mixedAudioTrackCount;
            anyRenderedSource = true;
        }
    }

    nativeVstHost.clearPluginGraph();

    if (!anyRenderedSource)
        return juce::Result::fail("No audible instrument or sample content was available inside the locator range.");

    return writeStereoBufferToAudioFile(file, finalMix, static_cast<double>(outSummary.sampleRate));
}

juce::Result exportProjectRangeToWavFile(const juce::File& file,
                                         const ProjectState& project,
                                         NativeVstHostSession& nativeVstHost,
                                         AudioExportSummary& outSummary,
                                         int sampleRate)
{
    return exportProjectRangeToAudioFile(file, project, nativeVstHost, outSummary, sampleRate);
}

juce::Result exportTrackRangeToAudioFile(const juce::File& file,
                                         const ProjectState& project,
                                         int trackIndex,
                                         NativeVstHostSession& nativeVstHost,
                                         AudioExportSummary& outSummary,
                                         int sampleRate)
{
    juce::AudioBuffer<float> trackBuffer;
    const auto result = renderTrackRangeToBuffer(project,
                                                 trackIndex,
                                                 nativeVstHost,
                                                 trackBuffer,
                                                 outSummary,
                                                 sampleRate);
    if (result.failed())
        return result;

    return writeStereoBufferToAudioFile(file, trackBuffer, static_cast<double>(outSummary.sampleRate));
}

juce::Result exportTrackRangeToWavFile(const juce::File& file,
                                       const ProjectState& project,
                                       int trackIndex,
                                       NativeVstHostSession& nativeVstHost,
                                       AudioExportSummary& outSummary,
                                       int sampleRate)
{
    return exportTrackRangeToAudioFile(file, project, trackIndex, nativeVstHost, outSummary, sampleRate);
}

juce::Result exportProjectStemsToFolder(const juce::File& folder,
                                        const ProjectState& project,
                                        NativeVstHostSession& nativeVstHost,
                                        AudioExportBatchSummary& outSummary,
                                        int sampleRate)
{
    outSummary = {};
    outSummary.sampleRate = juce::jmax(1, sampleRate);

    if (!folder.createDirectory())
        return juce::Result::fail("Could not create the stems folder.");

    for (int trackIndex = 0; trackIndex < static_cast<int>(project.tracks.size()); ++trackIndex)
    {
        const auto& track = project.tracks[static_cast<size_t>(trackIndex)];
        AudioExportSummary trackSummary;
        juce::AudioBuffer<float> trackBuffer;
        const auto renderResult = renderTrackRangeToBuffer(project,
                                                           trackIndex,
                                                           nativeVstHost,
                                                           trackBuffer,
                                                           trackSummary,
                                                           outSummary.sampleRate);
        if (renderResult.failed())
        {
            outSummary.warnings.addIfNotAlreadyThere(describeTrackWarningsPrefix(track, trackIndex)
                                                     + renderResult.getErrorMessage());
            outSummary.warnings.addArray(trackSummary.warnings);
            continue;
        }

        const auto stemFile = folder.getChildFile(makeStemFileName(track, trackIndex));
        const auto writeResult = writeStereoBufferToWavFile(stemFile,
                                                            trackBuffer,
                                                            static_cast<double>(outSummary.sampleRate));
        if (writeResult.failed())
        {
            outSummary.warnings.addIfNotAlreadyThere(describeTrackWarningsPrefix(track, trackIndex)
                                                     + writeResult.getErrorMessage());
            outSummary.warnings.addArray(trackSummary.warnings);
            continue;
        }

        AudioExportStemFile exportedFile;
        exportedFile.trackIndex = trackIndex;
        exportedFile.trackName = trackDisplayName(track, trackIndex);
        exportedFile.filePath = stemFile.getFullPathName();
        outSummary.exportedFiles.push_back(exportedFile);
        outSummary.stemPaths.add(stemFile.getFullPathName());
        ++outSummary.exportedTrackCount;
        outSummary.warnings.addArray(trackSummary.warnings);
    }

    nativeVstHost.clearPluginGraph();

    if (outSummary.exportedTrackCount == 0)
        return juce::Result::fail("No track stems could be exported from the current locator range.");

    return juce::Result::ok();
}

} // namespace aims
