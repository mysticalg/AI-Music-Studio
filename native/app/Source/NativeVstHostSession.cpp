#include "NativeVstHostSession.h"

#include <array>
#include <cmath>
#include <cstring>

namespace
{
extern "C"
{
    struct AimsVstHostTransportSnapshot
    {
        double sampleRate;
        double cpuUsage;
        double inprocessTransportPositionFrame;
        double audioEnginePositionFrame;
        float pluginOutputPeakLevel;
        float masterPeakLeft;
        float masterPeakRight;
        int inprocessTransportRunning;
        int audioEngineRunning;
        int audioEngineTailing;
        int editorOpen;
        int stateGeneration;
        int audioEngineTrackCount;
    };

    struct AimsVstHostParameterSnapshot
    {
        int editorOpen;
        int stateGeneration;
        int parameterCount;
    };

    struct AimsVstHostParameterSnapshotEntry
    {
        char name[256];
        float normalizedValue;
    };

    struct AimsVstHostTrackNoteEntry
    {
        int startTick;
        int endTick;
        int pitch;
        int velocity;
        int channel;
    };

    struct AimsVstHostTrackControllerEventEntry
    {
        int tick;
        int control;
        int value;
        int channel;
    };

    struct AimsVstHostStringEntry
    {
        char value[256];
    };

    struct AimsVstHostAudioDeviceSnapshot
    {
        char audioDeviceType[256];
        char audioDeviceName[256];
        double sampleRate;
        int bufferSize;
        int availableAudioDeviceTypeCount;
        int availableAudioOutputDeviceCount;
        int availableAudioSampleRateCount;
        int availableAudioBufferSizeCount;
    };

    void* aims_vst_host_create(const char* pluginPath,
                               int restoreLastPluginOnStartup,
                               int openEditor,
                               double sampleRate,
                               int bufferSize,
                               const char* audioDeviceType,
                               const char* audioOutputDeviceName,
                               char* errorBuffer,
                               int errorBufferBytes);

    void* aims_vst_host_create_ex(const char* pluginPath,
                                  int restoreLastPluginOnStartup,
                                  int openEditor,
                                  double sampleRate,
                                  int bufferSize,
                                  const char* audioDeviceType,
                                  const char* audioOutputDeviceName,
                                  int bridgeModeEnabled,
                                  char* errorBuffer,
                                  int errorBufferBytes);

    int aims_vst_host_command(void* handle,
                              const char* requestLine,
                              char* responseBuffer,
                              int responseBufferBytes);

    int aims_vst_host_open_audio_engine_track_editor(void* handle,
                                                     int trackIndex,
                                                     char* errorBuffer,
                                                     int errorBufferBytes);

    int aims_vst_host_close_audio_engine_track_editor(void* handle,
                                                      int trackIndex,
                                                      char* errorBuffer,
                                                      int errorBufferBytes);

    int aims_vst_host_get_audio_engine_track_editor_state(void* handle,
                                                          int trackIndex,
                                                          int* isOpen,
                                                          char* errorBuffer,
                                                          int errorBufferBytes);

    int aims_vst_host_audio_engine_track_note_on(void* handle,
                                                 int trackIndex,
                                                 int note,
                                                 int channel,
                                                 float velocity,
                                                 char* errorBuffer,
                                                 int errorBufferBytes);

    int aims_vst_host_audio_engine_track_note_off(void* handle,
                                                  int trackIndex,
                                                  int note,
                                                  int channel,
                                                  float velocity,
                                                  char* errorBuffer,
                                                  int errorBufferBytes);

    int aims_vst_host_audio_engine_track_all_notes_off(void* handle,
                                                       int trackIndex,
                                                       int channel,
                                                       char* errorBuffer,
                                                       int errorBufferBytes);

    int aims_vst_host_set_audio_engine_state_json(void* handle,
                                                  const char* tracksJson,
                                                  double bpm,
                                                  int ticksPerBeat,
                                                  int loopEnabled,
                                                  int64_t loopStartTick,
                                                  int64_t loopEndTick,
                                                  int metronomeEnabled,
                                                  int preserveTransport,
                                                  char* errorBuffer,
                                                  int errorBufferBytes);

    int aims_vst_host_set_audio_engine_transport(void* handle,
                                                 double bpm,
                                                 int ticksPerBeat,
                                                 int loopEnabled,
                                                 int64_t loopStartTick,
                                                 int64_t loopEndTick,
                                                 int metronomeEnabled,
                                                 char* errorBuffer,
                                                 int errorBufferBytes);

    int aims_vst_host_set_audio_engine_track_parameters(void* handle,
                                                        int trackIndex,
                                                        const AimsVstHostParameterSnapshotEntry* parameterEntries,
                                                        int parameterCount,
                                                        const char* parameterSignature,
                                                        char* errorBuffer,
                                                        int errorBufferBytes);

    int aims_vst_host_set_audio_engine_track_notes(void* handle,
                                                   int trackIndex,
                                                   const AimsVstHostTrackNoteEntry* noteEntries,
                                                   int noteCount,
                                                   const AimsVstHostTrackControllerEventEntry* controllerEntries,
                                                   int controllerCount,
                                                   int* appliedNoteCount,
                                                   char* errorBuffer,
                                                   int errorBufferBytes);

    int aims_vst_host_set_audio_engine_track_mix_state(void* handle,
                                                       int trackIndex,
                                                       float volume,
                                                       float outputGainDb,
                                                       float pan,
                                                       int audible,
                                                       char* errorBuffer,
                                                       int errorBufferBytes);

    int aims_vst_host_start_audio_engine(void* handle,
                                         int64_t startTick,
                                         char* errorBuffer,
                                         int errorBufferBytes);

    int aims_vst_host_stop_audio_engine(void* handle,
                                        int allowTail,
                                        char* errorBuffer,
                                        int errorBufferBytes);

    int aims_vst_host_configure_audio_device(void* handle,
                                             const char* audioDeviceType,
                                             const char* audioDeviceName,
                                             double sampleRate,
                                             int bufferSize,
                                             char* errorBuffer,
                                             int errorBufferBytes);

    int aims_vst_host_get_transport_snapshot(void* handle,
                                             AimsVstHostTransportSnapshot* snapshot,
                                             float* trackPeakLevels,
                                             int trackPeakLevelCapacity,
                                             char* errorBuffer,
                                             int errorBufferBytes);

    int aims_vst_host_get_audio_engine_track_parameter_snapshot(void* handle,
                                                                int trackIndex,
                                                                AimsVstHostParameterSnapshot* snapshot,
                                                                AimsVstHostParameterSnapshotEntry* parameterEntries,
                                                                int parameterEntryCapacity,
                                                                char* errorBuffer,
                                                                int errorBufferBytes);

    int aims_vst_host_save_audio_engine_track_plugin_state(void* handle,
                                                           int trackIndex,
                                                           const char* statePath,
                                                           char* errorBuffer,
                                                           int errorBufferBytes);

    int aims_vst_host_get_audio_device_snapshot(void* handle,
                                                AimsVstHostAudioDeviceSnapshot* snapshot,
                                                AimsVstHostStringEntry* availableDeviceTypes,
                                                int availableDeviceTypeCapacity,
                                                AimsVstHostStringEntry* availableOutputDevices,
                                                int availableOutputDeviceCapacity,
                                                double* availableSampleRates,
                                                int availableSampleRateCapacity,
                                                int* availableBufferSizes,
                                                int availableBufferSizeCapacity,
                                                char* errorBuffer,
                                                int errorBufferBytes);

    void aims_vst_host_set_editor_state_callback(void* handle,
                                                 void (*callback)(int isOpen, void* userData),
                                                 void* userData);

    void aims_vst_host_destroy(void* handle);
}

constexpr int kHostErrorBufferBytes = 4096;
constexpr int kHostResponseBufferBytes = 8 * 1024 * 1024;
constexpr double kFallbackSampleRate = 44100.0;

juce::Result resultFromResponse(const juce::var& response, const juce::String& fallbackError)
{
    auto* object = response.getDynamicObject();
    if (object == nullptr)
        return juce::Result::fail(fallbackError);

    if (!static_cast<bool>(object->getProperty("ok")))
    {
        const auto message = object->getProperty("message").toString().trim();
        return juce::Result::fail(message.isNotEmpty() ? message : fallbackError);
    }

    return juce::Result::ok();
}

bool trackIsAudible(const aims::ProjectState& project, int trackIndex)
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

juce::var buildParameterPayload(const juce::NamedValueSet& parameters)
{
    auto* params = new juce::DynamicObject();
    for (int index = 0; index < parameters.size(); ++index)
        params->setProperty(parameters.getName(index), parameters.getValueAt(index));
    return juce::var(params);
}

juce::var buildParameterPayload(const aims::TrackState& track)
{
    return buildParameterPayload(track.vstiParameters);
}

juce::String parameterPayloadSignature(const juce::NamedValueSet& values)
{
    juce::StringArray entries;
    for (int index = 0; index < values.size(); ++index)
    {
        const auto name = values.getName(index).toString().trim();
        if (name.isEmpty())
            continue;

        const auto value = static_cast<double>(values.getValueAt(index));
        entries.add(name + "=" + juce::String(std::isfinite(value) ? value : 0.0, 4));
    }

    entries.sort(true);
    return entries.joinIntoString("|");
}

juce::Array<juce::var> buildNotePayload(const aims::TrackState& track)
{
    juce::Array<juce::var> notes;
    for (const auto& note : track.notes)
    {
        auto* noteObject = new juce::DynamicObject();
        noteObject->setProperty("start_tick", juce::jmax(0, note.startTick));
        noteObject->setProperty("end_tick", juce::jmax(note.startTick + 1, note.startTick + note.durationTick));
        noteObject->setProperty("pitch", juce::jlimit(0, 127, note.pitch));
        noteObject->setProperty("velocity", juce::jlimit(1, 127, note.velocity));
        noteObject->setProperty("channel", juce::jlimit(1, 16, track.midiChannel + 1));
        notes.add(juce::var(noteObject));
    }
    return notes;
}

juce::Array<juce::var> buildControllerEventPayload(const aims::ProjectState& project, int trackIndex)
{
    juce::Array<juce::var> events;
    const auto controllerEvents = aims::collectTrackControllerEvents(project, trackIndex);
    for (const auto& event : controllerEvents)
    {
        auto* eventObject = new juce::DynamicObject();
        eventObject->setProperty("tick", juce::jmax(0, event.tick));
        eventObject->setProperty("control", juce::jlimit(0, 127, event.controller));
        eventObject->setProperty("value", juce::jlimit(0, 127, event.value));
        eventObject->setProperty("channel", juce::jlimit(1, 16, event.channel));
        events.add(juce::var(eventObject));
    }
    return events;
}

juce::Array<juce::var> buildClipPayload(const aims::ProjectState& project, int trackIndex, double sampleRate)
{
    juce::Array<juce::var> clips;

    for (const auto& clip : project.sampleClips)
    {
        if (clip.trackIndex != trackIndex)
            continue;
        if (clip.path.isEmpty())
            continue;

        auto* clipObject = new juce::DynamicObject();
        clipObject->setProperty("path", clip.path);
        clipObject->setProperty("start_frame", static_cast<double>(juce::jmax<int64_t>(0, static_cast<int64_t>(std::llround(clip.startSec * sampleRate)))));
        clips.add(juce::var(clipObject));
    }

    if (juce::isPositiveAndBelow(trackIndex, static_cast<int>(project.tracks.size())))
    {
        const auto& track = project.tracks[static_cast<size_t>(trackIndex)];
        if (track.renderedAudioPath.isNotEmpty() && juce::File(track.renderedAudioPath).existsAsFile())
        {
            auto* clipObject = new juce::DynamicObject();
            clipObject->setProperty("path", track.renderedAudioPath);
            clipObject->setProperty("start_frame", 0.0);
            clips.add(juce::var(clipObject));
        }
    }

    return clips;
}

juce::Array<juce::var> buildAutomationPayload(const aims::TrackState& track)
{
    juce::Array<juce::var> automation;

    for (const auto& lane : track.automationLanes)
    {
        if (!lane.enabled || lane.points.empty())
            continue;

        auto* laneObject = new juce::DynamicObject();
        laneObject->setProperty("target", lane.target);

        juce::Array<juce::var> points;
        for (const auto& point : lane.points)
        {
            auto* pointObject = new juce::DynamicObject();
            pointObject->setProperty("tick", juce::jmax(0, point.tick));
            pointObject->setProperty("value", point.value);
            points.add(juce::var(pointObject));
        }

        laneObject->setProperty("points", juce::var(points));
        automation.add(juce::var(laneObject));
    }

    return automation;
}

juce::var buildAudioEngineTrackPayload(const aims::ProjectState& project, int trackIndex, double sampleRate)
{
    const auto& track = project.tracks[static_cast<size_t>(trackIndex)];
    const auto clips = buildClipPayload(project, trackIndex, sampleRate);
    const auto automation = buildAutomationPayload(track);
    const bool hasAudioContent = !clips.isEmpty();

    auto* payload = new juce::DynamicObject();
    payload->setProperty("name", track.name.isNotEmpty() ? track.name : "Track " + juce::String(trackIndex + 1));
    payload->setProperty("volume", track.volume);
    payload->setProperty("output_gain_db", track.vstiOutputGainDb);
    payload->setProperty("pan", juce::jlimit(-1.0, 1.0, track.pan));
    payload->setProperty("audible", trackIsAudible(project, trackIndex));

    const auto pluginPath = aims::resolveRackPluginPath(project, track);
    const auto canUseInstrument = pluginPath.isNotEmpty()
        && juce::File(pluginPath).exists();
    if (canUseInstrument)
    {
        payload->setProperty("track_type", "instrument");
        payload->setProperty("plugin_path", pluginPath);
        payload->setProperty("state_path", track.vstiStatePath);
        payload->setProperty("parameters", buildParameterPayload(track));
        payload->setProperty("notes", juce::var(buildNotePayload(track)));
        payload->setProperty("controller_events", juce::var(buildControllerEventPayload(project, trackIndex)));
    }
    else
    {
        payload->setProperty("track_type", "audio");
        payload->setProperty("plugin_path", "");
        payload->setProperty("state_path", "");
        payload->setProperty("parameters", juce::var(new juce::DynamicObject()));
        payload->setProperty("clips", juce::var(clips));
    }

    if (!automation.isEmpty())
        payload->setProperty("automation", juce::var(automation));

    if (!canUseInstrument && !hasAudioContent)
        payload->setProperty("audible", false);

    return juce::var(payload);
}
} // namespace

namespace aims
{

NativeVstHostSession::NativeVstHostSession(bool shouldUseBridgeMode) noexcept
    : bridgeModeEnabled(shouldUseBridgeMode)
{
}

NativeVstHostSession::~NativeVstHostSession()
{
    destroy();
}

juce::Result NativeVstHostSession::ensureCreated()
{
    if (handle != nullptr)
        return juce::Result::ok();

    juce::HeapBlock<char> errorBuffer(static_cast<size_t>(kHostErrorBufferBytes));
    std::memset(errorBuffer.get(), 0, static_cast<size_t>(kHostErrorBufferBytes));

    handle = aims_vst_host_create_ex("",
                                     0,
                                     0,
                                     0.0,
                                     0,
                                     "",
                                     "",
                                     bridgeModeEnabled ? 1 : 0,
                                     errorBuffer.get(),
                                     kHostErrorBufferBytes);
    if (handle == nullptr)
    {
        const auto error = juce::String::fromUTF8(errorBuffer.get()).trim();
        return juce::Result::fail(error.isNotEmpty() ? error : "Failed to create native VST host session.");
    }

    aims_vst_host_set_editor_state_callback(handle,
                                            editorStateCallback != nullptr ? &NativeVstHostSession::editorStateThunk : nullptr,
                                            editorStateCallback != nullptr ? this : nullptr);

    return juce::Result::ok();
}

void NativeVstHostSession::destroy()
{
    if (handle == nullptr)
        return;

    aims_vst_host_set_editor_state_callback(handle, nullptr, nullptr);
    aims_vst_host_destroy(handle);
    handle = nullptr;
}

bool NativeVstHostSession::isReady() const noexcept
{
    return handle != nullptr;
}

juce::Result NativeVstHostSession::clearPluginGraph()
{
    juce::DynamicObject::Ptr request = new juce::DynamicObject();
    request->setProperty("command", "clear_plugin_graph");

    juce::var response;
    const auto result = runCommand(juce::var(request.get()), response);
    if (result.failed())
        return result;

    return resultFromResponse(response, "Native host clear plugin graph command failed.");
}

juce::Result NativeVstHostSession::renderOfflinePluginGraph(const juce::var& tracks,
                                                            int frames,
                                                            juce::var& response)
{
    juce::DynamicObject::Ptr request = new juce::DynamicObject();
    request->setProperty("command", "render_plugin_graph");
    request->setProperty("tracks", tracks);
    request->setProperty("frames", juce::jmax(1, frames));

    const auto result = runCommand(juce::var(request.get()), response);
    if (result.failed())
        return result;

    return resultFromResponse(response, "Native host offline graph render failed.");
}

juce::Result NativeVstHostSession::configureAudioDevice(const juce::String& audioDeviceType,
                                                        const juce::String& audioDeviceName,
                                                        double sampleRate,
                                                        int bufferSize)
{
    if (handle == nullptr)
        return juce::Result::fail("Native VST host session is not ready.");

    juce::HeapBlock<char> errorBuffer(static_cast<size_t>(kHostErrorBufferBytes));
    std::memset(errorBuffer.get(), 0, static_cast<size_t>(kHostErrorBufferBytes));

    const auto ok = aims_vst_host_configure_audio_device(handle,
                                                         audioDeviceType.toRawUTF8(),
                                                         audioDeviceName.toRawUTF8(),
                                                         sampleRate,
                                                         bufferSize,
                                                         errorBuffer.get(),
                                                         kHostErrorBufferBytes);
    if (ok != 0)
        return juce::Result::ok();

    const auto error = juce::String::fromUTF8(errorBuffer.get()).trim();
    return juce::Result::fail(error.isNotEmpty() ? error : "Native host audio device update failed.");
}

juce::Result NativeVstHostSession::setAudioEngineState(const ProjectState& project, bool preserveTransport)
{
    if (handle == nullptr)
        return juce::Result::fail("Native VST host session is not ready.");

    double sampleRate = cachedSampleRate > 0.0 ? cachedSampleRate : kFallbackSampleRate;
    if (cachedSampleRate <= 0.0)
    {
        AudioDeviceSnapshot deviceSnapshot;
        if (queryAudioDeviceSnapshot(deviceSnapshot).wasOk() && deviceSnapshot.sampleRate > 0.0)
            sampleRate = deviceSnapshot.sampleRate;
    }

    juce::Array<juce::var> tracks;
    tracks.ensureStorageAllocated(static_cast<int>(project.tracks.size()));
    for (int trackIndex = 0; trackIndex < static_cast<int>(project.tracks.size()); ++trackIndex)
        tracks.add(buildAudioEngineTrackPayload(project, trackIndex, sampleRate));

    const auto tracksJson = juce::JSON::toString(juce::var(tracks), false);
    juce::HeapBlock<char> errorBuffer(static_cast<size_t>(kHostErrorBufferBytes));
    std::memset(errorBuffer.get(), 0, static_cast<size_t>(kHostErrorBufferBytes));

    const auto ok = aims_vst_host_set_audio_engine_state_json(handle,
                                                              tracksJson.toRawUTF8(),
                                                              project.bpm,
                                                              kTicksPerBeat,
                                                              project.loopEnabled ? 1 : 0,
                                                              project.leftLocatorTick,
                                                              project.rightLocatorTick,
                                                              project.metronomeEnabled ? 1 : 0,
                                                              preserveTransport ? 1 : 0,
                                                              errorBuffer.get(),
                                                              kHostErrorBufferBytes);
    if (ok != 0)
        return juce::Result::ok();

    const auto error = juce::String::fromUTF8(errorBuffer.get()).trim();
    return juce::Result::fail(error.isNotEmpty() ? error : "Native audio engine state update failed.");
}

juce::Result NativeVstHostSession::updateAudioEngineTransport(const ProjectState& project)
{
    if (handle == nullptr)
        return juce::Result::fail("Native VST host session is not ready.");

    juce::HeapBlock<char> errorBuffer(static_cast<size_t>(kHostErrorBufferBytes));
    std::memset(errorBuffer.get(), 0, static_cast<size_t>(kHostErrorBufferBytes));

    const auto ok = aims_vst_host_set_audio_engine_transport(handle,
                                                             project.bpm,
                                                             kTicksPerBeat,
                                                             project.loopEnabled ? 1 : 0,
                                                             project.leftLocatorTick,
                                                             project.rightLocatorTick,
                                                             project.metronomeEnabled ? 1 : 0,
                                                             errorBuffer.get(),
                                                             kHostErrorBufferBytes);
    if (ok != 0)
        return juce::Result::ok();

    const auto error = juce::String::fromUTF8(errorBuffer.get()).trim();
    return juce::Result::fail(error.isNotEmpty() ? error : "Native audio engine transport update failed.");
}

juce::Result NativeVstHostSession::setAudioEngineTrackParameters(int trackIndex, const TrackState& track)
{
    if (handle == nullptr)
        return juce::Result::fail("Native VST host session is not ready.");

    juce::Array<AimsVstHostParameterSnapshotEntry> parameterEntries;
    parameterEntries.ensureStorageAllocated(track.vstiParameters.size());
    for (int index = 0; index < track.vstiParameters.size(); ++index)
    {
        AimsVstHostParameterSnapshotEntry entry{};
        const auto name = track.vstiParameters.getName(index).toString().trim();
        if (name.isEmpty())
            continue;

        name.copyToUTF8(entry.name, static_cast<int>(sizeof(entry.name)));
        const auto value = static_cast<double>(track.vstiParameters.getValueAt(index));
        entry.normalizedValue = static_cast<float>(juce::jlimit(0.0, 100.0, std::isfinite(value) ? value : 0.0) / 100.0);
        parameterEntries.add(entry);
    }

    juce::HeapBlock<char> errorBuffer(static_cast<size_t>(kHostErrorBufferBytes));
    std::memset(errorBuffer.get(), 0, static_cast<size_t>(kHostErrorBufferBytes));
    const auto signature = parameterPayloadSignature(track.vstiParameters);

    const auto ok = aims_vst_host_set_audio_engine_track_parameters(handle,
                                                                    trackIndex,
                                                                    parameterEntries.isEmpty() ? nullptr : parameterEntries.getRawDataPointer(),
                                                                    parameterEntries.size(),
                                                                    signature.toRawUTF8(),
                                                                    errorBuffer.get(),
                                                                    kHostErrorBufferBytes);
    if (ok != 0)
        return juce::Result::ok();

    const auto error = juce::String::fromUTF8(errorBuffer.get()).trim();
    return juce::Result::fail(error.isNotEmpty() ? error : "Native audio engine track parameter update failed.");
}

juce::Result NativeVstHostSession::setAudioEngineTrackNotes(int trackIndex, const ProjectState& project, const TrackState& track)
{
    if (handle == nullptr)
        return juce::Result::fail("Native VST host session is not ready.");

    juce::Array<AimsVstHostTrackNoteEntry> noteEntries;
    noteEntries.ensureStorageAllocated(static_cast<int>(track.notes.size()));
    for (const auto& note : track.notes)
    {
        AimsVstHostTrackNoteEntry entry{};
        entry.startTick = juce::jmax(0, note.startTick);
        entry.endTick = juce::jmax(note.startTick + 1, note.startTick + note.durationTick);
        entry.pitch = juce::jlimit(0, 127, note.pitch);
        entry.velocity = juce::jlimit(1, 127, note.velocity);
        entry.channel = juce::jlimit(1, 16, track.midiChannel + 1);
        noteEntries.add(entry);
    }

    const auto controllerEvents = aims::collectTrackControllerEvents(project, trackIndex);
    juce::Array<AimsVstHostTrackControllerEventEntry> controllerEntries;
    controllerEntries.ensureStorageAllocated(static_cast<int>(controllerEvents.size()));
    for (const auto& event : controllerEvents)
    {
        AimsVstHostTrackControllerEventEntry entry{};
        entry.tick = juce::jmax(0, event.tick);
        entry.control = juce::jlimit(0, 127, event.controller);
        entry.value = juce::jlimit(0, 127, event.value);
        entry.channel = juce::jlimit(1, 16, event.channel);
        controllerEntries.add(entry);
    }

    juce::HeapBlock<char> errorBuffer(static_cast<size_t>(kHostErrorBufferBytes));
    std::memset(errorBuffer.get(), 0, static_cast<size_t>(kHostErrorBufferBytes));
    int appliedNoteCount = 0;
    const auto ok = aims_vst_host_set_audio_engine_track_notes(handle,
                                                               trackIndex,
                                                               noteEntries.isEmpty() ? nullptr : noteEntries.getRawDataPointer(),
                                                               noteEntries.size(),
                                                               controllerEntries.isEmpty() ? nullptr : controllerEntries.getRawDataPointer(),
                                                               controllerEntries.size(),
                                                               &appliedNoteCount,
                                                               errorBuffer.get(),
                                                               kHostErrorBufferBytes);
    if (ok != 0)
        return juce::Result::ok();

    const auto error = juce::String::fromUTF8(errorBuffer.get()).trim();
    return juce::Result::fail(error.isNotEmpty() ? error : "Native audio engine track note update failed.");
}

juce::Result NativeVstHostSession::setAudioEngineTrackMixState(const ProjectState& project, int trackIndex, const TrackState& track)
{
    if (handle == nullptr)
        return juce::Result::fail("Native VST host session is not ready.");

    juce::HeapBlock<char> errorBuffer(static_cast<size_t>(kHostErrorBufferBytes));
    std::memset(errorBuffer.get(), 0, static_cast<size_t>(kHostErrorBufferBytes));

    const auto ok = aims_vst_host_set_audio_engine_track_mix_state(handle,
                                                                   trackIndex,
                                                                   static_cast<float>(track.volume),
                                                                   static_cast<float>(track.vstiOutputGainDb),
                                                                   static_cast<float>(juce::jlimit(-1.0, 1.0, track.pan)),
                                                                   trackIsAudible(project, trackIndex) ? 1 : 0,
                                                                   errorBuffer.get(),
                                                                   kHostErrorBufferBytes);
    if (ok != 0)
        return juce::Result::ok();

    const auto error = juce::String::fromUTF8(errorBuffer.get()).trim();
    return juce::Result::fail(error.isNotEmpty() ? error : "Native audio engine track mix update failed.");
}

juce::Result NativeVstHostSession::noteOnAudioEngineTrack(int trackIndex, int note, int channel, float velocity)
{
    if (handle == nullptr)
        return juce::Result::fail("Native VST host session is not ready.");

    juce::HeapBlock<char> errorBuffer(static_cast<size_t>(kHostErrorBufferBytes));
    std::memset(errorBuffer.get(), 0, static_cast<size_t>(kHostErrorBufferBytes));

    const auto ok = aims_vst_host_audio_engine_track_note_on(handle,
                                                             trackIndex,
                                                             note,
                                                             channel,
                                                             velocity,
                                                             errorBuffer.get(),
                                                             kHostErrorBufferBytes);
    if (ok != 0)
        return juce::Result::ok();

    const auto error = juce::String::fromUTF8(errorBuffer.get()).trim();
    return juce::Result::fail(error.isNotEmpty() ? error : "Native audio engine live note-on failed.");
}

juce::Result NativeVstHostSession::noteOffAudioEngineTrack(int trackIndex, int note, int channel, float velocity)
{
    if (handle == nullptr)
        return juce::Result::fail("Native VST host session is not ready.");

    juce::HeapBlock<char> errorBuffer(static_cast<size_t>(kHostErrorBufferBytes));
    std::memset(errorBuffer.get(), 0, static_cast<size_t>(kHostErrorBufferBytes));

    const auto ok = aims_vst_host_audio_engine_track_note_off(handle,
                                                              trackIndex,
                                                              note,
                                                              channel,
                                                              velocity,
                                                              errorBuffer.get(),
                                                              kHostErrorBufferBytes);
    if (ok != 0)
        return juce::Result::ok();

    const auto error = juce::String::fromUTF8(errorBuffer.get()).trim();
    return juce::Result::fail(error.isNotEmpty() ? error : "Native audio engine live note-off failed.");
}

juce::Result NativeVstHostSession::allNotesOffAudioEngineTrack(int trackIndex, int channel)
{
    if (handle == nullptr)
        return juce::Result::fail("Native VST host session is not ready.");

    juce::HeapBlock<char> errorBuffer(static_cast<size_t>(kHostErrorBufferBytes));
    std::memset(errorBuffer.get(), 0, static_cast<size_t>(kHostErrorBufferBytes));

    const auto ok = aims_vst_host_audio_engine_track_all_notes_off(handle,
                                                                   trackIndex,
                                                                   channel,
                                                                   errorBuffer.get(),
                                                                   kHostErrorBufferBytes);
    if (ok != 0)
        return juce::Result::ok();

    const auto error = juce::String::fromUTF8(errorBuffer.get()).trim();
    return juce::Result::fail(error.isNotEmpty() ? error : "Native audio engine live all-notes-off failed.");
}

juce::Result NativeVstHostSession::startAudioEngine(int startTick)
{
    if (handle == nullptr)
        return juce::Result::fail("Native VST host session is not ready.");

    juce::HeapBlock<char> errorBuffer(static_cast<size_t>(kHostErrorBufferBytes));
    std::memset(errorBuffer.get(), 0, static_cast<size_t>(kHostErrorBufferBytes));

    const auto ok = aims_vst_host_start_audio_engine(handle,
                                                     juce::jmax(0, startTick),
                                                     errorBuffer.get(),
                                                     kHostErrorBufferBytes);
    if (ok != 0)
        return juce::Result::ok();

    const auto error = juce::String::fromUTF8(errorBuffer.get()).trim();
    return juce::Result::fail(error.isNotEmpty() ? error : "Native audio engine start failed.");
}

juce::Result NativeVstHostSession::stopAudioEngine(bool allowTail)
{
    if (handle == nullptr)
        return juce::Result::fail("Native VST host session is not ready.");

    juce::HeapBlock<char> errorBuffer(static_cast<size_t>(kHostErrorBufferBytes));
    std::memset(errorBuffer.get(), 0, static_cast<size_t>(kHostErrorBufferBytes));

    const auto ok = aims_vst_host_stop_audio_engine(handle,
                                                    allowTail ? 1 : 0,
                                                    errorBuffer.get(),
                                                    kHostErrorBufferBytes);
    if (ok != 0)
        return juce::Result::ok();

    const auto error = juce::String::fromUTF8(errorBuffer.get()).trim();
    return juce::Result::fail(error.isNotEmpty() ? error : "Native audio engine stop failed.");
}

void NativeVstHostSession::setEditorStateCallback(std::function<void(bool)> callback)
{
    editorStateCallback = std::move(callback);

    if (handle != nullptr)
    {
        aims_vst_host_set_editor_state_callback(handle,
                                                editorStateCallback != nullptr ? &NativeVstHostSession::editorStateThunk : nullptr,
                                                editorStateCallback != nullptr ? this : nullptr);
    }
}

juce::Result NativeVstHostSession::openAudioEngineTrackEditor(int trackIndex)
{
    if (handle == nullptr)
        return juce::Result::fail("Native VST host session is not ready.");

    juce::HeapBlock<char> errorBuffer(static_cast<size_t>(kHostErrorBufferBytes));
    std::memset(errorBuffer.get(), 0, static_cast<size_t>(kHostErrorBufferBytes));

    const auto ok = aims_vst_host_open_audio_engine_track_editor(handle,
                                                                 trackIndex,
                                                                 errorBuffer.get(),
                                                                 kHostErrorBufferBytes);
    if (ok != 0)
        return juce::Result::ok();

    const auto error = juce::String::fromUTF8(errorBuffer.get()).trim();
    return juce::Result::fail(error.isNotEmpty() ? error : "Native host live track editor command failed.");
}

juce::Result NativeVstHostSession::closeAudioEngineTrackEditor(int trackIndex)
{
    if (handle == nullptr)
        return juce::Result::fail("Native VST host session is not ready.");

    juce::HeapBlock<char> errorBuffer(static_cast<size_t>(kHostErrorBufferBytes));
    std::memset(errorBuffer.get(), 0, static_cast<size_t>(kHostErrorBufferBytes));

    const auto ok = aims_vst_host_close_audio_engine_track_editor(handle,
                                                                  trackIndex,
                                                                  errorBuffer.get(),
                                                                  kHostErrorBufferBytes);
    if (ok != 0)
        return juce::Result::ok();

    const auto error = juce::String::fromUTF8(errorBuffer.get()).trim();
    return juce::Result::fail(error.isNotEmpty() ? error : "Native host close live track editor command failed.");
}

juce::Result NativeVstHostSession::queryAudioEngineTrackEditorOpen(int trackIndex, bool& isOpen) const
{
    isOpen = false;

    if (handle == nullptr)
        return juce::Result::fail("Native VST host session is not ready.");

    juce::HeapBlock<char> errorBuffer(static_cast<size_t>(kHostErrorBufferBytes));
    std::memset(errorBuffer.get(), 0, static_cast<size_t>(kHostErrorBufferBytes));

    int rawOpen = 0;
    const auto ok = aims_vst_host_get_audio_engine_track_editor_state(handle,
                                                                      trackIndex,
                                                                      &rawOpen,
                                                                      errorBuffer.get(),
                                                                      kHostErrorBufferBytes);
    if (ok == 0)
    {
        const auto error = juce::String::fromUTF8(errorBuffer.get()).trim();
        return juce::Result::fail(error.isNotEmpty() ? error : "Native host live track editor state query failed.");
    }

    isOpen = rawOpen != 0;
    return juce::Result::ok();
}

juce::Result NativeVstHostSession::queryTransportSnapshot(TransportSnapshot& snapshot, bool includeTrackPeakLevels) const
{
    snapshot = {};

    if (handle == nullptr)
        return juce::Result::fail("Native VST host session is not ready.");

    juce::HeapBlock<char> errorBuffer(static_cast<size_t>(kHostErrorBufferBytes));
    std::memset(errorBuffer.get(), 0, static_cast<size_t>(kHostErrorBufferBytes));
    std::array<float, 2048> trackPeakLevels {};

    AimsVstHostTransportSnapshot rawSnapshot{};
    const auto ok = aims_vst_host_get_transport_snapshot(handle,
                                                         &rawSnapshot,
                                                         includeTrackPeakLevels ? trackPeakLevels.data() : nullptr,
                                                         includeTrackPeakLevels ? static_cast<int>(trackPeakLevels.size()) : 0,
                                                         errorBuffer.get(),
                                                         kHostErrorBufferBytes);
    if (ok == 0)
    {
        const auto error = juce::String::fromUTF8(errorBuffer.get()).trim();
        return juce::Result::fail(error.isNotEmpty() ? error : "Native host transport snapshot failed.");
    }

    snapshot.sampleRate = juce::jmax(0.0, rawSnapshot.sampleRate);
    snapshot.cpuUsage = juce::jmax(0.0, rawSnapshot.cpuUsage);
    snapshot.inprocessTransportRunning = rawSnapshot.inprocessTransportRunning != 0;
    snapshot.inprocessTransportPositionFrame = static_cast<int64_t>(std::llround(rawSnapshot.inprocessTransportPositionFrame));
    snapshot.audioEngineRunning = rawSnapshot.audioEngineRunning != 0;
    snapshot.audioEngineTailing = rawSnapshot.audioEngineTailing != 0;
    snapshot.audioEnginePositionFrame = static_cast<int64_t>(std::llround(rawSnapshot.audioEnginePositionFrame));
    snapshot.pluginOutputPeakLevel = juce::jlimit(0.0f, 1.0f, rawSnapshot.pluginOutputPeakLevel);
    snapshot.masterPeakLeft = juce::jlimit(0.0f, 1.0f, rawSnapshot.masterPeakLeft);
    snapshot.masterPeakRight = juce::jlimit(0.0f, 1.0f, rawSnapshot.masterPeakRight);
    snapshot.editorOpen = rawSnapshot.editorOpen != 0;
    snapshot.stateGeneration = rawSnapshot.stateGeneration;
    if (includeTrackPeakLevels)
    {
        snapshot.audioEngineTrackPeakLevels.ensureStorageAllocated(rawSnapshot.audioEngineTrackCount);
        for (int index = 0; index < rawSnapshot.audioEngineTrackCount && index < static_cast<int>(trackPeakLevels.size()); ++index)
            snapshot.audioEngineTrackPeakLevels.add(juce::jlimit(0.0f, 1.0f, trackPeakLevels[static_cast<size_t>(index)]));
    }

    if (snapshot.sampleRate > 0.0)
        cachedSampleRate = snapshot.sampleRate;

    return juce::Result::ok();
}

juce::Result NativeVstHostSession::queryAudioEngineTrackParameterSnapshot(int trackIndex,
                                                                          RackParameterSnapshot& snapshot) const
{
    snapshot = {};

    if (handle == nullptr)
        return juce::Result::fail("Native VST host session is not ready.");

    thread_local std::array<char, static_cast<size_t>(kHostErrorBufferBytes)> errorBuffer{};
    errorBuffer.fill('\0');

    constexpr int kMaxParameterEntries = 2048;
    thread_local std::array<AimsVstHostParameterSnapshotEntry, static_cast<size_t>(kMaxParameterEntries)> parameterEntries{};
    std::memset(parameterEntries.data(),
                0,
                sizeof(AimsVstHostParameterSnapshotEntry) * static_cast<size_t>(kMaxParameterEntries));

    AimsVstHostParameterSnapshot rawSnapshot{};
    const auto ok = aims_vst_host_get_audio_engine_track_parameter_snapshot(handle,
                                                                            trackIndex,
                                                                            &rawSnapshot,
                                                                            parameterEntries.data(),
                                                                            kMaxParameterEntries,
                                                                            errorBuffer.data(),
                                                                            kHostErrorBufferBytes);
    if (ok == 0)
    {
        const auto error = juce::String::fromUTF8(errorBuffer.data()).trim();
        return juce::Result::fail(error.isNotEmpty() ? error : "Native host live track parameter snapshot failed.");
    }

    snapshot.editorOpen = rawSnapshot.editorOpen != 0;
    snapshot.stateGeneration = rawSnapshot.stateGeneration;
    for (int index = 0; index < rawSnapshot.parameterCount && index < kMaxParameterEntries; ++index)
    {
        const auto name = juce::String::fromUTF8(parameterEntries[static_cast<size_t>(index)].name).trim();
        if (name.isEmpty())
            continue;

        snapshot.parameterValues.set(juce::Identifier(name),
                                     juce::jlimit(0.0, 100.0,
                                                  static_cast<double>(parameterEntries[static_cast<size_t>(index)].normalizedValue) * 100.0));
    }

    return juce::Result::ok();
}

juce::Result NativeVstHostSession::saveAudioEngineTrackPluginState(int trackIndex, const juce::String& statePath)
{
    if (handle == nullptr)
        return juce::Result::fail("Native VST host session is not ready.");

    juce::HeapBlock<char> errorBuffer(static_cast<size_t>(kHostErrorBufferBytes));
    std::memset(errorBuffer.get(), 0, static_cast<size_t>(kHostErrorBufferBytes));

    const auto ok = aims_vst_host_save_audio_engine_track_plugin_state(handle,
                                                                       trackIndex,
                                                                       statePath.toRawUTF8(),
                                                                       errorBuffer.get(),
                                                                       kHostErrorBufferBytes);
    if (ok != 0)
        return juce::Result::ok();

    const auto error = juce::String::fromUTF8(errorBuffer.get()).trim();
    return juce::Result::fail(error.isNotEmpty() ? error : "Native host live track state save failed.");
}

juce::Result NativeVstHostSession::queryAudioDeviceSnapshot(AudioDeviceSnapshot& snapshot) const
{
    snapshot = {};

    if (handle == nullptr)
        return juce::Result::fail("Native VST host session is not ready.");

    juce::HeapBlock<char> errorBuffer(static_cast<size_t>(kHostErrorBufferBytes));
    std::memset(errorBuffer.get(), 0, static_cast<size_t>(kHostErrorBufferBytes));

    constexpr int kMaxDeviceTypes = 32;
    constexpr int kMaxOutputDevices = 128;
    constexpr int kMaxSampleRates = 64;
    constexpr int kMaxBufferSizes = 64;

    juce::HeapBlock<AimsVstHostStringEntry> deviceTypeEntries(static_cast<size_t>(kMaxDeviceTypes));
    juce::HeapBlock<AimsVstHostStringEntry> outputDeviceEntries(static_cast<size_t>(kMaxOutputDevices));
    juce::HeapBlock<double> sampleRates(static_cast<size_t>(kMaxSampleRates));
    juce::HeapBlock<int> bufferSizes(static_cast<size_t>(kMaxBufferSizes));
    std::memset(deviceTypeEntries.get(), 0, sizeof(AimsVstHostStringEntry) * static_cast<size_t>(kMaxDeviceTypes));
    std::memset(outputDeviceEntries.get(), 0, sizeof(AimsVstHostStringEntry) * static_cast<size_t>(kMaxOutputDevices));
    std::memset(sampleRates.get(), 0, sizeof(double) * static_cast<size_t>(kMaxSampleRates));
    std::memset(bufferSizes.get(), 0, sizeof(int) * static_cast<size_t>(kMaxBufferSizes));

    AimsVstHostAudioDeviceSnapshot rawSnapshot{};
    const auto ok = aims_vst_host_get_audio_device_snapshot(handle,
                                                            &rawSnapshot,
                                                            deviceTypeEntries.get(),
                                                            kMaxDeviceTypes,
                                                            outputDeviceEntries.get(),
                                                            kMaxOutputDevices,
                                                            sampleRates.get(),
                                                            kMaxSampleRates,
                                                            bufferSizes.get(),
                                                            kMaxBufferSizes,
                                                            errorBuffer.get(),
                                                            kHostErrorBufferBytes);
    if (ok == 0)
    {
        const auto error = juce::String::fromUTF8(errorBuffer.get()).trim();
        return juce::Result::fail(error.isNotEmpty() ? error : "Native host audio device snapshot failed.");
    }

    snapshot.audioDeviceType = juce::String::fromUTF8(rawSnapshot.audioDeviceType).trim();
    snapshot.audioDeviceName = juce::String::fromUTF8(rawSnapshot.audioDeviceName).trim();
    snapshot.sampleRate = juce::jmax(0.0, rawSnapshot.sampleRate);
    snapshot.bufferSize = juce::jmax(0, rawSnapshot.bufferSize);

    for (int index = 0; index < rawSnapshot.availableAudioDeviceTypeCount && index < kMaxDeviceTypes; ++index)
    {
        const auto text = juce::String::fromUTF8(deviceTypeEntries[index].value).trim();
        if (text.isNotEmpty())
            snapshot.availableAudioDeviceTypes.add(text);
    }

    for (int index = 0; index < rawSnapshot.availableAudioOutputDeviceCount && index < kMaxOutputDevices; ++index)
    {
        const auto text = juce::String::fromUTF8(outputDeviceEntries[index].value).trim();
        if (text.isNotEmpty())
            snapshot.availableAudioOutputDevices.add(text);
    }

    for (int index = 0; index < rawSnapshot.availableAudioSampleRateCount && index < kMaxSampleRates; ++index)
    {
        const auto value = sampleRates[index];
        if (value > 0.0)
            snapshot.availableAudioSampleRates.add(value);
    }

    for (int index = 0; index < rawSnapshot.availableAudioBufferSizeCount && index < kMaxBufferSizes; ++index)
    {
        const auto value = bufferSizes[index];
        if (value > 0)
            snapshot.availableAudioBufferSizes.add(value);
    }

    if (snapshot.sampleRate > 0.0)
        cachedSampleRate = snapshot.sampleRate;
    if (snapshot.bufferSize > 0)
        cachedBufferSize = snapshot.bufferSize;

    return juce::Result::ok();
}

void NativeVstHostSession::editorStateThunk(int isOpen, void* userData)
{
    auto* session = static_cast<NativeVstHostSession*>(userData);
    if (session == nullptr || session->editorStateCallback == nullptr)
        return;

    session->editorStateCallback(isOpen != 0);
}

juce::Result NativeVstHostSession::runCommand(const juce::var& request, juce::var& response) const
{
    response = juce::var();

    if (handle == nullptr)
        return juce::Result::fail("Native VST host session is not ready.");

    const auto requestText = juce::JSON::toString(request, false);
    juce::HeapBlock<char> responseBuffer(static_cast<size_t>(kHostResponseBufferBytes));
    std::memset(responseBuffer.get(), 0, static_cast<size_t>(kHostResponseBufferBytes));

    const auto ok = aims_vst_host_command(handle,
                                          requestText.toRawUTF8(),
                                          responseBuffer.get(),
                                          kHostResponseBufferBytes);
    if (ok == 0)
        return juce::Result::fail("Native VST host command failed.");

    const auto responseText = juce::String::fromUTF8(responseBuffer.get()).trim();
    response = juce::JSON::parse(responseText);
    if (response.isVoid())
        return juce::Result::fail("Native VST host returned invalid JSON.");

    return juce::Result::ok();
}

} // namespace aims
