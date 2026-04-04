#pragma once

#include "ProjectModel.h"

#include <juce_core/juce_core.h>

#include <functional>

namespace aims
{

class NativeVstHostSession final
{
public:
    struct TransportSnapshot
    {
        double sampleRate = 0.0;
        double cpuUsage = 0.0;
        bool inprocessTransportRunning = false;
        int64_t inprocessTransportPositionFrame = 0;
        bool audioEngineRunning = false;
        bool audioEngineTailing = false;
        int64_t audioEnginePositionFrame = 0;
        float pluginOutputPeakLevel = 0.0f;
        float masterPeakLeft = 0.0f;
        float masterPeakRight = 0.0f;
        bool editorOpen = false;
        int stateGeneration = 0;
        juce::Array<float> audioEngineTrackPeakLevels;
    };

    struct RackParameterSnapshot
    {
        bool editorOpen = false;
        int stateGeneration = 0;
        juce::NamedValueSet parameterValues;
    };

    struct AudioDeviceSnapshot
    {
        juce::String audioDeviceType;
        juce::String audioDeviceName;
        double sampleRate = 0.0;
        int bufferSize = 0;
        juce::StringArray availableAudioDeviceTypes;
        juce::StringArray availableAudioOutputDevices;
        juce::Array<double> availableAudioSampleRates;
        juce::Array<int> availableAudioBufferSizes;
    };

    explicit NativeVstHostSession(bool bridgeModeEnabled = false) noexcept;
    ~NativeVstHostSession();

    juce::Result ensureCreated();
    void destroy();

    bool isReady() const noexcept;

    juce::Result clearPluginGraph();
    juce::Result renderOfflinePluginGraph(const juce::var& tracks, int frames, juce::var& response);
    juce::Result configureAudioDevice(const juce::String& audioDeviceType,
                                      const juce::String& audioDeviceName,
                                      double sampleRate,
                                      int bufferSize);
    juce::Result setAudioEngineState(const ProjectState& project, bool preserveTransport = false);
    juce::Result updateAudioEngineTransport(const ProjectState& project);
    juce::Result setAudioEngineTrackParameters(int trackIndex, const TrackState& track);
    juce::Result setAudioEngineTrackNotes(int trackIndex, const ProjectState& project, const TrackState& track);
    juce::Result setAudioEngineTrackMixState(const ProjectState& project, int trackIndex, const TrackState& track);
    juce::Result noteOnAudioEngineTrack(int trackIndex, int note, int channel, float velocity = 1.0f);
    juce::Result noteOffAudioEngineTrack(int trackIndex, int note, int channel, float velocity = 0.0f);
    juce::Result allNotesOffAudioEngineTrack(int trackIndex, int channel = 0);
    juce::Result startAudioEngine(int startTick);
    juce::Result seekAudioEngine(int startTick);
    juce::Result stopAudioEngine(bool allowTail = false);
    void setEditorStateCallback(std::function<void(bool)> callback);
    juce::Result openAudioEngineTrackEditor(int trackIndex);
    juce::Result openAudioEngineTrackEffectEditor(int trackIndex, int effectIndex);
    juce::Result openAudioEngineSharedEffectBusEditor(const juce::String& busId);
    juce::Result openAudioEngineMasterEffectEditor(int effectIndex);
    juce::Result closeAudioEngineTrackEditor(int trackIndex);
    juce::Result queryAudioEngineTrackEditorOpen(int trackIndex, bool& isOpen) const;
    juce::Result queryTransportSnapshot(TransportSnapshot& snapshot, bool includeTrackPeakLevels = true) const;
    juce::Result queryAudioEngineTrackParameterSnapshot(int trackIndex, RackParameterSnapshot& snapshot) const;
    juce::Result queryAudioDeviceSnapshot(AudioDeviceSnapshot& snapshot) const;
    juce::Result saveAudioEngineTrackPluginState(int trackIndex, const juce::String& statePath);

private:
    static void editorStateThunk(int isOpen, void* userData);
    juce::Result runCommand(const juce::var& request, juce::var& response) const;

    void* handle = nullptr;
    bool bridgeModeEnabled = false;
    std::function<void(bool)> editorStateCallback;
    mutable double cachedSampleRate = 0.0;
    mutable int cachedBufferSize = 0;

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR(NativeVstHostSession)
};

} // namespace aims
