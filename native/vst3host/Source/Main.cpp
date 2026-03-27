#include <juce_audio_devices/juce_audio_devices.h>
#include <juce_audio_formats/juce_audio_formats.h>
#include <juce_audio_processors/juce_audio_processors.h>
#include <juce_audio_utils/juce_audio_utils.h>
#include <juce_gui_extra/juce_gui_extra.h>
#if JUCE_WINDOWS
#include <windows.h>
#include <crtdbg.h>
#endif
#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <mutex>
#include <utility>
#include <vector>

namespace
{
constexpr int kDefaultWindowWidth = 1040;
constexpr int kDefaultWindowHeight = 720;
constexpr int kDefaultMidiChannel = 1;
constexpr int kDefaultCommandPort = 47653;
const juce::String kMainAudioStreamId = "main";
const juce::String kPreviewAudioStreamId = "preview";
const juce::String kLiveMidiAudioStreamId = "live_midi";

juce::String normaliseVst3PathForSettings(const juce::String& path)
{
    auto normalised = path.trim();
    const auto lower = normalised.toLowerCase();
    const auto vst3Index = lower.indexOf(".vst3");

    if (vst3Index >= 0)
        return normalised.substring(0, vst3Index + 5);

    return normalised;
}

juce::var makeResponse(bool ok, const juce::String& message)
{
    auto* object = new juce::DynamicObject();
    object->setProperty("ok", ok);
    object->setProperty("message", message);
    return juce::var(object);
}

void setResponseField(juce::var& response, const juce::Identifier& name, const juce::var& value)
{
    if (auto* object = response.getDynamicObject())
        object->setProperty(name, value);
}

int clampMidiChannel(int channel)
{
    return juce::jlimit(1, 16, channel);
}

int clampMidiNote(int note)
{
    return juce::jlimit(0, 127, note);
}

float clampMidiVelocity(float velocity)
{
    return juce::jlimit(0.0f, 1.0f, velocity);
}

juce::String normaliseQueuedAudioStreamId(const juce::String& value)
{
    const auto streamId = value.trim().toLowerCase();
    if (streamId == kPreviewAudioStreamId || streamId == kLiveMidiAudioStreamId)
        return streamId;
    return kMainAudioStreamId;
}

#if JUCE_WINDOWS
void suppressWindowsCrashDialogs()
{
    const auto currentMode = ::SetErrorMode(0);
    ::SetErrorMode(currentMode | SEM_FAILCRITICALERRORS | SEM_NOGPFAULTERRORBOX | SEM_NOOPENFILEERRORBOX);
    _set_abort_behavior(0, _WRITE_ABORT_MSG | _CALL_REPORTFAULT);
}
#endif
}

class HostComponent;

class HostCommandServer final : private juce::Thread
{
public:
    HostCommandServer(HostComponent& ownerIn, int requestedPortIn);
    ~HostCommandServer() override;

    bool startServer();
    void stopServer();
    int getBoundPort() const noexcept;

private:
    void run() override;
    void handleClient(juce::StreamingSocket& client);

    HostComponent& owner;
    const int requestedPort;
    std::atomic<int> boundPort{0};
    std::unique_ptr<juce::StreamingSocket> listener;
};

class PluginEditorWindow final : public juce::DocumentWindow,
                                 private juce::Timer
{
public:
    PluginEditorWindow(juce::AudioProcessor& processor,
                       const juce::String& key,
                       juce::PropertiesFile& settings)
        : juce::DocumentWindow("VST3 Editor",
                               juce::Colours::black,
                               juce::DocumentWindow::allButtons),
          editorKey(key),
          appSettings(settings)
    {
        setUsingNativeTitleBar(true);
        setResizable(true, true);
        setAlwaysOnTop(true);

        if (auto* editor = processor.createEditorIfNeeded())
        {
            setContentOwned(editor, true);
            setName(processor.getName());
            centreAroundComponent(nullptr, editor->getWidth(), editor->getHeight());
            restoreBounds();
        }
        else
        {
            auto* label = new juce::Label();
            label->setText("This plugin does not expose a native editor.", juce::dontSendNotification);
            label->setJustificationType(juce::Justification::centred);
            setContentOwned(label, true);
            centreWithSize(440, 140);
        }
    }

    ~PluginEditorWindow() override
    {
        stopTimer();
        saveBounds();
    }

    std::function<void()> onWindowClosed;

    void forceTopmostFront()
    {
        setAlwaysOnTop(true);
        setMinimised(false);
        setVisible(true);
        toFront(true);
        grabKeyboardFocus();
    }

    void showBridgeEditorWindow()
    {
        setAlwaysOnTop(true);
        setMinimised(false);
        setVisible(true);
        toFront(true);
        grabKeyboardFocus();
    }

    void beginTopmostWarmup()
    {
        stopTimer();
    }

private:
    void closeButtonPressed() override
    {
        saveBounds();
        setVisible(false);
        auto safeThis = juce::Component::SafePointer<PluginEditorWindow>(this);
        juce::MessageManager::callAsync([safeThis]
        {
            if (safeThis != nullptr && safeThis->onWindowClosed != nullptr)
                safeThis->onWindowClosed();
        });
    }

    void moved() override
    {
        juce::DocumentWindow::moved();
        saveBounds();
    }

    void resized() override
    {
        juce::DocumentWindow::resized();
        saveBounds();
    }

    void activeWindowStatusChanged() override
    {
        juce::DocumentWindow::activeWindowStatusChanged();
    }

    void timerCallback() override
    {
        stopTimer();
    }

    void restoreBounds()
    {
        const auto saved = appSettings.getValue(editorKey + "_bounds");
        if (saved.isNotEmpty())
        {
            const auto rect = juce::Rectangle<int>::fromString(saved);
            if (!rect.isEmpty())
            {
                setBounds(rect);
                return;
            }
        }

        if (auto* desktop = juce::Desktop::getInstance().getDisplays().getPrimaryDisplay())
        {
            const auto area = desktop->userArea;
            setBounds(area.withSizeKeepingCentre(getWidth(), getHeight()));
        }
    }

    void saveBounds() const
    {
        appSettings.setValue(editorKey + "_bounds", getBounds().toString());
    }

    juce::String editorKey;
    juce::PropertiesFile& appSettings;
    int topmostWarmupPassesRemaining = 0;
};

class HostComponent final : public juce::Component,
                            private juce::Button::Listener,
                            private juce::ComboBox::Listener,
                            private juce::AudioIODeviceCallback,
                            private juce::AudioProcessorListener
{
public:
    HostComponent(juce::PropertiesFile& settings,
                  const juce::String& startupPluginPath,
                  const juce::String& startupStatePath,
                  bool restoreLastPluginOnStartup,
                  bool shouldOpenEditorOnStartup,
                  int requestedCommandPort,
                  bool bridgeModeEnabled,
                  double startupSampleRate,
                  int startupBufferSize,
                  const juce::String& startupAudioDeviceType,
                  const juce::String& startupAudioOutputDeviceName)
        : appSettings(settings),
          bridgeMode(bridgeModeEnabled),
          managedStateFile(startupStatePath.isNotEmpty() ? juce::File(startupStatePath) : juce::File()),
          keyboardComponent(keyboardState, juce::MidiKeyboardComponent::horizontalKeyboard)
    {
        formatManager.addFormat(std::make_unique<juce::VST3PluginFormat>());
        audioFormatManager.registerBasicFormats();

        addAndMakeVisible(pathLabel);
        pathLabel.setText("Plugin path", juce::dontSendNotification);
        pathLabel.setColour(juce::Label::textColourId, juce::Colours::whitesmoke);

        addAndMakeVisible(pathEditor);
        pathEditor.setText(appSettings.getValue("last_plugin_path"));

        configureButton(browseButton, "Browse...");
        configureButton(loadButton, "Load");
        configureButton(unloadButton, "Unload");
        configureButton(editorButton, "Open Editor");

        addAndMakeVisible(statusLabel);
        statusLabel.setColour(juce::Label::textColourId, juce::Colours::lightgreen);
        statusLabel.setJustificationType(juce::Justification::centredLeft);
        statusLabel.setText("Ready", juce::dontSendNotification);

        addAndMakeVisible(deviceLabel);
        deviceLabel.setColour(juce::Label::textColourId, juce::Colours::whitesmoke);

        if (!bridgeMode)
        {
            addAndMakeVisible(deviceTypeBox);
            deviceTypeBox.addListener(this);

            addAndMakeVisible(outputDeviceBox);
            outputDeviceBox.addListener(this);
        }

        addAndMakeVisible(sampleRateBox);
        sampleRateBox.addListener(this);

        addAndMakeVisible(bufferSizeBox);
        bufferSizeBox.addListener(this);

        if (!bridgeMode)
        {
            addAndMakeVisible(keyboardComponent);
            keyboardComponent.setAvailableRange(24, 96);
            keyboardComponent.setKeyWidth(18.0f);
            keyboardComponent.setLowestVisibleKey(36);
        }

        setOpaque(true);

        if (!bridgeMode)
        {
            initialiseAudio();
            restoreAudioPreferences(startupSampleRate, startupBufferSize, startupAudioDeviceType, startupAudioOutputDeviceName);
        }
        else
        {
            currentSampleRate = startupSampleRate > 0.0
                ? startupSampleRate
                : appSettings.getDoubleValue("audio_sample_rate", currentSampleRate);
            currentBlockSize = startupBufferSize > 0
                ? startupBufferSize
                : appSettings.getIntValue("audio_buffer_size", currentBlockSize);
            currentSampleRate = currentSampleRate > 0.0 ? currentSampleRate : 44100.0;
            currentBlockSize = juce::jmax(64, currentBlockSize);
        }
        updateDeviceBoxes();
        updateDeviceLabel();
        updateButtons();

        const auto startupPath = startupPluginPath.isNotEmpty()
            ? startupPluginPath
            : (restoreLastPluginOnStartup ? appSettings.getValue("last_plugin_path") : juce::String());
        if (startupPath.isNotEmpty())
        {
            loadPlugin(startupPath);
            if (pluginInstance != nullptr && managedStateFile.existsAsFile())
                loadPluginStateFromFile(managedStateFile);
            if (shouldOpenEditorOnStartup && pluginInstance != nullptr)
                openEditorWindow();
        }

        if (requestedCommandPort > 0)
        {
            commandServer = std::make_unique<HostCommandServer>(*this, requestedCommandPort);
            if (commandServer->startServer())
                statusLabel.setText("Ready  |  Command port " + juce::String(commandServer->getBoundPort()),
                                    juce::dontSendNotification);
            else
                statusLabel.setText("Ready  |  Command server failed", juce::dontSendNotification);
        }
    }

    ~HostComponent() override
    {
        persistManagedState();
        commandServer.reset();
        if (audioInitialised)
        {
            deviceManager.removeAudioCallback(this);
            deviceManager.closeAudioDevice();
        }
        closeGraphEditorWindow();
        clearPersistentPluginGraph();
        clearAudioEngine();
        closeEditorWindow();
        unloadPlugin();
    }

    void paint(juce::Graphics& g) override
    {
        g.fillAll(juce::Colour::fromRGB(18, 20, 25));
        g.setColour(juce::Colour::fromRGB(42, 46, 56));
        g.drawRoundedRectangle(getLocalBounds().toFloat().reduced(6.0f), 10.0f, 1.0f);
    }

    void resized() override
    {
        auto area = getLocalBounds().reduced(14);

        auto pathRow = area.removeFromTop(34);
        pathLabel.setBounds(pathRow.removeFromLeft(86));
        browseButton.setBounds(pathRow.removeFromRight(92));
        loadButton.setBounds(pathRow.removeFromRight(78).reduced(4, 0));
        pathEditor.setBounds(pathRow.reduced(4, 0));

        area.removeFromTop(8);

        auto deviceRow = area.removeFromTop(34);
        if (!bridgeMode)
        {
            deviceTypeBox.setBounds(deviceRow.removeFromLeft(210));
            deviceRow.removeFromLeft(8);
            outputDeviceBox.setBounds(deviceRow.removeFromLeft(250));
            deviceRow.removeFromLeft(8);
            sampleRateBox.setBounds(deviceRow.removeFromLeft(140));
            deviceRow.removeFromLeft(8);
            bufferSizeBox.setBounds(deviceRow.removeFromLeft(140));
            deviceRow.removeFromLeft(8);
            editorButton.setBounds(deviceRow.removeFromLeft(120));
            deviceRow.removeFromLeft(8);
            unloadButton.setBounds(deviceRow.removeFromLeft(90));

            area.removeFromTop(8);
            deviceLabel.setBounds(area.removeFromTop(24));
        }
        else
        {
            deviceLabel.setBounds(deviceRow.removeFromLeft(300));
            deviceRow.removeFromLeft(8);
            sampleRateBox.setBounds(deviceRow.removeFromLeft(140));
            deviceRow.removeFromLeft(8);
            bufferSizeBox.setBounds(deviceRow.removeFromLeft(140));
            deviceRow.removeFromLeft(8);
            editorButton.setBounds(deviceRow.removeFromLeft(120));
            deviceRow.removeFromLeft(8);
            unloadButton.setBounds(deviceRow.removeFromLeft(90));
        }

        area.removeFromTop(8);
        statusLabel.setBounds(area.removeFromTop(24));
        if (!bridgeMode)
        {
            area.removeFromTop(10);
            keyboardComponent.setBounds(area.removeFromTop(120));
        }
    }

    juce::String handleRemoteCommandLine(const juce::String& line)
    {
        const auto request = juce::JSON::parse(line);
        if (request.isVoid())
            return juce::JSON::toString(makeResponse(false, "Invalid JSON command"), true);

        // Performance-critical commands (render_audio, schedule_midi, status,
        // ping) are safe to run directly on the calling thread because they
        // use their own locks (pluginLock, scheduledMidiLock).  Dispatching
        // them through MessageManager::callSync adds ~15-18 ms of thread
        // synchronisation overhead per call on Windows, which is far too
        // slow for realtime audio rendering.
        auto* requestObject = request.getDynamicObject();
        if (requestObject != nullptr)
        {
            const auto command = requestObject->getProperty("command")
                                     .toString().trim().toLowerCase();
            if (command == "render_audio" || command == "render_plugin_graph"
                || command == "schedule_midi"
                || command == "schedule_graph_midi"
                || command == "status" || command == "ping"
                || command == "note_on" || command == "note_off"
                || command == "all_notes_off" || command == "panic"
                || command == "start_graph_transport"
                || command == "stop_graph_transport"
                || command == "set_graph_slot_mix"
                || command == "set_graph_transport_state"
                || command == "set_graph_slot_parameters"
                || command == "load_graph_slot_state"
                || command == "queue_audio" || command == "clear_audio_queue"
                || command == "start_audio_stream"
                || command == "set_audio_engine_transport"
                || command == "set_audio_engine_track_parameters"
                || command == "set_audio_engine_track_state"
                || command == "seek_audio_engine"
                || command == "set_transport_notes"
                || command == "start_inprocess_transport"
                || command == "stop_inprocess_transport")
            {
                const auto result = handleRemoteCommand(request);
                return juce::JSON::toString(result, true);
            }
        }

        // All other commands (load_plugin, open_editor, etc.) touch GUI
        // state and must run on the message thread.
        const auto result = juce::MessageManager::callSync([this, request]() mutable
        {
            return handleRemoteCommand(request);
        });

        if (!result.has_value())
            return juce::JSON::toString(makeResponse(false, "Failed to dispatch command to message thread"), true);

        // Use compact (single-line) JSON so newlines can be used as message
        // delimiters for the persistent TCP connection protocol.
        return juce::JSON::toString(*result, true);
    }

    int getActiveCommandPort() const noexcept
    {
        return commandServer != nullptr ? commandServer->getBoundPort() : 0;
    }

private:
    struct ScheduledMidiEvent
    {
        int64_t frame = 0;
        int priority = 0;
        int64_t sequence = 0;
        int64_t loopEpoch = 0;
        juce::MidiMessage message;
    };

    // In-process transport: the host owns the note list and generates MIDI
    // sample-accurately inside the audio callback, eliminating IPC drift.
    struct TransportNote
    {
        int startTick = 0;
        int endTick = 0;
        int pitch = 60;
        int velocity = 100;
        int channel = 1;
    };

    struct InProcessTransport
    {
        std::vector<TransportNote> notes;
        double bpm = 120.0;
        int ticksPerBeat = 480;
        int64_t positionFrame = 0;      // current playback position in samples
        int64_t loopStartTick = 0;
        int64_t loopEndTick = 0;
        bool loopEnabled = false;
        bool running = false;

        // Convert tick to sample frame (floor division, matching Python side)
        int64_t tickToFrame(int64_t tick, double sampleRate) const
        {
            return static_cast<int64_t>(
                (static_cast<double>(tick) * sampleRate * 60.0)
                / (bpm * static_cast<double>(ticksPerBeat))
            );
        }

        // Convert sample frame to tick (floor)
        int64_t frameToTick(int64_t frame, double sampleRate) const
        {
            return static_cast<int64_t>(
                (static_cast<double>(frame) * bpm * static_cast<double>(ticksPerBeat))
                / (sampleRate * 60.0)
            );
        }

        // Generate MIDI events for a block of numSamples starting at positionFrame.
        // When looping, splits the block at the loop boundary so each sub-block
        // maps cleanly onto the note range with no iteration math drift.
        void generateMidi(juce::MidiBuffer& midi, int numSamples, double sampleRate)
        {
            if (!running || notes.empty())
                return;

            // Compute loop frame boundaries once, consistently from tick values
            const auto lsFrame = tickToFrame(loopStartTick, sampleRate);
            const auto leFrame = tickToFrame(loopEndTick, sampleRate);
            const bool looping = loopEnabled && leFrame > lsFrame;

            int samplesRemaining = numSamples;
            int sampleOffset = 0;  // offset into the output MidiBuffer

            while (samplesRemaining > 0)
            {
                // How many samples until the loop boundary (or end of block)?
                int chunkSamples = samplesRemaining;
                if (looping)
                {
                    const auto framesUntilLoopEnd = leFrame - positionFrame;
                    if (framesUntilLoopEnd > 0 && framesUntilLoopEnd < chunkSamples)
                        chunkSamples = static_cast<int>(framesUntilLoopEnd);
                }

                const auto chunkStart = positionFrame;
                const auto chunkEnd = positionFrame + chunkSamples;

                // Emit MIDI events for notes that fall within [chunkStart, chunkEnd)
                const auto noteOffAdvance = juce::jmax<int64_t>(1, static_cast<int64_t>(sampleRate) / 1000);
                for (const auto& note : notes)
                {
                    const auto nsFrame = tickToFrame(note.startTick, sampleRate);
                    auto neFrame = tickToFrame(note.endTick, sampleRate);
                    neFrame = juce::jmax(nsFrame + 1, neFrame - noteOffAdvance);
                    // Clamp note end just before the loop boundary so the
                    // note-off fires inside the pre-wrap chunk (not on the
                    // exact boundary frame which equals chunkEnd and would
                    // be skipped by the < chunkEnd check).
                    if (looping && nsFrame < leFrame && neFrame >= leFrame)
                        neFrame = juce::jmax(nsFrame + 1, leFrame - 1);

                    if (nsFrame >= chunkStart && nsFrame < chunkEnd)
                    {
                        midi.addEvent(
                            juce::MidiMessage::noteOn(note.channel, note.pitch,
                                static_cast<float>(note.velocity) / 127.0f),
                            sampleOffset + static_cast<int>(nsFrame - chunkStart));
                    }
                    if (neFrame >= chunkStart && neFrame < chunkEnd)
                    {
                        midi.addEvent(
                            juce::MidiMessage::noteOff(note.channel, note.pitch, 0.0f),
                            sampleOffset + static_cast<int>(neFrame - chunkStart));
                    }
                }

                positionFrame += chunkSamples;
                sampleOffset += chunkSamples;
                samplesRemaining -= chunkSamples;

                // Wrap at loop boundary — send explicit noteOff for every note
                // to avoid overlap. CC 123 (allNotesOff) is unreliable across
                // VST plugins, so we send per-note noteOff messages instead.
                if (looping && positionFrame >= leFrame)
                {
                    for (const auto& n : notes)
                    {
                        midi.addEvent(
                            juce::MidiMessage::noteOff(n.channel, n.pitch, 0.0f),
                            juce::jmax(0, sampleOffset - 1));
                    }
                    positionFrame = lsFrame;
                }
            }
        }
    };

    struct BufferedOutputStream
    {
        juce::MemoryBlock data;
        size_t readOffset = 0;
        int channelCount = 2;
        int64_t inputFrames = 0;
        int64_t outputFrames = 0;
        int64_t underruns = 0;
    };

    struct PersistentGraphSlot
    {
        struct GraphEffectSlot
        {
            juce::String name;
            juce::String pluginPath;
            juce::String statePath;
            juce::String parameterSignature;
            std::unique_ptr<juce::AudioPluginInstance> plugin;
        };

        juce::String name;
        juce::String pluginPath;
        juce::String statePath;
        juce::String parameterSignature;
        juce::String fxChainSignature;
        std::unique_ptr<juce::AudioPluginInstance> plugin;
        std::vector<GraphEffectSlot> fxChain;
        juce::Array<ScheduledMidiEvent> scheduledEvents;
        float gain = 1.0f;
        float pan = 0.0f;
        float peakLevel = 0.0f;
    };

    struct OfflineGraphTrack
    {
        juce::String name;
        juce::String pluginPath;
        PersistentGraphSlot* slot = nullptr;
        juce::AudioPluginInstance* plugin = nullptr;
        float gain = 1.0f;
        float pan = 0.0f;
        std::vector<ScheduledMidiEvent> events;
    };

    struct AudioEngineTrackNote
    {
        int startTick = 0;
        int endTick = 0;
        int pitch = 60;
        int velocity = 100;
        int channel = 1;
    };

    enum class AudioEngineTrackKind
    {
        instrument,
        audio
    };

    struct AudioEngineAudioClip
    {
        juce::String path;
        int64_t startFrame = 0;
        juce::AudioBuffer<float> audioData;
    };

    enum class AudioEngineAutomationTargetType
    {
        volume,
        pan,
        outputGainDb,
        vstParameter
    };

    struct AudioEngineAutomationPoint
    {
        int tick = 0;
        float value = 0.0f;
    };

    struct AudioEngineAutomationLane
    {
        AudioEngineAutomationTargetType type = AudioEngineAutomationTargetType::volume;
        juce::String target;
        juce::String parameterName;
        int parameterIndex = -1;
        float defaultValue = 0.0f;
        float lastAppliedValue = std::numeric_limits<float>::quiet_NaN();
        std::vector<AudioEngineAutomationPoint> points;
    };

    struct AudioEngineTrack
    {
        struct EffectSlot
        {
            juce::String name;
            juce::String pluginPath;
            juce::String statePath;
            juce::String parameterSignature;
            std::unique_ptr<juce::AudioPluginInstance> plugin;
        };

        AudioEngineTrackKind kind = AudioEngineTrackKind::instrument;
        juce::String name;
        juce::String pluginPath;
        juce::String statePath;
        juce::String parameterSignature;
        juce::String fxChainSignature;
        std::unique_ptr<juce::AudioPluginInstance> plugin;
        std::vector<EffectSlot> fxChain;
        std::vector<AudioEngineTrackNote> notes;
        std::vector<AudioEngineAudioClip> audioClips;
        std::vector<AudioEngineAutomationLane> automationLanes;
        float baseVolume = 1.0f;
        float basePan = 0.0f;
        float baseOutputGainDb = 0.0f;
        float peakLevel = 0.0f;
        juce::AudioBuffer<float> processBuffer;
        juce::AudioBuffer<float> stereoBuffer;
    };

    struct AudioEngineState
    {
        std::vector<AudioEngineTrack> tracks;
        double bpm = 120.0;
        int ticksPerBeat = 480;
        int64_t positionFrame = 0;
        int64_t loopStartTick = 0;
        int64_t loopEndTick = 0;
        bool loopEnabled = false;
        bool running = false;
        bool bootstrapPending = false;
        bool metronomeEnabled = false;
        juce::AudioBuffer<float> metronomeClickBuffer;
        juce::AudioBuffer<float> metronomeAccentBuffer;

        int64_t tickToFrame(int64_t tick, double sampleRate) const
        {
            return static_cast<int64_t>(
                (static_cast<double>(tick) * sampleRate * 60.0)
                / (bpm * static_cast<double>(ticksPerBeat))
            );
        }

        int64_t frameToTick(int64_t frame, double sampleRate) const
        {
            return static_cast<int64_t>(
                (static_cast<double>(frame) * bpm * static_cast<double>(ticksPerBeat))
                / (sampleRate * 60.0)
            );
        }

        double frameToTickDouble(int64_t frame, double sampleRate) const
        {
            return (
                (static_cast<double>(frame) * bpm * static_cast<double>(ticksPerBeat))
                / (sampleRate * 60.0)
            );
        }
    };

    juce::var handleRemoteCommand(const juce::var& request)
    {
        auto* object = request.getDynamicObject();
        if (object == nullptr)
            return makeResponse(false, "Command payload must be a JSON object");

        const auto command = object->getProperty("command").toString().trim().toLowerCase();
        if (command.isEmpty())
            return makeResponse(false, "Missing command");

        if (command == "ping")
        {
            auto response = makeResponse(true, "pong");
            appendStatusFields(response);
            return response;
        }

        if (command == "status")
        {
            auto response = makeResponse(true, "status");
            appendStatusFields(response);
            return response;
        }

        if (command == "configure_buses")
        {
            if (pluginInstance == nullptr)
                return makeResponse(false, "No plugin loaded");

            juce::String error;
            if (!configureBuses(request, error))
                return makeResponse(false, error.isNotEmpty() ? error : "Could not configure plugin buses");

            auto response = makeResponse(true, "Plugin buses configured");
            appendStatusFields(response);
            return response;
        }

        if (command == "load")
        {
            const auto path = object->getProperty("path").toString().trim();
            if (path.isEmpty())
                return makeResponse(false, "Missing plugin path");

            loadPlugin(path);

            auto response = makeResponse(pluginInstance != nullptr,
                                         pluginInstance != nullptr ? "Plugin loaded" : "Plugin load failed");
            appendStatusFields(response);
            return response;
        }

        if (command == "unload")
        {
            unloadPlugin();
            auto response = makeResponse(true, "Plugin unloaded");
            appendStatusFields(response);
            return response;
        }

        if (command == "load_state")
        {
            if (pluginInstance == nullptr)
                return makeResponse(false, "No plugin loaded");

            const auto path = object->getProperty("path").toString().trim();
            if (path.isEmpty())
                return makeResponse(false, "Missing state path");

            const auto stateFile = juce::File(path);
            if (!stateFile.existsAsFile())
                return makeResponse(false, "State file not found");

            if (!loadPluginStateFromFile(stateFile))
                return makeResponse(false, "Could not load plugin state");

            auto response = makeResponse(true, "Plugin state loaded");
            appendStatusFields(response);
            setResponseField(response, "state_path", stateFile.getFullPathName());
            return response;
        }

        if (command == "save_state")
        {
            if (pluginInstance == nullptr)
                return makeResponse(false, "No plugin loaded");

            const auto path = object->getProperty("path").toString().trim();
            if (path.isEmpty())
                return makeResponse(false, "Missing state path");

            const auto stateFile = juce::File(path);
            if (!savePluginStateToFile(stateFile))
                return makeResponse(false, "Could not save plugin state");

            auto response = makeResponse(true, "Plugin state saved");
            appendStatusFields(response);
            setResponseField(response, "state_path", stateFile.getFullPathName());
            return response;
        }

        if (command == "open_editor")
        {
            if (pluginInstance == nullptr)
                return makeResponse(false, "No plugin loaded");

            juce::MessageManager::callAsync([this]
            {
                if (pluginInstance != nullptr)
                    openEditorWindow();
            });
            auto response = makeResponse(true, "Editor opened");
            appendStatusFields(response);
            return response;
        }

        if (command == "close_editor")
        {
            closeEditorWindow();
            auto response = makeResponse(true, "Editor closed");
            appendStatusFields(response);
            return response;
        }

        if (command == "configure_plugin_graph")
        {
            juce::Array<juce::var> trackErrors;
            juce::String error;
            if (!configurePersistentPluginGraph(object->getProperty("tracks"), trackErrors, error))
            {
                auto response = makeResponse(false, error.isNotEmpty() ? error : "Could not configure plugin graph");
                if (!trackErrors.isEmpty())
                    setResponseField(response, "track_errors", juce::var(trackErrors));
                return response;
            }

            auto response = makeResponse(true, "Plugin graph configured");
            appendStatusFields(response);
            setResponseField(response, "graph_slot_count", static_cast<int>(persistentGraphSlots.size()));
            if (!trackErrors.isEmpty())
                setResponseField(response, "track_errors", juce::var(trackErrors));
            return response;
        }

        if (command == "clear_plugin_graph")
        {
            clearPersistentPluginGraph();
            auto response = makeResponse(true, "Plugin graph cleared");
            appendStatusFields(response);
            return response;
        }

        if (command == "start_graph_transport")
        {
            {
                juce::ScopedLock lock(pluginLock);
                audioEngine.running = false;
                audioEngine.bootstrapPending = false;
                for (auto& slot : persistentGraphSlots)
                    resetPersistentGraphSlotProcessingState(slot);
            }
            graphTransportEnabled.store(true);
            auto response = makeResponse(true, "Graph transport started");
            appendStatusFields(response);
            return response;
        }

        if (command == "stop_graph_transport")
        {
            const auto sendPanic = static_cast<bool>(object->hasProperty("panic")
                ? object->getProperty("panic")
                : juce::var(true));
            {
                juce::ScopedLock lock(pluginLock);
                graphTransportEnabled.store(false);
                for (auto& slot : persistentGraphSlots)
                {
                    if (slot.plugin == nullptr)
                        continue;
                    slot.scheduledEvents.clear();
                    if (sendPanic)
                        panicPersistentGraphSlot(slot, 0);
                    resetPersistentGraphSlotProcessingState(slot);
                }
            }
            auto response = makeResponse(true, "Graph transport stopped");
            appendStatusFields(response);
            return response;
        }

        if (command == "open_graph_slot_editor")
        {
            const auto slotIndex = static_cast<int>(object->hasProperty("slot_index")
                ? object->getProperty("slot_index")
                : juce::var(-1));
            if (!juce::isPositiveAndBelow(slotIndex, static_cast<int>(persistentGraphSlots.size()))
                || persistentGraphSlots[static_cast<size_t>(slotIndex)].plugin == nullptr)
            {
                return makeResponse(false, "Graph slot not loaded");
            }

            juce::MessageManager::callAsync([this, slotIndex]
            {
                openGraphEditorWindow(slotIndex);
            });
            auto response = makeResponse(true, "Graph editor opened");
            appendStatusFields(response);
            setResponseField(response, "graph_editor_slot", slotIndex);
            return response;
        }

        if (command == "close_graph_editor")
        {
            closeGraphEditorWindow();
            auto response = makeResponse(true, "Graph editor closed");
            appendStatusFields(response);
            return response;
        }

        if (command == "note_on")
        {
            const auto note = clampMidiNote(static_cast<int>(object->getProperty("note")));
            const auto channel = clampMidiChannel(static_cast<int>(object->hasProperty("channel")
                                                                       ? object->getProperty("channel")
                                                                       : juce::var(kDefaultMidiChannel)));
            const auto velocity = clampMidiVelocity(static_cast<float>(object->hasProperty("velocity")
                                                                          ? static_cast<double>(object->getProperty("velocity"))
                                                                          : 1.0));

            enqueueMidiMessage(juce::MidiMessage::noteOn(channel, note, velocity));
            auto response = makeResponse(true, "Note on");
            appendStatusFields(response);
            setResponseField(response, "note", note);
            setResponseField(response, "channel", channel);
            return response;
        }

        if (command == "note_off")
        {
            const auto note = clampMidiNote(static_cast<int>(object->getProperty("note")));
            const auto channel = clampMidiChannel(static_cast<int>(object->hasProperty("channel")
                                                                       ? object->getProperty("channel")
                                                                       : juce::var(kDefaultMidiChannel)));
            const auto velocity = clampMidiVelocity(static_cast<float>(object->hasProperty("velocity")
                                                                          ? static_cast<double>(object->getProperty("velocity"))
                                                                          : 0.0));

            enqueueMidiMessage(juce::MidiMessage::noteOff(channel, note, velocity));
            auto response = makeResponse(true, "Note off");
            appendStatusFields(response);
            setResponseField(response, "note", note);
            setResponseField(response, "channel", channel);
            return response;
        }

        if (command == "all_notes_off")
        {
            const auto channel = object->hasProperty("channel")
                ? clampMidiChannel(static_cast<int>(object->getProperty("channel")))
                : 0;
            clearScheduledMidiEvents(channel);
            enqueuePanicMessages(channel);
            auto response = makeResponse(true, "All notes off");
            appendStatusFields(response);
            setResponseField(response, "channel", channel);
            return response;
        }

        if (command == "panic")
        {
            clearScheduledMidiEvents();
            keyboardState.reset();
            enqueuePanicMessages(0);
            auto response = makeResponse(true, "Panic sent");
            appendStatusFields(response);
            return response;
        }

        if (command == "set_output_mix")
        {
            const auto gain = object->hasProperty("gain")
                ? static_cast<float>(static_cast<double>(object->getProperty("gain")))
                : 1.0f;
            const auto pan = object->hasProperty("pan")
                ? static_cast<float>(static_cast<double>(object->getProperty("pan")))
                : 0.0f;
            pluginOutputGain.store(juce::jmax(0.0f, gain));
            pluginOutputPan.store(juce::jlimit(-1.0f, 1.0f, pan));
            auto response = makeResponse(true, "Output mix updated");
            appendStatusFields(response);
            return response;
        }

        if (command == "set_parameters")
        {
            if (pluginInstance == nullptr)
                return makeResponse(false, "No plugin loaded");

            juce::String error;
            int appliedCount = 0;
            juce::StringArray unmatchedParameters;
            if (!applyRequestedParameterValues(object->getProperty("parameters"), error, appliedCount, unmatchedParameters))
                return makeResponse(false, error.isNotEmpty() ? error : "Could not update plugin parameters");

            auto response = makeResponse(true, "Plugin parameters updated");
            appendStatusFields(response);
            setResponseField(response, "applied_parameter_count", appliedCount);
            if (!unmatchedParameters.isEmpty())
            {
                juce::Array<juce::var> unmatched;
                for (const auto& name : unmatchedParameters)
                    unmatched.add(name);
                setResponseField(response, "unmatched_parameters", juce::var(unmatched));
            }
            return response;
        }

        if (command == "set_graph_slot_parameters")
        {
            const auto slotIndex = static_cast<int>(object->hasProperty("slot_index")
                ? object->getProperty("slot_index")
                : juce::var(-1));
            if (!juce::isPositiveAndBelow(slotIndex, static_cast<int>(persistentGraphSlots.size())))
                return makeResponse(false, "Graph slot not loaded");

            auto& slot = persistentGraphSlots[static_cast<size_t>(slotIndex)];
            if (slot.plugin == nullptr)
                return makeResponse(false, "Graph slot not loaded");

            juce::String error;
            int appliedCount = 0;
            juce::StringArray unmatchedParameters;
            {
                juce::ScopedLock lock(pluginLock);
                if (!applyRequestedParameterValuesToPlugin(
                        *slot.plugin,
                        object->getProperty("parameters"),
                        error,
                        appliedCount,
                        unmatchedParameters))
                {
                    return makeResponse(false, error.isNotEmpty() ? error : "Could not update graph slot parameters");
                }
            }

            const auto parameterSignature = object->hasProperty("parameter_signature")
                ? object->getProperty("parameter_signature").toString().trim()
                : parameterPayloadSignature(object->getProperty("parameters"));
            slot.parameterSignature = parameterSignature;
            auto response = makeResponse(true, "Graph slot parameters updated");
            appendStatusFields(response);
            setResponseField(response, "graph_editor_slot", slotIndex);
            setResponseField(response, "applied_parameter_count", appliedCount);
            if (!unmatchedParameters.isEmpty())
            {
                juce::Array<juce::var> unmatched;
                for (const auto& name : unmatchedParameters)
                    unmatched.add(name);
                setResponseField(response, "unmatched_parameters", juce::var(unmatched));
            }
            return response;
        }

        if (command == "load_graph_slot_state")
        {
            const auto slotIndex = static_cast<int>(object->hasProperty("slot_index")
                ? object->getProperty("slot_index")
                : juce::var(-1));
            if (!juce::isPositiveAndBelow(slotIndex, static_cast<int>(persistentGraphSlots.size())))
                return makeResponse(false, "Graph slot not loaded");

            auto& slot = persistentGraphSlots[static_cast<size_t>(slotIndex)];
            if (slot.plugin == nullptr)
                return makeResponse(false, "Graph slot not loaded");

            const auto path = object->getProperty("path").toString().trim();
            if (path.isEmpty())
                return makeResponse(false, "Missing state path");

            const auto stateFile = juce::File(path);
            if (!stateFile.existsAsFile())
                return makeResponse(false, "State file not found");

            juce::String error;
            int appliedCount = 0;
            juce::StringArray unmatchedParameters;
            {
                juce::ScopedLock lock(pluginLock);
                clearPersistentGraphScheduledEvents(slot, 0);
                panicPersistentGraphSlot(slot, 0);
                resetPersistentGraphSlotProcessingState(slot);
                if (!loadPluginStateIntoInstance(*slot.plugin, stateFile))
                    return makeResponse(false, "Could not load graph slot state");

                if (object->hasProperty("parameters") && object->getProperty("parameters").isObject())
                {
                    if (!applyRequestedParameterValuesToPlugin(
                            *slot.plugin,
                            object->getProperty("parameters"),
                            error,
                            appliedCount,
                            unmatchedParameters))
                    {
                        return makeResponse(false, error.isNotEmpty() ? error : "Could not update graph slot parameters");
                    }
                    const auto parameterSignature = object->hasProperty("parameter_signature")
                        ? object->getProperty("parameter_signature").toString().trim()
                        : parameterPayloadSignature(object->getProperty("parameters"));
                    slot.parameterSignature = parameterSignature;
                }

                slot.statePath = stateFile.getFullPathName();
            }

            auto response = makeResponse(true, "Graph slot state loaded");
            appendStatusFields(response);
            setResponseField(response, "graph_editor_slot", slotIndex);
            setResponseField(response, "state_path", stateFile.getFullPathName());
            setResponseField(response, "applied_parameter_count", appliedCount);
            if (!unmatchedParameters.isEmpty())
            {
                juce::Array<juce::var> unmatched;
                for (const auto& name : unmatchedParameters)
                    unmatched.add(name);
                setResponseField(response, "unmatched_parameters", juce::var(unmatched));
            }
            return response;
        }

        if (command == "set_graph_slot_mix")
        {
            const auto slotIndex = static_cast<int>(object->hasProperty("slot_index")
                ? object->getProperty("slot_index")
                : juce::var(-1));
            if (!juce::isPositiveAndBelow(slotIndex, static_cast<int>(persistentGraphSlots.size())))
                return makeResponse(false, "Graph slot not loaded");

            auto& slot = persistentGraphSlots[static_cast<size_t>(slotIndex)];
            if (slot.plugin == nullptr)
                return makeResponse(false, "Graph slot not loaded");

            const auto gain = object->hasProperty("gain")
                ? static_cast<float>(static_cast<double>(object->getProperty("gain")))
                : slot.gain;
            const auto pan = object->hasProperty("pan")
                ? static_cast<float>(static_cast<double>(object->getProperty("pan")))
                : slot.pan;

            {
                juce::ScopedLock lock(pluginLock);
                slot.gain = juce::jmax(0.0f, gain);
                slot.pan = juce::jlimit(-1.0f, 1.0f, pan);
            }

            auto response = makeResponse(true, "Graph slot mix updated");
            appendStatusFields(response);
            setResponseField(response, "graph_editor_slot", slotIndex);
            setResponseField(response, "gain", static_cast<double>(slot.gain));
            setResponseField(response, "pan", static_cast<double>(slot.pan));
            return response;
        }

        if (command == "set_graph_transport_state")
        {
            const auto slotMixesVar = object->getProperty("slot_mixes");
            const auto resetSlotsVar = object->getProperty("reset_slots");
            const auto baseOffsetFrames = juce::jmax<int64_t>(
                0,
                static_cast<int64_t>(object->hasProperty("base_offset_frames")
                    ? static_cast<double>(object->getProperty("base_offset_frames"))
                    : 0.0)
            );

            juce::String error;
            int updatedMixCount = 0;
            int scheduledCount = 0;
            {
                juce::ScopedLock lock(pluginLock);
                if (slotMixesVar.isArray())
                {
                    auto* slotMixesArray = slotMixesVar.getArray();
                    for (const auto& slotVar : *slotMixesArray)
                    {
                        auto* slotObject = slotVar.getDynamicObject();
                        if (slotObject == nullptr)
                            continue;

                        const auto slotIndex = static_cast<int>(slotObject->hasProperty("slot_index")
                            ? slotObject->getProperty("slot_index")
                            : juce::var(-1));
                        if (!juce::isPositiveAndBelow(slotIndex, static_cast<int>(persistentGraphSlots.size())))
                            return makeResponse(false, "Graph slot not loaded");

                        auto& slot = persistentGraphSlots[static_cast<size_t>(slotIndex)];
                        if (slot.plugin == nullptr)
                            return makeResponse(false, "Graph slot not loaded");

                        const auto gain = slotObject->hasProperty("gain")
                            ? static_cast<float>(static_cast<double>(slotObject->getProperty("gain")))
                            : slot.gain;
                        const auto pan = slotObject->hasProperty("pan")
                            ? static_cast<float>(static_cast<double>(slotObject->getProperty("pan")))
                            : slot.pan;

                        slot.gain = juce::jmax(0.0f, gain);
                        slot.pan = juce::jlimit(-1.0f, 1.0f, pan);
                        ++updatedMixCount;
                    }
                }

                if (resetSlotsVar.isArray())
                {
                    if (!schedulePersistentGraphMidi(resetSlotsVar, baseOffsetFrames, scheduledCount, error))
                        return makeResponse(false, error.isNotEmpty() ? error : "Could not update graph transport state");
                }
            }

            auto response = makeResponse(true, "Graph transport state updated");
            appendStatusFields(response);
            setResponseField(response, "updated_slot_mix_count", updatedMixCount);
            setResponseField(response, "scheduled_count", scheduledCount);
            setResponseField(response, "base_offset_frames", static_cast<double>(baseOffsetFrames));
            return response;
        }

        if (command == "schedule_graph_midi")
        {
            const auto baseOffsetFrames = juce::jmax<int64_t>(
                0,
                static_cast<int64_t>(object->hasProperty("base_offset_frames")
                    ? static_cast<double>(object->getProperty("base_offset_frames"))
                    : 0.0)
            );
            const auto slotsVar = object->getProperty("slots");
            if (!slotsVar.isArray())
                return makeResponse(false, "schedule_graph_midi requires a slots array");

            juce::String error;
            int scheduledCount = 0;
            {
                juce::ScopedLock lock(pluginLock);
                if (!schedulePersistentGraphMidi(slotsVar, baseOffsetFrames, scheduledCount, error))
                    return makeResponse(false, error.isNotEmpty() ? error : "Could not schedule graph MIDI");
            }

            auto response = makeResponse(true, "Scheduled graph MIDI events");
            appendStatusFields(response);
            setResponseField(response, "scheduled_count", scheduledCount);
            setResponseField(response, "base_offset_frames", static_cast<double>(baseOffsetFrames));
            return response;
        }

        if (command == "queue_audio")
        {
            const auto audioBase64 = object->getProperty("audio_b64").toString();
            if (audioBase64.isEmpty())
                return makeResponse(false, "queue_audio requires audio_b64");
            const auto streamId = normaliseQueuedAudioStreamId(
                object->hasProperty("stream_id")
                    ? object->getProperty("stream_id").toString()
                    : kMainAudioStreamId
            );

            const auto format = object->hasProperty("format")
                ? object->getProperty("format").toString().trim().toLowerCase()
                : juce::String("f32le-interleaved");
            if (format.isNotEmpty() && format != "f32le-interleaved")
                return makeResponse(false, "queue_audio currently supports only f32le-interleaved audio");

            const auto channels = juce::jmax(
                1,
                static_cast<int>(object->hasProperty("channels")
                    ? object->getProperty("channels")
                    : juce::var(2))
            );

            juce::String error;
            int appendedFrames = 0;
            if (!queueStreamedAudio(streamId, audioBase64, channels, error, appendedFrames))
                return makeResponse(false, error.isNotEmpty() ? error : "Could not queue audio");

            auto response = makeResponse(true, "Queued audio");
            appendStatusFields(response);
            setResponseField(response, "stream_id", streamId);
            setResponseField(response, "appended_audio_frames", appendedFrames);
            return response;
        }

        if (command == "clear_audio_queue")
        {
            const auto streamId = normaliseQueuedAudioStreamId(
                object->hasProperty("stream_id")
                    ? object->getProperty("stream_id").toString()
                    : kMainAudioStreamId
            );
            const auto resetCounters = static_cast<bool>(object->hasProperty("reset_counters")
                ? object->getProperty("reset_counters")
                : juce::var(false));
            clearStreamedAudioQueue(streamId, resetCounters);
            auto response = makeResponse(true, "Cleared audio queue");
            appendStatusFields(response);
            setResponseField(response, "stream_id", streamId);
            return response;
        }

        if (command == "start_audio_stream")
        {
            const auto streamId = normaliseQueuedAudioStreamId(
                object->hasProperty("stream_id")
                    ? object->getProperty("stream_id").toString()
                    : kMainAudioStreamId
            );
            if (streamId == kMainAudioStreamId)
                audioStreamEnabled.store(true);
            auto response = makeResponse(true, "Audio stream started");
            appendStatusFields(response);
            setResponseField(response, "stream_id", streamId);
            return response;
        }

        if (command == "set_transport_notes")
        {
            const auto bpm = object->hasProperty("bpm")
                ? static_cast<double>(object->getProperty("bpm")) : 120.0;
            const auto ticksPerBeat = object->hasProperty("ticks_per_beat")
                ? static_cast<int>(object->getProperty("ticks_per_beat")) : 480;
            const auto loopEnabled = object->hasProperty("loop_enabled")
                ? static_cast<bool>(object->getProperty("loop_enabled")) : false;
            const auto loopStartTick = object->hasProperty("loop_start_tick")
                ? static_cast<int64_t>(static_cast<double>(object->getProperty("loop_start_tick"))) : int64_t(0);
            const auto loopEndTick = object->hasProperty("loop_end_tick")
                ? static_cast<int64_t>(static_cast<double>(object->getProperty("loop_end_tick"))) : int64_t(0);
            const auto notesVar = object->getProperty("notes");

            std::vector<TransportNote> notes;
            if (auto* notesArray = notesVar.getArray())
            {
                notes.reserve(static_cast<size_t>(notesArray->size()));
                for (const auto& noteVar : *notesArray)
                {
                    auto* noteObj = noteVar.getDynamicObject();
                    if (noteObj == nullptr) continue;
                    TransportNote note;
                    note.startTick = static_cast<int>(noteObj->getProperty("start_tick"));
                    note.endTick = static_cast<int>(noteObj->getProperty("end_tick"));
                    note.pitch = clampMidiNote(static_cast<int>(noteObj->getProperty("pitch")));
                    note.velocity = juce::jlimit(1, 127, static_cast<int>(noteObj->getProperty("velocity")));
                    note.channel = clampMidiChannel(
                        noteObj->hasProperty("channel")
                            ? static_cast<int>(noteObj->getProperty("channel"))
                            : kDefaultMidiChannel);
                    notes.push_back(note);
                }
            }

            {
                juce::ScopedLock lock(pluginLock);
                inProcessTransport.notes = std::move(notes);
                inProcessTransport.bpm = juce::jmax(1.0, bpm);
                inProcessTransport.ticksPerBeat = juce::jmax(1, ticksPerBeat);
                inProcessTransport.loopEnabled = loopEnabled;
                inProcessTransport.loopStartTick = loopStartTick;
                inProcessTransport.loopEndTick = loopEndTick;
            }

            auto response = makeResponse(true, "Transport notes updated");
            appendStatusFields(response);
            setResponseField(response, "note_count", static_cast<int>(inProcessTransport.notes.size()));
            return response;
        }

        if (command == "start_inprocess_transport")
        {
            const auto startTick = object->hasProperty("start_tick")
                ? static_cast<int64_t>(static_cast<double>(object->getProperty("start_tick")))
                : int64_t(0);

            {
                juce::ScopedLock lock(pluginLock);
                inProcessTransport.positionFrame = inProcessTransport.tickToFrame(
                    startTick, currentSampleRate);
                inProcessTransport.running = true;
                audioEngine.running = false;
                audioEngine.bootstrapPending = false;
                // Clear any leftover scheduled events from the old IPC-based path
                clearScheduledMidiEvents();
                renderedSampleFrames.store(0);
            }

            auto response = makeResponse(true, "In-process transport started");
            appendStatusFields(response);
            setResponseField(response, "start_tick", static_cast<double>(startTick));
            setResponseField(response, "position_frame", static_cast<double>(inProcessTransport.positionFrame));
            return response;
        }

        if (command == "stop_inprocess_transport")
        {
            {
                juce::ScopedLock lock(pluginLock);
                inProcessTransport.running = false;
                enqueuePanicMessages(0);
            }

            auto response = makeResponse(true, "In-process transport stopped");
            appendStatusFields(response);
            return response;
        }

        if (command == "set_audio_engine_state")
        {
            const auto bpm = object->hasProperty("bpm")
                ? static_cast<double>(object->getProperty("bpm")) : 120.0;
            const auto ticksPerBeat = object->hasProperty("ticks_per_beat")
                ? static_cast<int>(object->getProperty("ticks_per_beat")) : 480;
            const auto loopEnabled = object->hasProperty("loop_enabled")
                ? static_cast<bool>(object->getProperty("loop_enabled")) : false;
            const auto loopStartTick = object->hasProperty("loop_start_tick")
                ? static_cast<int64_t>(static_cast<double>(object->getProperty("loop_start_tick"))) : int64_t(0);
            const auto loopEndTick = object->hasProperty("loop_end_tick")
                ? static_cast<int64_t>(static_cast<double>(object->getProperty("loop_end_tick"))) : int64_t(0);
            const auto metronomeEnabled = object->hasProperty("metronome_enabled")
                ? static_cast<bool>(object->getProperty("metronome_enabled")) : false;

            juce::Array<juce::var> trackErrors;
            juce::String error;
            {
                juce::ScopedLock lock(pluginLock);
                if (!configureAudioEngine(
                        object->getProperty("tracks"),
                        bpm,
                        ticksPerBeat,
                        loopEnabled,
                        loopStartTick,
                        loopEndTick,
                        metronomeEnabled,
                        trackErrors,
                        error))
                {
                    auto failedResponse = makeResponse(false, error.isNotEmpty() ? error : "Could not configure audio engine");
                    if (!trackErrors.isEmpty())
                        setResponseField(failedResponse, "track_errors", juce::var(trackErrors));
                    return failedResponse;
                }
            }

            auto response = makeResponse(true, "Audio engine state updated");
            appendStatusFields(response);
            setResponseField(response, "track_count", static_cast<int>(audioEngine.tracks.size()));
            if (!trackErrors.isEmpty())
                setResponseField(response, "track_errors", juce::var(trackErrors));
            return response;
        }

        if (command == "set_audio_engine_transport")
        {
            const auto bpm = object->hasProperty("bpm")
                ? static_cast<double>(object->getProperty("bpm")) : audioEngine.bpm;
            const auto ticksPerBeat = object->hasProperty("ticks_per_beat")
                ? static_cast<int>(object->getProperty("ticks_per_beat")) : audioEngine.ticksPerBeat;
            const auto loopEnabled = object->hasProperty("loop_enabled")
                ? static_cast<bool>(object->getProperty("loop_enabled")) : audioEngine.loopEnabled;
            const auto loopStartTick = object->hasProperty("loop_start_tick")
                ? static_cast<int64_t>(static_cast<double>(object->getProperty("loop_start_tick"))) : audioEngine.loopStartTick;
            const auto loopEndTick = object->hasProperty("loop_end_tick")
                ? static_cast<int64_t>(static_cast<double>(object->getProperty("loop_end_tick"))) : audioEngine.loopEndTick;
            const auto metronomeEnabled = object->hasProperty("metronome_enabled")
                ? static_cast<bool>(object->getProperty("metronome_enabled")) : audioEngine.metronomeEnabled;

            {
                juce::ScopedLock lock(pluginLock);
                const auto sampleRate = juce::jmax(1.0, currentSampleRate);
                const auto preservedTick = audioEngine.frameToTickDouble(audioEngine.positionFrame, sampleRate);

                audioEngine.bpm = juce::jmax(1.0, bpm);
                audioEngine.ticksPerBeat = juce::jmax(1, ticksPerBeat);
                audioEngine.loopEnabled = loopEnabled;
                audioEngine.loopStartTick = loopStartTick;
                audioEngine.loopEndTick = loopEndTick;
                audioEngine.metronomeEnabled = metronomeEnabled;

                if (currentSampleRate > 0.0 && currentBlockSize > 0)
                    prepareAudioEngineMetronome();

                auto targetFrame = audioEngine.tickToFrame(static_cast<int64_t>(std::llround(preservedTick)), sampleRate);
                const auto loopStartFrame = audioEngine.tickToFrame(audioEngine.loopStartTick, sampleRate);
                const auto loopEndFrame = audioEngine.tickToFrame(audioEngine.loopEndTick, sampleRate);
                if (audioEngine.loopEnabled && loopEndFrame > loopStartFrame
                    && (targetFrame < loopStartFrame || targetFrame >= loopEndFrame))
                {
                    targetFrame = loopStartFrame;
                }

                audioEngine.positionFrame = juce::jmax<int64_t>(0, targetFrame);
                if (audioEngine.running)
                {
                    resetAudioEngineProcessingState();
                    audioEngine.bootstrapPending = true;
                }
            }

            auto response = makeResponse(true, "Audio engine transport updated");
            appendStatusFields(response);
            return response;
        }

        if (command == "set_audio_engine_track_parameters")
        {
            const auto trackIndex = static_cast<int>(object->hasProperty("track_index")
                ? object->getProperty("track_index") : juce::var(-1));

            juce::ScopedLock lock(pluginLock);
            if (!juce::isPositiveAndBelow(trackIndex, static_cast<int>(audioEngine.tracks.size())))
                return makeResponse(false, "Invalid audio engine track index");

            auto& track = audioEngine.tracks[static_cast<size_t>(trackIndex)];
            if (track.plugin == nullptr)
                return makeResponse(false, "Audio engine track has no plugin loaded");

            juce::String error;
            int appliedCount = 0;
            juce::StringArray unmatchedParameters;
            if (!applyRequestedParameterValuesToPlugin(
                    *track.plugin,
                    object->getProperty("parameters"),
                    error, appliedCount, unmatchedParameters))
                return makeResponse(false, error.isNotEmpty() ? error : "Could not update track parameters");

            auto response = makeResponse(true, "Audio engine track parameters updated");
            appendStatusFields(response);
            setResponseField(response, "applied_parameter_count", appliedCount);
            return response;
        }

        if (command == "set_audio_engine_track_state")
        {
            const auto trackIndex = static_cast<int>(object->hasProperty("track_index")
                ? object->getProperty("track_index") : juce::var(-1));
            const auto path = object->getProperty("path").toString().trim();

            juce::ScopedLock lock(pluginLock);
            if (!juce::isPositiveAndBelow(trackIndex, static_cast<int>(audioEngine.tracks.size())))
                return makeResponse(false, "Invalid audio engine track index");

            auto& track = audioEngine.tracks[static_cast<size_t>(trackIndex)];
            if (track.plugin == nullptr)
                return makeResponse(false, "Audio engine track has no plugin loaded");

            if (path.isEmpty())
                return makeResponse(false, "Missing state path");

            const auto stateFile = juce::File(path);
            if (!stateFile.existsAsFile())
                return makeResponse(false, "State file not found");

            if (!loadPluginStateIntoInstance(*track.plugin, stateFile))
                return makeResponse(false, "Could not load plugin state into audio engine track");

            track.statePath = path;
            track.parameterSignature = {};

            // Apply parameter overlay if provided.
            if (object->hasProperty("parameters"))
            {
                juce::String error;
                int appliedCount = 0;
                juce::StringArray unmatchedParameters;
                applyRequestedParameterValuesToPlugin(
                    *track.plugin,
                    object->getProperty("parameters"),
                    error, appliedCount, unmatchedParameters);
                track.parameterSignature = parameterPayloadSignature(object->getProperty("parameters"));
            }

            auto response = makeResponse(true, "Audio engine track state loaded");
            appendStatusFields(response);
            setResponseField(response, "state_path", stateFile.getFullPathName());
            return response;
        }

        if (command == "seek_audio_engine")
        {
            const auto startTick = object->hasProperty("start_tick")
                ? static_cast<int64_t>(static_cast<double>(object->getProperty("start_tick")))
                : int64_t(0);

            {
                juce::ScopedLock lock(pluginLock);
                if (currentSampleRate <= 0.0 || currentBlockSize <= 0)
                    return makeResponse(false, "Audio device is not ready");

                auto targetFrame = audioEngine.tickToFrame(startTick, currentSampleRate);
                const auto loopStartFrame = audioEngine.tickToFrame(audioEngine.loopStartTick, currentSampleRate);
                const auto loopEndFrame = audioEngine.tickToFrame(audioEngine.loopEndTick, currentSampleRate);
                if (audioEngine.loopEnabled && loopEndFrame > loopStartFrame
                    && (targetFrame < loopStartFrame || targetFrame >= loopEndFrame))
                {
                    targetFrame = loopStartFrame;
                }

                audioEngine.positionFrame = juce::jmax<int64_t>(0, targetFrame);
                resetAudioEngineProcessingState();
                audioEngine.bootstrapPending = audioEngine.running;
            }

            auto response = makeResponse(true, "Audio engine seek updated");
            appendStatusFields(response);
            setResponseField(response, "start_tick", static_cast<double>(startTick));
            setResponseField(response, "position_frame", static_cast<double>(audioEngine.positionFrame));
            return response;
        }

        if (command == "start_audio_engine")
        {
            const auto startTick = object->hasProperty("start_tick")
                ? static_cast<int64_t>(static_cast<double>(object->getProperty("start_tick")))
                : int64_t(0);

            {
                juce::ScopedLock lock(pluginLock);
                if (audioEngine.tracks.empty() && !audioEngine.metronomeEnabled)
                    return makeResponse(false, "Audio engine has no tracks");
                if (currentSampleRate <= 0.0 || currentBlockSize <= 0)
                    return makeResponse(false, "Audio device is not ready");

                const auto loopStartFrame = audioEngine.tickToFrame(audioEngine.loopStartTick, currentSampleRate);
                const auto loopEndFrame = audioEngine.tickToFrame(audioEngine.loopEndTick, currentSampleRate);
                auto targetFrame = audioEngine.tickToFrame(startTick, currentSampleRate);
                if (audioEngine.loopEnabled && loopEndFrame > loopStartFrame
                    && (targetFrame < loopStartFrame || targetFrame >= loopEndFrame))
                {
                    targetFrame = loopStartFrame;
                }

                graphTransportEnabled.store(false);
                inProcessTransport.running = false;
                clearScheduledMidiEvents();
                resetAudioEngineProcessingState();
                audioEngine.positionFrame = juce::jmax<int64_t>(0, targetFrame);
                audioEngine.running = true;
                audioEngine.bootstrapPending = true;
                renderedSampleFrames.store(0);
            }

            auto response = makeResponse(true, "Audio engine started");
            appendStatusFields(response);
            setResponseField(response, "start_tick", static_cast<double>(startTick));
            setResponseField(response, "position_frame", static_cast<double>(audioEngine.positionFrame));
            return response;
        }

        if (command == "stop_audio_engine")
        {
            {
                juce::ScopedLock lock(pluginLock);
                audioEngine.running = false;
                audioEngine.bootstrapPending = false;
                resetAudioEngineProcessingState();
            }

            auto response = makeResponse(true, "Audio engine stopped");
            appendStatusFields(response);
            return response;
        }

        if (command == "schedule_midi")
        {
            const auto baseOffsetFrames = juce::jmax<int64_t>(
                0,
                static_cast<int64_t>(object->hasProperty("base_offset_frames")
                    ? static_cast<double>(object->getProperty("base_offset_frames"))
                    : 0.0)
            );
            const auto loopEpoch = juce::jmax<int64_t>(
                0,
                static_cast<int64_t>(object->hasProperty("loop_epoch")
                    ? static_cast<double>(object->getProperty("loop_epoch"))
                    : 0.0)
            );
            const auto clearChannelsVar = object->getProperty("clear_channels");
            const auto resetChannelsVar = object->getProperty("reset_channels");
            const auto eventsVar = object->getProperty("events");
            if (!eventsVar.isArray())
                return makeResponse(false, "schedule_midi requires an events array");
            const auto currentFrame = renderedSampleFrames.load();
            const auto resetFrame = currentFrame + baseOffsetFrames;
            juce::Array<int> resetChannels;

            if (clearChannelsVar.isArray())
            {
                if (auto* clearArray = clearChannelsVar.getArray())
                {
                    for (const auto& channelVar : *clearArray)
                    {
                        const auto channel = clampMidiChannel(static_cast<int>(channelVar));
                        clearScheduledMidiEvents(channel, loopEpoch, resetFrame);
                    }
                }
            }

            if (resetChannelsVar.isArray())
            {
                if (auto* resetArray = resetChannelsVar.getArray())
                {
                    for (const auto& channelVar : *resetArray)
                    {
                        const auto channel = clampMidiChannel(static_cast<int>(channelVar));
                        clearScheduledMidiEvents(channel, loopEpoch, resetFrame);
                        resetChannels.addIfNotAlreadyThere(channel);
                    }
                }
            }

            int scheduledCount = 0;
            if (auto* array = eventsVar.getArray())
            {
                juce::ScopedLock lock(scheduledMidiLock);
                for (const auto channel : resetChannels)
                    scheduledCount += appendScheduledPanicEvents(scheduledMidiEvents, channel, resetFrame, loopEpoch);
                for (const auto& eventVar : *array)
                {
                    auto* eventObject = eventVar.getDynamicObject();
                    if (eventObject == nullptr)
                        continue;

                    const auto type = eventObject->getProperty("type").toString().trim().toLowerCase();
                    const auto channel = clampMidiChannel(static_cast<int>(eventObject->hasProperty("channel")
                                                                               ? eventObject->getProperty("channel")
                                                                               : juce::var(kDefaultMidiChannel)));
                    const auto note = clampMidiNote(static_cast<int>(eventObject->hasProperty("note")
                                                                         ? eventObject->getProperty("note")
                                                                         : juce::var(60)));
                    const auto velocity = clampMidiVelocity(static_cast<float>(eventObject->hasProperty("velocity")
                                                                                  ? static_cast<double>(eventObject->getProperty("velocity"))
                                                                                  : 0.0));
                    const auto controller = juce::jlimit(
                        0,
                        127,
                        static_cast<int>(eventObject->hasProperty("control")
                            ? eventObject->getProperty("control")
                            : juce::var(0))
                    );
                    const auto controllerValue = juce::jlimit(
                        0,
                        127,
                        static_cast<int>(eventObject->hasProperty("value")
                            ? eventObject->getProperty("value")
                            : juce::var(0))
                    );
                    const auto sampleOffset = juce::jmax<int64_t>(
                        0,
                        static_cast<int64_t>(eventObject->hasProperty("sample_offset")
                            ? static_cast<double>(eventObject->getProperty("sample_offset"))
                            : 0.0)
                    );
                    const auto priority = static_cast<int>(eventObject->hasProperty("priority")
                        ? eventObject->getProperty("priority")
                        : juce::var(0));

                    juce::MidiMessage message;
                    if (type == "note_on")
                        message = juce::MidiMessage::noteOn(channel, note, velocity);
                    else if (type == "note_off")
                        message = juce::MidiMessage::noteOff(channel, note, velocity);
                    else if (type == "control_change")
                        message = juce::MidiMessage::controllerEvent(channel, controller, controllerValue);
                    else
                        continue;

                    const auto targetFrame = currentFrame + baseOffsetFrames + sampleOffset;
                    scheduledMidiEvents.add({
                        targetFrame,
                        priority,
                        scheduledMidiSequence.fetch_add(1),
                        loopEpoch,
                        message,
                    });
                    ++scheduledCount;
                }
            }

            auto response = makeResponse(true, "Scheduled MIDI events");
            appendStatusFields(response);
            setResponseField(response, "scheduled_count", scheduledCount);
            setResponseField(response, "base_offset_frames", static_cast<double>(baseOffsetFrames));
            return response;
        }

        if (command == "render_audio")
        {
            if (pluginInstance == nullptr)
                return makeResponse(false, "No plugin loaded");

            const auto requestedFrames = static_cast<int>(object->hasProperty("frames")
                ? object->getProperty("frames")
                : juce::var(currentBlockSize));
            const auto frames = juce::jlimit(1, 4096, requestedFrames);
            auto response = renderOfflineAudioBlock(frames, object->getProperty("input_buses"));
            setResponseField(response, "plugin_loaded", pluginInstance != nullptr);
            setResponseField(response, "plugin_name", pluginDescription.name);
            setResponseField(response, "plugin_path",
                             normaliseVst3PathForSettings(pathEditor.getText().trim()));
            setResponseField(response, "editor_open", editorWindow != nullptr);
            setResponseField(response, "sample_rate", currentSampleRate);
            setResponseField(response, "buffer_size", currentBlockSize);
            setResponseField(response, "command_port", getActiveCommandPort());
            setResponseField(response, "bridge_mode", bridgeMode);
            setResponseField(response, "input_buses", describeBuses(true));
            setResponseField(response, "frames", frames);
            return response;
        }

        if (command == "render_plugin_graph")
        {
            const auto requestedFrames = static_cast<int>(object->hasProperty("frames")
                ? object->getProperty("frames")
                : juce::var(currentBlockSize));
            const auto frames = juce::jlimit(1, 262144, requestedFrames);
            auto response = renderOfflinePluginGraph(object->getProperty("tracks"), frames);
            setResponseField(response, "sample_rate", currentSampleRate);
            setResponseField(response, "buffer_size", currentBlockSize);
            setResponseField(response, "frames", frames);
            return response;
        }

        if (command == "quit")
        {
            juce::JUCEApplication::getInstance()->systemRequestedQuit();
            return makeResponse(true, "Quit requested");
        }

        return makeResponse(false, "Unknown command: " + command);
    }

    void appendStatusFields(juce::var& response) const
    {
        bool pluginIsInstrument = false;
        bool pluginIsEffect = false;
        juce::String pluginCategory = pluginDescription.category;
        juce::StringArray parameterNames;
        juce::Array<juce::var> parameters;
        juce::String audioDeviceType;
        juce::String audioDeviceName;
        juce::StringArray availableAudioDeviceTypes;
        juce::StringArray availableAudioOutputDevices;
        juce::Array<juce::var> availableAudioSampleRates;
        juce::Array<juce::var> availableAudioBufferSizes;
        juce::Array<juce::var> graphSlots;
        float currentPluginOutputPeakLevel = 0.0f;

        {
            juce::ScopedLock lock(pluginLock);
            currentPluginOutputPeakLevel = pluginOutputPeakLevel;
            if (pluginInstance != nullptr)
            {
                pluginIsInstrument = pluginDescription.isInstrument
                    || (pluginInstance->acceptsMidi() && pluginInstance->getTotalNumInputChannels() == 0);
                pluginIsEffect = !pluginIsInstrument;

                auto pluginParameters = pluginInstance->getParameters();
                for (int index = 0; index < pluginParameters.size(); ++index)
                {
                    auto* parameter = pluginParameters[index];
                    if (parameter == nullptr)
                        continue;

                    auto parameterName = parameter->getName(256).trim();
                    if (parameterName.isEmpty())
                        parameterName = "Param " + juce::String(index + 1);

                    auto* parameterObject = new juce::DynamicObject();
                    parameterObject->setProperty("index", index);
                    parameterObject->setProperty("name", parameterName);
                    parameterObject->setProperty("label", parameter->getLabel().trim());
                    parameterObject->setProperty("normalized_value", parameter->getValue());
                    parameters.add(juce::var(parameterObject));
                    parameterNames.addIfNotAlreadyThere(parameterName);
                }
            }
            for (int index = 0; index < static_cast<int>(persistentGraphSlots.size()); ++index)
            {
                const auto& slot = persistentGraphSlots[static_cast<size_t>(index)];
                if (slot.plugin == nullptr)
                    continue;
                auto* slotObject = new juce::DynamicObject();
                slotObject->setProperty("slot_index", index);
                slotObject->setProperty("name", slot.name);
                slotObject->setProperty("plugin_path", slot.pluginPath);
                slotObject->setProperty("state_path", slot.statePath);
                slotObject->setProperty("gain", slot.gain);
                slotObject->setProperty("pan", slot.pan);
                slotObject->setProperty("peak_level", slot.peakLevel);
                slotObject->setProperty("fx_count", static_cast<int>(slot.fxChain.size()));
                graphSlots.add(juce::var(slotObject));
            }
        }

        if (audioInitialised)
        {
            audioDeviceType = deviceManager.getCurrentAudioDeviceType();
            if (auto* device = deviceManager.getCurrentAudioDevice())
            {
                audioDeviceName = device->getName();
                for (const auto rate : device->getAvailableSampleRates())
                    availableAudioSampleRates.add(static_cast<int>(std::round(rate)));
                for (const auto size : device->getAvailableBufferSizes())
                    availableAudioBufferSizes.add(size);
            }
            for (auto* type : const_cast<juce::AudioDeviceManager&>(deviceManager).getAvailableDeviceTypes())
            {
                if (type == nullptr || ! isUsableAudioDeviceType(*type))
                    continue;
                availableAudioDeviceTypes.addIfNotAlreadyThere(type->getTypeName());
            }
            if (auto* type = deviceManager.getCurrentDeviceTypeObject())
                availableAudioOutputDevices = type->getDeviceNames(false);
        }

        setResponseField(response, "plugin_loaded", pluginInstance != nullptr);
        setResponseField(response, "plugin_name", pluginDescription.name);
        setResponseField(response, "plugin_path",
                         normaliseVst3PathForSettings(pathEditor.getText().trim()));
        setResponseField(response, "plugin_is_instrument", pluginIsInstrument);
        setResponseField(response, "plugin_is_effect", pluginIsEffect);
        setResponseField(response, "plugin_category", pluginCategory);
        setResponseField(response, "parameter_names", juce::var(parameterNames));
        setResponseField(response, "plugin_parameters", juce::var(parameters));
        setResponseField(response, "editor_open", editorWindow != nullptr);

        // Detect plugin state changes that don't fire audioProcessorChanged
        // by checksumming the actual state blob on each status poll.
        // Only check when editor is open to avoid false positives from
        // plugins that embed playback-related data in their state.
        if (pluginInstance != nullptr && editorWindow != nullptr)
        {
            juce::MemoryBlock stateBlock;
            pluginInstance->getStateInformation(stateBlock);
            uint64_t checksum = 0xcbf29ce484222325ULL; // FNV-1a offset basis
            for (size_t i = 0; i < stateBlock.getSize(); ++i)
            {
                checksum ^= static_cast<uint64_t>(static_cast<const uint8_t*>(stateBlock.getData())[i]);
                checksum *= 0x100000001b3ULL; // FNV-1a prime
            }
            if (checksum != lastPluginStateChecksum && lastPluginStateChecksum != 0)
                pluginStateGeneration.fetch_add(1, std::memory_order_relaxed);
            lastPluginStateChecksum = checksum;
        }

        setResponseField(response, "state_generation", pluginStateGeneration.load(std::memory_order_relaxed));
        setResponseField(response, "sample_rate", currentSampleRate);
        setResponseField(response, "buffer_size", currentBlockSize);
        setResponseField(response, "rendered_sample_frames", static_cast<double>(renderedSampleFrames.load()));
        setResponseField(response, "command_port", getActiveCommandPort());
        setResponseField(response, "bridge_mode", bridgeMode);
        setResponseField(response, "plugin_output_gain", pluginOutputGain.load());
        setResponseField(response, "plugin_output_pan", pluginOutputPan.load());
        setResponseField(response, "plugin_output_peak_level", currentPluginOutputPeakLevel);
        setResponseField(response, "graph_slot_count", static_cast<int>(graphSlots.size()));
        setResponseField(response, "graph_slots", juce::var(graphSlots));
        setResponseField(response, "graph_editor_open", graphEditorWindow != nullptr);
        setResponseField(response, "graph_editor_slot", graphEditorSlotIndex);
        setResponseField(response, "graph_transport_enabled", static_cast<bool>(graphTransportEnabled.load()));
        setResponseField(response, "inprocess_transport_running", inProcessTransport.running);
        setResponseField(response, "inprocess_transport_position_frame",
                         static_cast<double>(inProcessTransport.positionFrame));
        juce::Array<juce::var> audioEngineTracks;
        for (int trackIndex = 0; trackIndex < static_cast<int>(audioEngine.tracks.size()); ++trackIndex)
        {
            const auto& track = audioEngine.tracks[static_cast<size_t>(trackIndex)];
            auto* trackObject = new juce::DynamicObject();
            trackObject->setProperty("track_index", trackIndex);
            trackObject->setProperty("name", track.name);
            trackObject->setProperty("track_type",
                track.kind == AudioEngineTrackKind::audio ? "audio" : "instrument");
            trackObject->setProperty("plugin_path", track.pluginPath);
            trackObject->setProperty("note_count", static_cast<int>(track.notes.size()));
            trackObject->setProperty("clip_count", static_cast<int>(track.audioClips.size()));
            trackObject->setProperty("peak_level", track.peakLevel);
            audioEngineTracks.add(juce::var(trackObject));
        }
        setResponseField(response, "audio_engine_track_count", static_cast<int>(audioEngine.tracks.size()));
        setResponseField(response, "audio_engine_running", audioEngine.running);
        setResponseField(response, "audio_engine_position_frame", static_cast<double>(audioEngine.positionFrame));
        setResponseField(response, "audio_engine_metronome_enabled", audioEngine.metronomeEnabled);
        setResponseField(response, "audio_engine_tracks", juce::var(audioEngineTracks));
        setResponseField(response, "graph_scheduled_event_count", totalPersistentGraphScheduledEventCount());
        setResponseField(response, "input_buses", describeBuses(true));
        setResponseField(response, "output_buses", describeBuses(false));
        setResponseField(response, "audio_device_type", audioDeviceType);
        setResponseField(response, "audio_device_name", audioDeviceName);
        setResponseField(response, "available_audio_device_types", juce::var(availableAudioDeviceTypes));
        setResponseField(response, "available_audio_output_devices", juce::var(availableAudioOutputDevices));
        setResponseField(response, "available_audio_sample_rates", juce::var(availableAudioSampleRates));
        setResponseField(response, "available_audio_buffer_sizes", juce::var(availableAudioBufferSizes));
        setResponseField(response, "audio_stream_enabled", static_cast<bool>(audioStreamEnabled.load()));
        setResponseField(response, "queued_audio_frames", queuedStreamAudioFrames(kMainAudioStreamId));
        setResponseField(response, "queued_audio_channels", queuedStreamAudioChannels(kMainAudioStreamId));
        setResponseField(response, "audio_queue_underruns", static_cast<double>(queuedStreamAudioUnderruns(kMainAudioStreamId)));
        setResponseField(response, "audio_streams", describeQueuedAudioStreams());
    }

    void configureButton(juce::TextButton& button, const juce::String& text)
    {
        addAndMakeVisible(button);
        button.setButtonText(text);
        button.addListener(this);
    }

    bool isUsableAudioDeviceType(const juce::AudioIODeviceType& type) const
    {
        return ! type.getDeviceNames(false).isEmpty()
            || ! type.getDeviceNames(true).isEmpty();
    }

    juce::String canonicalAudioDeviceTypeName(const juce::String& requestedType)
    {
        const auto target = requestedType.trim();
        if (target.isEmpty())
            return {};

        for (auto* type : deviceManager.getAvailableDeviceTypes())
        {
            if (type == nullptr)
                continue;
            const auto typeName = type->getTypeName();
            if (typeName.equalsIgnoreCase(target))
                return typeName;
        }

        return {};
    }

    juce::String canonicalAudioOutputDeviceName(const juce::String& requestedName) const
    {
        const auto target = requestedName.trim();
        if (target.isEmpty())
            return {};

        if (auto* type = deviceManager.getCurrentDeviceTypeObject())
        {
            for (const auto& deviceName : type->getDeviceNames(false))
            {
                if (deviceName.equalsIgnoreCase(target))
                    return deviceName;
            }
        }

        return {};
    }

    bool applyAudioDeviceType(const juce::String& requestedType, bool treatAsChosenDevice)
    {
        if (!audioInitialised)
            return false;

        const auto typeName = canonicalAudioDeviceTypeName(requestedType);
        if (typeName.isEmpty())
            return false;

        if (! deviceManager.getCurrentAudioDeviceType().equalsIgnoreCase(typeName))
            deviceManager.setCurrentAudioDeviceType(typeName, treatAsChosenDevice);

        const auto activeType = deviceManager.getCurrentAudioDeviceType();
        if (treatAsChosenDevice && activeType.isNotEmpty())
            appSettings.setValue("audio_device_type", activeType);
        return activeType.equalsIgnoreCase(typeName);
    }

    void initialiseAudio()
    {
        deviceManager.initialise(0, 2, nullptr, true, {}, nullptr);
        deviceManager.addAudioCallback(this);
        audioInitialised = true;
    }

    void restoreAudioPreferences(double startupSampleRate,
                                 int startupBufferSize,
                                 const juce::String& startupDeviceType = {},
                                 const juce::String& startupOutputDeviceName = {})
    {
        if (!audioInitialised)
            return;

        const auto preferredType = startupDeviceType.isNotEmpty()
            ? startupDeviceType.trim()
            : appSettings.getValue("audio_device_type").trim();
        if (preferredType.isNotEmpty())
            applyAudioDeviceType(preferredType, true);

        const auto wantedRate = startupSampleRate > 0.0
            ? startupSampleRate
            : appSettings.getDoubleValue("audio_sample_rate", 0.0);
        const auto wantedBuffer = startupBufferSize > 0
            ? startupBufferSize
            : appSettings.getIntValue("audio_buffer_size", 0);
        const auto wantedOutputDevice = startupOutputDeviceName.isNotEmpty()
            ? startupOutputDeviceName.trim()
            : appSettings.getValue("audio_output_device_name").trim();
        if (auto* device = deviceManager.getCurrentAudioDevice())
        {
            auto setup = deviceManager.getAudioDeviceSetup();
            bool changed = false;

            if (wantedOutputDevice.isNotEmpty())
            {
                const auto outputDeviceName = canonicalAudioOutputDeviceName(wantedOutputDevice);
                if (outputDeviceName.isNotEmpty() && ! setup.outputDeviceName.equalsIgnoreCase(outputDeviceName))
                {
                    setup.outputDeviceName = outputDeviceName;
                    changed = true;
                }
            }

            if (wantedRate > 0.0)
            {
                for (const auto rate : device->getAvailableSampleRates())
                {
                    if (std::abs(rate - wantedRate) < 0.1)
                    {
                        setup.sampleRate = rate;
                        changed = true;
                        break;
                    }
                }
            }

            if (wantedBuffer > 0)
            {
                for (const auto size : device->getAvailableBufferSizes())
                {
                    if (size == wantedBuffer)
                    {
                        setup.bufferSize = size;
                        changed = true;
                        break;
                    }
                }
            }

            if (changed)
                deviceManager.setAudioDeviceSetup(setup, true);
        }

        if (deviceManager.getCurrentAudioDevice() == nullptr)
        {
            const auto failedType = deviceManager.getCurrentAudioDeviceType();
            for (auto* type : deviceManager.getAvailableDeviceTypes())
            {
                if (type == nullptr || ! isUsableAudioDeviceType(*type))
                    continue;

                const auto typeName = type->getTypeName();
                if (typeName.equalsIgnoreCase(failedType))
                    continue;

                deviceManager.setCurrentAudioDeviceType(typeName, false);
                if (deviceManager.getCurrentAudioDevice() != nullptr)
                    break;
            }
        }

        if (auto* device = deviceManager.getCurrentAudioDevice())
        {
            const auto activeType = deviceManager.getCurrentAudioDeviceType();
            if (activeType.isNotEmpty())
                appSettings.setValue("audio_device_type", activeType);
            appSettings.setValue("audio_output_device_name", device->getName());
        }
        if (startupSampleRate > 0.0)
            appSettings.setValue("audio_sample_rate", startupSampleRate);
        if (startupBufferSize > 0)
            appSettings.setValue("audio_buffer_size", startupBufferSize);
    }

    void updateOutputDeviceBox()
    {
        if (bridgeMode)
            return;

        outputDeviceBox.clear(juce::dontSendNotification);

        juce::String currentDeviceName;
        if (auto* device = deviceManager.getCurrentAudioDevice())
            currentDeviceName = device->getName();

        int selectedId = 0;
        int itemId = 1;
        if (auto* type = deviceManager.getCurrentDeviceTypeObject())
        {
            for (const auto& deviceName : type->getDeviceNames(false))
            {
                outputDeviceBox.addItem(deviceName, itemId);
                if (deviceName.equalsIgnoreCase(currentDeviceName))
                    selectedId = itemId;
                ++itemId;
            }
        }

        if (selectedId != 0)
            outputDeviceBox.setSelectedId(selectedId, juce::dontSendNotification);
        else if (outputDeviceBox.getNumItems() > 0)
            outputDeviceBox.setSelectedItemIndex(0, juce::dontSendNotification);

        outputDeviceBox.setEnabled(outputDeviceBox.getNumItems() > 0);
    }

    void updateDeviceTypeBox()
    {
        if (bridgeMode)
            return;

        deviceTypeBox.clear(juce::dontSendNotification);

        const auto currentType = deviceManager.getCurrentAudioDeviceType();
        int selectedId = 0;
        int itemId = 1;

        for (auto* type : deviceManager.getAvailableDeviceTypes())
        {
            if (type == nullptr || ! isUsableAudioDeviceType(*type))
                continue;

            const auto typeName = type->getTypeName();
            deviceTypeBox.addItem(typeName, itemId);
            if (typeName.equalsIgnoreCase(currentType))
                selectedId = itemId;
            ++itemId;
        }

        if (selectedId != 0)
            deviceTypeBox.setSelectedId(selectedId, juce::dontSendNotification);
        else if (deviceTypeBox.getNumItems() > 0)
            deviceTypeBox.setSelectedItemIndex(0, juce::dontSendNotification);

        deviceTypeBox.setEnabled(deviceTypeBox.getNumItems() > 1);
    }

    void updateDeviceBoxes()
    {
        updateDeviceTypeBox();
        updateOutputDeviceBox();
        sampleRateBox.clear(juce::dontSendNotification);
        bufferSizeBox.clear(juce::dontSendNotification);

        if (bridgeMode)
        {
            const auto rateValue = juce::jmax(1, static_cast<int>(std::round(currentSampleRate)));
            const auto bufferValue = juce::jmax(64, currentBlockSize);
            sampleRateBox.addItem(juce::String(rateValue) + " Hz", rateValue);
            sampleRateBox.setSelectedId(rateValue, juce::dontSendNotification);
            bufferSizeBox.addItem(juce::String(bufferValue) + " samples", bufferValue);
            bufferSizeBox.setSelectedId(bufferValue, juce::dontSendNotification);
            return;
        }

        if (auto* device = deviceManager.getCurrentAudioDevice())
        {
            int selectedRate = 0;
            int selectedBuffer = 0;
            for (const auto rate : device->getAvailableSampleRates())
            {
                const int value = static_cast<int>(std::round(rate));
                sampleRateBox.addItem(juce::String(value) + " Hz", value);
                if (std::abs(rate - device->getCurrentSampleRate()) < 0.1)
                    selectedRate = value;
            }
            for (const auto size : device->getAvailableBufferSizes())
            {
                bufferSizeBox.addItem(juce::String(size) + " samples", size);
                if (size == device->getCurrentBufferSizeSamples())
                    selectedBuffer = size;
            }
            if (selectedRate != 0)
                sampleRateBox.setSelectedId(selectedRate, juce::dontSendNotification);
            if (selectedBuffer != 0)
                bufferSizeBox.setSelectedId(selectedBuffer, juce::dontSendNotification);
        }
    }

    void updateDeviceLabel()
    {
        if (bridgeMode)
        {
            deviceLabel.setText(
                "Bridge render mode  "
                    + juce::String(static_cast<int>(std::round(currentSampleRate))) + " Hz"
                    + "  " + juce::String(currentBlockSize) + " samples",
                juce::dontSendNotification
            );
            return;
        }

        if (auto* device = deviceManager.getCurrentAudioDevice())
        {
            const auto currentType = deviceManager.getCurrentAudioDeviceType();
            deviceLabel.setText(
                "Audio: " + (currentType.isNotEmpty() ? currentType + "  |  " : juce::String())
                + device->getName()
                + "  " + juce::String(static_cast<int>(std::round(device->getCurrentSampleRate()))) + " Hz"
                + "  " + juce::String(device->getCurrentBufferSizeSamples()) + " samples",
                juce::dontSendNotification
            );
            return;
        }
        deviceLabel.setText("Audio: unavailable", juce::dontSendNotification);
    }

    void updateButtons()
    {
        const auto loaded = (pluginInstance != nullptr);
        loadButton.setEnabled(true);
        unloadButton.setEnabled(loaded);
        editorButton.setEnabled(loaded);
    }

    // AudioProcessorListener — detect preset / state changes in the main plugin.
    void audioProcessorParameterChanged(juce::AudioProcessor*, int, float) override {}
    void audioProcessorChanged(juce::AudioProcessor*, const juce::AudioProcessorListener::ChangeDetails&) override
    {
        pluginStateGeneration.fetch_add(1, std::memory_order_relaxed);
    }

    void buttonClicked(juce::Button* button) override
    {
        if (button == &browseButton)
        {
            fileChooser = std::make_unique<juce::FileChooser>("Select a VST3 plugin",
                                                              juce::File(pathEditor.getText().trim()),
                                                              "*.vst3");
            fileChooser->launchAsync(
                juce::FileBrowserComponent::openMode | juce::FileBrowserComponent::canSelectFiles,
                [this](const juce::FileChooser& chooser)
                {
                    const auto result = chooser.getResult();
                    if (result.exists())
                        pathEditor.setText(result.getFullPathName(), juce::dontSendNotification);
                    fileChooser.reset();
                }
            );
            return;
        }

        if (button == &loadButton)
        {
            loadPlugin(pathEditor.getText().trim());
            return;
        }

        if (button == &unloadButton)
        {
            unloadPlugin();
            return;
        }

        if (button == &editorButton)
        {
            openEditorWindow();
            return;
        }
    }

    void comboBoxChanged(juce::ComboBox* box) override
    {
        if (box == &deviceTypeBox && ! bridgeMode)
        {
            const auto selectedType = deviceTypeBox.getText().trim();
            if (selectedType.isNotEmpty())
            {
                const auto wantedRate = sampleRateBox.getSelectedId() > 0
                    ? static_cast<double>(sampleRateBox.getSelectedId())
                    : appSettings.getDoubleValue("audio_sample_rate", 0.0);
                const auto wantedBuffer = bufferSizeBox.getSelectedId() > 0
                    ? bufferSizeBox.getSelectedId()
                    : appSettings.getIntValue("audio_buffer_size", 0);

                if (applyAudioDeviceType(selectedType, true))
                {
                    restoreAudioPreferences(wantedRate, wantedBuffer, selectedType);
                    updateDeviceBoxes();
                    updateDeviceLabel();
                    preparePluginForPlayback();
                }
            }
            return;
        }

        auto setup = deviceManager.getAudioDeviceSetup();
        bool changed = false;

        if (box == &outputDeviceBox && ! bridgeMode)
        {
            const auto outputDeviceName = canonicalAudioOutputDeviceName(outputDeviceBox.getText());
            if (outputDeviceName.isNotEmpty() && ! setup.outputDeviceName.equalsIgnoreCase(outputDeviceName))
            {
                setup.outputDeviceName = outputDeviceName;
                appSettings.setValue("audio_output_device_name", outputDeviceName);
                changed = true;
            }
        }

        if (box == &sampleRateBox && sampleRateBox.getSelectedId() > 0)
        {
            setup.sampleRate = static_cast<double>(sampleRateBox.getSelectedId());
            appSettings.setValue("audio_sample_rate", setup.sampleRate);
            changed = true;
        }

        if (box == &bufferSizeBox && bufferSizeBox.getSelectedId() > 0)
        {
            setup.bufferSize = bufferSizeBox.getSelectedId();
            appSettings.setValue("audio_buffer_size", setup.bufferSize);
            changed = true;
        }

        if (changed)
        {
            const auto error = deviceManager.setAudioDeviceSetup(setup, true);
            if (error.isNotEmpty())
                statusLabel.setText(error, juce::dontSendNotification);
            updateDeviceBoxes();
            updateDeviceLabel();
            preparePluginForPlayback();
        }
    }

    void audioDeviceAboutToStart(juce::AudioIODevice* device) override
    {
        currentSampleRate = device != nullptr ? device->getCurrentSampleRate() : 44100.0;
        currentBlockSize = device != nullptr ? device->getCurrentBufferSizeSamples() : 512;
        if (! bridgeMode)
        {
            const auto activeType = deviceManager.getCurrentAudioDeviceType();
            if (activeType.isNotEmpty())
                appSettings.setValue("audio_device_type", activeType);
            if (device != nullptr)
                appSettings.setValue("audio_output_device_name", device->getName());
            updateDeviceTypeBox();
            updateOutputDeviceBox();
            updateDeviceLabel();
        }
        keyboardState.reset();
        ensureScratchBufferSize(currentBlockSize);
        preparePluginForPlayback();
        preparePersistentPluginGraphForPlayback();
        prepareAudioEngineForPlayback();
    }

    void audioDeviceStopped() override
    {
        releasePluginResources();
        releasePersistentPluginGraphResources();
        releaseAudioEngineResources();
    }

    void audioDeviceIOCallbackWithContext(const float* const* /*inputChannelData*/,
                                          int /*numInputChannels*/,
                                          float* const* outputChannelData,
                                          int numOutputChannels,
                                          int numSamples,
                                          const juce::AudioIODeviceCallbackContext& /*context*/) override
    {
        juce::AudioBuffer<float> buffer(outputChannelData, numOutputChannels, numSamples);
        buffer.clear();
        ensureScratchBufferSize(numSamples);

        {
            juce::ScopedLock lock(pluginLock);
            auto* plugin = pluginInstance.get();
            if (plugin != nullptr)
            {
                juce::MidiBuffer midi;
                keyboardState.processNextMidiBuffer(midi, 0, numSamples, true);
                appendPendingMidiMessages(midi);
                if (inProcessTransport.running)
                    inProcessTransport.generateMidi(midi, numSamples, currentSampleRate);
                else
                    appendScheduledMidiMessages(midi, numSamples);
                plugin->processBlock(buffer, midi);
                applyPluginOutputMix(buffer);
                pluginOutputPeakLevel = peakLevelForBuffer(buffer);
            }
            else
            {
                pluginOutputPeakLevel = 0.0f;
            }
            if (graphTransportEnabled.load())
                processPersistentPluginGraphTransport(buffer, numSamples, renderedSampleFrames.load());
            if (audioEngine.running)
                processAudioEngine(buffer, numSamples);
        }

        mixQueuedAudioStreamsIntoBuffer(buffer);

        renderedSampleFrames.fetch_add(numSamples);
    }

    void applyPluginOutputMix(juce::AudioBuffer<float>& buffer)
    {
        applyOutputMixToBuffer(
            buffer,
            juce::jmax(0.0f, pluginOutputGain.load()),
            juce::jlimit(-1.0f, 1.0f, pluginOutputPan.load())
        );
    }

    static void applyOutputMixToBuffer(juce::AudioBuffer<float>& buffer, float gain, float pan)
    {
        gain = juce::jmax(0.0f, gain);
        pan = juce::jlimit(-1.0f, 1.0f, pan);
        const auto angle = (pan + 1.0f) * juce::MathConstants<float>::pi * 0.25f;
        const auto leftGain = std::cos(angle) * gain;
        const auto rightGain = std::sin(angle) * gain;

        if (buffer.getNumChannels() > 0)
            buffer.applyGain(0, 0, buffer.getNumSamples(), leftGain);
        if (buffer.getNumChannels() > 1)
            buffer.applyGain(1, 0, buffer.getNumSamples(), rightGain);
        for (int channel = 2; channel < buffer.getNumChannels(); ++channel)
            buffer.applyGain(channel, 0, buffer.getNumSamples(), gain);
    }

    static float peakLevelForBuffer(const juce::AudioBuffer<float>& buffer)
    {
        return juce::jlimit(0.0f, 1.0f, buffer.getMagnitude(0, buffer.getNumSamples()));
    }

    static void copyStereoBufferToPluginInput(
        juce::AudioPluginInstance& plugin,
        juce::AudioBuffer<float>& processBuffer,
        const juce::AudioBuffer<float>& inputBuffer
    )
    {
        if (plugin.getBusCount(true) > 0)
        {
            auto inputBus = plugin.getBusBuffer(processBuffer, true, 0);
            for (int channel = 0; channel < inputBus.getNumChannels(); ++channel)
            {
                const auto sourceChannel = channel < inputBuffer.getNumChannels() ? channel : 0;
                inputBus.copyFrom(channel, 0, inputBuffer, sourceChannel, 0, inputBuffer.getNumSamples());
            }
            return;
        }

        for (int channel = 0; channel < processBuffer.getNumChannels(); ++channel)
        {
            const auto sourceChannel = channel < inputBuffer.getNumChannels() ? channel : 0;
            processBuffer.copyFrom(channel, 0, inputBuffer, sourceChannel, 0, inputBuffer.getNumSamples());
        }
    }

    static void extractPluginOutputToStereo(
        juce::AudioPluginInstance& plugin,
        juce::AudioBuffer<float>& processBuffer,
        juce::AudioBuffer<float>& stereoBuffer
    )
    {
        stereoBuffer.clear();
        if (plugin.getBusCount(false) > 0)
        {
            auto outputBus = plugin.getBusBuffer(processBuffer, false, 0);
            if (outputBus.getNumChannels() > 0)
            {
                const auto leftChannel = 0;
                const auto rightChannel = outputBus.getNumChannels() > 1 ? 1 : 0;
                stereoBuffer.copyFrom(0, 0, outputBus, leftChannel, 0, stereoBuffer.getNumSamples());
                stereoBuffer.copyFrom(1, 0, outputBus, rightChannel, 0, stereoBuffer.getNumSamples());
            }
            return;
        }

        if (processBuffer.getNumChannels() <= 0)
            return;
        const auto rightChannel = processBuffer.getNumChannels() > 1 ? 1 : 0;
        stereoBuffer.copyFrom(0, 0, processBuffer, 0, 0, stereoBuffer.getNumSamples());
        stereoBuffer.copyFrom(1, 0, processBuffer, rightChannel, 0, stereoBuffer.getNumSamples());
    }

    static juce::AudioBuffer<float> ensureStereoAudioEngineBuffer(const juce::AudioBuffer<float>& input)
    {
        juce::AudioBuffer<float> stereo(2, juce::jmax(0, input.getNumSamples()));
        stereo.clear();
        if (input.getNumChannels() <= 0 || input.getNumSamples() <= 0)
            return stereo;

        const auto rightChannel = input.getNumChannels() > 1 ? 1 : 0;
        stereo.copyFrom(0, 0, input, 0, 0, input.getNumSamples());
        stereo.copyFrom(1, 0, input, rightChannel, 0, input.getNumSamples());
        return stereo;
    }

    static juce::AudioBuffer<float> resampleAudioEngineBuffer(const juce::AudioBuffer<float>& input,
                                                              double sourceSampleRate,
                                                              double targetSampleRate)
    {
        if (input.getNumSamples() <= 0 || input.getNumChannels() <= 0)
            return ensureStereoAudioEngineBuffer(input);
        if (sourceSampleRate <= 0.0 || targetSampleRate <= 0.0
            || std::abs(sourceSampleRate - targetSampleRate) < 0.01)
        {
            return ensureStereoAudioEngineBuffer(input);
        }

        const auto ratio = targetSampleRate / sourceSampleRate;
        const auto targetSamples = juce::jmax(1, static_cast<int>(std::llround(input.getNumSamples() * ratio)));
        juce::AudioBuffer<float> output(input.getNumChannels(), targetSamples);
        output.clear();

        for (int channel = 0; channel < input.getNumChannels(); ++channel)
        {
            const auto* source = input.getReadPointer(channel);
            auto* destination = output.getWritePointer(channel);
            for (int sample = 0; sample < targetSamples; ++sample)
            {
                const auto sourcePosition = (static_cast<double>(sample) * sourceSampleRate) / targetSampleRate;
                const auto lowerIndex = juce::jlimit(0, juce::jmax(0, input.getNumSamples() - 1), static_cast<int>(std::floor(sourcePosition)));
                const auto upperIndex = juce::jlimit(0, juce::jmax(0, input.getNumSamples() - 1), lowerIndex + 1);
                const auto fraction = static_cast<float>(sourcePosition - static_cast<double>(lowerIndex));
                destination[sample] = source[lowerIndex] + ((source[upperIndex] - source[lowerIndex]) * fraction);
            }
        }

        return ensureStereoAudioEngineBuffer(output);
    }

    bool loadAudioEngineClipBuffer(const juce::String& path,
                                   juce::AudioBuffer<float>& destination,
                                   juce::String& error)
    {
        if (currentSampleRate <= 0.0)
        {
            error = "Audio device is not ready for audio clips";
            return false;
        }

        const juce::File audioFile(path);
        if (!audioFile.existsAsFile())
        {
            error = "Audio clip file not found";
            return false;
        }

        auto reader = std::unique_ptr<juce::AudioFormatReader>(audioFormatManager.createReaderFor(audioFile));
        if (reader == nullptr)
        {
            error = "Could not open audio clip";
            return false;
        }

        const auto sourceSamples = juce::jmax<int>(1, static_cast<int>(reader->lengthInSamples));
        const auto sourceChannels = juce::jmax(1, static_cast<int>(reader->numChannels));
        juce::AudioBuffer<float> source(sourceChannels, sourceSamples);
        source.clear();
        if (!reader->read(&source, 0, sourceSamples, 0, true, true))
        {
            error = "Could not read audio clip";
            return false;
        }

        auto resampled = resampleAudioEngineBuffer(source, reader->sampleRate, currentSampleRate);
        destination.makeCopyOf(resampled);
        return true;
    }

    static void prepareAudioEngineMetronomeBuffer(juce::AudioBuffer<float>& buffer,
                                                  double sampleRate,
                                                  bool accented)
    {
        const auto clickFrames = juce::jmax(8, static_cast<int>(sampleRate * (accented ? 0.032 : 0.022)));
        buffer.setSize(2, clickFrames, false, false, true);
        buffer.clear();

        const auto frequency = accented ? 1760.0f : 1320.0f;
        const auto gain = accented ? 0.22f : 0.14f;
        const auto decayRate = accented ? 78.0f : 96.0f;
        const auto harmonic = accented ? 0.28f : 0.18f;
        auto* left = buffer.getWritePointer(0);
        auto* right = buffer.getWritePointer(1);
        for (int sample = 0; sample < clickFrames; ++sample)
        {
            const auto t = static_cast<float>(sample / sampleRate);
            const auto envelope = std::exp(-t * decayRate);
            const auto value = (std::sin(2.0f * juce::MathConstants<float>::pi * frequency * t)
                + (harmonic * std::sin(2.0f * juce::MathConstants<float>::pi * frequency * 2.0f * t)))
                * envelope * gain;
            left[sample] = value;
            right[sample] = value;
        }
    }

    void prepareAudioEngineMetronome()
    {
        if (!audioEngine.metronomeEnabled || currentSampleRate <= 0.0)
        {
            audioEngine.metronomeClickBuffer.setSize(0, 0);
            audioEngine.metronomeAccentBuffer.setSize(0, 0);
            return;
        }

        prepareAudioEngineMetronomeBuffer(audioEngine.metronomeClickBuffer, currentSampleRate, false);
        prepareAudioEngineMetronomeBuffer(audioEngine.metronomeAccentBuffer, currentSampleRate, true);
    }

    static void renderAudioEngineTrackClips(AudioEngineTrack& track,
                                            int64_t chunkStartFrame,
                                            int chunkSamples)
    {
        track.stereoBuffer.clear();
        if (track.audioClips.empty() || chunkSamples <= 0)
            return;

        const auto chunkEndFrame = chunkStartFrame + static_cast<int64_t>(chunkSamples);
        for (const auto& clip : track.audioClips)
        {
            const auto clipSamples = clip.audioData.getNumSamples();
            if (clipSamples <= 0)
                continue;

            const auto clipStartFrame = clip.startFrame;
            const auto clipEndFrame = clipStartFrame + static_cast<int64_t>(clipSamples);
            const auto overlapStart = juce::jmax(chunkStartFrame, clipStartFrame);
            const auto overlapEnd = juce::jmin(chunkEndFrame, clipEndFrame);
            if (overlapEnd <= overlapStart)
                continue;

            const auto sourceOffset = static_cast<int>(overlapStart - clipStartFrame);
            const auto destinationOffset = static_cast<int>(overlapStart - chunkStartFrame);
            const auto overlapSamples = static_cast<int>(overlapEnd - overlapStart);
            track.stereoBuffer.addFrom(0, destinationOffset, clip.audioData, 0, sourceOffset, overlapSamples);
            track.stereoBuffer.addFrom(1, destinationOffset, clip.audioData, 1, sourceOffset, overlapSamples);
        }
    }

    void mixAudioEngineMetronomeChunk(juce::AudioBuffer<float>& destination,
                                      int sampleOffset,
                                      int64_t chunkStartFrame,
                                      int chunkSamples,
                                      double sampleRate)
    {
        if (!audioEngine.metronomeEnabled || chunkSamples <= 0 || audioEngine.ticksPerBeat <= 0)
            return;

        const auto chunkEndFrame = chunkStartFrame + static_cast<int64_t>(chunkSamples);
        int64_t beatIndex = juce::jmax<int64_t>(
            0,
            (audioEngine.frameToTick(chunkStartFrame, sampleRate) / audioEngine.ticksPerBeat) - 1
        );

        while (audioEngine.tickToFrame((beatIndex + 1) * static_cast<int64_t>(audioEngine.ticksPerBeat), sampleRate) <= chunkStartFrame)
            ++beatIndex;
        while (beatIndex > 0
            && audioEngine.tickToFrame(beatIndex * static_cast<int64_t>(audioEngine.ticksPerBeat), sampleRate) > chunkStartFrame)
        {
            --beatIndex;
        }

        while (true)
        {
            const auto beatFrame = audioEngine.tickToFrame(
                beatIndex * static_cast<int64_t>(audioEngine.ticksPerBeat),
                sampleRate
            );
            const auto& clickBuffer = (beatIndex % 4) == 0
                ? audioEngine.metronomeAccentBuffer
                : audioEngine.metronomeClickBuffer;
            const auto clickSamples = clickBuffer.getNumSamples();
            if (clickSamples <= 0 && beatFrame >= chunkEndFrame)
                break;
            if (clickSamples > 0)
            {
                const auto clickEndFrame = beatFrame + static_cast<int64_t>(clickSamples);
                const auto overlapStart = juce::jmax(chunkStartFrame, beatFrame);
                const auto overlapEnd = juce::jmin(chunkEndFrame, clickEndFrame);
                if (overlapEnd > overlapStart)
                {
                    const auto sourceOffset = static_cast<int>(overlapStart - beatFrame);
                    const auto destinationOffset = sampleOffset + static_cast<int>(overlapStart - chunkStartFrame);
                    const auto overlapSamples = static_cast<int>(overlapEnd - overlapStart);
                    destination.addFrom(0, destinationOffset, clickBuffer, 0, sourceOffset, overlapSamples);
                    destination.addFrom(1, destinationOffset, clickBuffer, 1, sourceOffset, overlapSamples);
                }
            }

            if (beatFrame >= chunkEndFrame)
                break;
            ++beatIndex;
        }
    }

    void enqueueMidiMessage(const juce::MidiMessage& message)
    {
        juce::ScopedLock lock(pendingMidiLock);
        pendingMidiMessages.addEvent(message, 0);
    }

    void appendPendingMidiMessages(juce::MidiBuffer& destination)
    {
        juce::MidiBuffer pending;
        {
            juce::ScopedLock lock(pendingMidiLock);
            if (pendingMidiMessages.isEmpty())
                return;
            std::swap(pending, pendingMidiMessages);
        }

        for (const auto metadata : pending)
            destination.addEvent(metadata.getMessage(), metadata.samplePosition);
    }

    void enqueuePanicMessages(int channel)
    {
        const auto enqueueChannel = [this](int midiChannel)
        {
            enqueueMidiMessage(juce::MidiMessage::controllerEvent(midiChannel, 64, 0));
            enqueueMidiMessage(juce::MidiMessage::controllerEvent(midiChannel, 66, 0));
            enqueueMidiMessage(juce::MidiMessage::controllerEvent(midiChannel, 67, 0));
            enqueueMidiMessage(juce::MidiMessage::controllerEvent(midiChannel, 120, 0));
            enqueueMidiMessage(juce::MidiMessage::controllerEvent(midiChannel, 123, 0));
        };

        if (channel <= 0)
        {
            for (int midiChannel = 1; midiChannel <= 16; ++midiChannel)
                enqueueChannel(midiChannel);
            keyboardState.reset();
            return;
        }

        enqueueChannel(clampMidiChannel(channel));
        keyboardState.allNotesOff(clampMidiChannel(channel));
    }

    int appendScheduledPanicEvents(juce::Array<ScheduledMidiEvent>& destination,
                                   int channel,
                                   int64_t targetFrame,
                                   int64_t loopEpoch)
    {
        int added = 0;
        const auto enqueueChannel = [&](int midiChannel)
        {
            const juce::MidiMessage messages[] = {
                juce::MidiMessage::controllerEvent(midiChannel, 64, 0),
                juce::MidiMessage::controllerEvent(midiChannel, 66, 0),
                juce::MidiMessage::controllerEvent(midiChannel, 67, 0),
                juce::MidiMessage::controllerEvent(midiChannel, 120, 0),
                juce::MidiMessage::controllerEvent(midiChannel, 123, 0),
            };
            for (const auto& message : messages)
            {
                destination.add({
                    targetFrame,
                    0,
                    scheduledMidiSequence.fetch_add(1),
                    loopEpoch,
                    message,
                });
                ++added;
            }
        };

        if (channel <= 0)
        {
            for (int midiChannel = 1; midiChannel <= 16; ++midiChannel)
                enqueueChannel(midiChannel);
            return added;
        }

        enqueueChannel(clampMidiChannel(channel));
        return added;
    }

    void appendScheduledMidiMessages(juce::MidiBuffer& destination, int numSamples)
    {
        const auto blockStart = renderedSampleFrames.load();
        const auto blockEnd = blockStart + juce::jmax(1, numSamples);

        juce::ScopedLock lock(scheduledMidiLock);
        if (scheduledMidiEvents.isEmpty())
            return;

        // Partition in-place: move future events to the front, collect ready ones
        int keepCount = 0;
        scheduledMidiReady.clearQuick();
        for (int i = 0; i < scheduledMidiEvents.size(); ++i)
        {
            auto& event = scheduledMidiEvents.getReference(i);
            if (event.frame >= blockEnd)
            {
                if (keepCount != i)
                    scheduledMidiEvents.getReference(keepCount) = std::move(event);
                ++keepCount;
            }
            else
            {
                if (event.frame < blockStart)
                    event.frame = blockStart;
                scheduledMidiReady.add(std::move(event));
            }
        }
        scheduledMidiEvents.removeRange(keepCount, scheduledMidiEvents.size() - keepCount);

        if (scheduledMidiReady.isEmpty())
            return;

        std::sort(scheduledMidiReady.begin(), scheduledMidiReady.end(),
            [](const ScheduledMidiEvent& a, const ScheduledMidiEvent& b)
        {
            if (a.frame != b.frame)
                return a.frame < b.frame;
            if (a.priority != b.priority)
                return a.priority < b.priority;
            return a.sequence < b.sequence;
        });

        for (const auto& event : scheduledMidiReady)
        {
            const auto samplePosition = static_cast<int>(
                juce::jlimit<int64_t>(0, juce::jmax(0, numSamples - 1), event.frame - blockStart)
            );
            destination.addEvent(event.message, samplePosition);
        }
    }

    void clearScheduledMidiEvents(int channel = 0,
                                  int64_t beforeLoopEpoch = -1,
                                  int64_t /*minFrame*/ = -1)
    {
        juce::ScopedLock lock(scheduledMidiLock);
        if (channel <= 0)
        {
            scheduledMidiEvents.clear();
            renderedSampleFrames.store(0);
            scheduledMidiSequence.store(0);
            return;
        }

        juce::Array<ScheduledMidiEvent> keep;
        const auto targetChannel = clampMidiChannel(channel);
        for (const auto& event : scheduledMidiEvents)
        {
            if (event.message.getChannel() != targetChannel)
            {
                keep.add(event);
                continue;
            }
            // Keep events from the current or future epoch unconditionally.
            if (beforeLoopEpoch >= 0 && event.loopEpoch >= beforeLoopEpoch)
            {
                keep.add(event);
                continue;
            }
            // Old-epoch events are removed regardless of frame position.
            // The bootstrap logic will re-trigger any notes that should
            // still be sounding after the loop wrap, so keeping "near
            // playback" old events just causes double triggers.
        }
        scheduledMidiEvents.swapWith(keep);
    }

    juce::var describeBuses(bool isInput) const
    {
        juce::Array<juce::var> buses;
        auto* plugin = pluginInstance.get();
        if (plugin == nullptr)
            return juce::var(buses);

        for (int busIndex = 0; busIndex < plugin->getBusCount(isInput); ++busIndex)
        {
            const auto* bus = plugin->getBus(isInput, busIndex);
            if (bus == nullptr)
                continue;

            auto* object = new juce::DynamicObject();
            object->setProperty("bus_index", busIndex);
            object->setProperty("name", bus->getName());
            object->setProperty("direction", isInput ? "input" : "output");
            object->setProperty("is_main", bus->isMain());
            object->setProperty("enabled", bus->isEnabled());
            object->setProperty("enabled_by_default", bus->isEnabledByDefault());
            object->setProperty("channel_count", bus->getNumberOfChannels());
            object->setProperty("default_channel_count", bus->getDefaultLayout().size());
            object->setProperty("last_enabled_channel_count", bus->getLastEnabledLayout().size());
            object->setProperty("max_supported_channels", bus->getMaxSupportedChannels(16));
            object->setProperty("layout", bus->getCurrentLayout().getDescription());
            buses.add(juce::var(object));
        }

        return juce::var(buses);
    }

    bool applyRequestedBusLayouts(const juce::var& busesVar,
                                  bool isInput,
                                  juce::AudioProcessor::BusesLayout& layout,
                                  juce::String& error) const
    {
        if (!busesVar.isArray())
            return true;

        auto* plugin = pluginInstance.get();
        if (plugin == nullptr)
        {
            error = "No plugin loaded";
            return false;
        }

        auto* busesArray = busesVar.getArray();
        auto& targets = isInput ? layout.inputBuses : layout.outputBuses;

        for (const auto& busVar : *busesArray)
        {
            auto* busObject = busVar.getDynamicObject();
            if (busObject == nullptr)
                continue;

            const auto busIndex = static_cast<int>(busObject->getProperty("bus_index"));
            if (busIndex < 0 || busIndex >= targets.size())
            {
                error = "Invalid bus index";
                return false;
            }

            auto* bus = plugin->getBus(isInput, busIndex);
            if (bus == nullptr)
            {
                error = "Bus not found";
                return false;
            }

            auto shouldEnable = !busObject->hasProperty("enabled")
                || static_cast<bool>(busObject->getProperty("enabled"));
            if (bus->isMain())
                shouldEnable = true;

            if (!shouldEnable)
            {
                targets.getReference(busIndex) = juce::AudioChannelSet::disabled();
                continue;
            }

            int channels = busObject->hasProperty("channels")
                ? static_cast<int>(busObject->getProperty("channels"))
                : 0;
            if (channels <= 0)
                channels = bus->getLastEnabledLayout().size();
            if (channels <= 0)
                channels = bus->getDefaultLayout().size();
            if (channels <= 0)
                channels = juce::jmax(1, bus->getNumberOfChannels());

            auto desiredLayout = bus->supportedLayoutWithChannels(channels);
            if (desiredLayout.isDisabled())
                desiredLayout = bus->getLastEnabledLayout();
            if (desiredLayout.isDisabled())
                desiredLayout = bus->getDefaultLayout();
            if (desiredLayout.isDisabled() && bus->isMain())
                desiredLayout = juce::AudioChannelSet::canonicalChannelSet(juce::jmax(1, channels));
            if (desiredLayout.isDisabled())
            {
                error = "Unsupported bus channel layout";
                return false;
            }

            targets.getReference(busIndex) = desiredLayout;
        }

        return true;
    }

    bool configureBuses(const juce::var& request, juce::String& error)
    {
        auto* requestObject = request.getDynamicObject();
        if (requestObject == nullptr)
        {
            error = "configure_buses requires an object payload";
            return false;
        }

        juce::AudioProcessor::BusesLayout layout;
        {
            juce::ScopedLock lock(pluginLock);
            if (pluginInstance == nullptr)
            {
                error = "No plugin loaded";
                return false;
            }

            layout = pluginInstance->getBusesLayout();
            if (requestObject->hasProperty("input_buses"))
            {
                for (int busIndex = 1; busIndex < layout.inputBuses.size(); ++busIndex)
                    layout.inputBuses.getReference(busIndex) = juce::AudioChannelSet::disabled();
            }
            if (requestObject->hasProperty("output_buses"))
            {
                for (int busIndex = 1; busIndex < layout.outputBuses.size(); ++busIndex)
                    layout.outputBuses.getReference(busIndex) = juce::AudioChannelSet::disabled();
            }

            if (!applyRequestedBusLayouts(requestObject->getProperty("input_buses"), true, layout, error))
                return false;
            if (!applyRequestedBusLayouts(requestObject->getProperty("output_buses"), false, layout, error))
                return false;

            pluginInstance->releaseResources();
            if (!pluginInstance->setBusesLayout(layout))
            {
                error = "Plugin rejected the requested bus layout";
                return false;
            }
        }

        preparePluginForPlayback();
        return true;
    }

    void loadPlugin(const juce::String& pluginPath)
    {
        const auto preferredPath = normaliseVst3PathForSettings(pluginPath);
        const juce::File pluginFile(preferredPath);
        if (!pluginFile.exists())
        {
            statusLabel.setText("Plugin not found", juce::dontSendNotification);
            return;
        }

        unloadPlugin();
        clearScheduledMidiEvents();

        auto descriptions = describePlugin(pluginFile);
        if (descriptions.isEmpty())
        {
            statusLabel.setText("No loadable VST3 plugin description found", juce::dontSendNotification);
            return;
        }

        juce::String error;
        const auto& description = *descriptions[0];

        auto instance = formatManager.createPluginInstance(
            description,
            currentSampleRate,
            currentBlockSize,
            error
        );

        if (instance == nullptr)
        {
            statusLabel.setText("Load failed: " + error, juce::dontSendNotification);
            return;
        }

        {
            juce::ScopedLock lock(pluginLock);
            pluginDescription = description;
            pluginInstance = std::move(instance);
            pluginInstance->disableNonMainBuses();
            pluginInstance->addListener(this);
            lastPluginStateChecksum = 0;
            pluginStateGeneration.store(0, std::memory_order_relaxed);
        }

        pathEditor.setText(pluginFile.getFullPathName(), juce::dontSendNotification);
        appSettings.setValue("last_plugin_path", pluginFile.getFullPathName());
        preparePluginForPlayback();
        statusLabel.setText("Loaded: " + pluginDescription.name, juce::dontSendNotification);
        updateButtons();
    }

    void unloadPlugin()
    {
        closeEditorWindow();
        clearScheduledMidiEvents();
        {
            juce::ScopedLock lock(pluginLock);
            if (pluginInstance != nullptr)
            {
                pluginInstance->removeListener(this);
                pluginInstance->suspendProcessing(true);
            }
        }
        releasePluginResources();
        {
            juce::ScopedLock lock(pluginLock);
            pluginInstance.reset();
            pluginDescription = {};
        }
        statusLabel.setText("Plugin unloaded", juce::dontSendNotification);
        updateButtons();
    }

    void preparePluginForPlayback()
    {
        clearScheduledMidiEvents();
        juce::ScopedLock lock(pluginLock);
        if (pluginInstance == nullptr)
            return;
        pluginInstance->setRateAndBufferSizeDetails(currentSampleRate, currentBlockSize);
        pluginInstance->prepareToPlay(currentSampleRate, currentBlockSize);
        pluginInstance->suspendProcessing(false);
    }

    void releasePluginResources()
    {
        juce::ScopedLock lock(pluginLock);
        if (pluginInstance != nullptr)
        {
            pluginInstance->suspendProcessing(true);
            pluginInstance->releaseResources();
        }
    }

    BufferedOutputStream& queuedAudioStreamForId(const juce::String& streamId)
    {
        if (streamId == kPreviewAudioStreamId)
            return previewOutputStream;
        if (streamId == kLiveMidiAudioStreamId)
            return liveMidiOutputStream;
        return mainOutputStream;
    }

    const BufferedOutputStream& queuedAudioStreamForId(const juce::String& streamId) const
    {
        if (streamId == kPreviewAudioStreamId)
            return previewOutputStream;
        if (streamId == kLiveMidiAudioStreamId)
            return liveMidiOutputStream;
        return mainOutputStream;
    }

    void compactQueuedAudioIfNeeded(BufferedOutputStream& stream)
    {
        if (stream.readOffset == 0)
            return;
        if (stream.readOffset >= stream.data.getSize())
        {
            stream.data.setSize(0);
            stream.readOffset = 0;
            return;
        }

        juce::MemoryBlock compacted(
            static_cast<const char*>(stream.data.getData()) + stream.readOffset,
            stream.data.getSize() - stream.readOffset
        );
        stream.data.swapWith(compacted);
        stream.readOffset = 0;
    }

    bool queueStreamedAudio(const juce::String& streamId,
                            const juce::String& audioBase64,
                            int channels,
                            juce::String& error,
                            int& appendedFrames)
    {
        appendedFrames = 0;
        const auto channelCount = juce::jmax(1, channels);
        juce::MemoryOutputStream decoded;
        if (!juce::Base64::convertFromBase64(decoded, audioBase64))
        {
            error = "Could not decode queued audio";
            return false;
        }

        const auto frameBytes = static_cast<size_t>(channelCount) * sizeof(float);
        if (frameBytes == 0 || (decoded.getDataSize() % frameBytes) != 0)
        {
            error = "Queued audio payload size does not match channel count";
            return false;
        }

        appendedFrames = static_cast<int>(decoded.getDataSize() / frameBytes);
        if (appendedFrames <= 0)
            return true;

        juce::ScopedLock lock(streamAudioLock);
        auto& stream = queuedAudioStreamForId(streamId);
        compactQueuedAudioIfNeeded(stream);
        stream.data.append(decoded.getData(), decoded.getDataSize());
        stream.channelCount = channelCount;
        stream.inputFrames += appendedFrames;
        return true;
    }

    void clearStreamedAudioQueue(const juce::String& streamId, bool resetCounters)
    {
        {
            juce::ScopedLock lock(streamAudioLock);
            auto& stream = queuedAudioStreamForId(streamId);
            stream.data.setSize(0);
            stream.readOffset = 0;
            stream.channelCount = 2;
            if (resetCounters)
            {
                stream.inputFrames = 0;
                stream.outputFrames = 0;
                stream.underruns = 0;
            }
        }
        if (streamId == kMainAudioStreamId)
            audioStreamEnabled.store(false);
    }

    int queuedStreamAudioFrames(const juce::String& streamId) const
    {
        juce::ScopedLock lock(streamAudioLock);
        const auto& stream = queuedAudioStreamForId(streamId);
        if (stream.channelCount <= 0)
            return 0;
        const auto frameBytes = static_cast<size_t>(stream.channelCount) * sizeof(float);
        if (frameBytes == 0 || stream.data.getSize() <= stream.readOffset)
            return 0;
        return static_cast<int>((stream.data.getSize() - stream.readOffset) / frameBytes);
    }

    int queuedStreamAudioChannels(const juce::String& streamId) const
    {
        juce::ScopedLock lock(streamAudioLock);
        return juce::jmax(1, queuedAudioStreamForId(streamId).channelCount);
    }

    int64_t queuedStreamAudioUnderruns(const juce::String& streamId) const
    {
        juce::ScopedLock lock(streamAudioLock);
        return queuedAudioStreamForId(streamId).underruns;
    }

    juce::var describeQueuedAudioStreams() const
    {
        juce::Array<juce::var> streams;
        const auto addStream = [&](const juce::String& streamId, const BufferedOutputStream& stream)
        {
            auto* object = new juce::DynamicObject();
            const auto frameBytes = static_cast<size_t>(juce::jmax(1, stream.channelCount)) * sizeof(float);
            const auto queuedFrames = (frameBytes == 0 || stream.data.getSize() <= stream.readOffset)
                ? 0
                : static_cast<int>((stream.data.getSize() - stream.readOffset) / frameBytes);
            object->setProperty("stream_id", streamId);
            object->setProperty("queued_frames", queuedFrames);
            object->setProperty("channels", juce::jmax(1, stream.channelCount));
            object->setProperty("underruns", static_cast<double>(stream.underruns));
            object->setProperty("active", streamId == kMainAudioStreamId ? static_cast<bool>(audioStreamEnabled.load()) : true);
            streams.add(juce::var(object));
        };

        juce::ScopedLock lock(streamAudioLock);
        addStream(kMainAudioStreamId, mainOutputStream);
        addStream(kPreviewAudioStreamId, previewOutputStream);
        addStream(kLiveMidiAudioStreamId, liveMidiOutputStream);
        return juce::var(streams);
    }

    bool mixQueuedAudioStreamIntoBuffer(const juce::String& streamId, juce::AudioBuffer<float>& buffer)
    {
        const auto numSamples = buffer.getNumSamples();
        const auto numChannels = buffer.getNumChannels();
        auto& stream = queuedAudioStreamForId(streamId);
        if (stream.channelCount <= 0)
            return false;
        const auto frameBytes = static_cast<size_t>(stream.channelCount) * sizeof(float);
        if (frameBytes == 0 || stream.data.getSize() <= stream.readOffset)
            return false;

        const auto availableFrames = static_cast<int>((stream.data.getSize() - stream.readOffset) / frameBytes);
        const auto copiedFrames = juce::jmin(numSamples, availableFrames);
        auto* source = reinterpret_cast<const float*>(
            static_cast<const char*>(stream.data.getData()) + stream.readOffset
        );

        if (stream.channelCount == 2 && numChannels >= 2)
        {
            // Fast path: stereo-to-stereo bulk de-interleave + add
            for (int sample = 0; sample < copiedFrames; ++sample)
            {
                deinterleaveLeft[sample] = source[sample * 2];
                deinterleaveRight[sample] = source[sample * 2 + 1];
            }
            juce::FloatVectorOperations::add(buffer.getWritePointer(0), deinterleaveLeft.data(), copiedFrames);
            juce::FloatVectorOperations::add(buffer.getWritePointer(1), deinterleaveRight.data(), copiedFrames);
            for (int channel = 2; channel < numChannels; ++channel)
                juce::FloatVectorOperations::add(buffer.getWritePointer(channel), deinterleaveRight.data(), copiedFrames);
        }
        else if (stream.channelCount == 1)
        {
            // Mono source: bulk add the same data to all output channels
            for (int channel = 0; channel < numChannels; ++channel)
                juce::FloatVectorOperations::add(buffer.getWritePointer(channel), source, copiedFrames);
        }
        else
        {
            // Generic fallback for unusual channel counts
            for (int sample = 0; sample < copiedFrames; ++sample)
            {
                for (int channel = 0; channel < numChannels; ++channel)
                {
                    const auto sourceChannel = juce::jmin(channel, stream.channelCount - 1);
                    buffer.addSample(channel, sample, source[(sample * stream.channelCount) + sourceChannel]);
                }
            }
        }

        stream.readOffset += static_cast<size_t>(copiedFrames) * frameBytes;
        stream.outputFrames += copiedFrames;
        if (stream.readOffset >= stream.data.getSize())
        {
            stream.data.setSize(0);
            stream.readOffset = 0;
        }
        if (copiedFrames < numSamples)
            ++stream.underruns;
        return true;
    }

    void mixQueuedAudioStreamsIntoBuffer(juce::AudioBuffer<float>& buffer)
    {
        juce::ScopedLock lock(streamAudioLock);
        if (audioStreamEnabled.load())
            mixQueuedAudioStreamIntoBuffer(kMainAudioStreamId, buffer);
        mixQueuedAudioStreamIntoBuffer(kPreviewAudioStreamId, buffer);
        mixQueuedAudioStreamIntoBuffer(kLiveMidiAudioStreamId, buffer);
    }

    bool loadPluginStateFromFile(const juce::File& stateFile)
    {
        juce::MemoryBlock rawState;
        if (!stateFile.loadFileAsData(rawState) || rawState.getSize() == 0)
            return false;

        clearScheduledMidiEvents();
        keyboardState.reset();
        juce::ScopedLock lock(pluginLock);
        if (pluginInstance == nullptr)
            return false;
        pluginInstance->setStateInformation(rawState.getData(), static_cast<int>(rawState.getSize()));
        return true;
    }

    static bool loadPluginStateIntoInstance(juce::AudioPluginInstance& plugin, const juce::File& stateFile)
    {
        juce::MemoryBlock rawState;
        if (!stateFile.loadFileAsData(rawState) || rawState.getSize() == 0)
            return false;
        plugin.setStateInformation(rawState.getData(), static_cast<int>(rawState.getSize()));
        return true;
    }

    bool savePluginStateToFile(const juce::File& stateFile)
    {
        juce::MemoryBlock rawState;
        {
            juce::ScopedLock lock(pluginLock);
            if (pluginInstance == nullptr)
                return false;
            pluginInstance->getStateInformation(rawState);
        }
        if (rawState.getSize() == 0)
            return false;
        const auto parentDir = stateFile.getParentDirectory();
        if (!parentDir.exists() && !parentDir.createDirectory())
            return false;
        return stateFile.replaceWithData(rawState.getData(), rawState.getSize());
    }

    static int resolvePluginParameterIndex(juce::AudioPluginInstance& plugin, const juce::String& rawName)
    {
        auto parameters = plugin.getParameters();
        const auto requestedName = rawName.trim();
        if (requestedName.isEmpty())
            return -1;

        if (requestedName.startsWithIgnoreCase("Param "))
        {
            const auto legacyNumber = requestedName.fromFirstOccurrenceOf(" ", false, false).trim().getIntValue();
            const auto legacyIndex = legacyNumber - 1;
            if (juce::isPositiveAndBelow(legacyIndex, parameters.size()))
                return legacyIndex;
        }

        for (int index = 0; index < parameters.size(); ++index)
        {
            auto* parameter = parameters[index];
            if (parameter == nullptr)
                continue;

            auto parameterName = parameter->getName(256).trim();
            if (parameterName.isEmpty())
                parameterName = "Param " + juce::String(index + 1);

            if (parameterName.equalsIgnoreCase(requestedName)
                || ("Param " + juce::String(index + 1)).equalsIgnoreCase(requestedName))
            {
                return index;
            }
        }

        return -1;
    }

    bool applyRequestedParameterValues(const juce::var& parametersVar,
                                       juce::String& error,
                                       int& appliedCount,
                                       juce::StringArray& unmatchedParameters)
    {
        auto* parameterObject = parametersVar.getDynamicObject();
        if (parameterObject == nullptr)
        {
            error = "set_parameters requires an object payload";
            return false;
        }

        juce::ScopedLock lock(pluginLock);
        if (pluginInstance == nullptr)
        {
            error = "No plugin loaded";
            return false;
        }

        return applyRequestedParameterValuesToPlugin(
            *pluginInstance,
            parametersVar,
            error,
            appliedCount,
            unmatchedParameters
        );
    }

    static bool applyRequestedParameterValuesToPlugin(juce::AudioPluginInstance& plugin,
                                                      const juce::var& parametersVar,
                                                      juce::String& error,
                                                      int& appliedCount,
                                                      juce::StringArray& unmatchedParameters)
    {
        auto* parameterObject = parametersVar.getDynamicObject();
        if (parameterObject == nullptr)
        {
            error = "set_parameters requires an object payload";
            return false;
        }

        auto parameters = plugin.getParameters();

        for (const auto& property : parameterObject->getProperties())
        {
            const auto parameterName = property.name.toString();
            const auto parameterIndex = resolvePluginParameterIndex(plugin, parameterName);
            if (!juce::isPositiveAndBelow(parameterIndex, parameters.size()))
            {
                unmatchedParameters.addIfNotAlreadyThere(parameterName);
                continue;
            }

            auto* parameter = parameters[parameterIndex];
            if (parameter == nullptr)
            {
                unmatchedParameters.addIfNotAlreadyThere(parameterName);
                continue;
            }

            const auto normalizedValue = juce::jlimit(
                0.0f,
                1.0f,
                static_cast<float>(static_cast<double>(property.value)) / 100.0f
            );
            parameter->beginChangeGesture();
            parameter->setValueNotifyingHost(normalizedValue);
            parameter->endChangeGesture();
            ++appliedCount;
        }

        if (appliedCount > 0)
            plugin.updateHostDisplay();
        return true;
    }

    static juce::String parameterPayloadSignature(const juce::var& parametersVar)
    {
        auto* parameterObject = parametersVar.getDynamicObject();
        if (parameterObject == nullptr)
            return {};

        juce::StringArray entries;
        for (const auto& property : parameterObject->getProperties())
        {
            const auto name = property.name.toString().trim();
            if (name.isEmpty())
                continue;
            const auto value = juce::String(static_cast<double>(property.value), 6);
            entries.add(name + "=" + value);
        }
        entries.sort(true);
        return entries.joinIntoString("|");
    }

    static juce::String graphEffectChainPayloadSignature(const juce::var& fxChainVar)
    {
        auto* fxArray = fxChainVar.getArray();
        if (fxArray == nullptr || fxArray->isEmpty())
            return {};

        juce::StringArray entries;
        entries.ensureStorageAllocated(fxArray->size());
        for (const auto& fxVar : *fxArray)
        {
            auto* fxObject = fxVar.getDynamicObject();
            if (fxObject == nullptr)
                continue;
            const auto pluginPath = normaliseVst3PathForSettings(
                fxObject->getProperty("plugin_path").toString().trim()
            );
            const auto statePath = fxObject->getProperty("state_path").toString().trim();
            const auto parameterSignature = parameterPayloadSignature(fxObject->getProperty("parameters"));
            entries.add(pluginPath + "|" + statePath + "|" + parameterSignature);
        }
        return entries.joinIntoString(">");
    }

    std::unique_ptr<juce::AudioPluginInstance> createPreparedGraphPluginInstance(
        const juce::String& pluginPath,
        const juce::String& statePath,
        const juce::var& parametersVar,
        juce::String& error
    )
    {
        const juce::File pluginFile(pluginPath);
        auto descriptions = describePlugin(pluginFile);
        if (descriptions.isEmpty())
        {
            error = "No loadable VST3 plugin description found";
            return {};
        }

        juce::String createError;
        auto instance = formatManager.createPluginInstance(*descriptions[0], currentSampleRate, currentBlockSize, createError);
        if (instance == nullptr)
        {
            error = createError.isNotEmpty() ? createError : "Could not create plugin instance";
            return {};
        }

        instance->disableNonMainBuses();
        instance->setRateAndBufferSizeDetails(currentSampleRate, currentBlockSize);
        instance->prepareToPlay(currentSampleRate, currentBlockSize);
        instance->suspendProcessing(false);

        if (statePath.isNotEmpty())
        {
            const auto stateFile = juce::File(statePath);
            if (stateFile.existsAsFile())
                loadPluginStateIntoInstance(*instance, stateFile);
        }

        if (parametersVar.isObject())
        {
            juce::String parameterError;
            int appliedParameterCount = 0;
            juce::StringArray unmatchedParameters;
            applyRequestedParameterValuesToPlugin(
                *instance,
                parametersVar,
                parameterError,
                appliedParameterCount,
                unmatchedParameters
            );
        }

        return instance;
    }

    void releasePersistentGraphEffectSlot(PersistentGraphSlot::GraphEffectSlot& slot)
    {
        if (slot.plugin != nullptr)
        {
            slot.plugin->suspendProcessing(true);
            slot.plugin->releaseResources();
        }
        slot.plugin.reset();
        slot.name.clear();
        slot.pluginPath.clear();
        slot.statePath.clear();
        slot.parameterSignature.clear();
    }

    void releasePersistentGraphSlot(PersistentGraphSlot& slot)
    {
        if (slot.plugin != nullptr)
        {
            slot.plugin->suspendProcessing(true);
            slot.plugin->releaseResources();
        }
        for (auto& effectSlot : slot.fxChain)
            releasePersistentGraphEffectSlot(effectSlot);
        slot.plugin.reset();
        slot.fxChain.clear();
        slot.name.clear();
        slot.pluginPath.clear();
        slot.statePath.clear();
        slot.parameterSignature.clear();
        slot.fxChainSignature.clear();
        slot.scheduledEvents.clear();
        slot.gain = 1.0f;
        slot.pan = 0.0f;
        slot.peakLevel = 0.0f;
    }

    void clearPersistentPluginGraph()
    {
        closeGraphEditorWindow();
        graphTransportEnabled.store(false);
        for (auto& slot : persistentGraphSlots)
            releasePersistentGraphSlot(slot);
        persistentGraphSlots.clear();
    }

    int totalPersistentGraphScheduledEventCount() const
    {
        int total = 0;
        for (const auto& slot : persistentGraphSlots)
            total += slot.scheduledEvents.size();
        return total;
    }

    void clearPersistentGraphScheduledEvents(PersistentGraphSlot& slot,
                                             int channel = 0,
                                             int64_t beforeLoopEpoch = -1,
                                             int64_t /* minFrame */ = -1)
    {
        if (channel <= 0)
        {
            slot.scheduledEvents.clear();
            return;
        }

        juce::Array<ScheduledMidiEvent> keep;
        const auto targetChannel = clampMidiChannel(channel);
        for (const auto& event : slot.scheduledEvents)
        {
            if (event.message.getChannel() != targetChannel)
            {
                keep.add(event);
                continue;
            }
            if (beforeLoopEpoch >= 0 && event.loopEpoch >= beforeLoopEpoch)
            {
                keep.add(event);
                continue;
            }
            // Old-epoch events removed unconditionally — bootstrap re-triggers
            // any notes that should still be sounding after loop wrap.
        }
        slot.scheduledEvents.swapWith(keep);
    }

    void panicPersistentGraphSlot(PersistentGraphSlot& slot, int channel)
    {
        if (slot.plugin == nullptr)
            return;

        const auto enqueueChannel = [](juce::MidiBuffer& midi, int midiChannel)
        {
            midi.addEvent(juce::MidiMessage::controllerEvent(midiChannel, 64, 0), 0);
            midi.addEvent(juce::MidiMessage::controllerEvent(midiChannel, 66, 0), 0);
            midi.addEvent(juce::MidiMessage::controllerEvent(midiChannel, 67, 0), 0);
            midi.addEvent(juce::MidiMessage::controllerEvent(midiChannel, 120, 0), 0);
            midi.addEvent(juce::MidiMessage::controllerEvent(midiChannel, 123, 0), 0);
        };

        const auto channelCount = juce::jmax(
            2,
            juce::jmax(slot.plugin->getTotalNumInputChannels(), slot.plugin->getTotalNumOutputChannels())
        );
        const auto blockSamples = juce::jmax(64, currentBlockSize > 0 ? currentBlockSize : 512);
        juce::AudioBuffer<float> processBuffer(channelCount, blockSamples);
        processBuffer.clear();
        juce::MidiBuffer midi;
        if (channel <= 0)
        {
            for (int midiChannel = 1; midiChannel <= 16; ++midiChannel)
                enqueueChannel(midi, midiChannel);
        }
        else
        {
            enqueueChannel(midi, clampMidiChannel(channel));
        }
        slot.plugin->processBlock(processBuffer, midi);
    }

    void appendPersistentGraphScheduledMidiMessages(PersistentGraphSlot& slot,
                                                    juce::MidiBuffer& destination,
                                                    int64_t blockStart,
                                                    int numSamples)
    {
        if (slot.scheduledEvents.isEmpty())
            return;

        const auto blockEnd = blockStart + juce::jmax(1, numSamples);

        // Partition in-place: compact future events to the front
        int keepCount = 0;
        graphMidiReady.clearQuick();
        for (int i = 0; i < slot.scheduledEvents.size(); ++i)
        {
            auto& event = slot.scheduledEvents.getReference(i);
            if (event.frame >= blockEnd)
            {
                if (keepCount != i)
                    slot.scheduledEvents.getReference(keepCount) = std::move(event);
                ++keepCount;
            }
            else
            {
                if (event.frame < blockStart)
                    event.frame = blockStart;
                graphMidiReady.add(std::move(event));
            }
        }
        slot.scheduledEvents.removeRange(keepCount, slot.scheduledEvents.size() - keepCount);

        if (graphMidiReady.isEmpty())
            return;

        std::sort(graphMidiReady.begin(), graphMidiReady.end(),
            [](const ScheduledMidiEvent& a, const ScheduledMidiEvent& b)
        {
            if (a.frame != b.frame)
                return a.frame < b.frame;
            if (a.priority != b.priority)
                return a.priority < b.priority;
            return a.sequence < b.sequence;
        });

        for (const auto& event : graphMidiReady)
        {
            const auto samplePosition = static_cast<int>(
                juce::jlimit<int64_t>(0, juce::jmax(0, numSamples - 1), event.frame - blockStart)
            );
            destination.addEvent(event.message, samplePosition);
        }
    }

    bool schedulePersistentGraphMidi(const juce::var& slotsVar,
                                     int64_t baseOffsetFrames,
                                     int& scheduledCount,
                                     juce::String& error)
    {
        auto* slotsArray = slotsVar.getArray();
        if (slotsArray == nullptr)
        {
            error = "schedule_graph_midi requires a slots array";
            return false;
        }

        const auto currentFrame = renderedSampleFrames.load();
        const auto resetFrame = currentFrame + baseOffsetFrames;
        scheduledCount = 0;

        for (const auto& slotVar : *slotsArray)
        {
            auto* slotObject = slotVar.getDynamicObject();
            if (slotObject == nullptr)
                continue;

            const auto slotIndex = static_cast<int>(slotObject->hasProperty("slot_index")
                ? slotObject->getProperty("slot_index")
                : juce::var(-1));
            if (!juce::isPositiveAndBelow(slotIndex, static_cast<int>(persistentGraphSlots.size())))
            {
                error = "Graph slot not loaded";
                return false;
            }

            auto& slot = persistentGraphSlots[static_cast<size_t>(slotIndex)];
            if (slot.plugin == nullptr)
            {
                error = "Graph slot not loaded";
                return false;
            }

            const auto loopEpoch = juce::jmax<int64_t>(
                0,
                static_cast<int64_t>(slotObject->hasProperty("loop_epoch")
                    ? static_cast<double>(slotObject->getProperty("loop_epoch"))
                    : 0.0)
            );
            const auto clearChannelsVar = slotObject->getProperty("clear_channels");
            const auto resetChannelsVar = slotObject->getProperty("reset_channels");
            const auto eventsVar = slotObject->getProperty("events");
            juce::Array<int> resetChannels;

            if (clearChannelsVar.isArray())
            {
                if (auto* clearArray = clearChannelsVar.getArray())
                {
                    for (const auto& channelVar : *clearArray)
                        clearPersistentGraphScheduledEvents(slot, static_cast<int>(channelVar), loopEpoch, resetFrame);
                }
            }

            if (resetChannelsVar.isArray())
            {
                if (auto* resetArray = resetChannelsVar.getArray())
                {
                    for (const auto& channelVar : *resetArray)
                    {
                        const auto channel = clampMidiChannel(static_cast<int>(channelVar));
                        clearPersistentGraphScheduledEvents(slot, channel, loopEpoch, resetFrame);
                        resetChannels.addIfNotAlreadyThere(channel);
                    }
                }
            }

            for (const auto channel : resetChannels)
                scheduledCount += appendScheduledPanicEvents(slot.scheduledEvents, channel, resetFrame, loopEpoch);

            auto* eventsArray = eventsVar.getArray();
            if (eventsArray == nullptr)
                continue;

            for (const auto& eventVar : *eventsArray)
            {
                auto* eventObject = eventVar.getDynamicObject();
                if (eventObject == nullptr)
                    continue;

                const auto type = eventObject->getProperty("type").toString().trim().toLowerCase();
                const auto channel = clampMidiChannel(static_cast<int>(eventObject->hasProperty("channel")
                    ? eventObject->getProperty("channel")
                    : juce::var(kDefaultMidiChannel)));
                const auto note = clampMidiNote(static_cast<int>(eventObject->hasProperty("note")
                    ? eventObject->getProperty("note")
                    : juce::var(60)));
                const auto velocity = clampMidiVelocity(static_cast<float>(eventObject->hasProperty("velocity")
                    ? static_cast<double>(eventObject->getProperty("velocity"))
                    : 0.0));
                const auto controller = juce::jlimit(
                    0,
                    127,
                    static_cast<int>(eventObject->hasProperty("control")
                        ? eventObject->getProperty("control")
                        : juce::var(0))
                );
                const auto controllerValue = juce::jlimit(
                    0,
                    127,
                    static_cast<int>(eventObject->hasProperty("value")
                        ? eventObject->getProperty("value")
                        : juce::var(0))
                );
                const auto sampleOffset = juce::jmax<int64_t>(
                    0,
                    static_cast<int64_t>(eventObject->hasProperty("sample_offset")
                        ? static_cast<double>(eventObject->getProperty("sample_offset"))
                        : 0.0)
                );
                const auto priority = static_cast<int>(eventObject->hasProperty("priority")
                    ? eventObject->getProperty("priority")
                    : juce::var(0));

                juce::MidiMessage message;
                if (type == "note_on")
                    message = juce::MidiMessage::noteOn(channel, note, velocity);
                else if (type == "note_off")
                    message = juce::MidiMessage::noteOff(channel, note, velocity);
                else if (type == "control_change")
                    message = juce::MidiMessage::controllerEvent(channel, controller, controllerValue);
                else
                    continue;

                const auto targetFrame = currentFrame + baseOffsetFrames + sampleOffset;
                slot.scheduledEvents.add({
                    targetFrame,
                    priority,
                    scheduledMidiSequence.fetch_add(1),
                    loopEpoch,
                    message,
                });
                ++scheduledCount;
            }
        }

        return true;
    }

    bool configurePersistentGraphEffectChain(
        PersistentGraphSlot& slot,
        const juce::var& fxChainVar,
        int trackIndex,
        juce::Array<juce::var>& trackErrors
    )
    {
        for (auto& effectSlot : slot.fxChain)
            releasePersistentGraphEffectSlot(effectSlot);
        slot.fxChain.clear();

        auto* fxArray = fxChainVar.getArray();
        if (fxArray == nullptr || fxArray->isEmpty())
            return true;

        slot.fxChain.reserve(static_cast<size_t>(fxArray->size()));
        for (int fxIndex = 0; fxIndex < fxArray->size(); ++fxIndex)
        {
            auto* fxObject = fxArray->getReference(fxIndex).getDynamicObject();
            if (fxObject == nullptr)
            {
                auto* errorObject = new juce::DynamicObject();
                errorObject->setProperty("track_index", trackIndex);
                errorObject->setProperty("fx_index", fxIndex);
                errorObject->setProperty("message", "FX payload must be an object");
                trackErrors.add(juce::var(errorObject));
                continue;
            }

            const auto pluginPath = normaliseVst3PathForSettings(
                fxObject->getProperty("plugin_path").toString().trim()
            );
            const auto statePath = fxObject->getProperty("state_path").toString().trim();
            const auto parameterSignature = parameterPayloadSignature(fxObject->getProperty("parameters"));
            const auto name = fxObject->hasProperty("name")
                ? fxObject->getProperty("name").toString().trim()
                : juce::String();

            if (pluginPath.isEmpty())
            {
                auto* errorObject = new juce::DynamicObject();
                errorObject->setProperty("track_index", trackIndex);
                errorObject->setProperty("fx_index", fxIndex);
                errorObject->setProperty("message", "Missing FX plugin path");
                trackErrors.add(juce::var(errorObject));
                continue;
            }

            juce::String createError;
            auto instance = createPreparedGraphPluginInstance(
                pluginPath,
                statePath,
                fxObject->getProperty("parameters"),
                createError
            );
            if (instance == nullptr)
            {
                auto* errorObject = new juce::DynamicObject();
                errorObject->setProperty("track_index", trackIndex);
                errorObject->setProperty("fx_index", fxIndex);
                errorObject->setProperty("plugin_path", pluginPath);
                errorObject->setProperty("message", createError.isNotEmpty() ? createError : "Could not create FX instance");
                trackErrors.add(juce::var(errorObject));
                continue;
            }

            PersistentGraphSlot::GraphEffectSlot effectSlot;
            effectSlot.name = name.isNotEmpty() ? name : juce::File(pluginPath).getFileNameWithoutExtension();
            effectSlot.pluginPath = pluginPath;
            effectSlot.statePath = statePath;
            effectSlot.parameterSignature = parameterSignature;
            effectSlot.plugin = std::move(instance);
            slot.fxChain.push_back(std::move(effectSlot));
        }

        return true;
    }

    void resetPersistentGraphSlotProcessingState(PersistentGraphSlot& slot)
    {
        if (slot.plugin != nullptr)
            slot.plugin->reset();
        for (auto& effectSlot : slot.fxChain)
        {
            if (effectSlot.plugin != nullptr)
                effectSlot.plugin->reset();
        }
        slot.peakLevel = 0.0f;
    }

    void processPersistentGraphEffectChain(PersistentGraphSlot& slot, juce::AudioBuffer<float>& stereoBuffer)
    {
        if (slot.fxChain.empty() || stereoBuffer.getNumSamples() <= 0)
            return;

        for (auto& effectSlot : slot.fxChain)
        {
            auto* plugin = effectSlot.plugin.get();
            if (plugin == nullptr)
                continue;

            const auto channelCount = juce::jmax(
                2,
                juce::jmax(plugin->getTotalNumInputChannels(), plugin->getTotalNumOutputChannels())
            );
            juce::AudioBuffer<float> processBuffer(channelCount, stereoBuffer.getNumSamples());
            processBuffer.clear();
            copyStereoBufferToPluginInput(*plugin, processBuffer, stereoBuffer);

            juce::MidiBuffer midi;
            plugin->processBlock(processBuffer, midi);
            extractPluginOutputToStereo(*plugin, processBuffer, stereoBuffer);
        }
    }

    void preparePersistentPluginGraphForPlayback()
    {
        for (auto& slot : persistentGraphSlots)
        {
            if (slot.plugin == nullptr)
                continue;
            slot.plugin->setRateAndBufferSizeDetails(currentSampleRate, currentBlockSize);
            slot.plugin->prepareToPlay(currentSampleRate, currentBlockSize);
            slot.plugin->suspendProcessing(false);
            for (auto& effectSlot : slot.fxChain)
            {
                if (effectSlot.plugin == nullptr)
                    continue;
                effectSlot.plugin->setRateAndBufferSizeDetails(currentSampleRate, currentBlockSize);
                effectSlot.plugin->prepareToPlay(currentSampleRate, currentBlockSize);
                effectSlot.plugin->suspendProcessing(false);
            }
            resetPersistentGraphSlotProcessingState(slot);
        }
    }

    void releasePersistentPluginGraphResources()
    {
        for (auto& slot : persistentGraphSlots)
        {
            if (slot.plugin == nullptr)
                continue;
            slot.plugin->suspendProcessing(true);
            slot.plugin->releaseResources();
            for (auto& effectSlot : slot.fxChain)
            {
                if (effectSlot.plugin == nullptr)
                    continue;
                effectSlot.plugin->suspendProcessing(true);
                effectSlot.plugin->releaseResources();
            }
        }
    }

    bool configurePersistentPluginGraph(const juce::var& tracksVar,
                                        juce::Array<juce::var>& trackErrors,
                                        juce::String& error)
    {
        auto* tracksArray = tracksVar.getArray();
        if (tracksArray == nullptr || tracksArray->isEmpty())
        {
            error = "configure_plugin_graph requires at least one track";
            return false;
        }

        if (persistentGraphSlots.size() > static_cast<size_t>(tracksArray->size()))
        {
            if (graphEditorSlotIndex >= tracksArray->size())
                closeGraphEditorWindow();
            for (size_t index = static_cast<size_t>(tracksArray->size()); index < persistentGraphSlots.size(); ++index)
                releasePersistentGraphSlot(persistentGraphSlots[index]);
            persistentGraphSlots.resize(static_cast<size_t>(tracksArray->size()));
        }
        else if (persistentGraphSlots.size() < static_cast<size_t>(tracksArray->size()))
        {
            persistentGraphSlots.resize(static_cast<size_t>(tracksArray->size()));
        }

        int loadedCount = 0;
        for (int index = 0; index < tracksArray->size(); ++index)
        {
            auto* trackObject = tracksArray->getReference(index).getDynamicObject();
            auto& slot = persistentGraphSlots[static_cast<size_t>(index)];
            if (trackObject == nullptr)
            {
                releasePersistentGraphSlot(slot);
                auto* errorObject = new juce::DynamicObject();
                errorObject->setProperty("track_index", index);
                errorObject->setProperty("message", "Track payload must be an object");
                trackErrors.add(juce::var(errorObject));
                continue;
            }

            const auto pluginPath = normaliseVst3PathForSettings(trackObject->getProperty("plugin_path").toString().trim());
            const auto statePath = trackObject->getProperty("state_path").toString().trim();
            const auto parameterSignature = parameterPayloadSignature(trackObject->getProperty("parameters"));
            const auto fxChainSignature = graphEffectChainPayloadSignature(trackObject->getProperty("fx_chain"));
            const auto name = trackObject->hasProperty("name")
                ? trackObject->getProperty("name").toString().trim()
                : juce::String();
            const auto gain = trackObject->hasProperty("gain")
                ? static_cast<float>(static_cast<double>(trackObject->getProperty("gain")))
                : 1.0f;
            const auto pan = trackObject->hasProperty("pan")
                ? static_cast<float>(static_cast<double>(trackObject->getProperty("pan")))
                : 0.0f;

            if (pluginPath.isEmpty())
            {
                releasePersistentGraphSlot(slot);
                auto* errorObject = new juce::DynamicObject();
                errorObject->setProperty("track_index", index);
                errorObject->setProperty("message", "Missing plugin path");
                trackErrors.add(juce::var(errorObject));
                continue;
            }

            const auto canReuse = slot.plugin != nullptr
                && slot.pluginPath == pluginPath
                && slot.statePath == statePath
                && slot.parameterSignature == parameterSignature
                && slot.fxChainSignature == fxChainSignature;
            if (canReuse)
            {
                slot.name = name.isNotEmpty() ? name : juce::File(pluginPath).getFileNameWithoutExtension();
                slot.scheduledEvents.clear();
                slot.gain = gain;
                slot.pan = pan;
                slot.peakLevel = 0.0f;
                ++loadedCount;
                continue;
            }

            if (graphEditorSlotIndex == index)
                closeGraphEditorWindow();
            releasePersistentGraphSlot(slot);

            juce::String createError;
            auto instance = createPreparedGraphPluginInstance(
                pluginPath,
                statePath,
                trackObject->getProperty("parameters"),
                createError
            );
            if (instance == nullptr)
            {
                auto* errorObject = new juce::DynamicObject();
                errorObject->setProperty("track_index", index);
                errorObject->setProperty("plugin_path", pluginPath);
                errorObject->setProperty("message", createError.isNotEmpty() ? createError : "Could not create plugin instance");
                trackErrors.add(juce::var(errorObject));
                continue;
            }

            slot.name = name.isNotEmpty() ? name : juce::File(pluginPath).getFileNameWithoutExtension();
            slot.pluginPath = pluginPath;
            slot.statePath = statePath;
            slot.parameterSignature = parameterSignature;
            slot.fxChainSignature = fxChainSignature;
            slot.plugin = std::move(instance);
            configurePersistentGraphEffectChain(slot, trackObject->getProperty("fx_chain"), index, trackErrors);
            slot.scheduledEvents.clear();
            slot.gain = gain;
            slot.pan = pan;
            slot.peakLevel = 0.0f;
            ++loadedCount;
        }

        if (loadedCount <= 0)
        {
            error = "Could not configure plugin graph";
            return false;
        }
        return true;
    }

    void persistManagedState()
    {
        if (managedStateFile == juce::File())
            return;
        savePluginStateToFile(managedStateFile);
    }

    static juce::String encodeInterleavedBusAudio(const juce::AudioBuffer<float>& buffer)
    {
        juce::MemoryOutputStream stream;
        const auto channelCount = buffer.getNumChannels();
        const auto sampleCount = buffer.getNumSamples();
        for (int sample = 0; sample < sampleCount; ++sample)
        {
            for (int channel = 0; channel < channelCount; ++channel)
            {
                const auto value = buffer.getSample(channel, sample);
                stream.write(&value, sizeof(value));
            }
        }
        return juce::Base64::toBase64(stream.getData(), stream.getDataSize());
    }

    void populateInputBuses(juce::AudioBuffer<float>& processBuffer, int numSamples, const juce::var& inputBusesVar)
    {
        auto* plugin = pluginInstance.get();
        if (plugin == nullptr || !inputBusesVar.isArray())
            return;

        if (auto* busesArray = inputBusesVar.getArray())
        {
            for (const auto& busVar : *busesArray)
            {
                auto* busObject = busVar.getDynamicObject();
                if (busObject == nullptr)
                    continue;

                const auto busIndex = static_cast<int>(busObject->getProperty("bus_index"));
                if (busIndex < 0 || busIndex >= plugin->getBusCount(true))
                    continue;

                auto busBuffer = plugin->getBusBuffer(processBuffer, true, busIndex);
                busBuffer.clear();
                if (busBuffer.getNumChannels() <= 0)
                    continue;

                const auto base64Audio = busObject->getProperty("audio_b64").toString();
                if (base64Audio.isEmpty())
                    continue;

                const auto providedChannels = juce::jmax(
                    1,
                    static_cast<int>(busObject->hasProperty("channels")
                        ? busObject->getProperty("channels")
                        : juce::var(busBuffer.getNumChannels())))
                ;

                juce::MemoryOutputStream decoded;
                if (!juce::Base64::convertFromBase64(decoded, base64Audio))
                    continue;

                const auto totalFloats = static_cast<int>(decoded.getDataSize() / sizeof(float));
                const auto availableFrames = providedChannels > 0 ? totalFloats / providedChannels : 0;
                auto* samples = static_cast<const float*>(decoded.getData());

                for (int channel = 0; channel < busBuffer.getNumChannels(); ++channel)
                {
                    const auto sourceChannel = channel < providedChannels ? channel : 0;
                    auto* destination = busBuffer.getWritePointer(channel);
                    for (int sample = 0; sample < numSamples; ++sample)
                    {
                        if (sample < availableFrames)
                            destination[sample] = samples[(sample * providedChannels) + sourceChannel];
                        else
                            destination[sample] = 0.0f;
                    }
                }
            }
        }
    }

    juce::var describeRenderedOutputBuses(juce::AudioBuffer<float>& processBuffer) const
    {
        juce::Array<juce::var> outputs;
        auto* plugin = pluginInstance.get();
        if (plugin == nullptr)
            return juce::var(outputs);

        for (int busIndex = 0; busIndex < plugin->getBusCount(false); ++busIndex)
        {
            const auto* bus = plugin->getBus(false, busIndex);
            if (bus == nullptr)
                continue;

            auto* object = new juce::DynamicObject();
            object->setProperty("bus_index", busIndex);
            object->setProperty("name", bus->getName());
            object->setProperty("is_main", bus->isMain());
            object->setProperty("enabled", bus->isEnabled());
            object->setProperty("channel_count", bus->getNumberOfChannels());

            auto busBuffer = plugin->getBusBuffer(processBuffer, false, busIndex);
            if (bus->isEnabled() && busBuffer.getNumChannels() > 0)
            {
                object->setProperty("audio_b64", encodeInterleavedBusAudio(busBuffer));
                object->setProperty("channels", busBuffer.getNumChannels());
                object->setProperty("frames", busBuffer.getNumSamples());
                object->setProperty("format", "f32le-interleaved");
            }

            outputs.add(juce::var(object));
        }

        return juce::var(outputs);
    }

    juce::var renderOfflineAudioBlock(int numSamples, const juce::var& inputBusesVar)
    {
        auto response = makeResponse(true, "Rendered audio");
        const auto channelCount = juce::jmax(
            pluginInstance != nullptr ? pluginInstance->getTotalNumInputChannels() : 0,
            pluginInstance != nullptr ? pluginInstance->getTotalNumOutputChannels() : 2
        );
        juce::AudioBuffer<float> buffer(juce::jmax(2, channelCount), juce::jmax(1, numSamples));
        buffer.clear();

        {
            juce::ScopedLock lock(pluginLock);
            if (pluginInstance == nullptr)
                return makeResponse(false, "No plugin loaded");

            populateInputBuses(buffer, numSamples, inputBusesVar);

            juce::MidiBuffer midi;
            keyboardState.processNextMidiBuffer(midi, 0, numSamples, true);
            appendPendingMidiMessages(midi);
            appendScheduledMidiMessages(midi, numSamples);
            pluginInstance->processBlock(buffer, midi);
            renderedSampleFrames.fetch_add(numSamples);
        }

        juce::AudioBuffer<float> mainBusBuffer(2, numSamples);
        mainBusBuffer.clear();
        if (pluginInstance != nullptr && pluginInstance->getBusCount(false) > 0)
        {
            auto renderedMainBus = pluginInstance->getBusBuffer(buffer, false, 0);
            if (renderedMainBus.getNumChannels() > 0)
            {
                const auto leftChannel = 0;
                const auto rightChannel = renderedMainBus.getNumChannels() > 1 ? 1 : 0;
                for (int sample = 0; sample < numSamples; ++sample)
                {
                    mainBusBuffer.setSample(0, sample, renderedMainBus.getSample(leftChannel, sample));
                    mainBusBuffer.setSample(1, sample, renderedMainBus.getSample(rightChannel, sample));
                }
            }
        }

        pluginOutputPeakLevel = peakLevelForBuffer(mainBusBuffer);
        setResponseField(response, "audio_b64", encodeInterleavedBusAudio(mainBusBuffer));
        setResponseField(response, "channels", 2);
        setResponseField(response, "format", "f32le-interleaved");
        setResponseField(response, "output_buses", describeRenderedOutputBuses(buffer));
        return response;
    }

    juce::var renderOfflinePluginGraph(const juce::var& tracksVar, int numSamples)
    {
        if (!tracksVar.isArray())
            return makeResponse(false, "render_plugin_graph requires a tracks array");

        auto response = makeResponse(true, "Rendered plugin graph");
        auto* tracksArray = tracksVar.getArray();
        if (tracksArray == nullptr || tracksArray->isEmpty())
            return makeResponse(false, "render_plugin_graph requires at least one track");

        juce::Array<juce::var> trackErrors;
        juce::String error;
        if (!configurePersistentPluginGraph(tracksVar, trackErrors, error))
        {
            auto failedResponse = makeResponse(false, error.isNotEmpty() ? error : "Could not build plugin graph");
            if (!trackErrors.isEmpty())
                setResponseField(failedResponse, "track_errors", juce::var(trackErrors));
            return failedResponse;
        }

        std::vector<OfflineGraphTrack> graphTracks;
        graphTracks.reserve(static_cast<size_t>(tracksArray->size()));
        for (int trackIndex = 0; trackIndex < tracksArray->size(); ++trackIndex)
        {
            auto* trackObject = tracksArray->getReference(trackIndex).getDynamicObject();
            if (trackObject == nullptr)
                continue;
            if (!juce::isPositiveAndBelow(trackIndex, static_cast<int>(persistentGraphSlots.size())))
                continue;

            auto& slot = persistentGraphSlots[static_cast<size_t>(trackIndex)];
            if (slot.plugin == nullptr)
                continue;

            OfflineGraphTrack graphTrack;
            graphTrack.name = slot.name;
            graphTrack.pluginPath = slot.pluginPath;
            graphTrack.slot = &slot;
            graphTrack.plugin = slot.plugin.get();
            graphTrack.gain = trackObject->hasProperty("gain")
                ? static_cast<float>(static_cast<double>(trackObject->getProperty("gain")))
                : slot.gain;
            graphTrack.pan = trackObject->hasProperty("pan")
                ? static_cast<float>(static_cast<double>(trackObject->getProperty("pan")))
                : slot.pan;

            const auto eventsVar = trackObject->getProperty("events");
            if (auto* eventsArray = eventsVar.getArray())
            {
                graphTrack.events.reserve(static_cast<size_t>(eventsArray->size()));
                for (const auto& eventVar : *eventsArray)
                {
                    auto* eventObject = eventVar.getDynamicObject();
                    if (eventObject == nullptr)
                        continue;

                    const auto type = eventObject->getProperty("type").toString().trim().toLowerCase();
                    const auto channel = clampMidiChannel(static_cast<int>(
                        eventObject->hasProperty("channel") ? eventObject->getProperty("channel") : juce::var(kDefaultMidiChannel)
                    ));
                    const auto note = clampMidiNote(static_cast<int>(
                        eventObject->hasProperty("note") ? eventObject->getProperty("note") : juce::var(60)
                    ));
                    const auto velocity = clampMidiVelocity(static_cast<float>(
                        eventObject->hasProperty("velocity") ? static_cast<double>(eventObject->getProperty("velocity")) : 0.0
                    ));
                    const auto controller = juce::jlimit(
                        0,
                        127,
                        static_cast<int>(eventObject->hasProperty("control")
                            ? eventObject->getProperty("control")
                            : juce::var(0))
                    );
                    const auto controllerValue = juce::jlimit(
                        0,
                        127,
                        static_cast<int>(eventObject->hasProperty("value")
                            ? eventObject->getProperty("value")
                            : juce::var(0))
                    );
                    const auto sampleOffset = juce::jlimit<int64_t>(
                        0,
                        juce::jmax<int64_t>(0, numSamples - 1),
                        static_cast<int64_t>(eventObject->hasProperty("sample_offset")
                            ? static_cast<double>(eventObject->getProperty("sample_offset"))
                            : 0.0)
                    );
                    const auto priority = static_cast<int>(
                        eventObject->hasProperty("priority") ? eventObject->getProperty("priority") : juce::var(0)
                    );

                    juce::MidiMessage message;
                    if (type == "note_on")
                        message = juce::MidiMessage::noteOn(channel, note, velocity);
                    else if (type == "note_off")
                        message = juce::MidiMessage::noteOff(channel, note, velocity);
                    else if (type == "control_change")
                        message = juce::MidiMessage::controllerEvent(channel, controller, controllerValue);
                    else
                        continue;

                    graphTrack.events.push_back({ sampleOffset, priority, static_cast<int64_t>(graphTrack.events.size()), 0, message });
                }
            }

            std::sort(graphTrack.events.begin(), graphTrack.events.end(), [](const ScheduledMidiEvent& a, const ScheduledMidiEvent& b)
            {
                if (a.frame != b.frame)
                    return a.frame < b.frame;
                if (a.priority != b.priority)
                    return a.priority < b.priority;
                return a.sequence < b.sequence;
            });

            graphTracks.push_back(std::move(graphTrack));
        }

        if (graphTracks.empty())
        {
            auto failedResponse = makeResponse(false, "Could not build plugin graph");
            if (!trackErrors.isEmpty())
                setResponseField(failedResponse, "track_errors", juce::var(trackErrors));
            return failedResponse;
        }

        juce::AudioBuffer<float> mixBuffer(2, juce::jmax(1, numSamples));
        mixBuffer.clear();
        const auto blockSize = juce::jmax(64, juce::jmin(juce::jmax(64, currentBlockSize > 0 ? currentBlockSize : 512), juce::jmax(1, numSamples)));

        juce::ScopedLock lock(pluginLock);
        for (int blockStart = 0; blockStart < numSamples; blockStart += blockSize)
        {
            const auto blockSamples = juce::jmin(blockSize, numSamples - blockStart);
            for (auto& graphTrack : graphTracks)
            {
                auto* plugin = graphTrack.plugin;
                if (plugin == nullptr)
                    continue;

                const auto trackChannelCount = juce::jmax(
                    2,
                    juce::jmax(plugin->getTotalNumInputChannels(), plugin->getTotalNumOutputChannels())
                );
                juce::AudioBuffer<float> processBuffer(trackChannelCount, blockSamples);
                processBuffer.clear();

                juce::MidiBuffer midi;
                for (const auto& event : graphTrack.events)
                {
                    if (event.frame < blockStart || event.frame >= blockStart + blockSamples)
                        continue;
                    midi.addEvent(event.message, static_cast<int>(event.frame - blockStart));
                }

                plugin->processBlock(processBuffer, midi);

                juce::AudioBuffer<float> stereoBuffer(2, blockSamples);
                extractPluginOutputToStereo(*plugin, processBuffer, stereoBuffer);
                processPersistentGraphEffectChain(*graphTrack.slot, stereoBuffer);
                applyOutputMixToBuffer(stereoBuffer, graphTrack.gain, graphTrack.pan);
                graphTrack.slot->peakLevel = peakLevelForBuffer(stereoBuffer);
                mixBuffer.addFrom(0, blockStart, stereoBuffer, 0, 0, blockSamples);
                mixBuffer.addFrom(1, blockStart, stereoBuffer, 1, 0, blockSamples);
            }
        }

        setResponseField(response, "audio_b64", encodeInterleavedBusAudio(mixBuffer));
        setResponseField(response, "channels", 2);
        setResponseField(response, "format", "f32le-interleaved");
        setResponseField(response, "rendered_track_count", static_cast<int>(graphTracks.size()));
        if (!trackErrors.isEmpty())
            setResponseField(response, "track_errors", juce::var(trackErrors));
        return response;
    }

    void processPersistentPluginGraphTransport(juce::AudioBuffer<float>& buffer,
                                               int numSamples,
                                               int64_t blockStart)
    {
        if (persistentGraphSlots.empty())
            return;

        // Reuse pre-allocated stereo scratch buffer
        graphStereoBuffer.setSize(2, numSamples, false, false, true);

        for (auto& slot : persistentGraphSlots)
        {
            auto* plugin = slot.plugin.get();
            if (plugin == nullptr)
                continue;

            const auto trackChannelCount = juce::jmax(
                2,
                juce::jmax(plugin->getTotalNumInputChannels(), plugin->getTotalNumOutputChannels())
            );
            // Reuse pre-allocated process buffer, resizing only when channel count grows
            graphProcessBuffer.setSize(trackChannelCount, numSamples, false, false, true);
            graphProcessBuffer.clear();

            juce::MidiBuffer midi;
            appendPersistentGraphScheduledMidiMessages(slot, midi, blockStart, numSamples);
            plugin->processBlock(graphProcessBuffer, midi);

            graphStereoBuffer.clear();
            extractPluginOutputToStereo(*plugin, graphProcessBuffer, graphStereoBuffer);
            processPersistentGraphEffectChain(slot, graphStereoBuffer);
            applyOutputMixToBuffer(graphStereoBuffer, slot.gain, slot.pan);
            slot.peakLevel = peakLevelForBuffer(graphStereoBuffer);
            buffer.addFrom(0, 0, graphStereoBuffer, 0, 0, numSamples);
            buffer.addFrom(1, 0, graphStereoBuffer, 1, 0, numSamples);
        }
    }

    void releaseAudioEngineEffectSlot(AudioEngineTrack::EffectSlot& slot)
    {
        if (slot.plugin != nullptr)
        {
            slot.plugin->suspendProcessing(true);
            slot.plugin->releaseResources();
        }
        slot.plugin.reset();
    }

    void releaseAudioEngineTrack(AudioEngineTrack& track)
    {
        if (track.plugin != nullptr)
        {
            track.plugin->suspendProcessing(true);
            track.plugin->releaseResources();
        }
        track.kind = AudioEngineTrackKind::instrument;
        track.plugin.reset();
        for (auto& effectSlot : track.fxChain)
            releaseAudioEngineEffectSlot(effectSlot);
        track.fxChain.clear();
        track.notes.clear();
        track.audioClips.clear();
        track.automationLanes.clear();
        track.baseVolume = 1.0f;
        track.basePan = 0.0f;
        track.baseOutputGainDb = 0.0f;
        track.processBuffer.setSize(0, 0);
        track.stereoBuffer.setSize(0, 0);
        track.peakLevel = 0.0f;
    }

    void clearAudioEngine()
    {
        audioEngine.running = false;
        audioEngine.bootstrapPending = false;
        for (auto& track : audioEngine.tracks)
            releaseAudioEngineTrack(track);
        audioEngine.tracks.clear();
        audioEngine.positionFrame = 0;
        audioEngine.metronomeEnabled = false;
        audioEngine.metronomeClickBuffer.setSize(0, 0);
        audioEngine.metronomeAccentBuffer.setSize(0, 0);
    }

    void resetAudioEngineTrackProcessingState(AudioEngineTrack& track)
    {
        if (track.plugin != nullptr)
            track.plugin->reset();
        for (auto& effectSlot : track.fxChain)
        {
            if (effectSlot.plugin != nullptr)
                effectSlot.plugin->reset();
        }
        for (auto& lane : track.automationLanes)
            lane.lastAppliedValue = std::numeric_limits<float>::quiet_NaN();
        track.peakLevel = 0.0f;
    }

    void resetAudioEngineProcessingState()
    {
        for (auto& track : audioEngine.tracks)
            resetAudioEngineTrackProcessingState(track);
    }

    static float evaluateAudioEngineAutomationLane(const AudioEngineAutomationLane& lane, double tick)
    {
        if (lane.points.empty())
            return lane.defaultValue;

        if (lane.points.size() == 1)
        {
            const auto& point = lane.points.front();
            return tick >= static_cast<double>(point.tick) ? point.value : lane.defaultValue;
        }

        auto previousTick = 0.0;
        auto previousValue = static_cast<double>(lane.defaultValue);
        for (const auto& point : lane.points)
        {
            const auto pointTick = static_cast<double>(point.tick);
            const auto pointValue = static_cast<double>(point.value);
            if (pointTick <= 0.0)
            {
                previousTick = pointTick;
                previousValue = pointValue;
                if (tick <= 0.0)
                    return point.value;
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

    static bool audioEngineTrackHasParameterAutomation(const AudioEngineTrack& track)
    {
        return std::any_of(track.automationLanes.begin(), track.automationLanes.end(), [](const AudioEngineAutomationLane& lane)
        {
            return lane.type == AudioEngineAutomationTargetType::vstParameter
                && lane.parameterIndex >= 0
                && !lane.points.empty();
        });
    }

    static void applyAudioEngineTrackParameterAutomation(AudioEngineTrack& track, double tick)
    {
        auto* plugin = track.plugin.get();
        if (plugin == nullptr)
            return;

        auto parameters = plugin->getParameters();
        for (auto& lane : track.automationLanes)
        {
            if (lane.type != AudioEngineAutomationTargetType::vstParameter
                || !juce::isPositiveAndBelow(lane.parameterIndex, parameters.size()))
            {
                continue;
            }

            auto* parameter = parameters[lane.parameterIndex];
            if (parameter == nullptr)
                continue;

            const auto value = juce::jlimit(0.0f, 100.0f, evaluateAudioEngineAutomationLane(lane, tick));
            if (std::isfinite(lane.lastAppliedValue)
                && std::abs(lane.lastAppliedValue - value) < 0.001f)
            {
                continue;
            }

            parameter->setValueNotifyingHost(juce::jlimit(0.0f, 1.0f, value / 100.0f));
            lane.lastAppliedValue = value;
        }
    }

    static void applyAudioEngineTrackMixAutomation(const AudioEngineState& engineState,
                                                   AudioEngineTrack& track,
                                                   juce::AudioBuffer<float>& stereoBuffer,
                                                   int64_t chunkStartFrame,
                                                   double sampleRate)
    {
        if (stereoBuffer.getNumSamples() <= 0)
            return;

        const AudioEngineAutomationLane* volumeLane = nullptr;
        const AudioEngineAutomationLane* panLane = nullptr;
        const AudioEngineAutomationLane* outputGainLane = nullptr;
        for (const auto& lane : track.automationLanes)
        {
            if (lane.points.empty())
                continue;
            if (lane.type == AudioEngineAutomationTargetType::volume)
                volumeLane = &lane;
            else if (lane.type == AudioEngineAutomationTargetType::pan)
                panLane = &lane;
            else if (lane.type == AudioEngineAutomationTargetType::outputGainDb)
                outputGainLane = &lane;
        }

        if (volumeLane == nullptr && panLane == nullptr && outputGainLane == nullptr)
        {
            const auto gain = juce::jmax(0.0f, track.baseVolume * std::pow(10.0f, track.baseOutputGainDb / 20.0f));
            applyOutputMixToBuffer(stereoBuffer, gain, track.basePan);
            return;
        }

        auto* left = stereoBuffer.getWritePointer(0);
        auto* right = stereoBuffer.getWritePointer(1);
        for (int sample = 0; sample < stereoBuffer.getNumSamples(); ++sample)
        {
            const auto tick = engineState.frameToTickDouble(chunkStartFrame + static_cast<int64_t>(sample), sampleRate);
            const auto volume = volumeLane != nullptr
                ? evaluateAudioEngineAutomationLane(*volumeLane, tick)
                : track.baseVolume;
            const auto pan = panLane != nullptr
                ? evaluateAudioEngineAutomationLane(*panLane, tick)
                : track.basePan;
            const auto outputGainDb = outputGainLane != nullptr
                ? evaluateAudioEngineAutomationLane(*outputGainLane, tick)
                : track.baseOutputGainDb;
            const auto gain = juce::jmax(0.0f, volume * std::pow(10.0f, outputGainDb / 20.0f));
            const auto clampedPan = juce::jlimit(-1.0f, 1.0f, pan);
            const auto angle = (clampedPan + 1.0f) * juce::MathConstants<float>::pi * 0.25f;
            left[sample] *= gain * std::cos(angle);
            right[sample] *= gain * std::sin(angle);
        }
    }

    void processAudioEngineEffectChain(AudioEngineTrack& track, juce::AudioBuffer<float>& stereoBuffer)
    {
        if (track.fxChain.empty() || stereoBuffer.getNumSamples() <= 0)
            return;

        for (auto& effectSlot : track.fxChain)
        {
            auto* plugin = effectSlot.plugin.get();
            if (plugin == nullptr)
                continue;

            const auto channelCount = juce::jmax(
                2,
                juce::jmax(plugin->getTotalNumInputChannels(), plugin->getTotalNumOutputChannels())
            );
            juce::AudioBuffer<float> processBuffer(channelCount, stereoBuffer.getNumSamples());
            processBuffer.clear();
            copyStereoBufferToPluginInput(*plugin, processBuffer, stereoBuffer);

            juce::MidiBuffer midi;
            plugin->processBlock(processBuffer, midi);
            extractPluginOutputToStereo(*plugin, processBuffer, stereoBuffer);
        }
    }

    void prepareAudioEngineForPlayback()
    {
        prepareAudioEngineMetronome();
        for (auto& track : audioEngine.tracks)
        {
            if (track.plugin == nullptr)
            {
                track.processBuffer.setSize(0, 0);
            }
            else
            {
                track.plugin->setRateAndBufferSizeDetails(currentSampleRate, currentBlockSize);
                track.plugin->prepareToPlay(currentSampleRate, currentBlockSize);
                track.plugin->suspendProcessing(false);
            }
            for (auto& effectSlot : track.fxChain)
            {
                if (effectSlot.plugin == nullptr)
                    continue;
                effectSlot.plugin->setRateAndBufferSizeDetails(currentSampleRate, currentBlockSize);
                effectSlot.plugin->prepareToPlay(currentSampleRate, currentBlockSize);
                effectSlot.plugin->suspendProcessing(false);
            }
            resetAudioEngineTrackProcessingState(track);
        }
    }

    void releaseAudioEngineResources()
    {
        for (auto& track : audioEngine.tracks)
        {
            if (track.plugin != nullptr)
            {
                track.plugin->suspendProcessing(true);
                track.plugin->releaseResources();
            }
            for (auto& effectSlot : track.fxChain)
            {
                if (effectSlot.plugin != nullptr)
                {
                    effectSlot.plugin->suspendProcessing(true);
                    effectSlot.plugin->releaseResources();
                }
            }
        }
        audioEngine.metronomeClickBuffer.setSize(0, 0);
        audioEngine.metronomeAccentBuffer.setSize(0, 0);
    }

    bool configureAudioEngineTrackEffectChain(AudioEngineTrack& track,
                                              const juce::var& fxChainVar,
                                              int trackIndex,
                                              juce::Array<juce::var>& trackErrors)
    {
        for (auto& effectSlot : track.fxChain)
            releaseAudioEngineEffectSlot(effectSlot);
        track.fxChain.clear();

        auto* fxArray = fxChainVar.getArray();
        if (fxArray == nullptr || fxArray->isEmpty())
            return true;

        track.fxChain.reserve(static_cast<size_t>(fxArray->size()));
        for (int fxIndex = 0; fxIndex < fxArray->size(); ++fxIndex)
        {
            auto* fxObject = fxArray->getReference(fxIndex).getDynamicObject();
            if (fxObject == nullptr)
            {
                auto* errorObject = new juce::DynamicObject();
                errorObject->setProperty("track_index", trackIndex);
                errorObject->setProperty("fx_index", fxIndex);
                errorObject->setProperty("message", "FX payload must be an object");
                trackErrors.add(juce::var(errorObject));
                continue;
            }

            const auto pluginPath = normaliseVst3PathForSettings(
                fxObject->getProperty("plugin_path").toString().trim()
            );
            const auto statePath = fxObject->getProperty("state_path").toString().trim();
            const auto parameterSignature = parameterPayloadSignature(fxObject->getProperty("parameters"));
            const auto name = fxObject->hasProperty("name")
                ? fxObject->getProperty("name").toString().trim()
                : juce::String();

            if (pluginPath.isEmpty())
            {
                auto* errorObject = new juce::DynamicObject();
                errorObject->setProperty("track_index", trackIndex);
                errorObject->setProperty("fx_index", fxIndex);
                errorObject->setProperty("message", "Missing FX plugin path");
                trackErrors.add(juce::var(errorObject));
                continue;
            }

            juce::String createError;
            auto instance = createPreparedGraphPluginInstance(
                pluginPath,
                statePath,
                fxObject->getProperty("parameters"),
                createError
            );
            if (instance == nullptr)
            {
                auto* errorObject = new juce::DynamicObject();
                errorObject->setProperty("track_index", trackIndex);
                errorObject->setProperty("fx_index", fxIndex);
                errorObject->setProperty("plugin_path", pluginPath);
                errorObject->setProperty("message", createError.isNotEmpty() ? createError : "Could not create FX instance");
                trackErrors.add(juce::var(errorObject));
                continue;
            }

            AudioEngineTrack::EffectSlot effectSlot;
            effectSlot.name = name.isNotEmpty() ? name : juce::File(pluginPath).getFileNameWithoutExtension();
            effectSlot.pluginPath = pluginPath;
            effectSlot.statePath = statePath;
            effectSlot.parameterSignature = parameterSignature;
            effectSlot.plugin = std::move(instance);
            track.fxChain.push_back(std::move(effectSlot));
        }

        return true;
    }

    bool configureAudioEngine(const juce::var& tracksVar,
                              double bpm,
                              int ticksPerBeat,
                              bool loopEnabled,
                              int64_t loopStartTick,
                              int64_t loopEndTick,
                              bool metronomeEnabled,
                              juce::Array<juce::var>& trackErrors,
                              juce::String& error)
    {
        auto* tracksArray = tracksVar.getArray();
        const auto requestedTrackCount = tracksArray != nullptr ? tracksArray->size() : 0;
        if (requestedTrackCount <= 0 && !metronomeEnabled)
        {
            error = "set_audio_engine_state requires track or metronome state";
            return false;
        }

        const auto shouldPrepare = currentSampleRate > 0.0 && currentBlockSize > 0;
        if (shouldPrepare)
            releaseAudioEngineResources();

        if (audioEngine.tracks.size() > static_cast<size_t>(requestedTrackCount))
        {
            for (size_t index = static_cast<size_t>(requestedTrackCount); index < audioEngine.tracks.size(); ++index)
                releaseAudioEngineTrack(audioEngine.tracks[index]);
            audioEngine.tracks.resize(static_cast<size_t>(requestedTrackCount));
        }
        else if (audioEngine.tracks.size() < static_cast<size_t>(requestedTrackCount))
        {
            audioEngine.tracks.resize(static_cast<size_t>(requestedTrackCount));
        }

        int loadedCount = 0;
        for (int trackIndex = 0; trackIndex < requestedTrackCount; ++trackIndex)
        {
            auto* trackObject = tracksArray->getReference(trackIndex).getDynamicObject();
            auto& track = audioEngine.tracks[static_cast<size_t>(trackIndex)];
            if (trackObject == nullptr)
            {
                releaseAudioEngineTrack(track);
                auto* errorObject = new juce::DynamicObject();
                errorObject->setProperty("track_index", trackIndex);
                errorObject->setProperty("message", "Track payload must be an object");
                trackErrors.add(juce::var(errorObject));
                continue;
            }

            const auto trackType = trackObject->hasProperty("track_type")
                ? trackObject->getProperty("track_type").toString().trim().toLowerCase()
                : juce::String("instrument");
            const auto requestedKind = trackType == "audio"
                ? AudioEngineTrackKind::audio
                : AudioEngineTrackKind::instrument;
            const auto pluginPath = normaliseVst3PathForSettings(trackObject->getProperty("plugin_path").toString().trim());
            const auto statePath = trackObject->getProperty("state_path").toString().trim();
            const auto parameterSignature = parameterPayloadSignature(trackObject->getProperty("parameters"));
            const auto fxChainSignature = graphEffectChainPayloadSignature(trackObject->getProperty("fx_chain"));
            const auto name = trackObject->hasProperty("name")
                ? trackObject->getProperty("name").toString().trim()
                : juce::String();
            const auto baseVolume = trackObject->hasProperty("volume")
                ? static_cast<float>(static_cast<double>(trackObject->getProperty("volume")))
                : (trackObject->hasProperty("gain")
                    ? static_cast<float>(static_cast<double>(trackObject->getProperty("gain")))
                    : 1.0f);
            const auto pan = trackObject->hasProperty("pan")
                ? static_cast<float>(static_cast<double>(trackObject->getProperty("pan")))
                : 0.0f;
            const auto outputGainDb = trackObject->hasProperty("output_gain_db")
                ? static_cast<float>(static_cast<double>(trackObject->getProperty("output_gain_db")))
                : 0.0f;

            std::vector<AudioEngineTrackNote> notes;
            if (auto* notesArray = trackObject->getProperty("notes").getArray())
            {
                notes.reserve(static_cast<size_t>(notesArray->size()));
                for (const auto& noteVar : *notesArray)
                {
                    auto* noteObject = noteVar.getDynamicObject();
                    if (noteObject == nullptr)
                        continue;

                    AudioEngineTrackNote note;
                    note.startTick = static_cast<int>(noteObject->hasProperty("start_tick")
                        ? noteObject->getProperty("start_tick")
                        : juce::var(0));
                    note.endTick = static_cast<int>(noteObject->hasProperty("end_tick")
                        ? noteObject->getProperty("end_tick")
                        : juce::var(note.startTick + 1));
                    note.pitch = clampMidiNote(static_cast<int>(noteObject->hasProperty("pitch")
                        ? noteObject->getProperty("pitch")
                        : juce::var(60)));
                    note.velocity = juce::jlimit(1, 127, static_cast<int>(noteObject->hasProperty("velocity")
                        ? noteObject->getProperty("velocity")
                        : juce::var(100)));
                    note.channel = clampMidiChannel(static_cast<int>(noteObject->hasProperty("channel")
                        ? noteObject->getProperty("channel")
                        : juce::var(kDefaultMidiChannel)));
                    note.endTick = juce::jmax(note.startTick + 1, note.endTick);
                    notes.push_back(note);
                }
            }
            std::sort(notes.begin(), notes.end(), [](const AudioEngineTrackNote& a, const AudioEngineTrackNote& b)
            {
                if (a.startTick != b.startTick)
                    return a.startTick < b.startTick;
                if (a.endTick != b.endTick)
                    return a.endTick < b.endTick;
                if (a.channel != b.channel)
                    return a.channel < b.channel;
                return a.pitch < b.pitch;
            });

            std::vector<AudioEngineAudioClip> audioClips;
            if (requestedKind == AudioEngineTrackKind::audio)
            {
                if (auto* clipsArray = trackObject->getProperty("clips").getArray())
                {
                    audioClips.reserve(static_cast<size_t>(clipsArray->size()));
                    for (int clipIndex = 0; clipIndex < clipsArray->size(); ++clipIndex)
                    {
                        auto* clipObject = clipsArray->getReference(clipIndex).getDynamicObject();
                        if (clipObject == nullptr)
                            continue;

                        const auto clipPath = clipObject->getProperty("path").toString().trim();
                        const auto startFrame = clipObject->hasProperty("start_frame")
                            ? static_cast<int64_t>(static_cast<double>(clipObject->getProperty("start_frame")))
                            : int64_t(0);
                        if (clipPath.isEmpty())
                        {
                            auto* errorObject = new juce::DynamicObject();
                            errorObject->setProperty("track_index", trackIndex);
                            errorObject->setProperty("clip_index", clipIndex);
                            errorObject->setProperty("message", "Missing audio clip path");
                            trackErrors.add(juce::var(errorObject));
                            continue;
                        }

                        AudioEngineAudioClip clip;
                        clip.path = clipPath;
                        clip.startFrame = juce::jmax<int64_t>(0, startFrame);
                        juce::String clipError;
                        if (!loadAudioEngineClipBuffer(clipPath, clip.audioData, clipError))
                        {
                            auto* errorObject = new juce::DynamicObject();
                            errorObject->setProperty("track_index", trackIndex);
                            errorObject->setProperty("clip_index", clipIndex);
                            errorObject->setProperty("path", clipPath);
                            errorObject->setProperty("message", clipError.isNotEmpty() ? clipError : "Could not load audio clip");
                            trackErrors.add(juce::var(errorObject));
                            continue;
                        }

                        audioClips.push_back(std::move(clip));
                    }
                }

                std::sort(audioClips.begin(), audioClips.end(), [](const AudioEngineAudioClip& a, const AudioEngineAudioClip& b)
                {
                    if (a.startFrame != b.startFrame)
                        return a.startFrame < b.startFrame;
                    return a.path.compareNatural(b.path) < 0;
                });
            }

            if (requestedKind == AudioEngineTrackKind::instrument && pluginPath.isEmpty())
            {
                releaseAudioEngineTrack(track);
                auto* errorObject = new juce::DynamicObject();
                errorObject->setProperty("track_index", trackIndex);
                errorObject->setProperty("message", "Missing plugin path");
                trackErrors.add(juce::var(errorObject));
                continue;
            }

            const auto canReuse = track.kind == requestedKind
                && track.statePath == statePath
                && track.parameterSignature == parameterSignature
                && track.fxChainSignature == fxChainSignature
                && (
                    (requestedKind == AudioEngineTrackKind::instrument
                        && track.plugin != nullptr
                        && track.pluginPath == pluginPath)
                    || (requestedKind == AudioEngineTrackKind::audio
                        && track.plugin == nullptr)
                );

            if (!canReuse)
            {
                releaseAudioEngineTrack(track);
                track.kind = requestedKind;

                if (requestedKind == AudioEngineTrackKind::instrument)
                {
                    juce::String createError;
                    auto instance = createPreparedGraphPluginInstance(
                        pluginPath,
                        statePath,
                        trackObject->getProperty("parameters"),
                        createError
                    );
                    if (instance == nullptr)
                    {
                        auto* errorObject = new juce::DynamicObject();
                        errorObject->setProperty("track_index", trackIndex);
                        errorObject->setProperty("plugin_path", pluginPath);
                        errorObject->setProperty("message", createError.isNotEmpty() ? createError : "Could not create plugin instance");
                        trackErrors.add(juce::var(errorObject));
                        continue;
                    }

                    track.pluginPath = pluginPath;
                    track.statePath = statePath;
                    track.parameterSignature = parameterSignature;
                    track.fxChainSignature = fxChainSignature;
                    track.plugin = std::move(instance);
                }
                else
                {
                    track.pluginPath = {};
                    track.statePath = statePath;
                    track.parameterSignature = parameterSignature;
                    track.fxChainSignature = fxChainSignature;
                }

                configureAudioEngineTrackEffectChain(track, trackObject->getProperty("fx_chain"), trackIndex, trackErrors);
            }

            std::vector<AudioEngineAutomationLane> automationLanes;
            if (auto* automationArray = trackObject->getProperty("automation").getArray())
            {
                automationLanes.reserve(static_cast<size_t>(automationArray->size()));
                for (const auto& laneVar : *automationArray)
                {
                    auto* laneObject = laneVar.getDynamicObject();
                    if (laneObject == nullptr)
                        continue;

                    const auto target = laneObject->getProperty("target").toString().trim();
                    if (target.isEmpty())
                        continue;

                    AudioEngineAutomationLane lane;
                    lane.target = target;
                    if (target.equalsIgnoreCase("volume"))
                    {
                        lane.type = AudioEngineAutomationTargetType::volume;
                        lane.defaultValue = baseVolume;
                    }
                    else if (target.equalsIgnoreCase("pan"))
                    {
                        lane.type = AudioEngineAutomationTargetType::pan;
                        lane.defaultValue = pan;
                    }
                    else if (target.equalsIgnoreCase("vsti_output_gain_db"))
                    {
                        lane.type = AudioEngineAutomationTargetType::outputGainDb;
                        lane.defaultValue = outputGainDb;
                    }
                    else if (target.startsWithIgnoreCase("vsti_parameter:"))
                    {
                        if (requestedKind != AudioEngineTrackKind::instrument || track.plugin == nullptr)
                            continue;
                        lane.type = AudioEngineAutomationTargetType::vstParameter;
                        lane.parameterName = target.fromFirstOccurrenceOf(":", false, false).trim();
                        lane.parameterIndex = track.plugin != nullptr
                            ? resolvePluginParameterIndex(*track.plugin, lane.parameterName)
                            : -1;
                        if (lane.parameterName.isEmpty() || lane.parameterIndex < 0)
                        {
                            auto* errorObject = new juce::DynamicObject();
                            errorObject->setProperty("track_index", trackIndex);
                            errorObject->setProperty("target", target);
                            errorObject->setProperty("message", "Unknown automation parameter");
                            trackErrors.add(juce::var(errorObject));
                            continue;
                        }
                        auto parameters = track.plugin->getParameters();
                        if (juce::isPositiveAndBelow(lane.parameterIndex, parameters.size())
                            && parameters[lane.parameterIndex] != nullptr)
                        {
                            lane.defaultValue = parameters[lane.parameterIndex]->getValue() * 100.0f;
                        }
                    }
                    else
                    {
                        continue;
                    }

                    std::vector<AudioEngineAutomationPoint> rawPoints;
                    if (auto* pointsArray = laneObject->getProperty("points").getArray())
                    {
                        for (const auto& pointVar : *pointsArray)
                        {
                            auto* pointObject = pointVar.getDynamicObject();
                            if (pointObject == nullptr)
                                continue;

                            AudioEngineAutomationPoint point;
                            point.tick = juce::jmax(0, static_cast<int>(pointObject->hasProperty("tick")
                                ? pointObject->getProperty("tick")
                                : juce::var(0)));
                            point.value = static_cast<float>(static_cast<double>(pointObject->hasProperty("value")
                                ? pointObject->getProperty("value")
                                : juce::var(lane.defaultValue)));
                            rawPoints.push_back(point);
                        }
                    }

                    std::sort(rawPoints.begin(), rawPoints.end(), [](const AudioEngineAutomationPoint& a, const AudioEngineAutomationPoint& b)
                    {
                        return a.tick < b.tick;
                    });
                    for (const auto& point : rawPoints)
                    {
                        if (!lane.points.empty() && lane.points.back().tick == point.tick)
                            lane.points.back().value = point.value;
                        else
                            lane.points.push_back(point);
                    }

                    if (!lane.points.empty())
                        automationLanes.push_back(std::move(lane));
                }
            }

            track.kind = requestedKind;
            track.name = name.isNotEmpty()
                ? name
                : (requestedKind == AudioEngineTrackKind::instrument
                    ? juce::File(pluginPath).getFileNameWithoutExtension()
                    : "Audio Track " + juce::String(trackIndex + 1));
            track.notes = requestedKind == AudioEngineTrackKind::instrument
                ? std::move(notes)
                : std::vector<AudioEngineTrackNote>{};
            track.audioClips = requestedKind == AudioEngineTrackKind::audio
                ? std::move(audioClips)
                : std::vector<AudioEngineAudioClip>{};
            track.automationLanes = std::move(automationLanes);
            track.baseVolume = baseVolume;
            track.basePan = pan;
            track.baseOutputGainDb = outputGainDb;
            track.peakLevel = 0.0f;
            if (requestedKind == AudioEngineTrackKind::instrument)
            {
                if (track.plugin != nullptr)
                    ++loadedCount;
            }
            else if (!track.audioClips.empty())
            {
                ++loadedCount;
            }
        }

        if (loadedCount <= 0 && !metronomeEnabled)
        {
            error = "Could not configure audio engine";
            clearAudioEngine();
            return false;
        }

        audioEngine.bpm = juce::jmax(1.0, bpm);
        audioEngine.ticksPerBeat = juce::jmax(1, ticksPerBeat);
        audioEngine.loopEnabled = loopEnabled;
        audioEngine.loopStartTick = loopStartTick;
        audioEngine.loopEndTick = loopEndTick;
        audioEngine.metronomeEnabled = metronomeEnabled;
        audioEngine.running = false;
        audioEngine.bootstrapPending = false;
        audioEngine.positionFrame = 0;

        if (shouldPrepare)
            prepareAudioEngineForPlayback();

        return true;
    }

    void appendAudioEngineNoteEvents(AudioEngineTrack& track,
                                     juce::MidiBuffer& midi,
                                     int64_t chunkStartFrame,
                                     int chunkSamples,
                                     double sampleRate,
                                     bool bootstrapActive,
                                     bool wrapAfterChunk,
                                     int64_t loopEndFrame)
    {
        if (track.notes.empty())
            return;

        const auto chunkEndFrame = chunkStartFrame + chunkSamples;
        const auto noteOffAdvance = juce::jmax<int64_t>(1, static_cast<int64_t>(sampleRate) / 1000);
        const auto looping = audioEngine.loopEnabled && loopEndFrame > audioEngine.tickToFrame(audioEngine.loopStartTick, sampleRate);

        for (const auto& note : track.notes)
        {
            const auto noteStartFrame = audioEngine.tickToFrame(note.startTick, sampleRate);
            auto noteEndFrame = juce::jmax<int64_t>(
                noteStartFrame + 1,
                audioEngine.tickToFrame(note.endTick, sampleRate)
            );
            if (looping && noteStartFrame < loopEndFrame && noteEndFrame > loopEndFrame)
                noteEndFrame = loopEndFrame;

            const auto noteOffFrame = juce::jmax<int64_t>(noteStartFrame, noteEndFrame - noteOffAdvance);
            if (bootstrapActive && noteStartFrame < chunkStartFrame && noteOffFrame > chunkStartFrame)
            {
                midi.addEvent(
                    juce::MidiMessage::noteOn(note.channel, note.pitch, static_cast<float>(note.velocity) / 127.0f),
                    0
                );
            }
            else if (noteStartFrame >= chunkStartFrame && noteStartFrame < chunkEndFrame)
            {
                midi.addEvent(
                    juce::MidiMessage::noteOn(note.channel, note.pitch, static_cast<float>(note.velocity) / 127.0f),
                    static_cast<int>(noteStartFrame - chunkStartFrame)
                );
            }

            if (noteOffFrame >= chunkStartFrame && noteOffFrame < chunkEndFrame)
            {
                midi.addEvent(
                    juce::MidiMessage::noteOff(note.channel, note.pitch, 0.0f),
                    static_cast<int>(noteOffFrame - chunkStartFrame)
                );
            }
        }

        if (wrapAfterChunk)
        {
            for (const auto& note : track.notes)
            {
                midi.addEvent(
                    juce::MidiMessage::noteOff(note.channel, note.pitch, 0.0f),
                    juce::jmax(0, chunkSamples - 1)
                );
            }
        }
    }

    void processAudioEngine(juce::AudioBuffer<float>& buffer, int numSamples)
    {
        if (!audioEngine.running || numSamples <= 0)
            return;

        const auto sampleRate = juce::jmax(1.0, currentSampleRate);
        const auto loopStartFrame = audioEngine.tickToFrame(audioEngine.loopStartTick, sampleRate);
        const auto loopEndFrame = audioEngine.tickToFrame(audioEngine.loopEndTick, sampleRate);
        const auto looping = audioEngine.loopEnabled && loopEndFrame > loopStartFrame;

        int samplesRemaining = numSamples;
        int sampleOffset = 0;

        while (samplesRemaining > 0)
        {
            int chunkSamples = samplesRemaining;
            if (looping)
            {
                const auto framesUntilLoopEnd = loopEndFrame - audioEngine.positionFrame;
                if (framesUntilLoopEnd > 0 && framesUntilLoopEnd < chunkSamples)
                    chunkSamples = static_cast<int>(framesUntilLoopEnd);
            }

            const auto chunkStartFrame = audioEngine.positionFrame;
            const auto wrapAfterChunk = looping && (chunkStartFrame + chunkSamples) >= loopEndFrame;
            const auto bootstrapActive = audioEngine.bootstrapPending;

            for (auto& track : audioEngine.tracks)
            {
                track.stereoBuffer.setSize(2, chunkSamples, false, false, true);
                track.stereoBuffer.clear();
                if (track.kind == AudioEngineTrackKind::instrument)
                {
                    auto* plugin = track.plugin.get();
                    if (plugin == nullptr)
                        continue;

                    const auto trackChannelCount = juce::jmax(
                        2,
                        juce::jmax(plugin->getTotalNumInputChannels(), plugin->getTotalNumOutputChannels())
                    );
                    track.processBuffer.setSize(trackChannelCount, chunkSamples, false, false, true);
                    track.processBuffer.clear();

                    juce::MidiBuffer midi;
                    appendAudioEngineNoteEvents(
                        track,
                        midi,
                        chunkStartFrame,
                        chunkSamples,
                        sampleRate,
                        bootstrapActive,
                        wrapAfterChunk,
                        loopEndFrame
                    );

                    applyAudioEngineTrackParameterAutomation(
                        track,
                        audioEngine.frameToTickDouble(chunkStartFrame, sampleRate)
                    );
                    plugin->processBlock(track.processBuffer, midi);
                    extractPluginOutputToStereo(*plugin, track.processBuffer, track.stereoBuffer);
                }
                else
                {
                    renderAudioEngineTrackClips(track, chunkStartFrame, chunkSamples);
                }

                processAudioEngineEffectChain(track, track.stereoBuffer);
                applyAudioEngineTrackMixAutomation(audioEngine, track, track.stereoBuffer, chunkStartFrame, sampleRate);
                track.peakLevel = peakLevelForBuffer(track.stereoBuffer);
                buffer.addFrom(0, sampleOffset, track.stereoBuffer, 0, 0, chunkSamples);
                buffer.addFrom(1, sampleOffset, track.stereoBuffer, 1, 0, chunkSamples);
            }

            if (audioEngine.metronomeEnabled)
                mixAudioEngineMetronomeChunk(buffer, sampleOffset, chunkStartFrame, chunkSamples, sampleRate);

            audioEngine.positionFrame += chunkSamples;
            audioEngine.bootstrapPending = false;
            sampleOffset += chunkSamples;
            samplesRemaining -= chunkSamples;

            if (wrapAfterChunk)
            {
                audioEngine.positionFrame = loopStartFrame;
                audioEngine.bootstrapPending = true;
            }
        }
    }

    void openEditorWindow()
    {
        if (pluginInstance == nullptr)
            return;

        if (editorWindow != nullptr)
        {
            editorWindow->forceTopmostFront();
            return;
        }

        auto key = pluginDescription.fileOrIdentifier;
        if (key.isEmpty())
            key = pathEditor.getText().trim();

        key = normaliseVst3PathForSettings(key);

        editorWindow = std::make_unique<PluginEditorWindow>(*pluginInstance, "editor_" + key, appSettings);
        editorWindow->onWindowClosed = [this]
        {
            editorWindow.reset();
            updateButtons();
        };
        if (bridgeMode)
        {
            editorWindow->showBridgeEditorWindow();
        }
        else
        {
            editorWindow->forceTopmostFront();
            editorWindow->beginTopmostWarmup();
            juce::MessageManager::callAsync([this]
            {
                if (editorWindow != nullptr)
                {
                    editorWindow->forceTopmostFront();
                    editorWindow->beginTopmostWarmup();
                }
            });
        }
    }

    void openGraphEditorWindow(int slotIndex)
    {
        if (!juce::isPositiveAndBelow(slotIndex, static_cast<int>(persistentGraphSlots.size())))
            return;

        auto& slot = persistentGraphSlots[static_cast<size_t>(slotIndex)];
        if (slot.plugin == nullptr)
            return;

        if (graphEditorWindow != nullptr && graphEditorSlotIndex == slotIndex)
        {
            graphEditorWindow->forceTopmostFront();
            return;
        }

        closeGraphEditorWindow();

        auto key = normaliseVst3PathForSettings(slot.pluginPath);
        if (key.isEmpty())
            key = "graph_slot_" + juce::String(slotIndex + 1);

        graphEditorWindow = std::make_unique<PluginEditorWindow>(*slot.plugin, "graph_editor_" + key + "_" + juce::String(slotIndex + 1), appSettings);
        graphEditorSlotIndex = slotIndex;
        graphEditorWindow->onWindowClosed = [this]
        {
            graphEditorWindow.reset();
            graphEditorSlotIndex = -1;
            updateButtons();
        };
        if (bridgeMode)
        {
            graphEditorWindow->showBridgeEditorWindow();
        }
        else
        {
            graphEditorWindow->forceTopmostFront();
            graphEditorWindow->beginTopmostWarmup();
            juce::MessageManager::callAsync([this]
            {
                if (graphEditorWindow != nullptr)
                {
                    graphEditorWindow->forceTopmostFront();
                    graphEditorWindow->beginTopmostWarmup();
                }
            });
        }
    }

    void closeEditorWindow()
    {
        if (editorWindow != nullptr)
            editorWindow.reset();
    }

    void closeGraphEditorWindow()
    {
        if (graphEditorWindow != nullptr)
            graphEditorWindow.reset();
        graphEditorSlotIndex = -1;
    }

    juce::OwnedArray<juce::PluginDescription> describePlugin(const juce::File& pluginFile)
    {
        juce::OwnedArray<juce::PluginDescription> results;
        for (auto* format : formatManager.getFormats())
        {
            if (!format->fileMightContainThisPluginType(pluginFile.getFullPathName()))
                continue;
            format->findAllTypesForFile(results, pluginFile.getFullPathName());
            if (!results.isEmpty())
                break;
        }
        return results;
    }

    juce::PropertiesFile& appSettings;
    juce::AudioPluginFormatManager formatManager;
    juce::AudioFormatManager audioFormatManager;
    juce::AudioDeviceManager deviceManager;
    juce::MidiKeyboardState keyboardState;
    juce::CriticalSection pluginLock;
    juce::CriticalSection pendingMidiLock;
    juce::MidiBuffer pendingMidiMessages;
    juce::CriticalSection scheduledMidiLock;
    juce::Array<ScheduledMidiEvent> scheduledMidiEvents;
    juce::Array<ScheduledMidiEvent> scheduledMidiReady;   // scratch buffer, avoids per-callback alloc
    juce::Array<ScheduledMidiEvent> graphMidiReady;      // scratch buffer for graph MIDI dispatch
    mutable juce::CriticalSection streamAudioLock;
    BufferedOutputStream mainOutputStream;
    BufferedOutputStream previewOutputStream;
    BufferedOutputStream liveMidiOutputStream;
    std::atomic<bool> audioStreamEnabled { false };
    std::atomic<float> pluginOutputGain { 1.0f };
    std::atomic<float> pluginOutputPan { 0.0f };
    std::atomic<bool> graphTransportEnabled { false };
    std::atomic<int64_t> renderedSampleFrames { 0 };
    std::atomic<int64_t> scheduledMidiSequence { 0 };
    InProcessTransport inProcessTransport;
    AudioEngineState audioEngine;

    juce::Label pathLabel;
    juce::TextEditor pathEditor;
    juce::TextButton browseButton;
    juce::TextButton loadButton;
    juce::TextButton unloadButton;
    juce::TextButton editorButton;
    juce::Label statusLabel;
    juce::Label deviceLabel;
    juce::ComboBox deviceTypeBox;
    juce::ComboBox outputDeviceBox;
    juce::ComboBox sampleRateBox;
    juce::ComboBox bufferSizeBox;
    juce::MidiKeyboardComponent keyboardComponent;

    juce::PluginDescription pluginDescription;
    std::unique_ptr<juce::AudioPluginInstance> pluginInstance;
    std::unique_ptr<PluginEditorWindow> editorWindow;
    std::vector<PersistentGraphSlot> persistentGraphSlots;
    std::unique_ptr<PluginEditorWindow> graphEditorWindow;
    int graphEditorSlotIndex = -1;
    std::unique_ptr<juce::FileChooser> fileChooser;
    std::unique_ptr<HostCommandServer> commandServer;
    bool bridgeMode = false;
    bool audioInitialised = false;
    juce::File managedStateFile;

    double currentSampleRate = 44100.0;
    int currentBlockSize = 512;
    float pluginOutputPeakLevel = 0.0f;
    mutable std::atomic<int> pluginStateGeneration { 0 };
    mutable uint64_t lastPluginStateChecksum { 0 };

    // Pre-allocated scratch buffers for realtime audio callback (avoid heap allocs)
    std::vector<float> deinterleaveLeft;
    std::vector<float> deinterleaveRight;
    juce::AudioBuffer<float> graphProcessBuffer;
    juce::AudioBuffer<float> graphStereoBuffer;

    void ensureScratchBufferSize(int numSamples)
    {
        const auto needed = static_cast<size_t>(juce::jmax(64, numSamples));
        if (deinterleaveLeft.size() < needed)
        {
            deinterleaveLeft.resize(needed, 0.0f);
            deinterleaveRight.resize(needed, 0.0f);
        }
        if (graphStereoBuffer.getNumSamples() < numSamples)
            graphStereoBuffer.setSize(2, numSamples, false, false, true);
    }
};

HostCommandServer::HostCommandServer(HostComponent& ownerIn, int requestedPortIn)
    : juce::Thread("AI Music Studio VST Host Command Server"),
      owner(ownerIn),
      requestedPort(requestedPortIn)
{
}

HostCommandServer::~HostCommandServer()
{
    stopServer();
}

bool HostCommandServer::startServer()
{
    listener = std::make_unique<juce::StreamingSocket>();
    if (!listener->createListener(requestedPort, "127.0.0.1"))
        return false;

    boundPort = listener->getBoundPort();
    startThread();
    return true;
}

void HostCommandServer::stopServer()
{
    signalThreadShouldExit();
    if (listener != nullptr)
        listener->close();
    stopThread(2000);
    listener.reset();
}

int HostCommandServer::getBoundPort() const noexcept
{
    return boundPort.load();
}

void HostCommandServer::run()
{
    while (!threadShouldExit())
    {
        if (listener == nullptr)
            break;

        const auto ready = listener->waitUntilReady(true, 200);
        if (ready <= 0)
            continue;

        std::unique_ptr<juce::StreamingSocket> client(listener->waitForNextConnection());
        if (client == nullptr)
            continue;

        // Disable Nagle's algorithm on the accepted socket so that small
        // responses are sent immediately.  Without this, TCP batches small
        // writes, adding ~15 ms of delayed-ACK latency per round-trip on
        // Windows — far too slow for realtime audio IPC.
        {
            int flag = 1;
            // IPPROTO_TCP = 6, TCP_NODELAY = 1
            ::setsockopt(static_cast<int>(client->getRawSocketHandle()),
                         6, 1,
                         reinterpret_cast<const char*>(&flag), sizeof(flag));
        }

        handleClient(*client);
    }
}

void HostCommandServer::handleClient(juce::StreamingSocket& client)
{
    // Persistent connection: keep reading newline-delimited commands until
    // the client disconnects or an idle timeout (2 s) expires.  This avoids
    // the ~10-16 ms TCP connection overhead per command on Windows that made
    // realtime VST rendering choppy.
    juce::String incoming;
    char buffer[4096]{};

    for (;;)
    {
        // Try to extract a complete line from any data already buffered.
        while (incoming.containsChar('\n'))
        {
            const auto line = incoming.upToFirstOccurrenceOf("\n", false, false).trim();
            incoming = incoming.fromFirstOccurrenceOf("\n", false, false);

            const auto response = owner.handleRemoteCommandLine(
                line.isNotEmpty() ? line : "{\"command\":\"status\"}");
            const auto responseLine = response + "\n";
            if (client.write(responseLine.toRawUTF8(),
                             static_cast<int>(responseLine.getNumBytesAsUTF8())) < 0)
                return;
        }

        if (threadShouldExit())
            return;

        // Wait for more data.  Use a 2-second idle timeout so that legacy
        // one-shot clients (which close their write-half after sending) are
        // handled promptly, while persistent clients can keep the socket
        // open across many commands.
        const auto ready = client.waitUntilReady(true, 2000);
        if (ready < 0)
            return;

        if (ready == 0)
        {
            // Idle timeout — if there is leftover data without a newline,
            // treat it as a final command (legacy one-shot protocol).
            if (incoming.isNotEmpty())
            {
                const auto line = incoming.trim();
                incoming.clear();
                const auto response = owner.handleRemoteCommandLine(
                    line.isNotEmpty() ? line : "{\"command\":\"status\"}");
                const auto responseLine = response + "\n";
                client.write(responseLine.toRawUTF8(),
                             static_cast<int>(responseLine.getNumBytesAsUTF8()));
            }
            return;
        }

        const auto bytesRead = client.read(buffer, static_cast<int>(std::size(buffer) - 1), false);
        if (bytesRead <= 0)
        {
            // Client disconnected.  Process any remaining buffered data.
            if (incoming.isNotEmpty())
            {
                const auto line = incoming.trim();
                const auto response = owner.handleRemoteCommandLine(
                    line.isNotEmpty() ? line : "{\"command\":\"status\"}");
                const auto responseLine = response + "\n";
                client.write(responseLine.toRawUTF8(),
                             static_cast<int>(responseLine.getNumBytesAsUTF8()));
            }
            return;
        }

        buffer[bytesRead] = '\0';
        incoming << juce::String::fromUTF8(buffer, bytesRead);
    }
}

class HostWindow final : public juce::DocumentWindow
{
public:
    HostWindow(juce::PropertiesFile& settings,
               const juce::String& startupPluginPath,
               const juce::String& startupStatePath,
               bool restoreLastPluginOnStartup,
               bool shouldOpenEditorOnStartup,
               int requestedCommandPort,
               bool bridgeModeEnabled,
               bool startHidden,
               double startupSampleRate,
               int startupBufferSize,
               const juce::String& startupAudioDeviceType,
               const juce::String& startupAudioOutputDeviceName)
        : juce::DocumentWindow("AI Music Studio VST Host",
                               juce::Colour::fromRGB(20, 24, 31),
                               juce::DocumentWindow::allButtons),
          appSettings(settings),
          bridgeMode(bridgeModeEnabled),
          hiddenOnStartup(startHidden)
    {
        setUsingNativeTitleBar(true);
        setResizable(true, true);
        setResizeLimits(760, 520, 1800, 1200);
        setContentOwned(new HostComponent(settings,
                                          startupPluginPath,
                                          startupStatePath,
                                          restoreLastPluginOnStartup,
                                          shouldOpenEditorOnStartup,
                                          requestedCommandPort,
                                          bridgeMode,
                                          startupSampleRate,
                                          startupBufferSize,
                                          startupAudioDeviceType,
                                          startupAudioOutputDeviceName),
                        true);
        restoreBounds();
        if (!hiddenOnStartup)
            setVisible(true);
    }

    ~HostWindow() override
    {
        saveBounds();
    }

private:
    void closeButtonPressed() override
    {
        saveBounds();
        juce::JUCEApplication::getInstance()->systemRequestedQuit();
    }

    void moved() override
    {
        juce::DocumentWindow::moved();
        saveBounds();
    }

    void resized() override
    {
        juce::DocumentWindow::resized();
        saveBounds();
    }

    void restoreBounds()
    {
        const auto saved = appSettings.getValue(bridgeMode ? "host_window_bounds_bridge" : "host_window_bounds");
        if (saved.isNotEmpty())
        {
            const auto rect = juce::Rectangle<int>::fromString(saved);
            if (!rect.isEmpty())
            {
                setBounds(rect);
                return;
            }
        }

        if (bridgeMode)
            centreWithSize(kDefaultWindowWidth, 220);
        else
            centreWithSize(kDefaultWindowWidth, kDefaultWindowHeight);
    }

    void saveBounds() const
    {
        appSettings.setValue(bridgeMode ? "host_window_bounds_bridge" : "host_window_bounds", getBounds().toString());
    }

    juce::PropertiesFile& appSettings;
    bool bridgeMode = false;
    bool hiddenOnStartup = false;
};

#if defined(AIMS_VST_HOST_BUILD_LIBRARY)
namespace
{
   #if JUCE_WINDOWS
    #define AIMS_VST_HOST_API extern "C" __declspec(dllexport)
   #else
    #define AIMS_VST_HOST_API extern "C"
   #endif

    juce::PropertiesFile::Options hostSettingsOptions()
    {
        juce::PropertiesFile::Options options;
        options.applicationName = "AI Music Studio VST Host";
        options.filenameSuffix = "settings";
        options.osxLibrarySubFolder = "Application Support";
        return options;
    }

    void copyUtf8ToBuffer(const juce::String& text, char* destination, int destinationBytes)
    {
        if (destination == nullptr || destinationBytes <= 0)
            return;
        const auto utf8 = text.toRawUTF8();
        const auto sourceBytes = std::strlen(utf8);
        const auto copyBytes = static_cast<size_t>(juce::jmax(0, destinationBytes - 1));
        const auto bytes = juce::jmin(copyBytes, sourceBytes);
        std::memcpy(destination, utf8, bytes);
        destination[bytes] = '\0';
    }

    class LibraryHostInstance final
    {
    public:
        LibraryHostInstance(const juce::String& pluginPath,
                            bool restoreLastPluginOnStartup,
                            bool shouldOpenEditorOnStartup,
                            double startupSampleRate,
                            int startupBufferSize,
                            const juce::String& startupAudioDeviceType,
                            const juce::String& startupAudioOutputDeviceName)
            : guiInitializer(std::make_unique<juce::ScopedJuceInitialiser_GUI>())
        {
            appProperties = std::make_unique<juce::ApplicationProperties>();
            appProperties->setStorageParameters(hostSettingsOptions());
            component = std::make_unique<HostComponent>(*appProperties->getUserSettings(),
                                                        pluginPath,
                                                        juce::String(),
                                                        restoreLastPluginOnStartup,
                                                        shouldOpenEditorOnStartup,
                                                        0,
                                                        true,
                                                        startupSampleRate,
                                                        startupBufferSize,
                                                        startupAudioDeviceType,
                                                        startupAudioOutputDeviceName);
        }

        juce::String command(const juce::String& requestLine)
        {
            return component != nullptr ? component->handleRemoteCommandLine(requestLine)
                                        : juce::JSON::toString(makeResponse(false, "Host component unavailable"), true);
        }

    private:
        std::unique_ptr<juce::ScopedJuceInitialiser_GUI> guiInitializer;
        std::unique_ptr<juce::ApplicationProperties> appProperties;
        std::unique_ptr<HostComponent> component;
    };
}

AIMS_VST_HOST_API void* aims_vst_host_create(const char* pluginPath,
                                             int restoreLastPluginOnStartup,
                                             int openEditor,
                                             double sampleRate,
                                             int bufferSize,
                                             const char* audioDeviceType,
                                             const char* audioOutputDeviceName,
                                             char* errorBuffer,
                                             int errorBufferBytes)
{
    try
    {
        auto instance = std::make_unique<LibraryHostInstance>(juce::String::fromUTF8(pluginPath != nullptr ? pluginPath : ""),
                                                              restoreLastPluginOnStartup != 0,
                                                              openEditor != 0,
                                                              sampleRate,
                                                              bufferSize,
                                                              juce::String::fromUTF8(audioDeviceType != nullptr ? audioDeviceType : ""),
                                                              juce::String::fromUTF8(audioOutputDeviceName != nullptr ? audioOutputDeviceName : ""));
        copyUtf8ToBuffer("", errorBuffer, errorBufferBytes);
        return instance.release();
    }
    catch (const std::exception& exc)
    {
        copyUtf8ToBuffer(juce::String(exc.what()), errorBuffer, errorBufferBytes);
    }
    catch (...)
    {
        copyUtf8ToBuffer("Unknown error creating in-process VST host", errorBuffer, errorBufferBytes);
    }
    return nullptr;
}

AIMS_VST_HOST_API int aims_vst_host_command(void* handle,
                                            const char* requestLine,
                                            char* responseBuffer,
                                            int responseBufferBytes)
{
    auto* instance = static_cast<LibraryHostInstance*>(handle);
    if (instance == nullptr)
    {
        copyUtf8ToBuffer(juce::JSON::toString(makeResponse(false, "Invalid host handle"), true), responseBuffer, responseBufferBytes);
        return 0;
    }
    const auto response = instance->command(juce::String::fromUTF8(requestLine != nullptr ? requestLine : ""));
    copyUtf8ToBuffer(response, responseBuffer, responseBufferBytes);
    return 1;
}

AIMS_VST_HOST_API void aims_vst_host_destroy(void* handle)
{
    auto* instance = static_cast<LibraryHostInstance*>(handle);
    if (instance == nullptr)
        return;

    if (auto* messageManager = juce::MessageManager::getInstanceWithoutCreating())
    {
        if (!messageManager->isThisTheMessageThread())
        {
            juce::MessageManager::callSync([instance]
            {
                delete instance;
            });
            return;
        }
    }

    delete instance;
}
#else
class HostApplication final : public juce::JUCEApplication
{
public:
    const juce::String getApplicationName() override      { return "AI Music Studio VST Host"; }
    const juce::String getApplicationVersion() override   { return "0.1.0"; }
    bool moreThanOneInstanceAllowed() override            { return true; }

    void initialise(const juce::String&) override
    {
#if JUCE_WINDOWS
        suppressWindowsCrashDialogs();
#endif
        juce::PropertiesFile::Options options;
        options.applicationName = "AI Music Studio VST Host";
        options.filenameSuffix = "settings";
        options.osxLibrarySubFolder = "Application Support";

        appProperties = std::make_unique<juce::ApplicationProperties>();
        appProperties->setStorageParameters(options);
        mainWindow = std::make_unique<HostWindow>(*appProperties->getUserSettings(),
                                                  parseStartupPluginPath(),
                                                  parseStartupStatePath(),
                                                  shouldRestoreLastPluginOnStartup(),
                                                  shouldOpenEditorOnStartup(),
                                                  parseCommandPort(),
                                                  isBridgeModeEnabled(),
                                                  shouldStartHidden(),
                                                  parseStartupSampleRate(),
                                                  parseStartupBufferSize(),
                                                  parseStartupAudioDeviceType(),
                                                  parseStartupAudioOutputDeviceName());
    }

    void shutdown() override
    {
        mainWindow.reset();
        appProperties.reset();
    }

    void systemRequestedQuit() override
    {
        quit();
    }

private:
    juce::String parseStartupPluginPath() const
    {
        const auto args = getCommandLineParameterArray();
        for (int i = 0; i < args.size(); ++i)
        {
            if (args[i] == "--plugin" && i + 1 < args.size())
                return args[i + 1].unquoted();
        }
        return {};
    }

    bool shouldOpenEditorOnStartup() const
    {
        const auto args = getCommandLineParameterArray();
        return args.contains("--open-editor");
    }

    juce::String parseStartupStatePath() const
    {
        const auto args = getCommandLineParameterArray();
        for (int i = 0; i < args.size(); ++i)
        {
            if (args[i] == "--state-file" && i + 1 < args.size())
                return args[i + 1].unquoted();
        }
        return {};
    }

    int parseCommandPort() const
    {
        const auto args = getCommandLineParameterArray();
        for (int i = 0; i < args.size(); ++i)
        {
            if (args[i] == "--port" && i + 1 < args.size())
                return juce::jmax(0, args[i + 1].getIntValue());
        }

        return 0;
    }

    double parseStartupSampleRate() const
    {
        const auto args = getCommandLineParameterArray();
        for (int i = 0; i < args.size(); ++i)
        {
            if (args[i] == "--sample-rate" && i + 1 < args.size())
                return juce::jmax(0.0, args[i + 1].getDoubleValue());
        }

        return 0.0;
    }

    int parseStartupBufferSize() const
    {
        const auto args = getCommandLineParameterArray();
        for (int i = 0; i < args.size(); ++i)
        {
            if (args[i] == "--buffer-size" && i + 1 < args.size())
                return juce::jmax(0, args[i + 1].getIntValue());
        }

        return 0;
    }

    juce::String parseStartupAudioDeviceType() const
    {
        const auto args = getCommandLineParameterArray();
        for (int i = 0; i < args.size(); ++i)
        {
            if (args[i] == "--audio-device-type" && i + 1 < args.size())
                return args[i + 1].unquoted();
        }

        return {};
    }

    juce::String parseStartupAudioOutputDeviceName() const
    {
        const auto args = getCommandLineParameterArray();
        for (int i = 0; i < args.size(); ++i)
        {
            if (args[i] == "--audio-output-device" && i + 1 < args.size())
                return args[i + 1].unquoted();
        }

        return {};
    }

    bool shouldRestoreLastPluginOnStartup() const
    {
        return ! getCommandLineParameterArray().contains("--no-restore-last-plugin");
    }

    bool isBridgeModeEnabled() const
    {
        return getCommandLineParameterArray().contains("--bridge-mode");
    }

    bool shouldStartHidden() const
    {
        return getCommandLineParameterArray().contains("--hidden");
    }

    std::unique_ptr<juce::ApplicationProperties> appProperties;
    std::unique_ptr<HostWindow> mainWindow;
};

START_JUCE_APPLICATION(HostApplication)
#endif
