#include <juce_audio_devices/juce_audio_devices.h>
#include <juce_audio_processors/juce_audio_processors.h>
#include <juce_audio_utils/juce_audio_utils.h>
#include <juce_gui_extra/juce_gui_extra.h>
#if JUCE_WINDOWS
#include <windows.h>
#endif
#include <algorithm>
#include <cstring>
#include <mutex>
#include <utility>

namespace
{
constexpr int kDefaultWindowWidth = 1040;
constexpr int kDefaultWindowHeight = 720;
constexpr int kDefaultMidiChannel = 1;
constexpr int kDefaultCommandPort = 47653;

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
        setAlwaysOnTop(false);
        setMinimised(false);
        setVisible(true);
        toFront(true);
        grabKeyboardFocus();
    }

    void showBridgeEditorWindow()
    {
        setAlwaysOnTop(false);
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
                            private juce::AudioIODeviceCallback
{
public:
    HostComponent(juce::PropertiesFile& settings,
                  const juce::String& startupPluginPath,
                  const juce::String& startupStatePath,
                  bool shouldOpenEditorOnStartup,
                  int requestedCommandPort,
                  bool bridgeModeEnabled,
                  double startupSampleRate,
                  int startupBufferSize)
        : appSettings(settings),
          bridgeMode(bridgeModeEnabled),
          managedStateFile(startupStatePath.isNotEmpty() ? juce::File(startupStatePath) : juce::File()),
          keyboardComponent(keyboardState, juce::MidiKeyboardComponent::horizontalKeyboard)
    {
        formatManager.addFormat(std::make_unique<juce::VST3PluginFormat>());

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

        initialiseAudio();
        restoreAudioPreferences(startupSampleRate, startupBufferSize);
        updateDeviceBoxes();
        updateDeviceLabel();
        updateButtons();

        const auto startupPath = startupPluginPath.isNotEmpty() ? startupPluginPath
                                                                : appSettings.getValue("last_plugin_path");
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
        deviceManager.removeAudioCallback(this);
        deviceManager.closeAudioDevice();
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
        deviceLabel.setBounds(deviceRow.removeFromLeft(300));
        deviceRow.removeFromLeft(8);
        sampleRateBox.setBounds(deviceRow.removeFromLeft(140));
        deviceRow.removeFromLeft(8);
        bufferSizeBox.setBounds(deviceRow.removeFromLeft(140));
        deviceRow.removeFromLeft(8);
        editorButton.setBounds(deviceRow.removeFromLeft(120));
        deviceRow.removeFromLeft(8);
        unloadButton.setBounds(deviceRow.removeFromLeft(90));

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
            return juce::JSON::toString(makeResponse(false, "Invalid JSON command"));

        const auto result = juce::MessageManager::callSync([this, request]() mutable
        {
            return handleRemoteCommand(request);
        });

        if (!result.has_value())
            return juce::JSON::toString(makeResponse(false, "Failed to dispatch command to message thread"));

        return juce::JSON::toString(*result);
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

            if (clearChannelsVar.isArray())
            {
                if (auto* clearArray = clearChannelsVar.getArray())
                {
                    for (const auto& channelVar : *clearArray)
                    {
                        const auto channel = clampMidiChannel(static_cast<int>(channelVar));
                        clearScheduledMidiEvents(channel, loopEpoch);
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
                        clearScheduledMidiEvents(channel, loopEpoch);
                        enqueuePanicMessages(channel);
                    }
                }
            }

            const auto currentFrame = renderedSampleFrames.load();
            int scheduledCount = 0;
            if (auto* array = eventsVar.getArray())
            {
                juce::ScopedLock lock(scheduledMidiLock);
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
                    else
                        continue;

                    const auto targetFrame = currentFrame + baseOffsetFrames + sampleOffset;
                    const auto isDuplicate = std::any_of(
                        scheduledMidiEvents.begin(),
                        scheduledMidiEvents.end(),
                        [&](const ScheduledMidiEvent& existing)
                        {
                            if (existing.loopEpoch != loopEpoch || existing.frame != targetFrame)
                                return false;
                            if (existing.message.getChannel() != message.getChannel())
                                return false;
                            if (existing.message.isNoteOn() && message.isNoteOn())
                                return existing.message.getNoteNumber() == message.getNoteNumber();
                            if (existing.message.isNoteOff() && message.isNoteOff())
                                return existing.message.getNoteNumber() == message.getNoteNumber();
                            return false;
                        }
                    );
                    if (isDuplicate)
                        continue;

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
            auto response = renderOfflineAudioBlock(frames);
            appendStatusFields(response);
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
        setResponseField(response, "plugin_loaded", pluginInstance != nullptr);
        setResponseField(response, "plugin_name", pluginDescription.name);
        setResponseField(response, "plugin_path",
                         normaliseVst3PathForSettings(pathEditor.getText().trim()));
        setResponseField(response, "editor_open", editorWindow != nullptr);
        setResponseField(response, "sample_rate", currentSampleRate);
        setResponseField(response, "buffer_size", currentBlockSize);
        setResponseField(response, "command_port", getActiveCommandPort());
    }

    void configureButton(juce::TextButton& button, const juce::String& text)
    {
        addAndMakeVisible(button);
        button.setButtonText(text);
        button.addListener(this);
    }

    void initialiseAudio()
    {
        deviceManager.initialise(0, 2, nullptr, true, {}, nullptr);
        deviceManager.addAudioCallback(this);
    }

    void restoreAudioPreferences(double startupSampleRate, int startupBufferSize)
    {
        const auto wantedRate = startupSampleRate > 0.0
            ? startupSampleRate
            : appSettings.getDoubleValue("audio_sample_rate", 0.0);
        const auto wantedBuffer = startupBufferSize > 0
            ? startupBufferSize
            : appSettings.getIntValue("audio_buffer_size", 0);
        if (auto* device = deviceManager.getCurrentAudioDevice())
        {
            auto setup = deviceManager.getAudioDeviceSetup();
            bool changed = false;

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

        if (startupSampleRate > 0.0)
            appSettings.setValue("audio_sample_rate", startupSampleRate);
        if (startupBufferSize > 0)
            appSettings.setValue("audio_buffer_size", startupBufferSize);
    }

    void updateDeviceBoxes()
    {
        sampleRateBox.clear(juce::dontSendNotification);
        bufferSizeBox.clear(juce::dontSendNotification);

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
        if (auto* device = deviceManager.getCurrentAudioDevice())
        {
            deviceLabel.setText(
                "Audio: " + device->getName()
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
        auto setup = deviceManager.getAudioDeviceSetup();
        bool changed = false;

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
        keyboardState.reset();
        preparePluginForPlayback();
    }

    void audioDeviceStopped() override
    {
        releasePluginResources();
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

        auto* plugin = pluginInstance.get();
        if (plugin == nullptr)
            return;

        juce::ScopedLock lock(pluginLock);
        juce::MidiBuffer midi;
        keyboardState.processNextMidiBuffer(midi, 0, numSamples, true);
        appendPendingMidiMessages(midi);
        appendScheduledMidiMessages(midi, numSamples);
        plugin->processBlock(buffer, midi);
        renderedSampleFrames.fetch_add(numSamples);
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

    void appendScheduledMidiMessages(juce::MidiBuffer& destination, int numSamples)
    {
        const auto blockStart = renderedSampleFrames.load();
        const auto blockEnd = blockStart + juce::jmax(1, numSamples);
        juce::Array<ScheduledMidiEvent> keep;
        juce::Array<ScheduledMidiEvent> ready;

        juce::ScopedLock lock(scheduledMidiLock);
        for (const auto& event : scheduledMidiEvents)
        {
            if (event.frame < blockStart)
            {
                if (event.message.isNoteOff())
                {
                    auto lateEvent = event;
                    lateEvent.frame = blockStart;
                    ready.add(lateEvent);
                }
            }
            else if (event.frame < blockEnd)
            {
                ready.add(event);
            }
            else
            {
                keep.add(event);
            }
        }
        scheduledMidiEvents.swapWith(keep);

        std::sort(ready.begin(), ready.end(), [](const ScheduledMidiEvent& a, const ScheduledMidiEvent& b)
        {
            if (a.frame != b.frame)
                return a.frame < b.frame;
            if (a.priority != b.priority)
                return a.priority < b.priority;
            return a.sequence < b.sequence;
        });

        for (const auto& event : ready)
        {
            const auto samplePosition = static_cast<int>(
                juce::jlimit<int64_t>(0, juce::jmax(0, numSamples - 1), event.frame - blockStart)
            );
            destination.addEvent(event.message, samplePosition);
        }
    }

    void clearScheduledMidiEvents(int channel = 0, int64_t beforeLoopEpoch = -1)
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
            if (event.message.getChannel() != targetChannel || (beforeLoopEpoch >= 0 && event.loopEpoch >= beforeLoopEpoch))
                keep.add(event);
        }
        scheduledMidiEvents.swapWith(keep);
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
        pluginInstance->enableAllBuses();
        pluginInstance->prepareToPlay(currentSampleRate, currentBlockSize);
    }

    void releasePluginResources()
    {
        juce::ScopedLock lock(pluginLock);
        if (pluginInstance != nullptr)
            pluginInstance->releaseResources();
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

    void persistManagedState()
    {
        if (managedStateFile == juce::File())
            return;
        savePluginStateToFile(managedStateFile);
    }

    juce::var renderOfflineAudioBlock(int numSamples)
    {
        auto response = makeResponse(true, "Rendered audio");
        juce::AudioBuffer<float> buffer(juce::jmax(2, pluginInstance != nullptr ? pluginInstance->getTotalNumOutputChannels() : 2),
                                        juce::jmax(1, numSamples));
        buffer.clear();

        {
            juce::ScopedLock lock(pluginLock);
            if (pluginInstance == nullptr)
                return makeResponse(false, "No plugin loaded");

            juce::MidiBuffer midi;
            keyboardState.processNextMidiBuffer(midi, 0, numSamples, true);
            appendPendingMidiMessages(midi);
            appendScheduledMidiMessages(midi, numSamples);
            pluginInstance->processBlock(buffer, midi);
            renderedSampleFrames.fetch_add(numSamples);
        }

        juce::MemoryOutputStream stream;
        const auto channelCount = buffer.getNumChannels();
        for (int sample = 0; sample < numSamples; ++sample)
        {
            const auto left = buffer.getSample(0, sample);
            const auto right = channelCount > 1 ? buffer.getSample(1, sample) : left;
            stream.write(&left, sizeof(left));
            stream.write(&right, sizeof(right));
        }

        setResponseField(response, "audio_b64", juce::Base64::toBase64(stream.getData(), stream.getDataSize()));
        setResponseField(response, "channels", 2);
        setResponseField(response, "format", "f32le-interleaved");
        return response;
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

    void closeEditorWindow()
    {
        if (editorWindow != nullptr)
            editorWindow.reset();
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
    juce::AudioDeviceManager deviceManager;
    juce::MidiKeyboardState keyboardState;
    juce::CriticalSection pluginLock;
    juce::CriticalSection pendingMidiLock;
    juce::MidiBuffer pendingMidiMessages;
    juce::CriticalSection scheduledMidiLock;
    juce::Array<ScheduledMidiEvent> scheduledMidiEvents;
    std::atomic<int64_t> renderedSampleFrames { 0 };
    std::atomic<int64_t> scheduledMidiSequence { 0 };

    juce::Label pathLabel;
    juce::TextEditor pathEditor;
    juce::TextButton browseButton;
    juce::TextButton loadButton;
    juce::TextButton unloadButton;
    juce::TextButton editorButton;
    juce::Label statusLabel;
    juce::Label deviceLabel;
    juce::ComboBox sampleRateBox;
    juce::ComboBox bufferSizeBox;
    juce::MidiKeyboardComponent keyboardComponent;

    juce::PluginDescription pluginDescription;
    std::unique_ptr<juce::AudioPluginInstance> pluginInstance;
    std::unique_ptr<PluginEditorWindow> editorWindow;
    std::unique_ptr<juce::FileChooser> fileChooser;
    std::unique_ptr<HostCommandServer> commandServer;
    bool bridgeMode = false;
    juce::File managedStateFile;

    double currentSampleRate = 44100.0;
    int currentBlockSize = 512;
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

        handleClient(*client);
    }
}

void HostCommandServer::handleClient(juce::StreamingSocket& client)
{
    juce::String incoming;
    char buffer[2048]{};

    while (!threadShouldExit())
    {
        const auto ready = client.waitUntilReady(true, 500);
        if (ready < 0)
            return;

        if (ready == 0)
        {
            if (incoming.isNotEmpty())
                break;
            continue;
        }

        const auto bytesRead = client.read(buffer, static_cast<int>(std::size(buffer) - 1), false);
        if (bytesRead <= 0)
            break;

        buffer[bytesRead] = '\0';
        incoming << juce::String::fromUTF8(buffer, bytesRead);

        if (incoming.containsChar('\n'))
            break;
    }

    const auto line = incoming.upToFirstOccurrenceOf("\n", false, false).trim();
    const auto response = owner.handleRemoteCommandLine(line.isNotEmpty() ? line
                                                                          : "{\"command\":\"status\"}");
    const auto responseLine = response + "\n";
    client.write(responseLine.toRawUTF8(), static_cast<int>(responseLine.getNumBytesAsUTF8()));
    client.close();
}

class HostWindow final : public juce::DocumentWindow
{
public:
    HostWindow(juce::PropertiesFile& settings,
               const juce::String& startupPluginPath,
               const juce::String& startupStatePath,
               bool shouldOpenEditorOnStartup,
               int requestedCommandPort,
               bool bridgeModeEnabled,
               bool startHidden,
               double startupSampleRate,
               int startupBufferSize)
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
                                          shouldOpenEditorOnStartup,
                                          requestedCommandPort,
                                          bridgeMode,
                                          startupSampleRate,
                                          startupBufferSize),
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
                            bool shouldOpenEditorOnStartup,
                            double startupSampleRate,
                            int startupBufferSize)
            : guiInitializer(std::make_unique<juce::ScopedJuceInitialiser_GUI>())
        {
            appProperties = std::make_unique<juce::ApplicationProperties>();
            appProperties->setStorageParameters(hostSettingsOptions());
            component = std::make_unique<HostComponent>(*appProperties->getUserSettings(),
                                                        pluginPath,
                                                        juce::String(),
                                                        shouldOpenEditorOnStartup,
                                                        0,
                                                        true,
                                                        startupSampleRate,
                                                        startupBufferSize);
        }

        juce::String command(const juce::String& requestLine)
        {
            return component != nullptr ? component->handleRemoteCommandLine(requestLine)
                                        : juce::JSON::toString(makeResponse(false, "Host component unavailable"));
        }

    private:
        std::unique_ptr<juce::ScopedJuceInitialiser_GUI> guiInitializer;
        std::unique_ptr<juce::ApplicationProperties> appProperties;
        std::unique_ptr<HostComponent> component;
    };
}

AIMS_VST_HOST_API void* aims_vst_host_create(const char* pluginPath,
                                             int openEditor,
                                             double sampleRate,
                                             int bufferSize,
                                             char* errorBuffer,
                                             int errorBufferBytes)
{
    try
    {
        auto instance = std::make_unique<LibraryHostInstance>(juce::String::fromUTF8(pluginPath != nullptr ? pluginPath : ""),
                                                              openEditor != 0,
                                                              sampleRate,
                                                              bufferSize);
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
        copyUtf8ToBuffer(juce::JSON::toString(makeResponse(false, "Invalid host handle")), responseBuffer, responseBufferBytes);
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
        juce::PropertiesFile::Options options;
        options.applicationName = "AI Music Studio VST Host";
        options.filenameSuffix = "settings";
        options.osxLibrarySubFolder = "Application Support";

        appProperties = std::make_unique<juce::ApplicationProperties>();
        appProperties->setStorageParameters(options);
        mainWindow = std::make_unique<HostWindow>(*appProperties->getUserSettings(),
                                                  parseStartupPluginPath(),
                                                  parseStartupStatePath(),
                                                  shouldOpenEditorOnStartup(),
                                                  parseCommandPort(),
                                                  isBridgeModeEnabled(),
                                                  shouldStartHidden(),
                                                  parseStartupSampleRate(),
                                                  parseStartupBufferSize());
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
