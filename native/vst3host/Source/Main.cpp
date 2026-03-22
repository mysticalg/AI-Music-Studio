#include <JuceHeader.h>

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

class PluginEditorWindow final : public juce::DocumentWindow
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
        saveBounds();
    }

    std::function<void()> onWindowClosed;

private:
    void closeButtonPressed() override
    {
        saveBounds();
        if (onWindowClosed != nullptr)
            onWindowClosed();
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
};

class HostComponent final : public juce::Component,
                            private juce::Button::Listener,
                            private juce::ComboBox::Listener,
                            private juce::AudioIODeviceCallback
{
public:
    HostComponent(juce::PropertiesFile& settings,
                  const juce::String& startupPluginPath,
                  bool shouldOpenEditorOnStartup,
                  int requestedCommandPort,
                  bool bridgeModeEnabled)
        : appSettings(settings),
          bridgeMode(bridgeModeEnabled),
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
        restoreAudioPreferences();
        updateDeviceBoxes();
        updateDeviceLabel();
        updateButtons();

        const auto startupPath = startupPluginPath.isNotEmpty() ? startupPluginPath
                                                                : appSettings.getValue("last_plugin_path");
        if (startupPath.isNotEmpty())
        {
            loadPlugin(startupPath);
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
        commandServer.reset();
        closeEditorWindow();
        unloadPlugin();
        deviceManager.removeAudioCallback(this);
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

        if (command == "open_editor")
        {
            if (pluginInstance == nullptr)
                return makeResponse(false, "No plugin loaded");

            openEditorWindow();
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

            keyboardState.noteOn(channel, note, velocity);
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

            keyboardState.noteOff(channel, note, velocity);
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
            keyboardState.allNotesOff(channel);
            auto response = makeResponse(true, "All notes off");
            appendStatusFields(response);
            setResponseField(response, "channel", channel);
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

    void restoreAudioPreferences()
    {
        const auto wantedRate = appSettings.getDoubleValue("audio_sample_rate", 0.0);
        const auto wantedBuffer = appSettings.getIntValue("audio_buffer_size", 0);
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
        plugin->processBlock(buffer, midi);
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

    void openEditorWindow()
    {
        if (pluginInstance == nullptr)
            return;

        if (editorWindow != nullptr)
        {
            editorWindow->toFront(true);
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
        editorWindow->setAlwaysOnTop(true);
        editorWindow->setVisible(true);
        editorWindow->toFront(true);
        editorWindow->grabKeyboardFocus();
        juce::MessageManager::callAsync([this]
        {
            if (editorWindow != nullptr)
                editorWindow->toFront(true);
        });
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
               bool shouldOpenEditorOnStartup,
               int requestedCommandPort,
               bool bridgeModeEnabled)
        : juce::DocumentWindow("AI Music Studio VST Host",
                               juce::Colour::fromRGB(20, 24, 31),
                               juce::DocumentWindow::allButtons),
          appSettings(settings),
          bridgeMode(bridgeModeEnabled)
    {
        setUsingNativeTitleBar(true);
        setResizable(true, true);
        setResizeLimits(760, 520, 1800, 1200);
        setContentOwned(new HostComponent(settings,
                                          startupPluginPath,
                                          shouldOpenEditorOnStartup,
                                          requestedCommandPort,
                                          bridgeMode),
                        true);
        restoreBounds();
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
};

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
                                                  shouldOpenEditorOnStartup(),
                                                  parseCommandPort(),
                                                  isBridgeModeEnabled());
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

    bool isBridgeModeEnabled() const
    {
        return getCommandLineParameterArray().contains("--bridge-mode");
    }

    std::unique_ptr<juce::ApplicationProperties> appProperties;
    std::unique_ptr<HostWindow> mainWindow;
};

START_JUCE_APPLICATION(HostApplication)
