#include "MainWindow.h"
#include "AudioExport.h"
#include "UiStyle.h"
#include <BinaryData.h>
#include <juce_audio_processors/juce_audio_processors.h>

#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace aims
{
namespace
{
juce::PropertiesFile::Options nativeWindowSettingsOptions()
{
    juce::PropertiesFile::Options options;
    options.applicationName = "Mutagen";
    options.filenameSuffix = "settings";
    options.osxLibrarySubFolder = "Application Support";
    return options;
}

juce::File nativeLogsDirectory()
{
    return juce::File::getSpecialLocation(juce::File::userApplicationDataDirectory)
        .getChildFile("Mutagen")
        .getChildFile("logs");
}

juce::File nativeAceStepServerLogFile()
{
    return nativeLogsDirectory().getChildFile("ace-step-server.log");
}

juce::File nativeSessionStateFile()
{
    return juce::File::getSpecialLocation(juce::File::userApplicationDataDirectory)
        .getChildFile("Mutagen")
        .getChildFile("last-session.aims");
}

juce::File nativeTemplatesDirectory()
{
    return juce::File::getSpecialLocation(juce::File::userApplicationDataDirectory)
        .getChildFile("Mutagen")
        .getChildFile("templates");
}

juce::File nativeStemSeparationDirectory()
{
    return juce::File::getSpecialLocation(juce::File::userApplicationDataDirectory)
        .getChildFile("Mutagen")
        .getChildFile("stem-separation");
}

juce::File nativeGeneratedAudioDirectory()
{
    return juce::File::getSpecialLocation(juce::File::userApplicationDataDirectory)
        .getChildFile("Mutagen")
        .getChildFile("generated-audio");
}

constexpr const char* kMidiInputSelectionDisabled = "__mutagen_midi_input_disabled__";
constexpr const char* kAceStepGitHubUrl = "https://github.com/ace-step/ACE-Step-1.5";
constexpr const char* kOllamaSiteUrl = "https://ollama.com/";
constexpr const char* kOllamaWindowsDownloadUrl = "https://ollama.com/download/windows";
constexpr const char* kDemucsInstallUrl = "https://github.com/facebookresearch/demucs";
constexpr const char* kBuiltInDefaultTemplateId = "__mutagen_builtin_default_template__";
constexpr const char* kProjectTemplateWildcard = "*.aimstpl";

bool aceStepBootstrapMessageMatches(const juce::String& text)
{
    const auto normalised = text.toLowerCase();
    return normalised.contains("first-run model setup in progress")
        || normalised.contains("still initializing models")
        || normalised.contains("still preparing first-run models")
        || normalised.contains("downloading or loading required models")
        || normalised.contains("downloading/loading models for first use")
        || normalised.contains("ace-step is listening on");
}

juce::String aceStepBootstrapUserMessage()
{
    return "ACE-Step is downloading or loading required models for first use.\n\n"
           "Leave ACE-Step and Mutagen running while setup finishes, then try Generate Audio again.\n\n"
           "Open Activity Log if you want to watch progress.";
}

juce::String aceStepBootstrapStatusText()
{
    return "ACE-Step is downloading/loading models for first use. Leave it running.";
}

juce::String sanitiseAceStepServerOutput(const juce::String& rawText)
{
    juce::String cleaned;
    cleaned.preallocateBytes(rawText.getNumBytesAsUTF8());

    bool insideEscapeSequence = false;
    auto characters = rawText.getCharPointer();
    while (!characters.isEmpty())
    {
        const auto character = characters.getAndAdvance();

        if (insideEscapeSequence)
        {
            if ((character >= '@' && character <= '~') || character == 'm' || character == 'K')
                insideEscapeSequence = false;
            continue;
        }

        if (character == 0x1b)
        {
            insideEscapeSequence = true;
            continue;
        }

        if (character == '\r')
        {
            cleaned << '\n';
            continue;
        }

        if (character != 0)
            cleaned << character;
    }

    return cleaned;
}

bool isAceStepProgressLine(const juce::String& text)
{
    const auto trimmed = text.trim();
    if (trimmed.isEmpty())
        return false;

    return trimmed.containsChar('%')
        || trimmed.containsIgnoreCase("model.safetensors:")
        || trimmed.containsIgnoreCase("diffusion_pytorch_model.safetensors:")
        || trimmed.containsIgnoreCase("tokenizer.json:")
        || trimmed.containsIgnoreCase("Downloading unified repo")
        || trimmed.containsIgnoreCase("Downloading ")
        || trimmed.containsIgnoreCase("already exists at ");
}

bool isAceStepInformationalLine(const juce::String& text)
{
    const auto trimmed = text.trim();
    if (trimmed.isEmpty())
        return false;

    if (trimmed.startsWith("INFO:") && trimmed.contains("HTTP/1.1"))
        return false;

    return trimmed.containsIgnoreCase("[API Server]")
        || trimmed.containsIgnoreCase("[Model Download]")
        || trimmed.containsIgnoreCase("Downloading Model from ")
        || trimmed.containsIgnoreCase("Started server process")
        || trimmed.containsIgnoreCase("Application startup complete")
        || trimmed.containsIgnoreCase("Uvicorn running on ")
        || trimmed.containsIgnoreCase("WARNING")
        || trimmed.containsIgnoreCase("ERROR");
}

enum class MidiImportAssignmentMode
{
    generalMidi,
    tryNativeRack
};

constexpr int kMidiImportModeGeneralMidiId = 1;
constexpr int kMidiImportModeNativeRackId = 2;

MidiImportAssignmentMode midiImportAssignmentModeFromComboId(int comboId)
{
    return comboId == kMidiImportModeNativeRackId
        ? MidiImportAssignmentMode::tryNativeRack
        : MidiImportAssignmentMode::generalMidi;
}

struct ProjectTemplateOption
{
    juce::String identifier;
    juce::String name;
    juce::File file;
    bool builtIn = false;
};

juce::String rackEntryDisplayName(const VstInstrument& entry)
{
    if (entry.name.trim().isNotEmpty())
        return entry.name.trim();
    if (entry.pluginName.trim().isNotEmpty())
        return entry.pluginName.trim();
    if (entry.path.trim().isNotEmpty())
        return juce::File(entry.path).getFileNameWithoutExtension();
    return {};
}

const VstInstrument* findRackEntryByReference(const ProjectState& project,
                                              const juce::String& reference,
                                              bool requireEffect = false)
{
    const auto trimmedReference = reference.trim();
    if (trimmedReference.isEmpty())
        return nullptr;

    const auto referenceFile = juce::File(trimmedReference);
    const auto referencePath = referenceFile.getFullPathName();
    const auto referenceFileName = referenceFile.getFileName();
    const auto referenceStem = referenceFile.getFileNameWithoutExtension();

    for (const auto& entry : project.vstRack)
    {
        if (requireEffect && !entry.isEffect)
            continue;

        const auto entryPath = entry.path.trim();
        const auto entryName = entry.name.trim();
        const auto entryPluginName = entry.pluginName.trim();

        if (trimmedReference.equalsIgnoreCase(entryPath)
            || trimmedReference.equalsIgnoreCase(entryName)
            || trimmedReference.equalsIgnoreCase(entryPluginName))
        {
            return &entry;
        }

        if (entryPath.isEmpty())
            continue;

        const auto entryFile = juce::File(entryPath);
        if (referencePath.equalsIgnoreCase(entryFile.getFullPathName())
            || trimmedReference.equalsIgnoreCase(entryFile.getFileName())
            || trimmedReference.equalsIgnoreCase(entryFile.getFileNameWithoutExtension())
            || referenceFileName.equalsIgnoreCase(entryFile.getFileName())
            || referenceStem.equalsIgnoreCase(entryFile.getFileNameWithoutExtension()))
        {
            return &entry;
        }
    }

    return nullptr;
}

juce::File ensureTemplateSuffix(const juce::File& file)
{
    return file.hasFileExtension(".aimstpl") ? file : file.withFileExtension(".aimstpl");
}

std::vector<ProjectTemplateOption> availableProjectTemplateOptions()
{
    std::vector<ProjectTemplateOption> options;
    options.push_back({ kBuiltInDefaultTemplateId, "Default Template", {}, true });

    const auto templateRoot = nativeTemplatesDirectory();
    if (!templateRoot.isDirectory())
        return options;

    const auto files = templateRoot.findChildFiles(juce::File::findFiles, false, "*.aimstpl");
    for (const auto& file : files)
    {
        auto name = file.getFileNameWithoutExtension().trim();
        if (name.isEmpty())
            name = "Template";

        options.push_back({ file.getFullPathName(), name, file, false });
    }

    std::sort(options.begin() + 1,
              options.end(),
              [] (const ProjectTemplateOption& lhs, const ProjectTemplateOption& rhs)
              {
                  return lhs.name.compareNatural(rhs.name) < 0;
              });
    return options;
}

juce::String suggestTemplateBaseName(const juce::File& currentProjectFile, const ProjectFileData& documentState)
{
    auto name = currentProjectFile.existsAsFile()
        ? currentProjectFile.getFileNameWithoutExtension().trim()
        : juce::String();

    if (name.isEmpty() && !documentState.project.tracks.empty())
        name = documentState.project.tracks.front().name.trim();
    if (name.isEmpty())
        name = "Mutagen Template";

    name = juce::File::createLegalFileName(name);
    return name.isNotEmpty() ? name : juce::String("Mutagen Template");
}

bool tryDetectDemucsCommand(juce::StringArray& outCommandPrefix)
{
    const std::array<juce::String, 2> launchers = { "py", "python" };
    for (const auto& launcher : launchers)
    {
        juce::ChildProcess probe;
        juce::StringArray probeCommand { launcher, "-m", "demucs", "--help" };
        if (!probe.start(probeCommand))
            continue;

        if (!probe.waitForProcessToFinish(12000))
            continue;

        if (probe.getExitCode() == 0)
        {
            outCommandPrefix = { launcher, "-m", "demucs" };
            return true;
        }
    }

    return false;
}

int stemSortRank(const juce::String& rawName)
{
    const auto name = rawName.trim().toLowerCase();
    if (name == "vocals")
        return 0;
    if (name == "drums")
        return 1;
    if (name == "bass")
        return 2;
    if (name == "guitar")
        return 3;
    if (name == "piano")
        return 4;
    if (name == "other")
        return 5;
    return 100;
}

juce::String stemDisplayName(const juce::File& file)
{
    auto name = file.getFileNameWithoutExtension().trim();
    if (name.isEmpty())
        name = "Stem";

    if (name.length() == 1)
        return name.toUpperCase();

    return name.substring(0, 1).toUpperCase() + name.substring(1);
}

bool matchesCommandShortcut(const juce::KeyPress& key, char letter)
{
    if (!key.getModifiers().isCommandDown())
        return false;

    const auto normalisedLetter = juce::CharacterFunctions::toLowerCase(static_cast<wchar_t>(letter));
    const auto normalisedKeyCode = juce::CharacterFunctions::toLowerCase(static_cast<wchar_t>(key.getKeyCode()));
    const auto normalisedTextCharacter = juce::CharacterFunctions::toLowerCase(static_cast<wchar_t>(key.getTextCharacter()));
    return normalisedKeyCode == normalisedLetter || normalisedTextCharacter == normalisedLetter;
}

juce::String midiInputDisplayName(const juce::MidiDeviceInfo& device)
{
    return device.name.trim().isNotEmpty() ? device.name.trim() : device.identifier.trim();
}

juce::String describeMidiInputSelection(const juce::String& selectedIdentifier,
                                        const juce::Array<juce::MidiDeviceInfo>& availableDevices,
                                        const juce::StringArray& activeDeviceNames)
{
    if (selectedIdentifier == kMidiInputSelectionDisabled)
        return "MIDI input disabled.";

    if (selectedIdentifier.isEmpty())
    {
        return activeDeviceNames.isEmpty()
            ? "No MIDI input devices detected."
            : "Listening to all MIDI inputs: " + activeDeviceNames.joinIntoString(", ");
    }

    for (const auto& device : availableDevices)
    {
        if (device.identifier == selectedIdentifier)
            return "Listening to MIDI input: " + midiInputDisplayName(device);
    }

    return "Selected MIDI input is unavailable.";
}

juce::File detectedOllamaExecutable()
{
    juce::StringArray candidates;

    const auto localAppData = juce::SystemStats::getEnvironmentVariable("LOCALAPPDATA", {}).trim();
    if (localAppData.isNotEmpty())
    {
        const juce::File localBase(localAppData);
        candidates.add(localBase.getChildFile("Programs").getChildFile("Ollama").getChildFile("ollama.exe").getFullPathName());
        candidates.add(localBase.getChildFile("Programs").getChildFile("Ollama").getChildFile("Ollama app.exe").getFullPathName());
    }

    const auto programFiles = juce::SystemStats::getEnvironmentVariable("ProgramFiles", {}).trim();
    if (programFiles.isNotEmpty())
    {
        const juce::File programFilesBase(programFiles);
        candidates.add(programFilesBase.getChildFile("Ollama").getChildFile("ollama.exe").getFullPathName());
        candidates.add(programFilesBase.getChildFile("Ollama").getChildFile("Ollama app.exe").getFullPathName());
    }

    const auto pathEntries = juce::StringArray::fromTokens(juce::SystemStats::getEnvironmentVariable("PATH", {}),
                                                           ";",
                                                           "\"");
    for (const auto& entry : pathEntries)
    {
        const auto trimmed = entry.trim();
        if (trimmed.isEmpty())
            continue;

        candidates.add(juce::File(trimmed).getChildFile("ollama.exe").getFullPathName());
        candidates.add(juce::File(trimmed).getChildFile("Ollama app.exe").getFullPathName());
    }

    for (const auto& candidate : candidates)
    {
        const juce::File file(candidate);
        if (file.existsAsFile())
            return file;
    }

    return {};
}

bool isOllamaInstalledLocally()
{
    return detectedOllamaExecutable().existsAsFile();
}

juce::File detectedAceStepInstallDirectory()
{
    juce::Array<juce::File> candidates;
    const auto currentWorkingDirectory = juce::File::getCurrentWorkingDirectory();
    candidates.add(currentWorkingDirectory.getChildFile("ACE-Step-1.5"));
    candidates.add(currentWorkingDirectory.getChildFile("tmp").getChildFile("ACE-Step-1.5"));

    const auto executableDirectory = juce::File::getSpecialLocation(juce::File::currentExecutableFile).getParentDirectory();
    candidates.add(executableDirectory.getChildFile("ACE-Step-1.5"));
    candidates.add(executableDirectory.getParentDirectory().getChildFile("ACE-Step-1.5"));

    const auto userHome = juce::File::getSpecialLocation(juce::File::userHomeDirectory);
    candidates.add(userHome.getChildFile("Documents").getChildFile("GitHub").getChildFile("ACE-Step-1.5"));
    candidates.add(userHome.getChildFile("OneDrive").getChildFile("Documents").getChildFile("GitHub").getChildFile("ACE-Step-1.5"));
    candidates.add(userHome.getChildFile("Downloads").getChildFile("ACE-Step-1.5"));
    candidates.add(userHome.getChildFile("ACE-Step-1.5"));
    const auto localAppData = juce::SystemStats::getEnvironmentVariable("LOCALAPPDATA", {}).trim();
    if (localAppData.isNotEmpty())
        candidates.add(juce::File(localAppData).getChildFile("Mutagen").getChildFile("ACE-Step-1.5"));

    for (const auto& candidate : candidates)
    {
        if (!candidate.isDirectory())
            continue;

       #if JUCE_WINDOWS
        if (candidate.getChildFile("mutagen-start-api-server.bat").existsAsFile())
            return candidate;
        if (candidate.getChildFile("start_api_server.bat").existsAsFile())
            return candidate;
       #elif JUCE_MAC
        if (candidate.getChildFile("start_api_server_macos.sh").existsAsFile())
            return candidate;
       #else
        if (candidate.getChildFile("start_api_server.sh").existsAsFile())
            return candidate;
       #endif
    }

    return {};
}

juce::String aceStepTimeSignatureValue(const ProjectState& project)
{
    if (project.timeSigNumerator == 2 && project.timeSigDenominator == 4)
        return "2";
    if (project.timeSigNumerator == 3 && project.timeSigDenominator == 4)
        return "3";
    if (project.timeSigNumerator == 4 && project.timeSigDenominator == 4)
        return "4";
    if (project.timeSigNumerator == 6 && project.timeSigDenominator == 8)
        return "6";
    return {};
}

juce::String aceStepKeyScaleValue(const ProjectState& project)
{
    const auto display = keyQuantizeDisplayName(project.keyQuantizeRoot, project.keyQuantizeScale).trim();
    if (display.equalsIgnoreCase("All Notes (Chromatic)"))
        return {};
    return display;
}

juce::StringArray detectedOllamaModelChoices(const AIClient& aiClient,
                                             const juce::String& baseUrl,
                                             const juce::String& preferredModel = {})
{
    juce::StringArray choices;

    try
    {
        choices = aiClient.availableOllamaModels(baseUrl);
    }
    catch (const std::exception&)
    {
    }

    const auto trimmedPreferredModel = preferredModel.trim();
    if (trimmedPreferredModel.isNotEmpty() && !choices.contains(trimmedPreferredModel))
        choices.insert(0, trimmedPreferredModel);

    return choices;
}

void populateComboBoxChoices(juce::ComboBox& comboBox,
                             const juce::StringArray& choices,
                             const juce::String& preferredText = {})
{
    comboBox.setEditableText(true);
    comboBox.clear(juce::dontSendNotification);

    int itemId = 1;
    for (const auto& choice : choices)
        comboBox.addItem(choice, itemId++);

    const auto desiredText = preferredText.trim();
    if (desiredText.isNotEmpty())
        comboBox.setText(desiredText, juce::dontSendNotification);
    else if (!choices.isEmpty())
        comboBox.setSelectedItemIndex(0, juce::dontSendNotification);
    else
        comboBox.setText({}, juce::dontSendNotification);
}

void refreshOllamaModelCombo(juce::ComboBox& modelBox,
                             const AIClient& aiClient,
                             const juce::String& baseUrl,
                             const juce::String& preferredModel = {})
{
    const auto resolvedBaseUrl = baseUrl.trim().isNotEmpty() ? baseUrl.trim() : aiClient.getOllamaBaseUrl();
    const auto choices = detectedOllamaModelChoices(aiClient, resolvedBaseUrl, preferredModel);
    populateComboBoxChoices(modelBox, choices, preferredModel);

    modelBox.setTooltip(choices.isEmpty()
                            ? "No Ollama models were detected at the configured endpoint. You can still type a model tag manually."
                            : "Detected Ollama models from " + resolvedBaseUrl + ". You can still type a custom model tag.");
}

const std::array<AIClient::Provider, 6>& aiSettingsProviderOrder()
{
    static const std::array<AIClient::Provider, 6> providers
    {
        AIClient::Provider::openAI,
        AIClient::Provider::anthropic,
        AIClient::Provider::xAI,
        AIClient::Provider::gemini,
        AIClient::Provider::openAICompatible,
        AIClient::Provider::ollama
    };

    return providers;
}

juce::StringArray aiSettingsProviderLabels()
{
    juce::StringArray labels;
    for (const auto provider : aiSettingsProviderOrder())
        labels.add(AIClient::providerDisplayName(provider));
    return labels;
}

AIClient::Provider aiProviderFromDisplayLabel(const juce::String& label)
{
    for (const auto provider : aiSettingsProviderOrder())
    {
        if (label.equalsIgnoreCase(AIClient::providerDisplayName(provider)))
            return provider;
    }

    return AIClient::Provider::openAI;
}

juce::String aiProviderHelpText(AIClient::Provider provider)
{
    switch (provider)
    {
        case AIClient::Provider::openAI:
            return "Uses OpenAI's native Responses API.";
        case AIClient::Provider::anthropic:
            return "Uses Claude's OpenAI-compatible endpoint.";
        case AIClient::Provider::xAI:
            return "Uses xAI's chat-completions API.";
        case AIClient::Provider::gemini:
            return "Uses Gemini's OpenAI compatibility endpoint.";
        case AIClient::Provider::openAICompatible:
            return "Enter any OpenAI-compatible /v1 endpoint.";
        case AIClient::Provider::ollama:
            break;
    }

    return {};
}

class AiSettingsContentComponent final : public juce::Component
{
public:
    AiSettingsContentComponent(const AIClient& aiClientIn,
                               const AceStepClient& aceStepClientIn,
                               bool ollamaDetectedIn,
                               const juce::File& installedOllamaIn,
                               const juce::File& detectedAceStepIn)
        : aiClient(aiClientIn),
          ollamaDetected(ollamaDetectedIn),
          installedOllama(installedOllamaIn)
    {
        const auto configureLabel = [this] (juce::Label& label, const juce::String& text, bool bold = false)
        {
            label.setText(text, juce::dontSendNotification);
            label.setJustificationType(juce::Justification::centredLeft);
            if (bold)
                label.setFont(label.getFont().boldened());
            addAndMakeVisible(label);
        };

        const auto configureEditor = [this] (juce::TextEditor& editor, const juce::String& text)
        {
            editor.setText(text, juce::dontSendNotification);
            editor.setSelectAllWhenFocused(true);
            addAndMakeVisible(editor);
        };

        configureLabel(providerSectionLabel, "Composition Provider", true);
        configureLabel(providerLabel, "Provider");
        providerBox.addItemList(aiSettingsProviderLabels(), 1);
        providerBox.onChange = [this] { handleProviderChanged(); };
        addAndMakeVisible(providerBox);

        configureLabel(remoteModelLabel, "Remote model");
        remoteModelBox.setEditableText(true);
        addAndMakeVisible(remoteModelBox);
        configureLabel(remoteApiKeyLabel, "API key");
        configureEditor(remoteApiKeyEditor, {});
        remoteApiKeyEditor.setPasswordCharacter(0x2022);
        configureLabel(remoteEndpointLabel, "Custom endpoint");
        configureEditor(remoteEndpointEditor, aiClient.getRemoteBaseUrl());
        configureLabel(remoteHelpLabel, {});

        configureLabel(ollamaSectionLabel, "Ollama", true);
        configureLabel(ollamaBaseUrlLabel, "Ollama endpoint");
        configureEditor(ollamaBaseUrlEditor, aiClient.getOllamaBaseUrl());
        ollamaBaseUrlEditor.onReturnKey = [this] { refreshOllamaModels(); };
        ollamaBaseUrlEditor.onFocusLost = [this] { refreshOllamaModels(); };
        configureLabel(ollamaModelLabel, "Ollama model");
        ollamaModelBox.setEditableText(true);
        addAndMakeVisible(ollamaModelBox);
        refreshOllamaButton.setButtonText("Refresh");
        refreshOllamaButton.onClick = [this] { refreshOllamaModels(); };
        addAndMakeVisible(refreshOllamaButton);
        configureLabel(ollamaStatusLabel,
                       ollamaDetected
                           ? "Detected local Ollama at " + installedOllama.getFullPathName()
                           : "Ollama was not detected on this system.");
        ollamaDownloadButton.setButtonText("Download Ollama");
        ollamaDownloadButton.onClick = []
        {
            juce::URL(kOllamaWindowsDownloadUrl).launchInDefaultBrowser();
        };
        addAndMakeVisible(ollamaDownloadButton);

        configureLabel(timeoutLabel, "Request timeout (seconds)");
        configureEditor(timeoutEditor, juce::String(aiClient.getRequestTimeoutSeconds()));

        configureLabel(aceStepSectionLabel, "ACE-Step Audio Generation", true);
        configureLabel(aceStepBaseUrlLabel, "ACE-Step endpoint");
        configureEditor(aceStepBaseUrlEditor, aceStepClientIn.getBaseUrl());
        configureLabel(aceStepApiKeyLabel, "ACE-Step API key");
        configureEditor(aceStepApiKeyEditor, {});
        aceStepApiKeyEditor.setPasswordCharacter(0x2022);
        configureLabel(aceStepInstallDirLabel, "ACE-Step install folder");
        configureEditor(aceStepInstallDirEditor,
                        aceStepClientIn.getInstallDirectory().trim().isNotEmpty()
                            ? aceStepClientIn.getInstallDirectory()
                            : detectedAceStepIn.getFullPathName());
        configureLabel(aceStepAutoStartLabel, "ACE-Step auto-start");
        aceStepAutoStartBox.addItemList({ "Enabled", "Disabled" }, 1);
        aceStepAutoStartBox.setSelectedItemIndex(aceStepClientIn.getAutoStartServer() ? 0 : 1, juce::dontSendNotification);
        addAndMakeVisible(aceStepAutoStartBox);
        configureLabel(aceStepDefaultModelLabel, "ACE-Step default model");
        configureEditor(aceStepDefaultModelEditor, aceStepClientIn.getDefaultModel());
        configureLabel(aceStepAudioFormatLabel, "ACE-Step output format");
        aceStepAudioFormatBox.addItemList({ "wav", "flac", "mp3" }, 1);
        aceStepAudioFormatBox.setSelectedItemIndex(juce::jmax(0,
                                                              juce::StringArray{ "wav", "flac", "mp3" }.indexOf(aceStepClientIn.getDefaultAudioFormat())),
                                                   juce::dontSendNotification);
        addAndMakeVisible(aceStepAudioFormatBox);
        configureLabel(aceStepStartupTimeoutLabel, "Startup timeout (seconds)");
        configureEditor(aceStepStartupTimeoutEditor, juce::String(aceStepClientIn.getStartupTimeoutSeconds()));
        configureLabel(aceStepJobTimeoutLabel, "Job timeout (seconds)");
        configureEditor(aceStepJobTimeoutEditor, juce::String(aceStepClientIn.getJobTimeoutSeconds()));
        configureLabel(aceStepHelpLabel, "Only set the install folder when Mutagen should launch ACE-Step for you.");
        aceStepRepoButton.setButtonText("Open ACE-Step Repository");
        aceStepRepoButton.onClick = []
        {
            juce::URL(kAceStepGitHubUrl).launchInDefaultBrowser();
        };
        addAndMakeVisible(aceStepRepoButton);

        int providerIndex = 0;
        for (int index = 0; index < static_cast<int>(aiSettingsProviderOrder().size()); ++index)
        {
            if (aiSettingsProviderOrder()[static_cast<size_t>(index)] == aiClient.getProvider())
            {
                providerIndex = index;
                break;
            }
        }

        providerBox.setSelectedItemIndex(providerIndex, juce::dontSendNotification);
        lastProvider = selectedProvider();
        remoteModelBox.setText(aiClient.getRemoteModel(), juce::dontSendNotification);
        refreshOllamaModels();
        handleProviderChanged();
    }

    int preferredDialogHeight() const
    {
        return juce::jlimit(620, 900, computeContentHeight() + 130);
    }

    AIClient::Provider selectedProvider() const
    {
        return aiProviderFromDisplayLabel(providerBox.getText());
    }

    juce::String remoteModel() const { return remoteModelBox.getText().trim(); }
    juce::String remoteApiKey() const { return remoteApiKeyEditor.getText().trim(); }
    juce::String remoteBaseUrl() const
    {
        return customRemoteEndpointVisible()
            ? remoteEndpointEditor.getText().trim()
            : AIClient::defaultRemoteBaseUrlForProvider(selectedProvider());
    }
    juce::String ollamaBaseUrl() const { return ollamaBaseUrlEditor.getText().trim(); }
    juce::String ollamaModel() const { return ollamaModelBox.getText().trim(); }
    int timeoutSeconds() const { return timeoutEditor.getText().getIntValue(); }
    juce::String aceStepBaseUrl() const { return aceStepBaseUrlEditor.getText().trim(); }
    juce::String aceStepApiKey() const { return aceStepApiKeyEditor.getText().trim(); }
    juce::String aceStepInstallDirectory() const { return aceStepInstallDirEditor.getText().trim(); }
    bool aceStepAutoStartEnabled() const { return aceStepAutoStartBox.getSelectedItemIndex() != 1; }
    juce::String aceStepDefaultModel() const { return aceStepDefaultModelEditor.getText().trim(); }
    juce::String aceStepAudioFormat() const { return aceStepAudioFormatBox.getText().trim(); }
    int aceStepStartupTimeoutSeconds() const { return aceStepStartupTimeoutEditor.getText().getIntValue(); }
    int aceStepJobTimeoutSeconds() const { return aceStepJobTimeoutEditor.getText().getIntValue(); }

    void resized() override
    {
        auto area = getLocalBounds().reduced(12);
        constexpr int labelWidth = 180;
        constexpr int rowHeight = 28;
        constexpr int sectionGap = 10;
        constexpr int rowGap = 6;

        const auto layoutSection = [&area, sectionGap] (juce::Label& label)
        {
            label.setBounds(area.removeFromTop(24));
            area.removeFromTop(sectionGap);
        };

        const auto layoutRow = [&area, labelWidth, rowHeight, rowGap] (juce::Label& label, juce::Component& field)
        {
            auto row = area.removeFromTop(rowHeight);
            label.setBounds(row.removeFromLeft(labelWidth));
            field.setBounds(row);
            area.removeFromTop(rowGap);
        };

        const auto layoutRowWithButton = [&area, labelWidth, rowHeight, rowGap] (juce::Label& label,
                                                                                  juce::Component& field,
                                                                                  juce::Component& button,
                                                                                  int buttonWidth)
        {
            auto row = area.removeFromTop(rowHeight);
            label.setBounds(row.removeFromLeft(labelWidth));
            button.setBounds(row.removeFromRight(buttonWidth));
            row.removeFromRight(6);
            field.setBounds(row);
            area.removeFromTop(rowGap);
        };

        const auto layoutStandalone = [&area, rowGap] (juce::Component& component, int height)
        {
            component.setBounds(area.removeFromTop(height));
            area.removeFromTop(rowGap);
        };

        layoutSection(providerSectionLabel);
        layoutRow(providerLabel, providerBox);

        if (remoteSectionVisible())
        {
            layoutRow(remoteModelLabel, remoteModelBox);
            layoutRow(remoteApiKeyLabel, remoteApiKeyEditor);
            if (customRemoteEndpointVisible())
                layoutRow(remoteEndpointLabel, remoteEndpointEditor);
            layoutStandalone(remoteHelpLabel, 22);
        }

        if (ollamaSectionVisible())
        {
            layoutSection(ollamaSectionLabel);
            layoutRow(ollamaBaseUrlLabel, ollamaBaseUrlEditor);
            layoutRowWithButton(ollamaModelLabel, ollamaModelBox, refreshOllamaButton, 84);
            layoutStandalone(ollamaStatusLabel, 22);
            if (!ollamaDetected)
                layoutStandalone(ollamaDownloadButton, 24);
        }

        layoutRow(timeoutLabel, timeoutEditor);
        area.removeFromTop(sectionGap);
        layoutSection(aceStepSectionLabel);
        layoutRow(aceStepBaseUrlLabel, aceStepBaseUrlEditor);
        layoutRow(aceStepApiKeyLabel, aceStepApiKeyEditor);
        layoutRow(aceStepInstallDirLabel, aceStepInstallDirEditor);
        layoutRow(aceStepAutoStartLabel, aceStepAutoStartBox);
        layoutRow(aceStepDefaultModelLabel, aceStepDefaultModelEditor);
        layoutRow(aceStepAudioFormatLabel, aceStepAudioFormatBox);
        layoutRow(aceStepStartupTimeoutLabel, aceStepStartupTimeoutEditor);
        layoutRow(aceStepJobTimeoutLabel, aceStepJobTimeoutEditor);
        layoutStandalone(aceStepHelpLabel, 22);
        layoutStandalone(aceStepRepoButton, 24);
    }

private:
    bool remoteSectionVisible() const noexcept { return selectedProvider() != AIClient::Provider::ollama; }
    bool ollamaSectionVisible() const noexcept { return selectedProvider() == AIClient::Provider::ollama; }
    bool customRemoteEndpointVisible() const noexcept { return selectedProvider() == AIClient::Provider::openAICompatible; }

    int computeContentHeight() const
    {
        int height = 24 + 10 + 28 + 6;

        if (remoteSectionVisible())
        {
            height += 28 + 6;
            height += 28 + 6;
            if (customRemoteEndpointVisible())
                height += 28 + 6;
            height += 22 + 6;
        }

        if (ollamaSectionVisible())
        {
            height += 24 + 10;
            height += 28 + 6;
            height += 28 + 6;
            height += 22 + 6;
            if (!ollamaDetected)
                height += 24 + 6;
        }

        height += 28 + 6;
        height += 10;
        height += 24 + 10;
        height += (28 + 6) * 8;
        height += 22 + 6;
        height += 24 + 6;
        return height + 24;
    }

    void refreshOllamaModels()
    {
        auto preferredModel = ollamaModelBox.getText().trim();
        if (preferredModel.isEmpty())
            preferredModel = aiClient.getOllamaModel().trim();
        refreshOllamaModelCombo(ollamaModelBox, aiClient, ollamaBaseUrlEditor.getText().trim(), preferredModel);
    }

    void handleProviderChanged()
    {
        const auto newProvider = selectedProvider();
        const auto previousDefaultModel = AIClient::defaultRemoteModelForProvider(lastProvider);
        auto currentModel = remoteModelBox.getText().trim();
        if (currentModel.isEmpty() || currentModel == previousDefaultModel)
            currentModel = AIClient::defaultRemoteModelForProvider(newProvider);

        remoteModelLabel.setText(AIClient::providerDisplayName(newProvider) + " model", juce::dontSendNotification);
        remoteApiKeyLabel.setText(AIClient::providerDisplayName(newProvider) + " API key", juce::dontSendNotification);
        remoteHelpLabel.setText(aiProviderHelpText(newProvider), juce::dontSendNotification);
        populateComboBoxChoices(remoteModelBox, aiClient.remoteModelChoices(newProvider), currentModel);

        const auto setVisibility = [] (bool visible, std::initializer_list<juce::Component*> components)
        {
            for (auto* component : components)
                if (component != nullptr)
                    component->setVisible(visible);
        };

        setVisibility(remoteSectionVisible(),
                      { &remoteModelLabel, &remoteModelBox, &remoteApiKeyLabel, &remoteApiKeyEditor, &remoteHelpLabel });
        setVisibility(customRemoteEndpointVisible(), { &remoteEndpointLabel, &remoteEndpointEditor });
        setVisibility(ollamaSectionVisible(),
                      { &ollamaSectionLabel, &ollamaBaseUrlLabel, &ollamaBaseUrlEditor, &ollamaModelLabel, &ollamaModelBox,
                        &refreshOllamaButton, &ollamaStatusLabel });
        ollamaDownloadButton.setVisible(ollamaSectionVisible() && !ollamaDetected);

        lastProvider = newProvider;
        resized();

        if (auto* alertWindow = findParentComponentOfClass<juce::AlertWindow>())
            alertWindow->setSize(alertWindow->getWidth(), preferredDialogHeight());
    }

    const AIClient& aiClient;
    bool ollamaDetected = false;
    juce::File installedOllama;
    AIClient::Provider lastProvider = AIClient::Provider::openAI;

    juce::Label providerSectionLabel;
    juce::Label providerLabel;
    juce::ComboBox providerBox;
    juce::Label remoteModelLabel;
    juce::ComboBox remoteModelBox;
    juce::Label remoteApiKeyLabel;
    juce::TextEditor remoteApiKeyEditor;
    juce::Label remoteEndpointLabel;
    juce::TextEditor remoteEndpointEditor;
    juce::Label remoteHelpLabel;
    juce::Label ollamaSectionLabel;
    juce::Label ollamaBaseUrlLabel;
    juce::TextEditor ollamaBaseUrlEditor;
    juce::Label ollamaModelLabel;
    juce::ComboBox ollamaModelBox;
    juce::TextButton refreshOllamaButton;
    juce::Label ollamaStatusLabel;
    juce::TextButton ollamaDownloadButton;
    juce::Label timeoutLabel;
    juce::TextEditor timeoutEditor;
    juce::Label aceStepSectionLabel;
    juce::Label aceStepBaseUrlLabel;
    juce::TextEditor aceStepBaseUrlEditor;
    juce::Label aceStepApiKeyLabel;
    juce::TextEditor aceStepApiKeyEditor;
    juce::Label aceStepInstallDirLabel;
    juce::TextEditor aceStepInstallDirEditor;
    juce::Label aceStepAutoStartLabel;
    juce::ComboBox aceStepAutoStartBox;
    juce::Label aceStepDefaultModelLabel;
    juce::TextEditor aceStepDefaultModelEditor;
    juce::Label aceStepAudioFormatLabel;
    juce::ComboBox aceStepAudioFormatBox;
    juce::Label aceStepStartupTimeoutLabel;
    juce::TextEditor aceStepStartupTimeoutEditor;
    juce::Label aceStepJobTimeoutLabel;
    juce::TextEditor aceStepJobTimeoutEditor;
    juce::Label aceStepHelpLabel;
    juce::TextButton aceStepRepoButton;
};

class TabbedAiSettingsContentComponent final : public juce::Component
{
public:
    TabbedAiSettingsContentComponent(const AIClient& aiClientIn,
                                     const AceStepClient& aceStepClientIn,
                                     bool ollamaDetectedIn,
                                     const juce::File& installedOllamaIn,
                                     const juce::File& detectedAceStepIn);

    int preferredDialogWidth() const noexcept;
    int preferredDialogHeight() const noexcept;

    AIClient::Provider selectedProvider() const;
    juce::String remoteModel() const;
    juce::String remoteApiKey() const;
    juce::String remoteBaseUrl() const;
    juce::String ollamaBaseUrl() const;
    juce::String ollamaModel() const;
    int timeoutSeconds() const;
    juce::String aceStepBaseUrl() const;
    juce::String aceStepApiKey() const;
    juce::String aceStepInstallDirectory() const;
    bool aceStepAutoStartEnabled() const;
    juce::String aceStepDefaultModel() const;
    juce::String aceStepAudioFormat() const;
    juce::String aceStepQualityPreset() const;
    juce::String aceStepVocalLanguage() const;
    bool aceStepThinkingEnabled() const;
    bool aceStepUseRandomSeed() const;
    int aceStepSeed() const;
    int aceStepInferenceSteps() const;
    double aceStepGuidanceScale() const;
    juce::String aceStepInferMethod() const;
    int aceStepStartupTimeoutSeconds() const;
    int aceStepJobTimeoutSeconds() const;

    void resized() override;

private:
    static bool aceStepModelLooksTurbo(const juce::String& model) noexcept;
    bool remoteSectionVisible() const noexcept;
    bool ollamaSectionVisible() const noexcept;
    bool customRemoteEndpointVisible() const noexcept;
    bool aceStepCustomQualitySelected() const noexcept;
    int aceStepPresetInferenceSteps(const juce::String& preset) const;
    int computeCompositionContentHeight() const;
    int computeAceStepContentHeight() const;
    void layoutCompositionPage();
    void layoutAceStepPage();
    void refreshOllamaModels();
    void applyAceStepQualityPreset(const juce::String& preset);
    void updateAceStepFieldStates();
    void handleAceStepQualityChanged();
    void handleAceStepModelChanged();
    void handleProviderChanged();

    const AIClient& aiClient;
    const AceStepClient& aceStepClient;
    bool ollamaDetected = false;
    bool updatingAceStepQualityControls = false;
    juce::File installedOllama;
    AIClient::Provider lastProvider = AIClient::Provider::openAI;
    juce::TabbedComponent tabs;
    juce::Viewport compositionViewport;
    juce::Viewport aceStepViewport;
    juce::Component compositionPage;
    juce::Component aceStepPage;

    juce::Label providerSectionLabel;
    juce::Label providerLabel;
    juce::ComboBox providerBox;
    juce::Label remoteModelLabel;
    juce::ComboBox remoteModelBox;
    juce::Label remoteApiKeyLabel;
    juce::TextEditor remoteApiKeyEditor;
    juce::Label remoteEndpointLabel;
    juce::TextEditor remoteEndpointEditor;
    juce::Label remoteHelpLabel;
    juce::Label ollamaSectionLabel;
    juce::Label ollamaBaseUrlLabel;
    juce::TextEditor ollamaBaseUrlEditor;
    juce::Label ollamaModelLabel;
    juce::ComboBox ollamaModelBox;
    juce::TextButton refreshOllamaButton;
    juce::Label ollamaStatusLabel;
    juce::TextButton ollamaDownloadButton;
    juce::Label requestSectionLabel;
    juce::Label timeoutLabel;
    juce::TextEditor timeoutEditor;

    juce::Label aceStepConnectionSectionLabel;
    juce::Label aceStepBaseUrlLabel;
    juce::TextEditor aceStepBaseUrlEditor;
    juce::Label aceStepApiKeyLabel;
    juce::TextEditor aceStepApiKeyEditor;
    juce::Label aceStepInstallDirLabel;
    juce::TextEditor aceStepInstallDirEditor;
    juce::Label aceStepAutoStartLabel;
    juce::ComboBox aceStepAutoStartBox;
    juce::Label aceStepStartupTimeoutLabel;
    juce::TextEditor aceStepStartupTimeoutEditor;
    juce::Label aceStepJobTimeoutLabel;
    juce::TextEditor aceStepJobTimeoutEditor;
    juce::Label aceStepHelpLabel;
    juce::Label aceStepRenderSectionLabel;
    juce::Label aceStepQualityLabel;
    juce::ComboBox aceStepQualityBox;
    juce::Label aceStepDefaultModelLabel;
    juce::TextEditor aceStepDefaultModelEditor;
    juce::Label aceStepAudioFormatLabel;
    juce::ComboBox aceStepAudioFormatBox;
    juce::Label aceStepLanguageLabel;
    juce::TextEditor aceStepLanguageEditor;
    juce::Label aceStepSeedModeLabel;
    juce::ComboBox aceStepSeedModeBox;
    juce::Label aceStepSeedLabel;
    juce::TextEditor aceStepSeedEditor;
    juce::Label aceStepAdvancedSectionLabel;
    juce::Label aceStepThinkingLabel;
    juce::ComboBox aceStepThinkingBox;
    juce::Label aceStepInferenceStepsLabel;
    juce::TextEditor aceStepInferenceStepsEditor;
    juce::Label aceStepGuidanceScaleLabel;
    juce::TextEditor aceStepGuidanceScaleEditor;
    juce::Label aceStepInferMethodLabel;
    juce::ComboBox aceStepInferMethodBox;
    juce::Label aceStepAdvancedHelpLabel;
    juce::TextButton aceStepRepoButton;
};

class AiSettingsDialogContainerComponent final : public juce::Component
{
public:
    using SubmitHandler = std::function<void(int, const TabbedAiSettingsContentComponent&)>;

    AiSettingsDialogContainerComponent(const AIClient& aiClientIn,
                                       const AceStepClient& aceStepClientIn,
                                       bool ollamaDetectedIn,
                                       const juce::File& installedOllamaIn,
                                       const juce::File& detectedAceStepIn,
                                       SubmitHandler submitHandlerIn)
        : settingsContent(aiClientIn,
                          aceStepClientIn,
                          ollamaDetectedIn,
                          installedOllamaIn,
                          detectedAceStepIn),
          submitHandler(std::move(submitHandlerIn))
    {
        addAndMakeVisible(settingsContent);

        saveButton.setButtonText("Save");
        saveButton.onClick = [this] { closeWithResult(1); };
        addAndMakeVisible(saveButton);

        clearAuthButton.setButtonText("Clear Auth");
        clearAuthButton.onClick = [this] { closeWithResult(2); };
        addAndMakeVisible(clearAuthButton);

        cancelButton.setButtonText("Cancel");
        cancelButton.onClick = [this] { closeWithResult(0); };
        addAndMakeVisible(cancelButton);

        setSize(settingsContent.preferredDialogWidth() + 24,
                settingsContent.preferredDialogHeight() + 72);
    }

    void paint(juce::Graphics& g) override
    {
        g.fillAll(juce::Colour::fromRGB(52, 63, 72));
    }

    void resized() override
    {
        auto area = getLocalBounds().reduced(12);
        auto buttonsArea = area.removeFromBottom(44);
        settingsContent.setBounds(area);

        constexpr int buttonWidth = 132;
        constexpr int buttonHeight = 38;
        constexpr int buttonGap = 12;
        const auto totalWidth = (buttonWidth * 3) + (buttonGap * 2);
        auto row = buttonsArea.withSizeKeepingCentre(totalWidth, buttonHeight);
        saveButton.setBounds(row.removeFromLeft(buttonWidth));
        row.removeFromLeft(buttonGap);
        clearAuthButton.setBounds(row.removeFromLeft(buttonWidth));
        row.removeFromLeft(buttonGap);
        cancelButton.setBounds(row.removeFromLeft(buttonWidth));
    }

private:
    void closeWithResult(int result)
    {
        if ((result == 1 || result == 2) && submitHandler != nullptr)
            submitHandler(result, settingsContent);

        if (auto* dialogWindow = findParentComponentOfClass<juce::DialogWindow>())
        {
            dialogWindow->exitModalState(result);
            dialogWindow->setVisible(false);
        }
    }

    TabbedAiSettingsContentComponent settingsContent;
    SubmitHandler submitHandler;
    juce::TextButton saveButton;
    juce::TextButton clearAuthButton;
    juce::TextButton cancelButton;
};

TabbedAiSettingsContentComponent::TabbedAiSettingsContentComponent(const AIClient& aiClientIn,
                                                                   const AceStepClient& aceStepClientIn,
                                                                   bool ollamaDetectedIn,
                                                                   const juce::File& installedOllamaIn,
                                                                   const juce::File& detectedAceStepIn)
    : aiClient(aiClientIn),
      aceStepClient(aceStepClientIn),
      ollamaDetected(ollamaDetectedIn),
      installedOllama(installedOllamaIn),
      tabs(juce::TabbedButtonBar::TabsAtTop)
{
    tabs.setTabBarDepth(32);
    compositionViewport.setViewedComponent(&compositionPage, false);
    compositionViewport.setScrollBarsShown(true, false);
    aceStepViewport.setViewedComponent(&aceStepPage, false);
    aceStepViewport.setScrollBarsShown(true, false);
    tabs.addTab("Composition", juce::Colour::fromRGB(34, 46, 62), &compositionViewport, false);
    tabs.addTab("ACE-Step", juce::Colour::fromRGB(34, 46, 62), &aceStepViewport, false);
    addAndMakeVisible(tabs);

    const auto configureLabel = [] (juce::Component& parent,
                                    juce::Label& label,
                                    const juce::String& text,
                                    bool bold = false)
    {
        label.setText(text, juce::dontSendNotification);
        label.setJustificationType(juce::Justification::centredLeft);
        if (bold)
            label.setFont(label.getFont().boldened());
        parent.addAndMakeVisible(label);
    };

    const auto configureEditor = [] (juce::Component& parent,
                                     juce::TextEditor& editor,
                                     const juce::String& text)
    {
        editor.setText(text, juce::dontSendNotification);
        editor.setSelectAllWhenFocused(true);
        parent.addAndMakeVisible(editor);
    };

    configureLabel(compositionPage, providerSectionLabel, "Composition Provider", true);
    configureLabel(compositionPage, providerLabel, "Provider");
    providerBox.addItemList(aiSettingsProviderLabels(), 1);
    providerBox.onChange = [this] { handleProviderChanged(); };
    compositionPage.addAndMakeVisible(providerBox);

    configureLabel(compositionPage, remoteModelLabel, "Remote model");
    remoteModelBox.setEditableText(true);
    compositionPage.addAndMakeVisible(remoteModelBox);
    configureLabel(compositionPage, remoteApiKeyLabel, "API key");
    configureEditor(compositionPage, remoteApiKeyEditor, {});
    remoteApiKeyEditor.setPasswordCharacter(0x2022);
    configureLabel(compositionPage, remoteEndpointLabel, "Custom endpoint");
    configureEditor(compositionPage, remoteEndpointEditor, aiClient.getRemoteBaseUrl());
    configureLabel(compositionPage, remoteHelpLabel, {});

    configureLabel(compositionPage, ollamaSectionLabel, "Ollama", true);
    configureLabel(compositionPage, ollamaBaseUrlLabel, "Ollama endpoint");
    configureEditor(compositionPage, ollamaBaseUrlEditor, aiClient.getOllamaBaseUrl());
    ollamaBaseUrlEditor.onReturnKey = [this] { refreshOllamaModels(); };
    ollamaBaseUrlEditor.onFocusLost = [this] { refreshOllamaModels(); };
    configureLabel(compositionPage, ollamaModelLabel, "Ollama model");
    ollamaModelBox.setEditableText(true);
    compositionPage.addAndMakeVisible(ollamaModelBox);
    refreshOllamaButton.setButtonText("Refresh");
    refreshOllamaButton.onClick = [this] { refreshOllamaModels(); };
    compositionPage.addAndMakeVisible(refreshOllamaButton);
    configureLabel(compositionPage,
                   ollamaStatusLabel,
                   ollamaDetected
                       ? "Detected local Ollama installation."
                       : "Ollama was not detected on this system.");
    if (ollamaDetected)
        ollamaStatusLabel.setTooltip(installedOllama.getFullPathName());
    ollamaDownloadButton.setButtonText("Download Ollama");
    ollamaDownloadButton.onClick = []
    {
        juce::URL(kOllamaWindowsDownloadUrl).launchInDefaultBrowser();
    };
    compositionPage.addAndMakeVisible(ollamaDownloadButton);

    configureLabel(compositionPage, requestSectionLabel, "Request Handling", true);
    configureLabel(compositionPage, timeoutLabel, "Request timeout (seconds)");
    configureEditor(compositionPage, timeoutEditor, juce::String(aiClient.getRequestTimeoutSeconds()));

    configureLabel(aceStepPage, aceStepConnectionSectionLabel, "Connection", true);
    configureLabel(aceStepPage, aceStepBaseUrlLabel, "ACE-Step endpoint");
    configureEditor(aceStepPage, aceStepBaseUrlEditor, aceStepClientIn.getBaseUrl());
    configureLabel(aceStepPage, aceStepApiKeyLabel, "ACE-Step API key");
    configureEditor(aceStepPage, aceStepApiKeyEditor, {});
    aceStepApiKeyEditor.setPasswordCharacter(0x2022);
    configureLabel(aceStepPage, aceStepInstallDirLabel, "ACE-Step install folder");
    configureEditor(aceStepPage,
                    aceStepInstallDirEditor,
                    aceStepClientIn.getInstallDirectory().trim().isNotEmpty()
                        ? aceStepClientIn.getInstallDirectory()
                        : detectedAceStepIn.getFullPathName());
    configureLabel(aceStepPage, aceStepAutoStartLabel, "Auto-start server");
    aceStepAutoStartBox.addItemList({ "Enabled", "Disabled" }, 1);
    aceStepAutoStartBox.setSelectedItemIndex(aceStepClientIn.getAutoStartServer() ? 0 : 1, juce::dontSendNotification);
    aceStepPage.addAndMakeVisible(aceStepAutoStartBox);
    configureLabel(aceStepPage, aceStepStartupTimeoutLabel, "Startup timeout (seconds)");
    configureEditor(aceStepPage, aceStepStartupTimeoutEditor, juce::String(aceStepClientIn.getStartupTimeoutSeconds()));
    configureLabel(aceStepPage, aceStepJobTimeoutLabel, "Job timeout (seconds)");
    configureEditor(aceStepPage, aceStepJobTimeoutEditor, juce::String(aceStepClientIn.getJobTimeoutSeconds()));
    configureLabel(aceStepPage, aceStepHelpLabel, "Install folder is only needed when Mutagen should launch ACE-Step for you.");

    configureLabel(aceStepPage, aceStepRenderSectionLabel, "Render Defaults", true);
    configureLabel(aceStepPage, aceStepQualityLabel, "Quality");
    aceStepQualityBox.addItemList({ "Fast", "Balanced", "High", "Custom" }, 1);
    aceStepQualityBox.onChange = [this] { handleAceStepQualityChanged(); };
    aceStepPage.addAndMakeVisible(aceStepQualityBox);
    configureLabel(aceStepPage, aceStepDefaultModelLabel, "Default model");
    configureEditor(aceStepPage, aceStepDefaultModelEditor, aceStepClientIn.getDefaultModel());
    aceStepDefaultModelEditor.onTextChange = [this] { handleAceStepModelChanged(); };
    configureLabel(aceStepPage, aceStepAudioFormatLabel, "Output format");
    aceStepAudioFormatBox.addItemList({ "wav", "flac", "mp3", "opus", "aac", "wav32" }, 1);
    aceStepAudioFormatBox.setSelectedItemIndex(juce::jmax(0,
                                                          juce::StringArray{ "wav", "flac", "mp3", "opus", "aac", "wav32" }
                                                              .indexOf(aceStepClientIn.getDefaultAudioFormat())),
                                               juce::dontSendNotification);
    aceStepPage.addAndMakeVisible(aceStepAudioFormatBox);
    configureLabel(aceStepPage, aceStepLanguageLabel, "Lyrics language");
    configureEditor(aceStepPage, aceStepLanguageEditor, aceStepClientIn.getDefaultVocalLanguage());
    configureLabel(aceStepPage, aceStepSeedModeLabel, "Seed");
    aceStepSeedModeBox.addItemList({ "Random", "Fixed" }, 1);
    aceStepSeedModeBox.setSelectedItemIndex(aceStepClientIn.getDefaultUseRandomSeed() ? 0 : 1, juce::dontSendNotification);
    aceStepSeedModeBox.onChange = [this] { updateAceStepFieldStates(); };
    aceStepPage.addAndMakeVisible(aceStepSeedModeBox);
    configureLabel(aceStepPage, aceStepSeedLabel, "Fixed seed");
    configureEditor(aceStepPage,
                    aceStepSeedEditor,
                    juce::String(aceStepClientIn.getDefaultUseRandomSeed() ? 0 : aceStepClientIn.getDefaultSeed()));

    configureLabel(aceStepPage, aceStepAdvancedSectionLabel, "Advanced", true);
    configureLabel(aceStepPage, aceStepThinkingLabel, "Thinking");
    aceStepThinkingBox.addItemList({ "Enabled", "Disabled" }, 1);
    aceStepThinkingBox.setSelectedItemIndex(aceStepClientIn.getDefaultThinking() ? 0 : 1, juce::dontSendNotification);
    aceStepPage.addAndMakeVisible(aceStepThinkingBox);
    configureLabel(aceStepPage, aceStepInferenceStepsLabel, "Inference steps");
    configureEditor(aceStepPage, aceStepInferenceStepsEditor, juce::String(aceStepClientIn.getDefaultInferenceSteps()));
    configureLabel(aceStepPage, aceStepGuidanceScaleLabel, "Guidance scale");
    configureEditor(aceStepPage, aceStepGuidanceScaleEditor, juce::String(aceStepClientIn.getDefaultGuidanceScale(), 2));
    configureLabel(aceStepPage, aceStepInferMethodLabel, "Infer method");
    aceStepInferMethodBox.addItemList({ "ode", "sde" }, 1);
    aceStepInferMethodBox.setSelectedItemIndex(juce::jmax(0,
                                                          juce::StringArray{ "ode", "sde" }.indexOf(aceStepClientIn.getDefaultInferMethod())),
                                               juce::dontSendNotification);
    aceStepPage.addAndMakeVisible(aceStepInferMethodBox);
    configureLabel(aceStepPage,
                   aceStepAdvancedHelpLabel,
                   "Switch Quality to Custom to edit thinking, inference steps, guidance, and infer method.");
    aceStepRepoButton.setButtonText("Open ACE-Step Repository");
    aceStepRepoButton.onClick = []
    {
        juce::URL(kAceStepGitHubUrl).launchInDefaultBrowser();
    };
    aceStepPage.addAndMakeVisible(aceStepRepoButton);

    int providerIndex = 0;
    for (int index = 0; index < static_cast<int>(aiSettingsProviderOrder().size()); ++index)
    {
        if (aiSettingsProviderOrder()[static_cast<size_t>(index)] == aiClient.getProvider())
        {
            providerIndex = index;
            break;
        }
    }

    providerBox.setSelectedItemIndex(providerIndex, juce::dontSendNotification);
    lastProvider = selectedProvider();
    remoteModelBox.setText(aiClient.getRemoteModel(), juce::dontSendNotification);
    aceStepQualityBox.setSelectedItemIndex(juce::jmax(0,
                                                      juce::StringArray{ "Fast", "Balanced", "High", "Custom" }
                                                          .indexOf(aceStepClientIn.getDefaultQualityPreset())),
                                           juce::dontSendNotification);
    refreshOllamaModels();
    handleProviderChanged();
    applyAceStepQualityPreset(aceStepQualityPreset());
    updateAceStepFieldStates();
    setSize(preferredDialogWidth(), preferredDialogHeight());
}

int TabbedAiSettingsContentComponent::preferredDialogWidth() const noexcept
{
    return 860;
}

int TabbedAiSettingsContentComponent::preferredDialogHeight() const noexcept
{
    return 620;
}

AIClient::Provider TabbedAiSettingsContentComponent::selectedProvider() const
{
    return aiProviderFromDisplayLabel(providerBox.getText());
}

juce::String TabbedAiSettingsContentComponent::remoteModel() const { return remoteModelBox.getText().trim(); }
juce::String TabbedAiSettingsContentComponent::remoteApiKey() const { return remoteApiKeyEditor.getText().trim(); }

juce::String TabbedAiSettingsContentComponent::remoteBaseUrl() const
{
    return customRemoteEndpointVisible()
        ? remoteEndpointEditor.getText().trim()
        : AIClient::defaultRemoteBaseUrlForProvider(selectedProvider());
}

juce::String TabbedAiSettingsContentComponent::ollamaBaseUrl() const { return ollamaBaseUrlEditor.getText().trim(); }
juce::String TabbedAiSettingsContentComponent::ollamaModel() const { return ollamaModelBox.getText().trim(); }
int TabbedAiSettingsContentComponent::timeoutSeconds() const { return timeoutEditor.getText().getIntValue(); }
juce::String TabbedAiSettingsContentComponent::aceStepBaseUrl() const { return aceStepBaseUrlEditor.getText().trim(); }
juce::String TabbedAiSettingsContentComponent::aceStepApiKey() const { return aceStepApiKeyEditor.getText().trim(); }
juce::String TabbedAiSettingsContentComponent::aceStepInstallDirectory() const { return aceStepInstallDirEditor.getText().trim(); }
bool TabbedAiSettingsContentComponent::aceStepAutoStartEnabled() const { return aceStepAutoStartBox.getSelectedItemIndex() != 1; }
juce::String TabbedAiSettingsContentComponent::aceStepDefaultModel() const { return aceStepDefaultModelEditor.getText().trim(); }
juce::String TabbedAiSettingsContentComponent::aceStepAudioFormat() const { return aceStepAudioFormatBox.getText().trim(); }
juce::String TabbedAiSettingsContentComponent::aceStepQualityPreset() const { return aceStepQualityBox.getText().trim(); }
juce::String TabbedAiSettingsContentComponent::aceStepVocalLanguage() const { return aceStepLanguageEditor.getText().trim(); }
bool TabbedAiSettingsContentComponent::aceStepThinkingEnabled() const { return aceStepThinkingBox.getSelectedItemIndex() != 1; }
bool TabbedAiSettingsContentComponent::aceStepUseRandomSeed() const { return aceStepSeedModeBox.getSelectedItemIndex() != 1; }
int TabbedAiSettingsContentComponent::aceStepSeed() const { return aceStepSeedEditor.getText().getIntValue(); }
int TabbedAiSettingsContentComponent::aceStepInferenceSteps() const { return aceStepInferenceStepsEditor.getText().getIntValue(); }
double TabbedAiSettingsContentComponent::aceStepGuidanceScale() const { return aceStepGuidanceScaleEditor.getText().getDoubleValue(); }
juce::String TabbedAiSettingsContentComponent::aceStepInferMethod() const { return aceStepInferMethodBox.getText().trim(); }
int TabbedAiSettingsContentComponent::aceStepStartupTimeoutSeconds() const { return aceStepStartupTimeoutEditor.getText().getIntValue(); }
int TabbedAiSettingsContentComponent::aceStepJobTimeoutSeconds() const { return aceStepJobTimeoutEditor.getText().getIntValue(); }

void TabbedAiSettingsContentComponent::resized()
{
    tabs.setBounds(getLocalBounds());
    const auto compositionVisibleWidth = juce::jmax(compositionViewport.getMaximumVisibleWidth(),
                                                    compositionViewport.getWidth() - compositionViewport.getScrollBarThickness() - 20);
    const auto aceStepVisibleWidth = juce::jmax(aceStepViewport.getMaximumVisibleWidth(),
                                                aceStepViewport.getWidth() - aceStepViewport.getScrollBarThickness() - 20);
    const auto compositionPageWidth = juce::jmax(320, compositionVisibleWidth - 8);
    const auto aceStepPageWidth = juce::jmax(320, aceStepVisibleWidth - 8);
    compositionPage.setSize(compositionPageWidth, computeCompositionContentHeight());
    aceStepPage.setSize(aceStepPageWidth, computeAceStepContentHeight());
    layoutCompositionPage();
    layoutAceStepPage();
}

bool TabbedAiSettingsContentComponent::aceStepModelLooksTurbo(const juce::String& model) noexcept
{
    return model.containsIgnoreCase("turbo");
}

bool TabbedAiSettingsContentComponent::remoteSectionVisible() const noexcept
{
    return selectedProvider() != AIClient::Provider::ollama;
}

bool TabbedAiSettingsContentComponent::ollamaSectionVisible() const noexcept
{
    return selectedProvider() == AIClient::Provider::ollama;
}

bool TabbedAiSettingsContentComponent::customRemoteEndpointVisible() const noexcept
{
    return selectedProvider() == AIClient::Provider::openAICompatible;
}

bool TabbedAiSettingsContentComponent::aceStepCustomQualitySelected() const noexcept
{
    return aceStepQualityPreset().equalsIgnoreCase("Custom");
}

int TabbedAiSettingsContentComponent::aceStepPresetInferenceSteps(const juce::String& preset) const
{
    if (preset.equalsIgnoreCase("Fast"))
        return 4;

    if (preset.equalsIgnoreCase("High"))
    {
        const auto effectiveModel = aceStepDefaultModel().isNotEmpty() ? aceStepDefaultModel() : aceStepClient.getDefaultModel();
        return aceStepModelLooksTurbo(effectiveModel) || effectiveModel.isEmpty() ? 16 : 32;
    }

    return 8;
}

int TabbedAiSettingsContentComponent::computeCompositionContentHeight() const
{
    constexpr int rowHeight = 28;
    constexpr int sectionGap = 10;
    constexpr int rowGap = 6;

    int height = 12 + 24 + sectionGap + rowHeight + rowGap;

    if (remoteSectionVisible())
    {
        height += rowHeight + rowGap;
        height += rowHeight + rowGap;
        if (customRemoteEndpointVisible())
            height += rowHeight + rowGap;
        height += 22 + rowGap;
    }

    if (ollamaSectionVisible())
    {
        height += 24 + sectionGap;
        height += rowHeight + rowGap;
        height += rowHeight + rowGap;
        height += 24 + rowGap;
        height += 22 + rowGap;
        if (!ollamaDetected)
            height += 24 + rowGap;
    }

    height += 24 + sectionGap;
    height += rowHeight + rowGap;
    return height + 12;
}

int TabbedAiSettingsContentComponent::computeAceStepContentHeight() const
{
    constexpr int rowHeight = 28;
    constexpr int sectionGap = 10;
    constexpr int rowGap = 6;

    int height = 12;
    height += 24 + sectionGap;
    height += (rowHeight + rowGap) * 6;
    height += 22 + rowGap;

    height += 24 + sectionGap;
    height += (rowHeight + rowGap) * 5;
    if (!aceStepUseRandomSeed())
        height += rowHeight + rowGap;

    height += 24 + sectionGap;
    height += (rowHeight + rowGap) * 4;
    height += 22 + rowGap;
    height += 24 + rowGap;
    return height + 12;
}

void TabbedAiSettingsContentComponent::layoutCompositionPage()
{
    auto area = compositionPage.getLocalBounds().reduced(16, 12);
    constexpr int labelWidth = 170;
    constexpr int fieldGap = 18;
    constexpr int maxFieldWidth = 520;
    constexpr int rowHeight = 28;
    constexpr int sectionGap = 10;
    constexpr int rowGap = 6;

    const auto layoutSection = [&area, sectionGap] (juce::Label& label)
    {
        label.setBounds(area.removeFromTop(24));
        area.removeFromTop(sectionGap);
    };

    const auto layoutRow = [&area, labelWidth, fieldGap, maxFieldWidth, rowHeight, rowGap] (juce::Label& label,
                                                                                              juce::Component& field)
    {
        auto row = area.removeFromTop(rowHeight);
        label.setBounds(row.removeFromLeft(labelWidth));
        row.removeFromLeft(fieldGap);
        field.setBounds(row.removeFromLeft(juce::jmin(maxFieldWidth, row.getWidth())));
        area.removeFromTop(rowGap);
    };

    const auto layoutStandalone = [&area, rowGap] (juce::Component& component, int height)
    {
        component.setBounds(area.removeFromTop(height));
        area.removeFromTop(rowGap);
    };

    const auto layoutIndentedButton = [&area, labelWidth, fieldGap, rowGap] (juce::Component& button, int width, int height)
    {
        auto row = area.removeFromTop(height);
        row.removeFromLeft(labelWidth + fieldGap);
        button.setBounds(row.removeFromLeft(width));
        area.removeFromTop(rowGap);
    };

    layoutSection(providerSectionLabel);
    layoutRow(providerLabel, providerBox);

    if (remoteSectionVisible())
    {
        layoutRow(remoteModelLabel, remoteModelBox);
        layoutRow(remoteApiKeyLabel, remoteApiKeyEditor);
        if (customRemoteEndpointVisible())
            layoutRow(remoteEndpointLabel, remoteEndpointEditor);
        layoutStandalone(remoteHelpLabel, 22);
    }

    if (ollamaSectionVisible())
    {
        layoutSection(ollamaSectionLabel);
        layoutRow(ollamaBaseUrlLabel, ollamaBaseUrlEditor);
        layoutRow(ollamaModelLabel, ollamaModelBox);
        layoutIndentedButton(refreshOllamaButton, 92, 24);
        layoutStandalone(ollamaStatusLabel, 22);
        if (!ollamaDetected)
            layoutStandalone(ollamaDownloadButton, 24);
    }

    layoutSection(requestSectionLabel);
    layoutRow(timeoutLabel, timeoutEditor);
}

void TabbedAiSettingsContentComponent::layoutAceStepPage()
{
    auto area = aceStepPage.getLocalBounds().reduced(16, 12);
    constexpr int labelWidth = 175;
    constexpr int fieldGap = 18;
    constexpr int maxFieldWidth = 600;
    constexpr int rowHeight = 28;
    constexpr int sectionGap = 10;
    constexpr int rowGap = 6;

    const auto layoutSection = [&area, sectionGap] (juce::Label& label)
    {
        label.setBounds(area.removeFromTop(24));
        area.removeFromTop(sectionGap);
    };

    const auto layoutRow = [&area, labelWidth, fieldGap, maxFieldWidth, rowHeight, rowGap] (juce::Label& label,
                                                                                              juce::Component& field)
    {
        auto row = area.removeFromTop(rowHeight);
        label.setBounds(row.removeFromLeft(labelWidth));
        row.removeFromLeft(fieldGap);
        field.setBounds(row.removeFromLeft(juce::jmin(maxFieldWidth, row.getWidth())));
        area.removeFromTop(rowGap);
    };

    const auto layoutStandalone = [&area, rowGap] (juce::Component& component, int height)
    {
        component.setBounds(area.removeFromTop(height));
        area.removeFromTop(rowGap);
    };

    const auto layoutIndentedButton = [&area, labelWidth, fieldGap, rowGap] (juce::Component& button, int width, int height)
    {
        auto row = area.removeFromTop(height);
        row.removeFromLeft(labelWidth + fieldGap);
        button.setBounds(row.removeFromLeft(width));
        area.removeFromTop(rowGap);
    };

    layoutSection(aceStepConnectionSectionLabel);
    layoutRow(aceStepBaseUrlLabel, aceStepBaseUrlEditor);
    layoutRow(aceStepApiKeyLabel, aceStepApiKeyEditor);
    layoutRow(aceStepInstallDirLabel, aceStepInstallDirEditor);
    layoutRow(aceStepAutoStartLabel, aceStepAutoStartBox);
    layoutRow(aceStepStartupTimeoutLabel, aceStepStartupTimeoutEditor);
    layoutRow(aceStepJobTimeoutLabel, aceStepJobTimeoutEditor);
    layoutStandalone(aceStepHelpLabel, 22);

    layoutSection(aceStepRenderSectionLabel);
    layoutRow(aceStepQualityLabel, aceStepQualityBox);
    layoutRow(aceStepDefaultModelLabel, aceStepDefaultModelEditor);
    layoutRow(aceStepAudioFormatLabel, aceStepAudioFormatBox);
    layoutRow(aceStepLanguageLabel, aceStepLanguageEditor);
    layoutRow(aceStepSeedModeLabel, aceStepSeedModeBox);
    if (!aceStepUseRandomSeed())
        layoutRow(aceStepSeedLabel, aceStepSeedEditor);

    layoutSection(aceStepAdvancedSectionLabel);
    layoutRow(aceStepThinkingLabel, aceStepThinkingBox);
    layoutRow(aceStepInferenceStepsLabel, aceStepInferenceStepsEditor);
    layoutRow(aceStepGuidanceScaleLabel, aceStepGuidanceScaleEditor);
    layoutRow(aceStepInferMethodLabel, aceStepInferMethodBox);
    layoutStandalone(aceStepAdvancedHelpLabel, 22);
    layoutIndentedButton(aceStepRepoButton, 190, 24);
}

void TabbedAiSettingsContentComponent::refreshOllamaModels()
{
    auto preferredModel = ollamaModelBox.getText().trim();
    if (preferredModel.isEmpty())
        preferredModel = aiClient.getOllamaModel().trim();
    refreshOllamaModelCombo(ollamaModelBox, aiClient, ollamaBaseUrlEditor.getText().trim(), preferredModel);
}

void TabbedAiSettingsContentComponent::applyAceStepQualityPreset(const juce::String& preset)
{
    if (preset.equalsIgnoreCase("Custom"))
    {
        updateAceStepFieldStates();
        return;
    }

    const juce::ScopedValueSetter<bool> guard(updatingAceStepQualityControls, true);
    aceStepThinkingBox.setSelectedItemIndex(preset.equalsIgnoreCase("Fast") ? 1 : 0, juce::dontSendNotification);
    aceStepInferenceStepsEditor.setText(juce::String(aceStepPresetInferenceSteps(preset)), juce::dontSendNotification);
    aceStepGuidanceScaleEditor.setText("7.0", juce::dontSendNotification);
    aceStepInferMethodBox.setSelectedItemIndex(0, juce::dontSendNotification);
    updateAceStepFieldStates();
}

void TabbedAiSettingsContentComponent::updateAceStepFieldStates()
{
    const auto customQuality = aceStepCustomQualitySelected();
    for (auto* component : { static_cast<juce::Component*>(&aceStepThinkingLabel),
                             static_cast<juce::Component*>(&aceStepThinkingBox),
                             static_cast<juce::Component*>(&aceStepInferenceStepsLabel),
                             static_cast<juce::Component*>(&aceStepInferenceStepsEditor),
                             static_cast<juce::Component*>(&aceStepGuidanceScaleLabel),
                             static_cast<juce::Component*>(&aceStepGuidanceScaleEditor),
                             static_cast<juce::Component*>(&aceStepInferMethodLabel),
                             static_cast<juce::Component*>(&aceStepInferMethodBox) })
    {
        component->setEnabled(customQuality);
    }

    aceStepSeedLabel.setVisible(!aceStepUseRandomSeed());
    aceStepSeedEditor.setVisible(!aceStepUseRandomSeed());
    aceStepAdvancedHelpLabel.setText(customQuality
                                         ? "Custom mode is active. Guidance scale mainly affects base ACE-Step models."
                                         : "Switch Quality to Custom to edit thinking, inference steps, guidance, and infer method.",
                                     juce::dontSendNotification);
    layoutAceStepPage();
}

void TabbedAiSettingsContentComponent::handleAceStepQualityChanged()
{
    if (updatingAceStepQualityControls)
        return;

    applyAceStepQualityPreset(aceStepQualityPreset());
}

void TabbedAiSettingsContentComponent::handleAceStepModelChanged()
{
    if (updatingAceStepQualityControls)
        return;

    if (!aceStepCustomQualitySelected())
        applyAceStepQualityPreset(aceStepQualityPreset());
}

void TabbedAiSettingsContentComponent::handleProviderChanged()
{
    const auto newProvider = selectedProvider();
    const auto previousDefaultModel = AIClient::defaultRemoteModelForProvider(lastProvider);
    auto currentModel = remoteModelBox.getText().trim();
    if (currentModel.isEmpty() || currentModel == previousDefaultModel)
        currentModel = AIClient::defaultRemoteModelForProvider(newProvider);

    remoteModelLabel.setText(AIClient::providerDisplayName(newProvider) + " model", juce::dontSendNotification);
    remoteApiKeyLabel.setText(AIClient::providerDisplayName(newProvider) + " API key", juce::dontSendNotification);
    remoteHelpLabel.setText(aiProviderHelpText(newProvider), juce::dontSendNotification);
    populateComboBoxChoices(remoteModelBox, aiClient.remoteModelChoices(newProvider), currentModel);

    const auto setVisibility = [] (bool visible, std::initializer_list<juce::Component*> components)
    {
        for (auto* component : components)
            if (component != nullptr)
                component->setVisible(visible);
    };

    setVisibility(remoteSectionVisible(),
                  { &remoteModelLabel, &remoteModelBox, &remoteApiKeyLabel, &remoteApiKeyEditor, &remoteHelpLabel });
    setVisibility(customRemoteEndpointVisible(), { &remoteEndpointLabel, &remoteEndpointEditor });
    setVisibility(ollamaSectionVisible(),
                  { &ollamaSectionLabel, &ollamaBaseUrlLabel, &ollamaBaseUrlEditor, &ollamaModelLabel, &ollamaModelBox,
                    &refreshOllamaButton, &ollamaStatusLabel });
    ollamaDownloadButton.setVisible(ollamaSectionVisible() && !ollamaDetected);

    lastProvider = newProvider;
    layoutCompositionPage();
}

juce::Image loadMutagenLogoBinaryData(bool preferSmallLogo)
{
    const auto loadFromBinary = [] (const void* data, int dataSize)
    {
        juce::MemoryInputStream stream(data, static_cast<size_t>(dataSize), false);
        return juce::ImageFileFormat::loadFrom(stream);
    };

    if (preferSmallLogo)
    {
        auto image = loadFromBinary(BinaryData::mutagenlogo80_png,
                                    BinaryData::mutagenlogo80_pngSize);
        if (image.isValid())
            return image;
    }

    return loadFromBinary(BinaryData::mutagenlogosource_png,
                          BinaryData::mutagenlogosource_pngSize);
}

void classifyRackPluginEntry(VstInstrument& entry)
{
    const auto pluginPath = entry.path.trim();
    if (pluginPath.isEmpty())
    {
        entry.isInstrument = false;
        entry.isEffect = false;
        entry.category = "Unknown";
        entry.hostSupported = false;
        entry.hostError = "Missing plugin path.";
        return;
    }

    const auto pluginFile = juce::File(pluginPath);
    if (!pluginFile.exists())
    {
        entry.isInstrument = false;
        entry.isEffect = false;
        entry.category = "Missing";
        entry.hostSupported = false;
        entry.hostError = "Plugin path does not exist.";
        return;
    }

    juce::AudioPluginFormatManager formatManager;
    formatManager.addFormat(std::make_unique<juce::VST3PluginFormat>());

    juce::OwnedArray<juce::PluginDescription> descriptions;
    for (int index = 0; index < formatManager.getNumFormats(); ++index)
    {
        auto* format = formatManager.getFormat(index);
        if (format == nullptr)
            continue;
        if (!format->fileMightContainThisPluginType(pluginFile.getFullPathName()))
            continue;

        format->findAllTypesForFile(descriptions, pluginFile.getFullPathName());
        if (!descriptions.isEmpty())
            break;
    }

    if (descriptions.isEmpty())
    {
        entry.isInstrument = false;
        entry.isEffect = false;
        entry.category = "Unknown";
        entry.hostSupported = false;
        entry.hostError = "No loadable VST3 plugin description found.";
        return;
    }

    const auto& description = *descriptions[0];
    if (entry.pluginName.trim().isEmpty())
        entry.pluginName = description.name.trim();
    if (entry.name.trim().isEmpty())
        entry.name = entry.pluginName.trim().isNotEmpty()
            ? entry.pluginName
            : pluginFile.getFileNameWithoutExtension();

    entry.isInstrument = description.isInstrument;
    entry.isEffect = !entry.isInstrument;
    entry.category = description.category.trim().isNotEmpty()
        ? description.category.trim()
        : (entry.isInstrument ? "Instrument" : "Effect");
    entry.hostSupported = true;
    entry.hostError.clear();
}

VstInstrument makeRackPluginEntry(const juce::File& pluginFile)
{
    VstInstrument entry;
    entry.name = pluginFile.getFileNameWithoutExtension();
    entry.path = pluginFile.getFullPathName();
    entry.pluginName = entry.name;
    classifyRackPluginEntry(entry);
    return entry;
}

int playbackRefreshRateForComponent(const juce::Component& component)
{
    const auto* display = juce::Desktop::getInstance().getDisplays().getDisplayForRect(component.getScreenBounds());
    const auto hz = display != nullptr && display->verticalFrequencyHz.has_value()
        ? *display->verticalFrequencyHz
        : 60.0;
    return juce::jlimit(60, 240, juce::roundToInt(hz));
}

struct SequenceTickOption
{
    int ticks;
    const char* label;
};

struct KeyQuantizeOption
{
    int id;
    int root;
    juce::String scaleId;
    juce::String label;
};

struct PianoRollPitchOption
{
    int id;
    int pitch;
    juce::String label;
};

struct VirtualPianoKeySpec
{
    int pitch;
    const char* primary;
    std::initializer_list<const char*> aliases;
};

struct ThemeSpec
{
    const char* name;
    juce::LookAndFeel_V4::ColourScheme scheme;
    juce::Colour mainBackground;
    juce::Colour headerStart;
    juce::Colour headerMid;
    juce::Colour headerEnd;
    juce::Colour surface;
    juce::Colour surfaceAlt;
    juce::Colour outline;
    juce::Colour primaryText;
    juce::Colour secondaryText;
    juce::Colour successText;
    juce::Colour infoText;
    juce::Colour warningText;
    juce::Colour editorPlaceholder;
    juce::Colour lcdBackground;
    juce::Colour lcdFrame;
    juce::Colour lcdGlow;
    juce::Colour lcdLabel;
    juce::Colour lcdGhost;
    juce::Colour lcdValue;
    juce::Colour buttonOff;
    juce::Colour buttonOn;
    juce::Colour buttonText;
};

const std::array<ThemeSpec, 8>& availableThemeSpecs()
{
    using Scheme = juce::LookAndFeel_V4::ColourScheme;

    static const std::array<ThemeSpec, 8> themes { {
        { "Mutagen Dark",
          Scheme(juce::Colour::fromRGB(14, 18, 24), juce::Colour::fromRGB(44, 50, 62), juce::Colour::fromRGB(18, 22, 28),
                 juce::Colour::fromRGB(62, 71, 86), juce::Colour::fromRGB(232, 238, 246), juce::Colour::fromRGB(72, 180, 138),
                 juce::Colour::fromRGB(255, 255, 255), juce::Colour::fromRGB(78, 128, 196), juce::Colour::fromRGB(232, 238, 246)),
          juce::Colour::fromRGB(13, 15, 20), juce::Colour::fromRGB(26, 35, 52), juce::Colour::fromRGB(17, 21, 28),
          juce::Colour::fromRGB(12, 14, 18), juce::Colour::fromRGB(20, 22, 28), juce::Colour::fromRGB(48, 54, 66),
          juce::Colour::fromRGB(56, 64, 79), juce::Colour::fromRGB(235, 239, 244), juce::Colour::fromRGB(190, 199, 210),
          juce::Colour::fromRGB(143, 225, 170), juce::Colour::fromRGB(156, 199, 239), juce::Colour::fromRGB(230, 190, 118),
          juce::Colour::fromRGB(122, 133, 149), juce::Colour::fromRGB(14, 22, 18), juce::Colour::fromRGB(58, 94, 72),
          juce::Colour::fromRGBA(120, 255, 174, 22), juce::Colour::fromRGB(135, 214, 164), juce::Colour::fromRGBA(106, 170, 121, 52),
          juce::Colour::fromRGB(175, 255, 196), juce::Colour::fromRGB(48, 54, 66), juce::Colour::fromRGB(72, 104, 160),
          juce::Colours::white },
        { "Graphite",
          Scheme(juce::Colour::fromRGB(20, 21, 25), juce::Colour::fromRGB(52, 56, 64), juce::Colour::fromRGB(22, 24, 28),
                 juce::Colour::fromRGB(83, 89, 99), juce::Colour::fromRGB(236, 239, 243), juce::Colour::fromRGB(154, 160, 171),
                 juce::Colour::fromRGB(255, 255, 255), juce::Colour::fromRGB(120, 150, 196), juce::Colour::fromRGB(236, 239, 243)),
          juce::Colour::fromRGB(15, 16, 19), juce::Colour::fromRGB(38, 40, 46), juce::Colour::fromRGB(28, 30, 36),
          juce::Colour::fromRGB(16, 17, 21), juce::Colour::fromRGB(22, 24, 29), juce::Colour::fromRGB(54, 58, 66),
          juce::Colour::fromRGB(74, 80, 92), juce::Colour::fromRGB(236, 239, 243), juce::Colour::fromRGB(188, 194, 203),
          juce::Colour::fromRGB(152, 214, 182), juce::Colour::fromRGB(165, 196, 232), juce::Colour::fromRGB(230, 196, 128),
          juce::Colour::fromRGB(128, 134, 146), juce::Colour::fromRGB(24, 26, 30), juce::Colour::fromRGB(78, 84, 92),
          juce::Colour::fromRGBA(192, 212, 230, 20), juce::Colour::fromRGB(180, 188, 196), juce::Colour::fromRGBA(154, 160, 171, 52),
          juce::Colour::fromRGB(235, 239, 243), juce::Colour::fromRGB(54, 58, 66), juce::Colour::fromRGB(94, 112, 154),
          juce::Colour::fromRGB(244, 246, 250) },
        { "Midnight Blue",
          Scheme(juce::Colour::fromRGB(10, 16, 28), juce::Colour::fromRGB(36, 50, 74), juce::Colour::fromRGB(12, 20, 34),
                 juce::Colour::fromRGB(66, 92, 126), juce::Colour::fromRGB(228, 238, 250), juce::Colour::fromRGB(78, 200, 224),
                 juce::Colour::fromRGB(255, 255, 255), juce::Colour::fromRGB(72, 136, 232), juce::Colour::fromRGB(228, 238, 250)),
          juce::Colour::fromRGB(8, 12, 20), juce::Colour::fromRGB(16, 32, 58), juce::Colour::fromRGB(12, 24, 44),
          juce::Colour::fromRGB(8, 12, 22), juce::Colour::fromRGB(12, 22, 38), juce::Colour::fromRGB(40, 56, 82),
          juce::Colour::fromRGB(62, 86, 118), juce::Colour::fromRGB(230, 239, 250), juce::Colour::fromRGB(176, 196, 222),
          juce::Colour::fromRGB(128, 224, 182), juce::Colour::fromRGB(120, 196, 255), juce::Colour::fromRGB(232, 196, 126),
          juce::Colour::fromRGB(122, 142, 170), juce::Colour::fromRGB(12, 24, 30), juce::Colour::fromRGB(42, 88, 118),
          juce::Colour::fromRGBA(126, 224, 255, 18), juce::Colour::fromRGB(146, 224, 234), juce::Colour::fromRGBA(118, 170, 214, 52),
          juce::Colour::fromRGB(188, 250, 255), juce::Colour::fromRGB(40, 56, 82), juce::Colour::fromRGB(58, 118, 204),
          juce::Colour::fromRGB(240, 247, 255) },
        { "Oxide Amber",
          Scheme(juce::Colour::fromRGB(28, 18, 12), juce::Colour::fromRGB(70, 46, 30), juce::Colour::fromRGB(24, 16, 12),
                 juce::Colour::fromRGB(112, 78, 54), juce::Colour::fromRGB(248, 235, 216), juce::Colour::fromRGB(236, 160, 76),
                 juce::Colour::fromRGB(32, 22, 14), juce::Colour::fromRGB(212, 118, 54), juce::Colour::fromRGB(248, 235, 216)),
          juce::Colour::fromRGB(18, 12, 9), juce::Colour::fromRGB(54, 30, 16), juce::Colour::fromRGB(35, 22, 13),
          juce::Colour::fromRGB(16, 10, 8), juce::Colour::fromRGB(26, 18, 13), juce::Colour::fromRGB(72, 48, 31),
          juce::Colour::fromRGB(94, 66, 44), juce::Colour::fromRGB(246, 236, 222), juce::Colour::fromRGB(214, 192, 170),
          juce::Colour::fromRGB(188, 224, 154), juce::Colour::fromRGB(244, 188, 118), juce::Colour::fromRGB(255, 208, 132),
          juce::Colour::fromRGB(162, 138, 118), juce::Colour::fromRGB(22, 18, 12), juce::Colour::fromRGB(112, 74, 32),
          juce::Colour::fromRGBA(255, 196, 110, 18), juce::Colour::fromRGB(255, 206, 132), juce::Colour::fromRGBA(190, 132, 76, 54),
          juce::Colour::fromRGB(255, 224, 176), juce::Colour::fromRGB(72, 48, 31), juce::Colour::fromRGB(132, 86, 42),
          juce::Colour::fromRGB(255, 242, 226) },
        { "Volt Green",
          Scheme(juce::Colour::fromRGB(10, 16, 10), juce::Colour::fromRGB(28, 52, 30), juce::Colour::fromRGB(10, 18, 12),
                 juce::Colour::fromRGB(60, 98, 64), juce::Colour::fromRGB(232, 246, 232), juce::Colour::fromRGB(126, 232, 94),
                 juce::Colour::fromRGB(10, 18, 12), juce::Colour::fromRGB(82, 182, 110), juce::Colour::fromRGB(232, 246, 232)),
          juce::Colour::fromRGB(8, 12, 8), juce::Colour::fromRGB(16, 40, 18), juce::Colour::fromRGB(12, 28, 14),
          juce::Colour::fromRGB(8, 12, 8), juce::Colour::fromRGB(12, 20, 12), juce::Colour::fromRGB(28, 56, 30),
          juce::Colour::fromRGB(48, 84, 52), juce::Colour::fromRGB(232, 246, 232), juce::Colour::fromRGB(176, 204, 176),
          juce::Colour::fromRGB(154, 236, 132), juce::Colour::fromRGB(146, 224, 178), juce::Colour::fromRGB(214, 214, 126),
          juce::Colour::fromRGB(120, 146, 122), juce::Colour::fromRGB(10, 24, 12), juce::Colour::fromRGB(42, 96, 46),
          juce::Colour::fromRGBA(118, 255, 108, 18), juce::Colour::fromRGB(168, 246, 160), juce::Colour::fromRGBA(108, 180, 102, 52),
          juce::Colour::fromRGB(198, 255, 190), juce::Colour::fromRGB(28, 56, 30), juce::Colour::fromRGB(54, 118, 62),
          juce::Colour::fromRGB(242, 250, 242) },
        { "Crimson Night",
          Scheme(juce::Colour::fromRGB(24, 12, 18), juce::Colour::fromRGB(64, 34, 50), juce::Colour::fromRGB(22, 12, 18),
                 juce::Colour::fromRGB(104, 58, 84), juce::Colour::fromRGB(246, 232, 238), juce::Colour::fromRGB(226, 92, 136),
                 juce::Colour::fromRGB(255, 255, 255), juce::Colour::fromRGB(192, 84, 148), juce::Colour::fromRGB(246, 232, 238)),
          juce::Colour::fromRGB(14, 10, 16), juce::Colour::fromRGB(44, 18, 34), juce::Colour::fromRGB(30, 14, 24),
          juce::Colour::fromRGB(12, 8, 14), juce::Colour::fromRGB(22, 14, 22), juce::Colour::fromRGB(62, 36, 52),
          juce::Colour::fromRGB(88, 56, 74), juce::Colour::fromRGB(246, 232, 238), juce::Colour::fromRGB(214, 182, 196),
          juce::Colour::fromRGB(164, 236, 186), juce::Colour::fromRGB(232, 158, 188), juce::Colour::fromRGB(238, 186, 122),
          juce::Colour::fromRGB(158, 132, 144), juce::Colour::fromRGB(20, 12, 18), juce::Colour::fromRGB(96, 44, 68),
          juce::Colour::fromRGBA(255, 134, 156, 18), juce::Colour::fromRGB(255, 164, 178), juce::Colour::fromRGBA(188, 88, 118, 52),
          juce::Colour::fromRGB(255, 192, 204), juce::Colour::fromRGB(62, 36, 52), juce::Colour::fromRGB(118, 52, 86),
          juce::Colour::fromRGB(250, 240, 244) },
        { "Ultraviolet",
          Scheme(juce::Colour::fromRGB(14, 10, 24), juce::Colour::fromRGB(42, 30, 72), juce::Colour::fromRGB(16, 12, 28),
                 juce::Colour::fromRGB(74, 60, 118), juce::Colour::fromRGB(236, 232, 250), juce::Colour::fromRGB(154, 112, 244),
                 juce::Colour::fromRGB(255, 255, 255), juce::Colour::fromRGB(112, 108, 226), juce::Colour::fromRGB(236, 232, 250)),
          juce::Colour::fromRGB(10, 8, 18), juce::Colour::fromRGB(24, 18, 52), juce::Colour::fromRGB(18, 14, 34),
          juce::Colour::fromRGB(10, 8, 18), juce::Colour::fromRGB(18, 14, 28), juce::Colour::fromRGB(44, 34, 74),
          juce::Colour::fromRGB(72, 60, 108), juce::Colour::fromRGB(238, 234, 250), juce::Colour::fromRGB(194, 186, 220),
          juce::Colour::fromRGB(172, 232, 188), juce::Colour::fromRGB(188, 172, 255), juce::Colour::fromRGB(230, 196, 126),
          juce::Colour::fromRGB(138, 132, 162), juce::Colour::fromRGB(16, 12, 28), juce::Colour::fromRGB(64, 52, 118),
          juce::Colour::fromRGBA(182, 148, 255, 18), juce::Colour::fromRGB(200, 188, 255), juce::Colour::fromRGBA(142, 118, 214, 52),
          juce::Colour::fromRGB(220, 212, 255), juce::Colour::fromRGB(44, 34, 74), juce::Colour::fromRGB(80, 68, 150),
          juce::Colour::fromRGB(244, 240, 255) },
        { "Slate Ice",
          Scheme(juce::Colour::fromRGB(18, 24, 28), juce::Colour::fromRGB(48, 60, 68), juce::Colour::fromRGB(18, 26, 30),
                 juce::Colour::fromRGB(86, 100, 110), juce::Colour::fromRGB(236, 242, 245), juce::Colour::fromRGB(118, 198, 214),
                 juce::Colour::fromRGB(18, 24, 28), juce::Colour::fromRGB(112, 164, 198), juce::Colour::fromRGB(236, 242, 245)),
          juce::Colour::fromRGB(12, 16, 20), juce::Colour::fromRGB(24, 34, 42), juce::Colour::fromRGB(18, 25, 31),
          juce::Colour::fromRGB(12, 16, 20), juce::Colour::fromRGB(16, 22, 28), juce::Colour::fromRGB(48, 60, 68),
          juce::Colour::fromRGB(74, 86, 94), juce::Colour::fromRGB(236, 242, 245), juce::Colour::fromRGB(194, 204, 208),
          juce::Colour::fromRGB(168, 228, 194), juce::Colour::fromRGB(166, 214, 232), juce::Colour::fromRGB(226, 198, 132),
          juce::Colour::fromRGB(132, 144, 152), juce::Colour::fromRGB(16, 22, 26), juce::Colour::fromRGB(52, 86, 96),
          juce::Colour::fromRGBA(158, 228, 236, 18), juce::Colour::fromRGB(184, 230, 236), juce::Colour::fromRGBA(132, 170, 178, 52),
          juce::Colour::fromRGB(214, 248, 250), juce::Colour::fromRGB(48, 60, 68), juce::Colour::fromRGB(78, 120, 148),
          juce::Colour::fromRGB(244, 248, 250) }
    } };

    return themes;
}

const ThemeSpec& themeSpecForIndex(int index)
{
    const auto& themes = availableThemeSpecs();
    return themes[static_cast<size_t>(juce::jlimit(0, static_cast<int>(themes.size()) - 1, index))];
}

void applyThemeToComponentTree(juce::Component& component, const ThemeSpec& theme)
{
    if (auto* button = dynamic_cast<juce::TextButton*>(&component))
    {
        button->setColour(juce::TextButton::buttonColourId, theme.buttonOff);
        button->setColour(juce::TextButton::buttonOnColourId, theme.buttonOn);
        button->setColour(juce::TextButton::textColourOffId, theme.buttonText);
        button->setColour(juce::TextButton::textColourOnId, theme.buttonText);
    }
    else if (auto* toggle = dynamic_cast<juce::ToggleButton*>(&component))
    {
        toggle->setColour(juce::ToggleButton::textColourId, theme.primaryText);
        toggle->setColour(juce::ToggleButton::tickColourId, theme.buttonOn);
        toggle->setColour(juce::ToggleButton::tickDisabledColourId, theme.outline);
    }
    else if (auto* editor = dynamic_cast<juce::TextEditor*>(&component))
    {
        editor->setColour(juce::TextEditor::backgroundColourId, theme.surface);
        editor->setColour(juce::TextEditor::textColourId, theme.primaryText);
        editor->setColour(juce::TextEditor::outlineColourId, theme.outline);
        editor->setColour(juce::TextEditor::focusedOutlineColourId, theme.buttonOn);
        editor->setColour(juce::CaretComponent::caretColourId, theme.primaryText);
    }
    else if (auto* combo = dynamic_cast<juce::ComboBox*>(&component))
    {
        combo->setColour(juce::ComboBox::backgroundColourId, theme.surface);
        combo->setColour(juce::ComboBox::textColourId, theme.primaryText);
        combo->setColour(juce::ComboBox::outlineColourId, theme.outline);
        combo->setColour(juce::ComboBox::buttonColourId, theme.surfaceAlt);
        combo->setColour(juce::ComboBox::arrowColourId, theme.secondaryText);
    }
    else if (auto* slider = dynamic_cast<juce::Slider*>(&component))
    {
        slider->setColour(juce::Slider::backgroundColourId, theme.surface);
        slider->setColour(juce::Slider::trackColourId, theme.buttonOn);
        slider->setColour(juce::Slider::thumbColourId, theme.lcdValue);
        slider->setColour(juce::Slider::textBoxBackgroundColourId, theme.surface);
        slider->setColour(juce::Slider::textBoxTextColourId, theme.primaryText);
        slider->setColour(juce::Slider::textBoxOutlineColourId, theme.outline);
    }
    else if (auto* list = dynamic_cast<juce::ListBox*>(&component))
    {
        list->setColour(juce::ListBox::backgroundColourId, theme.surface);
        list->setColour(juce::ListBox::outlineColourId, theme.outline);
        list->setColour(juce::ListBox::textColourId, theme.primaryText);
    }
    else if (auto* header = dynamic_cast<juce::TableHeaderComponent*>(&component))
    {
        header->setColour(juce::TableHeaderComponent::backgroundColourId, theme.surfaceAlt);
        header->setColour(juce::TableHeaderComponent::outlineColourId, theme.outline);
        header->setColour(juce::TableHeaderComponent::textColourId, theme.primaryText);
    }
    else if (auto* label = dynamic_cast<juce::Label*>(&component))
    {
        label->setColour(juce::Label::textColourId, theme.primaryText);
    }

    for (auto* child : component.getChildren())
        applyThemeToComponentTree(*child, theme);
}

class CompactHeaderLookAndFeel final : public juce::LookAndFeel_V4
{
public:
    juce::Font getTextButtonFont(juce::TextButton&, int buttonHeight) override
    {
        return juce::FontOptions(juce::jlimit(ui::scaleValue(ui::kTinyTextSize),
                                              ui::scaleValue(ui::kStrongTextSize),
                                              static_cast<float>(buttonHeight) * 0.38f),
                                 juce::Font::bold);
    }

    juce::Font getComboBoxFont(juce::ComboBox&) override
    {
        return ui::font();
    }

    juce::Font getPopupMenuFont() override
    {
        return ui::font();
    }

    juce::Label* createSliderTextBox(juce::Slider& slider) override
    {
        auto* label = juce::LookAndFeel_V4::createSliderTextBox(slider);
        label->setFont(ui::tinyFont());
        label->setJustificationType(juce::Justification::centred);
        label->setMinimumHorizontalScale(1.0f);
        return label;
    }
};

std::vector<SequenceTickOption> sequenceTickOptionsForProject(const ProjectState& project)
{
    const auto barTicks = ticksPerBar(project);
    return {
        { kTicksPerBeat / 16, "1/64 note" },
        { kTicksPerBeat / 8, "1/32 note" },
        { kTicksPerBeat / 4, "1/16 note" },
        { kTicksPerBeat / 2, "1/8 note" },
        { kTicksPerBeat, "1 beat" },
        { kTicksPerBeat * 2, "1/2 note" },
        { barTicks, "1 bar" },
        { barTicks * 2, "2 bars" },
        { barTicks * 4, "4 bars" },
        { barTicks * 8, "8 bars" }
    };
}

juce::String sequenceTickLabel(int ticks, const ProjectState& project)
{
    const auto projectBarTicks = ticksPerBar(project);
    const auto normalisedTicks = juce::jmax(kMinSequenceSnapTicks, ticks);
    for (const auto& option : sequenceTickOptionsForProject(project))
    {
        if (option.ticks == normalisedTicks)
            return option.label;
    }

    if (projectBarTicks > 0 && (normalisedTicks % projectBarTicks) == 0)
    {
        const auto bars = normalisedTicks / projectBarTicks;
        return juce::String(bars) + (bars == 1 ? " bar" : " bars");
    }

    if (normalisedTicks % kTicksPerBeat == 0)
    {
        const auto beats = normalisedTicks / kTicksPerBeat;
        return juce::String(beats) + (beats == 1 ? " beat" : " beats");
    }

    return juce::String(normalisedTicks) + " ticks";
}

juce::StringArray buildUiFontChoices()
{
    juce::StringArray choices;
    choices.add("Default System");

    auto installedFonts = juce::Font::findAllTypefaceNames();
    installedFonts.sort(true);

    const auto addInstalledFont = [&choices, &installedFonts] (const juce::String& requestedName)
    {
        for (const auto& installedFont : installedFonts)
        {
            if (!installedFont.equalsIgnoreCase(requestedName))
                continue;

            choices.addIfNotAlreadyThere(installedFont);
            return;
        }

        for (const auto& installedFont : installedFonts)
        {
            if (!installedFont.startsWithIgnoreCase(requestedName)
                && !installedFont.containsIgnoreCase(requestedName))
                continue;

            choices.addIfNotAlreadyThere(installedFont);
            return;
        }
    };

    // Prefer compact, broadcast-style sans fonts that fit a DAW UI.
    constexpr std::array<const char*, 20> preferredFonts
    {
        "Bahnschrift SemiCondensed",
        "Bahnschrift Condensed",
        "Bahnschrift SemiBold SemiConden",
        "Bahnschrift",
        "Agency FB",
        "Franklin Gothic Medium",
        "Franklin Gothic Demi",
        "Gill Sans MT Condensed",
        "Gill Sans MT",
        "Arial Narrow",
        "Segoe UI",
        "Trebuchet MS",
        "Verdana",
        "Tahoma",
        "Century Gothic",
        "Calibri",
        "Corbel",
        "Candara",
        "Consolas",
        "Cascadia Mono"
    };

    for (const auto* preferredFont : preferredFonts)
        addInstalledFont(preferredFont);

    const auto looksLikeDawFont = [] (const juce::String& fontName)
    {
        constexpr std::array<const char*, 16> allowedTokens
        {
            "bahnschrift",
            "agency",
            "gothic",
            "segoe",
            "arial",
            "verdana",
            "tahoma",
            "trebuchet",
            "calibri",
            "corbel",
            "candara",
            "consolas",
            "cascadia",
            "dubai",
            "dejavu sans",
            "sans"
        };

        constexpr std::array<const char*, 13> rejectedTokens
        {
            "script",
            "hand",
            "blackadder",
            "brush",
            "comic",
            "curlz",
            "chiller",
            "broadway",
            "gigi",
            "harrington",
            "jokerman",
            "symbol",
            "wingding"
        };

        for (const auto* rejectedToken : rejectedTokens)
        {
            if (fontName.containsIgnoreCase(rejectedToken))
                return false;
        }

        for (const auto* allowedToken : allowedTokens)
        {
            if (fontName.containsIgnoreCase(allowedToken))
                return true;
        }

        return false;
    };

    for (const auto& installedFont : installedFonts)
    {
        if (choices.size() >= 15)
            break;

        if (looksLikeDawFont(installedFont))
            choices.addIfNotAlreadyThere(installedFont);
    }

    return choices;
}

struct UiFontSizeOption
{
    const char* label = "100%";
    float scale = 1.0f;
};

const std::array<UiFontSizeOption, 6>& uiFontSizeOptions()
{
    static constexpr std::array<UiFontSizeOption, 6> options
    {{
        { "80%", 0.80f },
        { "90%", 0.90f },
        { "100%", 1.00f },
        { "110%", 1.10f },
        { "125%", 1.25f },
        { "140%", 1.40f }
    }};
    return options;
}

int resolveUiFontSizeIndex(float scale)
{
    const auto& options = uiFontSizeOptions();
    int bestIndex = 0;
    float bestDistance = std::numeric_limits<float>::max();

    for (int index = 0; index < static_cast<int>(options.size()); ++index)
    {
        const auto distance = std::abs(options[static_cast<size_t>(index)].scale - scale);
        if (distance >= bestDistance)
            continue;

        bestDistance = distance;
        bestIndex = index;
    }

    return bestIndex;
}

juce::Font remapExplicitUiFont(const juce::Font& currentFont, float scaleRatio)
{
    juce::Font remapped(juce::FontOptions(juce::jmax(1.0f, currentFont.getHeight() * scaleRatio)));
    remapped.setStyleFlags(currentFont.getStyleFlags());
    return remapped;
}

void refreshExplicitUiFontsInComponentTree(juce::Component& component, float scaleRatio)
{
    if (auto* label = dynamic_cast<juce::Label*>(&component))
        label->setFont(remapExplicitUiFont(label->getFont(), scaleRatio));
    else if (auto* editor = dynamic_cast<juce::TextEditor*>(&component))
        editor->setFont(remapExplicitUiFont(editor->getFont(), scaleRatio));

    for (auto* child : component.getChildren())
        refreshExplicitUiFontsInComponentTree(*child, scaleRatio);
}

void populateSequenceTickBox(juce::ComboBox& box, const ProjectState& project)
{
    box.clear(juce::dontSendNotification);
    for (const auto& option : sequenceTickOptionsForProject(project))
        box.addItem(option.label, option.ticks);
}

const std::vector<KeyQuantizeOption>& keyQuantizeOptions()
{
    static const std::vector<KeyQuantizeOption> options = []
    {
        std::vector<KeyQuantizeOption> values;
        values.push_back({ 1, 0, "chromatic", "All Notes (Chromatic)" });

        int nextId = 2;
        const auto scales = availableKeyQuantizeScales();
        for (int root = 0; root < 12; ++root)
        {
            for (const auto& scaleId : scales)
            {
                if (scaleId.equalsIgnoreCase("chromatic"))
                    continue;

                values.push_back({ nextId++, root, scaleId, keyQuantizeDisplayName(root, scaleId) });
            }
        }

        return values;
    }();

    return options;
}

int keyQuantizeOptionId(const ProjectState& project)
{
    const auto scaleId = normaliseKeyQuantizeScale(project.keyQuantizeScale);
    const auto root = juce::negativeAwareModulo(project.keyQuantizeRoot, 12);
    for (const auto& option : keyQuantizeOptions())
    {
        if (option.root == root && option.scaleId.equalsIgnoreCase(scaleId))
            return option.id;
    }

    return 1;
}

const KeyQuantizeOption* findKeyQuantizeOptionById(int id)
{
    for (const auto& option : keyQuantizeOptions())
    {
        if (option.id == id)
            return &option;
    }

    return nullptr;
}

juce::String pianoRollPitchOptionLabel(int pitch)
{
    static const char* names[] = { "C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B" };
    const auto clampedPitch = juce::jlimit(kEditableMidiPitchMin, kEditableMidiPitchMax, pitch);
    return juce::String(names[clampedPitch % 12]) + juce::String((clampedPitch / 12) - 1);
}

const std::vector<PianoRollPitchOption>& pianoRollPitchOptions()
{
    static const std::vector<PianoRollPitchOption> options = []
    {
        std::vector<PianoRollPitchOption> values;
        values.reserve(static_cast<size_t>((kEditableMidiPitchMax - kEditableMidiPitchMin) + 1));
        for (int pitch = kEditableMidiPitchMin; pitch <= kEditableMidiPitchMax; ++pitch)
            values.push_back({ pitch + 1, pitch, pianoRollPitchOptionLabel(pitch) });
        return values;
    }();

    return options;
}

int pianoRollPitchOptionId(int pitch)
{
    return juce::jlimit(kEditableMidiPitchMin, kEditableMidiPitchMax, pitch) + 1;
}

const PianoRollPitchOption* findPianoRollPitchOptionById(int id)
{
    for (const auto& option : pianoRollPitchOptions())
    {
        if (option.id == id)
            return &option;
    }

    return nullptr;
}

const std::array<VirtualPianoKeySpec, 24>& virtualPianoKeySpecs()
{
    static const std::array<VirtualPianoKeySpec, 24> specs { {
        { 60, "Z", {} },
        { 61, "S", {} },
        { 62, "X", {} },
        { 63, "D", {} },
        { 64, "C", {} },
        { 65, "V", {} },
        { 66, "G", {} },
        { 67, "B", {} },
        { 68, "H", {} },
        { 69, "N", {} },
        { 70, "J", {} },
        { 71, "M", {} },
        { 72, "Q", { "," } },
        { 73, "2", { "L" } },
        { 74, "W", { "." } },
        { 75, "3", { ";" } },
        { 76, "E", { "/" } },
        { 77, "R", {} },
        { 78, "5", {} },
        { 79, "T", {} },
        { 80, "6", {} },
        { 81, "Y", {} },
        { 82, "7", {} },
        { 83, "U", {} }
    } };

    return specs;
}

juce::String virtualPianoHintText()
{
    return "Click the keys or play from the computer keyboard. "
           "Lower octave: Z S X D C V G B H N J M. "
           "Upper octave: Q 2 W 3 E R 5 T 6 Y 7 U. "
           "Aliases: , L . ; /.";
}

juce::String virtualPianoShortcutLabel(const VirtualPianoKeySpec& spec)
{
    juce::StringArray labels;
    labels.add(spec.primary);
    for (const auto* alias : spec.aliases)
        labels.add(alias);
    return labels.joinIntoString(" / ");
}

juce::String formatLcdSeconds(double seconds)
{
    auto clamped = juce::jmax(0.0, seconds);
    const auto wholeSeconds = static_cast<int>(std::floor(clamped));
    const auto centiseconds = juce::jlimit(0, 99, juce::roundToInt((clamped - static_cast<double>(wholeSeconds)) * 100.0));
    const auto secondsPart = wholeSeconds % 60;
    const auto minutesPart = (wholeSeconds / 60) % 60;
    const auto hoursPart = wholeSeconds / 3600;

    if (hoursPart > 0)
        return juce::String(hoursPart) + ":" + juce::String(minutesPart).paddedLeft('0', 2)
            + ":" + juce::String(secondsPart).paddedLeft('0', 2)
            + "." + juce::String(centiseconds).paddedLeft('0', 2);

    return juce::String(wholeSeconds / 60).paddedLeft('0', 2)
        + ":" + juce::String(secondsPart).paddedLeft('0', 2)
        + "." + juce::String(centiseconds).paddedLeft('0', 2);
}

juce::String lcdGhostForValue(const juce::String& value)
{
    juce::String ghost;
    ghost.preallocateBytes(value.getNumBytesAsUTF8() + 1);
    for (const auto character : value)
    {
        if (juce::CharacterFunctions::isDigit(character))
            ghost << '8';
        else
            ghost << character;
    }
    return ghost;
}

double projectSequenceLengthSeconds(const ProjectState& project)
{
    double endSeconds = juce::jmax(project.rightLocatorSec, project.playheadSec);

    for (const auto& section : project.midiSections)
        endSeconds = juce::jmax(endSeconds, tickToSeconds(project, section.startTick + juce::jmax(kMinSequenceSnapTicks, section.lengthTicks)));

    for (const auto& clip : project.sampleClips)
        endSeconds = juce::jmax(endSeconds, juce::jmax(0.0, clip.startSec + juce::jmax(0.0, clip.durationSec)));

    return endSeconds;
}

int projectSequenceLengthTicks(const ProjectState& project)
{
    auto endTick = juce::jmax(ticksPerBar(project), juce::jmax(project.rightLocatorTick, project.playheadTick));

    for (const auto& section : project.midiSections)
        endTick = juce::jmax(endTick, section.startTick + juce::jmax(kMinSequenceSnapTicks, section.lengthTicks));

    for (const auto& clip : project.sampleClips)
        endTick = juce::jmax(endTick,
                             secondsToTick(project,
                                           juce::jmax(0.0, clip.startSec + juce::jmax(0.0, clip.durationSec))));

    return endTick;
}

juce::String noteNameLabel(int pitch)
{
    static constexpr const char* kNames[] = { "C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B" };
    return juce::String(kNames[pitch % 12]) + juce::String((pitch / 12) - 1);
}

bool isVirtualPianoBlackKey(int pitch)
{
    switch (pitch % 12)
    {
        case 1:
        case 3:
        case 6:
        case 8:
        case 10:
            return true;
        default:
            return false;
    }
}

juce::String normaliseVirtualPianoShortcutKey(const juce::KeyPress& key)
{
    const auto character = key.getTextCharacter();
    if (character == 0)
        return {};

    juce::String text(juce::String::charToString(character).trim());
    if (text.isEmpty())
        return {};

    if (text.length() == 1)
    {
        const auto upper = juce::CharacterFunctions::toUpperCase(text[0]);
        return juce::String::charToString(upper);
    }

    return text.toUpperCase();
}

enum class AppProfileSection
{
    timerCallback,
    refreshFloatingWindows,
    count
};

constexpr auto kAppProfileSectionCount = static_cast<size_t>(AppProfileSection::count);

const char* appProfileSectionName(AppProfileSection section)
{
    switch (section)
    {
        case AppProfileSection::timerCallback: return "timer_callback";
        case AppProfileSection::refreshFloatingWindows: return "refresh_floating_windows";
        case AppProfileSection::count: break;
    }

    return "unknown";
}

int64_t appProfileNowMicroseconds() noexcept
{
    return std::chrono::duration_cast<std::chrono::microseconds>(
        std::chrono::steady_clock::now().time_since_epoch()
    ).count();
}

struct AppProfileCounter
{
    std::atomic<int64_t> count{0};
    std::atomic<int64_t> totalMicros{0};
    std::atomic<int64_t> maxMicros{0};

    void addSample(int64_t durationMicros) noexcept
    {
        count.fetch_add(1, std::memory_order_relaxed);
        totalMicros.fetch_add(durationMicros, std::memory_order_relaxed);

        auto currentMax = maxMicros.load(std::memory_order_relaxed);
        while (durationMicros > currentMax
               && !maxMicros.compare_exchange_weak(currentMax, durationMicros, std::memory_order_relaxed))
        {
        }
    }
};

class AppRuntimeProfiler final
{
public:
    AppRuntimeProfiler()
    {
        const auto logPath = juce::SystemStats::getEnvironmentVariable("AIMS_NATIVE_PROFILE_LOG", {}).trim();
        enabled = logPath.isNotEmpty();
        if (enabled)
            logFile = juce::File(logPath);
    }

    bool isEnabled() const noexcept
    {
        return enabled;
    }

    void addSample(AppProfileSection section, int64_t durationMicros) noexcept
    {
        if (!enabled)
            return;

        counters[static_cast<size_t>(section)].addSample(durationMicros);
    }

    void dump(const juce::String& sectionTitle)
    {
        if (!enabled)
            return;

        bool expected = false;
        if (!dumped.compare_exchange_strong(expected, true, std::memory_order_relaxed))
            return;

        if (logFile == juce::File())
            return;

        logFile.getParentDirectory().createDirectory();
        juce::FileOutputStream output(logFile);
        if (!output.openedOk())
            return;

        output.setPosition(output.getFile().getSize());

        juce::StringArray lines;
        lines.add("[" + sectionTitle + "] time=" + juce::Time::getCurrentTime().toString(true, true));
        for (size_t index = 0; index < counters.size(); ++index)
        {
            const auto count = counters[index].count.load(std::memory_order_relaxed);
            if (count <= 0)
                continue;

            const auto totalMicros = counters[index].totalMicros.load(std::memory_order_relaxed);
            const auto maxMicros = counters[index].maxMicros.load(std::memory_order_relaxed);
            const auto averageMicros = static_cast<double>(totalMicros) / static_cast<double>(count);
            lines.add(juce::String(appProfileSectionName(static_cast<AppProfileSection>(index)))
                      + ": count=" + juce::String(count)
                      + " avg_us=" + juce::String(averageMicros, 2)
                      + " max_us=" + juce::String(maxMicros)
                      + " total_ms=" + juce::String(static_cast<double>(totalMicros) / 1000.0, 2));
        }
        lines.add({});

        output.writeText(lines.joinIntoString("\n"), false, false, nullptr);
    }

private:
    bool enabled = false;
    std::atomic<bool> dumped{false};
    juce::File logFile;
    std::array<AppProfileCounter, kAppProfileSectionCount> counters;
};

AppRuntimeProfiler& appRuntimeProfiler()
{
    static AppRuntimeProfiler profiler;
    return profiler;
}

constexpr juce::uint32 kPlaybackRackEditorSyncIntervalMs = 96;
constexpr juce::uint32 kDeferredEngineParameterFlushIntervalMs = 24;
constexpr int kPlaybackRefreshRateWithOpenEditorHz = 30;
constexpr int kIdleRefreshRateWithOpenEditorHz = 2;
constexpr int kPlaybackRefreshRateWithOpenPianoRollHz = 45;
constexpr int kPlaybackHeavyUiRefreshDivisorWithOpenEditor = 2;
constexpr int kPlaybackHeavyUiRefreshDivisorWithOpenPianoRoll = 2;
constexpr int kPlaybackFloatingRefreshDivisorWithOpenEditor = 6;
constexpr int kPlaybackFloatingRefreshDivisorWithOpenPianoRoll = 12;
constexpr int kPlaybackFloatingPianoRollRefreshDivisor = 2;

class ScopedAppProfileSample final
{
public:
    explicit ScopedAppProfileSample(AppProfileSection sectionIn) noexcept
        : section(sectionIn),
          startMicros(appRuntimeProfiler().isEnabled() ? appProfileNowMicroseconds() : 0)
    {
    }

    ~ScopedAppProfileSample()
    {
        if (startMicros <= 0)
            return;

        appRuntimeProfiler().addSample(section, appProfileNowMicroseconds() - startMicros);
    }

private:
    AppProfileSection section;
    int64_t startMicros = 0;
};

}

class HeaderLcdDisplay final : public juce::Component
{
public:
    enum class Mode
    {
        standard,
        yautja,
        virus
    };

    HeaderLcdDisplay()
    {
        yautjaToggle.setButtonText("Y");
        yautjaToggle.setClickingTogglesState(true);
        yautjaToggle.setTooltip("Yautja timecode");
        yautjaToggle.onClick = [this]
        {
            setMode(yautjaToggle.getToggleState() ? Mode::yautja : Mode::standard);
        };
        addAndMakeVisible(yautjaToggle);

        virusToggle.setButtonText("V");
        virusToggle.setClickingTogglesState(true);
        virusToggle.setTooltip("Virus style LCD");
        virusToggle.onClick = [this]
        {
            setMode(virusToggle.getToggleState() ? Mode::virus : Mode::standard);
        };
        addAndMakeVisible(virusToggle);

        setTheme(themeSpecForIndex(0));
    }

    void setTheme(const ThemeSpec& theme)
    {
        lcdBackgroundColour = theme.lcdBackground;
        lcdFrameColour = theme.lcdFrame;
        lcdGlowColour = theme.lcdGlow;
        lcdLabelColour = theme.lcdLabel;
        lcdGhostColour = theme.lcdGhost;
        lcdValueColour = theme.lcdValue;
        yautjaToggle.setColour(juce::TextButton::buttonColourId, theme.surface);
        yautjaToggle.setColour(juce::TextButton::buttonOnColourId, juce::Colour::fromRGB(78, 26, 26));
        yautjaToggle.setColour(juce::TextButton::textColourOffId, theme.lcdLabel);
        yautjaToggle.setColour(juce::TextButton::textColourOnId, juce::Colour::fromRGB(255, 124, 112));
        virusToggle.setColour(juce::TextButton::buttonColourId, theme.surface);
        virusToggle.setColour(juce::TextButton::buttonOnColourId, juce::Colour::fromRGB(68, 90, 116));
        virusToggle.setColour(juce::TextButton::textColourOffId, theme.lcdLabel);
        virusToggle.setColour(juce::TextButton::textColourOnId, juce::Colour::fromRGB(224, 244, 255));
        repaint();
    }

    void setMode(Mode newMode)
    {
        if (mode == newMode)
        {
            yautjaToggle.setToggleState(mode == Mode::yautja, juce::dontSendNotification);
            virusToggle.setToggleState(mode == Mode::virus, juce::dontSendNotification);
            return;
        }

        mode = newMode;
        yautjaToggle.setToggleState(mode == Mode::yautja, juce::dontSendNotification);
        virusToggle.setToggleState(mode == Mode::virus, juce::dontSendNotification);
        repaint();
    }

    void setValues(double playheadSecondsIn,
                   double totalSecondsIn,
                   double leftLocatorSecondsIn,
                   double rightLocatorSecondsIn)
    {
        const auto nextPosition = formatLcdSeconds(playheadSecondsIn);
        const auto nextLength = formatLcdSeconds(totalSecondsIn);
        const auto nextLeft = formatLcdSeconds(leftLocatorSecondsIn);
        const auto nextRight = formatLcdSeconds(rightLocatorSecondsIn);

        if (positionValue == nextPosition
            && totalLengthValue == nextLength
            && leftLocatorValue == nextLeft
            && rightLocatorValue == nextRight)
            return;

        positionValue = nextPosition;
        totalLengthValue = nextLength;
        leftLocatorValue = nextLeft;
        rightLocatorValue = nextRight;
        repaint();
    }

    void resized() override
    {
        auto area = getLocalBounds().reduced(6, 4);
        auto toggleStrip = area.removeFromBottom(10);
        auto centred = toggleStrip.withSizeKeepingCentre(32, 10);
        yautjaToggle.setBounds(centred.removeFromLeft(14));
        centred.removeFromLeft(4);
        virusToggle.setBounds(centred.removeFromLeft(14));
    }

    void paint(juce::Graphics& g) override
    {
        auto bounds = getLocalBounds().toFloat();
        if (bounds.isEmpty())
            return;

        g.setColour(lcdBackgroundColour);
        g.fillRoundedRectangle(bounds, 6.0f);
        g.setColour(lcdFrameColour);
        g.drawRoundedRectangle(bounds.reduced(0.5f), 6.0f, 1.0f);

        auto inner = bounds.reduced(6.0f, 4.0f);
        inner.removeFromBottom(10.0f);
        const auto useYautja = mode == Mode::yautja;
        const auto useVirus = mode == Mode::virus;
        g.setColour(useYautja ? juce::Colour::fromRGBA(255, 96, 96, 18)
                              : (useVirus ? juce::Colour::fromRGBA(168, 212, 255, 24)
                                          : lcdGlowColour));
        g.fillRoundedRectangle(inner, 4.5f);

        if (useVirus)
        {
            g.setColour(juce::Colour::fromRGBA(228, 242, 255, 10));
            for (float y = inner.getY() + 2.0f; y < inner.getBottom(); y += 4.0f)
                g.fillRect(inner.withY(y).withHeight(1.0f));
        }

        auto titleArea = inner.removeFromTop(8.0f);
        drawLabelText(g,
                      titleArea,
                      "TIME",
                      useYautja ? juce::Colour::fromRGB(255, 110, 104)
                                : (useVirus ? juce::Colour::fromRGB(196, 226, 255)
                                            : lcdLabelColour),
                      7.4f,
                      mode);

        inner.removeFromTop(2.0f);
        auto topRow = inner.removeFromTop(15.0f);
        auto bottomRow = inner.removeFromTop(14.0f);

        drawCell(g, topRow.removeFromLeft(topRow.getWidth() * 0.5f), "POS", positionValue, mode);
        drawCell(g, topRow, "LEN", totalLengthValue, mode);
        drawCell(g, bottomRow.removeFromLeft(bottomRow.getWidth() * 0.5f), "L", leftLocatorValue, mode);
        drawCell(g, bottomRow, "R", rightLocatorValue, mode);
    }

private:
    enum GlyphSegment
    {
        segN  = 1 << 0,
        segNE = 1 << 1,
        segE  = 1 << 2,
        segSE = 1 << 3,
        segS  = 1 << 4,
        segSW = 1 << 5,
        segW  = 1 << 6,
        segNW = 1 << 7,
        segC  = 1 << 8
    };

    static unsigned int yautjaMaskForChar(juce::juce_wchar ch)
    {
        switch (juce::CharacterFunctions::toUpperCase(ch))
        {
            case '0': return segN | segNE | segE | segSE | segS | segSW | segW | segNW;
            case '1': return segN | segS;
            case '2': return segN | segNE | segE | segC | segSW | segS;
            case '3': return segN | segNE | segE | segC | segSE | segS;
            case '4': return segNW | segW | segC | segNE | segE;
            case '5': return segN | segNW | segW | segC | segSE | segS;
            case '6': return segN | segNW | segW | segSW | segS | segSE | segC;
            case '7': return segN | segNE | segE;
            case '8': return segN | segNE | segE | segSE | segS | segSW | segW | segNW | segC;
            case '9': return segN | segNE | segE | segSE | segS | segNW | segW | segC;
            case 'A': return segN | segNE | segE | segW | segNW | segC;
            case 'E': return segN | segW | segNW | segS | segSW | segC;
            case 'I': return segN | segS | segC;
            case 'L': return segW | segSW | segS;
            case 'M': return segNW | segN | segNE | segSW | segSE;
            case 'N': return segNW | segNE | segSW | segSE | segC;
            case 'O': return segN | segNE | segE | segSE | segS | segSW | segW | segNW;
            case 'P': return segN | segNE | segE | segW | segNW | segC;
            case 'R': return segN | segNE | segE | segW | segNW | segC | segSE;
            case 'S': return segN | segNW | segW | segC | segSE | segS;
            case 'T': return segN | segC | segS;
            default: return 0;
        }
    }

    static void drawYautjaGlyph(juce::Graphics& g,
                                juce::Rectangle<float> area,
                                juce::juce_wchar ch,
                                juce::Colour colour)
    {
        if (ch == ' ')
            return;

        if (ch == ':')
        {
            const auto radius = juce::jmax(1.2f, area.getWidth() * 0.10f);
            g.setColour(colour);
            g.fillEllipse(area.getCentreX() - radius, area.getY() + area.getHeight() * 0.28f, radius * 2.0f, radius * 2.0f);
            g.fillEllipse(area.getCentreX() - radius, area.getY() + area.getHeight() * 0.64f, radius * 2.0f, radius * 2.0f);
            return;
        }

        if (ch == '.')
        {
            const auto radius = juce::jmax(1.2f, area.getWidth() * 0.10f);
            g.setColour(colour);
            g.fillEllipse(area.getCentreX() - radius, area.getBottom() - (radius * 2.6f), radius * 2.0f, radius * 2.0f);
            return;
        }

        const auto mask = yautjaMaskForChar(ch);
        if (mask == 0)
            return;

        juce::Path path;
        const auto x = area.getX();
        const auto y = area.getY();
        const auto w = area.getWidth();
        const auto h = area.getHeight();
        const auto cx = x + (w * 0.5f);
        const auto cy = y + (h * 0.54f);
        const auto top = y + (h * 0.10f);
        const auto upper = y + (h * 0.30f);
        const auto lower = y + (h * 0.74f);
        const auto bottom = y + (h * 0.92f);
        const auto left = x + (w * 0.10f);
        const auto innerLeft = x + (w * 0.28f);
        const auto innerRight = x + (w * 0.72f);
        const auto right = x + (w * 0.90f);

        const auto addSegment = [&path] (float x1, float y1, float x2, float y2)
        {
            path.startNewSubPath(x1, y1);
            path.lineTo(x2, y2);
        };

        if ((mask & segN) != 0)  addSegment(cx, upper, cx, top);
        if ((mask & segNE) != 0) addSegment(cx + (w * 0.04f), cy - (h * 0.06f), right, top);
        if ((mask & segE) != 0)  addSegment(innerRight, cy, right, cy);
        if ((mask & segSE) != 0) addSegment(cx + (w * 0.04f), cy + (h * 0.04f), innerRight, lower);
        if ((mask & segS) != 0)  addSegment(cx, lower, cx, bottom);
        if ((mask & segSW) != 0) addSegment(cx - (w * 0.04f), cy + (h * 0.04f), left, bottom);
        if ((mask & segW) != 0)  addSegment(left, cy, innerLeft, cy);
        if ((mask & segNW) != 0) addSegment(cx - (w * 0.04f), cy - (h * 0.06f), left, top);
        if ((mask & segC) != 0)  addSegment(innerLeft, cy, innerRight, cy);

        g.setColour(colour);
        g.strokePath(path, juce::PathStrokeType(juce::jmax(1.2f, w * 0.10f),
                                                juce::PathStrokeType::curved,
                                                juce::PathStrokeType::rounded));
    }

    static void drawYautjaText(juce::Graphics& g,
                               juce::Rectangle<float> area,
                               const juce::String& text,
                               juce::Colour colour,
                               float glyphHeight,
                               float xOffset = 0.0f)
    {
        const auto effectiveHeight = juce::jmax(8.0f, glyphHeight);
        auto glyphWidth = effectiveHeight * 0.58f;
        auto glyphGap = effectiveHeight * 0.18f;
        const auto totalWidth = (glyphWidth * static_cast<float>(text.length())) + (glyphGap * juce::jmax(0, text.length() - 1));
        const auto scale = totalWidth > area.getWidth() && totalWidth > 0.0f
            ? (area.getWidth() / totalWidth)
            : 1.0f;
        glyphWidth *= scale;
        glyphGap *= scale;

        auto x = area.getX() + xOffset;
        const auto y = area.getCentreY() - (effectiveHeight * scale * 0.5f);
        for (int index = 0; index < text.length(); ++index)
        {
            drawYautjaGlyph(g,
                            { x, y, glyphWidth, effectiveHeight * scale },
                            text[index],
                            colour);
            x += glyphWidth + glyphGap;
        }
    }

    static void drawLabelText(juce::Graphics& g,
                              juce::Rectangle<float> area,
                              const juce::String& text,
                              juce::Colour colour,
                              float normalFontHeight,
                              Mode displayMode = Mode::standard)
    {
        if (displayMode == Mode::yautja)
        {
            drawYautjaText(g, area, text, colour, area.getHeight() * 0.82f);
            return;
        }

        g.setColour(colour);
        g.setFont(juce::FontOptions(ui::scaleValue(displayMode == Mode::virus ? normalFontHeight + 0.3f : normalFontHeight),
                                    juce::Font::bold));
        g.drawText(text, area, juce::Justification::centredLeft, false);
    }

    void drawCell(juce::Graphics& g,
                  juce::Rectangle<float> area,
                  const juce::String& label,
                  const juce::String& value,
                  Mode displayMode) const
    {
        const auto useYautja = displayMode == Mode::yautja;
        const auto useVirus = displayMode == Mode::virus;
        auto cell = area.reduced(4.0f, 0.0f);
        auto labelArea = cell.removeFromLeft(useYautja ? 28.0f : (useVirus ? 26.0f : 22.0f));
        auto valueArea = cell;

        const auto labelColour = useYautja ? juce::Colour::fromRGB(255, 120, 112)
                                           : (useVirus ? juce::Colour::fromRGB(198, 226, 255)
                                                       : lcdLabelColour);
        const auto ghostColour = useYautja ? juce::Colour::fromRGBA(172, 72, 68, 44)
                                           : (useVirus ? juce::Colour::fromRGBA(124, 156, 188, 56)
                                                       : lcdGhostColour);
        const auto valueColour = useYautja ? juce::Colour::fromRGB(255, 132, 124)
                                           : (useVirus ? juce::Colour::fromRGB(232, 244, 255)
                                                       : lcdValueColour);

        drawLabelText(g, labelArea, label, labelColour, 7.4f, displayMode);

        const auto ghostText = lcdGhostForValue(value);
        if (useYautja)
        {
            drawYautjaText(g, valueArea, ghostText, ghostColour, ui::scaleValue(10.8f), 1.0f);
            drawYautjaText(g, valueArea, value, valueColour, ui::scaleValue(10.8f), 1.0f);
        }
        else
        {
            g.setFont(ui::strongFont());
            g.setColour(ghostColour);
            g.drawText(ghostText, valueArea, juce::Justification::centredLeft, false);
            g.setColour(valueColour);
            g.drawText(value, valueArea, juce::Justification::centredLeft, false);
        }
    }

    juce::String positionValue = "00:00.00";
    juce::String totalLengthValue = "00:00.00";
    juce::String leftLocatorValue = "00:00.00";
    juce::String rightLocatorValue = "00:00.00";
    Mode mode = Mode::standard;
    juce::Colour lcdBackgroundColour = juce::Colour::fromRGB(14, 22, 18);
    juce::Colour lcdFrameColour = juce::Colour::fromRGB(58, 94, 72);
    juce::Colour lcdGlowColour = juce::Colour::fromRGBA(120, 255, 174, 22);
    juce::Colour lcdLabelColour = juce::Colour::fromRGB(135, 214, 164);
    juce::Colour lcdGhostColour = juce::Colour::fromRGBA(106, 170, 121, 52);
    juce::Colour lcdValueColour = juce::Colour::fromRGB(175, 255, 196);
    juce::TextButton yautjaToggle;
    juce::TextButton virusToggle;
};

class FloatingPanelWindow final : public juce::DocumentWindow
{
public:
    explicit FloatingPanelWindow(const juce::String& title, bool alwaysOnTop = false)
        : juce::DocumentWindow(title,
                               juce::Colour::fromRGB(11, 13, 17),
                               juce::DocumentWindow::allButtons)
    {
        setUsingNativeTitleBar(true);
        setResizable(true, true);
        setAlwaysOnTop(alwaysOnTop);
        setIcon(loadMutagenLogoBinaryData(true));
    }

    std::function<void()> onClosePressed;
    std::function<bool(const juce::KeyPress&)> onKeyPressed;

private:
    bool keyPressed(const juce::KeyPress& key) override
    {
        if (onKeyPressed != nullptr && onKeyPressed(key))
            return true;
        return juce::DocumentWindow::keyPressed(key);
    }

    void closeButtonPressed() override
    {
        setVisible(false);
        if (onClosePressed != nullptr)
            onClosePressed();
    }
};

class ActivityLogWindowComponent final : public juce::Component
{
public:
    ActivityLogWindowComponent(std::function<juce::String()> getLogTextIn,
                               std::function<void()> clearLogIn)
        : getLogText(std::move(getLogTextIn)),
          clearLog(std::move(clearLogIn))
    {
        titleLabel.setText("Activity Log", juce::dontSendNotification);
        titleLabel.setColour(juce::Label::textColourId, juce::Colour::fromRGB(230, 235, 242));
        titleLabel.setFont(ui::titleFont());
        addAndMakeVisible(titleLabel);

        clearButton.setButtonText("Clear");
        clearButton.onClick = [this]
        {
            if (clearLog != nullptr)
                clearLog();
        };
        addAndMakeVisible(clearButton);

        logEditor.setMultiLine(true);
        logEditor.setReadOnly(true);
        logEditor.setScrollbarsShown(true);
        logEditor.setCaretVisible(false);
        logEditor.setPopupMenuEnabled(true);
        logEditor.setColour(juce::TextEditor::backgroundColourId, juce::Colour::fromRGB(15, 18, 23));
        logEditor.setColour(juce::TextEditor::textColourId, juce::Colour::fromRGB(220, 226, 235));
        logEditor.setColour(juce::TextEditor::outlineColourId, juce::Colour::fromRGB(62, 71, 86));
        logEditor.setColour(juce::TextEditor::focusedOutlineColourId, juce::Colour::fromRGB(88, 143, 212));
        logEditor.setFont(ui::font());
        addAndMakeVisible(logEditor);
    }

    void refreshFromModel()
    {
        const auto text = getLogText != nullptr ? getLogText() : juce::String();
        if (text == lastRenderedText)
            return;

        lastRenderedText = text;
        logEditor.setText(text, false);
        logEditor.moveCaretToEnd();
    }

    void resized() override
    {
        auto area = getLocalBounds().reduced(10);
        auto header = area.removeFromTop(28);
        titleLabel.setBounds(header.removeFromLeft(180));
        clearButton.setBounds(header.removeFromRight(90));
        area.removeFromTop(8);
        logEditor.setBounds(area);
    }

private:
    std::function<juce::String()> getLogText;
    std::function<void()> clearLog;
    juce::String lastRenderedText;
    juce::Label titleLabel;
    juce::TextButton clearButton;
    juce::TextEditor logEditor;
};

class VstFolderManagerComponent final : public juce::Component
{
public:
    VstFolderManagerComponent(std::function<juce::String()> defaultFolderGetterIn,
                              std::function<juce::StringArray()> userFoldersGetterIn,
                              std::function<void()> addFolderIn,
                              std::function<void(const juce::String&)> removeFolderIn,
                              std::function<void()> refreshCatalogIn)
        : defaultFolderGetter(std::move(defaultFolderGetterIn)),
          userFoldersGetter(std::move(userFoldersGetterIn)),
          addFolder(std::move(addFolderIn)),
          removeFolder(std::move(removeFolderIn)),
          refreshCatalog(std::move(refreshCatalogIn)),
          folderListModel(*this)
    {
        titleLabel.setText("VST Folder Manager", juce::dontSendNotification);
        titleLabel.setColour(juce::Label::textColourId, juce::Colour::fromRGB(230, 235, 242));
        titleLabel.setFont(ui::titleFont());
        addAndMakeVisible(titleLabel);

        defaultFolderLabel.setText("Default VST Folder", juce::dontSendNotification);
        defaultFolderLabel.setColour(juce::Label::textColourId, juce::Colour::fromRGB(190, 199, 210));
        addAndMakeVisible(defaultFolderLabel);

        defaultFolderEditor.setReadOnly(true);
        defaultFolderEditor.setMultiLine(false);
        defaultFolderEditor.setScrollbarsShown(true);
        defaultFolderEditor.setPopupMenuEnabled(true);
        defaultFolderEditor.setColour(juce::TextEditor::backgroundColourId, juce::Colour::fromRGB(15, 18, 23));
        defaultFolderEditor.setColour(juce::TextEditor::textColourId, juce::Colour::fromRGB(226, 232, 240));
        defaultFolderEditor.setColour(juce::TextEditor::outlineColourId, juce::Colour::fromRGB(62, 71, 86));
        defaultFolderEditor.setColour(juce::TextEditor::focusedOutlineColourId, juce::Colour::fromRGB(88, 143, 212));
        addAndMakeVisible(defaultFolderEditor);

        userFoldersLabel.setText("User VST Folders", juce::dontSendNotification);
        userFoldersLabel.setColour(juce::Label::textColourId, juce::Colour::fromRGB(190, 199, 210));
        addAndMakeVisible(userFoldersLabel);

        folderList.setModel(&folderListModel);
        folderList.setRowHeight(30);
        folderList.setColour(juce::ListBox::backgroundColourId, juce::Colour::fromRGB(20, 22, 28));
        folderList.setColour(juce::ListBox::outlineColourId, juce::Colour::fromRGB(62, 71, 86));
        addAndMakeVisible(folderList);

        helperLabel.setText("The default VST folder is always scanned. Add any extra plugin folders below.",
                            juce::dontSendNotification);
        helperLabel.setColour(juce::Label::textColourId, juce::Colour::fromRGB(150, 159, 170));
        addAndMakeVisible(helperLabel);

        configureButton(addButton, "Add Folder...");
        addButton.onClick = [this]
        {
            if (addFolder != nullptr)
                addFolder();
        };

        configureButton(removeButton, "Remove");
        removeButton.onClick = [this]
        {
            if (removeFolder == nullptr)
                return;

            const auto folder = selectedFolderPath();
            if (folder.isNotEmpty())
                removeFolder(folder);
        };

        configureButton(refreshButton, "Refresh Catalog");
        refreshButton.onClick = [this]
        {
            if (refreshCatalog != nullptr)
                refreshCatalog();
        };

        addAndMakeVisible(addButton);
        addAndMakeVisible(removeButton);
        addAndMakeVisible(refreshButton);

        refreshFromModel();
    }

    void refreshFromModel()
    {
        const auto defaultFolder = defaultFolderGetter != nullptr ? defaultFolderGetter().trim() : juce::String();
        const auto defaultText = defaultFolder.isNotEmpty() ? defaultFolder : juce::String("(Default VST folder not found)");
        if (defaultFolderEditor.getText() != defaultText)
            defaultFolderEditor.setText(defaultText, false);

        const auto selectedFolder = selectedFolderPath();
        const auto latestFolders = userFoldersGetter != nullptr ? userFoldersGetter() : juce::StringArray();
        if (folderPaths.joinIntoString("\n") != latestFolders.joinIntoString("\n"))
        {
            folderPaths = latestFolders;
            folderList.updateContent();
            if (folderPaths.isEmpty())
            {
                folderList.deselectAllRows();
            }
            else
            {
                const auto selectedIndex = folderPaths.indexOf(selectedFolder);
                folderList.selectRow(selectedIndex >= 0 ? selectedIndex : 0);
            }
        }

        removeButton.setEnabled(folderList.getSelectedRow() >= 0);
    }

    void resized() override
    {
        auto area = getLocalBounds().reduced(10);
        auto header = area.removeFromTop(28);
        titleLabel.setBounds(header.removeFromLeft(220));
        area.removeFromTop(6);

        defaultFolderLabel.setBounds(area.removeFromTop(22));
        defaultFolderEditor.setBounds(area.removeFromTop(30));
        area.removeFromTop(8);

        userFoldersLabel.setBounds(area.removeFromTop(22));
        auto buttonRow = area.removeFromBottom(38);
        helperLabel.setBounds(area.removeFromBottom(22));
        area.removeFromBottom(6);
        folderList.setBounds(area);

        addButton.setBounds(buttonRow.removeFromLeft(120));
        buttonRow.removeFromLeft(8);
        removeButton.setBounds(buttonRow.removeFromLeft(100));
        refreshButton.setBounds(buttonRow.removeFromRight(140));
    }

private:
    class FolderListModel final : public juce::ListBoxModel
    {
    public:
        explicit FolderListModel(VstFolderManagerComponent& ownerIn) : owner(ownerIn) {}

        int getNumRows() override
        {
            return owner.folderPaths.size();
        }

        void paintListBoxItem(int rowNumber,
                              juce::Graphics& g,
                              int width,
                              int height,
                              bool rowIsSelected) override
        {
            if (!juce::isPositiveAndBelow(rowNumber, owner.folderPaths.size()))
                return;

            g.fillAll(rowIsSelected ? juce::Colour::fromRGB(46, 88, 138)
                                    : ((rowNumber % 2) == 0 ? juce::Colour::fromRGB(26, 30, 37)
                                                            : juce::Colour::fromRGB(21, 25, 31)));
            g.setColour(rowIsSelected ? juce::Colours::white : juce::Colour::fromRGB(226, 232, 240));
            g.setFont(ui::font());
            g.drawText(owner.folderPaths[rowNumber], 8, 0, width - 12, height, juce::Justification::centredLeft, true);
        }

        void selectedRowsChanged(int) override
        {
            owner.refreshFromModel();
        }

    private:
        VstFolderManagerComponent& owner;
    };

    static void configureButton(juce::TextButton& button, const juce::String& text)
    {
        button.setButtonText(text);
        button.setColour(juce::TextButton::buttonColourId, juce::Colour::fromRGB(46, 52, 64));
        button.setColour(juce::TextButton::buttonOnColourId, juce::Colour::fromRGB(72, 104, 160));
        button.setColour(juce::TextButton::textColourOffId, juce::Colour::fromRGB(235, 239, 244));
    }

    juce::String selectedFolderPath() const
    {
        const auto row = folderList.getSelectedRow();
        if (!juce::isPositiveAndBelow(row, folderPaths.size()))
            return {};
        return folderPaths[row];
    }

    std::function<juce::String()> defaultFolderGetter;
    std::function<juce::StringArray()> userFoldersGetter;
    std::function<void()> addFolder;
    std::function<void(const juce::String&)> removeFolder;
    std::function<void()> refreshCatalog;
    FolderListModel folderListModel;
    juce::StringArray folderPaths;
    juce::Label titleLabel;
    juce::Label defaultFolderLabel;
    juce::TextEditor defaultFolderEditor;
    juce::Label userFoldersLabel;
    juce::Label helperLabel;
    juce::ListBox folderList;
    juce::TextButton addButton;
    juce::TextButton removeButton;
    juce::TextButton refreshButton;
};

class TransportPanelComponent final : public juce::Component
{
public:
    TransportPanelComponent(std::function<void()> jumpToStartIn,
                            std::function<void()> playProjectIn,
                            std::function<void()> playTrackIn,
                            std::function<void()> stopPlaybackIn,
                            std::function<void(int)> setPlayheadTickIn,
                            std::function<void(int)> setLeftLocatorTickIn,
                            std::function<void(int)> setRightLocatorTickIn,
                            std::function<void(int)> setTempoIn,
                            std::function<void(bool)> setLoopEnabledIn,
                            std::function<void(bool)> setMetronomeEnabledIn,
                            std::function<void(bool)> setRecordEnabledIn)
        : jumpToStart(std::move(jumpToStartIn)),
          playProject(std::move(playProjectIn)),
          playTrack(std::move(playTrackIn)),
          stopPlayback(std::move(stopPlaybackIn)),
          setPlayheadTick(std::move(setPlayheadTickIn)),
          setLeftLocatorTick(std::move(setLeftLocatorTickIn)),
          setRightLocatorTick(std::move(setRightLocatorTickIn)),
          setTempo(std::move(setTempoIn)),
          setLoopEnabled(std::move(setLoopEnabledIn)),
          setMetronomeEnabled(std::move(setMetronomeEnabledIn)),
          setRecordEnabled(std::move(setRecordEnabledIn))
    {
        configureButton(homeButton, "Home");
        homeButton.onClick = [this] { if (jumpToStart != nullptr) jumpToStart(); };

        configureButton(playProjectButton, "Play Project");
        playProjectButton.onClick = [this] { if (playProject != nullptr) playProject(); };

        configureButton(playTrackButton, "Play Track");
        playTrackButton.onClick = [this] { if (playTrack != nullptr) playTrack(); };

        configureButton(stopButton, "Stop");
        stopButton.onClick = [this] { if (stopPlayback != nullptr) stopPlayback(); };

        tempoLabel.setText("Tempo", juce::dontSendNotification);
        tempoLabel.setColour(juce::Label::textColourId, juce::Colour::fromRGB(214, 219, 227));
        addAndMakeVisible(tempoLabel);

        tempoSlider.setSliderStyle(juce::Slider::LinearHorizontal);
        tempoSlider.setRange(20.0, 300.0, 1.0);
        tempoSlider.setTextBoxStyle(juce::Slider::TextBoxRight, false, 70, 22);
        tempoSlider.setTextValueSuffix(" BPM");
        tempoSlider.onValueChange = [this]
        {
            if (!syncing && setTempo != nullptr)
                setTempo(juce::roundToInt(tempoSlider.getValue()));
        };
        addAndMakeVisible(tempoSlider);

        loopToggle.setButtonText("Loop");
        loopToggle.onClick = [this]
        {
            if (!syncing && setLoopEnabled != nullptr)
                setLoopEnabled(loopToggle.getToggleState());
        };
        addAndMakeVisible(loopToggle);

        metronomeToggle.setButtonText("Metronome");
        metronomeToggle.onClick = [this]
        {
            if (!syncing && setMetronomeEnabled != nullptr)
                setMetronomeEnabled(metronomeToggle.getToggleState());
        };
        addAndMakeVisible(metronomeToggle);

        recordToggle.setButtonText("Rec");
        recordToggle.setClickingTogglesState(true);
        recordToggle.setColour(juce::TextButton::buttonColourId, juce::Colour::fromRGB(58, 34, 40));
        recordToggle.setColour(juce::TextButton::buttonOnColourId, juce::Colour::fromRGB(184, 62, 74));
        recordToggle.setColour(juce::TextButton::textColourOffId, juce::Colours::white);
        recordToggle.setColour(juce::TextButton::textColourOnId, juce::Colours::white);
        recordToggle.setTooltip("Record incoming MIDI to the selected track during project playback.");
        recordToggle.onClick = [this]
        {
            if (!syncing && setRecordEnabled != nullptr)
                setRecordEnabled(recordToggle.getToggleState());
        };
        addAndMakeVisible(recordToggle);

        configureButton(setLeftButton, "Set L");
        setLeftButton.setTooltip("Set the left locator to the current playhead.");
        setLeftButton.onClick = [this]
        {
            if (setLeftLocatorTick != nullptr)
                setLeftLocatorTick(displayedProject.playheadTick);
        };

        configureButton(setRightButton, "Set R");
        setRightButton.setTooltip("Set the right locator to the current playhead.");
        setRightButton.onClick = [this]
        {
            if (setRightLocatorTick != nullptr)
                setRightLocatorTick(displayedProject.playheadTick);
        };

        statusLabel.setColour(juce::Label::textColourId, juce::Colour::fromRGB(230, 235, 242));
        statusLabel.setJustificationType(juce::Justification::centredLeft);
        addAndMakeVisible(statusLabel);

        playheadLabel.setColour(juce::Label::textColourId, juce::Colour::fromRGB(156, 199, 239));
        playheadLabel.setJustificationType(juce::Justification::centredLeft);
        addAndMakeVisible(playheadLabel);

        cpuUsageLabel.setColour(juce::Label::textColourId, juce::Colour::fromRGB(230, 235, 242));
        cpuUsageLabel.setJustificationType(juce::Justification::centred);
        cpuUsageLabel.setFont(ui::font());
        addAndMakeVisible(cpuUsageLabel);

        addAndMakeVisible(setLeftButton);
        addAndMakeVisible(setRightButton);
    }

    void refreshFromState(const ProjectState& project,
                          bool hasTrackSelection,
                          bool rackPlaying,
                          bool projectPlaying,
                          bool recordEnabled,
                          const juce::String& statusText,
                          double cpuUsagePercent,
                          float masterPeakLeftIn,
                          float masterPeakRightIn)
    {
        if (activeMarkerDrag == MarkerDragTarget::none)
        {
            displayedProject = project;
            displayedSequenceLengthTicks = juce::jmax(ticksPerBar(project),
                                                      projectSequenceLengthTicks(project) + ticksPerBar(project));
        }

        const auto& transportDisplayProject = activeMarkerDrag == MarkerDragTarget::none ? project : displayedProject;
        syncing = true;
        tempoSlider.setValue(transportDisplayProject.bpm, juce::dontSendNotification);
        loopToggle.setToggleState(transportDisplayProject.loopEnabled, juce::dontSendNotification);
        metronomeToggle.setToggleState(transportDisplayProject.metronomeEnabled, juce::dontSendNotification);
        recordToggle.setToggleState(recordEnabled, juce::dontSendNotification);
        syncing = false;
        masterPeakLeft = juce::jlimit(0.0f, 1.0f, masterPeakLeftIn);
        masterPeakRight = juce::jlimit(0.0f, 1.0f, masterPeakRightIn);
        cpuUsageLabel.setText("CPU\n" + juce::String(juce::roundToInt(cpuUsagePercent)) + "%", juce::dontSendNotification);

        playTrackButton.setEnabled(hasTrackSelection);
        stopButton.setEnabled(rackPlaying || projectPlaying);
        recordToggle.setEnabled(hasTrackSelection || !project.tracks.empty());

        const auto playheadSec = tickToSeconds(transportDisplayProject.playheadTick, transportDisplayProject.bpm);
        playheadLabel.setText("Playhead: tick "
                                  + juce::String(transportDisplayProject.playheadTick)
                                  + "  |  "
                                  + juce::String(playheadSec, 2)
                                  + " s  |  "
                                  + "Locators "
                                  + juce::String(transportDisplayProject.leftLocatorTick)
                                  + " - "
                                  + juce::String(transportDisplayProject.rightLocatorTick),
                              juce::dontSendNotification);

        juce::String transportState;
        if (projectPlaying && recordEnabled)
            transportState = "Project recording active";
        else if (projectPlaying)
            transportState = "Project playback active";
        else if (rackPlaying)
            transportState = "Track playback active";
        else
            transportState = "Transport idle";

        if (statusText.trim().isNotEmpty())
            transportState << "  |  " << statusText.trim();
        statusLabel.setText(transportState, juce::dontSendNotification);
        repaint(locatorStripBounds.expanded(4));
    }

    void paint(juce::Graphics& g) override
    {
        g.fillAll(juce::Colour::fromRGB(13, 15, 20));
        g.setColour(juce::Colour::fromRGB(31, 35, 44));
        g.drawRect(getLocalBounds(), 1);

        g.setColour(juce::Colour::fromRGB(36, 41, 52));
        g.fillRoundedRectangle(masterMeterBounds.toFloat(), 6.0f);
        g.setColour(juce::Colour::fromRGB(66, 72, 86));
        g.drawRoundedRectangle(masterMeterBounds.toFloat(), 6.0f, 1.0f);

        auto leftBounds = masterMeterBounds.removeFromLeft(masterMeterBounds.getWidth() / 2).reduced(4, 4);
        auto rightBounds = masterMeterBounds.reduced(4, 4);
        paintMeterBar(g, leftBounds, masterPeakLeft);
        paintMeterBar(g, rightBounds, masterPeakRight);

        if (!locatorStripBounds.isEmpty())
            paintLocatorStrip(g);
    }

    void resized() override
    {
        auto area = getLocalBounds().reduced(12);
        auto controls = area.removeFromTop(30);
        homeButton.setBounds(controls.removeFromLeft(72));
        controls.removeFromLeft(6);
        playProjectButton.setBounds(controls.removeFromLeft(116));
        controls.removeFromLeft(6);
        recordToggle.setBounds(controls.removeFromLeft(60));
        controls.removeFromLeft(6);
        playTrackButton.setBounds(controls.removeFromLeft(96));
        controls.removeFromLeft(6);
        stopButton.setBounds(controls.removeFromLeft(72));
        controls.removeFromLeft(14);
        tempoLabel.setBounds(controls.removeFromLeft(44));
        tempoSlider.setBounds(controls.removeFromLeft(220));
        controls.removeFromLeft(10);
        loopToggle.setBounds(controls.removeFromLeft(72));
        controls.removeFromLeft(6);
        metronomeToggle.setBounds(controls.removeFromLeft(110));
        controls.removeFromLeft(10);
        setLeftButton.setBounds(controls.removeFromLeft(58));
        controls.removeFromLeft(6);
        setRightButton.setBounds(controls.removeFromLeft(58));
        controls.removeFromLeft(10);
        cpuUsageLabel.setBounds(controls.removeFromLeft(60));
        controls.removeFromLeft(8);
        masterMeterBounds = controls.removeFromLeft(40).reduced(0, 2);

        area.removeFromTop(8);
        playheadLabel.setBounds(area.removeFromTop(22));
        area.removeFromTop(4);
        locatorStripBounds = area.removeFromTop(34);
        area.removeFromTop(6);
        statusLabel.setBounds(area.removeFromTop(20));
    }

    void mouseDown(const juce::MouseEvent& event) override
    {
        if (!locatorStripBounds.toFloat().contains(event.position))
            return;

        const auto target = hitTestLocatorMarker(event.position);
        if (target != MarkerDragTarget::none)
        {
            activeMarkerDrag = target;
            applyLocatorTarget(target, tickForLocatorPosition(event.position.x), false);
            return;
        }

        if (event.mods.isShiftDown())
        {
            applyLocatorTarget(MarkerDragTarget::leftLocator, tickForLocatorPosition(event.position.x), true);
            return;
        }

        if (event.mods.isAltDown() || event.mods.isRightButtonDown())
        {
            applyLocatorTarget(MarkerDragTarget::rightLocator, tickForLocatorPosition(event.position.x), true);
            return;
        }

        applyLocatorTarget(MarkerDragTarget::playhead, tickForLocatorPosition(event.position.x), true);
    }

    void mouseDrag(const juce::MouseEvent& event) override
    {
        if (activeMarkerDrag == MarkerDragTarget::none)
            return;

        applyLocatorTarget(activeMarkerDrag, tickForLocatorPosition(event.position.x), false);
    }

    void mouseUp(const juce::MouseEvent&) override
    {
        if (activeMarkerDrag != MarkerDragTarget::none)
            applyLocatorTarget(activeMarkerDrag, currentTickForTarget(activeMarkerDrag), true);
        activeMarkerDrag = MarkerDragTarget::none;
    }

private:
    enum class MarkerDragTarget
    {
        none,
        playhead,
        leftLocator,
        rightLocator
    };

    void configureButton(juce::TextButton& button, const juce::String& text)
    {
        button.setButtonText(text);
        button.setColour(juce::TextButton::buttonColourId, juce::Colour::fromRGB(44, 50, 62));
        button.setColour(juce::TextButton::textColourOffId, juce::Colours::white);
        addAndMakeVisible(button);
    }

    static void paintMeterBar(juce::Graphics& g, juce::Rectangle<int> bounds, float level)
    {
        g.setColour(juce::Colour::fromRGB(20, 23, 30));
        g.fillRoundedRectangle(bounds.toFloat(), 4.0f);

        const auto filledHeight = juce::roundToInt(static_cast<float>(bounds.getHeight()) * juce::jlimit(0.0f, 1.0f, level));
        if (filledHeight <= 0)
            return;

        auto filledBounds = bounds.withTrimmedTop(bounds.getHeight() - filledHeight);
        juce::ColourGradient gradient(juce::Colour::fromRGB(61, 210, 122),
                                      filledBounds.getBottomLeft().toFloat(),
                                      juce::Colour::fromRGB(242, 99, 84),
                                      filledBounds.getTopLeft().toFloat(),
                                      false);
        g.setGradientFill(gradient);
        g.fillRoundedRectangle(filledBounds.toFloat(), 4.0f);
    }

    void paintLocatorStrip(juce::Graphics& g) const
    {
        auto strip = locatorStripBounds.toFloat().reduced(0.5f);
        if (strip.isEmpty())
            return;

        const auto leftColour = juce::Colour::fromRGB(120, 212, 255);
        const auto rightColour = juce::Colour::fromRGB(255, 209, 102);
        const auto playheadColour = juce::Colour::fromRGB(255, 102, 102);

        g.setColour(juce::Colour::fromRGB(22, 26, 34));
        g.fillRoundedRectangle(strip, 6.0f);
        g.setColour(juce::Colour::fromRGB(66, 72, 86));
        g.drawRoundedRectangle(strip, 6.0f, 1.0f);

        auto content = strip.reduced(12.0f, 8.0f);
        const auto centreY = content.getCentreY();
        g.setColour(juce::Colour::fromRGB(82, 92, 108));
        g.drawLine(content.getX(), centreY, content.getRight(), centreY, 2.0f);

        const auto leftX = locatorXForTick(displayedProject.leftLocatorTick);
        const auto rightX = locatorXForTick(displayedProject.rightLocatorTick);
        const auto playheadX = locatorXForTick(displayedProject.playheadTick);

        auto region = juce::Rectangle<float>(leftX,
                                             content.getY() + 2.0f,
                                             juce::jmax(2.0f, rightX - leftX),
                                             content.getHeight() - 4.0f);
        g.setColour(leftColour.interpolatedWith(rightColour, 0.5f).withAlpha(0.26f));
        g.fillRoundedRectangle(region, 4.0f);
        g.setColour(leftColour.interpolatedWith(rightColour, 0.55f));
        g.drawRoundedRectangle(region, 4.0f, 1.0f);

        paintLocatorMarker(g, strip, leftX, leftColour, "L");
        paintLocatorMarker(g, strip, rightX, rightColour, "R");
        paintLocatorMarker(g, strip, playheadX, playheadColour, "P");
    }

    static void paintLocatorMarker(juce::Graphics& g,
                                   juce::Rectangle<float> strip,
                                   float x,
                                   juce::Colour colour,
                                   const juce::String& label)
    {
        g.setColour(colour);
        g.drawVerticalLine(juce::roundToInt(x), strip.getY() + 3.0f, strip.getBottom() - 4.0f);

        auto cap = juce::Rectangle<float>(x - 10.0f, strip.getY() + 2.0f, 20.0f, 12.0f);
        g.fillRoundedRectangle(cap, 4.0f);
        g.setColour(juce::Colours::black.withAlpha(0.85f));
        g.setFont(ui::strongFont());
        g.drawText(label, cap.toNearestInt(), juce::Justification::centred);

        juce::Path pointer;
        pointer.addTriangle(x,
                            strip.getBottom() - 2.0f,
                            x - 4.0f,
                            strip.getBottom() - 9.0f,
                            x + 4.0f,
                            strip.getBottom() - 9.0f);
        g.setColour(colour);
        g.fillPath(pointer);
    }

    float locatorXForTick(int tick) const
    {
        auto content = locatorStripBounds.toFloat().reduced(12.0f, 8.0f);
        if (content.getWidth() <= 0.0f)
            return content.getX();

        const auto safeTotalTicks = juce::jmax(1, displayedSequenceLengthTicks);
        const auto proportion = static_cast<float>(juce::jlimit(0, safeTotalTicks, tick))
            / static_cast<float>(safeTotalTicks);
        return content.getX() + (content.getWidth() * proportion);
    }

    int tickForLocatorPosition(float x) const
    {
        auto content = locatorStripBounds.toFloat().reduced(12.0f, 8.0f);
        if (content.getWidth() <= 0.0f)
            return 0;

        const auto safeTotalTicks = juce::jmax(1, displayedSequenceLengthTicks);
        const auto clampedX = juce::jlimit(content.getX(), content.getRight(), x);
        const auto proportion = static_cast<double>(clampedX - content.getX()) / static_cast<double>(content.getWidth());
        auto tick = juce::roundToInt(proportion * static_cast<double>(safeTotalTicks));
        const auto snapTicks = juce::jmax(1, displayedProject.arrangementSnapTicks);
        tick = static_cast<int>(std::llround(static_cast<double>(tick) / static_cast<double>(snapTicks))) * snapTicks;
        return juce::jmax(0, tick);
    }

    MarkerDragTarget hitTestLocatorMarker(juce::Point<float> position) const
    {
        if (!locatorStripBounds.toFloat().contains(position))
            return MarkerDragTarget::none;

        const auto threshold = 9.0f;
        const auto distanceTo = [&position, threshold] (float markerX)
        {
            return std::abs(position.x - markerX) <= threshold;
        };

        if (distanceTo(locatorXForTick(displayedProject.leftLocatorTick)))
            return MarkerDragTarget::leftLocator;
        if (distanceTo(locatorXForTick(displayedProject.rightLocatorTick)))
            return MarkerDragTarget::rightLocator;
        if (distanceTo(locatorXForTick(displayedProject.playheadTick)))
            return MarkerDragTarget::playhead;

        return MarkerDragTarget::none;
    }

    int currentTickForTarget(MarkerDragTarget target) const
    {
        switch (target)
        {
            case MarkerDragTarget::playhead: return displayedProject.playheadTick;
            case MarkerDragTarget::leftLocator: return displayedProject.leftLocatorTick;
            case MarkerDragTarget::rightLocator: return displayedProject.rightLocatorTick;
            case MarkerDragTarget::none: break;
        }

        return displayedProject.playheadTick;
    }

    void applyLocatorTarget(MarkerDragTarget target, int tick, bool commit)
    {
        const auto minimumSpan = juce::jmax(1, ticksPerTimeSignatureBeat(displayedProject));
        switch (target)
        {
            case MarkerDragTarget::playhead:
                displayedProject.playheadTick = juce::jmax(0, tick);
                displayedProject.recalculateTimeFields();
                repaint(locatorStripBounds.expanded(4));
                if (commit && setPlayheadTick != nullptr)
                    setPlayheadTick(displayedProject.playheadTick);
                break;

            case MarkerDragTarget::leftLocator:
                displayedProject.leftLocatorTick = juce::jmin(tick,
                                                              juce::jmax(0, displayedProject.rightLocatorTick - minimumSpan));
                displayedProject.recalculateTimeFields();
                repaint(locatorStripBounds.expanded(4));
                if (commit && setLeftLocatorTick != nullptr)
                    setLeftLocatorTick(displayedProject.leftLocatorTick);
                break;

            case MarkerDragTarget::rightLocator:
                displayedProject.rightLocatorTick = juce::jmax(displayedProject.leftLocatorTick + minimumSpan, tick);
                displayedProject.recalculateTimeFields();
                repaint(locatorStripBounds.expanded(4));
                if (commit && setRightLocatorTick != nullptr)
                    setRightLocatorTick(displayedProject.rightLocatorTick);
                break;

            case MarkerDragTarget::none:
                break;
        }
    }

    std::function<void()> jumpToStart;
    std::function<void()> playProject;
    std::function<void()> playTrack;
    std::function<void()> stopPlayback;
    std::function<void(int)> setPlayheadTick;
    std::function<void(int)> setLeftLocatorTick;
    std::function<void(int)> setRightLocatorTick;
    std::function<void(int)> setTempo;
    std::function<void(bool)> setLoopEnabled;
    std::function<void(bool)> setMetronomeEnabled;
    std::function<void(bool)> setRecordEnabled;
    bool syncing = false;

    juce::TextButton homeButton;
    juce::TextButton playProjectButton;
    juce::TextButton recordToggle;
    juce::TextButton playTrackButton;
    juce::TextButton stopButton;
    juce::Label tempoLabel;
    juce::Slider tempoSlider;
    juce::ToggleButton loopToggle;
    juce::ToggleButton metronomeToggle;
    juce::TextButton setLeftButton;
    juce::TextButton setRightButton;
    juce::Label cpuUsageLabel;
    juce::Label playheadLabel;
    juce::Label statusLabel;
    juce::Rectangle<int> masterMeterBounds;
    juce::Rectangle<int> locatorStripBounds;
    ProjectState displayedProject;
    int displayedSequenceLengthTicks = kTicksPerBar;
    MarkerDragTarget activeMarkerDrag = MarkerDragTarget::none;
    float masterPeakLeft = 0.0f;
    float masterPeakRight = 0.0f;
};

class AudioSettingsPanelComponent final : public juce::Component
{
public:
    AudioSettingsPanelComponent(std::function<juce::Result(const juce::String&,
                                                           const juce::String&,
                                                           int,
                                                           int)> applySettingsIn,
                                std::function<juce::Result(const juce::String&)> applyMidiInputSettingIn,
                                std::function<void()> refreshStatusIn)
        : applySettings(std::move(applySettingsIn)),
          applyMidiInputSetting(std::move(applyMidiInputSettingIn)),
          refreshStatus(std::move(refreshStatusIn))
    {
        summaryLabel.setColour(juce::Label::textColourId, juce::Colour::fromRGB(230, 235, 242));
        summaryLabel.setJustificationType(juce::Justification::topLeft);
        addAndMakeVisible(summaryLabel);

        configureCombo(driverTypeLabel, driverTypeCombo, "Driver Backend");
        configureCombo(outputDeviceLabel, outputDeviceCombo, "Output Device");
        configureCombo(midiInputLabel, midiInputCombo, "MIDI Input");
        configureCombo(sampleRateLabel, sampleRateCombo, "Sample Rate");
        configureCombo(bufferSizeLabel, bufferSizeCombo, "Buffer Size");

        driverTypeCombo.onChange = [this]
        {
            if (!syncing)
                applyDriverTypeSelection();
        };

        outputDeviceCombo.onChange = [this]
        {
            if (!syncing)
                applyOutputDeviceSelection();
        };

        midiInputCombo.onChange = [this]
        {
            if (!syncing)
                applyMidiInputSelectionChange();
        };

        sampleRateCombo.onChange = [this]
        {
            if (!syncing)
                applyFormatSelection();
        };

        bufferSizeCombo.onChange = [this]
        {
            if (!syncing)
                applyFormatSelection();
        };

        refreshButton.setButtonText("Refresh");
        refreshButton.onClick = [this]
        {
            if (refreshStatus != nullptr)
                refreshStatus();
        };
        addAndMakeVisible(refreshButton);

        applyButton.setButtonText("Apply");
        applyButton.onClick = [this] { applyCurrentSelection(); };
        addAndMakeVisible(applyButton);

        footerLabel.setColour(juce::Label::textColourId, juce::Colour::fromRGB(156, 199, 239));
        footerLabel.setJustificationType(juce::Justification::topLeft);
        footerLabel.setText("Playback uses the shared JUCE host output device. MIDI input selection controls live audition and note insert.",
                            juce::dontSendNotification);
        addAndMakeVisible(footerLabel);
    }

    void applyHostStatus(const juce::var& status)
    {
        auto* object = status.getDynamicObject();
        if (object == nullptr)
        {
            setStatusMessage("Could not read native audio device status.", true);
            return;
        }

        const auto driverType = object->getProperty("audio_device_type").toString().trim();
        const auto outputDevice = object->getProperty("audio_device_name").toString().trim();
        const auto sampleRate = juce::roundToInt(static_cast<double>(object->getProperty("sample_rate")));
        const auto bufferSize = static_cast<int>(object->getProperty("buffer_size"));

        summaryLabel.setText("Driver: " + (driverType.isNotEmpty() ? driverType : "Default")
                                 + "\nDevice: " + (outputDevice.isNotEmpty() ? outputDevice : "Unavailable")
                                 + "\nFormat: " + juce::String(juce::jmax(1, sampleRate)) + " Hz"
                                 + "  |  " + juce::String(juce::jmax(1, bufferSize)) + " samples",
                             juce::dontSendNotification);

        syncing = true;
        populateStringCombo(driverTypeCombo,
                            object->getProperty("available_audio_device_types"),
                            driverType,
                            true);
        populateStringCombo(outputDeviceCombo,
                            object->getProperty("available_audio_output_devices"),
                            outputDevice,
                            false);
        populateIntCombo(sampleRateCombo,
                         object->getProperty("available_audio_sample_rates"),
                         sampleRate,
                         " Hz");
        populateIntCombo(bufferSizeCombo,
                         object->getProperty("available_audio_buffer_sizes"),
                         bufferSize,
                         " samples");
        syncing = false;

        appliedDriverType = driverType;
        appliedOutputDevice = outputDevice;
        appliedSampleRate = juce::jmax(0, sampleRate);
        appliedBufferSize = juce::jmax(0, bufferSize);

        setStatusMessage("Native audio settings ready.", false);
    }

    void applyAudioDeviceSnapshot(const NativeVstHostSession::AudioDeviceSnapshot& snapshot)
    {
        const auto sampleRate = juce::roundToInt(snapshot.sampleRate);
        const auto bufferSize = snapshot.bufferSize;

        summaryLabel.setText("Driver: " + (snapshot.audioDeviceType.isNotEmpty() ? snapshot.audioDeviceType : "Default")
                                 + "\nDevice: " + (snapshot.audioDeviceName.isNotEmpty() ? snapshot.audioDeviceName : "Unavailable")
                                 + "\nFormat: " + juce::String(juce::jmax(1, sampleRate)) + " Hz"
                                 + "  |  " + juce::String(juce::jmax(1, bufferSize)) + " samples",
                             juce::dontSendNotification);

        syncing = true;
        populateStringCombo(driverTypeCombo, snapshot.availableAudioDeviceTypes, snapshot.audioDeviceType, true);
        populateStringCombo(outputDeviceCombo, snapshot.availableAudioOutputDevices, snapshot.audioDeviceName, false);
        populateIntCombo(sampleRateCombo, snapshot.availableAudioSampleRates, sampleRate, " Hz");
        populateIntCombo(bufferSizeCombo, snapshot.availableAudioBufferSizes, bufferSize, " samples");
        syncing = false;

        appliedDriverType = snapshot.audioDeviceType;
        appliedOutputDevice = snapshot.audioDeviceName;
        appliedSampleRate = juce::jmax(0, sampleRate);
        appliedBufferSize = juce::jmax(0, bufferSize);

        setStatusMessage("Native audio settings ready.", false);
    }

    void applyMidiInputSnapshot(const juce::Array<juce::MidiDeviceInfo>& devices,
                                const juce::String& selectedIdentifier)
    {
        syncing = true;
        midiInputCombo.clear(juce::dontSendNotification);
        midiInputIdentifiers.clear();

        int itemId = 1;
        int selectedId = 0;

        midiInputCombo.addItem("All MIDI Inputs", itemId);
        midiInputIdentifiers.add({});
        if (selectedIdentifier.isEmpty())
            selectedId = itemId;
        ++itemId;

        midiInputCombo.addItem("No MIDI Input", itemId);
        midiInputIdentifiers.add(kMidiInputSelectionDisabled);
        if (selectedIdentifier == kMidiInputSelectionDisabled)
            selectedId = itemId;
        ++itemId;

        for (const auto& device : devices)
        {
            const auto label = midiInputDisplayName(device);
            if (label.isEmpty())
                continue;

            midiInputCombo.addItem(label, itemId);
            midiInputIdentifiers.add(device.identifier);
            if (device.identifier == selectedIdentifier)
                selectedId = itemId;
            ++itemId;
        }

        if (selectedId == 0 && selectedIdentifier.isNotEmpty() && selectedIdentifier != kMidiInputSelectionDisabled)
        {
            midiInputCombo.addItem("Unavailable MIDI Device", itemId);
            midiInputIdentifiers.add(selectedIdentifier);
            selectedId = itemId;
        }

        if (selectedId != 0)
            midiInputCombo.setSelectedId(selectedId, juce::dontSendNotification);
        else if (midiInputCombo.getNumItems() > 0)
            midiInputCombo.setSelectedItemIndex(0, juce::dontSendNotification);

        appliedMidiInputIdentifier = selectedIdentifier;
        syncing = false;
    }

    void setStatusMessage(const juce::String& message, bool isError)
    {
        footerLabel.setColour(juce::Label::textColourId,
                              isError ? juce::Colour::fromRGB(244, 144, 144)
                                      : juce::Colour::fromRGB(156, 199, 239));
        footerLabel.setText(message, juce::dontSendNotification);
    }

    void paint(juce::Graphics& g) override
    {
        g.fillAll(juce::Colour::fromRGB(13, 15, 20));
        g.setColour(juce::Colour::fromRGB(31, 35, 44));
        g.drawRect(getLocalBounds(), 1);
    }

    void resized() override
    {
        auto area = getLocalBounds().reduced(14);
        summaryLabel.setBounds(area.removeFromTop(72));
        area.removeFromTop(10);
        layoutRow(area, driverTypeLabel, driverTypeCombo);
        area.removeFromTop(8);
        layoutRow(area, outputDeviceLabel, outputDeviceCombo);
        area.removeFromTop(8);
        layoutRow(area, midiInputLabel, midiInputCombo);
        area.removeFromTop(8);
        layoutRow(area, sampleRateLabel, sampleRateCombo);
        area.removeFromTop(8);
        layoutRow(area, bufferSizeLabel, bufferSizeCombo);
        area.removeFromTop(12);

        auto buttonRow = area.removeFromTop(30);
        refreshButton.setBounds(buttonRow.removeFromLeft(96));
        buttonRow.removeFromLeft(8);
        applyButton.setBounds(buttonRow.removeFromLeft(96));
        area.removeFromTop(10);
        footerLabel.setBounds(area.removeFromTop(40));
    }

private:
    void configureCombo(juce::Label& label, juce::ComboBox& combo, const juce::String& text)
    {
        label.setText(text, juce::dontSendNotification);
        label.setColour(juce::Label::textColourId, juce::Colour::fromRGB(214, 219, 227));
        addAndMakeVisible(label);
        addAndMakeVisible(combo);
    }

    void layoutRow(juce::Rectangle<int>& area, juce::Label& label, juce::ComboBox& combo)
    {
        auto row = area.removeFromTop(28);
        label.setBounds(row.removeFromLeft(120));
        combo.setBounds(row);
    }

    void populateStringCombo(juce::ComboBox& combo,
                             const juce::var& valuesVar,
                             const juce::String& selectedValue,
                             bool includeAutoItem)
    {
        combo.clear(juce::dontSendNotification);
        int itemId = 1;
        int selectedId = 0;

        if (includeAutoItem)
        {
            combo.addItem("Auto / Default", itemId);
            if (selectedValue.isEmpty())
                selectedId = itemId;
            ++itemId;
        }

        if (auto* values = valuesVar.getArray())
        {
            for (const auto& valueVar : *values)
            {
                const auto text = valueVar.toString().trim();
                if (text.isEmpty())
                    continue;
                combo.addItem(text, itemId);
                if (text.equalsIgnoreCase(selectedValue))
                    selectedId = itemId;
                ++itemId;
            }
        }

        if (selectedId != 0)
            combo.setSelectedId(selectedId, juce::dontSendNotification);
        else if (combo.getNumItems() > 0)
            combo.setSelectedItemIndex(0, juce::dontSendNotification);
    }

    void populateStringCombo(juce::ComboBox& combo,
                             const juce::StringArray& values,
                             const juce::String& selectedValue,
                             bool includeAutoItem)
    {
        combo.clear(juce::dontSendNotification);
        int itemId = 1;
        int selectedId = 0;

        if (includeAutoItem)
        {
            combo.addItem("Auto / Default", itemId);
            if (selectedValue.isEmpty())
                selectedId = itemId;
            ++itemId;
        }

        for (const auto& text : values)
        {
            if (text.trim().isEmpty())
                continue;
            combo.addItem(text, itemId);
            if (text.equalsIgnoreCase(selectedValue))
                selectedId = itemId;
            ++itemId;
        }

        if (selectedId != 0)
            combo.setSelectedId(selectedId, juce::dontSendNotification);
        else if (combo.getNumItems() > 0)
            combo.setSelectedItemIndex(0, juce::dontSendNotification);
    }

    void populateIntCombo(juce::ComboBox& combo,
                          const juce::var& valuesVar,
                          int selectedValue,
                          const juce::String& suffix)
    {
        combo.clear(juce::dontSendNotification);
        int selectedId = 0;

        if (auto* values = valuesVar.getArray())
        {
            for (const auto& valueVar : *values)
            {
                const auto value = static_cast<int>(valueVar);
                if (value <= 0)
                    continue;
                combo.addItem(juce::String(value) + suffix, value);
                if (value == selectedValue)
                    selectedId = value;
            }
        }

        if (selectedId != 0)
            combo.setSelectedId(selectedId, juce::dontSendNotification);
        else if (combo.getNumItems() > 0)
            combo.setSelectedItemIndex(0, juce::dontSendNotification);
    }

    template <typename ValueType>
    void populateIntCombo(juce::ComboBox& combo,
                          const juce::Array<ValueType>& values,
                          int selectedValue,
                          const juce::String& suffix)
    {
        combo.clear(juce::dontSendNotification);
        int selectedId = 0;

        for (int index = 0; index < values.size(); ++index)
        {
            const auto value = static_cast<int>(std::llround(static_cast<double>(values.getReference(index))));
            if (value <= 0)
                continue;
            combo.addItem(juce::String(value) + suffix, value);
            if (value == selectedValue)
                selectedId = value;
        }

        if (selectedId != 0)
            combo.setSelectedId(selectedId, juce::dontSendNotification);
        else if (combo.getNumItems() > 0)
            combo.setSelectedItemIndex(0, juce::dontSendNotification);
    }

    int selectedIntValue(const juce::ComboBox& combo) const
    {
        return combo.getSelectedId();
    }

    juce::String selectedDriverType() const
    {
        const auto selected = driverTypeCombo.getText().trim();
        if (selected == "Auto / Default")
            return {};
        return selected;
    }

    void applyDriverTypeSelection()
    {
        if (applySettings == nullptr)
            return;

        const auto requestedType = selectedDriverType();
        if (requestedType.isEmpty() || requestedType.equalsIgnoreCase(appliedDriverType))
            return;

        const auto result = applySettings(requestedType, {}, 0, 0);
        setStatusMessage(result.wasOk() ? "Switched audio driver backend. Choose a device if needed."
                                        : result.getErrorMessage(),
                         result.failed());
    }

    void applyOutputDeviceSelection()
    {
        if (applySettings == nullptr)
            return;

        const auto requestedOutput = outputDeviceCombo.getText().trim();
        if (requestedOutput.isEmpty() || requestedOutput.equalsIgnoreCase(appliedOutputDevice))
            return;

        const auto result = applySettings({}, requestedOutput, 0, 0);
        setStatusMessage(result.wasOk() ? "Switched audio output device. Choose sample rate or buffer if needed."
                                        : result.getErrorMessage(),
                         result.failed());
    }

    juce::String selectedMidiInputIdentifier() const
    {
        const auto index = midiInputCombo.getSelectedItemIndex();
        if (!juce::isPositiveAndBelow(index, midiInputIdentifiers.size()))
            return appliedMidiInputIdentifier;
        return midiInputIdentifiers[index];
    }

    void applyMidiInputSelectionChange()
    {
        if (applyMidiInputSetting == nullptr)
            return;

        const auto requestedIdentifier = selectedMidiInputIdentifier();
        if (requestedIdentifier == appliedMidiInputIdentifier)
            return;

        const auto result = applyMidiInputSetting(requestedIdentifier);
        setStatusMessage(result.wasOk() ? "Updated MIDI input selection."
                                        : result.getErrorMessage(),
                         result.failed());
    }

    void applyFormatSelection()
    {
        if (applySettings == nullptr)
            return;

        const auto selectedRate = selectedIntValue(sampleRateCombo);
        const auto selectedBuffer = selectedIntValue(bufferSizeCombo);
        if (selectedRate <= 0 && selectedBuffer <= 0)
            return;
        if (selectedRate == appliedSampleRate && selectedBuffer == appliedBufferSize)
            return;

        const auto result = applySettings({}, {}, selectedRate, selectedBuffer);
        setStatusMessage(result.wasOk() ? "Updated native sample rate and buffer size."
                                        : result.getErrorMessage(),
                         result.failed());
    }

    void applyCurrentSelection()
    {
        if (applySettings == nullptr)
            return;

        bool changed = false;

        const auto requestedType = selectedDriverType();
        if (requestedType.isNotEmpty() && !requestedType.equalsIgnoreCase(appliedDriverType))
        {
            const auto result = applySettings(requestedType, {}, 0, 0);
            if (result.failed())
            {
                setStatusMessage(result.getErrorMessage(), true);
                return;
            }
            changed = true;
        }

        const auto requestedOutput = outputDeviceCombo.getText().trim();
        if (requestedOutput.isNotEmpty() && !requestedOutput.equalsIgnoreCase(appliedOutputDevice))
        {
            const auto result = applySettings({}, requestedOutput, 0, 0);
            if (result.failed())
            {
                setStatusMessage(result.getErrorMessage(), true);
                return;
            }
            changed = true;
        }

        const auto requestedMidiInput = selectedMidiInputIdentifier();
        if (requestedMidiInput != appliedMidiInputIdentifier)
        {
            if (applyMidiInputSetting == nullptr)
            {
                setStatusMessage("MIDI input settings are unavailable.", true);
                return;
            }

            const auto result = applyMidiInputSetting(requestedMidiInput);
            if (result.failed())
            {
                setStatusMessage(result.getErrorMessage(), true);
                return;
            }
            changed = true;
        }

        const auto selectedRate = selectedIntValue(sampleRateCombo);
        const auto selectedBuffer = selectedIntValue(bufferSizeCombo);
        if ((selectedRate > 0 || selectedBuffer > 0)
            && (selectedRate != appliedSampleRate || selectedBuffer != appliedBufferSize))
        {
            const auto result = applySettings({}, {}, selectedRate, selectedBuffer);
            if (result.failed())
            {
                setStatusMessage(result.getErrorMessage(), true);
                return;
            }
            changed = true;
        }

        setStatusMessage(changed ? "Updated native audio settings."
                                 : "Audio settings already match the current driver.",
                         false);
    }

    std::function<juce::Result(const juce::String&, const juce::String&, int, int)> applySettings;
    std::function<juce::Result(const juce::String&)> applyMidiInputSetting;
    std::function<void()> refreshStatus;
    bool syncing = false;
    juce::String appliedDriverType;
    juce::String appliedOutputDevice;
    juce::String appliedMidiInputIdentifier;
    int appliedSampleRate = 0;
    int appliedBufferSize = 0;
    juce::StringArray midiInputIdentifiers;

    juce::Label summaryLabel;
    juce::Label driverTypeLabel;
    juce::ComboBox driverTypeCombo;
    juce::Label outputDeviceLabel;
    juce::ComboBox outputDeviceCombo;
    juce::Label midiInputLabel;
    juce::ComboBox midiInputCombo;
    juce::Label sampleRateLabel;
    juce::ComboBox sampleRateCombo;
    juce::Label bufferSizeLabel;
    juce::ComboBox bufferSizeCombo;
    juce::TextButton refreshButton;
    juce::TextButton applyButton;
    juce::Label footerLabel;
};

class AudioWorkspaceWindowComponent final : public juce::Component
{
public:
    AudioWorkspaceWindowComponent(std::function<void()> jumpToStartIn,
                                  std::function<void()> playProjectIn,
                                  std::function<void()> playTrackIn,
                                  std::function<void()> stopPlaybackIn,
                                  std::function<void()> openAudioSettingsIn,
                                  std::function<void(int)> setPlayheadTickIn,
                                  std::function<void(int)> setLeftLocatorTickIn,
                                  std::function<void(int)> setRightLocatorTickIn,
                                  std::function<void(int)> setTempoIn,
                                  std::function<void(bool)> setLoopEnabledIn,
                                  std::function<void(bool)> setMetronomeEnabledIn,
                                  std::function<void(bool)> setRecordEnabledIn,
                                  std::function<void()> exportMixIn,
                                  std::function<void()> exportStemsIn,
                                  MixerComponent::ProjectGetter projectGetterIn,
                                  MixerComponent::TrackWriter trackWriterIn,
                                  MixerComponent::ProjectWriter projectWriterIn,
                                  MixerComponent::MeterGetter meterGetterIn,
                                  MixerComponent::MasterMeterGetter masterMeterGetterIn,
                                  std::function<juce::String()> audioSummaryGetterIn)
        : audioSummaryGetter(std::move(audioSummaryGetterIn))
    {
        titleLabel.setText("Audio", juce::dontSendNotification);
        titleLabel.setFont(ui::titleFont());
        titleLabel.setColour(juce::Label::textColourId, juce::Colour::fromRGB(235, 239, 244));
        addAndMakeVisible(titleLabel);

        configureButton(audioSettingsButton, "Audio Settings");
        audioSettingsButton.onClick = [openAudioSettings = std::move(openAudioSettingsIn)]
        {
            if (openAudioSettings != nullptr)
                openAudioSettings();
        };

        configureButton(exportMixButton, "Export Mix");
        exportMixButton.onClick = [exportMix = std::move(exportMixIn)]
        {
            if (exportMix != nullptr)
                exportMix();
        };

        configureButton(exportStemsButton, "Export Stems");
        exportStemsButton.onClick = [exportStems = std::move(exportStemsIn)]
        {
            if (exportStems != nullptr)
                exportStems();
        };

        audioSummaryLabel.setColour(juce::Label::textColourId, juce::Colour::fromRGB(190, 199, 210));
        audioSummaryLabel.setJustificationType(juce::Justification::centredLeft);
        addAndMakeVisible(audioSummaryLabel);

        transportPanel = std::make_unique<TransportPanelComponent>(std::move(jumpToStartIn),
                                                                   std::move(playProjectIn),
                                                                   std::move(playTrackIn),
                                                                   std::move(stopPlaybackIn),
                                                                   std::move(setPlayheadTickIn),
                                                                   std::move(setLeftLocatorTickIn),
                                                                   std::move(setRightLocatorTickIn),
                                                                   std::move(setTempoIn),
                                                                   std::move(setLoopEnabledIn),
                                                                   std::move(setMetronomeEnabledIn),
                                                                   std::move(setRecordEnabledIn));
        addAndMakeVisible(*transportPanel);

        mixer = std::make_unique<MixerComponent>(std::move(projectGetterIn),
                                                 std::move(trackWriterIn),
                                                 std::move(projectWriterIn),
                                                 std::move(meterGetterIn),
                                                 std::move(masterMeterGetterIn));
        mixerViewport.setViewedComponent(mixer.get(), false);
        mixerViewport.setScrollBarsShown(true, false);
        addAndMakeVisible(mixerViewport);
    }

    void refreshFromModel(const ProjectState& project,
                          bool hasTrackSelection,
                          bool rackPlaying,
                          bool projectPlaying,
                          bool recordEnabled,
                          const juce::String& statusText,
                          double cpuUsagePercent,
                          float masterPeakLeft,
                          float masterPeakRight,
                          bool deferAudioSummaryRefresh = false)
    {
        if (transportPanel != nullptr)
            transportPanel->refreshFromState(project,
                                             hasTrackSelection,
                                             rackPlaying,
                                             projectPlaying,
                                             recordEnabled,
                                             statusText,
                                             cpuUsagePercent,
                                             masterPeakLeft,
                                             masterPeakRight);

        if (mixer != nullptr)
        {
            mixer->refreshFromModel();
            mixer->refreshMeters();
        }

        if (!deferAudioSummaryRefresh)
        {
            audioSummaryLabel.setText(audioSummaryGetter != nullptr
                                          ? audioSummaryGetter()
                                          : juce::String("Audio settings unavailable."),
                                      juce::dontSendNotification);
        }
    }

    void focusWorkspace()
    {
        if (transportPanel != nullptr)
            transportPanel->grabKeyboardFocus();
    }

    void paint(juce::Graphics& g) override
    {
        g.fillAll(juce::Colour::fromRGB(13, 15, 20));
        g.setColour(juce::Colour::fromRGB(31, 35, 44));
        g.drawRect(getLocalBounds(), 1);
    }

    void resized() override
    {
        auto area = getLocalBounds().reduced(14);
        auto header = area.removeFromTop(30);
        titleLabel.setBounds(header.removeFromLeft(90));
        audioSettingsButton.setBounds(header.removeFromLeft(120));
        header.removeFromLeft(6);
        exportMixButton.setBounds(header.removeFromLeft(100));
        header.removeFromLeft(6);
        exportStemsButton.setBounds(header.removeFromLeft(110));
        header.removeFromLeft(10);
        audioSummaryLabel.setBounds(header);

        area.removeFromTop(8);
        if (transportPanel != nullptr)
            transportPanel->setBounds(area.removeFromTop(148));
        area.removeFromTop(10);
        mixerViewport.setBounds(area);
    }

private:
    void configureButton(juce::TextButton& button, const juce::String& text)
    {
        button.setButtonText(text);
        button.setColour(juce::TextButton::buttonColourId, juce::Colour::fromRGB(48, 54, 66));
        button.setColour(juce::TextButton::textColourOffId, juce::Colours::white);
        addAndMakeVisible(button);
    }

    std::function<juce::String()> audioSummaryGetter;
    juce::Label titleLabel;
    juce::TextButton audioSettingsButton;
    juce::TextButton exportMixButton;
    juce::TextButton exportStemsButton;
    juce::Label audioSummaryLabel;
    std::unique_ptr<TransportPanelComponent> transportPanel;
    juce::Viewport mixerViewport;
    std::unique_ptr<MixerComponent> mixer;
};

class PanelsWindowComponent final : public juce::Component
{
public:
    PanelsWindowComponent(ArrangementOverviewComponent::ProjectGetter projectGetter,
                          ArrangementOverviewComponent::SelectedSectionGetter selectedSectionGetter,
                          ArrangementOverviewComponent::SectionSelectCallback sectionSelectCallback,
                          AutomationEditorComponent::SelectedTrackIndexGetter selectedTrackIndexGetter,
                          AutomationEditorComponent::TrackWriter trackWriter,
                          ArrangementOverviewComponent::ProjectWriter projectWriter,
                          std::function<EditorToolMode()> toolModeGetterIn,
                          std::function<void(EditorToolMode)> setToolModeIn)
        : projectGetter(std::move(projectGetter)),
          projectWriter(std::move(projectWriter)),
          toolModeGetter(std::move(toolModeGetterIn)),
          setToolMode(std::move(setToolModeIn)),
          tabs(juce::TabbedButtonBar::TabsAtTop)
    {
        tabs.setTabBarDepth(30);
        addAndMakeVisible(tabs);

        keyQuantizeLabel.setText("Key Quantize", juce::dontSendNotification);
        keyQuantizeLabel.setColour(juce::Label::textColourId, juce::Colour::fromRGB(214, 219, 227));
        addAndMakeVisible(keyQuantizeLabel);

        for (const auto& option : keyQuantizeOptions())
            keyQuantizeBox.addItem(option.label, option.id);
        keyQuantizeBox.setTextWhenNothingSelected("All Notes");
        keyQuantizeBox.onChange = [this]
        {
            if (syncingKeyQuantizeControls)
                return;

            const auto* option = findKeyQuantizeOptionById(keyQuantizeBox.getSelectedId());
            if (option == nullptr)
                return;

            auto updatedProject = this->projectGetter();
            const auto currentScale = normaliseKeyQuantizeScale(updatedProject.keyQuantizeScale);
            const auto currentRoot = juce::negativeAwareModulo(updatedProject.keyQuantizeRoot, 12);
            if (currentRoot == option->root && currentScale.equalsIgnoreCase(option->scaleId))
                return;

            updatedProject.keyQuantizeRoot = option->root;
            updatedProject.keyQuantizeScale = option->scaleId;
            updatedProject.recalculateTimeFields();
            this->projectWriter(updatedProject, true, "Change Key Quantize");
        };
        addAndMakeVisible(keyQuantizeBox);

        noteRangeLabel.setText("Range", juce::dontSendNotification);
        noteRangeLabel.setColour(juce::Label::textColourId, juce::Colour::fromRGB(214, 219, 227));
        addAndMakeVisible(noteRangeLabel);

        for (const auto& option : pianoRollPitchOptions())
        {
            noteRangeMinBox.addItem(option.label, option.id);
            noteRangeMaxBox.addItem(option.label, option.id);
        }
        noteRangeMinBox.setTextWhenNothingSelected("Low");
        noteRangeMaxBox.setTextWhenNothingSelected("High");
        noteRangeMinBox.setJustificationType(juce::Justification::centredLeft);
        noteRangeMaxBox.setJustificationType(juce::Justification::centredLeft);
        noteRangeMinBox.onChange = [this] { handlePianoRollRangeChanged(true); };
        noteRangeMaxBox.onChange = [this] { handlePianoRollRangeChanged(false); };
        addAndMakeVisible(noteRangeMinBox);

        noteRangeSeparatorLabel.setText("to", juce::dontSendNotification);
        noteRangeSeparatorLabel.setJustificationType(juce::Justification::centred);
        noteRangeSeparatorLabel.setColour(juce::Label::textColourId, juce::Colour::fromRGB(190, 199, 210));
        addAndMakeVisible(noteRangeSeparatorLabel);
        addAndMakeVisible(noteRangeMaxBox);

        auto arrangementProjectGetter = this->projectGetter;
        auto arrangementProjectWriter = this->projectWriter;
        auto arrangementSelectedSectionGetter = selectedSectionGetter;
        auto arrangementSectionSelectCallback = sectionSelectCallback;
        auto* arrangement = new ArrangementOverviewComponent(arrangementProjectGetter,
                                                             arrangementProjectWriter,
                                                             arrangementSelectedSectionGetter,
                                                             arrangementSectionSelectCallback,
                                                             [this] { return toolModeGetter != nullptr ? toolModeGetter() : EditorToolMode::pencil; },
                                                             [this] (EditorToolMode mode)
                                                             {
                                                                 if (setToolMode != nullptr)
                                                                     setToolMode(mode);
                                                             });
        arrangement->setZoomChangedCallback([this] (float pixelsPerBar)
                                            {
                                                if (arrangementZoomChangedCallback != nullptr)
                                                    arrangementZoomChangedCallback(pixelsPerBar);
                                            });
        arrangement->setLaneHeightChangedCallback([this] (float laneHeightPixels)
                                                  {
                                                      if (arrangementLaneHeightChangedCallback != nullptr)
                                                          arrangementLaneHeightChangedCallback(laneHeightPixels);
                                                  });
        arrangementView = arrangement;
        tabs.addTab("Arrangement",
                    juce::Colour::fromRGB(26, 35, 52),
                    arrangement,
                    true);

        auto automationProjectGetter = this->projectGetter;
        auto automationSelectedTrackIndexGetter = selectedTrackIndexGetter;
        auto automationTrackWriter = trackWriter;
        auto* automation = new AutomationEditorComponent(automationProjectGetter,
                                                         automationSelectedTrackIndexGetter,
                                                         automationTrackWriter);
        automationView = automation;
        tabs.addTab("Automation",
                    juce::Colour::fromRGB(26, 35, 52),
                    automation,
                    true);

        auto sampleProjectGetter = this->projectGetter;
        auto sampleProjectWriter = this->projectWriter;
        auto* samples = new SampleTimelineComponent(sampleProjectGetter,
                                                    sampleProjectWriter);
        sampleTimelineView = samples;
        tabs.addTab("Samples",
                    juce::Colour::fromRGB(26, 35, 52),
                    samples,
                    true);

        auto pianoProjectGetter = this->projectGetter;
        auto pianoTrackIndexGetter = selectedTrackIndexGetter;
        auto pianoSelectedSectionGetter = selectedSectionGetter;
        auto pianoProjectWriter = this->projectWriter;
        auto* piano = new PianoRollComponent(pianoProjectGetter,
                                             pianoTrackIndexGetter,
                                             pianoSelectedSectionGetter,
                                             pianoProjectWriter);
        piano->setToolModeChangeCallback(setToolMode);
        piano->setZoomChangedCallback([this] (float pixelsPerBeat)
                                      {
                                          if (pianoRollZoomChangedCallback != nullptr)
                                              pianoRollZoomChangedCallback(pixelsPerBeat);
                                      });
        piano->setRowHeightChangedCallback([this] (float rowHeightPixels)
                                           {
                                               if (pianoRollRowHeightChangedCallback != nullptr)
                                                   pianoRollRowHeightChangedCallback(rowHeightPixels);
                                           });
        pianoRollView = piano;
        pianoRollViewport.setViewedComponent(piano, true);
        pianoRollViewport.setScrollBarsShown(true, true);
        tabs.addTab("Piano Roll",
                    juce::Colour::fromRGB(26, 35, 52),
                    &pianoRollViewport,
                    false);
    }

    void setNotePreviewCallbacks(PianoRollComponent::NotePreviewCallback noteOnCallback,
                                 PianoRollComponent::NotePreviewCallback noteOffCallback,
                                 PianoRollComponent::PreviewStopCallback stopPreviewCallback = {})
    {
        if (pianoRollView != nullptr)
        {
            pianoRollView->setNotePreviewCallbacks(std::move(noteOnCallback),
                                                   std::move(noteOffCallback),
                                                   std::move(stopPreviewCallback));
        }
    }

    void setKeyHandlerCallback(PianoRollComponent::KeyHandlerCallback keyHandlerCallback)
    {
        if (pianoRollView != nullptr)
            pianoRollView->setKeyHandlerCallback(std::move(keyHandlerCallback));
    }

    void setArrangementZoomChangedCallback(std::function<void(float)> arrangementZoomChangedCallbackIn)
    {
        arrangementZoomChangedCallback = std::move(arrangementZoomChangedCallbackIn);
        if (arrangementView != nullptr)
            arrangementView->setZoomChangedCallback(arrangementZoomChangedCallback);
    }

    void setArrangementLaneHeightChangedCallback(std::function<void(float)> arrangementLaneHeightChangedCallbackIn)
    {
        arrangementLaneHeightChangedCallback = std::move(arrangementLaneHeightChangedCallbackIn);
        if (arrangementView != nullptr)
            arrangementView->setLaneHeightChangedCallback(arrangementLaneHeightChangedCallback);
    }

    void setPianoRollZoomChangedCallback(std::function<void(float)> pianoRollZoomChangedCallbackIn)
    {
        pianoRollZoomChangedCallback = std::move(pianoRollZoomChangedCallbackIn);
        if (pianoRollView != nullptr)
            pianoRollView->setZoomChangedCallback(pianoRollZoomChangedCallback);
    }

    void setPianoRollRowHeightChangedCallback(std::function<void(float)> pianoRollRowHeightChangedCallbackIn)
    {
        pianoRollRowHeightChangedCallback = std::move(pianoRollRowHeightChangedCallbackIn);
        if (pianoRollView != nullptr)
            pianoRollView->setRowHeightChangedCallback(pianoRollRowHeightChangedCallback);
    }

    void setArrangementViewScale(float pixelsPerBar, float laneHeightPixels)
    {
        if (arrangementView != nullptr)
        {
            arrangementView->setHorizontalZoom(pixelsPerBar);
            arrangementView->setLaneHeight(laneHeightPixels);
        }
    }

    void setPianoRollViewScale(float pixelsPerBeat, float rowHeightPixels)
    {
        if (pianoRollView != nullptr)
        {
            pianoRollView->setHorizontalZoom(pixelsPerBeat);
            pianoRollView->setNoteRowHeight(rowHeightPixels);
        }
    }

    void refreshFromModel()
    {
        syncingKeyQuantizeControls = true;
        const auto& currentProject = projectGetter();
        keyQuantizeBox.setSelectedId(keyQuantizeOptionId(currentProject), juce::dontSendNotification);
        noteRangeMinBox.setSelectedId(pianoRollPitchOptionId(currentProject.pianoRollVisiblePitchMin), juce::dontSendNotification);
        noteRangeMaxBox.setSelectedId(pianoRollPitchOptionId(currentProject.pianoRollVisiblePitchMax), juce::dontSendNotification);
        syncingKeyQuantizeControls = false;

        if (arrangementView != nullptr)
            arrangementView->refreshFromModel();
        if (automationView != nullptr)
            automationView->refreshFromModel();
        if (sampleTimelineView != nullptr)
            sampleTimelineView->refreshFromModel();
        if (pianoRollView != nullptr)
        {
            if (toolModeGetter != nullptr)
                pianoRollView->setToolMode(toolModeGetter());
            pianoRollView->refreshFromModel();
            focusInitialPianoRollViewportIfNeeded();
        }
    }

    void refreshMidiEditState()
    {
        syncingKeyQuantizeControls = true;
        const auto& currentProject = projectGetter();
        keyQuantizeBox.setSelectedId(keyQuantizeOptionId(currentProject), juce::dontSendNotification);
        noteRangeMinBox.setSelectedId(pianoRollPitchOptionId(currentProject.pianoRollVisiblePitchMin), juce::dontSendNotification);
        noteRangeMaxBox.setSelectedId(pianoRollPitchOptionId(currentProject.pianoRollVisiblePitchMax), juce::dontSendNotification);
        syncingKeyQuantizeControls = false;

        if (arrangementView != nullptr)
            arrangementView->refreshFromModel();

        if (pianoRollView != nullptr)
        {
            if (toolModeGetter != nullptr)
                pianoRollView->setToolMode(toolModeGetter());
            pianoRollView->refreshFromModel();
            focusInitialPianoRollViewportIfNeeded();
        }
    }

    void showArrangementTab()
    {
        tabs.setCurrentTabIndex(0);
        if (arrangementView != nullptr)
            arrangementView->grabKeyboardFocus();
    }

    void showAutomationTab()
    {
        tabs.setCurrentTabIndex(1);
        if (automationView != nullptr)
            automationView->grabKeyboardFocus();
    }

    void showSamplesTab()
    {
        tabs.setCurrentTabIndex(2);
        if (sampleTimelineView != nullptr)
            sampleTimelineView->grabKeyboardFocus();
    }

    void showPianoRollTab()
    {
        tabs.setCurrentTabIndex(3);
        if (pianoRollView != nullptr)
            pianoRollView->grabKeyboardFocus();
    }

    bool selectAllNotes()
    {
        return pianoRollView != nullptr && pianoRollView->selectAllNotes();
    }

    bool selectAllSections()
    {
        return arrangementView != nullptr && arrangementView->selectAllSections();
    }

    std::vector<int> getSelectedSectionIndices() const
    {
        return arrangementView != nullptr ? arrangementView->getSelectedSectionIndices() : std::vector<int>{};
    }

    bool hasArrangementKeyboardFocus() const
    {
        return arrangementView != nullptr && arrangementView->hasKeyboardFocus(true);
    }

    bool hasPianoRollKeyboardFocus() const
    {
        return pianoRollView != nullptr && pianoRollView->hasKeyboardFocus(true);
    }

    void paint(juce::Graphics& g) override
    {
        g.fillAll(juce::Colour::fromRGB(13, 15, 20));
    }

    void resized() override
    {
        tabs.setBounds(getLocalBounds());

        auto header = getLocalBounds().removeFromTop(tabs.getTabBarDepth()).reduced(8, 3);
        noteRangeMaxBox.setBounds(header.removeFromRight(92));
        header.removeFromRight(6);
        noteRangeSeparatorLabel.setBounds(header.removeFromRight(20));
        header.removeFromRight(6);
        noteRangeMinBox.setBounds(header.removeFromRight(92));
        header.removeFromRight(8);
        noteRangeLabel.setBounds(header.removeFromRight(44));
        header.removeFromRight(14);
        keyQuantizeBox.setBounds(header.removeFromRight(196));
        header.removeFromRight(8);
        keyQuantizeLabel.setBounds(header.removeFromRight(92));
        focusInitialPianoRollViewportIfNeeded();
    }

private:
    void handlePianoRollRangeChanged(bool minimumChanged)
    {
        if (syncingKeyQuantizeControls)
            return;

        const auto* minimumOption = findPianoRollPitchOptionById(noteRangeMinBox.getSelectedId());
        const auto* maximumOption = findPianoRollPitchOptionById(noteRangeMaxBox.getSelectedId());
        if (minimumOption == nullptr || maximumOption == nullptr)
            return;

        auto updatedProject = projectGetter();
        auto minimumPitch = minimumOption->pitch;
        auto maximumPitch = maximumOption->pitch;
        if (minimumPitch > maximumPitch)
        {
            if (minimumChanged)
                maximumPitch = minimumPitch;
            else
                minimumPitch = maximumPitch;
        }

        if (updatedProject.pianoRollVisiblePitchMin == minimumPitch
            && updatedProject.pianoRollVisiblePitchMax == maximumPitch)
        {
            return;
        }

        updatedProject.pianoRollVisiblePitchMin = minimumPitch;
        updatedProject.pianoRollVisiblePitchMax = maximumPitch;
        updatedProject.recalculateTimeFields();
        projectWriter(updatedProject, true, "Change Piano Roll Range");
    }

    void focusInitialPianoRollViewportIfNeeded()
    {
        if (!pianoRollInitialViewPending || pianoRollView == nullptr || pianoRollViewport.getHeight() <= 0)
            return;

        pianoRollViewport.setViewPosition(pianoRollViewport.getViewPositionX(),
                                          pianoRollView->viewPositionYForPitch(60, pianoRollViewport.getHeight()));
        pianoRollInitialViewPending = false;
    }

    ArrangementOverviewComponent::ProjectGetter projectGetter;
    ArrangementOverviewComponent::ProjectWriter projectWriter;
    std::function<EditorToolMode()> toolModeGetter;
    std::function<void(EditorToolMode)> setToolMode;
    std::function<void(float)> arrangementZoomChangedCallback;
    std::function<void(float)> arrangementLaneHeightChangedCallback;
    std::function<void(float)> pianoRollZoomChangedCallback;
    std::function<void(float)> pianoRollRowHeightChangedCallback;
    juce::TabbedComponent tabs;
    juce::Label keyQuantizeLabel;
    juce::ComboBox keyQuantizeBox;
    juce::Label noteRangeLabel;
    juce::ComboBox noteRangeMinBox;
    juce::Label noteRangeSeparatorLabel;
    juce::ComboBox noteRangeMaxBox;
    bool syncingKeyQuantizeControls = false;
    ArrangementOverviewComponent* arrangementView = nullptr;
    AutomationEditorComponent* automationView = nullptr;
    SampleTimelineComponent* sampleTimelineView = nullptr;
    juce::Viewport pianoRollViewport;
    PianoRollComponent* pianoRollView = nullptr;
    bool pianoRollInitialViewPending = true;
};

class SampleWorkspaceWindowComponent final : public juce::Component
{
public:
    using ProjectGetter = SampleTimelineComponent::ProjectGetter;
    using ProjectWriter = SampleTimelineComponent::ProjectWriter;

    SampleWorkspaceWindowComponent(ProjectGetter projectGetterIn,
                                   ProjectWriter projectWriterIn,
                                   std::function<int()> selectedAssetGetterIn,
                                   std::function<void(int)> selectedAssetSetterIn,
                                   std::function<void()> importSampleIn,
                                   std::function<void()> placeSampleIn,
                                   std::function<bool()> canPlaceSampleIn)
        : projectGetter(std::move(projectGetterIn)),
          selectedAssetGetter(std::move(selectedAssetGetterIn)),
          importSample(std::move(importSampleIn)),
          placeSample(std::move(placeSampleIn)),
          canPlaceSample(std::move(canPlaceSampleIn)),
          assetListModel(projectGetter, std::move(selectedAssetSetterIn))
    {
        titleLabel.setText("Samples", juce::dontSendNotification);
        titleLabel.setFont(ui::titleFont());
        titleLabel.setColour(juce::Label::textColourId, juce::Colour::fromRGB(235, 239, 244));
        addAndMakeVisible(titleLabel);

        configureButton(importButton, "Import Sample");
        importButton.onClick = [this]
        {
            if (importSample != nullptr)
                importSample();
        };

        configureButton(placeButton, "Place At Playhead");
        placeButton.onClick = [this]
        {
            if (placeSample != nullptr)
                placeSample();
        };

        libraryLabel.setText("Library", juce::dontSendNotification);
        libraryLabel.setColour(juce::Label::textColourId, juce::Colour::fromRGB(190, 199, 210));
        addAndMakeVisible(libraryLabel);

        timelineLabel.setText("Timeline", juce::dontSendNotification);
        timelineLabel.setColour(juce::Label::textColourId, juce::Colour::fromRGB(190, 199, 210));
        addAndMakeVisible(timelineLabel);

        assetList.setModel(&assetListModel);
        assetList.setRowHeight(28);
        assetList.setColour(juce::ListBox::backgroundColourId, juce::Colour::fromRGB(20, 22, 28));
        assetList.setOutlineThickness(1);
        addAndMakeVisible(assetList);

        timeline = std::make_unique<SampleTimelineComponent>(projectGetter,
                                                             std::move(projectWriterIn));
        timelineViewport.setViewedComponent(timeline.get(), false);
        timelineViewport.setScrollBarsShown(true, true);
        addAndMakeVisible(timelineViewport);
    }

    void refreshFromModel()
    {
        const auto selectedRow = selectedAssetGetter != nullptr ? selectedAssetGetter() : -1;
        assetList.updateContent();
        if (selectedRow >= 0)
            assetList.selectRow(selectedRow, false, true);
        else
            assetList.deselectAllRows();

        if (timeline != nullptr)
            timeline->refreshFromModel();

        placeButton.setEnabled(canPlaceSample != nullptr && canPlaceSample());
    }

    void focusLibrary()
    {
        assetList.grabKeyboardFocus();
    }

    void focusTimeline()
    {
        if (timeline != nullptr)
            timeline->grabKeyboardFocus();
    }

    void paint(juce::Graphics& g) override
    {
        g.fillAll(juce::Colour::fromRGB(13, 15, 20));
        g.setColour(juce::Colour::fromRGB(31, 35, 44));
        g.drawRect(getLocalBounds(), 1);
    }

    void resized() override
    {
        auto area = getLocalBounds().reduced(14);
        auto header = area.removeFromTop(30);
        titleLabel.setBounds(header.removeFromLeft(140));
        importButton.setBounds(header.removeFromLeft(120));
        header.removeFromLeft(8);
        placeButton.setBounds(header.removeFromLeft(146));

        area.removeFromTop(10);
        auto content = area;
        auto libraryArea = content.removeFromLeft(juce::jmin(320, juce::roundToInt(static_cast<float>(content.getWidth()) * 0.3f)));
        libraryLabel.setBounds(libraryArea.removeFromTop(24));
        libraryArea.removeFromTop(6);
        assetList.setBounds(libraryArea);

        content.removeFromLeft(10);
        timelineLabel.setBounds(content.removeFromTop(24));
        content.removeFromTop(6);
        timelineViewport.setBounds(content);
    }

private:
    class AssetListModel final : public juce::ListBoxModel
    {
    public:
        AssetListModel(ProjectGetter projectGetterIn,
                       std::function<void(int)> selectedAssetSetterIn)
            : projectGetter(std::move(projectGetterIn)),
              selectedAssetSetter(std::move(selectedAssetSetterIn))
        {
        }

        int getNumRows() override
        {
            return static_cast<int>(projectGetter().sampleAssets.size());
        }

        void paintListBoxItem(int rowNumber,
                              juce::Graphics& g,
                              int width,
                              int height,
                              bool rowIsSelected) override
        {
            juce::ignoreUnused(height);
            const auto& project = projectGetter();
            if (!juce::isPositiveAndBelow(rowNumber, static_cast<int>(project.sampleAssets.size())))
                return;

            const auto& asset = project.sampleAssets[static_cast<size_t>(rowNumber)];
            const auto text = juce::File(asset.path).getFileName()
                + "  (" + juce::String(asset.durationSec, 2) + " s)";

            g.fillAll(rowIsSelected ? juce::Colour::fromRGB(46, 88, 138)
                                    : ((rowNumber % 2) == 0 ? juce::Colour::fromRGB(26, 30, 37)
                                                            : juce::Colour::fromRGB(21, 25, 31)));
            g.setColour(rowIsSelected ? juce::Colours::white : juce::Colour::fromRGB(226, 232, 240));
            g.setFont(ui::font());
            g.drawText(text, 8, 0, width - 12, height, juce::Justification::centredLeft, true);
        }

        void selectedRowsChanged(int lastRowSelected) override
        {
            if (selectedAssetSetter != nullptr)
                selectedAssetSetter(lastRowSelected);
        }

    private:
        ProjectGetter projectGetter;
        std::function<void(int)> selectedAssetSetter;
    };

    class AssetListBox final : public juce::ListBox
    {
    public:
        explicit AssetListBox(ProjectGetter projectGetterIn)
            : projectGetter(std::move(projectGetterIn))
        {
        }

        void mouseDown(const juce::MouseEvent& event) override
        {
            dragTriggered = false;
            juce::ListBox::mouseDown(event);
        }

        void mouseDrag(const juce::MouseEvent& event) override
        {
            juce::ListBox::mouseDrag(event);

            if (dragTriggered || event.getDistanceFromDragStart() < 6)
                return;

            const auto row = getSelectedRow();
            const auto& project = projectGetter();
            if (!juce::isPositiveAndBelow(row, static_cast<int>(project.sampleAssets.size())))
                return;

            const juce::File file(project.sampleAssets[static_cast<size_t>(row)].path);
            if (!file.existsAsFile())
                return;

            dragTriggered = juce::DragAndDropContainer::performExternalDragDropOfFiles({ file.getFullPathName() },
                                                                                       false,
                                                                                       this);
        }

    private:
        ProjectGetter projectGetter;
        bool dragTriggered = false;
    };

    void configureButton(juce::TextButton& button, const juce::String& text)
    {
        button.setButtonText(text);
        button.setColour(juce::TextButton::buttonColourId, juce::Colour::fromRGB(48, 54, 66));
        button.setColour(juce::TextButton::textColourOffId, juce::Colours::white);
        addAndMakeVisible(button);
    }

    ProjectGetter projectGetter;
    std::function<int()> selectedAssetGetter;
    std::function<void()> importSample;
    std::function<void()> placeSample;
    std::function<bool()> canPlaceSample;
    AssetListModel assetListModel;

    juce::Label titleLabel;
    juce::TextButton importButton;
    juce::TextButton placeButton;
    juce::Label libraryLabel;
    juce::Label timelineLabel;
    AssetListBox assetList { projectGetter };
    juce::Viewport timelineViewport;
    std::unique_ptr<SampleTimelineComponent> timeline;
};

class PianoRollWindowComponent final : public juce::Component,
                                       private juce::ScrollBar::Listener
{
public:
    using ProjectGetter = PianoRollComponent::ProjectGetter;
    using TrackIndexGetter = PianoRollComponent::TrackIndexGetter;
    using SelectedSectionIndexGetter = PianoRollComponent::SelectedSectionIndexGetter;
    using ProjectWriter = PianoRollComponent::ProjectWriter;

    PianoRollWindowComponent(ProjectGetter projectGetterIn,
                             TrackIndexGetter trackIndexGetterIn,
                             SelectedSectionIndexGetter selectedSectionIndexGetterIn,
                             ProjectWriter projectWriterIn,
                             std::function<EditorToolMode()> toolModeGetterIn,
                             std::function<void(EditorToolMode)> setToolModeIn,
                             float initialPixelsPerBeat,
                             float initialRowHeightPixels,
                             std::function<void(float)> zoomChangedIn,
                             std::function<void(float)> rowHeightChangedIn,
                             bool showHeaderIn = true)
        : projectGetter(std::move(projectGetterIn)),
          projectWriter(std::move(projectWriterIn)),
          toolModeGetter(std::move(toolModeGetterIn)),
          setToolMode(std::move(setToolModeIn)),
          zoomChanged(std::move(zoomChangedIn)),
          rowHeightChanged(std::move(rowHeightChangedIn)),
          showHeader(showHeaderIn)
    {
        titleLabel.setText("Piano Roll", juce::dontSendNotification);
        titleLabel.setFont(ui::titleFont());
        titleLabel.setColour(juce::Label::textColourId, juce::Colour::fromRGB(235, 239, 244));
        addChildComponent(titleLabel);

        toolHintLabel.setJustificationType(juce::Justification::centredRight);
        toolHintLabel.setColour(juce::Label::textColourId, juce::Colour::fromRGB(190, 199, 210));
        addChildComponent(toolHintLabel);

        zoomLabel.setText("Zoom", juce::dontSendNotification);
        zoomLabel.setColour(juce::Label::textColourId, juce::Colour::fromRGB(214, 219, 227));
        addChildComponent(zoomLabel);

        zoomSlider.setSliderStyle(juce::Slider::LinearHorizontal);
        zoomSlider.setRange(12.0, 96.0, 1.0);
        zoomSlider.setChangeNotificationOnlyOnRelease(false);
        zoomSlider.setTextBoxStyle(juce::Slider::TextBoxRight, false, 52, 22);
        zoomSlider.setTextValueSuffix(" px/beat");
        zoomSlider.setValue(initialPixelsPerBeat, juce::dontSendNotification);
        zoomSlider.onValueChange = [this]
        {
            if (! syncingControls && zoomChanged != nullptr)
                zoomChanged(static_cast<float>(zoomSlider.getValue()));
        };
        addChildComponent(zoomSlider);

        rowHeightLabel.setText("Row", juce::dontSendNotification);
        rowHeightLabel.setColour(juce::Label::textColourId, juce::Colour::fromRGB(214, 219, 227));
        addChildComponent(rowHeightLabel);

        rowHeightSlider.setSliderStyle(juce::Slider::LinearHorizontal);
        rowHeightSlider.setRange(8.0, 32.0, 1.0);
        rowHeightSlider.setChangeNotificationOnlyOnRelease(false);
        rowHeightSlider.setTextBoxStyle(juce::Slider::TextBoxRight, false, 52, 22);
        rowHeightSlider.setTextValueSuffix(" px");
        rowHeightSlider.setValue(initialRowHeightPixels, juce::dontSendNotification);
        rowHeightSlider.onValueChange = [this]
        {
            if (! syncingControls && rowHeightChanged != nullptr)
                rowHeightChanged(static_cast<float>(rowHeightSlider.getValue()));
        };
        addChildComponent(rowHeightSlider);

        keyQuantizeLabel.setText("Key Quantize", juce::dontSendNotification);
        keyQuantizeLabel.setColour(juce::Label::textColourId, juce::Colour::fromRGB(214, 219, 227));
        addChildComponent(keyQuantizeLabel);

        for (const auto& option : keyQuantizeOptions())
            keyQuantizeBox.addItem(option.label, option.id);
        keyQuantizeBox.setTextWhenNothingSelected("All Notes");
        keyQuantizeBox.onChange = [this]
        {
            if (syncingControls)
                return;

            const auto* option = findKeyQuantizeOptionById(keyQuantizeBox.getSelectedId());
            if (option == nullptr)
                return;

            auto updatedProject = projectGetter();
            const auto currentScale = normaliseKeyQuantizeScale(updatedProject.keyQuantizeScale);
            const auto currentRoot = juce::negativeAwareModulo(updatedProject.keyQuantizeRoot, 12);
            if (currentRoot == option->root && currentScale.equalsIgnoreCase(option->scaleId))
                return;

            updatedProject.keyQuantizeRoot = option->root;
            updatedProject.keyQuantizeScale = option->scaleId;
            updatedProject.recalculateTimeFields();
            projectWriter(updatedProject, true, "Change Key Quantize");
        };
        addChildComponent(keyQuantizeBox);

        noteRangeLabel.setText("Range", juce::dontSendNotification);
        noteRangeLabel.setColour(juce::Label::textColourId, juce::Colour::fromRGB(214, 219, 227));
        addChildComponent(noteRangeLabel);

        for (const auto& option : pianoRollPitchOptions())
        {
            noteRangeMinBox.addItem(option.label, option.id);
            noteRangeMaxBox.addItem(option.label, option.id);
        }
        noteRangeMinBox.setTextWhenNothingSelected("Low");
        noteRangeMaxBox.setTextWhenNothingSelected("High");
        noteRangeMinBox.setJustificationType(juce::Justification::centredLeft);
        noteRangeMaxBox.setJustificationType(juce::Justification::centredLeft);
        noteRangeMinBox.onChange = [this] { handlePianoRollRangeChanged(true); };
        noteRangeMaxBox.onChange = [this] { handlePianoRollRangeChanged(false); };
        addChildComponent(noteRangeMinBox);

        noteRangeSeparatorLabel.setText("to", juce::dontSendNotification);
        noteRangeSeparatorLabel.setJustificationType(juce::Justification::centred);
        noteRangeSeparatorLabel.setColour(juce::Label::textColourId, juce::Colour::fromRGB(190, 199, 210));
        addChildComponent(noteRangeSeparatorLabel);
        addChildComponent(noteRangeMaxBox);

        noteEditor = std::make_unique<PianoRollComponent>(projectGetter,
                                                          trackIndexGetterIn,
                                                          selectedSectionIndexGetterIn,
                                                          projectWriter);
        noteEditor->setSurfaceMode(PianoRollComponent::SurfaceMode::notesOnly);
        noteEditor->setToolModeChangeCallback(setToolMode);
        noteEditor->setZoomChangedCallback([this] (float pixelsPerBeat)
                                           {
                                               if (zoomChanged != nullptr)
                                                   zoomChanged(pixelsPerBeat);
                                           });
        noteEditor->setRowHeightChangedCallback([this] (float rowHeightPixels)
                                                {
                                                    if (rowHeightChanged != nullptr)
                                                        rowHeightChanged(rowHeightPixels);
                                                });
        noteViewport.setViewedComponent(noteEditor.get(), false);
        noteViewport.setScrollBarsShown(true, true);
        addAndMakeVisible(noteViewport);

        controllerEditor = std::make_unique<PianoRollComponent>(projectGetter,
                                                                std::move(trackIndexGetterIn),
                                                                std::move(selectedSectionIndexGetterIn),
                                                                projectWriter);
        controllerEditor->setSurfaceMode(PianoRollComponent::SurfaceMode::controllerOnly);
        controllerEditor->setToolModeChangeCallback(setToolMode);
        controllerEditor->setZoomChangedCallback([this] (float pixelsPerBeat)
                                                 {
                                                     if (zoomChanged != nullptr)
                                                         zoomChanged(pixelsPerBeat);
                                                 });
        controllerEditor->setRowHeightChangedCallback([this] (float rowHeightPixels)
                                                      {
                                                          if (rowHeightChanged != nullptr)
                                                              rowHeightChanged(rowHeightPixels);
                                                      });
        controllerViewport.setViewedComponent(controllerEditor.get(), false);
        controllerViewport.setScrollBarsShown(false, true);
        addAndMakeVisible(controllerViewport);

        noteViewport.getHorizontalScrollBar().addListener(this);
        controllerViewport.getHorizontalScrollBar().addListener(this);

        titleLabel.setVisible(showHeader);
        toolHintLabel.setVisible(showHeader);
        zoomLabel.setVisible(showHeader);
        zoomSlider.setVisible(showHeader);
        rowHeightLabel.setVisible(showHeader);
        rowHeightSlider.setVisible(showHeader);
        keyQuantizeLabel.setVisible(showHeader);
        keyQuantizeBox.setVisible(showHeader);
        noteRangeLabel.setVisible(showHeader);
        noteRangeMinBox.setVisible(showHeader);
        noteRangeSeparatorLabel.setVisible(showHeader);
        noteRangeMaxBox.setVisible(showHeader);
    }

    ~PianoRollWindowComponent() override
    {
        noteViewport.getHorizontalScrollBar().removeListener(this);
        controllerViewport.getHorizontalScrollBar().removeListener(this);
    }

    void setNotePreviewCallbacks(PianoRollComponent::NotePreviewCallback noteOnCallback,
                                 PianoRollComponent::NotePreviewCallback noteOffCallback,
                                 PianoRollComponent::PreviewStopCallback stopPreviewCallback = {})
    {
        if (noteEditor != nullptr)
        {
            noteEditor->setNotePreviewCallbacks(noteOnCallback,
                                                noteOffCallback,
                                                stopPreviewCallback);
        }

        if (controllerEditor != nullptr)
        {
            controllerEditor->setNotePreviewCallbacks(std::move(noteOnCallback),
                                                      std::move(noteOffCallback),
                                                      std::move(stopPreviewCallback));
        }
    }

    void setKeyHandlerCallback(PianoRollComponent::KeyHandlerCallback keyHandlerCallback)
    {
        if (noteEditor != nullptr)
            noteEditor->setKeyHandlerCallback(keyHandlerCallback);

        if (controllerEditor != nullptr)
            controllerEditor->setKeyHandlerCallback(std::move(keyHandlerCallback));
    }

    void setViewScale(float pixelsPerBeat, float rowHeightPixels)
    {
        syncingControls = true;
        zoomSlider.setValue(pixelsPerBeat, juce::dontSendNotification);
        rowHeightSlider.setValue(rowHeightPixels, juce::dontSendNotification);
        syncingControls = false;

        if (noteEditor != nullptr)
        {
            noteEditor->setHorizontalZoom(pixelsPerBeat);
            noteEditor->setNoteRowHeight(rowHeightPixels);
        }

        if (controllerEditor != nullptr)
        {
            controllerEditor->setHorizontalZoom(pixelsPerBeat);
            controllerEditor->setNoteRowHeight(rowHeightPixels);
        }
    }

    void refreshFromModel()
    {
        const auto mode = toolModeGetter != nullptr ? toolModeGetter() : EditorToolMode::pencil;
        if (showHeader)
        {
            toolHintLabel.setText("Tool: " + editorToolModeLabel(mode) + "  |  Right-click for tools",
                                  juce::dontSendNotification);
            syncingControls = true;
            const auto& currentProject = projectGetter();
            keyQuantizeBox.setSelectedId(keyQuantizeOptionId(currentProject), juce::dontSendNotification);
            noteRangeMinBox.setSelectedId(pianoRollPitchOptionId(currentProject.pianoRollVisiblePitchMin), juce::dontSendNotification);
            noteRangeMaxBox.setSelectedId(pianoRollPitchOptionId(currentProject.pianoRollVisiblePitchMax), juce::dontSendNotification);
            syncingControls = false;
        }

        if (noteEditor != nullptr)
        {
            noteEditor->setToolMode(mode);
            noteEditor->refreshFromModel();
        }

        if (controllerEditor != nullptr)
        {
            controllerEditor->setToolMode(mode);
            controllerEditor->refreshFromModel();
        }

        focusInitialNoteViewportIfNeeded();
        syncHorizontalScrollFrom(noteViewport);
    }

    void refreshPlaybackState()
    {
        noteViewport.repaint();
        controllerViewport.repaint();
    }

    void focusEditor()
    {
        if (noteEditor != nullptr)
            noteEditor->grabKeyboardFocus();
    }

    bool selectAllNotes()
    {
        return noteEditor != nullptr && noteEditor->selectAllNotes();
    }

    void paint(juce::Graphics& g) override
    {
        g.fillAll(juce::Colour::fromRGB(13, 15, 20));
        g.setColour(juce::Colour::fromRGB(31, 35, 44));
        g.drawRect(getLocalBounds(), 1);

        if (!controllerViewport.getBounds().isEmpty())
        {
            g.setColour(juce::Colour::fromRGB(36, 42, 52));
            g.drawHorizontalLine(controllerViewport.getY() - 3, 12.0f, static_cast<float>(getWidth() - 12));
        }
    }

    void resized() override
    {
        auto area = getLocalBounds().reduced(14);
        if (showHeader)
        {
            auto header = area.removeFromTop(30);
            titleLabel.setBounds(header.removeFromLeft(120));
            auto controls = header.removeFromRight(juce::jmin(960, juce::jmax(0, header.getWidth())));
            rowHeightSlider.setBounds(controls.removeFromRight(140));
            controls.removeFromRight(8);
            rowHeightLabel.setBounds(controls.removeFromRight(34));
            controls.removeFromRight(10);
            zoomSlider.setBounds(controls.removeFromRight(140));
            controls.removeFromRight(8);
            zoomLabel.setBounds(controls.removeFromRight(42));
            controls.removeFromRight(14);
            noteRangeMaxBox.setBounds(controls.removeFromRight(92));
            controls.removeFromRight(6);
            noteRangeSeparatorLabel.setBounds(controls.removeFromRight(20));
            controls.removeFromRight(6);
            noteRangeMinBox.setBounds(controls.removeFromRight(92));
            controls.removeFromRight(8);
            noteRangeLabel.setBounds(controls.removeFromRight(44));
            controls.removeFromRight(14);
            keyQuantizeBox.setBounds(controls.removeFromRight(188));
            controls.removeFromRight(8);
            keyQuantizeLabel.setBounds(controls.removeFromRight(92));
            controls.removeFromRight(14);
            toolHintLabel.setBounds(header);
            area.removeFromTop(8);
        }
        else
        {
            zoomLabel.setBounds({});
            zoomSlider.setBounds({});
            rowHeightLabel.setBounds({});
            rowHeightSlider.setBounds({});
            keyQuantizeLabel.setBounds({});
            keyQuantizeBox.setBounds({});
            noteRangeLabel.setBounds({});
            noteRangeMinBox.setBounds({});
            noteRangeSeparatorLabel.setBounds({});
            noteRangeMaxBox.setBounds({});
        }

        const auto controllerPaneHeight = juce::jlimit(144, 220, area.getHeight() / 3);
        auto controllerArea = area.removeFromBottom(controllerPaneHeight);
        area.removeFromBottom(6);
        noteViewport.setBounds(area);
        controllerViewport.setBounds(controllerArea);
        focusInitialNoteViewportIfNeeded();
    }

private:
    void handlePianoRollRangeChanged(bool minimumChanged)
    {
        if (syncingControls)
            return;

        const auto* minimumOption = findPianoRollPitchOptionById(noteRangeMinBox.getSelectedId());
        const auto* maximumOption = findPianoRollPitchOptionById(noteRangeMaxBox.getSelectedId());
        if (minimumOption == nullptr || maximumOption == nullptr)
            return;

        auto updatedProject = projectGetter();
        auto minimumPitch = minimumOption->pitch;
        auto maximumPitch = maximumOption->pitch;
        if (minimumPitch > maximumPitch)
        {
            if (minimumChanged)
                maximumPitch = minimumPitch;
            else
                minimumPitch = maximumPitch;
        }

        if (updatedProject.pianoRollVisiblePitchMin == minimumPitch
            && updatedProject.pianoRollVisiblePitchMax == maximumPitch)
        {
            return;
        }

        updatedProject.pianoRollVisiblePitchMin = minimumPitch;
        updatedProject.pianoRollVisiblePitchMax = maximumPitch;
        updatedProject.recalculateTimeFields();
        projectWriter(updatedProject, true, "Change Piano Roll Range");
    }

    void focusInitialNoteViewportIfNeeded()
    {
        if (!initialNoteViewPending || noteEditor == nullptr || noteViewport.getHeight() <= 0)
            return;

        noteViewport.setViewPosition(noteViewport.getViewPositionX(),
                                     noteEditor->viewPositionYForPitch(60, noteViewport.getHeight()));
        initialNoteViewPending = false;
    }

    ProjectGetter projectGetter;
    ProjectWriter projectWriter;
    void scrollBarMoved(juce::ScrollBar* scrollBarThatHasMoved, double newRangeStart) override
    {
        juce::ignoreUnused(newRangeStart);
        if (syncingScroll)
            return;

        if (scrollBarThatHasMoved == &noteViewport.getHorizontalScrollBar())
            syncHorizontalScrollFrom(noteViewport);
        else if (scrollBarThatHasMoved == &controllerViewport.getHorizontalScrollBar())
            syncHorizontalScrollFrom(controllerViewport);
    }

    void syncHorizontalScrollFrom(juce::Viewport& source)
    {
        if (syncingScroll)
            return;

        const auto targetX = source.getViewPositionX();
        syncingScroll = true;
        if (&source != &noteViewport)
            noteViewport.setViewPosition(targetX, noteViewport.getViewPositionY());
        if (&source != &controllerViewport)
            controllerViewport.setViewPosition(targetX, controllerViewport.getViewPositionY());
        syncingScroll = false;
    }

    std::function<EditorToolMode()> toolModeGetter;
    std::function<void(EditorToolMode)> setToolMode;
    std::function<void(float)> zoomChanged;
    std::function<void(float)> rowHeightChanged;
    bool showHeader = true;
    bool syncingScroll = false;
    bool syncingControls = false;
    bool initialNoteViewPending = true;

    juce::Label titleLabel;
    juce::Label toolHintLabel;
    juce::Label zoomLabel;
    juce::Slider zoomSlider;
    juce::Label rowHeightLabel;
    juce::Slider rowHeightSlider;
    juce::Label keyQuantizeLabel;
    juce::ComboBox keyQuantizeBox;
    juce::Label noteRangeLabel;
    juce::ComboBox noteRangeMinBox;
    juce::Label noteRangeSeparatorLabel;
    juce::ComboBox noteRangeMaxBox;
    juce::Viewport noteViewport;
    juce::Viewport controllerViewport;
    std::unique_ptr<PianoRollComponent> noteEditor;
    std::unique_ptr<PianoRollComponent> controllerEditor;
};

class VirtualPianoKeyboardComponent final : public juce::Component,
                                            private juce::Timer
{
public:
    VirtualPianoKeyboardComponent()
    {
        setOpaque(true);
        setWantsKeyboardFocus(true);
        setScalePercent(100);
    }

    void setScalePercent(int percent)
    {
        const auto clampedPercent = juce::jlimit(50, 150, percent);
        if (scalePercent == clampedPercent && !keyLayouts.empty())
            return;

        scalePercent = clampedPercent;
        rebuildKeyLayouts();
        repaint();
    }

    int getScalePercent() const noexcept
    {
        return scalePercent;
    }

    void setNoteTriggeredCallback(std::function<void(int)> callback)
    {
        noteTriggered = std::move(callback);
    }

    void flashPitch(int pitch)
    {
        flashedPitch = pitch;
        startTimer(140);
        repaint();
    }

    void paint(juce::Graphics& g) override
    {
        g.fillAll(juce::Colour::fromRGB(20, 22, 28));

        for (const auto& key : keyLayouts)
        {
            if (key.isBlack)
                continue;

            auto fill = juce::Colour::fromRGB(252, 252, 252);
            auto accent = juce::Colour::fromRGB(208, 214, 224);
            if (flashedPitch == key.pitch)
            {
                fill = juce::Colour::fromRGB(202, 235, 255);
                accent = juce::Colour::fromRGB(112, 186, 255);
            }

            juce::ColourGradient gradient(fill, key.rect.getTopLeft(), accent, key.rect.getBottomLeft(), false);
            g.setGradientFill(gradient);
            g.setColour(fill);
            g.fillRoundedRectangle(key.rect, 4.0f);
            g.setColour(juce::Colour::fromRGB(72, 78, 88));
            g.drawRoundedRectangle(key.rect, 4.0f, 1.0f);

            g.setColour(juce::Colour::fromRGB(44, 48, 56));
            g.setFont(juce::FontOptions(ui::scaleValue(juce::jmax(9.0f, 9.0f * scaleFactor())), juce::Font::bold));
            g.drawText(noteNameLabel(key.pitch),
                       key.rect.withTrimmedTop(key.rect.getHeight() * 0.74f).toNearestInt(),
                       juce::Justification::centred);

            g.setColour(juce::Colour::fromRGB(96, 104, 118));
            g.setFont(juce::FontOptions(ui::scaleValue(juce::jmax(8.0f, 8.0f * scaleFactor()))));
            g.drawText(key.shortcutLabel,
                       key.rect.withTrimmedTop(key.rect.getHeight() * 0.86f).toNearestInt(),
                       juce::Justification::centred);
        }

        for (const auto& key : keyLayouts)
        {
            if (!key.isBlack)
                continue;

            auto fill = juce::Colour::fromRGB(72, 82, 96);
            auto accent = juce::Colour::fromRGB(18, 22, 28);
            if (flashedPitch == key.pitch)
            {
                fill = juce::Colour::fromRGB(78, 170, 255);
                accent = juce::Colour::fromRGB(20, 88, 180);
            }

            juce::ColourGradient gradient(fill, key.rect.getTopLeft(), accent, key.rect.getBottomLeft(), false);
            g.setGradientFill(gradient);
            g.fillRoundedRectangle(key.rect, 4.0f);
            g.setColour(juce::Colour::fromRGB(12, 14, 18));
            g.drawRoundedRectangle(key.rect, 4.0f, 1.0f);

            g.setColour(juce::Colour::fromRGB(232, 240, 248));
            g.setFont(juce::FontOptions(ui::scaleValue(juce::jmax(8.0f, 8.0f * scaleFactor())), juce::Font::bold));
            g.drawText(noteNameLabel(key.pitch),
                       key.rect.withTrimmedTop(key.rect.getHeight() * 0.7f).toNearestInt(),
                       juce::Justification::centred);

            g.setColour(juce::Colour::fromRGB(176, 188, 204));
            g.setFont(juce::FontOptions(ui::scaleValue(juce::jmax(7.0f, 7.0f * scaleFactor()))));
            g.drawText(key.shortcutLabel,
                       key.rect.withTrimmedTop(key.rect.getHeight() * 0.84f).toNearestInt(),
                       juce::Justification::centred);
        }
    }

    void mouseDown(const juce::MouseEvent& event) override
    {
        if (!event.mods.isLeftButtonDown())
            return;

        grabKeyboardFocus();

        if (const auto* key = hitTestKey(event.position))
        {
            flashPitch(key->pitch);
            if (noteTriggered != nullptr)
                noteTriggered(key->pitch);
        }
    }

private:
    struct KeyLayout
    {
        int pitch = 60;
        bool isBlack = false;
        juce::String shortcutLabel;
        juce::Rectangle<float> rect;
    };

    float scaleFactor() const noexcept
    {
        return static_cast<float>(scalePercent) / 100.0f;
    }

    void timerCallback() override
    {
        stopTimer();
        flashedPitch = -1;
        repaint();
    }

    void rebuildKeyLayouts()
    {
        keyLayouts.clear();
        keyLayouts.reserve(virtualPianoKeySpecs().size());

        const auto whiteKeyWidth = 54.0f * scaleFactor();
        const auto whiteKeyHeight = 184.0f * scaleFactor();
        const auto naturalWidth = (whiteKeyWidth * 14.0f) + 24.0f;
        const auto naturalHeight = whiteKeyHeight + 24.0f;

        setSize(juce::jmax(280, juce::roundToInt(naturalWidth)),
                juce::jmax(96, juce::roundToInt(naturalHeight)));

        const auto area = getLocalBounds().toFloat().reduced(12.0f);
        const auto blackKeyWidth = whiteKeyWidth * 0.62f;
        const auto blackKeyHeight = whiteKeyHeight * 0.62f;

        auto whiteCursor = 0;
        for (const auto& spec : virtualPianoKeySpecs())
        {
            KeyLayout layout;
            layout.pitch = spec.pitch;
            layout.isBlack = isVirtualPianoBlackKey(spec.pitch);
            layout.shortcutLabel = virtualPianoShortcutLabel(spec);

            if (layout.isBlack)
            {
                const auto centerX = area.getX() + (static_cast<float>(whiteCursor) * whiteKeyWidth);
                layout.rect = { centerX - (blackKeyWidth * 0.5f), area.getY(), blackKeyWidth, blackKeyHeight };
            }
            else
            {
                layout.rect = { area.getX() + (static_cast<float>(whiteCursor) * whiteKeyWidth),
                                area.getY(),
                                whiteKeyWidth + 0.5f,
                                whiteKeyHeight };
                ++whiteCursor;
            }

            keyLayouts.push_back(std::move(layout));
        }
    }

    const KeyLayout* hitTestKey(juce::Point<float> position) const
    {
        for (const auto& key : keyLayouts)
        {
            if (key.isBlack && key.rect.contains(position))
                return &key;
        }

        for (const auto& key : keyLayouts)
        {
            if (!key.isBlack && key.rect.contains(position))
                return &key;
        }

        return nullptr;
    }

    std::function<void(int)> noteTriggered;
    std::vector<KeyLayout> keyLayouts;
    int scalePercent = 100;
    int flashedPitch = -1;
};

class VirtualPianoWindowComponent final : public juce::Component
{
public:
    explicit VirtualPianoWindowComponent(std::function<void(int)> triggerPitchIn,
                                         std::function<bool(const juce::KeyPress&)> shortcutHandlerIn)
        : triggerPitch(std::move(triggerPitchIn)),
          shortcutHandler(std::move(shortcutHandlerIn))
    {
        setWantsKeyboardFocus(true);

        titleLabel.setText("Virtual Piano", juce::dontSendNotification);
        titleLabel.setFont(ui::titleFont());
        titleLabel.setColour(juce::Label::textColourId, juce::Colour::fromRGB(235, 239, 244));
        addAndMakeVisible(titleLabel);

        addAndMakeVisible(scaleCombo);

        scaleLabel.setText("Key Scale", juce::dontSendNotification);
        scaleLabel.setColour(juce::Label::textColourId, juce::Colour::fromRGB(210, 218, 228));
        scaleLabel.attachToComponent(&scaleCombo, true);
        addAndMakeVisible(scaleLabel);

        for (const auto percent : { 50, 75, 100, 125, 150 })
            scaleCombo.addItem(juce::String(percent) + "%", percent);
        scaleCombo.setSelectedId(100, juce::dontSendNotification);
        scaleCombo.onChange = [this]
        {
            if (keyboard != nullptr)
                keyboard->setScalePercent(scaleCombo.getSelectedId());
        };

        hintLabel.setText(virtualPianoHintText(), juce::dontSendNotification);
        hintLabel.setColour(juce::Label::textColourId, juce::Colour::fromRGB(199, 208, 220));
        hintLabel.setJustificationType(juce::Justification::centredLeft);
        hintLabel.setFont(ui::font());
        addAndMakeVisible(hintLabel);

        keyboard = std::make_unique<VirtualPianoKeyboardComponent>();
        keyboard->setNoteTriggeredCallback([this] (int pitch)
        {
            if (triggerPitch != nullptr)
                triggerPitch(pitch);
        });

        keyboardViewport.setViewedComponent(keyboard.get(), false);
        keyboardViewport.setScrollBarsShown(true, false);
        addAndMakeVisible(keyboardViewport);
    }

    bool keyPressed(const juce::KeyPress& key) override
    {
        return shortcutHandler != nullptr && shortcutHandler(key);
    }

    void flashPitch(int pitch)
    {
        if (keyboard != nullptr)
            keyboard->flashPitch(pitch);
    }

    void focusKeyboard()
    {
        grabKeyboardFocus();
    }

    void resized() override
    {
        auto area = getLocalBounds().reduced(14);
        auto header = area.removeFromTop(28);
        titleLabel.setBounds(header.removeFromLeft(180));
        auto scaleArea = header.removeFromRight(180);
        scaleCombo.setBounds(scaleArea.removeFromRight(88));
        area.removeFromTop(8);
        hintLabel.setBounds(area.removeFromTop(40));
        area.removeFromTop(8);
        keyboardViewport.setBounds(area);
    }

    void paint(juce::Graphics& g) override
    {
        g.fillAll(juce::Colour::fromRGB(13, 15, 20));
        g.setColour(juce::Colour::fromRGB(31, 35, 44));
        g.drawRect(getLocalBounds(), 1);
    }

private:
    std::function<void(int)> triggerPitch;
    std::function<bool(const juce::KeyPress&)> shortcutHandler;

    juce::Label titleLabel;
    juce::Label scaleLabel;
    juce::ComboBox scaleCombo;
    juce::Label hintLabel;
    juce::Viewport keyboardViewport;
    std::unique_ptr<VirtualPianoKeyboardComponent> keyboard;
};

class TracksWorkspaceWindowComponent final : public juce::Component
{
public:
    using ProjectGetter = std::function<const ProjectState&()>;
    using SelectedTrackGetter = std::function<int()>;
    using SelectedTrackSetter = std::function<void(int)>;
    using SelectedTrackMutator = std::function<void(std::function<void(TrackState&)>, const juce::String&)>;
    using TrackSummaryGetter = std::function<juce::String(int)>;

    TracksWorkspaceWindowComponent(ProjectGetter projectGetterIn,
                                   SelectedTrackGetter selectedTrackGetterIn,
                                   SelectedTrackSetter selectedTrackSetterIn,
                                   SelectedTrackMutator mutateSelectedTrackIn,
                                   TrackSummaryGetter trackSummaryGetterIn,
                                   std::function<void()> addTrackIn,
                                   std::function<void()> duplicateTrackIn,
                                   std::function<void()> removeTrackIn,
                                   std::function<void()> openRackEditorIn,
                                   std::function<void()> saveRackStateIn,
                                   std::function<void()> playTrackIn,
                                   std::function<void()> stopPlaybackIn)
        : projectGetter(std::move(projectGetterIn)),
          selectedTrackGetter(std::move(selectedTrackGetterIn)),
          selectedTrackSetter(std::move(selectedTrackSetterIn)),
          mutateSelectedTrack(std::move(mutateSelectedTrackIn)),
          trackSummaryGetter(std::move(trackSummaryGetterIn)),
          addTrack(std::move(addTrackIn)),
          duplicateTrack(std::move(duplicateTrackIn)),
          removeTrack(std::move(removeTrackIn)),
          openRackEditor(std::move(openRackEditorIn)),
          saveRackState(std::move(saveRackStateIn)),
          playTrack(std::move(playTrackIn)),
          stopPlayback(std::move(stopPlaybackIn)),
          trackListModel(projectGetter, selectedTrackSetter)
    {
        titleLabel.setText("Tracks", juce::dontSendNotification);
        titleLabel.setFont(ui::titleFont());
        titleLabel.setColour(juce::Label::textColourId, juce::Colour::fromRGB(235, 239, 244));
        addAndMakeVisible(titleLabel);

        configureButton(addTrackButton, "Add");
        addTrackButton.onClick = [this]
        {
            if (addTrack != nullptr)
                addTrack();
        };

        configureButton(duplicateTrackButton, "Duplicate");
        duplicateTrackButton.onClick = [this]
        {
            if (duplicateTrack != nullptr)
                duplicateTrack();
        };

        configureButton(removeTrackButton, "Remove");
        removeTrackButton.onClick = [this]
        {
            if (removeTrack != nullptr)
                removeTrack();
        };

        configureButton(openRackEditorButton, "Edit VST");
        openRackEditorButton.onClick = [this]
        {
            if (openRackEditor != nullptr)
                openRackEditor();
        };

        configureButton(saveRackStateButton, "Save Rack");
        saveRackStateButton.onClick = [this]
        {
            if (saveRackState != nullptr)
                saveRackState();
        };

        configureButton(playTrackButton, "Play Track");
        playTrackButton.onClick = [this]
        {
            if (playTrack != nullptr)
                playTrack();
        };

        configureButton(stopButton, "Stop");
        stopButton.onClick = [this]
        {
            if (stopPlayback != nullptr)
                stopPlayback();
        };

        trackList.setModel(&trackListModel);
        trackList.setRowHeight(46);
        trackList.setColour(juce::ListBox::backgroundColourId, juce::Colour::fromRGB(20, 22, 28));
        trackList.setOutlineThickness(1);
        addAndMakeVisible(trackList);

        inspectorLabel.setText("Inspector", juce::dontSendNotification);
        inspectorLabel.setFont(ui::sectionFont());
        inspectorLabel.setColour(juce::Label::textColourId, juce::Colour::fromRGB(235, 239, 244));
        addAndMakeVisible(inspectorLabel);

        configureEditor(trackNameEditor, "Track name");
        trackNameEditor.onFocusLost = [this]
        {
            if (syncingControls)
                return;

            const auto value = trackNameEditor.getText().trim();
            applyTrackMutation([value] (TrackState& track)
                               {
                                   track.name = value.isNotEmpty() ? value : "Track";
                               },
                               "Rename Track");
        };
        trackNameEditor.onReturnKey = [this] { trackNameEditor.giveAwayKeyboardFocus(); };

        configureEditor(trackTypeEditor, "Track type");
        trackTypeEditor.onFocusLost = [this]
        {
            if (syncingControls)
                return;

            const auto value = trackTypeEditor.getText().trim();
            applyTrackMutation([value] (TrackState& track)
                               {
                                   track.trackType = value.isNotEmpty() ? value : "instrument";
                               },
                               "Change Track Type");
        };
        trackTypeEditor.onReturnKey = [this] { trackTypeEditor.giveAwayKeyboardFocus(); };

        configureEditor(instrumentModeEditor, "Instrument mode");
        instrumentModeEditor.onFocusLost = [this]
        {
            if (syncingControls)
                return;

            const auto value = instrumentModeEditor.getText().trim();
            applyTrackMutation([value] (TrackState& track)
                               {
                                   track.instrumentMode = value.isNotEmpty() ? value : "General MIDI";
                               },
                               "Change Instrument Mode");
        };
        instrumentModeEditor.onReturnKey = [this] { instrumentModeEditor.giveAwayKeyboardFocus(); };

        configureEditor(instrumentEditor, "Instrument name");
        instrumentEditor.onFocusLost = [this]
        {
            if (syncingControls)
                return;

            const auto value = instrumentEditor.getText().trim();
            applyTrackMutation([value] (TrackState& track)
                               {
                                   track.instrument = value.isNotEmpty() ? value : "Piano";
                               },
                               "Change Instrument");
        };
        instrumentEditor.onReturnKey = [this] { instrumentEditor.giveAwayKeyboardFocus(); };

        configureEditor(rackVstEditor, "Rack VST path or name");
        rackVstEditor.onFocusLost = [this]
        {
            if (syncingControls)
                return;

            const auto value = rackVstEditor.getText().trim();
            applyTrackMutation([value] (TrackState& track)
                               {
                                   track.rackVst = value;
                               },
                               "Change Rack VST");
        };
        rackVstEditor.onReturnKey = [this] { rackVstEditor.giveAwayKeyboardFocus(); };

        configureSlider(midiChannelSlider, 1.0, 16.0, 1.0, " ch");
        midiChannelSlider.onValueChange = [this]
        {
            if (syncingControls)
                return;

            applyTrackMutation([value = juce::roundToInt(midiChannelSlider.getValue())] (TrackState& track)
                               {
                                   track.midiChannel = juce::jlimit(0, 15, value - 1);
                               },
                               "Change MIDI Channel");
        };

        configureSlider(midiProgramSlider, 0.0, 127.0, 1.0, " pgm");
        midiProgramSlider.onValueChange = [this]
        {
            if (syncingControls)
                return;

            applyTrackMutation([value = juce::roundToInt(midiProgramSlider.getValue())] (TrackState& track)
                               {
                                   track.midiProgram = juce::jlimit(0, 127, value);
                               },
                               "Change MIDI Program");
        };

        configureSlider(volumeSlider, 0.0, 1.0, 0.01, "");
        volumeSlider.onValueChange = [this]
        {
            if (syncingControls)
                return;

            applyTrackMutation([value = volumeSlider.getValue()] (TrackState& track)
                               {
                                   track.volume = juce::jlimit(0.0, 1.0, value);
                               },
                               "Change Track Volume");
        };

        configureSlider(panSlider, -1.0, 1.0, 0.01, "");
        panSlider.onValueChange = [this]
        {
            if (syncingControls)
                return;

            applyTrackMutation([value = panSlider.getValue()] (TrackState& track)
                               {
                                   track.pan = juce::jlimit(-1.0, 1.0, value);
                               },
                               "Change Track Pan");
        };

        muteToggle.setButtonText("Mute");
        muteToggle.onClick = [this]
        {
            if (syncingControls)
                return;

            applyTrackMutation([value = muteToggle.getToggleState()] (TrackState& track)
                               {
                                   track.mute = value;
                               },
                               "Toggle Mute");
        };
        addAndMakeVisible(muteToggle);

        soloToggle.setButtonText("Solo");
        soloToggle.onClick = [this]
        {
            if (syncingControls)
                return;

            applyTrackMutation([value = soloToggle.getToggleState()] (TrackState& track)
                               {
                                   track.solo = value;
                               },
                               "Toggle Solo");
        };
        addAndMakeVisible(soloToggle);

        liveArmToggle.setButtonText("Arm");
        liveArmToggle.onClick = [this]
        {
            if (syncingControls)
                return;

            applyTrackMutation([value = liveArmToggle.getToggleState()] (TrackState& track)
                               {
                                   track.liveArmed = value;
                               },
                               "Toggle Record Arm");
        };
        addAndMakeVisible(liveArmToggle);

        summaryEditor.setMultiLine(true);
        summaryEditor.setReadOnly(true);
        summaryEditor.setScrollbarsShown(true);
        summaryEditor.setColour(juce::TextEditor::backgroundColourId, juce::Colour::fromRGB(18, 20, 25));
        summaryEditor.setColour(juce::TextEditor::textColourId, juce::Colour::fromRGB(226, 230, 237));
        summaryEditor.setColour(juce::TextEditor::outlineColourId, juce::Colour::fromRGB(56, 64, 79));
        summaryEditor.setFont(ui::font());
        addAndMakeVisible(summaryEditor);
    }

    void refreshFromModel()
    {
        const auto& project = projectGetter();
        const auto selectedTrack = selectedTrackGetter != nullptr ? selectedTrackGetter() : -1;

        trackList.updateContent();
        if (!project.tracks.empty() && selectedTrack >= 0)
            trackList.selectRow(selectedTrack, false, true);
        else
            trackList.deselectAllRows();

        refreshInspectorControls();

        const auto hasSelection = selectedTrack >= 0 && selectedTrack < static_cast<int>(project.tracks.size());
        duplicateTrackButton.setEnabled(hasSelection);
        removeTrackButton.setEnabled(hasSelection);
        openRackEditorButton.setEnabled(hasSelection);
        saveRackStateButton.setEnabled(hasSelection);
        playTrackButton.setEnabled(hasSelection);
        stopButton.setEnabled(hasSelection);
    }

    void focusTrackList()
    {
        trackList.grabKeyboardFocus();
    }

    void paint(juce::Graphics& g) override
    {
        g.fillAll(juce::Colour::fromRGB(13, 15, 20));
        g.setColour(juce::Colour::fromRGB(31, 35, 44));
        g.drawRect(getLocalBounds(), 1);
    }

    void resized() override
    {
        auto area = getLocalBounds().reduced(14);
        auto header = area.removeFromTop(30);
        titleLabel.setBounds(header.removeFromLeft(120));
        addTrackButton.setBounds(header.removeFromLeft(70));
        header.removeFromLeft(6);
        duplicateTrackButton.setBounds(header.removeFromLeft(90));
        header.removeFromLeft(6);
        removeTrackButton.setBounds(header.removeFromLeft(82));
        header.removeFromLeft(10);
        openRackEditorButton.setBounds(header.removeFromLeft(88));
        header.removeFromLeft(6);
        saveRackStateButton.setBounds(header.removeFromLeft(88));
        header.removeFromLeft(6);
        playTrackButton.setBounds(header.removeFromLeft(86));
        header.removeFromLeft(6);
        stopButton.setBounds(header.removeFromLeft(68));

        area.removeFromTop(10);
        auto leftArea = area.removeFromLeft(juce::jmin(360, juce::roundToInt(static_cast<float>(area.getWidth()) * 0.34f)));
        trackList.setBounds(leftArea);
        area.removeFromLeft(12);

        inspectorLabel.setBounds(area.removeFromTop(24));
        area.removeFromTop(6);
        trackNameEditor.setBounds(area.removeFromTop(24));
        area.removeFromTop(6);

        auto typeRow = area.removeFromTop(24);
        auto typeWidth = juce::roundToInt(static_cast<float>(typeRow.getWidth()) * 0.44f);
        trackTypeEditor.setBounds(typeRow.removeFromLeft(typeWidth));
        typeRow.removeFromLeft(6);
        instrumentModeEditor.setBounds(typeRow);
        area.removeFromTop(6);

        instrumentEditor.setBounds(area.removeFromTop(24));
        area.removeFromTop(6);
        rackVstEditor.setBounds(area.removeFromTop(24));
        area.removeFromTop(6);

        auto midiRow = area.removeFromTop(24);
        auto halfWidth = juce::roundToInt(static_cast<float>(midiRow.getWidth()) * 0.5f) - 3;
        midiChannelSlider.setBounds(midiRow.removeFromLeft(halfWidth));
        midiRow.removeFromLeft(6);
        midiProgramSlider.setBounds(midiRow);
        area.removeFromTop(6);

        auto mixRow = area.removeFromTop(24);
        volumeSlider.setBounds(mixRow.removeFromLeft(halfWidth));
        mixRow.removeFromLeft(6);
        panSlider.setBounds(mixRow);
        area.removeFromTop(8);

        auto toggleRow = area.removeFromTop(24);
        muteToggle.setBounds(toggleRow.removeFromLeft(70));
        toggleRow.removeFromLeft(8);
        soloToggle.setBounds(toggleRow.removeFromLeft(70));
        toggleRow.removeFromLeft(8);
        liveArmToggle.setBounds(toggleRow.removeFromLeft(70));
        area.removeFromTop(8);

        summaryEditor.setBounds(area);
    }

private:
    class TrackListModel final : public juce::ListBoxModel
    {
    public:
        TrackListModel(ProjectGetter projectGetterIn,
                       SelectedTrackSetter selectedTrackSetterIn)
            : projectGetter(std::move(projectGetterIn)),
              selectedTrackSetter(std::move(selectedTrackSetterIn))
        {
        }

        int getNumRows() override
        {
            return static_cast<int>(projectGetter().tracks.size());
        }

        void paintListBoxItem(int rowNumber,
                              juce::Graphics& g,
                              int width,
                              int height,
                              bool rowIsSelected) override
        {
            juce::ignoreUnused(height);
            const auto& project = projectGetter();
            if (!juce::isPositiveAndBelow(rowNumber, static_cast<int>(project.tracks.size())))
                return;

            const auto& track = project.tracks[static_cast<size_t>(rowNumber)];
            juce::StringArray trackFlags;
            if (track.mute)
                trackFlags.add("M");
            if (track.solo)
                trackFlags.add("S");
            if (track.liveArmed)
                trackFlags.add("ARM");

            const auto background = (rowNumber % 2) == 0 ? juce::Colour::fromRGB(26, 30, 37)
                                                         : juce::Colour::fromRGB(21, 25, 31);
            const auto trackColour = trackDisplayColour(track, rowNumber);
            g.fillAll(background);
            g.setColour(trackColour.withAlpha(rowIsSelected ? 0.78f : 0.14f));
            g.fillRect(0, 0, width, height);
            g.setColour(trackColour);
            g.fillRect(0, 0, 5, height);

            g.setColour(rowIsSelected ? trackTextColour(trackColour) : trackColour.brighter(0.18f));
            g.setFont(ui::sectionFont());
            g.drawText(track.name, 8, 2, width - 16, 18, juce::Justification::centredLeft, true);

            juce::String detail = track.trackType + " | "
                + (track.instrument.isNotEmpty() ? track.instrument : track.rackVst);
            if (trackFlags.size() > 0)
                detail << " | " << trackFlags.joinIntoString(" ");

            g.setFont(ui::font());
            g.drawText(detail, 8, 22, width - 16, 18, juce::Justification::centredLeft, true);
        }

        void selectedRowsChanged(int lastRowSelected) override
        {
            if (selectedTrackSetter != nullptr)
                selectedTrackSetter(lastRowSelected);
        }

    private:
        ProjectGetter projectGetter;
        SelectedTrackSetter selectedTrackSetter;
    };

    void configureButton(juce::TextButton& button, const juce::String& text)
    {
        button.setButtonText(text);
        button.setColour(juce::TextButton::buttonColourId, juce::Colour::fromRGB(48, 54, 66));
        button.setColour(juce::TextButton::textColourOffId, juce::Colours::white);
        addAndMakeVisible(button);
    }

    void configureEditor(juce::TextEditor& editor, const juce::String& placeholder)
    {
        editor.setMultiLine(false);
        editor.setScrollbarsShown(false);
        editor.setTextToShowWhenEmpty(placeholder, juce::Colour::fromRGB(122, 133, 149));
        editor.setColour(juce::TextEditor::backgroundColourId, juce::Colour::fromRGB(18, 20, 25));
        editor.setColour(juce::TextEditor::textColourId, juce::Colour::fromRGB(235, 239, 244));
        editor.setColour(juce::TextEditor::outlineColourId, juce::Colour::fromRGB(56, 64, 79));
        editor.setColour(juce::CaretComponent::caretColourId, juce::Colour::fromRGB(237, 242, 249));
        editor.setFont(ui::font());
        addAndMakeVisible(editor);
    }

    void configureSlider(juce::Slider& slider,
                         double minimum,
                         double maximum,
                         double step,
                         const juce::String& suffix)
    {
        slider.setSliderStyle(juce::Slider::LinearHorizontal);
        slider.setRange(minimum, maximum, step);
        slider.setChangeNotificationOnlyOnRelease(true);
        slider.setTextBoxStyle(juce::Slider::TextBoxRight, false, 60, 22);
        slider.setTextValueSuffix(suffix);
        addAndMakeVisible(slider);
    }

    void applyTrackMutation(std::function<void(TrackState&)> mutation, const juce::String& actionName)
    {
        if (mutateSelectedTrack != nullptr)
            mutateSelectedTrack(std::move(mutation), actionName);
    }

    void refreshInspectorControls()
    {
        const auto& project = projectGetter();
        const auto selectedTrack = selectedTrackGetter != nullptr ? selectedTrackGetter() : -1;
        const auto* track = juce::isPositiveAndBelow(selectedTrack, static_cast<int>(project.tracks.size()))
            ? &project.tracks[static_cast<size_t>(selectedTrack)]
            : nullptr;

        if (track != nullptr)
        {
            syncingControls = true;
            trackNameEditor.setText(track->name, juce::dontSendNotification);
            trackTypeEditor.setText(track->trackType, juce::dontSendNotification);
            instrumentModeEditor.setText(track->instrumentMode, juce::dontSendNotification);
            instrumentEditor.setText(track->instrument, juce::dontSendNotification);
            rackVstEditor.setText(track->rackVst, juce::dontSendNotification);
            midiChannelSlider.setValue(track->midiChannel + 1, juce::dontSendNotification);
            midiProgramSlider.setValue(track->midiProgram, juce::dontSendNotification);
            volumeSlider.setValue(track->volume, juce::dontSendNotification);
            panSlider.setValue(track->pan, juce::dontSendNotification);
            muteToggle.setToggleState(track->mute, juce::dontSendNotification);
            soloToggle.setToggleState(track->solo, juce::dontSendNotification);
            liveArmToggle.setToggleState(track->liveArmed, juce::dontSendNotification);
            syncingControls = false;

            trackNameEditor.setEnabled(true);
            trackTypeEditor.setEnabled(true);
            instrumentModeEditor.setEnabled(true);
            instrumentEditor.setEnabled(true);
            rackVstEditor.setEnabled(true);
            midiChannelSlider.setEnabled(true);
            midiProgramSlider.setEnabled(true);
            volumeSlider.setEnabled(true);
            panSlider.setEnabled(true);
            muteToggle.setEnabled(true);
            soloToggle.setEnabled(true);
            liveArmToggle.setEnabled(true);
            summaryEditor.setText(trackSummaryGetter != nullptr ? trackSummaryGetter(selectedTrack) : track->name, false);
            return;
        }

        syncingControls = true;
        trackNameEditor.setText({}, juce::dontSendNotification);
        trackTypeEditor.setText({}, juce::dontSendNotification);
        instrumentModeEditor.setText({}, juce::dontSendNotification);
        instrumentEditor.setText({}, juce::dontSendNotification);
        rackVstEditor.setText({}, juce::dontSendNotification);
        midiChannelSlider.setValue(1.0, juce::dontSendNotification);
        midiProgramSlider.setValue(0.0, juce::dontSendNotification);
        volumeSlider.setValue(0.8, juce::dontSendNotification);
        panSlider.setValue(0.0, juce::dontSendNotification);
        muteToggle.setToggleState(false, juce::dontSendNotification);
        soloToggle.setToggleState(false, juce::dontSendNotification);
        liveArmToggle.setToggleState(false, juce::dontSendNotification);
        syncingControls = false;

        trackNameEditor.setEnabled(false);
        trackTypeEditor.setEnabled(false);
        instrumentModeEditor.setEnabled(false);
        instrumentEditor.setEnabled(false);
        rackVstEditor.setEnabled(false);
        midiChannelSlider.setEnabled(false);
        midiProgramSlider.setEnabled(false);
        volumeSlider.setEnabled(false);
        panSlider.setEnabled(false);
        muteToggle.setEnabled(false);
        soloToggle.setEnabled(false);
        liveArmToggle.setEnabled(false);
        summaryEditor.setText("No track selected.", false);
    }

    ProjectGetter projectGetter;
    SelectedTrackGetter selectedTrackGetter;
    SelectedTrackSetter selectedTrackSetter;
    SelectedTrackMutator mutateSelectedTrack;
    TrackSummaryGetter trackSummaryGetter;
    std::function<void()> addTrack;
    std::function<void()> duplicateTrack;
    std::function<void()> removeTrack;
    std::function<void()> openRackEditor;
    std::function<void()> saveRackState;
    std::function<void()> playTrack;
    std::function<void()> stopPlayback;
    bool syncingControls = false;
    TrackListModel trackListModel;

    juce::Label titleLabel;
    juce::TextButton addTrackButton;
    juce::TextButton duplicateTrackButton;
    juce::TextButton removeTrackButton;
    juce::TextButton openRackEditorButton;
    juce::TextButton saveRackStateButton;
    juce::TextButton playTrackButton;
    juce::TextButton stopButton;
    juce::ListBox trackList;
    juce::Label inspectorLabel;
    juce::TextEditor trackNameEditor;
    juce::TextEditor trackTypeEditor;
    juce::TextEditor instrumentModeEditor;
    juce::TextEditor instrumentEditor;
    juce::TextEditor rackVstEditor;
    juce::Slider midiChannelSlider;
    juce::Slider midiProgramSlider;
    juce::Slider volumeSlider;
    juce::Slider panSlider;
    juce::ToggleButton muteToggle;
    juce::ToggleButton soloToggle;
    juce::ToggleButton liveArmToggle;
    juce::TextEditor summaryEditor;
};

class ModulationMatrixWindowComponent final : public juce::Component
{
public:
    using ProjectGetter = std::function<const ProjectState&()>;
    using AddInstrumentTrack = std::function<void(const juce::String&)>;
    using AddSharedEffectBus = std::function<void(const juce::String&, int)>;
    using ReplaceSharedEffectBus = std::function<void(const juce::String&, const juce::String&)>;
    using RemoveSharedEffectBus = std::function<void(const juce::String&)>;
    using RouteTrackTarget = std::function<void(int, const juce::String&)>;
    using OpenTrackEditor = std::function<void(int)>;
    using OpenSharedEffectEditor = std::function<void(const juce::String&)>;
    using ClearSharedEffectBusOutputs = std::function<void(const juce::String&)>;
    using SetSharedEffectBusOutputTargetEnabled = std::function<void(const juce::String&, const juce::String&, bool)>;

    ModulationMatrixWindowComponent(ProjectGetter projectGetterIn,
                                    AddInstrumentTrack addInstrumentTrackIn,
                                    AddSharedEffectBus addSharedEffectBusIn,
                                    ReplaceSharedEffectBus replaceSharedEffectBusIn,
                                    RemoveSharedEffectBus removeSharedEffectBusIn,
                                    RouteTrackTarget routeTrackTargetIn,
                                    OpenTrackEditor openTrackEditorIn,
                                    OpenSharedEffectEditor openSharedEffectEditorIn,
                                    ClearSharedEffectBusOutputs clearSharedEffectBusOutputsIn,
                                    SetSharedEffectBusOutputTargetEnabled setSharedEffectBusOutputTargetEnabledIn)
        : projectGetter(std::move(projectGetterIn)),
          addInstrumentTrack(std::move(addInstrumentTrackIn)),
          addSharedEffectBus(std::move(addSharedEffectBusIn)),
          replaceSharedEffectBus(std::move(replaceSharedEffectBusIn)),
          removeSharedEffectBus(std::move(removeSharedEffectBusIn)),
          routeTrackTarget(std::move(routeTrackTargetIn)),
          openTrackEditor(std::move(openTrackEditorIn)),
          openSharedEffectEditor(std::move(openSharedEffectEditorIn)),
          clearSharedEffectBusOutputs(std::move(clearSharedEffectBusOutputsIn)),
          setSharedEffectBusOutputTargetEnabled(std::move(setSharedEffectBusOutputTargetEnabledIn))
    {
        titleLabel.setText("Modulation Matrix", juce::dontSendNotification);
        titleLabel.setFont(ui::titleFont());
        titleLabel.setColour(juce::Label::textColourId, juce::Colour::fromRGB(235, 239, 244));
        addAndMakeVisible(titleLabel);

        hintLabel.setText("Double-click instruments or FX to open their editor. Drag nodes to place them. Drag from a node handle to route. Drop on empty space to disconnect.",
                          juce::dontSendNotification);
        hintLabel.setFont(ui::font());
        hintLabel.setColour(juce::Label::textColourId, juce::Colour::fromRGB(155, 166, 184));
        addAndMakeVisible(hintLabel);
    }

    void refreshFromModel()
    {
        syncNodeStateFromProject();
        rebuildNodes();
        repaint();
    }

    void paint(juce::Graphics& g) override
    {
        g.fillAll(juce::Colour::fromRGB(13, 15, 20));
        g.setColour(juce::Colour::fromRGB(31, 35, 44));
        g.drawRect(getLocalBounds(), 1);

        g.setColour(juce::Colour::fromRGB(16, 19, 25));
        g.fillRoundedRectangle(canvasBounds.toFloat(), 12.0f);

        g.setColour(juce::Colour::fromRGB(25, 29, 38));
        for (int x = canvasBounds.getX() + 40; x < canvasBounds.getRight(); x += 52)
            g.drawVerticalLine(x, static_cast<float>(canvasBounds.getY()), static_cast<float>(canvasBounds.getBottom()));
        for (int y = canvasBounds.getY() + 40; y < canvasBounds.getBottom(); y += 44)
            g.drawHorizontalLine(y, static_cast<float>(canvasBounds.getX()), static_cast<float>(canvasBounds.getRight()));

        drawConnections(g);
        drawNodes(g);

        if (nodes.empty())
        {
            g.setColour(juce::Colour::fromRGB(132, 142, 158));
            g.setFont(ui::sectionFont());
            g.drawText("Right-click to add an instrument or shared FX bus.",
                       canvasBounds.reduced(24),
                       juce::Justification::centred);
        }
    }

    void resized() override
    {
        auto area = getLocalBounds().reduced(14);
        titleLabel.setBounds(area.removeFromTop(28));
        hintLabel.setBounds(area.removeFromTop(20));
        area.removeFromTop(8);
        canvasBounds = area;
        rebuildNodes();
    }

    void mouseMove(const juce::MouseEvent& event) override
    {
        if (routeDragActive)
        {
            dragCurrentPosition = event.position;
            updateRouteDragTarget(event.position);
            repaint();
            return;
        }

        setMouseCursor(hitTestNode(event.position) != nullptr
                           ? juce::MouseCursor::PointingHandCursor
                           : juce::MouseCursor::NormalCursor);
    }

    void mouseExit(const juce::MouseEvent&) override
    {
        setMouseCursor(juce::MouseCursor::NormalCursor);
    }

    void mouseDown(const juce::MouseEvent& event) override
    {
        if (event.mods.isRightButtonDown())
        {
            showContextMenu(event.getScreenPosition(), event.position);
            return;
        }

        if (!event.mods.isLeftButtonDown())
            return;

        const auto hit = hitTestNodePart(event.position);
        if (hit.node == nullptr)
        {
            cancelRouteDrag();
            cancelNodeMove();
            pendingRouteTrackIndex = -1;
            repaint();
            return;
        }

        if (hit.part == NodePart::outputPort
            && (hit.node->kind == NodeKind::track || hit.node->kind == NodeKind::bus))
        {
            routeSourceKind = hit.node->kind == NodeKind::track ? RouteSourceKind::track : RouteSourceKind::bus;
            routeCandidateTrackIndex = hit.node->trackIndex;
            routeCandidateBusId = hit.node->busId;
            dragStartPosition = event.position;
            dragCurrentPosition = event.position;
            return;
        }

        if (hit.part == NodePart::body
            && (hit.node->kind == NodeKind::track || hit.node->kind == NodeKind::bus || hit.node->kind == NodeKind::master))
        {
            moveCandidateTrackIndex = -1;
            moveCandidateBusId.clear();
            moveCandidateMaster = false;
            moveCandidateTrackIndex = hit.node->kind == NodeKind::track ? hit.node->trackIndex : -1;
            moveCandidateBusId = hit.node->kind == NodeKind::bus ? hit.node->busId : juce::String();
            moveCandidateMaster = hit.node->kind == NodeKind::master;
            dragStartPosition = event.position;
            dragCurrentPosition = event.position;
            dragNodeOffset = event.position - hit.node->bounds.getPosition();
            if (hit.node->kind == NodeKind::track)
                pendingRouteTrackIndex = hit.node->trackIndex;
            repaint();
        }
    }

    void mouseDrag(const juce::MouseEvent& event) override
    {
        if (!event.mods.isLeftButtonDown())
            return;

        if (routeSourceKind != RouteSourceKind::none)
        {
            if (!routeDragActive)
            {
                if (event.position.getDistanceFrom(dragStartPosition) < 6.0f)
                    return;

                routeDragActive = true;
                dragRouteTrackIndex = routeSourceKind == RouteSourceKind::track ? routeCandidateTrackIndex : -1;
                dragRouteBusId = routeSourceKind == RouteSourceKind::bus ? routeCandidateBusId : juce::String();
                if (routeSourceKind == RouteSourceKind::track)
                    pendingRouteTrackIndex = dragRouteTrackIndex;
            }

            dragCurrentPosition = event.position;
            updateRouteDragTarget(event.position);
            repaint();
            return;
        }

        if (moveCandidateTrackIndex < 0 && moveCandidateBusId.isEmpty() && !moveCandidateMaster)
            return;

        if (!nodeMoveActive && event.position.getDistanceFrom(dragStartPosition) < 2.0f)
            return;

        nodeMoveActive = true;
        dragCurrentPosition = event.position;
        if (moveCandidateTrackIndex >= 0 && juce::isPositiveAndBelow(moveCandidateTrackIndex, static_cast<int>(trackNodePositions.size())))
        {
            trackNodePositions[static_cast<size_t>(moveCandidateTrackIndex)] =
                clampNodePosition(event.position - dragNodeOffset, trackNodeSize());
        }
        else if (moveCandidateBusId.isNotEmpty())
        {
            setBusNodePosition(moveCandidateBusId,
                               clampNodePosition(event.position - dragNodeOffset, busNodeSize()));
        }
        else if (moveCandidateMaster)
        {
            masterNodePosition = clampNodePosition(event.position - dragNodeOffset, masterNodeSize());
        }

        rebuildNodes();
        repaint();
    }

    void mouseUp(const juce::MouseEvent& event) override
    {
        if (routeDragActive)
        {
            updateRouteDragTarget(event.position);
            applyRouteDrag();
            return;
        }

        cancelRouteDrag();
        cancelNodeMove();
        repaint();
    }

    void mouseDoubleClick(const juce::MouseEvent& event) override
    {
        const auto hit = hitTestNodePart(event.position);
        if (hit.node == nullptr || hit.part != NodePart::body)
            return;

        if (hit.node->kind == NodeKind::track)
        {
            if (openTrackEditor != nullptr)
                openTrackEditor(hit.node->trackIndex);
            return;
        }

        if (hit.node->kind == NodeKind::bus && openSharedEffectEditor != nullptr)
            openSharedEffectEditor(hit.node->busId);
    }

private:
    enum class NodeKind
    {
        track,
        bus,
        master
    };

    enum class NodePart
    {
        none,
        body,
        inputPort,
        outputPort
    };

    enum class RouteSourceKind
    {
        none,
        track,
        bus
    };

    struct Node
    {
        NodeKind kind = NodeKind::track;
        juce::Rectangle<float> bounds;
        juce::Rectangle<float> inputPortBounds;
        juce::Rectangle<float> outputPortBounds;
        int trackIndex = -1;
        juce::String busId;
        juce::String title;
        juce::String subtitle;
        juce::Colour colour;
    };

    struct HitResult
    {
        const Node* node = nullptr;
        NodePart part = NodePart::none;
    };

    void rebuildNodes()
    {
        nodes.clear();
        if (canvasBounds.isEmpty())
            return;

        const auto& project = projectGetter();
        trackNodePositions.resize(project.tracks.size(), { -1.0f, -1.0f });
        const auto content = canvasBounds.reduced(26, 24);
        if (content.isEmpty())
            return;

        const auto trackSize = trackNodeSize();
        const auto busSize = busNodeSize();
        const auto masterSize = masterNodeSize();
        const auto defaultMasterPosition = juce::Point<float>(
            static_cast<float>(content.getCentreX()) - masterSize.x * 0.5f,
            static_cast<float>(content.getCentreY()) - masterSize.y * 0.5f);
        const auto clampedMasterPosition = clampNodePosition(
            (masterNodePosition.x < 0.0f || masterNodePosition.y < 0.0f) ? defaultMasterPosition : masterNodePosition,
            masterSize);
        masterNodePosition = clampedMasterPosition;
        const juce::Rectangle<float> masterBounds(
            clampedMasterPosition.x,
            clampedMasterPosition.y,
            masterSize.x,
            masterSize.y);

        for (int trackIndex = 0; trackIndex < static_cast<int>(project.tracks.size()); ++trackIndex)
        {
            const auto& track = project.tracks[static_cast<size_t>(trackIndex)];
            Node node;
            node.kind = NodeKind::track;
            node.trackIndex = trackIndex;
            auto position = trackNodePositions[static_cast<size_t>(trackIndex)];
            if (position.x < 0.0f || position.y < 0.0f)
            {
                position = pendingTrackPlacement.x >= 0.0f
                    ? clampNodePosition(pendingTrackPlacement, trackSize)
                    : defaultTrackNodePosition(trackIndex,
                                               static_cast<int>(project.tracks.size()),
                                               content,
                                               masterBounds,
                                               trackSize);
                trackNodePositions[static_cast<size_t>(trackIndex)] = position;
                if (pendingTrackPlacement.x >= 0.0f)
                    pendingTrackPlacement = { -1.0f, -1.0f };
            }
            const auto clampedPosition = clampNodePosition(position, trackSize);
            node.bounds = juce::Rectangle<float>(clampedPosition.x,
                                                 clampedPosition.y,
                                                 trackSize.x,
                                                 trackSize.y);
            node.title = track.name.trim().isNotEmpty() ? track.name.trim() : ("Track " + juce::String(trackIndex + 1));
            const auto rackName = displayRackName(project, track).trim();
            if (track.trackType.equalsIgnoreCase("sample"))
                node.subtitle = "Sample";
            else if (rackName.isNotEmpty())
                node.subtitle = rackName;
            else if (track.instrument.trim().isNotEmpty())
                node.subtitle = track.instrument.trim();
            else
                node.subtitle = track.trackType.trim().isNotEmpty() ? track.trackType.trim() : "Instrument";
            node.colour = trackDisplayColour(track, trackIndex);
            assignNodePortBounds(node);
            nodes.push_back(std::move(node));
        }

        for (int busIndex = 0; busIndex < static_cast<int>(project.sharedFxBuses.size()); ++busIndex)
        {
            const auto& bus = project.sharedFxBuses[static_cast<size_t>(busIndex)];
            Node node;
            node.kind = NodeKind::bus;
            node.busId = bus.id.trim();
            auto position = findBusNodePosition(bus.id);
            if (position.x < 0.0f || position.y < 0.0f)
            {
                position = pendingBusPlacement.x >= 0.0f
                    ? clampNodePosition(pendingBusPlacement, busSize)
                    : defaultBusNodePosition(busIndex,
                                             static_cast<int>(project.sharedFxBuses.size()),
                                             content,
                                             masterBounds,
                                             busSize);
                setBusNodePosition(bus.id, position);
                if (pendingBusPlacement.x >= 0.0f)
                    pendingBusPlacement = { -1.0f, -1.0f };
            }
            const auto clampedPosition = clampNodePosition(position, busSize);
            node.bounds = juce::Rectangle<float>(clampedPosition.x,
                                                 clampedPosition.y,
                                                 busSize.x,
                                                 busSize.y);
            node.title = bus.name.trim().isNotEmpty() ? bus.name.trim() : ("FX Bus " + juce::String(busIndex + 1));
            if (const auto* entry = findRackEntryByReference(project, bus.effect, true))
                node.subtitle = rackEntryDisplayName(*entry);
            else
                node.subtitle = bus.effect.trim().isNotEmpty() ? bus.effect.trim() : "Shared Effect";
            node.colour = juce::Colour::fromRGB(74, 119, 193);
            assignNodePortBounds(node);
            nodes.push_back(std::move(node));
        }

        Node masterNode;
        masterNode.kind = NodeKind::master;
        masterNode.bounds = masterBounds;
        masterNode.title = "Master Out";
        masterNode.subtitle = project.masterFxChain.size() == 0
            ? "Final output"
            : (juce::String(project.masterFxChain.size()) + " master FX");
        masterNode.colour = juce::Colour::fromRGB(86, 190, 152);
        assignNodePortBounds(masterNode);
        nodes.push_back(std::move(masterNode));
    }

    void syncNodeStateFromProject()
    {
        const auto& project = projectGetter();
        if (!juce::isPositiveAndBelow(pendingRouteTrackIndex, static_cast<int>(project.tracks.size())))
            pendingRouteTrackIndex = -1;
        if (!juce::isPositiveAndBelow(routeCandidateTrackIndex, static_cast<int>(project.tracks.size())))
            routeCandidateTrackIndex = -1;
        if (!juce::isPositiveAndBelow(dragRouteTrackIndex, static_cast<int>(project.tracks.size())))
            dragRouteTrackIndex = -1;
        if (!juce::isPositiveAndBelow(moveCandidateTrackIndex, static_cast<int>(project.tracks.size())))
            moveCandidateTrackIndex = -1;

        trackNodePositions.resize(project.tracks.size(), { -1.0f, -1.0f });
        busNodePositions.erase(std::remove_if(busNodePositions.begin(),
                                              busNodePositions.end(),
                                              [&project] (const auto& entry)
                                              {
                                                  return std::none_of(project.sharedFxBuses.begin(),
                                                                      project.sharedFxBuses.end(),
                                                                      [&entry] (const SharedEffectBusState& bus)
                                                                      {
                                                                          return bus.id.equalsIgnoreCase(entry.first);
                                                                      });
                                              }),
                               busNodePositions.end());
    }

    void drawConnections(juce::Graphics& g)
    {
        const auto& project = projectGetter();
        const auto* masterNode = findNode(NodeKind::master, {}, -1);
        if (masterNode == nullptr)
            return;

        for (const auto& node : nodes)
        {
            if (node.kind != NodeKind::track)
                continue;

            juce::Point<float> endPoint;
            juce::Colour lineColour = node.colour.withAlpha(0.78f);
            const auto targetId = juce::isPositiveAndBelow(node.trackIndex, static_cast<int>(project.tracks.size()))
                ? project.tracks[static_cast<size_t>(node.trackIndex)].routingTarget.trim()
                : juce::String();
            if (targetId.equalsIgnoreCase("none"))
                continue;

            if (const auto* busNode = findNode(NodeKind::bus, targetId, -1))
                endPoint = busNode->inputPortBounds.getCentre();
            else
                endPoint = masterNode->inputPortBounds.getCentre();

            drawArrow(g,
                      node.outputPortBounds.getCentre(),
                      endPoint,
                      lineColour,
                      pendingRouteTrackIndex == node.trackIndex ? 3.0f : 2.2f);
        }

        for (const auto& node : nodes)
        {
            if (node.kind != NodeKind::bus)
                continue;

            const auto busIt = std::find_if(project.sharedFxBuses.begin(),
                                            project.sharedFxBuses.end(),
                                            [&node] (const SharedEffectBusState& bus)
                                            {
                                                return bus.id.equalsIgnoreCase(node.busId);
                                            });
            if (busIt == project.sharedFxBuses.end())
                continue;

            for (const auto& target : busIt->outputTargets)
            {
                auto endPoint = masterNode->inputPortBounds.getCentre();
                if (const auto* busNode = findNode(NodeKind::bus, target, -1))
                    endPoint = busNode->inputPortBounds.getCentre();

                drawArrow(g,
                          node.outputPortBounds.getCentre(),
                          endPoint,
                          node.colour.withAlpha(0.85f),
                          2.4f);
            }
        }

        if (routeDragActive)
        {
            const auto* sourceNode = routeSourceKind == RouteSourceKind::track
                ? findNode(NodeKind::track, {}, dragRouteTrackIndex)
                : findNode(NodeKind::bus, dragRouteBusId, -1);
            if (sourceNode != nullptr)
            {
                auto endPoint = dragCurrentPosition;
                if (dragHoverMaster && masterNode != nullptr)
                    endPoint = masterNode->inputPortBounds.getCentre();
                else if (const auto* busNode = findNode(NodeKind::bus, dragHoverBusId, -1))
                    endPoint = busNode->inputPortBounds.getCentre();

                drawArrow(g,
                          sourceNode->outputPortBounds.getCentre(),
                          endPoint,
                          sourceNode->colour.withAlpha(0.95f),
                          3.4f);
            }
        }
    }

    void drawNodes(juce::Graphics& g)
    {
        for (const auto& node : nodes)
        {
            const auto isPendingTrack = node.kind == NodeKind::track && node.trackIndex == pendingRouteTrackIndex;
            const auto isDragTrack = (node.kind == NodeKind::track && node.trackIndex == dragRouteTrackIndex)
                || (node.kind == NodeKind::bus && node.busId.equalsIgnoreCase(dragRouteBusId));
            const auto isHoverTarget = (node.kind == NodeKind::master && dragHoverMaster)
                || (node.kind == NodeKind::bus && node.busId.equalsIgnoreCase(dragHoverBusId));
            const auto baseColour = node.kind == NodeKind::master
                ? node.colour.withAlpha(0.24f)
                : node.colour.withAlpha((isPendingTrack || isDragTrack) ? 0.34f : 0.2f);

            g.setColour(baseColour);
            g.fillRoundedRectangle(node.bounds, 12.0f);

            g.setColour(node.colour.withAlpha((isPendingTrack || isDragTrack || isHoverTarget) ? 1.0f : 0.82f));
            g.drawRoundedRectangle(node.bounds, 12.0f, (isPendingTrack || isDragTrack || isHoverTarget) ? 2.6f : 1.6f);

            auto textBounds = node.bounds.toNearestInt().reduced(12, 10);
            g.setColour(juce::Colour::fromRGB(242, 246, 252));
            g.setFont(ui::sectionFont());
            g.drawText(node.title, textBounds.removeFromTop(20), juce::Justification::centredLeft, true);
            g.setColour(juce::Colour::fromRGB(171, 182, 198));
            g.setFont(ui::font());
            g.drawText(node.subtitle, textBounds, juce::Justification::centredLeft, true);

            auto drawPort = [&g, &node] (juce::Rectangle<float> bounds, bool highlighted)
            {
                const auto colour = node.colour.withAlpha(highlighted ? 1.0f : 0.8f);
                g.setColour(colour.withAlpha(0.22f));
                g.fillEllipse(bounds.expanded(2.0f));
                g.setColour(colour);
                g.fillEllipse(bounds);
            };

            drawPort(node.inputPortBounds, isHoverTarget);
            if (node.kind != NodeKind::master)
                drawPort(node.outputPortBounds, isDragTrack);
        }
    }

    HitResult hitTestNodePart(juce::Point<float> position) const
    {
        for (const auto& node : nodes)
        {
            if (node.outputPortBounds.contains(position) && node.kind != NodeKind::master)
                return { &node, NodePart::outputPort };
            if (node.inputPortBounds.contains(position))
                return { &node, NodePart::inputPort };
            if (node.bounds.contains(position))
                return { &node, NodePart::body };
        }

        return {};
    }

    const Node* hitTestNode(juce::Point<float> position) const
    {
        return hitTestNodePart(position).node;
    }

    const Node* findNode(NodeKind kind, const juce::String& busId, int trackIndex) const
    {
        for (const auto& node : nodes)
        {
            if (node.kind != kind)
                continue;
            if (kind == NodeKind::bus && !node.busId.equalsIgnoreCase(busId))
                continue;
            if (kind == NodeKind::track && node.trackIndex != trackIndex)
                continue;
            return &node;
        }
        return nullptr;
    }

    void updateRouteDragTarget(juce::Point<float> position)
    {
        dragHoverBusId.clear();
        dragHoverMaster = false;

        if (const auto* node = hitTestNode(position))
        {
            if (node->kind == NodeKind::bus
                && (routeSourceKind != RouteSourceKind::bus || !node->busId.equalsIgnoreCase(routeCandidateBusId)))
                dragHoverBusId = node->busId;
            else if (node->kind == NodeKind::master)
                dragHoverMaster = true;
        }
    }

    void applyRouteDrag()
    {
        if (routeSourceKind == RouteSourceKind::track
            && routeTrackTarget != nullptr
            && dragRouteTrackIndex >= 0)
        {
            if (dragHoverMaster)
                routeTrackTarget(dragRouteTrackIndex, "master");
            else if (dragHoverBusId.isNotEmpty())
                routeTrackTarget(dragRouteTrackIndex, dragHoverBusId);
            else
                routeTrackTarget(dragRouteTrackIndex, "none");
        }
        else if (routeSourceKind == RouteSourceKind::bus && dragRouteBusId.isNotEmpty())
        {
            if (dragHoverMaster && setSharedEffectBusOutputTargetEnabled != nullptr)
                setSharedEffectBusOutputTargetEnabled(dragRouteBusId, "master", true);
            else if (dragHoverBusId.isNotEmpty() && setSharedEffectBusOutputTargetEnabled != nullptr)
                setSharedEffectBusOutputTargetEnabled(dragRouteBusId, dragHoverBusId, true);
            else if (clearSharedEffectBusOutputs != nullptr)
                clearSharedEffectBusOutputs(dragRouteBusId);
        }

        cancelRouteDrag();
        repaint();
    }

    void cancelRouteDrag()
    {
        routeSourceKind = RouteSourceKind::none;
        routeCandidateTrackIndex = -1;
        routeCandidateBusId.clear();
        routeDragActive = false;
        dragRouteTrackIndex = -1;
        dragRouteBusId.clear();
        pendingRouteTrackIndex = -1;
        dragHoverBusId.clear();
        dragHoverMaster = false;
    }

    void cancelNodeMove()
    {
        moveCandidateTrackIndex = -1;
        moveCandidateBusId.clear();
        moveCandidateMaster = false;
        nodeMoveActive = false;
    }

    void assignNodePortBounds(Node& node) const
    {
        constexpr float portSize = 10.0f;
        node.inputPortBounds = { node.bounds.getX() - portSize * 0.5f,
                                 node.bounds.getCentreY() - portSize * 0.5f,
                                 portSize,
                                 portSize };
        node.outputPortBounds = { node.bounds.getRight() - portSize * 0.5f,
                                  node.bounds.getCentreY() - portSize * 0.5f,
                                  portSize,
                                  portSize };
    }

    static juce::Point<float> trackNodeSize() { return { 244.0f, 64.0f }; }
    static juce::Point<float> busNodeSize() { return { 212.0f, 60.0f }; }
    static juce::Point<float> masterNodeSize() { return { 188.0f, 92.0f }; }

    juce::Point<float> clampNodePosition(juce::Point<float> position, juce::Point<float> nodeSize) const
    {
        return {
            juce::jlimit(static_cast<float>(canvasBounds.getX()) + 8.0f,
                         static_cast<float>(canvasBounds.getRight()) - nodeSize.x - 8.0f,
                         position.x),
            juce::jlimit(static_cast<float>(canvasBounds.getY()) + 8.0f,
                         static_cast<float>(canvasBounds.getBottom()) - nodeSize.y - 8.0f,
                         position.y)
        };
    }

    static juce::Point<float> radialNodePosition(int index,
                                                 int count,
                                                 juce::Rectangle<int> content,
                                                 juce::Rectangle<float> masterBounds,
                                                 juce::Point<float> nodeSize,
                                                 float startAngle,
                                                 float endAngle)
    {
        const auto t = count <= 1 ? 0.5f : static_cast<float>(index) / static_cast<float>(juce::jmax(1, count - 1));
        const auto angle = juce::jmap(t, startAngle, endAngle);
        const auto radiusX = juce::jmax(120.0f, static_cast<float>(content.getWidth()) * 0.32f);
        const auto radiusY = juce::jmax(100.0f, static_cast<float>(content.getHeight()) * 0.33f);
        return {
            masterBounds.getCentreX() + std::cos(angle) * radiusX - nodeSize.x * 0.5f,
            masterBounds.getCentreY() + std::sin(angle) * radiusY - nodeSize.y * 0.5f
        };
    }

    static juce::Point<float> defaultTrackNodePosition(int index,
                                                       int count,
                                                       juce::Rectangle<int> content,
                                                       juce::Rectangle<float> masterBounds,
                                                       juce::Point<float> nodeSize)
    {
        return radialNodePosition(index,
                                  count,
                                  content,
                                  masterBounds,
                                  nodeSize,
                                  juce::MathConstants<float>::pi * 0.78f,
                                  juce::MathConstants<float>::pi * 1.22f);
    }

    static juce::Point<float> defaultBusNodePosition(int index,
                                                     int count,
                                                     juce::Rectangle<int> content,
                                                     juce::Rectangle<float> masterBounds,
                                                     juce::Point<float> nodeSize)
    {
        return radialNodePosition(index,
                                  count,
                                  content,
                                  masterBounds,
                                  nodeSize,
                                  juce::MathConstants<float>::pi * -0.22f,
                                  juce::MathConstants<float>::pi * 0.22f);
    }

    juce::Point<float> findBusNodePosition(const juce::String& busId) const
    {
        for (const auto& entry : busNodePositions)
        {
            if (entry.first.equalsIgnoreCase(busId))
                return entry.second;
        }
        return { -1.0f, -1.0f };
    }

    void setBusNodePosition(const juce::String& busId, juce::Point<float> position)
    {
        for (auto& entry : busNodePositions)
        {
            if (entry.first.equalsIgnoreCase(busId))
            {
                entry.second = position;
                return;
            }
        }

        busNodePositions.push_back({ busId, position });
    }

    static void drawArrow(juce::Graphics& g,
                          juce::Point<float> start,
                          juce::Point<float> end,
                          juce::Colour colour,
                          float thickness)
    {
        juce::Path line;
        line.startNewSubPath(start);
        line.lineTo(end);
        g.setColour(colour);
        g.strokePath(line, juce::PathStrokeType(thickness, juce::PathStrokeType::curved, juce::PathStrokeType::rounded));

        auto delta = end - start;
        const auto length = delta.getDistanceFromOrigin();
        if (length <= 1.0f)
            return;

        delta /= length;
        const juce::Point<float> orthogonal(-delta.y, delta.x);
        const auto tip = end;
        const auto base = end - delta * 12.0f;
        juce::Path head;
        head.addTriangle(tip, base + orthogonal * 5.5f, base - orthogonal * 5.5f);
        g.fillPath(head);
    }

    void showContextMenu(juce::Point<int> screenPosition, juce::Point<float> localPosition)
    {
        if (const auto* node = hitTestNode(localPosition))
        {
            switch (node->kind)
            {
                case NodeKind::track: showTrackMenu(*node, screenPosition); break;
                case NodeKind::bus: showBusMenu(*node, screenPosition); break;
                case NodeKind::master: showMasterMenu(screenPosition); break;
            }
            return;
        }

        showBackgroundMenu(screenPosition, localPosition);
    }

    void showBackgroundMenu(juce::Point<int> screenPosition, juce::Point<float> localPosition)
    {
        const auto& project = projectGetter();
        juce::PopupMenu menu;
        juce::PopupMenu instrumentsMenu;
        juce::PopupMenu effectsMenu;
        std::vector<juce::String> instrumentReferences;
        std::vector<juce::String> effectReferences;

        constexpr int instrumentBaseId = 1000;
        constexpr int effectBaseId = 2000;

        for (const auto& entry : project.vstRack)
        {
            const auto label = rackEntryDisplayName(entry);
            if (label.isEmpty())
                continue;

            if (entry.isInstrument)
            {
                instrumentReferences.push_back(entry.path.isNotEmpty() ? entry.path : label);
                instrumentsMenu.addItem(instrumentBaseId + static_cast<int>(instrumentReferences.size()) - 1, label);
            }

            if (entry.isEffect)
            {
                effectReferences.push_back(entry.path.isNotEmpty() ? entry.path : label);
                effectsMenu.addItem(effectBaseId + static_cast<int>(effectReferences.size()) - 1, label);
            }
        }

        menu.addSubMenu("Add Instrument Track", instrumentsMenu, !instrumentReferences.empty());
        menu.addSubMenu("Add Shared Effect", effectsMenu, !effectReferences.empty());

        auto options = juce::PopupMenu::Options()
            .withTargetScreenArea(juce::Rectangle<int>(screenPosition.x, screenPosition.y, 1, 1))
            .withMinimumWidth(240);

        menu.showMenuAsync(options,
                           [safeThis = juce::Component::SafePointer<ModulationMatrixWindowComponent>(this),
                            instrumentPlacement = clampNodePosition(localPosition - trackNodeSize() * 0.5f, trackNodeSize()),
                            effectPlacement = clampNodePosition(localPosition - busNodeSize() * 0.5f, busNodeSize()),
                            instrumentReferences,
                            effectReferences] (int result)
                           {
                               if (safeThis == nullptr || result == 0)
                                   return;

                               if (result >= 1000 && result < 1000 + static_cast<int>(instrumentReferences.size()))
                               {
                                   if (safeThis->addInstrumentTrack != nullptr)
                                   {
                                       safeThis->pendingTrackPlacement = instrumentPlacement;
                                       safeThis->addInstrumentTrack(instrumentReferences[static_cast<size_t>(result - 1000)]);
                                   }
                                   return;
                               }

                               if (result >= 2000 && result < 2000 + static_cast<int>(effectReferences.size()))
                               {
                                    if (safeThis->addSharedEffectBus != nullptr)
                                    {
                                        safeThis->pendingBusPlacement = effectPlacement;
                                        safeThis->addSharedEffectBus(effectReferences[static_cast<size_t>(result - 2000)], -1);
                                    }
                               }
                           });
    }

    void showTrackMenu(const Node& node, juce::Point<int> screenPosition)
    {
        const auto& project = projectGetter();
        if (!juce::isPositiveAndBelow(node.trackIndex, static_cast<int>(project.tracks.size())))
            return;

        const auto& track = project.tracks[static_cast<size_t>(node.trackIndex)];
        juce::PopupMenu menu;
        juce::PopupMenu routeMenu;
        juce::PopupMenu effectsMenu;
        std::vector<juce::String> busIds;
        std::vector<juce::String> effectReferences;

        constexpr int openEditorId = 1;
        constexpr int routeMasterId = 10;
        constexpr int routeDisconnectId = 11;
        constexpr int routeBusBaseId = 100;
        constexpr int insertEffectBaseId = 1000;

        menu.addItem(openEditorId, "Open Instrument Editor");
        menu.addSeparator();

        routeMenu.addItem(routeMasterId, "Master Out", true, track.routingTarget.equalsIgnoreCase("master"));
        for (const auto& bus : project.sharedFxBuses)
        {
            busIds.push_back(bus.id.trim());
            routeMenu.addItem(routeBusBaseId + static_cast<int>(busIds.size()) - 1,
                              bus.name.trim().isNotEmpty() ? bus.name.trim() : "FX Bus",
                              true,
                              track.routingTarget.equalsIgnoreCase(bus.id));
        }
        routeMenu.addSeparator();
        routeMenu.addItem(routeDisconnectId, "Disconnected", true, track.routingTarget.equalsIgnoreCase("none"));

        for (const auto& entry : project.vstRack)
        {
            if (!entry.isEffect)
                continue;

            const auto label = rackEntryDisplayName(entry);
            if (label.isEmpty())
                continue;

            effectReferences.push_back(entry.path.isNotEmpty() ? entry.path : label);
            effectsMenu.addItem(insertEffectBaseId + static_cast<int>(effectReferences.size()) - 1, label);
        }

        menu.addSubMenu("Route To", routeMenu, true);
        menu.addSubMenu("Insert Shared Effect", effectsMenu, !effectReferences.empty());

        auto options = juce::PopupMenu::Options()
            .withTargetScreenArea(juce::Rectangle<int>(screenPosition.x, screenPosition.y, 1, 1))
            .withMinimumWidth(240);

        menu.showMenuAsync(options,
                           [safeThis = juce::Component::SafePointer<ModulationMatrixWindowComponent>(this),
                            trackIndex = node.trackIndex,
                            busPlacement = clampNodePosition(juce::Point<float>(node.bounds.getRight() + 36.0f,
                                                                                node.bounds.getY() - 6.0f),
                                                             busNodeSize()),
                            busIds,
                            effectReferences] (int result)
                           {
                               if (safeThis == nullptr || result == 0)
                                   return;

                               if (result == 1)
                               {
                                   if (safeThis->openTrackEditor != nullptr)
                                       safeThis->openTrackEditor(trackIndex);
                                   return;
                               }

                               if (result == 10)
                               {
                                   if (safeThis->routeTrackTarget != nullptr)
                                       safeThis->routeTrackTarget(trackIndex, "master");
                                   return;
                               }

                               if (result == 11)
                               {
                                   if (safeThis->routeTrackTarget != nullptr)
                                       safeThis->routeTrackTarget(trackIndex, "none");
                                   return;
                               }

                               if (result >= 100 && result < 100 + static_cast<int>(busIds.size()))
                               {
                                   if (safeThis->routeTrackTarget != nullptr)
                                       safeThis->routeTrackTarget(trackIndex, busIds[static_cast<size_t>(result - 100)]);
                                   return;
                               }

                               if (result >= 1000 && result < 1000 + static_cast<int>(effectReferences.size()))
                               {
                                   if (safeThis->addSharedEffectBus != nullptr)
                                   {
                                       safeThis->pendingBusPlacement = busPlacement;
                                       safeThis->addSharedEffectBus(effectReferences[static_cast<size_t>(result - 1000)],
                                                                   trackIndex);
                                   }
                               }
                           });
    }

    void showBusMenu(const Node& node, juce::Point<int> screenPosition)
    {
        const auto& project = projectGetter();
        juce::PopupMenu menu;
        juce::PopupMenu outputsMenu;
        juce::PopupMenu replaceMenu;
        std::vector<juce::String> effectReferences;
        std::vector<juce::String> outputTargets;

        constexpr int openEditorId = 1;
        constexpr int routeSelectedTrackHereId = 2;
        constexpr int clearOutputsId = 3;
        constexpr int outputBaseId = 100;
        constexpr int removeBusId = 900;
        constexpr int replaceEffectBaseId = 1000;

        menu.addItem(openEditorId, "Open FX Editor");

        if (juce::isPositiveAndBelow(pendingRouteTrackIndex, static_cast<int>(project.tracks.size())))
        {
            menu.addSeparator();
            menu.addItem(routeSelectedTrackHereId,
                         "Route " + project.tracks[static_cast<size_t>(pendingRouteTrackIndex)].name + " Here");
            menu.addSeparator();
        }

        auto enabledTargets = juce::StringArray();
        if (const auto busIt = std::find_if(project.sharedFxBuses.begin(),
                                            project.sharedFxBuses.end(),
                                            [&node] (const SharedEffectBusState& bus)
                                            {
                                                return bus.id.equalsIgnoreCase(node.busId);
                                            });
            busIt != project.sharedFxBuses.end())
        {
            enabledTargets = busIt->outputTargets;
        }

        outputTargets.push_back("master");
        outputsMenu.addItem(outputBaseId, "Master Out", true, enabledTargets.contains("master", true));
        for (const auto& bus : project.sharedFxBuses)
        {
            if (bus.id.equalsIgnoreCase(node.busId))
                continue;

            outputTargets.push_back(bus.id);
            outputsMenu.addItem(outputBaseId + static_cast<int>(outputTargets.size()) - 1,
                                bus.name.trim().isNotEmpty() ? bus.name.trim() : "FX Bus",
                                true,
                                enabledTargets.contains(bus.id, true));
        }

        for (const auto& entry : project.vstRack)
        {
            if (!entry.isEffect)
                continue;

            const auto label = rackEntryDisplayName(entry);
            if (label.isEmpty())
                continue;

            effectReferences.push_back(entry.path.isNotEmpty() ? entry.path : label);
            replaceMenu.addItem(replaceEffectBaseId + static_cast<int>(effectReferences.size()) - 1, label);
        }

        menu.addSubMenu("Outputs", outputsMenu, true);
        menu.addItem(clearOutputsId, "Disconnect All Outputs", !enabledTargets.isEmpty());
        menu.addSubMenu("Replace Effect", replaceMenu, !effectReferences.empty());
        menu.addItem(removeBusId, "Remove Effect Bus");

        auto options = juce::PopupMenu::Options()
            .withTargetScreenArea(juce::Rectangle<int>(screenPosition.x, screenPosition.y, 1, 1))
            .withMinimumWidth(240);

        menu.showMenuAsync(options,
                           [safeThis = juce::Component::SafePointer<ModulationMatrixWindowComponent>(this),
                            busId = node.busId,
                            outputTargets,
                            effectReferences] (int result)
                           {
                               if (safeThis == nullptr || result == 0)
                                   return;

                               if (result == 1)
                               {
                                   if (safeThis->openSharedEffectEditor != nullptr)
                                       safeThis->openSharedEffectEditor(busId);
                                   return;
                               }

                               if (result == 2)
                               {
                                   if (safeThis->routeTrackTarget != nullptr && safeThis->pendingRouteTrackIndex >= 0)
                                   {
                                       safeThis->routeTrackTarget(safeThis->pendingRouteTrackIndex, busId);
                                       safeThis->pendingRouteTrackIndex = -1;
                                       safeThis->repaint();
                                   }
                                   return;
                               }

                               if (result == 3)
                               {
                                   if (safeThis->clearSharedEffectBusOutputs != nullptr)
                                       safeThis->clearSharedEffectBusOutputs(busId);
                                   return;
                               }

                               if (result >= 100 && result < 100 + static_cast<int>(outputTargets.size()))
                               {
                                   if (safeThis->setSharedEffectBusOutputTargetEnabled != nullptr)
                                   {
                                       const auto targetId = outputTargets[static_cast<size_t>(result - 100)];
                                       const auto& project = safeThis->projectGetter();
                                       const auto busIt = std::find_if(project.sharedFxBuses.begin(),
                                                                       project.sharedFxBuses.end(),
                                                                       [busId] (const SharedEffectBusState& bus)
                                                                       {
                                                                           return bus.id.equalsIgnoreCase(busId);
                                                                       });
                                       if (busIt != project.sharedFxBuses.end())
                                       {
                                           const auto enable = !busIt->outputTargets.contains(targetId, true);
                                           safeThis->setSharedEffectBusOutputTargetEnabled(busId, targetId, enable);
                                       }
                                   }
                                   return;
                               }

                               if (result == 900)
                               {
                                   if (safeThis->removeSharedEffectBus != nullptr)
                                       safeThis->removeSharedEffectBus(busId);
                                   return;
                               }

                               if (result >= 1000 && result < 1000 + static_cast<int>(effectReferences.size()))
                               {
                                   if (safeThis->replaceSharedEffectBus != nullptr)
                                   {
                                       safeThis->replaceSharedEffectBus(busId,
                                                                       effectReferences[static_cast<size_t>(result - 1000)]);
                                   }
                               }
                           });
    }

    void showMasterMenu(juce::Point<int> screenPosition)
    {
        const auto& project = projectGetter();
        if (!juce::isPositiveAndBelow(pendingRouteTrackIndex, static_cast<int>(project.tracks.size())))
            return;

        juce::PopupMenu menu;
        menu.addItem(1, "Route " + project.tracks[static_cast<size_t>(pendingRouteTrackIndex)].name + " to Master Out");

        auto options = juce::PopupMenu::Options()
            .withTargetScreenArea(juce::Rectangle<int>(screenPosition.x, screenPosition.y, 1, 1))
            .withMinimumWidth(240);

        menu.showMenuAsync(options,
                           [safeThis = juce::Component::SafePointer<ModulationMatrixWindowComponent>(this)] (int result)
                           {
                               if (safeThis == nullptr || result != 1 || safeThis->pendingRouteTrackIndex < 0)
                                   return;

                               if (safeThis->routeTrackTarget != nullptr)
                                   safeThis->routeTrackTarget(safeThis->pendingRouteTrackIndex, "master");
                               safeThis->pendingRouteTrackIndex = -1;
                               safeThis->repaint();
                           });
    }

    ProjectGetter projectGetter;
    AddInstrumentTrack addInstrumentTrack;
    AddSharedEffectBus addSharedEffectBus;
    ReplaceSharedEffectBus replaceSharedEffectBus;
    RemoveSharedEffectBus removeSharedEffectBus;
    RouteTrackTarget routeTrackTarget;
    OpenTrackEditor openTrackEditor;
    OpenSharedEffectEditor openSharedEffectEditor;
    ClearSharedEffectBusOutputs clearSharedEffectBusOutputs;
    SetSharedEffectBusOutputTargetEnabled setSharedEffectBusOutputTargetEnabled;
    juce::Label titleLabel;
    juce::Label hintLabel;
    juce::Rectangle<int> canvasBounds;
    std::vector<Node> nodes;
    int pendingRouteTrackIndex = -1;
    int routeCandidateTrackIndex = -1;
    juce::String routeCandidateBusId;
    int dragRouteTrackIndex = -1;
    juce::String dragRouteBusId;
    int moveCandidateTrackIndex = -1;
    juce::String moveCandidateBusId;
    bool moveCandidateMaster = false;
    bool nodeMoveActive = false;
    RouteSourceKind routeSourceKind = RouteSourceKind::none;
    bool routeDragActive = false;
    bool dragHoverMaster = false;
    juce::String dragHoverBusId;
    juce::Point<float> dragStartPosition;
    juce::Point<float> dragCurrentPosition;
    juce::Point<float> dragNodeOffset;
    juce::Point<float> pendingTrackPlacement { -1.0f, -1.0f };
    juce::Point<float> pendingBusPlacement { -1.0f, -1.0f };
    juce::Point<float> masterNodePosition { -1.0f, -1.0f };
    std::vector<juce::Point<float>> trackNodePositions;
    std::vector<std::pair<juce::String, juce::Point<float>>> busNodePositions;
};

class RackBrowserWindowComponent final : public juce::Component
{
public:
    using ProjectGetter = std::function<const ProjectState&()>;
    using SelectedTrackGetter = std::function<int()>;

    RackBrowserWindowComponent(ProjectGetter projectGetterIn,
                               SelectedTrackGetter selectedTrackGetterIn,
                               std::function<void(const juce::String&)> assignRackIn,
                               std::function<void()> clearRackIn,
                               std::function<void()> autoAssignRackIn,
                               std::function<void()> importRackPluginIn,
                               std::function<void()> refreshCatalogIn,
                               std::function<void()> openRackEditorIn,
                               std::function<void()> saveRackStateIn,
                               std::function<void()> playTrackIn,
                               std::function<void()> stopPlaybackIn)
        : projectGetter(std::move(projectGetterIn)),
          selectedTrackGetter(std::move(selectedTrackGetterIn)),
          assignRack(std::move(assignRackIn)),
          clearRack(std::move(clearRackIn)),
          autoAssignRack(std::move(autoAssignRackIn)),
          importRackPlugin(std::move(importRackPluginIn)),
          refreshCatalog(std::move(refreshCatalogIn)),
          openRackEditor(std::move(openRackEditorIn)),
          saveRackState(std::move(saveRackStateIn)),
          playTrack(std::move(playTrackIn)),
          stopPlayback(std::move(stopPlaybackIn)),
          rackListModel(*this)
    {
        titleLabel.setText("Rack Browser", juce::dontSendNotification);
        titleLabel.setFont(ui::titleFont());
        titleLabel.setColour(juce::Label::textColourId, juce::Colour::fromRGB(235, 239, 244));
        addAndMakeVisible(titleLabel);

        selectedTrackLabel.setColour(juce::Label::textColourId, juce::Colour::fromRGB(190, 199, 210));
        addAndMakeVisible(selectedTrackLabel);

        configureButton(refreshButton, "Refresh Folders");
        refreshButton.onClick = [this]
        {
            if (refreshCatalog != nullptr)
                refreshCatalog();
        };

        configureButton(importButton, "Import Plugin");
        importButton.onClick = [this]
        {
            if (importRackPlugin != nullptr)
                importRackPlugin();
        };

        configureButton(assignButton, "Assign To Track");
        assignButton.onClick = [this]
        {
            if (assignRack == nullptr)
                return;
            if (const auto* entry = currentEntry())
                assignRack(entry->path.isNotEmpty() ? entry->path : entry->name);
        };

        configureButton(autoAssignButton, "Auto Assign");
        autoAssignButton.onClick = [this]
        {
            if (autoAssignRack != nullptr)
                autoAssignRack();
        };

        configureButton(clearButton, "Clear Track Rack");
        clearButton.onClick = [this]
        {
            if (clearRack != nullptr)
                clearRack();
        };

        configureButton(openEditorButton, "Open Editor");
        openEditorButton.onClick = [this]
        {
            if (openRackEditor != nullptr)
                openRackEditor();
        };

        configureButton(saveStateButton, "Save State");
        saveStateButton.onClick = [this]
        {
            if (saveRackState != nullptr)
                saveRackState();
        };

        configureButton(playTrackButton, "Play Track");
        playTrackButton.onClick = [this]
        {
            if (playTrack != nullptr)
                playTrack();
        };

        configureButton(stopButton, "Stop");
        stopButton.onClick = [this]
        {
            if (stopPlayback != nullptr)
                stopPlayback();
        };

        rackList.setModel(&rackListModel);
        rackList.setRowHeight(48);
        rackList.setColour(juce::ListBox::backgroundColourId, juce::Colour::fromRGB(20, 22, 28));
        rackList.setOutlineThickness(1);
        addAndMakeVisible(rackList);

        detailsEditor.setMultiLine(true);
        detailsEditor.setReadOnly(true);
        detailsEditor.setScrollbarsShown(true);
        detailsEditor.setColour(juce::TextEditor::backgroundColourId, juce::Colour::fromRGB(18, 20, 25));
        detailsEditor.setColour(juce::TextEditor::textColourId, juce::Colour::fromRGB(226, 230, 237));
        detailsEditor.setColour(juce::TextEditor::outlineColourId, juce::Colour::fromRGB(56, 64, 79));
        detailsEditor.setFont(ui::font());
        addAndMakeVisible(detailsEditor);
    }

    void refreshFromModel()
    {
        const auto& project = projectGetter();
        if (!juce::isPositiveAndBelow(selectedRackRow, static_cast<int>(project.vstRack.size())))
            selectedRackRow = -1;

        const auto selectedTrackIndex = selectedTrackGetter != nullptr ? selectedTrackGetter() : -1;
        if (selectedRackRow < 0)
        {
            if (const auto assignedIndex = assignedRackIndexForSelectedTrack(); assignedIndex >= 0)
                selectedRackRow = assignedIndex;
        }

        rackList.updateContent();
        if (selectedRackRow >= 0)
            rackList.selectRow(selectedRackRow, false, true);
        else
            rackList.deselectAllRows();

        if (juce::isPositiveAndBelow(selectedTrackIndex, static_cast<int>(project.tracks.size())))
        {
            const auto& track = project.tracks[static_cast<size_t>(selectedTrackIndex)];
            selectedTrackLabel.setText("Selected track: " + track.name
                                           + "  |  Mode: " + track.instrumentMode
                                           + "  |  Rack: " + (displayRackName(project, track).isNotEmpty() ? displayRackName(project, track) : "(none)"),
                                       juce::dontSendNotification);
        }
        else
        {
            selectedTrackLabel.setText("No track selected.", juce::dontSendNotification);
        }

        detailsEditor.setText(describeSelection(), false);

        const auto hasTrackSelection = juce::isPositiveAndBelow(selectedTrackIndex, static_cast<int>(project.tracks.size()));
        const auto hasRackSelection = currentEntry() != nullptr;
        assignButton.setEnabled(hasTrackSelection && hasRackSelection);
        clearButton.setEnabled(hasTrackSelection);
        autoAssignButton.setEnabled(hasTrackSelection);
        openEditorButton.setEnabled(hasTrackSelection);
        saveStateButton.setEnabled(hasTrackSelection);
        playTrackButton.setEnabled(hasTrackSelection);
        stopButton.setEnabled(hasTrackSelection);
    }

    void focusRackList()
    {
        rackList.grabKeyboardFocus();
    }

    void selectAssignedRack()
    {
        if (const auto assignedIndex = assignedRackIndexForSelectedTrack(); assignedIndex >= 0)
            selectRackByReference(projectGetter().vstRack[static_cast<size_t>(assignedIndex)].path);
    }

    void selectRackByReference(const juce::String& reference)
    {
        const auto index = findRackInstrumentIndexByReference(projectGetter(), reference);
        if (index < 0)
            return;

        selectedRackRow = index;
        rackList.selectRow(index, true, true);
        detailsEditor.setText(describeSelection(), false);
    }

    void paint(juce::Graphics& g) override
    {
        g.fillAll(juce::Colour::fromRGB(13, 15, 20));
        g.setColour(juce::Colour::fromRGB(31, 35, 44));
        g.drawRect(getLocalBounds(), 1);
    }

    void resized() override
    {
        auto area = getLocalBounds().reduced(14);
        auto header = area.removeFromTop(28);
        titleLabel.setBounds(header.removeFromLeft(160));
        selectedTrackLabel.setBounds(header);

        area.removeFromTop(8);
        auto row1 = area.removeFromTop(30);
        refreshButton.setBounds(row1.removeFromLeft(126));
        row1.removeFromLeft(6);
        importButton.setBounds(row1.removeFromLeft(116));
        row1.removeFromLeft(6);
        assignButton.setBounds(row1.removeFromLeft(126));
        row1.removeFromLeft(6);
        autoAssignButton.setBounds(row1.removeFromLeft(100));
        row1.removeFromLeft(6);
        clearButton.setBounds(row1.removeFromLeft(128));

        area.removeFromTop(8);
        auto row2 = area.removeFromTop(30);
        openEditorButton.setBounds(row2.removeFromLeft(96));
        row2.removeFromLeft(6);
        saveStateButton.setBounds(row2.removeFromLeft(90));
        row2.removeFromLeft(6);
        playTrackButton.setBounds(row2.removeFromLeft(90));
        row2.removeFromLeft(6);
        stopButton.setBounds(row2.removeFromLeft(70));

        area.removeFromTop(10);
        auto listArea = area.removeFromLeft(juce::jmin(420, juce::roundToInt(static_cast<float>(area.getWidth()) * 0.4f)));
        rackList.setBounds(listArea);
        area.removeFromLeft(12);
        detailsEditor.setBounds(area);
    }

private:
    class RackListModel final : public juce::ListBoxModel
    {
    public:
        explicit RackListModel(RackBrowserWindowComponent& ownerIn)
            : owner(ownerIn)
        {
        }

        int getNumRows() override
        {
            return static_cast<int>(owner.projectGetter().vstRack.size());
        }

        void paintListBoxItem(int rowNumber,
                              juce::Graphics& g,
                              int width,
                              int height,
                              bool rowIsSelected) override
        {
            juce::ignoreUnused(height);
            const auto& project = owner.projectGetter();
            if (!juce::isPositiveAndBelow(rowNumber, static_cast<int>(project.vstRack.size())))
                return;

            const auto& entry = project.vstRack[static_cast<size_t>(rowNumber)];
            const auto name = entry.name.isNotEmpty() ? entry.name
                : (entry.pluginName.isNotEmpty() ? entry.pluginName : juce::File(entry.path).getFileNameWithoutExtension());
            const auto detail = entry.path.isNotEmpty() ? entry.path : "(no plugin path)";
            const auto exists = entry.path.isNotEmpty() && juce::File(entry.path).exists();
            const auto badgeText = !entry.hostSupported || !exists
                ? juce::String("CHECK")
                : (entry.isEffect ? juce::String("FX") : juce::String("INST"));
            const auto badgeColour = !entry.hostSupported || !exists
                ? juce::Colour::fromRGB(137, 76, 76)
                : (entry.isEffect ? juce::Colour::fromRGB(76, 111, 168)
                                  : juce::Colour::fromRGB(66, 139, 92));

            g.fillAll(rowIsSelected ? juce::Colour::fromRGB(46, 88, 138)
                                    : ((rowNumber % 2) == 0 ? juce::Colour::fromRGB(26, 30, 37)
                                                            : juce::Colour::fromRGB(21, 25, 31)));

            g.setColour(rowIsSelected ? juce::Colours::white : juce::Colour::fromRGB(235, 239, 244));
            g.setFont(ui::sectionFont());
            g.drawText(name, 8, 2, width - 90, 18, juce::Justification::centredLeft, true);

            g.setFont(ui::font());
            g.drawText(detail, 8, 22, width - 90, 18, juce::Justification::centredLeft, true);

            const auto badgeBounds = juce::Rectangle<int>(width - 78, 12, 68, 20);
            g.setColour(badgeColour);
            g.fillRoundedRectangle(badgeBounds.toFloat(), 5.0f);
            g.setColour(juce::Colours::white);
            g.setFont(ui::tinyFont(true));
            g.drawFittedText(badgeText,
                             badgeBounds,
                             juce::Justification::centred,
                             1);
        }

        void selectedRowsChanged(int lastRowSelected) override
        {
            owner.selectedRackRow = lastRowSelected;
            owner.detailsEditor.setText(owner.describeSelection(), false);
        }

    private:
        RackBrowserWindowComponent& owner;
    };

    const VstInstrument* currentEntry() const
    {
        const auto& project = projectGetter();
        if (!juce::isPositiveAndBelow(selectedRackRow, static_cast<int>(project.vstRack.size())))
            return nullptr;
        return &project.vstRack[static_cast<size_t>(selectedRackRow)];
    }

    int assignedRackIndexForSelectedTrack() const
    {
        const auto& project = projectGetter();
        const auto selectedTrackIndex = selectedTrackGetter != nullptr ? selectedTrackGetter() : -1;
        if (!juce::isPositiveAndBelow(selectedTrackIndex, static_cast<int>(project.tracks.size())))
            return -1;
        return findRackInstrumentIndexByReference(project, project.tracks[static_cast<size_t>(selectedTrackIndex)].rackVst);
    }

    juce::String describeSelection() const
    {
        juce::StringArray lines;
        const auto& project = projectGetter();
        const auto selectedTrackIndex = selectedTrackGetter != nullptr ? selectedTrackGetter() : -1;
        if (juce::isPositiveAndBelow(selectedTrackIndex, static_cast<int>(project.tracks.size())))
        {
            const auto& track = project.tracks[static_cast<size_t>(selectedTrackIndex)];
            lines.add("Track");
            lines.add("Name: " + track.name);
            lines.add("Mode: " + track.instrumentMode);
            lines.add("Rack: " + (displayRackName(project, track).isNotEmpty() ? displayRackName(project, track) : "(none)"));
            lines.add("Resolved path: " + (resolveRackPluginPath(project, track).isNotEmpty() ? resolveRackPluginPath(project, track) : "(unresolved)"));
            lines.add("State path: " + (track.vstiStatePath.isNotEmpty() ? track.vstiStatePath : "(none)"));
        }
        else
        {
            lines.add("Track");
            lines.add("No track selected.");
        }

        lines.add({});
        lines.add("Rack Entry");

        if (const auto* entry = currentEntry())
        {
            const auto exists = entry->path.isNotEmpty() && juce::File(entry->path).exists();
            lines.add("Name: " + (entry->name.isNotEmpty() ? entry->name : "(unnamed)"));
            lines.add("Plugin name: " + (entry->pluginName.isNotEmpty() ? entry->pluginName : "(unknown)"));
            lines.add("Path: " + (entry->path.isNotEmpty() ? entry->path : "(none)"));
            lines.add("Category: " + (entry->category.isNotEmpty() ? entry->category : "(none)"));
            lines.add("Instrument: " + juce::String(entry->isInstrument ? "yes" : "no"));
            lines.add("Effect: " + juce::String(entry->isEffect ? "yes" : "no"));
            lines.add("Host supported: " + juce::String(entry->hostSupported ? "yes" : "no"));
            lines.add("Path exists: " + juce::String(exists ? "yes" : "no"));
            if (entry->hostError.isNotEmpty())
                lines.add("Host error: " + entry->hostError);
        }
        else
        {
            lines.add("No rack entry selected.");
        }

        return lines.joinIntoString("\n");
    }

    void configureButton(juce::TextButton& button, const juce::String& text)
    {
        button.setButtonText(text);
        button.setColour(juce::TextButton::buttonColourId, juce::Colour::fromRGB(48, 54, 66));
        button.setColour(juce::TextButton::textColourOffId, juce::Colours::white);
        addAndMakeVisible(button);
    }

    ProjectGetter projectGetter;
    SelectedTrackGetter selectedTrackGetter;
    std::function<void(const juce::String&)> assignRack;
    std::function<void()> clearRack;
    std::function<void()> autoAssignRack;
    std::function<void()> importRackPlugin;
    std::function<void()> refreshCatalog;
    std::function<void()> openRackEditor;
    std::function<void()> saveRackState;
    std::function<void()> playTrack;
    std::function<void()> stopPlayback;
    int selectedRackRow = -1;
    RackListModel rackListModel;

    juce::Label titleLabel;
    juce::Label selectedTrackLabel;
    juce::TextButton refreshButton;
    juce::TextButton importButton;
    juce::TextButton assignButton;
    juce::TextButton autoAssignButton;
    juce::TextButton clearButton;
    juce::TextButton openEditorButton;
    juce::TextButton saveStateButton;
    juce::TextButton playTrackButton;
    juce::TextButton stopButton;
    juce::ListBox rackList;
    juce::TextEditor detailsEditor;
};

class RenderManagerWindowComponent final : public juce::Component
{
public:
    using ProjectGetter = std::function<const ProjectState&()>;
    using SelectedTrackGetter = std::function<int()>;
    using SelectedTrackSetter = std::function<void(int)>;

    RenderManagerWindowComponent(ProjectGetter projectGetterIn,
                                 SelectedTrackGetter selectedTrackGetterIn,
                                 SelectedTrackSetter selectedTrackSetterIn,
                                 std::function<void()> exportTrackIn,
                                 std::function<void()> exportStemsIn,
                                 std::function<void()> relinkRenderIn,
                                 std::function<void()> clearRenderIn,
                                 std::function<void()> importRenderIn,
                                 std::function<void()> placeRenderIn)
        : projectGetter(std::move(projectGetterIn)),
          selectedTrackGetter(std::move(selectedTrackGetterIn)),
          selectedTrackSetter(std::move(selectedTrackSetterIn)),
          exportTrack(std::move(exportTrackIn)),
          exportStems(std::move(exportStemsIn)),
          relinkRender(std::move(relinkRenderIn)),
          clearRender(std::move(clearRenderIn)),
          importRender(std::move(importRenderIn)),
          placeRender(std::move(placeRenderIn)),
          trackListModel(*this)
    {
        titleLabel.setText("Render Manager", juce::dontSendNotification);
        titleLabel.setFont(ui::titleFont());
        titleLabel.setColour(juce::Label::textColourId, juce::Colour::fromRGB(235, 239, 244));
        addAndMakeVisible(titleLabel);

        configureButton(exportTrackButton, "Export Track");
        exportTrackButton.onClick = [this]
        {
            if (exportTrack != nullptr)
                exportTrack();
        };

        configureButton(exportStemsButton, "Export Stems");
        exportStemsButton.onClick = [this]
        {
            if (exportStems != nullptr)
                exportStems();
        };

        configureButton(relinkButton, "Relink File");
        relinkButton.onClick = [this]
        {
            if (relinkRender != nullptr)
                relinkRender();
        };

        configureButton(clearButton, "Clear Render");
        clearButton.onClick = [this]
        {
            if (clearRender != nullptr)
                clearRender();
        };

        configureButton(importButton, "Add To Library");
        importButton.onClick = [this]
        {
            if (importRender != nullptr)
                importRender();
        };

        configureButton(placeButton, "Place At Playhead");
        placeButton.onClick = [this]
        {
            if (placeRender != nullptr)
                placeRender();
        };

        trackList.setModel(&trackListModel);
        trackList.setRowHeight(50);
        trackList.setColour(juce::ListBox::backgroundColourId, juce::Colour::fromRGB(20, 22, 28));
        trackList.setOutlineThickness(1);
        addAndMakeVisible(trackList);

        detailsEditor.setMultiLine(true);
        detailsEditor.setReadOnly(true);
        detailsEditor.setScrollbarsShown(true);
        detailsEditor.setColour(juce::TextEditor::backgroundColourId, juce::Colour::fromRGB(18, 20, 25));
        detailsEditor.setColour(juce::TextEditor::textColourId, juce::Colour::fromRGB(226, 230, 237));
        detailsEditor.setColour(juce::TextEditor::outlineColourId, juce::Colour::fromRGB(56, 64, 79));
        detailsEditor.setFont(ui::font());
        addAndMakeVisible(detailsEditor);
    }

    void refreshFromModel()
    {
        const auto selectedTrack = selectedTrackGetter != nullptr ? selectedTrackGetter() : -1;
        trackList.updateContent();
        if (selectedTrack >= 0)
            trackList.selectRow(selectedTrack, false, true);
        else
            trackList.deselectAllRows();

        detailsEditor.setText(describeSelection(), false);

        const auto hasSelection = selectedTrack >= 0
            && selectedTrack < static_cast<int>(projectGetter().tracks.size());
        exportTrackButton.setEnabled(hasSelection);
        relinkButton.setEnabled(hasSelection);
        clearButton.setEnabled(hasSelection);
        importButton.setEnabled(hasSelection && selectedTrackHasRenderFile());
        placeButton.setEnabled(hasSelection && selectedTrackHasRenderFile());
        exportStemsButton.setEnabled(!projectGetter().tracks.empty());
    }

    void focusTrackList()
    {
        trackList.grabKeyboardFocus();
    }

    void paint(juce::Graphics& g) override
    {
        g.fillAll(juce::Colour::fromRGB(13, 15, 20));
        g.setColour(juce::Colour::fromRGB(31, 35, 44));
        g.drawRect(getLocalBounds(), 1);
    }

    void resized() override
    {
        auto area = getLocalBounds().reduced(14);
        auto header = area.removeFromTop(30);
        titleLabel.setBounds(header.removeFromLeft(160));
        exportTrackButton.setBounds(header.removeFromLeft(104));
        header.removeFromLeft(6);
        exportStemsButton.setBounds(header.removeFromLeft(102));
        header.removeFromLeft(6);
        relinkButton.setBounds(header.removeFromLeft(94));
        header.removeFromLeft(6);
        clearButton.setBounds(header.removeFromLeft(100));
        header.removeFromLeft(6);
        importButton.setBounds(header.removeFromLeft(102));
        header.removeFromLeft(6);
        placeButton.setBounds(header.removeFromLeft(126));

        area.removeFromTop(10);
        auto listArea = area.removeFromLeft(juce::jmin(360, juce::roundToInt(static_cast<float>(area.getWidth()) * 0.33f)));
        trackList.setBounds(listArea);
        area.removeFromLeft(12);
        detailsEditor.setBounds(area);
    }

private:
    class TrackListModel final : public juce::ListBoxModel
    {
    public:
        explicit TrackListModel(RenderManagerWindowComponent& ownerIn)
            : owner(ownerIn)
        {
        }

        int getNumRows() override
        {
            return static_cast<int>(owner.projectGetter().tracks.size());
        }

        void paintListBoxItem(int rowNumber,
                              juce::Graphics& g,
                              int width,
                              int height,
                              bool rowIsSelected) override
        {
            juce::ignoreUnused(height);
            const auto& project = owner.projectGetter();
            if (!juce::isPositiveAndBelow(rowNumber, static_cast<int>(project.tracks.size())))
                return;

            const auto& track = project.tracks[static_cast<size_t>(rowNumber)];
            const auto renderFile = juce::File(track.renderedAudioPath);
            const auto hasRender = track.renderedAudioPath.isNotEmpty();
            const auto renderReady = hasRender && renderFile.existsAsFile();

            const auto background = (rowNumber % 2) == 0 ? juce::Colour::fromRGB(26, 30, 37)
                                                         : juce::Colour::fromRGB(21, 25, 31);
            const auto trackColour = trackDisplayColour(track, rowNumber);
            g.fillAll(background);
            g.setColour(trackColour.withAlpha(rowIsSelected ? 0.78f : 0.14f));
            g.fillRect(0, 0, width, height);
            g.setColour(trackColour);
            g.fillRect(0, 0, 5, height);

            g.setColour(rowIsSelected ? trackTextColour(trackColour) : trackColour.brighter(0.18f));
            g.setFont(ui::sectionFont());
            g.drawText(track.name.trim().isNotEmpty() ? track.name : ("Track " + juce::String(rowNumber + 1)),
                       8, 2, width - 90, 18, juce::Justification::centredLeft, true);

            juce::String detail = track.trackType + " | ";
            if (renderReady)
                detail << renderFile.getFileName();
            else if (hasRender)
                detail << "missing render";
            else
                detail << "no render";

            g.setFont(ui::font());
            g.drawText(detail, 8, 24, width - 90, 18, juce::Justification::centredLeft, true);

            const auto badgeBounds = juce::Rectangle<int>(width - 78, 14, 68, 20);
            g.setColour(renderReady ? juce::Colour::fromRGB(66, 139, 92)
                                    : (hasRender ? juce::Colour::fromRGB(151, 109, 62)
                                                 : juce::Colour::fromRGB(86, 92, 104)));
            g.fillRoundedRectangle(badgeBounds.toFloat(), 5.0f);
            g.setColour(juce::Colours::white);
            g.setFont(ui::strongFont());
            g.drawFittedText(renderReady ? "READY" : (hasRender ? "MISSING" : "EMPTY"),
                             badgeBounds,
                             juce::Justification::centred,
                             1);
        }

        void selectedRowsChanged(int lastRowSelected) override
        {
            if (owner.selectedTrackSetter != nullptr)
                owner.selectedTrackSetter(lastRowSelected);
            owner.detailsEditor.setText(owner.describeSelection(), false);
        }

    private:
        RenderManagerWindowComponent& owner;
    };

    bool selectedTrackHasRenderFile() const
    {
        const auto& project = projectGetter();
        const auto selectedTrack = selectedTrackGetter != nullptr ? selectedTrackGetter() : -1;
        if (!juce::isPositiveAndBelow(selectedTrack, static_cast<int>(project.tracks.size())))
            return false;
        const auto path = project.tracks[static_cast<size_t>(selectedTrack)].renderedAudioPath.trim();
        return path.isNotEmpty() && juce::File(path).existsAsFile();
    }

    juce::String describeSelection() const
    {
        const auto& project = projectGetter();
        const auto selectedTrack = selectedTrackGetter != nullptr ? selectedTrackGetter() : -1;
        if (!juce::isPositiveAndBelow(selectedTrack, static_cast<int>(project.tracks.size())))
            return "No track selected.";

        const auto& track = project.tracks[static_cast<size_t>(selectedTrack)];
        const auto renderPath = track.renderedAudioPath.trim();
        const auto renderFile = juce::File(renderPath);

        int clipReferenceCount = 0;
        for (const auto& clip : project.sampleClips)
        {
            if (clip.path.equalsIgnoreCase(renderPath))
                ++clipReferenceCount;
        }

        juce::StringArray lines;
        lines.add("Track");
        lines.add("Name: " + (track.name.trim().isNotEmpty() ? track.name : ("Track " + juce::String(selectedTrack + 1))));
        lines.add("Type: " + track.trackType);
        lines.add("Instrument: " + track.instrument);
        lines.add("Rack: " + (displayRackName(project, track).isNotEmpty() ? displayRackName(project, track) : "(none)"));
        lines.add("Notes: " + juce::String(static_cast<int>(track.notes.size())));
        lines.add("Locator render path: " + (renderPath.isNotEmpty() ? renderPath : "(none)"));
        lines.add("Render exists: " + juce::String(renderPath.isNotEmpty() && renderFile.existsAsFile() ? "yes" : "no"));
        lines.add("Clip references: " + juce::String(clipReferenceCount));
        if (renderPath.isNotEmpty() && renderFile.existsAsFile())
        {
            lines.add("Render file: " + renderFile.getFileName());
            lines.add("Render folder: " + renderFile.getParentDirectory().getFullPathName());
        }
        return lines.joinIntoString("\n");
    }

    void configureButton(juce::TextButton& button, const juce::String& text)
    {
        button.setButtonText(text);
        button.setColour(juce::TextButton::buttonColourId, juce::Colour::fromRGB(48, 54, 66));
        button.setColour(juce::TextButton::textColourOffId, juce::Colours::white);
        addAndMakeVisible(button);
    }

    ProjectGetter projectGetter;
    SelectedTrackGetter selectedTrackGetter;
    SelectedTrackSetter selectedTrackSetter;
    std::function<void()> exportTrack;
    std::function<void()> exportStems;
    std::function<void()> relinkRender;
    std::function<void()> clearRender;
    std::function<void()> importRender;
    std::function<void()> placeRender;
    TrackListModel trackListModel;

    juce::Label titleLabel;
    juce::TextButton exportTrackButton;
    juce::TextButton exportStemsButton;
    juce::TextButton relinkButton;
    juce::TextButton clearButton;
    juce::TextButton importButton;
    juce::TextButton placeButton;
    juce::ListBox trackList;
    juce::TextEditor detailsEditor;
};

namespace
{
constexpr int kColumnMute = 1;
constexpr int kColumnSolo = 2;
constexpr int kColumnVstView = 3;
constexpr int kColumnName = 4;
constexpr int kColumnType = 5;
constexpr int kColumnMode = 6;
constexpr int kColumnRack = 7;
constexpr int kColumnNotes = 8;
constexpr int kColumnChannel = 9;
constexpr int kColumnVolume = 10;
constexpr int kColumnPan = 11;
constexpr int kColumnFlags = 12;
constexpr int kDefaultWindowWidth = 1480;
constexpr int kDefaultWindowHeight = 920;
const char* kProjectFileWildcard = "*.aims;*.json";
const char* kJsonProjectWildcard = "*.json";
const char* kMidiFileWildcard = "*.mid;*.midi";
const char* kAudioExportFileWildcard = "*.wav;*.mp3";

void paintTrackLevelBar(juce::Graphics& g,
                        juce::Rectangle<int> bounds,
                        float level,
                        float volumeMarker,
                        juce::Colour accentColour)
{
    bounds = bounds.reduced(2);
    if (bounds.isEmpty())
        return;

    g.setColour(juce::Colour::fromRGB(20, 23, 30));
    g.fillRect(bounds);
    g.setColour(accentColour.withAlpha(0.42f));
    g.drawRect(bounds, 1);

    const auto filledHeight = juce::roundToInt(static_cast<float>(bounds.getHeight()) * juce::jlimit(0.0f, 1.0f, level));
    if (filledHeight > 0)
    {
        auto filledBounds = bounds.withTrimmedTop(bounds.getHeight() - filledHeight).reduced(1, 1);
        const auto meterColour = level >= 0.85f
            ? juce::Colour::fromRGB(242, 99, 84)
            : (level >= 0.60f ? juce::Colour::fromRGB(234, 192, 78)
                              : juce::Colour::fromRGB(61, 210, 122));
        g.setColour(meterColour);
        g.fillRect(filledBounds);
    }

    const auto clampedMarker = juce::jlimit(0.0f, 1.0f, volumeMarker);
    const auto markerY = juce::roundToInt(static_cast<float>(bounds.getBottom()) - (static_cast<float>(bounds.getHeight()) * clampedMarker));
    g.setColour(accentColour.brighter(0.3f));
    g.drawLine(static_cast<float>(bounds.getX() + 2),
               static_cast<float>(markerY),
               static_cast<float>(bounds.getRight() - 2),
               static_cast<float>(markerY),
               1.6f);
}

void discoverBundledVstEntriesRecursive(const juce::File& directory, std::vector<VstInstrument>& entries)
{
    juce::Array<juce::File> children;
    directory.findChildFiles(children, juce::File::findFilesAndDirectories, false);

    for (const auto& child : children)
    {
        if (child.isDirectory())
        {
            const auto name = child.getFileName();
            if (name.endsWithIgnoreCase(".disabled"))
                continue;

            if (name.endsWithIgnoreCase(".vst3"))
            {
                entries.push_back(makeRackPluginEntry(child));
                continue;
            }

            discoverBundledVstEntriesRecursive(child, entries);
            continue;
        }

        if (!child.hasFileExtension(".dll;.so;.vst3"))
            continue;

        entries.push_back(makeRackPluginEntry(child));
    }
}

juce::File findBundledVstDirectory()
{
    juce::Array<juce::File> baseDirectories;
    baseDirectories.add(juce::File::getCurrentWorkingDirectory());
    baseDirectories.add(juce::File::getSpecialLocation(juce::File::currentApplicationFile).getParentDirectory());

    for (const auto& baseDirectory : baseDirectories)
    {
        auto probe = baseDirectory;
        for (int depth = 0; depth < 8 && probe != juce::File(); ++depth)
        {
            const auto candidate = probe.getChildFile("vsti");
            if (candidate.isDirectory())
                return candidate;

            probe = probe.getParentDirectory();
        }
    }

    return {};
}

juce::File defaultSystemVstDirectory()
{
#if JUCE_WINDOWS
    return juce::File::getSpecialLocation(juce::File::globalApplicationsDirectory)
        .getChildFile("Common Files")
        .getChildFile("VST3");
#else
    return {};
#endif
}

std::vector<VstInstrument> discoverVstCatalogInDirectory(const juce::File& directory)
{
    if (directory == juce::File() || !directory.isDirectory())
        return {};

    std::vector<VstInstrument> entries;
    discoverBundledVstEntriesRecursive(directory, entries);

    std::sort(entries.begin(),
              entries.end(),
              [] (const VstInstrument& lhs, const VstInstrument& rhs)
              {
                  return lhs.path.compareIgnoreCase(rhs.path) < 0;
              });

    entries.erase(std::unique(entries.begin(),
                              entries.end(),
                              [] (const VstInstrument& lhs, const VstInstrument& rhs)
                              {
                                  return lhs.path.equalsIgnoreCase(rhs.path);
                              }),
                  entries.end());
    return entries;
}

juce::String makeTrackFlags(const TrackState& track)
{
    juce::StringArray flags;
    if (track.liveArmed)
        flags.add("ARM");
    return flags.joinIntoString(" ");
}

juce::String describeTrack(const ProjectState& project, const TrackState& track)
{
    juce::StringArray lines;
    lines.add("Name: " + track.name);
    lines.add("Type: " + track.trackType);
    lines.add("Instrument: " + track.instrument);
    lines.add("Mode: " + track.instrumentMode);
    lines.add("Rack VST: " + (displayRackName(project, track).isNotEmpty() ? displayRackName(project, track) : "(none)"));
    const auto resolvedRackPath = resolveRackPluginPath(project, track);
    lines.add("Rack path: " + (resolvedRackPath.isNotEmpty() ? resolvedRackPath : "(unresolved)"));
    lines.add("MIDI channel: " + juce::String(track.midiChannel + 1));
    lines.add("Program: " + juce::String(track.midiProgram));
    lines.add("Notes: " + juce::String(static_cast<int>(track.notes.size())));
    lines.add("Automation lanes: " + juce::String(static_cast<int>(track.automationLanes.size())));
    lines.add("Volume: " + juce::String(track.volume, 2));
    lines.add("Pan: " + juce::String(track.pan, 2));
    lines.add("Render path: " + (track.renderedAudioPath.isNotEmpty() ? track.renderedAudioPath : "(none)"));

    if (!track.notes.empty())
    {
        lines.add({});
        lines.add("First notes:");
        const auto previewCount = juce::jmin<int>(10, static_cast<int>(track.notes.size()));
        for (int index = 0; index < previewCount; ++index)
        {
            const auto& note = track.notes[static_cast<size_t>(index)];
            lines.add("  Tick " + juce::String(note.startTick)
                + " len " + juce::String(note.durationTick)
                + " pitch " + juce::String(note.pitch)
                + " vel " + juce::String(note.velocity)
                + (note.selected ? " [selected]" : ""));
        }
    }

    return lines.joinIntoString("\n");
}

juce::String aiComposeModeLabel(AIComposeTargetMode mode)
{
    switch (mode)
    {
        case AIComposeTargetMode::replaceCurrentTrack: return "Replace Current Track";
        case AIComposeTargetMode::replaceAllTracks: return "Replace All Tracks";
        case AIComposeTargetMode::addToCurrentTrack: return "Add To Current Track";
        case AIComposeTargetMode::addToAllTracks: return "Add To All Tracks";
    }

    return "Replace All Tracks";
}

int aiComposeModeComboId(AIComposeTargetMode mode)
{
    switch (mode)
    {
        case AIComposeTargetMode::replaceCurrentTrack: return 1;
        case AIComposeTargetMode::replaceAllTracks: return 2;
        case AIComposeTargetMode::addToCurrentTrack: return 3;
        case AIComposeTargetMode::addToAllTracks: return 4;
    }

    return 2;
}

AIComposeTargetMode aiComposeModeFromComboId(int comboId)
{
    switch (comboId)
    {
        case 1: return AIComposeTargetMode::replaceCurrentTrack;
        case 3: return AIComposeTargetMode::addToCurrentTrack;
        case 4: return AIComposeTargetMode::addToAllTracks;
        case 2:
        default:
            return AIComposeTargetMode::replaceAllTracks;
    }
}

juce::StringArray aiComposeStyleChoices()
{
    juce::StringArray values;
    values.add("Balanced");
    values.add("Ambient");
    values.add("Cinematic");
    values.add("Electronic");
    values.add("Lo-fi");
    values.add("Orchestral");
    values.add("Aggressive");
    values.add("Experimental");
    return values;
}

juce::StringArray aiComposeEnergyChoices()
{
    juce::StringArray values;
    values.add("Low");
    values.add("Medium");
    values.add("High");
    values.add("Explosive");
    return values;
}

juce::StringArray aiComposeDensityChoices()
{
    juce::StringArray values;
    values.add("Sparse");
    values.add("Balanced");
    values.add("Busy");
    values.add("Dense");
    return values;
}

juce::StringArray aiComposeVariationChoices()
{
    juce::StringArray values;
    values.add("Minimal");
    values.add("Moderate");
    values.add("Evolving");
    values.add("Wild");
    return values;
}

juce::StringArray aiComposeRegisterChoices()
{
    juce::StringArray values;
    values.add("Natural");
    values.add("Low");
    values.add("Mid");
    values.add("High");
    values.add("Wide");
    return values;
}

juce::String aiComposeGridLabel(const ProjectState& project)
{
    const auto division = juce::jmax(1, project.quantizeDiv);
    auto label = "1/" + juce::String(division);
    if (project.quantizeTriplet)
        label << " triplet";
    return label;
}

int aiComposeGridTick(const ProjectState& project)
{
    const auto quantizeDiv = juce::jmax(1, project.quantizeDiv);
    auto beats = 4.0 / static_cast<double>(quantizeDiv);
    if (project.quantizeTriplet)
        beats *= (2.0 / 3.0);
    return juce::jmax(1, static_cast<int>(std::round(beats * static_cast<double>(kTicksPerBeat))));
}

int aiComposeRoundToGrid(int tick, int gridTick)
{
    const auto safeGrid = juce::jmax(1, gridTick);
    return juce::jmax(0,
                      static_cast<int>(std::llround(static_cast<double>(juce::jmax(0, tick))
                                                    / static_cast<double>(safeGrid)))
                          * safeGrid);
}

int aiComposeQuantizedPitch(const ProjectState& project, int pitch)
{
    const auto clampedPitch = juce::jlimit(kEditableMidiPitchMin, kEditableMidiPitchMax, pitch);
    if (!projectUsesKeyQuantize(project))
        return clampedPitch;

    const auto visiblePitches = visiblePitchesForProjectScale(project, kEditableMidiPitchMin, kEditableMidiPitchMax);
    if (visiblePitches.empty())
        return clampedPitch;

    auto bestPitch = clampedPitch;
    auto bestDistance = 999;
    for (const auto candidate : visiblePitches)
    {
        const auto distance = std::abs(candidate - clampedPitch);
        if (distance < bestDistance)
        {
            bestDistance = distance;
            bestPitch = candidate;
        }
    }

    return juce::jlimit(kEditableMidiPitchMin, kEditableMidiPitchMax, bestPitch);
}

juce::String aiComposeTickLocationLabel(const ProjectState& project, int tick)
{
    const auto safeTick = juce::jmax(0, tick);
    const auto projectBarTicks = ticksPerBar(project);
    const auto projectBeatTicks = ticksPerTimeSignatureBeat(project);
    const auto bar = (safeTick / projectBarTicks) + 1;
    const auto beat = ((safeTick % projectBarTicks) / juce::jmax(1, projectBeatTicks)) + 1;
    return "bar " + juce::String(bar)
        + ", beat " + juce::String(beat)
        + " (tick " + juce::String(safeTick) + ")";
}

juce::String aiComposeTrackContextLine(const ProjectState& project, int trackIndex)
{
    if (!juce::isPositiveAndBelow(trackIndex, static_cast<int>(project.tracks.size())))
        return "Track context unavailable.";

    const auto& track = project.tracks[static_cast<size_t>(trackIndex)];
    auto sectionCount = 0;
    for (const auto& section : project.midiSections)
    {
        if (section.trackIndex == trackIndex)
            ++sectionCount;
    }

    auto lowestPitch = 127;
    auto highestPitch = 0;
    for (const auto& note : track.notes)
    {
        lowestPitch = juce::jmin(lowestPitch, note.pitch);
        highestPitch = juce::jmax(highestPitch, note.pitch);
    }

    juce::StringArray fields;
    fields.add("Track " + juce::String(trackIndex + 1) + ": " + (track.name.trim().isNotEmpty() ? track.name : "Track"));
    fields.add("instrument " + (track.instrument.trim().isNotEmpty() ? track.instrument : "(unset)"));
    fields.add("rack " + (displayRackName(project, track).trim().isNotEmpty() ? displayRackName(project, track) : "(none)"));
    fields.add("mode " + (track.instrumentMode.trim().isNotEmpty() ? track.instrumentMode : "(unset)"));
    fields.add("MIDI ch " + juce::String(track.midiChannel + 1));
    fields.add("program " + juce::String(track.midiProgram));
    fields.add("sections " + juce::String(sectionCount));
    fields.add("notes " + juce::String(static_cast<int>(track.notes.size())));
    if (!track.notes.empty())
        fields.add("pitch range " + juce::String(lowestPitch) + "-" + juce::String(highestPitch));
    return fields.joinIntoString(" | ");
}

juce::String parameterSetSignature(const juce::NamedValueSet& values)
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

juce::String noteTransportSignature(const std::vector<MidiNote>& notes)
{
    juce::StringArray entries;
    entries.ensureStorageAllocated(static_cast<int>(notes.size()));
    for (const auto& note : notes)
    {
        entries.add(juce::String(note.startTick)
            + ":" + juce::String(note.durationTick)
            + ":" + juce::String(note.pitch)
            + ":" + juce::String(note.velocity));
    }
    return entries.joinIntoString("|");
}

juce::String controllerEventSignature(const std::vector<MidiControllerEvent>& events)
{
    juce::StringArray entries;
    entries.ensureStorageAllocated(static_cast<int>(events.size()));
    for (const auto& event : events)
    {
        entries.add(juce::String(event.tick)
            + ":" + juce::String(event.controller)
            + ":" + juce::String(event.value)
            + ":" + juce::String(event.channel));
    }
    return entries.joinIntoString("|");
}

juce::String automationLaneSignature(const std::vector<AutomationLane>& lanes)
{
    juce::StringArray laneEntries;
    laneEntries.ensureStorageAllocated(static_cast<int>(lanes.size()));
    for (const auto& lane : lanes)
    {
        juce::StringArray pointEntries;
        pointEntries.ensureStorageAllocated(static_cast<int>(lane.points.size()));
        for (const auto& point : lane.points)
            pointEntries.add(juce::String(point.tick) + ":" + juce::String(point.value, 6));

        laneEntries.add(lane.target.trim().toLowerCase()
                        + ":" + (lane.enabled ? "1" : "0")
                        + ":" + pointEntries.joinIntoString(","));
    }

    laneEntries.sort(true);
    return laneEntries.joinIntoString("|");
}

bool sameStringArray(const juce::StringArray& lhs, const juce::StringArray& rhs)
{
    if (lhs.size() != rhs.size())
        return false;

    for (int index = 0; index < lhs.size(); ++index)
    {
        if (lhs[index] != rhs[index])
            return false;
    }

    return true;
}

bool sameBoolVector(const std::vector<bool>& lhs, const std::vector<bool>& rhs)
{
    if (lhs.size() != rhs.size())
        return false;

    for (size_t index = 0; index < lhs.size(); ++index)
    {
        if (lhs[index] != rhs[index])
            return false;
    }

    return true;
}

bool sameTempoMarkers(const std::vector<TempoMarker>& lhs, const std::vector<TempoMarker>& rhs)
{
    if (lhs.size() != rhs.size())
        return false;

    for (size_t index = 0; index < lhs.size(); ++index)
    {
        if (lhs[index].tick != rhs[index].tick || lhs[index].bpm != rhs[index].bpm)
            return false;
    }

    return true;
}

bool sameSampleClips(const std::vector<SampleClip>& lhs, const std::vector<SampleClip>& rhs)
{
    if (lhs.size() != rhs.size())
        return false;

    for (size_t index = 0; index < lhs.size(); ++index)
    {
        const auto& left = lhs[index];
        const auto& right = rhs[index];
        if (left.path != right.path
            || left.trackIndex != right.trackIndex
            || std::abs(left.startSec - right.startSec) > 1.0e-6
            || std::abs(left.durationSec - right.durationSec) > 1.0e-6
            || std::abs(left.sourceOffsetSec - right.sourceOffsetSec) > 1.0e-6
            || std::abs(left.sourceFileDurationSec - right.sourceFileDurationSec) > 1.0e-6
            || left.sampleRate != right.sampleRate)
        {
            return false;
        }
    }

    return true;
}

bool sameSharedEffectBuses(const std::vector<SharedEffectBusState>& lhs,
                           const std::vector<SharedEffectBusState>& rhs)
{
    if (lhs.size() != rhs.size())
        return false;

    for (size_t index = 0; index < lhs.size(); ++index)
    {
        const auto& a = lhs[index];
        const auto& b = rhs[index];
        if (!a.id.equalsIgnoreCase(b.id)
            || !a.name.equalsIgnoreCase(b.name)
            || !a.effect.equalsIgnoreCase(b.effect)
            || !sameStringArray(a.outputTargets, b.outputTargets)
            || !a.statePath.equalsIgnoreCase(b.statePath)
            || a.bypassed != b.bypassed
            || parameterSetSignature(a.parameters) != parameterSetSignature(b.parameters))
        {
            return false;
        }
    }

    return true;
}

bool sameMidiSections(const std::vector<MidiSection>& lhs, const std::vector<MidiSection>& rhs)
{
    if (lhs.size() != rhs.size())
        return false;

    for (size_t index = 0; index < lhs.size(); ++index)
    {
        const auto& a = lhs[index];
        const auto& b = rhs[index];
        if (a.trackIndex != b.trackIndex
            || a.startTick != b.startTick
            || a.lengthTicks != b.lengthTicks
            || !a.name.equalsIgnoreCase(b.name)
            || !a.patternId.equalsIgnoreCase(b.patternId))
        {
            return false;
        }
    }

    return true;
}

bool sameMidiPatterns(const std::vector<MidiPattern>& lhs, const std::vector<MidiPattern>& rhs)
{
    if (lhs.size() != rhs.size())
        return false;

    for (size_t index = 0; index < lhs.size(); ++index)
    {
        const auto& a = lhs[index];
        const auto& b = rhs[index];
        if (!a.id.equalsIgnoreCase(b.id)
            || !a.name.equalsIgnoreCase(b.name)
            || a.lengthTicks != b.lengthTicks
            || noteTransportSignature(a.notes) != noteTransportSignature(b.notes)
            || automationLaneSignature(a.controllerLanes) != automationLaneSignature(b.controllerLanes))
        {
            return false;
        }
    }

    return true;
}

struct TrackEngineDiff
{
    bool rackBindingChanged = false;
    bool noteContentChanged = false;
    bool controllerContentChanged = false;
    bool parameterContentChanged = false;
    bool mixStateChanged = false;
    bool fxStateChanged = false;
    bool automationContentChanged = false;
    bool renderedAudioChanged = false;
    bool instrumentActivationChanged = false;
    bool requiresFullEngineState = false;
};

TrackEngineDiff analyseTrackEngineDiff(const ProjectState& previousProject,
                                       const ProjectState& currentProject,
                                       int trackIndex)
{
    TrackEngineDiff diff;
    if (!juce::isPositiveAndBelow(trackIndex, static_cast<int>(previousProject.tracks.size()))
        || !juce::isPositiveAndBelow(trackIndex, static_cast<int>(currentProject.tracks.size())))
    {
        diff.requiresFullEngineState = true;
        return diff;
    }

    const auto& previousTrack = previousProject.tracks[static_cast<size_t>(trackIndex)];
    const auto& currentTrack = currentProject.tracks[static_cast<size_t>(trackIndex)];
    const auto mixValueChanged = [] (double lhs, double rhs)
    {
        return std::abs(lhs - rhs) > 1.0e-6;
    };
    const auto trackIsAudibleAtIndex = [] (const ProjectState& project, int index)
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

        if (!juce::isPositiveAndBelow(index, static_cast<int>(project.tracks.size())))
            return false;

        const auto& track = project.tracks[static_cast<size_t>(index)];
        if (track.mute)
            return false;
        if (anySolo && !track.solo)
            return false;
        return true;
    };

    diff.rackBindingChanged = !previousTrack.rackVst.equalsIgnoreCase(currentTrack.rackVst)
        || !previousTrack.instrumentMode.equalsIgnoreCase(currentTrack.instrumentMode)
        || !previousTrack.trackType.equalsIgnoreCase(currentTrack.trackType)
        || !previousTrack.vstiStatePath.equalsIgnoreCase(currentTrack.vstiStatePath);
    diff.noteContentChanged = noteTransportSignature(previousTrack.notes) != noteTransportSignature(currentTrack.notes);
    diff.controllerContentChanged =
        controllerEventSignature(collectTrackControllerEvents(previousProject, trackIndex))
            != controllerEventSignature(collectTrackControllerEvents(currentProject, trackIndex));
    diff.parameterContentChanged = parameterSetSignature(previousTrack.vstiParameters) != parameterSetSignature(currentTrack.vstiParameters);
    diff.mixStateChanged = mixValueChanged(previousTrack.volume, currentTrack.volume)
        || mixValueChanged(previousTrack.pan, currentTrack.pan)
        || mixValueChanged(previousTrack.vstiOutputGainDb, currentTrack.vstiOutputGainDb)
        || previousTrack.mute != currentTrack.mute
        || previousTrack.solo != currentTrack.solo
        || trackIsAudibleAtIndex(previousProject, trackIndex) != trackIsAudibleAtIndex(currentProject, trackIndex);
    diff.fxStateChanged = previousTrack.vstFxBypassed != currentTrack.vstFxBypassed
        || !sameStringArray(previousTrack.vstFxChain, currentTrack.vstFxChain)
        || !sameBoolVector(previousTrack.vstFxSlotBypassed, currentTrack.vstFxSlotBypassed);
    diff.automationContentChanged = automationLaneSignature(previousTrack.automationLanes)
        != automationLaneSignature(currentTrack.automationLanes);
    diff.renderedAudioChanged = !previousTrack.renderedAudioPath.equalsIgnoreCase(currentTrack.renderedAudioPath);

    const auto previousRackPath = resolveRackPluginPath(previousProject, previousTrack);
    const auto currentRackPath = resolveRackPluginPath(currentProject, currentTrack);
    diff.instrumentActivationChanged = (!previousRackPath.isEmpty() || !currentRackPath.isEmpty())
        && (previousTrack.notes.empty() != currentTrack.notes.empty());
    diff.requiresFullEngineState = diff.rackBindingChanged
        || !previousTrack.routingTarget.equalsIgnoreCase(currentTrack.routingTarget)
        || diff.fxStateChanged
        || diff.automationContentChanged
        || diff.renderedAudioChanged
        || diff.instrumentActivationChanged;
    return diff;
}

bool projectTrackIsAudible(const ProjectState& project, int trackIndex)
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

TrackState makeFallbackDefaultTrack()
{
    TrackState track;
    track.name = "Track 1";
    track.instrument = "AI Piano";
    track.instrumentMode = "VSTI Rack";
    track.rackVst = "AI Piano";
    track.midiChannel = 0;
    track.synthProfile = "keys";
    track.followThemeTrackColour = true;
    track.themeColourSlot = 0;
    return track;
}

void normaliseTrack(TrackState& track)
{
    std::sort(track.notes.begin(),
              track.notes.end(),
              [] (const MidiNote& lhs, const MidiNote& rhs)
              {
                  if (lhs.startTick != rhs.startTick)
                      return lhs.startTick < rhs.startTick;
                  if (lhs.pitch != rhs.pitch)
                      return lhs.pitch > rhs.pitch;
                  return lhs.durationTick < rhs.durationTick;
              });

    track.midiProgram = juce::jlimit(0, 127, track.midiProgram);
    track.midiChannel = juce::jlimit(0, 15, track.midiChannel);
    track.volume = juce::jlimit(0.0, 2.0, track.volume);
    track.pan = juce::jlimit(-1.0, 1.0, track.pan);
    track.vstiOutputGainDb = juce::jlimit(-48.0, 24.0, track.vstiOutputGainDb);

    juce::StringArray cleanedFxChain;
    for (const auto& entry : track.vstFxChain)
    {
        const auto trimmed = entry.trim();
        if (trimmed.isNotEmpty())
            cleanedFxChain.addIfNotAlreadyThere(trimmed);
    }
    track.vstFxChain = cleanedFxChain;

    sanitiseAutomationLanes(track);
    sanitiseArrangementAutomationLaneVisibility(track);
}

void normaliseProject(ProjectState& project)
{
    if (project.tracks.empty())
        project.tracks.push_back(makeFallbackDefaultTrack());

    for (size_t index = 0; index < project.tracks.size(); ++index)
    {
        auto& track = project.tracks[index];
        if (track.themeColourSlot < 0)
            track.themeColourSlot = static_cast<int>(index);
        if (track.colorHex.trim().isEmpty())
            track.colorHex = defaultTrackColour(track.themeColourSlot).toDisplayString(false);
        normaliseTrack(track);
    }

    project.masterVolume = juce::jlimit(0.0, 2.0, project.masterVolume);
    juce::StringArray cleanedMasterFxChain;
    for (const auto& entry : project.masterFxChain)
    {
        const auto trimmed = entry.trim();
        if (trimmed.isNotEmpty())
            cleanedMasterFxChain.addIfNotAlreadyThere(trimmed);
    }
    project.masterFxChain = cleanedMasterFxChain;

    project.recalculateTimeFields();
}

class TrackEditAction final : public juce::UndoableAction
{
public:
    TrackEditAction(std::function<void(const TrackState&)> applyStateIn,
                    TrackState beforeStateIn,
                    TrackState afterStateIn)
        : applyState(std::move(applyStateIn)),
          beforeState(std::move(beforeStateIn)),
          afterState(std::move(afterStateIn))
    {
    }

    bool perform() override
    {
        applyState(afterState);
        return true;
    }

    bool undo() override
    {
        applyState(beforeState);
        return true;
    }

    int getSizeInUnits() override
    {
        return 32;
    }

private:
    std::function<void(const TrackState&)> applyState;
    TrackState beforeState;
    TrackState afterState;
};

class ProjectEditAction final : public juce::UndoableAction
{
public:
    ProjectEditAction(std::function<void(const ProjectState&)> applyStateIn,
                      ProjectState beforeStateIn,
                      ProjectState afterStateIn)
        : applyState(std::move(applyStateIn)),
          beforeState(std::move(beforeStateIn)),
          afterState(std::move(afterStateIn))
    {
    }

    bool perform() override
    {
        applyState(afterState);
        return true;
    }

    bool undo() override
    {
        applyState(beforeState);
        return true;
    }

    int getSizeInUnits() override
    {
        return 64;
    }

private:
    std::function<void(const ProjectState&)> applyState;
    ProjectState beforeState;
    ProjectState afterState;
};
} // namespace

StudioShellComponent::TrackTableModel::TrackTableModel(StudioShellComponent& ownerIn)
    : owner(ownerIn)
{
}

int StudioShellComponent::TrackTableModel::getNumRows()
{
    return static_cast<int>(owner.documentState.project.tracks.size()) + 1;
}

void StudioShellComponent::TrackTableModel::backgroundClicked(const juce::MouseEvent& event)
{
    if (event.mods.isLeftButtonDown() && event.getNumberOfClicks() >= 2)
        owner.addTrack();
}

void StudioShellComponent::TrackTableModel::paintRowBackground(juce::Graphics& g,
                                                               int rowNumber,
                                                               int width,
                                                               int height,
                                                               bool rowIsSelected)
{
    juce::ignoreUnused(height);

    const auto background = (rowNumber % 2) == 0 ? juce::Colour::fromRGB(26, 29, 36)
                                                 : juce::Colour::fromRGB(21, 24, 30);
    g.fillAll(background);

    if (!juce::isPositiveAndBelow(rowNumber, static_cast<int>(owner.documentState.project.tracks.size())))
    {
        g.setColour(juce::Colour::fromRGB(42, 47, 58));
        g.drawHorizontalLine(height - 1, 0.0f, static_cast<float>(width));
        return;
    }

    const auto trackColour = trackDisplayColour(owner.documentState.project.tracks[static_cast<size_t>(rowNumber)], rowNumber);
    g.setColour(trackColour.withAlpha(rowIsSelected ? 0.78f : 0.16f));
    g.fillRect(0, 0, width, height);
    g.setColour(trackColour);
    g.fillRect(0, 0, 5, height);
}

void StudioShellComponent::TrackTableModel::paintCell(juce::Graphics& g,
                                                      int rowNumber,
                                                      int columnId,
                                                      int width,
                                                      int height,
                                                      bool rowIsSelected)
{
    juce::ignoreUnused(height);
    if (rowNumber < 0)
        return;

    if (rowNumber >= static_cast<int>(owner.documentState.project.tracks.size()))
    {
        if (columnId == kColumnRack || columnId == kColumnName)
        {
            g.setColour(rowIsSelected ? juce::Colours::white : juce::Colour::fromRGB(144, 154, 171));
            g.setFont(columnId == kColumnName ? ui::sectionFont() : ui::font());
            const auto text = columnId == kColumnName ? "Double-click to add track" : "New Track";
            g.drawText(text, 8, 0, width - 12, height, juce::Justification::centredLeft, true);
        }
        return;
    }

    if (rowNumber < 0 || rowNumber >= static_cast<int>(owner.documentState.project.tracks.size()))
        return;

    const auto& track = owner.documentState.project.tracks[static_cast<size_t>(rowNumber)];
    const auto trackColour = trackDisplayColour(track, rowNumber);

    if (columnId == kColumnMute || columnId == kColumnSolo || columnId == kColumnVstView)
    {
        const bool isMuteColumn = columnId == kColumnMute;
        const bool isSoloColumn = columnId == kColumnSolo;
        const bool isVstColumn = columnId == kColumnVstView;
        const bool canToggleVstView = track.trackType != "sample"
            && track.instrumentMode.containsIgnoreCase("VST");
        const auto active = isMuteColumn ? track.mute
                           : (isSoloColumn ? track.solo
                                           : (canToggleVstView && owner.isTrackBeingLiveEdited(rowNumber)));
        const auto label = isMuteColumn ? "M"
                         : (isSoloColumn ? "S" : "V");
        const auto accentColour = isMuteColumn
            ? juce::Colour::fromRGB(232, 96, 96)
            : (isSoloColumn
                   ? juce::Colour::fromRGB(244, 211, 94)
                   : juce::Colour::fromRGB(108, 212, 255));
        const auto chipWidth = juce::jmax(24, width - 14);
        const auto chipHeight = juce::jlimit(18, 22, height - 10);
        auto chipBounds = juce::Rectangle<int>(0, 0, chipWidth, chipHeight)
            .withCentre(juce::Point<int>(width / 2, height / 2));

        const auto disabled = isVstColumn && !canToggleVstView;
        const auto inactiveChipColour = disabled
            ? juce::Colour::fromRGB(30, 34, 42)
            : juce::Colour::fromRGB(38, 43, 52).interpolatedWith(accentColour, rowIsSelected ? 0.14f : 0.09f);
        g.setColour(active ? accentColour : inactiveChipColour);
        g.fillRoundedRectangle(chipBounds.toFloat(), 6.0f);
        g.setColour(active ? accentColour.brighter(0.12f)
                           : (disabled
                                  ? juce::Colour::fromRGB(52, 57, 70)
                                  : accentColour.withAlpha(rowIsSelected ? 0.82f : 0.54f)));
        g.drawRoundedRectangle(chipBounds.toFloat(), 6.0f, 1.2f);
        g.setColour(active
                        ? (isSoloColumn ? juce::Colour::fromRGB(28, 28, 28) : juce::Colours::white)
                        : (disabled
                               ? juce::Colour::fromRGB(112, 120, 134)
                               : juce::Colour::fromRGB(222, 228, 236)));
        g.setFont(ui::strongFont());
        g.drawText(label, chipBounds, juce::Justification::centred, false);

        g.setColour(juce::Colour::fromRGB(45, 50, 62));
        g.fillRect(width - 1, 0, 1, height);
        return;
    }

    if (columnId == kColumnVolume)
    {
        const auto meterLevel = juce::isPositiveAndBelow(rowNumber, static_cast<int>(owner.trackMeterLevels.size()))
            ? owner.trackMeterLevels[static_cast<size_t>(rowNumber)]
            : 0.0f;
        auto meterBounds = juce::Rectangle<int>(0, 0, width, height).reduced(10, 5);
        meterBounds = meterBounds.withSizeKeepingCentre(18, juce::jmax(18, meterBounds.getHeight()));
        paintTrackLevelBar(g, meterBounds, meterLevel, static_cast<float>(track.volume), trackColour);

        g.setColour(juce::Colour::fromRGB(45, 50, 62));
        g.fillRect(width - 1, 0, 1, height);
        return;
    }

    juce::String text;

    switch (columnId)
    {
        case kColumnName: text = track.name; break;
        case kColumnType: text = track.trackType; break;
        case kColumnMode: text = track.instrumentMode; break;
        case kColumnRack: text = displayRackName(owner.documentState.project, track); break;
        case kColumnNotes: text = juce::String(static_cast<int>(track.notes.size())); break;
        case kColumnChannel: text = juce::String(track.midiChannel + 1); break;
        case kColumnPan: text = juce::String(track.pan, 2); break;
        case kColumnFlags: text = makeTrackFlags(track); break;
        default: break;
    }

    auto textColour = rowIsSelected ? trackTextColour(trackColour)
                                    : juce::Colour::fromRGB(227, 233, 241);
    if (!rowIsSelected && columnId == kColumnName)
        textColour = trackColour.brighter(0.18f);

    g.setColour(textColour);
    g.setFont(columnId == kColumnName ? ui::sectionFont() : ui::font());
    g.drawText(text, 8, 0, width - 12, height, juce::Justification::centredLeft, true);

    g.setColour(juce::Colour::fromRGB(45, 50, 62));
    g.fillRect(width - 1, 0, 1, height);
}

void StudioShellComponent::TrackTableModel::selectedRowsChanged(int)
{
    if (!juce::isPositiveAndBelow(owner.trackTable.getSelectedRow(),
                                  static_cast<int>(owner.documentState.project.tracks.size())))
        return;

    owner.ensureSelectedMidiSectionForTrack(owner.getSelectedTrackIndex());
    owner.refreshInspector();
    owner.updateEditorState();
    owner.scheduleSelectedTrackRackPreviewWarmup();
}

void StudioShellComponent::TrackTableModel::cellClicked(int rowNumber, int columnId, const juce::MouseEvent& event)
{
    if (!juce::isPositiveAndBelow(rowNumber, static_cast<int>(owner.documentState.project.tracks.size())))
    {
        if (event.mods.isLeftButtonDown() && event.getNumberOfClicks() >= 2)
            owner.addTrack();
        return;
    }

    owner.setSelectedTrackIndex(rowNumber);
    if (!event.mods.isRightButtonDown()
        && (columnId == kColumnMute || columnId == kColumnSolo || columnId == kColumnVstView || columnId == kColumnVolume))
    {
        auto updatedTrack = owner.documentState.project.tracks[static_cast<size_t>(rowNumber)];
        if (columnId == kColumnMute)
        {
            updatedTrack.mute = !updatedTrack.mute;
            owner.applyTrackStateEdit(rowNumber, updatedTrack, "Toggle Mute");
            owner.trackTable.repaint(owner.trackTable.getRowPosition(rowNumber, true));
        }
        else
        {
            if (columnId == kColumnSolo)
            {
                updatedTrack.solo = !updatedTrack.solo;
                owner.applyTrackStateEdit(rowNumber, updatedTrack, "Toggle Solo");
                owner.trackTable.repaint(owner.trackTable.getRowPosition(rowNumber, true));
            }
            else
            {
                if (columnId == kColumnVolume)
                {
                    if (owner.isMixerWindowVisible())
                        owner.setMixerWindowVisible(false);
                    else
                        owner.focusMixerPanel();
                }
                else
                {
                    const bool canToggleVstView = updatedTrack.trackType != "sample"
                        && updatedTrack.instrumentMode.containsIgnoreCase("VST");
                    if (!canToggleVstView)
                        return;

                    if (auto* session = owner.findRackEditorSession(rowNumber);
                        session != nullptr && session->editorOpen)
                    {
                        const auto closeResult = owner.nativeVstHost.closeAudioEngineTrackEditor(rowNumber);
                        if (closeResult.failed())
                        {
                            juce::AlertWindow::showMessageBoxAsync(juce::AlertWindow::WarningIcon,
                                                                   "Editor Close Failed",
                                                                   closeResult.getErrorMessage());
                        }
                        else
                        {
                            owner.closeRackEditorSession(rowNumber);
                            owner.refreshPollingTimerState();
                            owner.updateEditorState();
                            owner.statusLabel.setText("Hid VST editor for " + updatedTrack.name + ".", juce::dontSendNotification);
                        }
                    }
                    else
                    {
                        owner.openSelectedTrackRackEditor();
                    }

                    owner.trackTable.repaint();
                }
            }
        }
        return;
    }

    if (event.mods.isRightButtonDown())
        owner.showTrackContextMenu(rowNumber, event.getScreenPosition().roundToInt());
}

StudioShellComponent::SampleAssetListModel::SampleAssetListModel(StudioShellComponent& ownerIn)
    : owner(ownerIn)
{
}

int StudioShellComponent::SampleAssetListModel::getNumRows()
{
    return static_cast<int>(owner.documentState.project.sampleAssets.size());
}

void StudioShellComponent::SampleAssetListModel::paintListBoxItem(int rowNumber,
                                                                  juce::Graphics& g,
                                                                  int width,
                                                                  int height,
                                                                  bool rowIsSelected)
{
    juce::ignoreUnused(height);
    if (!juce::isPositiveAndBelow(rowNumber, static_cast<int>(owner.documentState.project.sampleAssets.size())))
        return;

    const auto& asset = owner.documentState.project.sampleAssets[static_cast<size_t>(rowNumber)];
    const auto text = juce::File(asset.path).getFileName()
        + "  (" + juce::String(asset.durationSec, 2) + " s)";

    g.fillAll(rowIsSelected ? juce::Colour::fromRGB(46, 88, 138)
                            : ((rowNumber % 2) == 0 ? juce::Colour::fromRGB(26, 30, 37)
                                                    : juce::Colour::fromRGB(21, 25, 31)));
    g.setColour(rowIsSelected ? juce::Colours::white : juce::Colour::fromRGB(226, 232, 240));
    g.setFont(ui::font());
    g.drawText(text, 8, 0, width - 12, height, juce::Justification::centredLeft, true);
}

void StudioShellComponent::SampleAssetListModel::selectedRowsChanged(int)
{
    owner.updateEditorState();
}

StudioShellComponent::StudioShellComponent()
    : tableModel(*this),
      trackTable("Tracks", &tableModel),
      sampleAssetListModel(*this)
{
    compactHeaderLookAndFeel = std::make_unique<CompactHeaderLookAndFeel>();
    documentState = makeDefaultProjectFile();
    activityLogFile = nativeLogsDirectory().getChildFile("native-activity-"
        + juce::Time::getCurrentTime().formatted("%Y%m%d-%H%M%S")
        + ".log");
    windowStateSettings = std::make_unique<juce::PropertiesFile>(nativeWindowSettingsOptions());
    availableUiFonts = buildUiFontChoices();
    restorePersistedThemeSelection();
    restorePersistedFontSelection();
    restorePersistedWindowVisibility();
    restorePersistedSessionState();
    syncBundledRackCatalogInProject();
    aiClient.setActivityLogCallback([safeThis = juce::Component::SafePointer<StudioShellComponent>(this)] (const juce::String& title,
                                                                                                         const juce::String& body)
    {
        juce::MessageManager::callAsync([safeThis, title, body]
        {
            if (safeThis == nullptr)
                return;

            safeThis->appendActivityLog(title, body);
        });
    });
    aceStepClient.setActivityLogCallback([safeThis = juce::Component::SafePointer<StudioShellComponent>(this)] (const juce::String& title,
                                                                                                                const juce::String& body)
    {
        juce::MessageManager::callAsync([safeThis, title, body]
        {
            if (safeThis == nullptr)
                return;

            safeThis->appendActivityLog(title, body);

            if (title == "ACE-Step" && aceStepBootstrapMessageMatches(body))
            {
                safeThis->statusLabel.setText(aceStepBootstrapStatusText(), juce::dontSendNotification);

                if (!safeThis->aceStepBootstrapNoticeShown)
                {
                    safeThis->aceStepBootstrapNoticeShown = true;
                    juce::AlertWindow::showMessageBoxAsync(juce::AlertWindow::InfoIcon,
                                                           "ACE-Step Setup In Progress",
                                                           aceStepBootstrapUserMessage());
                }
            }
        });
    });
    if (aceStepClient.getInstallDirectory().trim().isEmpty())
    {
        const auto detectedAceStep = detectedAceStepInstallDirectory();
        if (detectedAceStep.isDirectory())
            aceStepClient.setInstallDirectory(detectedAceStep.getFullPathName());
    }
    nativeVstHost.setEditorStateCallback([safeThis = juce::Component::SafePointer<StudioShellComponent>(this)] (bool isOpen)
    {
        auto applyState = [safeThis, isOpen]
        {
            if (safeThis == nullptr)
                return;

            safeThis->loadedRackEditorOpen = isOpen;
            safeThis->lastRackEditorSessionSyncMs = 0;
            safeThis->refreshPollingTimerState();
        };

        if (auto* messageManager = juce::MessageManager::getInstanceWithoutCreating();
            messageManager != nullptr && messageManager->isThisTheMessageThread())
        {
            applyState();
        }
        else
        {
            juce::MessageManager::callAsync([applyState]
            {
                applyState();
            });
        }
    });

    headerLogoImage = loadMutagenLogoBinaryData(false);
    if (headerLogoImage.isValid())
        headerLogo.setImage(headerLogoImage);
    headerLogo.setImagePlacement(juce::RectanglePlacement::centred);
    addAndMakeVisible(headerLogo);
    headerLabel.setVisible(false);

    fileLabel.setColour(juce::Label::textColourId, juce::Colour::fromRGB(165, 176, 191));
    fileLabel.setFont(ui::strongFont());
    addAndMakeVisible(fileLabel);

    headerTimecodeDisplay = std::make_unique<HeaderLcdDisplay>();
    addAndMakeVisible(*headerTimecodeDisplay);

    statsLabel.setColour(juce::Label::textColourId, juce::Colour::fromRGB(210, 216, 224));
    statsLabel.setFont(ui::font());
    addChildComponent(statsLabel);

    statusLabel.setColour(juce::Label::textColourId, juce::Colour::fromRGB(143, 225, 170));
    statusLabel.setFont(ui::font());
    addAndMakeVisible(statusLabel);

    aiStatusSummaryLabel.setColour(juce::Label::textColourId, juce::Colour::fromRGB(156, 199, 239));
    aiStatusSummaryLabel.setFont(ui::font());
    aiStatusSummaryLabel.setJustificationType(juce::Justification::centredRight);
    addAndMakeVisible(aiStatusSummaryLabel);

    inspectorLabel.setText("Selected Track", juce::dontSendNotification);
    inspectorLabel.setFont(ui::sectionFont());
    inspectorLabel.setColour(juce::Label::textColourId, juce::Colour::fromRGB(235, 239, 244));
    addAndMakeVisible(inspectorLabel);

    configureInspectorTextEditor(trackNameEditor, "Track name");
    trackNameEditor.onFocusLost = [this]
    {
        if (syncingInspectorControls)
            return;
        const auto name = trackNameEditor.getText().trim();
        applySelectedTrackMutation([name] (TrackState& track)
                                   {
                                       track.name = name.isNotEmpty() ? name : "Track";
                                   },
                                   "Rename Track");
    };
    trackNameEditor.onReturnKey = [this] { trackNameEditor.giveAwayKeyboardFocus(); };

    configureInspectorTextEditor(trackTypeEditor, "Track type");
    trackTypeEditor.onFocusLost = [this]
    {
        if (syncingInspectorControls)
            return;
        const auto value = trackTypeEditor.getText().trim();
        applySelectedTrackMutation([value] (TrackState& track)
                                   {
                                       track.trackType = value.isNotEmpty() ? value : "instrument";
                                   },
                                   "Change Track Type");
    };
    trackTypeEditor.onReturnKey = [this] { trackTypeEditor.giveAwayKeyboardFocus(); };

    configureInspectorTextEditor(instrumentModeEditor, "Instrument mode");
    instrumentModeEditor.onFocusLost = [this]
    {
        if (syncingInspectorControls)
            return;
        const auto value = instrumentModeEditor.getText().trim();
        applySelectedTrackMutation([value] (TrackState& track)
                                   {
                                       track.instrumentMode = value.isNotEmpty() ? value : "General MIDI";
                                   },
                                   "Change Instrument Mode");
    };
    instrumentModeEditor.onReturnKey = [this] { instrumentModeEditor.giveAwayKeyboardFocus(); };

    configureInspectorTextEditor(instrumentEditor, "Instrument name");
    instrumentEditor.onFocusLost = [this]
    {
        if (syncingInspectorControls)
            return;
        const auto value = instrumentEditor.getText().trim();
        applySelectedTrackMutation([value] (TrackState& track)
                                   {
                                       track.instrument = value.isNotEmpty() ? value : "Piano";
                                   },
                                   "Change Instrument");
    };
    instrumentEditor.onReturnKey = [this] { instrumentEditor.giveAwayKeyboardFocus(); };

    configureInspectorTextEditor(rackVstEditor, "Rack VST path or name");
    rackVstEditor.onFocusLost = [this]
    {
        if (syncingInspectorControls)
            return;
        const auto value = rackVstEditor.getText().trim();
        applySelectedTrackMutation([value] (TrackState& track)
                                   {
                                       track.rackVst = value;
                                   },
                                   "Change Rack VST");
    };
    rackVstEditor.onReturnKey = [this] { rackVstEditor.giveAwayKeyboardFocus(); };

    configureInspectorSlider(midiChannelSlider, 1.0, 16.0, 1.0, " ch");
    midiChannelSlider.onValueChange = [this]
    {
        if (syncingInspectorControls)
            return;
        applySelectedTrackMutation([value = juce::roundToInt(midiChannelSlider.getValue())] (TrackState& track)
                                   {
                                       track.midiChannel = juce::jlimit(0, 15, value - 1);
                                   },
                                   "Change MIDI Channel");
    };

    configureInspectorSlider(midiProgramSlider, 0.0, 127.0, 1.0, " pgm");
    midiProgramSlider.onValueChange = [this]
    {
        if (syncingInspectorControls)
            return;
        applySelectedTrackMutation([value = juce::roundToInt(midiProgramSlider.getValue())] (TrackState& track)
                                   {
                                       track.midiProgram = juce::jlimit(0, 127, value);
                                   },
                                   "Change MIDI Program");
    };

    configureInspectorSlider(volumeSlider, 0.0, 1.0, 0.01, "");
    volumeSlider.onValueChange = [this]
    {
        if (syncingInspectorControls)
            return;
        applySelectedTrackMutation([value = volumeSlider.getValue()] (TrackState& track)
                                   {
                                       track.volume = juce::jlimit(0.0, 1.0, value);
                                   },
                                   "Change Track Volume");
    };

    configureInspectorSlider(panSlider, -1.0, 1.0, 0.01, "");
    panSlider.onValueChange = [this]
    {
        if (syncingInspectorControls)
            return;
        applySelectedTrackMutation([value = panSlider.getValue()] (TrackState& track)
                                   {
                                       track.pan = juce::jlimit(-1.0, 1.0, value);
                                   },
                                   "Change Track Pan");
    };

    muteToggle.setButtonText("Mute");
    muteToggle.onClick = [this]
    {
        if (syncingInspectorControls)
            return;
        applySelectedTrackMutation([value = muteToggle.getToggleState()] (TrackState& track)
                                   {
                                       track.mute = value;
                                   },
                                   "Toggle Mute");
    };
    addAndMakeVisible(muteToggle);

    soloToggle.setButtonText("Solo");
    soloToggle.onClick = [this]
    {
        if (syncingInspectorControls)
            return;
        applySelectedTrackMutation([value = soloToggle.getToggleState()] (TrackState& track)
                                   {
                                       track.solo = value;
                                   },
                                   "Toggle Solo");
    };
    addAndMakeVisible(soloToggle);

    liveArmToggle.setButtonText("Arm");
    liveArmToggle.onClick = [this]
    {
        if (syncingInspectorControls)
            return;
        applySelectedTrackMutation([value = liveArmToggle.getToggleState()] (TrackState& track)
                                   {
                                       track.liveArmed = value;
                                   },
                                   "Toggle Record Arm");
    };
    addAndMakeVisible(liveArmToggle);

    inspectorEditor.setMultiLine(true);
    inspectorEditor.setReadOnly(true);
    inspectorEditor.setScrollbarsShown(true);
    inspectorEditor.setColour(juce::TextEditor::backgroundColourId, juce::Colour::fromRGB(18, 20, 25));
    inspectorEditor.setColour(juce::TextEditor::textColourId, juce::Colour::fromRGB(226, 230, 237));
    inspectorEditor.setColour(juce::TextEditor::outlineColourId, juce::Colour::fromRGB(56, 64, 79));
    inspectorEditor.setFont(ui::font());
    addAndMakeVisible(inspectorEditor);

    inspectorViewport.setScrollBarsShown(true, false);
    inspectorViewport.setScrollBarThickness(10);
    inspectorViewport.setViewedComponent(&inspectorContent, false);
    addAndMakeVisible(inspectorViewport);

    auto attachInspectorComponent = [this] (juce::Component& component)
    {
        inspectorContent.addAndMakeVisible(component);
    };

    attachInspectorComponent(inspectorLabel);
    attachInspectorComponent(trackNameEditor);
    attachInspectorComponent(trackTypeEditor);
    attachInspectorComponent(instrumentModeEditor);
    attachInspectorComponent(instrumentEditor);
    attachInspectorComponent(rackVstEditor);
    attachInspectorComponent(midiChannelSlider);
    attachInspectorComponent(midiProgramSlider);
    attachInspectorComponent(volumeSlider);
    attachInspectorComponent(panSlider);
    attachInspectorComponent(muteToggle);
    attachInspectorComponent(soloToggle);
    attachInspectorComponent(liveArmToggle);
    attachInspectorComponent(inspectorEditor);

    auto restoreInspectorComponentToShell = [this] (juce::Component& component)
    {
        if (auto* parent = component.getParentComponent(); parent != nullptr && parent != this)
            parent->removeChildComponent(&component);
        addAndMakeVisible(component);
    };

    restoreInspectorComponentToShell(inspectorLabel);
    restoreInspectorComponentToShell(trackNameEditor);
    restoreInspectorComponentToShell(trackTypeEditor);
    restoreInspectorComponentToShell(instrumentModeEditor);
    restoreInspectorComponentToShell(instrumentEditor);
    restoreInspectorComponentToShell(rackVstEditor);
    restoreInspectorComponentToShell(midiChannelSlider);
    restoreInspectorComponentToShell(midiProgramSlider);
    restoreInspectorComponentToShell(volumeSlider);
    restoreInspectorComponentToShell(panSlider);
    restoreInspectorComponentToShell(muteToggle);
    restoreInspectorComponentToShell(soloToggle);
    restoreInspectorComponentToShell(liveArmToggle);
    restoreInspectorComponentToShell(inspectorEditor);
    inspectorViewport.setVisible(false);

    mixerLabel.setText("Mixer", juce::dontSendNotification);
    mixerLabel.setFont(ui::sectionFont());
    mixerLabel.setColour(juce::Label::textColourId, juce::Colour::fromRGB(235, 239, 244));
    addAndMakeVisible(mixerLabel);

    mixerComponent = std::make_unique<MixerComponent>(
        [this] () -> const ProjectState& { return documentState.project; },
        [this] (int trackIndex, const TrackState& track, bool undoable, const juce::String& actionName)
        {
            if (trackIndex < 0)
                return;

            if (undoable)
                applyTrackStateEdit(trackIndex, track, actionName);
            else
                replaceTrackStateNoUndo(trackIndex, track);
        },
        [this] (const ProjectState& project, bool undoable, const juce::String& actionName)
        {
            if (undoable)
                applyProjectStateEdit(project, actionName);
            else
                setProjectStateInternal(project);
        },
        [this] (int trackIndex) -> float
        {
            if (!juce::isPositiveAndBelow(trackIndex, static_cast<int>(trackMeterLevels.size())))
                return 0.0f;
            return trackMeterLevels[static_cast<size_t>(trackIndex)];
        },
        [this] () -> std::pair<float, float>
        {
            return { transportMasterPeakLeft, transportMasterPeakRight };
        });
    mixerViewport.setViewedComponent(mixerComponent.get(), false);
    mixerViewport.setScrollBarsShown(true, false);
    addAndMakeVisible(mixerViewport);

    samplesLabel.setText("Samples", juce::dontSendNotification);
    samplesLabel.setFont(ui::sectionFont());
    samplesLabel.setColour(juce::Label::textColourId, juce::Colour::fromRGB(235, 239, 244));
    addAndMakeVisible(samplesLabel);

    sampleLibraryLabel.setText("Library", juce::dontSendNotification);
    sampleLibraryLabel.setColour(juce::Label::textColourId, juce::Colour::fromRGB(190, 199, 210));
    addAndMakeVisible(sampleLibraryLabel);

    sampleTimelineLabel.setText("Timeline", juce::dontSendNotification);
    sampleTimelineLabel.setColour(juce::Label::textColourId, juce::Colour::fromRGB(190, 199, 210));
    addAndMakeVisible(sampleTimelineLabel);

    sampleAssetList.setModel(&sampleAssetListModel);
    sampleAssetList.setRowHeight(28);
    sampleAssetList.setColour(juce::ListBox::backgroundColourId, juce::Colour::fromRGB(20, 22, 28));
    sampleAssetList.setOutlineThickness(1);
    addAndMakeVisible(sampleAssetList);

    sampleTimeline = std::make_unique<SampleTimelineComponent>(
        [this] () -> const ProjectState& { return documentState.project; },
        [this] (const ProjectState& project, bool undoable, const juce::String& actionName)
        {
            if (undoable)
                applyProjectStateEdit(project, actionName);
            else
                setProjectStateInternal(project);
        });
    sampleTimelineViewport.setViewedComponent(sampleTimeline.get(), false);
    sampleTimelineViewport.setScrollBarsShown(true, true);
    addAndMakeVisible(sampleTimelineViewport);

    automationLabel.setText("Automation", juce::dontSendNotification);
    automationLabel.setFont(ui::sectionFont());
    automationLabel.setColour(juce::Label::textColourId, juce::Colour::fromRGB(235, 239, 244));
    addAndMakeVisible(automationLabel);

    automationEditor = std::make_unique<AutomationEditorComponent>(
        [this] () -> const ProjectState& { return documentState.project; },
        [this] () -> int { return getSelectedTrackIndex(); },
        [this] (int trackIndex, const TrackState& track, bool undoable, const juce::String& actionName)
        {
            if (trackIndex < 0)
                return;

            if (undoable)
                applyTrackStateEdit(trackIndex, track, actionName);
            else
                replaceTrackStateNoUndo(trackIndex, track);
        });
    addAndMakeVisible(*automationEditor);

    pianoRollLabel.setText("Piano Roll", juce::dontSendNotification);
    pianoRollLabel.setFont(ui::sectionFont());
    pianoRollLabel.setColour(juce::Label::textColourId, juce::Colour::fromRGB(235, 239, 244));
    addAndMakeVisible(pianoRollLabel);
    pianoRollLabel.setVisible(false);

    arrangementLabel.setText("Sequencer", juce::dontSendNotification);
    arrangementLabel.setFont(ui::sectionFont());
    arrangementLabel.setColour(juce::Label::textColourId, juce::Colour::fromRGB(235, 239, 244));
    addAndMakeVisible(arrangementLabel);
    arrangementLabel.setVisible(false);

    arrangementOverview = std::make_unique<ArrangementOverviewComponent>(
        [this] () -> const ProjectState& { return documentState.project; },
        [this] (const ProjectState& project, bool undoable, const juce::String& actionName)
        {
            if (undoable)
                applyProjectStateEdit(project, actionName);
            else
                setProjectStateInternal(project);
        },
        [this] () { return getSelectedMidiSectionIndex(); },
        [this] (int sectionIndex, bool focusEditor)
        {
            setSelectedMidiSectionIndex(sectionIndex, true);
            if (focusEditor)
                focusMidiSectionInPianoRoll(sectionIndex);
        },
        [this] () { return editorToolMode; },
        [this] (EditorToolMode mode)
        {
            setEditorToolMode(mode);
        });
    arrangementOverview->setZoomChangedCallback([this] (float pixelsPerBar)
                                                {
                                                    if (std::abs(arrangementZoomPixelsPerBar - pixelsPerBar) < 0.01f)
                                                        return;

                                                    arrangementZoomPixelsPerBar = pixelsPerBar;
                                                    applyEditorViewScaleState();
                                                });
    arrangementOverview->setLaneHeightChangedCallback([this] (float laneHeightPixels)
                                                      {
                                                          if (std::abs(arrangementLaneHeightPixels - laneHeightPixels) < 0.01f)
                                                              return;

                                                          arrangementLaneHeightPixels = laneHeightPixels;
                                                          applyEditorViewScaleState();
                                                      });
    arrangementOverview->setSampleClipStemSeparationCallback([this] (int clipIndex)
                                                             {
                                                                 separateSampleClipToStems(clipIndex);
                                                             });
    arrangementOverview->setSampleFileDropCallback([this] (const juce::StringArray& files,
                                                           int targetTrackIndex,
                                                           int targetStartTick)
                                                   {
                                                       for (const auto& filePath : files)
                                                       {
                                                           const juce::File file(filePath);
                                                           if (!file.existsAsFile())
                                                               continue;

                                                           juce::String trackName;
                                                           const auto result = placeSampleFileOnTrackAtTick(file,
                                                                                                            targetTrackIndex,
                                                                                                            targetStartTick,
                                                                                                            "Drop Sample Clip",
                                                                                                            trackName);
                                                           if (result.wasOk())
                                                           {
                                                               statusLabel.setText("Dropped sample on " + trackName + ".", juce::dontSendNotification);
                                                               return result;
                                                           }

                                                           return result;
                                                       }

                                                       return juce::Result::fail("No valid audio file was dropped.");
                                                   });
    arrangementViewport.setViewedComponent(arrangementOverview.get(), false);
    arrangementViewport.setScrollBarsShown(true, true);
    addAndMakeVisible(arrangementViewport);

    pianoRoll = std::make_unique<PianoRollComponent>([this] () -> const ProjectState& { return documentState.project; },
                                                     [this] () { return getSelectedTrackIndex(); },
                                                     [this] () { return getSelectedMidiSectionIndex(); },
                                                     [this] (const ProjectState& project, bool undoable, const juce::String& actionName)
                                                     {
                                                         if (undoable)
                                                             applyProjectStateEdit(project, actionName);
                                                         else
                                                             setProjectStateInternal(project);
                                                     });
    pianoRoll->setZoomChangedCallback([this] (float pixelsPerBeat)
                                      {
                                          if (std::abs(pianoRollZoomPixelsPerBeat - pixelsPerBeat) < 0.01f)
                                              return;

                                          pianoRollZoomPixelsPerBeat = pixelsPerBeat;
                                          applyEditorViewScaleState();
                                      });
    pianoRoll->setRowHeightChangedCallback([this] (float rowHeightPixels)
                                           {
                                               if (std::abs(pianoRollRowHeightPixels - rowHeightPixels) < 0.01f)
                                                   return;

                                               pianoRollRowHeightPixels = rowHeightPixels;
                                               applyEditorViewScaleState();
                                           });
    pianoRoll->setNotePreviewCallbacks([this] (int pitch, int velocity)
                                       {
                                           previewSelectedTrackMidiNoteOn(pitch, velocity);
                                       },
                                       [this] (int pitch, int velocity)
                                       {
                                           previewSelectedTrackMidiNoteOff(pitch, velocity);
                                       },
                                       [this]
                                       {
                                           stopSelectedTrackMidiPreview();
                                       });
    pianoRoll->setToolModeChangeCallback([this] (EditorToolMode mode)
                                         {
                                             setEditorToolMode(mode);
                                         });
    pianoRoll->setKeyHandlerCallback([this] (const juce::KeyPress& key)
                                     {
                                         return keyPressed(key);
                                     });
    pianoRollViewport.setViewedComponent(pianoRoll.get(), false);
    pianoRollViewport.setScrollBarsShown(true, true);
    addAndMakeVisible(pianoRollViewport);
    pianoRollViewport.setVisible(false);

    createToolbarButton(newButton, "New");
    createToolbarButton(openButton, "Open");
    createToolbarButton(saveButton, "Save");
    createToolbarButton(saveAsButton, "Save As");
    createToolbarButton(importMidiButton, "Import MIDI");
    createToolbarButton(exportWavButton, "Export WAV");
    createToolbarButton(exportMidiButton, "Export MIDI");
    createToolbarButton(importSampleButton, "Import Sample");
    createToolbarButton(placeSampleButton, "Place At Playhead");
    createToolbarButton(aiSettingsButton, "AI Settings");
    createToolbarButton(aiComposeButton, "Compose");
    createToolbarButton(aceStepGenerateButton, "Generate Audio");
    createToolbarButton(playProjectButton, "Play");
    createToolbarButton(recordToggle, "Rec");
    createToolbarButton(undoButton, "Undo");
    createToolbarButton(redoButton, "Redo");

    newButton.onClick = [this] { createNewProject(); };
    openButton.onClick = [this] { promptOpenProject(); };
    saveButton.onClick = [this] { saveProject(); };
    saveAsButton.onClick = [this] { saveProjectAs(); };
    importMidiButton.onClick = [this] { promptImportMidi(); };
    exportWavButton.onClick = [this] { promptExportWav(); };
    exportMidiButton.onClick = [this] { promptExportMidi(); };
    importSampleButton.onClick = [this] { promptImportSample(); };
    placeSampleButton.onClick = [this] { placeSelectedSampleAtPlayhead(); };
    aiSettingsButton.onClick = [this] { showAiSettingsDialog(); };
    aiComposeButton.onClick = [this] { composeWithAi(); };
    aceStepGenerateButton.onClick = [this] { generateAudioWithAceStep(); };
    playProjectButton.onClick = [this]
    {
        if (rackPreviewRunning || projectPreviewRunning)
            stopRackPreview();
        else
            playFullProjectThroughNativeEngine();
    };
    recordToggle.setClickingTogglesState(true);
    recordToggle.setToggleState(transportRecordEnabled, juce::dontSendNotification);
    recordToggle.setTooltip("Record incoming MIDI to the selected track during project playback.");
    recordToggle.setLookAndFeel(compactHeaderLookAndFeel.get());
    recordToggle.onClick = [this] { setTransportRecordEnabled(recordToggle.getToggleState()); };
    undoButton.onClick = [this] { undo(); };
    redoButton.onClick = [this] { redo(); };

    aiComposeButton.setTooltip("AI Compose");
    aceStepGenerateButton.setTooltip("Generate audio onto the selected sample track with ACE-Step.");

    playProjectButton.setLookAndFeel(compactHeaderLookAndFeel.get());
    recordToggle.setLookAndFeel(compactHeaderLookAndFeel.get());
    aiComposeButton.setLookAndFeel(compactHeaderLookAndFeel.get());
    aceStepGenerateButton.setLookAndFeel(compactHeaderLookAndFeel.get());
    refreshPlaybackToggleButton();

    tempoLabel.setText("Tempo", juce::dontSendNotification);
    tempoLabel.setColour(juce::Label::textColourId, juce::Colour::fromRGB(214, 219, 227));
    tempoLabel.setFont(ui::font());
    addAndMakeVisible(tempoLabel);

    tempoSlider.setRange(20.0, 300.0, 1.0);
    tempoSlider.setTextBoxStyle(juce::Slider::TextBoxRight, false, 62, 20);
    tempoSlider.setValue(documentState.project.bpm, juce::dontSendNotification);
    tempoSlider.onValueChange = [this] { handleTempoChanged(); };
    tempoSlider.setLookAndFeel(compactHeaderLookAndFeel.get());
    addAndMakeVisible(tempoSlider);

    timeSignatureLabel.setText("Time Sig", juce::dontSendNotification);
    timeSignatureLabel.setColour(juce::Label::textColourId, juce::Colour::fromRGB(214, 219, 227));
    timeSignatureLabel.setFont(ui::font());
    addAndMakeVisible(timeSignatureLabel);

    for (int numerator = 1; numerator <= 16; ++numerator)
        timeSignatureNumeratorBox.addItem(juce::String(numerator), numerator);
    timeSignatureNumeratorBox.setTextWhenNothingSelected("4");
    timeSignatureNumeratorBox.onChange = [this] { handleTimeSignatureChanged(); };
    timeSignatureNumeratorBox.setLookAndFeel(compactHeaderLookAndFeel.get());
    timeSignatureNumeratorBox.setJustificationType(juce::Justification::centredLeft);
    addAndMakeVisible(timeSignatureNumeratorBox);

    timeSignatureSlashLabel.setText("/", juce::dontSendNotification);
    timeSignatureSlashLabel.setJustificationType(juce::Justification::centred);
    timeSignatureSlashLabel.setColour(juce::Label::textColourId, juce::Colour::fromRGB(214, 219, 227));
    timeSignatureSlashLabel.setFont(ui::strongFont());
    addAndMakeVisible(timeSignatureSlashLabel);

    for (const auto denominator : { 1, 2, 4, 8, 16, 32 })
        timeSignatureDenominatorBox.addItem(juce::String(denominator), denominator);
    timeSignatureDenominatorBox.setTextWhenNothingSelected("4");
    timeSignatureDenominatorBox.onChange = [this] { handleTimeSignatureChanged(); };
    timeSignatureDenominatorBox.setLookAndFeel(compactHeaderLookAndFeel.get());
    timeSignatureDenominatorBox.setJustificationType(juce::Justification::centredLeft);
    addAndMakeVisible(timeSignatureDenominatorBox);

    patternBarsLabel.setText("Pattern Size", juce::dontSendNotification);
    patternBarsLabel.setColour(juce::Label::textColourId, juce::Colour::fromRGB(214, 219, 227));
    patternBarsLabel.setFont(ui::font());
    addAndMakeVisible(patternBarsLabel);

    populateSequenceTickBox(patternBarsBox, documentState.project);
    patternBarsBox.setTextWhenNothingSelected("Pattern size");
    patternBarsBox.onChange = [this] { handlePatternBarsChanged(); };
    patternBarsBox.setLookAndFeel(compactHeaderLookAndFeel.get());
    patternBarsBox.setJustificationType(juce::Justification::centredLeft);
    addAndMakeVisible(patternBarsBox);

    keyQuantizeLabel.setText("Key Quantize", juce::dontSendNotification);
    keyQuantizeLabel.setColour(juce::Label::textColourId, juce::Colour::fromRGB(214, 219, 227));
    keyQuantizeLabel.setFont(ui::font());
    addAndMakeVisible(keyQuantizeLabel);

    for (const auto& option : keyQuantizeOptions())
        keyQuantizeBox.addItem(option.label, option.id);
    keyQuantizeBox.setTextWhenNothingSelected("All Notes");
    keyQuantizeBox.onChange = [this] { handleKeyQuantizeChanged(); };
    keyQuantizeBox.setLookAndFeel(compactHeaderLookAndFeel.get());
    keyQuantizeBox.setJustificationType(juce::Justification::centredLeft);
    addAndMakeVisible(keyQuantizeBox);

    arrangementSnapLabel.setText("Seq Snap", juce::dontSendNotification);
    arrangementSnapLabel.setColour(juce::Label::textColourId, juce::Colour::fromRGB(214, 219, 227));
    arrangementSnapLabel.setFont(ui::font());
    addAndMakeVisible(arrangementSnapLabel);

    populateSequenceTickBox(arrangementSnapBox, documentState.project);
    arrangementSnapBox.setTextWhenNothingSelected("Sequencer snap");
    arrangementSnapBox.onChange = [this] { handleArrangementSnapChanged(); };
    arrangementSnapBox.setLookAndFeel(compactHeaderLookAndFeel.get());
    arrangementSnapBox.setJustificationType(juce::Justification::centredLeft);
    addAndMakeVisible(arrangementSnapBox);

    auto configureViewSlider = [this] (juce::Slider& slider,
                                       double minimum,
                                       double maximum,
                                       double step,
                                       const juce::String& suffix,
                                       double value,
                                       std::function<void()> onChange)
    {
        slider.setSliderStyle(juce::Slider::LinearHorizontal);
        slider.setRange(minimum, maximum, step);
        slider.setChangeNotificationOnlyOnRelease(false);
        slider.setTextBoxStyle(juce::Slider::TextBoxRight, false, 46, 20);
        slider.setTextValueSuffix(suffix);
        slider.setValue(value, juce::dontSendNotification);
        slider.onValueChange = std::move(onChange);
        slider.setLookAndFeel(compactHeaderLookAndFeel.get());
        addAndMakeVisible(slider);
    };

    arrangementZoomLabel.setText("Seq Zoom", juce::dontSendNotification);
    arrangementZoomLabel.setColour(juce::Label::textColourId, juce::Colour::fromRGB(214, 219, 227));
    arrangementZoomLabel.setFont(ui::font());
    addAndMakeVisible(arrangementZoomLabel);
    configureViewSlider(arrangementZoomSlider,
                        kArrangementMinPixelsPerBar,
                        kArrangementMaxPixelsPerBar,
                        1.0,
                        "",
                        arrangementZoomPixelsPerBar,
                        [this] { handleArrangementZoomChanged(); });

    arrangementLaneHeightLabel.setText("Seq Lane", juce::dontSendNotification);
    arrangementLaneHeightLabel.setColour(juce::Label::textColourId, juce::Colour::fromRGB(214, 219, 227));
    arrangementLaneHeightLabel.setFont(ui::font());
    addAndMakeVisible(arrangementLaneHeightLabel);
    configureViewSlider(arrangementLaneHeightSlider,
                        22.0,
                        80.0,
                        1.0,
                        " px",
                        arrangementLaneHeightPixels,
                        [this] { handleArrangementLaneHeightChanged(); });

    pianoRollZoomLabel.setVisible(false);
    pianoRollZoomSlider.setVisible(false);
    pianoRollRowHeightLabel.setVisible(false);
    pianoRollRowHeightSlider.setVisible(false);

    createToolbarButton(loopToggle, "Loop");
    loopToggle.setClickingTogglesState(true);
    loopToggle.setToggleState(documentState.project.loopEnabled, juce::dontSendNotification);
    loopToggle.setTooltip("Loop Playback");
    loopToggle.setLookAndFeel(compactHeaderLookAndFeel.get());
    loopToggle.onClick = [this] { setTransportLoopEnabled(loopToggle.getToggleState()); };

    createToolbarButton(metronomeToggle, "Metro");
    metronomeToggle.setClickingTogglesState(true);
    metronomeToggle.setToggleState(documentState.project.metronomeEnabled, juce::dontSendNotification);
    metronomeToggle.setTooltip("Metronome");
    metronomeToggle.setLookAndFeel(compactHeaderLookAndFeel.get());
    metronomeToggle.onClick = [this] { setTransportMetronomeEnabled(metronomeToggle.getToggleState()); };

    createToolbarButton(midiInsertToggle, "MIDI In");
    midiInsertToggle.setClickingTogglesState(true);
    midiInsertToggle.setToggleState(midiInsertEnabled, juce::dontSendNotification);
    midiInsertToggle.setTooltip("Insert incoming MIDI notes into the selected pattern instead of audition-only.");
    midiInsertToggle.setLookAndFeel(compactHeaderLookAndFeel.get());
    midiInsertToggle.onClick = [this]
    {
        midiInsertEnabled = midiInsertToggle.getToggleState();
        persistSessionState();
    };

    trackTable.getHeader().addColumn("Mute", kColumnMute, 58, 52, 72);
    trackTable.getHeader().addColumn("Solo", kColumnSolo, 58, 52, 72);
    trackTable.getHeader().addColumn("VST", kColumnVstView, 54, 48, 68);
    trackTable.getHeader().addColumn("Rack / Instrument", kColumnRack, 220, 140, 340);
    trackTable.getHeader().addColumn("Track", kColumnName, 180, 80, 320);
    trackTable.getHeader().addColumn("Vol", kColumnVolume, 52, 42, 72);
    trackTable.getHeader().addColumn("Type", kColumnType, 90, 60, 140);
    trackTable.getHeader().addColumn("Mode", kColumnMode, 140, 100, 220);
    trackTable.getHeader().addColumn("Notes", kColumnNotes, 70, 50, 100);
    trackTable.getHeader().addColumn("Ch", kColumnChannel, 55, 40, 70);
    trackTable.getHeader().addColumn("Pan", kColumnPan, 70, 50, 90);
    trackTable.getHeader().addColumn("Arm", kColumnFlags, 72, 56, 120);
    trackTable.setHeaderHeight(24);
    trackTable.setRowHeight(juce::roundToInt(arrangementLaneHeightPixels));
    trackTable.setColour(juce::ListBox::backgroundColourId, juce::Colour::fromRGB(20, 22, 28));
    trackTable.setOutlineThickness(1);
    addAndMakeVisible(trackTable);

    midiInputDeviceConnection = juce::MidiDeviceListConnection::make(
        [safeThis = juce::Component::SafePointer<StudioShellComponent>(this)]
        {
            juce::MessageManager::callAsync([safeThis]
            {
                if (safeThis != nullptr)
                    safeThis->refreshMidiInputDevices();
            });
        });
    refreshMidiInputDevices();

    setupFloatingWindows();
    applyEditorViewScaleState();
    setWantsKeyboardFocus(true);
    refreshPollingTimerState();
    refreshUi();
    applyTheme();
    updateAiStatusSummary();
    trackTable.selectRow(0);
    pianoRoll->grabKeyboardFocus();
    juce::Timer::callAfterDelay(0,
                                [safeThis = juce::Component::SafePointer<StudioShellComponent>(this)]
                                {
                                    if (safeThis != nullptr)
                                        safeThis->scheduleSelectedTrackRackPreviewWarmup(0);
                                });
    appendActivityLog("Log File", "Activity session log\n" + activityLogFile.getFullPathName());
    appendActivityLog("App", "Mutagen shell started.");
}

StudioShellComponent::~StudioShellComponent()
{
    persistSessionState();
    cancelPendingUpdate();
    midiInputDeviceConnection.reset();
    midiInputs.clear();
    playProjectButton.setLookAndFeel(nullptr);
    recordToggle.setLookAndFeel(nullptr);
    aiComposeButton.setLookAndFeel(nullptr);
    aceStepGenerateButton.setLookAndFeel(nullptr);
    loopToggle.setLookAndFeel(nullptr);
    metronomeToggle.setLookAndFeel(nullptr);
    midiInsertToggle.setLookAndFeel(nullptr);
    tempoSlider.setLookAndFeel(nullptr);
    timeSignatureNumeratorBox.setLookAndFeel(nullptr);
    timeSignatureDenominatorBox.setLookAndFeel(nullptr);
    patternBarsBox.setLookAndFeel(nullptr);
    keyQuantizeBox.setLookAndFeel(nullptr);
    arrangementSnapBox.setLookAndFeel(nullptr);
    arrangementZoomSlider.setLookAndFeel(nullptr);
    arrangementLaneHeightSlider.setLookAndFeel(nullptr);
    closeAllRackEditorSessions();
    nativeVstHost.setEditorStateCallback({});
    appRuntimeProfiler().dump("native_app_profile");
}

void StudioShellComponent::restorePersistedThemeSelection()
{
    if (windowStateSettings == nullptr)
        return;

    currentThemeIndex = juce::jlimit(0,
                                     static_cast<int>(availableThemeSpecs().size()) - 1,
                                     windowStateSettings->getIntValue("ui_theme", currentThemeIndex));

    if (auto* lookAndFeel = dynamic_cast<juce::LookAndFeel_V4*>(&juce::LookAndFeel::getDefaultLookAndFeel()))
        lookAndFeel->setColourScheme(themeSpecForIndex(currentThemeIndex).scheme);

    const auto& theme = themeSpecForIndex(currentThemeIndex);
    setTrackColourTheme(theme.buttonOn, theme.surface, theme.outline);
}

void StudioShellComponent::restorePersistedFontSelection()
{
    if (windowStateSettings == nullptr)
        return;

    if (availableUiFonts.isEmpty())
        availableUiFonts = buildUiFontChoices();

    const auto savedFontName = windowStateSettings->getValue("ui_font_name", getFontName(currentFontIndex));
    int resolvedIndex = 0;
    for (int index = 0; index < availableUiFonts.size(); ++index)
    {
        if (!availableUiFonts[index].equalsIgnoreCase(savedFontName))
            continue;

        resolvedIndex = index;
        break;
    }

    currentFontIndex = resolvedIndex;
    currentFontSizeIndex = resolveUiFontSizeIndex(static_cast<float>(windowStateSettings->getDoubleValue("ui_font_scale",
                                                                                                          uiFontSizeOptions()[static_cast<size_t>(currentFontSizeIndex)].scale)));
    applyUiFont();
}

void StudioShellComponent::persistThemeSelection() const
{
    if (windowStateSettings == nullptr)
        return;

    windowStateSettings->setValue("ui_theme", currentThemeIndex);
    windowStateSettings->saveIfNeeded();
}

void StudioShellComponent::persistFontSelection() const
{
    if (windowStateSettings == nullptr)
        return;

    windowStateSettings->setValue("ui_font_name", getFontName(currentFontIndex));
    windowStateSettings->setValue("ui_font_scale", uiFontSizeOptions()[static_cast<size_t>(currentFontSizeIndex)].scale);
    windowStateSettings->saveIfNeeded();
}

juce::String StudioShellComponent::currentDefaultTemplateIdentifier() const
{
    if (windowStateSettings == nullptr)
        return kBuiltInDefaultTemplateId;

    const auto identifier = windowStateSettings->getValue("project_default_template_id",
                                                          juce::String(kBuiltInDefaultTemplateId)).trim();
    return identifier.isNotEmpty() ? identifier : juce::String(kBuiltInDefaultTemplateId);
}

void StudioShellComponent::restorePersistedSessionState()
{
    if (windowStateSettings == nullptr)
        return;

    arrangementZoomPixelsPerBar = juce::jlimit(kArrangementMinPixelsPerBar,
                                               kArrangementMaxPixelsPerBar,
                                               static_cast<float>(windowStateSettings->getDoubleValue("session_arrangement_zoom_pixels_per_bar",
                                                                                                      arrangementZoomPixelsPerBar)));
    arrangementLaneHeightPixels = juce::jlimit(22.0f,
                                               80.0f,
                                               static_cast<float>(windowStateSettings->getDoubleValue("session_arrangement_lane_height_pixels",
                                                                                                      arrangementLaneHeightPixels)));
    pianoRollZoomPixelsPerBeat = juce::jlimit(12.0f,
                                              96.0f,
                                              static_cast<float>(windowStateSettings->getDoubleValue("session_piano_roll_zoom_pixels_per_beat",
                                                                                                     pianoRollZoomPixelsPerBeat)));
    pianoRollRowHeightPixels = juce::jlimit(8.0f,
                                            32.0f,
                                            static_cast<float>(windowStateSettings->getDoubleValue("session_piano_roll_row_height_pixels",
                                                                                                   pianoRollRowHeightPixels)));
    leftPaneWidthPixels = juce::jmax(0, windowStateSettings->getIntValue("session_left_pane_width_pixels", leftPaneWidthPixels));
    leftPaneTrackListHeightPixels = juce::jmax(0,
                                               windowStateSettings->getIntValue("session_left_pane_track_list_height_pixels",
                                                                                leftPaneTrackListHeightPixels));
    midiInsertEnabled = windowStateSettings->getBoolValue("session_midi_insert_enabled", midiInsertEnabled);
    transportRecordEnabled = windowStateSettings->getBoolValue("session_transport_record_enabled", transportRecordEnabled);
    preferredMidiInputIdentifier = windowStateSettings->getValue("session_midi_input_identifier",
                                                                 preferredMidiInputIdentifier).trim();

    ProjectFileData restored;
    const auto sessionFile = nativeSessionStateFile();
    if (sessionFile.existsAsFile() && loadProjectFile(sessionFile, restored).wasOk())
    {
        documentState = std::move(restored);
        const auto sessionProjectPath = windowStateSettings->getValue("session_project_file");
        currentProjectFile = sessionProjectPath.trim().isNotEmpty() ? juce::File(sessionProjectPath) : juce::File();
        dirty = windowStateSettings->getBoolValue("session_dirty", dirty);
    }
    else
    {
        documentState = makeDefaultProjectFile();
        const auto defaultTemplateId = currentDefaultTemplateIdentifier();
        if (defaultTemplateId != kBuiltInDefaultTemplateId)
        {
            ProjectFileData templateProject;
            if (loadProjectFile(juce::File(defaultTemplateId), templateProject).wasOk())
                documentState = std::move(templateProject);
            else if (windowStateSettings != nullptr)
            {
                windowStateSettings->setValue("project_default_template_id", kBuiltInDefaultTemplateId);
                windowStateSettings->saveIfNeeded();
            }
        }

        currentProjectFile = juce::File();
        dirty = false;
    }

    normaliseProject(documentState.project);
}

void StudioShellComponent::persistSessionState() const
{
    if (windowStateSettings == nullptr)
        return;

    windowStateSettings->setValue("session_project_file", currentProjectFile.getFullPathName());
    windowStateSettings->setValue("session_dirty", dirty);
    windowStateSettings->setValue("session_arrangement_zoom_pixels_per_bar", arrangementZoomPixelsPerBar);
    windowStateSettings->setValue("session_arrangement_lane_height_pixels", arrangementLaneHeightPixels);
    windowStateSettings->setValue("session_piano_roll_zoom_pixels_per_beat", pianoRollZoomPixelsPerBeat);
    windowStateSettings->setValue("session_piano_roll_row_height_pixels", pianoRollRowHeightPixels);
    windowStateSettings->setValue("session_left_pane_width_pixels", leftPaneWidthPixels);
    windowStateSettings->setValue("session_left_pane_track_list_height_pixels", leftPaneTrackListHeightPixels);
    windowStateSettings->setValue("session_midi_insert_enabled", midiInsertEnabled);
    windowStateSettings->setValue("session_transport_record_enabled", transportRecordEnabled);
    windowStateSettings->setValue("session_midi_input_identifier", preferredMidiInputIdentifier);

    const auto sessionFile = nativeSessionStateFile();
    ignoreUnused(saveProjectFile(sessionFile, documentState));
    windowStateSettings->saveIfNeeded();
}

void StudioShellComponent::refreshPlaybackToggleButton()
{
    const bool playbackRunning = rackPreviewRunning || projectPreviewRunning;
    playProjectButton.setButtonText(playbackRunning ? "Stop" : "Play");
    playProjectButton.setTooltip(playbackRunning ? "Stop Playback" : "Play Project");
    playProjectButton.setToggleState(playbackRunning, juce::dontSendNotification);
    playProjectButton.setEnabled(playbackRunning || !documentState.project.tracks.empty() || documentState.project.metronomeEnabled);
    recordToggle.setToggleState(transportRecordEnabled, juce::dontSendNotification);
    recordToggle.setEnabled(!documentState.project.tracks.empty());
    recordToggle.setTooltip(projectPreviewRunning && transportRecordEnabled
                                ? "Recording incoming MIDI to the selected track."
                                : "Record incoming MIDI to the selected track during project playback.");
}

int StudioShellComponent::getCurrentThemeIndex() const noexcept
{
    return currentThemeIndex;
}

int StudioShellComponent::getThemeCount() const noexcept
{
    return static_cast<int>(availableThemeSpecs().size());
}

juce::String StudioShellComponent::getThemeName(int index) const
{
    return themeSpecForIndex(index).name;
}

int StudioShellComponent::getCurrentFontIndex() const noexcept
{
    return currentFontIndex;
}

int StudioShellComponent::getFontCount() const noexcept
{
    return availableUiFonts.size();
}

juce::String StudioShellComponent::getFontName(int index) const
{
    if (!juce::isPositiveAndBelow(index, availableUiFonts.size()))
        return "Default System";

    return availableUiFonts[index];
}

int StudioShellComponent::getCurrentFontSizeIndex() const noexcept
{
    return currentFontSizeIndex;
}

int StudioShellComponent::getFontSizeCount() const noexcept
{
    return static_cast<int>(uiFontSizeOptions().size());
}

juce::String StudioShellComponent::getFontSizeLabel(int index) const
{
    if (!juce::isPositiveAndBelow(index, getFontSizeCount()))
        return uiFontSizeOptions()[2].label;

    return uiFontSizeOptions()[static_cast<size_t>(index)].label;
}

void StudioShellComponent::setThemeIndex(int index)
{
    const auto clamped = juce::jlimit(0, getThemeCount() - 1, index);
    if (currentThemeIndex == clamped)
        return;

    currentThemeIndex = clamped;
    applyTheme();
    appendActivityLog("Theme", "Switched to " + getThemeName(currentThemeIndex) + ".");
}

void StudioShellComponent::setFontIndex(int index)
{
    if (availableUiFonts.isEmpty())
        availableUiFonts = buildUiFontChoices();

    const auto clamped = juce::jlimit(0, juce::jmax(0, getFontCount() - 1), index);
    if (currentFontIndex == clamped)
        return;

    currentFontIndex = clamped;
    applyUiFont();
    appendActivityLog("Font", "Switched to " + getFontName(currentFontIndex) + ".");
}

void StudioShellComponent::setFontSizeIndex(int index)
{
    const auto clamped = juce::jlimit(0, getFontSizeCount() - 1, index);
    if (currentFontSizeIndex == clamped)
        return;

    currentFontSizeIndex = clamped;
    applyUiFont();
    appendActivityLog("Font", "Set UI text size to " + getFontSizeLabel(currentFontSizeIndex) + ".");
}

void StudioShellComponent::applyUiFont()
{
    if (availableUiFonts.isEmpty())
        availableUiFonts = buildUiFontChoices();

    const auto previousScale = ui::fontScale();
    const auto nextScale = uiFontSizeOptions()[static_cast<size_t>(juce::jlimit(0,
                                                                                getFontSizeCount() - 1,
                                                                                currentFontSizeIndex))].scale;
    ui::setFontScale(nextScale);
    const auto scaleRatio = previousScale > 0.0f ? nextScale / previousScale : 1.0f;

    const auto selectedFont = getFontName(currentFontIndex);
    const auto resolvedFont = selectedFont.equalsIgnoreCase("Default System") ? juce::String() : selectedFont;

    if (auto* lookAndFeel = dynamic_cast<juce::LookAndFeel_V4*>(&juce::LookAndFeel::getDefaultLookAndFeel()))
        lookAndFeel->setDefaultSansSerifTypefaceName(resolvedFont);

    if (compactHeaderLookAndFeel != nullptr)
        compactHeaderLookAndFeel->setDefaultSansSerifTypefaceName(resolvedFont);

    sendLookAndFeelChange();
    refreshExplicitUiFontsInComponentTree(*this, scaleRatio);

    const auto refreshFloatingWindowFonts = [scaleRatio] (FloatingPanelWindow* window)
    {
        if (window == nullptr)
            return;

        window->sendLookAndFeelChange();
        if (auto* content = window->getContentComponent())
        {
            content->sendLookAndFeelChange();
            refreshExplicitUiFontsInComponentTree(*content, scaleRatio);
        }
        window->repaint();
    };

    refreshFloatingWindowFonts(transportWindow.get());
    refreshFloatingWindowFonts(mixerWindow.get());
    refreshFloatingWindowFonts(audioWindow.get());
    refreshFloatingWindowFonts(panelsWindow.get());
    refreshFloatingWindowFonts(tracksWindow.get());
    refreshFloatingWindowFonts(modulationMatrixWindow.get());
    refreshFloatingWindowFonts(rackBrowserWindow.get());
    refreshFloatingWindowFonts(renderManagerWindow.get());
    refreshFloatingWindowFonts(arrangementWindow.get());
    refreshFloatingWindowFonts(automationWindow.get());
    refreshFloatingWindowFonts(samplesWindow.get());
    refreshFloatingWindowFonts(pianoRollWindow.get());
    refreshFloatingWindowFonts(virtualPianoWindow.get());
    refreshFloatingWindowFonts(activityLogWindow.get());
    refreshFloatingWindowFonts(audioSettingsWindow.get());
    refreshFloatingWindowFonts(vstFolderManagerWindow.get());

    trackTable.repaint();
    if (arrangementOverview != nullptr)
        arrangementOverview->repaint();
    if (automationEditor != nullptr)
        automationEditor->repaint();
    if (sampleTimeline != nullptr)
        sampleTimeline->repaint();
    if (pianoRoll != nullptr)
        pianoRoll->repaint();
    if (mixerComponent != nullptr)
        mixerComponent->refreshFromModel();
    if (floatingMixerComponent != nullptr)
        floatingMixerComponent->refreshFromModel();
    repaint();

    persistFontSelection();
}

void StudioShellComponent::applyTheme()
{
    const auto& theme = themeSpecForIndex(currentThemeIndex);

    if (auto* lookAndFeel = dynamic_cast<juce::LookAndFeel_V4*>(&juce::LookAndFeel::getDefaultLookAndFeel()))
        lookAndFeel->setColourScheme(theme.scheme);

    setTrackColourTheme(theme.buttonOn, theme.surface, theme.outline);

    applyThemeToComponentTree(*this, theme);

    if (headerTimecodeDisplay != nullptr)
        headerTimecodeDisplay->setTheme(theme);

    fileLabel.setColour(juce::Label::textColourId, theme.secondaryText);
    statsLabel.setColour(juce::Label::textColourId, theme.primaryText.withAlpha(0.92f));
    statusLabel.setColour(juce::Label::textColourId, theme.successText);
    inspectorLabel.setColour(juce::Label::textColourId, theme.primaryText);
    mixerLabel.setColour(juce::Label::textColourId, theme.primaryText);
    samplesLabel.setColour(juce::Label::textColourId, theme.primaryText);
    sampleLibraryLabel.setColour(juce::Label::textColourId, theme.secondaryText);
    sampleTimelineLabel.setColour(juce::Label::textColourId, theme.secondaryText);
    automationLabel.setColour(juce::Label::textColourId, theme.primaryText);
    pianoRollLabel.setColour(juce::Label::textColourId, theme.primaryText);
    arrangementLabel.setColour(juce::Label::textColourId, theme.primaryText);
    tempoLabel.setColour(juce::Label::textColourId, theme.primaryText);
    patternBarsLabel.setColour(juce::Label::textColourId, theme.primaryText);
    keyQuantizeLabel.setColour(juce::Label::textColourId, theme.primaryText);
    arrangementSnapLabel.setColour(juce::Label::textColourId, theme.primaryText);
    arrangementZoomLabel.setColour(juce::Label::textColourId, theme.primaryText);
    arrangementLaneHeightLabel.setColour(juce::Label::textColourId, theme.primaryText);

    trackTable.setColour(juce::ListBox::backgroundColourId, theme.surface);
    trackTable.setColour(juce::ListBox::outlineColourId, theme.outline);
    trackTable.getHeader().setColour(juce::TableHeaderComponent::backgroundColourId, theme.surfaceAlt);
    trackTable.getHeader().setColour(juce::TableHeaderComponent::outlineColourId, theme.outline);
    trackTable.getHeader().setColour(juce::TableHeaderComponent::textColourId, theme.primaryText);
    sampleAssetList.setColour(juce::ListBox::backgroundColourId, theme.surface);
    sampleAssetList.setColour(juce::ListBox::outlineColourId, theme.outline);
    inspectorEditor.setTextToShowWhenEmpty("Track details", theme.editorPlaceholder);

    const auto styleFloatingWindow = [&theme] (FloatingPanelWindow* window)
    {
        if (window == nullptr)
            return;

        window->setColour(juce::ResizableWindow::backgroundColourId, theme.mainBackground);
        if (auto* content = window->getContentComponent())
            applyThemeToComponentTree(*content, theme);
        window->repaint();
    };

    styleFloatingWindow(transportWindow.get());
    styleFloatingWindow(mixerWindow.get());
    styleFloatingWindow(audioWindow.get());
    styleFloatingWindow(panelsWindow.get());
    styleFloatingWindow(tracksWindow.get());
    styleFloatingWindow(modulationMatrixWindow.get());
    styleFloatingWindow(rackBrowserWindow.get());
    styleFloatingWindow(renderManagerWindow.get());
    styleFloatingWindow(arrangementWindow.get());
    styleFloatingWindow(automationWindow.get());
    styleFloatingWindow(samplesWindow.get());
    styleFloatingWindow(pianoRollWindow.get());
    styleFloatingWindow(virtualPianoWindow.get());
    styleFloatingWindow(activityLogWindow.get());
    styleFloatingWindow(audioSettingsWindow.get());
    styleFloatingWindow(vstFolderManagerWindow.get());

    trackTable.repaint();
    if (arrangementOverview != nullptr)
        arrangementOverview->repaint();
    if (automationEditor != nullptr)
        automationEditor->repaint();
    if (sampleTimeline != nullptr)
        sampleTimeline->repaint();
    if (pianoRoll != nullptr)
        pianoRoll->repaint();
    if (mixerComponent != nullptr)
        mixerComponent->refreshFromModel();
    if (floatingMixerComponent != nullptr)
        floatingMixerComponent->refreshFromModel();

    updateAiStatusSummary();
    repaint();
    if (auto* topLevel = getTopLevelComponent())
        topLevel->repaint();

    persistThemeSelection();
}

void StudioShellComponent::createToolbarButton(juce::TextButton& button, const juce::String& text)
{
    const auto& theme = themeSpecForIndex(currentThemeIndex);
    button.setButtonText(text);
    button.setColour(juce::TextButton::buttonColourId, theme.buttonOff);
    button.setColour(juce::TextButton::buttonOnColourId, theme.buttonOn);
    button.setColour(juce::TextButton::textColourOffId, theme.buttonText);
    button.setColour(juce::TextButton::textColourOnId, theme.buttonText);
    addAndMakeVisible(button);
}

void StudioShellComponent::configureInspectorTextEditor(juce::TextEditor& editor, const juce::String& placeholder)
{
    const auto& theme = themeSpecForIndex(currentThemeIndex);
    editor.setMultiLine(false);
    editor.setScrollbarsShown(false);
    editor.setTextToShowWhenEmpty(placeholder, theme.editorPlaceholder);
    editor.setColour(juce::TextEditor::backgroundColourId, theme.surface);
    editor.setColour(juce::TextEditor::textColourId, theme.primaryText);
    editor.setColour(juce::TextEditor::outlineColourId, theme.outline);
    editor.setColour(juce::CaretComponent::caretColourId, theme.primaryText);
    editor.setFont(ui::font());
    addAndMakeVisible(editor);
}

void StudioShellComponent::configureInspectorSlider(juce::Slider& slider,
                                                    double minimum,
                                                    double maximum,
                                                    double step,
                                                    const juce::String& suffix)
{
    slider.setSliderStyle(juce::Slider::LinearHorizontal);
    slider.setRange(minimum, maximum, step);
    slider.setChangeNotificationOnlyOnRelease(true);
    slider.setTextBoxStyle(juce::Slider::TextBoxRight, false, 60, 22);
    slider.setTextValueSuffix(suffix);
    addAndMakeVisible(slider);
}

void StudioShellComponent::paint(juce::Graphics& g)
{
    const auto& theme = themeSpecForIndex(currentThemeIndex);
    g.fillAll(theme.mainBackground);

    juce::ColourGradient background(theme.headerStart,
                                    0.0f,
                                    0.0f,
                                    theme.headerEnd,
                                    0.0f,
                                    460.0f,
                                    false);
    background.addColour(0.45, theme.headerMid);
    g.setGradientFill(background);
    g.fillRect(getLocalBounds());

    if (!leftPaneResizeHandleBounds.isEmpty())
    {
        auto handle = leftPaneResizeHandleBounds.toFloat();
        g.setColour(theme.outline.withAlpha(leftPaneResizeDragging ? 0.7f : 0.35f));
        g.fillRoundedRectangle(handle.reduced(handle.getWidth() * 0.35f, 12.0f), 2.0f);
    }

    if (!leftPaneTrackListResizeHandleBounds.isEmpty())
    {
        auto handle = leftPaneTrackListResizeHandleBounds.toFloat();
        g.setColour(theme.outline.withAlpha(leftPaneTrackListResizeDragging ? 0.7f : 0.35f));
        g.fillRoundedRectangle(handle.reduced(18.0f, handle.getHeight() * 0.35f), 2.0f);
    }
}

void StudioShellComponent::resized()
{
    auto bounds = getLocalBounds();
    auto headerArea = bounds.removeFromTop(68);
    headerLogo.setBounds(8, 5, 50, 50);

    auto headerContent = headerArea.reduced(10, 4);
    headerContent.removeFromLeft(56);
    headerContent.removeFromLeft(8);

    auto lcdArea = headerContent.removeFromRight(juce::jlimit(152, 196, headerContent.getWidth() / 7));
    headerContent.removeFromRight(8);
    if (headerTimecodeDisplay != nullptr)
        headerTimecodeDisplay->setBounds(lcdArea.reduced(0, 1));

    auto fileArea = headerContent.removeFromTop(12);
    auto fileTextArea = fileArea;
    aiStatusSummaryLabel.setBounds(fileTextArea.removeFromRight(juce::jlimit(140, 280, fileArea.getWidth() / 5)));
    fileTextArea.removeFromRight(6);
    fileLabel.setBounds(fileTextArea);

    headerContent.removeFromTop(2);
    auto toolbar1 = headerContent.removeFromTop(20);
    playProjectButton.setBounds(toolbar1.removeFromLeft(64));
    toolbar1.removeFromLeft(3);
    recordToggle.setBounds(toolbar1.removeFromLeft(46));
    toolbar1.removeFromLeft(3);
    aiComposeButton.setBounds(toolbar1.removeFromLeft(76));
    toolbar1.removeFromLeft(6);
    aceStepGenerateButton.setBounds(toolbar1.removeFromLeft(112));
    toolbar1.removeFromLeft(6);
    loopToggle.setBounds(toolbar1.removeFromLeft(48));
    toolbar1.removeFromLeft(4);
    metronomeToggle.setBounds(toolbar1.removeFromLeft(52));
    toolbar1.removeFromLeft(4);
    midiInsertToggle.setBounds(toolbar1.removeFromLeft(60));

    headerContent.removeFromTop(2);
    auto toolbar2 = headerContent.removeFromTop(20);
    tempoLabel.setBounds(toolbar2.removeFromLeft(36));
    tempoSlider.setBounds(toolbar2.removeFromLeft(118));
    toolbar2.removeFromLeft(4);
    timeSignatureLabel.setBounds(toolbar2.removeFromLeft(42));
    timeSignatureNumeratorBox.setBounds(toolbar2.removeFromLeft(36));
    timeSignatureSlashLabel.setBounds(toolbar2.removeFromLeft(10));
    timeSignatureDenominatorBox.setBounds(toolbar2.removeFromLeft(36));
    toolbar2.removeFromLeft(4);
    patternBarsLabel.setBounds(toolbar2.removeFromLeft(64));
    patternBarsBox.setBounds(toolbar2.removeFromLeft(84));
    toolbar2.removeFromLeft(4);
    keyQuantizeLabel.setBounds(toolbar2.removeFromLeft(66));
    keyQuantizeBox.setBounds(toolbar2.removeFromLeft(138));
    toolbar2.removeFromLeft(4);
    arrangementSnapLabel.setBounds(toolbar2.removeFromLeft(50));
    arrangementSnapBox.setBounds(toolbar2.removeFromLeft(82));
    toolbar2.removeFromLeft(4);
    arrangementZoomLabel.setBounds(toolbar2.removeFromLeft(50));
    arrangementZoomSlider.setBounds(toolbar2.removeFromLeft(102));
    toolbar2.removeFromLeft(4);
    arrangementLaneHeightLabel.setBounds(toolbar2.removeFromLeft(46));
    arrangementLaneHeightSlider.setBounds(toolbar2.removeFromLeft(96));

    newButton.setBounds({});
    openButton.setBounds({});
    saveButton.setBounds({});
    saveAsButton.setBounds({});
    importMidiButton.setBounds({});
    exportWavButton.setBounds({});
    exportMidiButton.setBounds({});
    aiSettingsButton.setBounds({});
    undoButton.setBounds({});
    redoButton.setBounds({});

    auto area = bounds.reduced(12, 0);
    area.removeFromTop(2);
    pianoRollZoomLabel.setBounds({});
    pianoRollZoomSlider.setBounds({});
    pianoRollRowHeightLabel.setBounds({});
    pianoRollRowHeightSlider.setBounds({});

    statsLabel.setBounds({});
    statusLabel.setBounds(area.removeFromTop(16));
    area.removeFromTop(5);

    auto workspaceArea = area;
    const auto minLeftWidth = 300;
    const auto maxLeftWidth = juce::jmax(minLeftWidth, workspaceArea.getWidth() - 420);
    if (leftPaneWidthPixels <= 0)
        leftPaneWidthPixels = juce::jlimit(minLeftWidth,
                                           juce::jmax(minLeftWidth, 520),
                                           juce::roundToInt(static_cast<float>(workspaceArea.getWidth()) * 0.34f));
    leftPaneWidthPixels = juce::jlimit(minLeftWidth, maxLeftWidth, leftPaneWidthPixels);

    auto leftColumn = workspaceArea.removeFromLeft(leftPaneWidthPixels);
    leftPaneResizeHandleBounds = workspaceArea.removeFromLeft(14);
    auto sequenceArea = workspaceArea;

    const auto leftHeight = leftColumn.getHeight();
    const auto trackCount = juce::jmax(1, static_cast<int>(documentState.project.tracks.size()));
    const auto resizeHandleHeight = 12;
    const auto minimumInspectorAreaHeight = 170;
    const auto minimumTrackTableHeight = trackTable.getHeaderHeight()
                                         + (juce::jmax(2, juce::jmin(trackCount, 2)) * trackTable.getRowHeight())
                                         + 2;
    const auto desiredTrackTableHeight = trackTable.getHeaderHeight() + (trackCount * trackTable.getRowHeight()) + 2;
    const auto maximumTrackTableHeight = juce::jmax(minimumTrackTableHeight,
                                                    leftHeight - minimumInspectorAreaHeight - resizeHandleHeight - 10);
    if (leftPaneTrackListHeightPixels <= 0)
        leftPaneTrackListHeightPixels = juce::jmin(desiredTrackTableHeight, maximumTrackTableHeight);
    leftPaneTrackListHeightPixels = juce::jlimit(minimumTrackTableHeight,
                                                 maximumTrackTableHeight,
                                                 leftPaneTrackListHeightPixels);

    trackTable.setBounds(leftColumn.removeFromTop(leftPaneTrackListHeightPixels));
    leftColumn.removeFromTop(6);
    leftPaneTrackListResizeHandleBounds = leftColumn.removeFromTop(resizeHandleHeight);
    leftColumn.removeFromTop(6);
    inspectorViewport.setBounds({});
    inspectorContent.setSize(0, 0);

    const auto controlHeight = 22;
    const auto smallGap = 4;
    const auto sectionGap = 6;
    auto inspectorArea = leftColumn;

    inspectorLabel.setBounds(inspectorArea.removeFromTop(20));
    inspectorArea.removeFromTop(smallGap);

    trackNameEditor.setBounds(inspectorArea.removeFromTop(controlHeight));
    inspectorArea.removeFromTop(smallGap);

    auto typeRow = inspectorArea.removeFromTop(controlHeight);
    auto typeWidth = juce::roundToInt(static_cast<float>(typeRow.getWidth()) * 0.44f);
    trackTypeEditor.setBounds(typeRow.removeFromLeft(typeWidth));
    typeRow.removeFromLeft(smallGap);
    instrumentModeEditor.setBounds(typeRow);
    inspectorArea.removeFromTop(smallGap);

    instrumentEditor.setBounds(inspectorArea.removeFromTop(controlHeight));
    inspectorArea.removeFromTop(smallGap);
    rackVstEditor.setBounds(inspectorArea.removeFromTop(controlHeight));
    inspectorArea.removeFromTop(smallGap);

    auto midiRow = inspectorArea.removeFromTop(controlHeight);
    auto halfWidth = juce::roundToInt(static_cast<float>(midiRow.getWidth()) * 0.5f) - (smallGap / 2);
    midiChannelSlider.setBounds(midiRow.removeFromLeft(halfWidth));
    midiRow.removeFromLeft(smallGap);
    midiProgramSlider.setBounds(midiRow);
    inspectorArea.removeFromTop(smallGap);

    auto mixRow = inspectorArea.removeFromTop(controlHeight);
    volumeSlider.setBounds(mixRow.removeFromLeft(halfWidth));
    mixRow.removeFromLeft(smallGap);
    panSlider.setBounds(mixRow);
    inspectorArea.removeFromTop(smallGap);

    auto toggleRow = inspectorArea.removeFromTop(controlHeight);
    muteToggle.setBounds(toggleRow.removeFromLeft(56));
    toggleRow.removeFromLeft(6);
    soloToggle.setBounds(toggleRow.removeFromLeft(56));
    toggleRow.removeFromLeft(6);
    liveArmToggle.setBounds(toggleRow.removeFromLeft(64));
    inspectorArea.removeFromTop(sectionGap);

    inspectorEditor.setBounds(inspectorArea);

    arrangementLabel.setBounds({});
    arrangementViewport.setBounds(sequenceArea);

    pianoRollLabel.setBounds({});
    pianoRollViewport.setBounds({});
    mixerLabel.setBounds({});
    mixerViewport.setBounds({});
    samplesLabel.setBounds({});
    sampleLibraryLabel.setBounds({});
    sampleTimelineLabel.setBounds({});
    sampleAssetList.setBounds({});
    sampleTimelineViewport.setBounds({});
    importSampleButton.setBounds({});
    placeSampleButton.setBounds({});
    automationLabel.setBounds({});
    if (automationEditor != nullptr)
        automationEditor->setBounds({});
}

bool StudioShellComponent::keyPressed(const juce::KeyPress& key)
{
    if (key.getKeyCode() == juce::KeyPress::F11Key && !key.getModifiers().isAnyModifierKeyDown())
    {
        if (auto* window = dynamic_cast<MainWindow*>(getTopLevelComponent()))
        {
            window->toggleBorderlessFullscreen();
            return true;
        }
    }

    if (key.getKeyCode() == juce::KeyPress::F2Key)
    {
        setTransportWindowVisible(!isTransportWindowVisible());
        return true;
    }

    if (key.getKeyCode() == juce::KeyPress::spaceKey && !key.getModifiers().isAnyModifierKeyDown())
    {
        if (rackPreviewRunning || projectPreviewRunning)
            stopRackPreview();
        else
            playFullProjectThroughNativeEngine();
        return true;
    }

    if (tryHandleVirtualPianoShortcut(key))
        return true;

    if (matchesCommandShortcut(key, 'a'))
    {
        if (floatingArrangementOverview != nullptr
            && arrangementWindow != nullptr
            && arrangementWindow->isVisible()
            && floatingArrangementOverview->hasKeyboardFocus(true))
        {
            floatingArrangementOverview->selectAllSections();
            return true;
        }

        if (panelsWindowContent != nullptr
            && panelsWindow != nullptr
            && panelsWindow->isVisible()
            && panelsWindowContent->hasArrangementKeyboardFocus())
        {
            panelsWindowContent->selectAllSections();
            return true;
        }

        if (arrangementOverview != nullptr && arrangementOverview->hasKeyboardFocus(true))
        {
            arrangementOverview->selectAllSections();
            return true;
        }

        if (floatingPianoRollWorkspace != nullptr
            && pianoRollWindow != nullptr
            && pianoRollWindow->isVisible()
            && floatingPianoRollWorkspace->hasKeyboardFocus(true))
        {
            floatingPianoRollWorkspace->selectAllNotes();
            return true;
        }

        if (panelsWindowContent != nullptr
            && panelsWindow != nullptr
            && panelsWindow->isVisible()
            && panelsWindowContent->hasPianoRollKeyboardFocus())
        {
            panelsWindowContent->selectAllNotes();
            return true;
        }

        if (pianoRoll != nullptr && pianoRoll->hasKeyboardFocus(true))
        {
            pianoRoll->selectAllNotes();
            return true;
        }

        return selectAllFromFocusedEditor(nullptr);
    }

    if (matchesCommandShortcut(key, 'c'))
    {
        if (pianoRoll != nullptr && pianoRoll->copySelected())
            return true;
    }

    if (matchesCommandShortcut(key, 'x'))
    {
        if (pianoRoll != nullptr && pianoRoll->cutSelected())
            return true;
    }

    if (matchesCommandShortcut(key, 'v'))
    {
        if (pianoRoll != nullptr && pianoRoll->pasteClipboard())
            return true;
    }

    if (matchesCommandShortcut(key, 'n'))
    {
        createNewProject();
        return true;
    }

    if (matchesCommandShortcut(key, 'o') && key.getModifiers().isShiftDown())
    {
        promptImportMidi();
        return true;
    }

    if (matchesCommandShortcut(key, 'o'))
    {
        promptOpenProject();
        return true;
    }

    if (matchesCommandShortcut(key, 's') && key.getModifiers().isShiftDown())
    {
        saveProjectAs();
        return true;
    }

    if (matchesCommandShortcut(key, 's'))
    {
        saveProject();
        return true;
    }

    if (matchesCommandShortcut(key, 'g'))
    {
        composeWithAi();
        return true;
    }

    if (key.getModifiers().isCommandDown() && key.getKeyCode() == ',')
    {
        showAiSettingsDialog();
        return true;
    }

    if (matchesCommandShortcut(key, 'e'))
    {
        if (key.getModifiers().isShiftDown())
            promptExportMidi();
        else
            promptExportWav();
        return true;
    }

    if (matchesCommandShortcut(key, 'z') && !key.getModifiers().isShiftDown())
    {
        undo();
        return true;
    }

    if ((matchesCommandShortcut(key, 'z') && key.getModifiers().isShiftDown())
        || matchesCommandShortcut(key, 'y'))
    {
        redo();
        return true;
    }

    if (matchesCommandShortcut(key, 'q'))
    {
        quantizeSelectedNotes();
        return true;
    }

    return false;
}

void StudioShellComponent::mouseMove(const juce::MouseEvent& event)
{
    if (leftPaneTrackListResizeHandleBounds.contains(event.getPosition()))
        setMouseCursor(juce::MouseCursor::UpDownResizeCursor);
    else if (leftPaneResizeHandleBounds.contains(event.getPosition()))
        setMouseCursor(juce::MouseCursor::LeftRightResizeCursor);
    else if (!leftPaneResizeDragging && !leftPaneTrackListResizeDragging)
        setMouseCursor(juce::MouseCursor::NormalCursor);
}

void StudioShellComponent::mouseExit(const juce::MouseEvent&)
{
    if (!leftPaneResizeDragging && !leftPaneTrackListResizeDragging)
        setMouseCursor(juce::MouseCursor::NormalCursor);
}

void StudioShellComponent::mouseDown(const juce::MouseEvent& event)
{
    if (leftPaneTrackListResizeHandleBounds.contains(event.getPosition()))
    {
        leftPaneTrackListResizeDragging = true;
        leftPaneTrackListDragStartHeight = leftPaneTrackListHeightPixels;
        setMouseCursor(juce::MouseCursor::UpDownResizeCursor);
        return;
    }

    if (!leftPaneResizeHandleBounds.contains(event.getPosition()))
        return;

    leftPaneResizeDragging = true;
    leftPaneDragStartWidth = leftPaneWidthPixels;
    setMouseCursor(juce::MouseCursor::LeftRightResizeCursor);
}

void StudioShellComponent::mouseDrag(const juce::MouseEvent& event)
{
    if (leftPaneTrackListResizeDragging)
    {
        leftPaneTrackListHeightPixels = leftPaneTrackListDragStartHeight + event.getDistanceFromDragStartY();
        resized();
        repaint(leftPaneTrackListResizeHandleBounds.expanded(2));
        return;
    }

    if (!leftPaneResizeDragging)
        return;

    leftPaneWidthPixels = leftPaneDragStartWidth + event.getDistanceFromDragStartX();
    resized();
    repaint(leftPaneResizeHandleBounds.expanded(2));
}

void StudioShellComponent::mouseUp(const juce::MouseEvent& event)
{
    juce::ignoreUnused(event);
    leftPaneResizeDragging = false;
    leftPaneTrackListResizeDragging = false;
    if (leftPaneTrackListResizeHandleBounds.contains(getMouseXYRelative()))
        setMouseCursor(juce::MouseCursor::UpDownResizeCursor);
    else if (leftPaneResizeHandleBounds.contains(getMouseXYRelative()))
        setMouseCursor(juce::MouseCursor::LeftRightResizeCursor);
    else
        setMouseCursor(juce::MouseCursor::NormalCursor);
}

void StudioShellComponent::timerCallback()
{
    const ScopedAppProfileSample profileSample(AppProfileSection::timerCallback);
    const auto nowMs = juce::Time::getMillisecondCounter();
    dispatchPendingMidiInputMessages();
    pollAceStepServerOutput();
    pollAiComposeFuture();
    pollAceStepGenerationFuture();
    pollStemSeparationFuture();

    const bool hasSharedRackHost = nativeVstHost.isReady();
    const bool hasOpenRackEditors = hasOpenRackEditorSessions();
    const bool hasAnyOpenRackEditorWindow = loadedRackEditorOpen || hasOpenRackEditors;

    if (!hasSharedRackHost && !hasOpenRackEditors)
    {
        refreshFloatingWindows();
        return;
    }

    if (!rackPreviewRunning && !projectPreviewRunning && hasAnyOpenRackEditorWindow)
    {
        syncOpenRackEditorSessions(false);
        refreshFloatingWindows(false);
        return;
    }

    if (!rackPreviewRunning && !projectPreviewRunning && !hasOpenRackEditors)
    {
        if (hasSharedRackHost)
        {
            NativeVstHostSession::TransportSnapshot transportSnapshot;
            if (nativeVstHost.queryTransportSnapshot(transportSnapshot, false).wasOk())
            {
                transportCpuUsagePercent = juce::jmax(0.0, transportSnapshot.cpuUsage * 100.0);
                transportMasterPeakLeft = transportSnapshot.masterPeakLeft;
                transportMasterPeakRight = transportSnapshot.masterPeakRight;
            }
        }

        syncOpenRackEditorSessions(false);
        refreshFloatingWindows();
        return;
    }

    const bool hasVisibleFloatingPianoRoll = floatingPianoRollWorkspace != nullptr
        && pianoRollWindow != nullptr
        && pianoRollWindow->isVisible();
    const bool shouldThrottleForRackEditor = hasAnyOpenRackEditorWindow;
    const bool shouldThrottleForPianoRoll = hasVisibleFloatingPianoRoll;
    const bool hasVisibleTrackMeterConsumers = trackTable.isShowing()
        || (mixerComponent != nullptr && mixerComponent->isShowing())
        || (floatingMixerComponent != nullptr && mixerWindow != nullptr && mixerWindow->isVisible());
    const bool shouldRefreshTrackMetersThisTick = hasVisibleTrackMeterConsumers
        && (rackPreviewRunning || ((playbackUiTickCounter % 3) == 0));
    NativeVstHostSession::TransportSnapshot transportSnapshot;
    const auto snapshotResult = hasSharedRackHost
        ? nativeVstHost.queryTransportSnapshot(transportSnapshot, shouldRefreshTrackMetersThisTick)
        : juce::Result::fail("Native VST host session is not ready.");
    if (snapshotResult.failed())
    {
        syncOpenRackEditorSessions(true);
        refreshFloatingWindows(false);
        return;
    }

    ++playbackUiTickCounter;
    const auto heavyUiRefreshDivisor = shouldThrottleForRackEditor ? kPlaybackHeavyUiRefreshDivisorWithOpenEditor
        : (shouldThrottleForPianoRoll ? kPlaybackHeavyUiRefreshDivisorWithOpenPianoRoll : 1);
    const auto floatingRefreshDivisor = shouldThrottleForRackEditor ? kPlaybackFloatingRefreshDivisorWithOpenEditor
        : (shouldThrottleForPianoRoll ? kPlaybackFloatingRefreshDivisorWithOpenPianoRoll : 4);
    const bool shouldRefreshTimelineViewsThisTick = (playbackUiTickCounter % heavyUiRefreshDivisor) == 0;
    const bool shouldRefreshEditorsThisTick = (playbackUiTickCounter % floatingRefreshDivisor) == 0;
    const bool shouldRefreshFloatingPianoRollThisTick = hasVisibleFloatingPianoRoll
        && (playbackUiTickCounter % kPlaybackFloatingPianoRollRefreshDivisor) == 0;
    const bool shouldFlushDeferredParameterSyncThisTick = (projectPreviewRunning || rackPreviewRunning)
        && (nowMs - lastDeferredParameterFlushMs) >= kDeferredEngineParameterFlushIntervalMs;

    if (shouldFlushDeferredParameterSyncThisTick
        && juce::isPositiveAndBelow(pendingLiveRackParameterEngineSyncTrack, static_cast<int>(documentState.project.tracks.size())))
    {
        const auto deferredTrackIndex = pendingLiveRackParameterEngineSyncTrack;
        const auto& deferredTrack = documentState.project.tracks[static_cast<size_t>(deferredTrackIndex)];
        const auto parameterSyncResult = nativeVstHost.setAudioEngineTrackParameters(deferredTrackIndex, deferredTrack);
        lastDeferredParameterFlushMs = nowMs;
        if (parameterSyncResult.wasOk())
        {
            pendingLiveRackParameterEngineSyncTrack = -1;
            audioEngineStateValid = true;
            audioEngineStateDirty = false;
        }
        else
        {
            pendingLiveRackParameterEngineSyncTrack = -1;
            statusLabel.setText("Live modulation sync failed: " + parameterSyncResult.getErrorMessage(),
                                juce::dontSendNotification);
        }
    }

    const auto sampleRate = juce::jmax(1.0, transportSnapshot.sampleRate);
    const auto inprocessRunning = transportSnapshot.inprocessTransportRunning;
    const auto audioEngineRunning = transportSnapshot.audioEngineRunning || transportSnapshot.audioEngineTailing;
    transportCpuUsagePercent = juce::jmax(0.0, transportSnapshot.cpuUsage * 100.0);
    transportMasterPeakLeft = transportSnapshot.masterPeakLeft;
    transportMasterPeakRight = transportSnapshot.masterPeakRight;
    syncOpenRackEditorSessions(true);

    const auto positionFrame = (projectPreviewRunning || rackPreviewRunning)
        ? transportSnapshot.audioEnginePositionFrame
        : transportSnapshot.inprocessTransportPositionFrame;
    const auto stillRunning = (projectPreviewRunning || rackPreviewRunning) ? audioEngineRunning : inprocessRunning;

    bool trackMeterLevelsChanged = false;
    if (projectPreviewRunning || rackPreviewRunning)
    {
        if (shouldRefreshTrackMetersThisTick)
        {
            std::fill(trackMeterLevels.begin(), trackMeterLevels.end(), 0.0f);
            for (int trackIndex = 0; trackIndex < transportSnapshot.audioEngineTrackPeakLevels.size(); ++trackIndex)
            {
                if (!juce::isPositiveAndBelow(trackIndex, static_cast<int>(trackMeterLevels.size())))
                    continue;
                trackMeterLevels[static_cast<size_t>(trackIndex)] = transportSnapshot.audioEngineTrackPeakLevels.getReference(trackIndex);
            }
            trackMeterLevelsChanged = true;
        }
    }

    documentState.project.playheadTick = juce::jmax(0, secondsToTick(static_cast<double>(positionFrame) / sampleRate,
                                                                     documentState.project.bpm));
    documentState.project.recalculateTimeFields();
    if (headerTimecodeDisplay != nullptr)
    {
        headerTimecodeDisplay->setValues(documentState.project.playheadSec,
                                         projectSequenceLengthSeconds(documentState.project),
                                         documentState.project.leftLocatorSec,
                                         documentState.project.rightLocatorSec);
    }
    if (shouldRefreshTimelineViewsThisTick)
    {
        if (pianoRoll != nullptr && !pianoRollViewport.getBounds().isEmpty())
            pianoRoll->repaint();
        if (arrangementOverview != nullptr && !arrangementViewport.getBounds().isEmpty())
            arrangementOverview->repaint();
        if (sampleTimeline != nullptr && !sampleTimelineViewport.getBounds().isEmpty())
            sampleTimeline->repaint();
        if (automationEditor != nullptr && !automationEditor->getBounds().isEmpty())
            automationEditor->refreshViewState();
    }
    if (mixerComponent != nullptr)
        mixerComponent->refreshMeters();
    if (trackMeterLevelsChanged)
        repaintTrackVolumeMeters();
    if (shouldRefreshFloatingPianoRollThisTick)
        floatingPianoRollWorkspace->refreshPlaybackState();

    if (!stillRunning)
    {
        if (!activeRealtimeRecordedNotes.empty())
            finishActiveRealtimeRecordedNotes(documentState.project.playheadTick);
        rackPreviewRunning = false;
        projectPreviewRunning = false;
        playbackUiTickCounter = 0;
        std::fill(trackMeterLevels.begin(), trackMeterLevels.end(), 0.0f);
        transportMasterPeakLeft = 0.0f;
        transportMasterPeakRight = 0.0f;
        if (mixerComponent != nullptr)
            mixerComponent->refreshMeters();
        repaintTrackVolumeMeters();
        refreshPlaybackToggleButton();
        updateEditorState();
        statusLabel.setText("Native preview finished.", juce::dontSendNotification);
    }

    refreshFloatingWindows(shouldRefreshEditorsThisTick);
}

juce::Result StudioShellComponent::openProjectFile(const juce::File& file)
{
    if (rackPreviewRunning && nativeVstHost.isReady())
        nativeVstHost.stopAudioEngine();
    if (projectPreviewRunning && nativeVstHost.isReady())
        nativeVstHost.stopAudioEngine();
    closeAllRackEditorSessions();
    resetRackHostTracking();
    rackPreviewRunning = false;
    projectPreviewRunning = false;
    playbackUiTickCounter = 0;
    pendingLiveRackParameterEngineSyncTrack = -1;
    audioEngineStateValid = false;
    audioEngineStateDirty = true;

    ProjectFileData loaded;
    const auto result = loadProjectFile(file, loaded);
    if (result.failed())
        return result;

    documentState = std::move(loaded);
    syncBundledRackCatalogInProject();
    currentProjectFile = file;
    clearDirty();
    undoManager.clearUndoHistory();
    refreshUi();
    trackTable.selectRow(0);
    scheduleSelectedTrackRackPreviewWarmup(0);
    statusLabel.setText("Loaded project: " + file.getFileName(), juce::dontSendNotification);
    appendActivityLog("Project", "Loaded project\n" + file.getFullPathName());
    return juce::Result::ok();
}

void StudioShellComponent::refreshUi()
{
    normaliseProject(documentState.project);
    trackMeterLevels.assign(documentState.project.tracks.size(), 0.0f);

    tempoSlider.setValue(documentState.project.bpm, juce::dontSendNotification);
    timeSignatureNumeratorBox.setSelectedId(documentState.project.timeSigNumerator, juce::dontSendNotification);
    timeSignatureDenominatorBox.setSelectedId(documentState.project.timeSigDenominator, juce::dontSendNotification);
    loopToggle.setToggleState(documentState.project.loopEnabled, juce::dontSendNotification);
    metronomeToggle.setToggleState(documentState.project.metronomeEnabled, juce::dontSendNotification);
    recordToggle.setToggleState(transportRecordEnabled, juce::dontSendNotification);
    midiInsertToggle.setToggleState(midiInsertEnabled, juce::dontSendNotification);
    midiInsertToggle.setTooltip(activeMidiInputNames.isEmpty()
                                    ? "No MIDI input devices detected. Connect one to audition or insert notes."
                                    : "Insert incoming MIDI notes into the selected pattern. Listening to: "
                                        + activeMidiInputNames.joinIntoString(", "));
    refreshProjectSummaryLabels();

    const auto trackCount = static_cast<int>(documentState.project.tracks.size());
    const auto selectedRow = getSelectedTrackIndex();
    const auto selectedSampleAssetRow = sampleAssetList.getSelectedRow();
    trackTable.updateContent();
    sampleAssetList.updateContent();
    if (trackCount > 0)
        trackTable.selectRow(juce::jlimit(0, trackCount - 1, selectedRow >= 0 ? selectedRow : 0), juce::dontSendNotification);
    if (!documentState.project.sampleAssets.empty())
        sampleAssetList.selectRow(juce::jlimit(0,
                                               static_cast<int>(documentState.project.sampleAssets.size()) - 1,
                                               selectedSampleAssetRow >= 0 ? selectedSampleAssetRow : 0),
                                  juce::dontSendNotification);
    else
        sampleAssetList.deselectAllRows();

    ensureSelectedMidiSectionForTrack(getSelectedTrackIndex());
    populateSequenceTickBox(patternBarsBox, documentState.project);
    populateSequenceTickBox(arrangementSnapBox, documentState.project);
    auto patternBarsSelection = defaultPatternLengthTicks(documentState.project);
    if (juce::isPositiveAndBelow(selectedMidiSectionIndex, static_cast<int>(documentState.project.midiSections.size())))
    {
        const auto& section = documentState.project.midiSections[static_cast<size_t>(selectedMidiSectionIndex)];
        if (const auto* pattern = findMidiPattern(documentState.project, section.patternId))
            patternBarsSelection = patternLengthTicks(*pattern);
    }
    patternBarsBox.setSelectedId(patternBarsSelection, juce::dontSendNotification);
    if (patternBarsBox.getSelectedId() != patternBarsSelection)
        patternBarsBox.setText(sequenceTickLabel(patternBarsSelection, documentState.project), juce::dontSendNotification);
    keyQuantizeBox.setSelectedId(keyQuantizeOptionId(documentState.project), juce::dontSendNotification);
    if (keyQuantizeBox.getSelectedId() == 0)
        keyQuantizeBox.setText(keyQuantizeDisplayName(documentState.project.keyQuantizeRoot, documentState.project.keyQuantizeScale),
                               juce::dontSendNotification);
    arrangementSnapBox.setSelectedId(arrangementSnapTickLength(documentState.project), juce::dontSendNotification);
    if (arrangementSnapBox.getSelectedId() != arrangementSnapTickLength(documentState.project))
        arrangementSnapBox.setText(sequenceTickLabel(arrangementSnapTickLength(documentState.project), documentState.project),
                                   juce::dontSendNotification);
    arrangementZoomSlider.setValue(arrangementZoomPixelsPerBar, juce::dontSendNotification);
    arrangementLaneHeightSlider.setValue(arrangementLaneHeightPixels, juce::dontSendNotification);
    pianoRollZoomSlider.setValue(pianoRollZoomPixelsPerBeat, juce::dontSendNotification);
    pianoRollRowHeightSlider.setValue(pianoRollRowHeightPixels, juce::dontSendNotification);
    applyEditorViewScaleState();
    refreshInspector();
    updateEditorState();
    updateAiStatusSummary();
}

void StudioShellComponent::refreshProjectSummaryLabels()
{
    const auto totalLengthSec = projectSequenceLengthSeconds(documentState.project);

    if (headerTimecodeDisplay != nullptr)
    {
        headerTimecodeDisplay->setValues(documentState.project.playheadSec,
                                         totalLengthSec,
                                         documentState.project.leftLocatorSec,
                                         documentState.project.rightLocatorSec);
    }

    auto displayPath = currentProjectFile.existsAsFile()
        ? currentProjectFile.getFullPathName()
        : juce::String("Unsaved native project");
    if (dirty)
        displayPath << " *";
    fileLabel.setText(displayPath, juce::dontSendNotification);
}

void StudioShellComponent::applyEditorViewScaleState()
{
    const auto desiredTrackTableHeaderHeight = 24;
    if (trackTable.getHeaderHeight() != desiredTrackTableHeaderHeight)
        trackTable.setHeaderHeight(desiredTrackTableHeaderHeight);

    const auto desiredTrackRowHeight = juce::roundToInt(arrangementLaneHeightPixels);
    if (trackTable.getRowHeight() != desiredTrackRowHeight)
        trackTable.setRowHeight(desiredTrackRowHeight);

    if (arrangementOverview != nullptr)
    {
        arrangementOverview->setHorizontalZoom(arrangementZoomPixelsPerBar);
        arrangementOverview->setLaneHeight(arrangementLaneHeightPixels);
    }

    if (floatingArrangementOverview != nullptr)
    {
        floatingArrangementOverview->setHorizontalZoom(arrangementZoomPixelsPerBar);
        floatingArrangementOverview->setLaneHeight(arrangementLaneHeightPixels);
    }

    if (pianoRoll != nullptr)
    {
        pianoRoll->setHorizontalZoom(pianoRollZoomPixelsPerBeat);
        pianoRoll->setNoteRowHeight(pianoRollRowHeightPixels);
    }

    if (floatingPianoRollWorkspace != nullptr)
        floatingPianoRollWorkspace->setViewScale(pianoRollZoomPixelsPerBeat, pianoRollRowHeightPixels);

    if (panelsWindowContent != nullptr)
    {
        panelsWindowContent->setArrangementViewScale(arrangementZoomPixelsPerBar, arrangementLaneHeightPixels);
        panelsWindowContent->setPianoRollViewScale(pianoRollZoomPixelsPerBeat, pianoRollRowHeightPixels);
    }
}

void StudioShellComponent::repaintTrackVolumeMeters()
{
    if (!trackTable.isShowing())
        return;

    const auto rowCount = static_cast<int>(documentState.project.tracks.size());
    for (int row = 0; row < rowCount; ++row)
    {
        const auto cellBounds = trackTable.getCellPosition(kColumnVolume, row, true);
        if (!cellBounds.isEmpty())
            trackTable.repaint(cellBounds);
    }
}

void StudioShellComponent::refreshInspector()
{
    if (const auto* track = getSelectedTrack())
    {
        syncingInspectorControls = true;
        trackNameEditor.setText(track->name, juce::dontSendNotification);
        trackTypeEditor.setText(track->trackType, juce::dontSendNotification);
        instrumentModeEditor.setText(track->instrumentMode, juce::dontSendNotification);
        instrumentEditor.setText(track->instrument, juce::dontSendNotification);
        rackVstEditor.setText(track->rackVst, juce::dontSendNotification);
        midiChannelSlider.setValue(track->midiChannel + 1, juce::dontSendNotification);
        midiProgramSlider.setValue(track->midiProgram, juce::dontSendNotification);
        volumeSlider.setValue(track->volume, juce::dontSendNotification);
        panSlider.setValue(track->pan, juce::dontSendNotification);
        muteToggle.setToggleState(track->mute, juce::dontSendNotification);
        soloToggle.setToggleState(track->solo, juce::dontSendNotification);
        liveArmToggle.setToggleState(track->liveArmed, juce::dontSendNotification);
        syncingInspectorControls = false;

        trackNameEditor.setEnabled(true);
        trackTypeEditor.setEnabled(true);
        instrumentModeEditor.setEnabled(true);
        instrumentEditor.setEnabled(true);
        rackVstEditor.setEnabled(true);
        midiChannelSlider.setEnabled(true);
        midiProgramSlider.setEnabled(true);
        volumeSlider.setEnabled(true);
        panSlider.setEnabled(true);
        muteToggle.setEnabled(true);
        soloToggle.setEnabled(true);
        liveArmToggle.setEnabled(true);
        inspectorEditor.setText(describeTrack(documentState.project, *track), false);
    }
    else
    {
        syncingInspectorControls = true;
        trackNameEditor.setText({}, juce::dontSendNotification);
        trackTypeEditor.setText({}, juce::dontSendNotification);
        instrumentModeEditor.setText({}, juce::dontSendNotification);
        instrumentEditor.setText({}, juce::dontSendNotification);
        rackVstEditor.setText({}, juce::dontSendNotification);
        midiChannelSlider.setValue(1.0, juce::dontSendNotification);
        midiProgramSlider.setValue(0.0, juce::dontSendNotification);
        volumeSlider.setValue(0.8, juce::dontSendNotification);
        panSlider.setValue(0.0, juce::dontSendNotification);
        muteToggle.setToggleState(false, juce::dontSendNotification);
        soloToggle.setToggleState(false, juce::dontSendNotification);
        liveArmToggle.setToggleState(false, juce::dontSendNotification);
        syncingInspectorControls = false;

        trackNameEditor.setEnabled(false);
        trackTypeEditor.setEnabled(false);
        instrumentModeEditor.setEnabled(false);
        instrumentEditor.setEnabled(false);
        rackVstEditor.setEnabled(false);
        midiChannelSlider.setEnabled(false);
        midiProgramSlider.setEnabled(false);
        volumeSlider.setEnabled(false);
        panSlider.setEnabled(false);
        muteToggle.setEnabled(false);
        soloToggle.setEnabled(false);
        liveArmToggle.setEnabled(false);
        inspectorEditor.setText("No track selected.", false);
    }
}

void StudioShellComponent::updateEditorState()
{
    timeSignatureNumeratorBox.setSelectedId(documentState.project.timeSigNumerator, juce::dontSendNotification);
    timeSignatureDenominatorBox.setSelectedId(documentState.project.timeSigDenominator, juce::dontSendNotification);
    auto patternBarsSelection = defaultPatternLengthTicks(documentState.project);
    if (juce::isPositiveAndBelow(selectedMidiSectionIndex, static_cast<int>(documentState.project.midiSections.size())))
    {
        const auto& section = documentState.project.midiSections[static_cast<size_t>(selectedMidiSectionIndex)];
        if (const auto* pattern = findMidiPattern(documentState.project, section.patternId))
            patternBarsSelection = patternLengthTicks(*pattern);
    }
    patternBarsBox.setSelectedId(patternBarsSelection, juce::dontSendNotification);
    if (patternBarsBox.getSelectedId() != patternBarsSelection)
        patternBarsBox.setText(sequenceTickLabel(patternBarsSelection, documentState.project), juce::dontSendNotification);
    keyQuantizeBox.setSelectedId(keyQuantizeOptionId(documentState.project), juce::dontSendNotification);
    if (keyQuantizeBox.getSelectedId() == 0)
        keyQuantizeBox.setText(keyQuantizeDisplayName(documentState.project.keyQuantizeRoot, documentState.project.keyQuantizeScale),
                               juce::dontSendNotification);
    arrangementSnapBox.setSelectedId(arrangementSnapTickLength(documentState.project), juce::dontSendNotification);
    if (arrangementSnapBox.getSelectedId() != arrangementSnapTickLength(documentState.project))
        arrangementSnapBox.setText(sequenceTickLabel(arrangementSnapTickLength(documentState.project), documentState.project),
                                   juce::dontSendNotification);

    if (mixerComponent != nullptr)
        mixerComponent->refreshFromModel();

    if (arrangementOverview != nullptr)
        arrangementOverview->refreshFromModel();

    if (automationEditor != nullptr)
        automationEditor->refreshFromModel();

    if (sampleTimeline != nullptr)
        sampleTimeline->refreshFromModel();

    if (pianoRoll != nullptr)
    {
        pianoRoll->setToolMode(editorToolMode);
        pianoRoll->refreshFromModel();
    }

    placeSampleButton.setEnabled(sampleAssetList.getSelectedRow() >= 0 && findPreferredSampleTrackIndex() >= 0);
    refreshPlaybackToggleButton();
    exportWavButton.setEnabled(!documentState.project.tracks.empty() || !documentState.project.sampleClips.empty());
    const auto selectedTrack = getSelectedTrack();
    const bool selectedTrackIsSample = selectedTrack != nullptr && selectedTrack->trackType.equalsIgnoreCase("sample");
    const bool aiTaskBusy = aiComposeBusy || aceStepGenerationBusy;
    aiSettingsButton.setEnabled(!aiTaskBusy);
    aiComposeButton.setEnabled(!aiTaskBusy);
    aceStepGenerateButton.setEnabled(!aiTaskBusy && selectedTrackIsSample);
    undoButton.setEnabled(undoManager.canUndo());
    redoButton.setEnabled(undoManager.canRedo());
    refreshFloatingWindows();
}

void StudioShellComponent::refreshMidiEditState()
{
    auto patternBarsSelection = defaultPatternLengthTicks(documentState.project);
    if (juce::isPositiveAndBelow(selectedMidiSectionIndex, static_cast<int>(documentState.project.midiSections.size())))
    {
        const auto& section = documentState.project.midiSections[static_cast<size_t>(selectedMidiSectionIndex)];
        if (const auto* pattern = findMidiPattern(documentState.project, section.patternId))
            patternBarsSelection = patternLengthTicks(*pattern);
    }

    patternBarsBox.setSelectedId(patternBarsSelection, juce::dontSendNotification);
    if (patternBarsBox.getSelectedId() != patternBarsSelection)
        patternBarsBox.setText(sequenceTickLabel(patternBarsSelection, documentState.project), juce::dontSendNotification);

    if (arrangementOverview != nullptr)
        arrangementOverview->refreshFromModel();

    if (pianoRoll != nullptr)
    {
        pianoRoll->setToolMode(editorToolMode);
        pianoRoll->refreshFromModel();
    }

    refreshPlaybackToggleButton();
    undoButton.setEnabled(undoManager.canUndo());
    redoButton.setEnabled(undoManager.canRedo());
    refreshFloatingWindowsForMidiEdit();
}

void StudioShellComponent::markDirty()
{
    dirty = true;
}

void StudioShellComponent::clearDirty()
{
    dirty = false;
}

void StudioShellComponent::restorePersistedWindowVisibility()
{
    if (windowStateSettings == nullptr)
        return;

    transportWindowVisible = windowStateSettings->getBoolValue("window_transport_visible", transportWindowVisible);
    mixerWindowVisible = windowStateSettings->getBoolValue("window_mixer_visible", mixerWindowVisible);
    audioWindowVisible = windowStateSettings->getBoolValue("window_audio_visible", audioWindowVisible);
    panelsWindowVisible = windowStateSettings->getBoolValue("window_panels_visible", panelsWindowVisible);
    tracksWindowVisible = windowStateSettings->getBoolValue("window_tracks_visible", tracksWindowVisible);
    modulationMatrixWindowVisible = windowStateSettings->getBoolValue("window_modulation_matrix_visible", modulationMatrixWindowVisible);
    rackBrowserWindowVisible = windowStateSettings->getBoolValue("window_rack_browser_visible", rackBrowserWindowVisible);
    renderManagerWindowVisible = windowStateSettings->getBoolValue("window_render_manager_visible", renderManagerWindowVisible);
    arrangementWindowVisible = windowStateSettings->getBoolValue("window_arrangement_visible", arrangementWindowVisible);
    automationWindowVisible = windowStateSettings->getBoolValue("window_automation_visible", automationWindowVisible);
    samplesWindowVisible = windowStateSettings->getBoolValue("window_samples_visible", samplesWindowVisible);
    pianoRollWindowVisible = windowStateSettings->getBoolValue("window_piano_roll_visible", pianoRollWindowVisible);
    virtualPianoWindowVisible = windowStateSettings->getBoolValue("window_virtual_piano_visible", virtualPianoWindowVisible);
    activityLogWindowVisible = windowStateSettings->getBoolValue("window_activity_log_visible", activityLogWindowVisible);
}

void StudioShellComponent::persistWindowVisibilityState() const
{
    if (windowStateSettings == nullptr)
        return;

    windowStateSettings->setValue("window_transport_visible", transportWindowVisible);
    windowStateSettings->setValue("window_mixer_visible", mixerWindowVisible);
    windowStateSettings->setValue("window_audio_visible", audioWindowVisible);
    windowStateSettings->setValue("window_panels_visible", panelsWindowVisible);
    windowStateSettings->setValue("window_tracks_visible", tracksWindowVisible);
    windowStateSettings->setValue("window_modulation_matrix_visible", modulationMatrixWindowVisible);
    windowStateSettings->setValue("window_rack_browser_visible", rackBrowserWindowVisible);
    windowStateSettings->setValue("window_render_manager_visible", renderManagerWindowVisible);
    windowStateSettings->setValue("window_arrangement_visible", arrangementWindowVisible);
    windowStateSettings->setValue("window_automation_visible", automationWindowVisible);
    windowStateSettings->setValue("window_samples_visible", samplesWindowVisible);
    windowStateSettings->setValue("window_piano_roll_visible", pianoRollWindowVisible);
    windowStateSettings->setValue("window_virtual_piano_visible", virtualPianoWindowVisible);
    windowStateSettings->setValue("window_activity_log_visible", activityLogWindowVisible);
    windowStateSettings->saveIfNeeded();
}

void StudioShellComponent::refreshMidiInputDevices()
{
    cancelPendingUpdate();

    {
        const juce::ScopedLock lock(midiInputQueueLock);
        pendingMidiInputMessages.clear();
    }

    midiInputs.clear();
    activeMidiInputNames.clear();
    const auto availableDevices = juce::MidiInput::getAvailableDevices();

    for (const auto& device : availableDevices)
    {
        if (preferredMidiInputIdentifier == kMidiInputSelectionDisabled)
            break;

        if (preferredMidiInputIdentifier.isNotEmpty()
            && preferredMidiInputIdentifier != device.identifier)
        {
            continue;
        }

        auto input = juce::MidiInput::openDevice(device.identifier, this);
        if (input == nullptr)
            continue;

        activeMidiInputNames.add(midiInputDisplayName(device));
        input->start();
        midiInputs.push_back(std::move(input));
    }

    const auto midiTooltip = describeMidiInputSelection(preferredMidiInputIdentifier,
                                                        availableDevices,
                                                        activeMidiInputNames)
        + " Use MIDI In to insert notes into the selected pattern.";
    midiInsertToggle.setTooltip(midiTooltip);

    if (audioSettingsPanel != nullptr)
        audioSettingsPanel->applyMidiInputSnapshot(availableDevices, preferredMidiInputIdentifier);
}

void StudioShellComponent::dispatchPendingMidiInputMessages()
{
    std::vector<juce::MidiMessage> messages;

    {
        const juce::ScopedLock lock(midiInputQueueLock);
        if (pendingMidiInputMessages.empty())
            return;

        messages.swap(pendingMidiInputMessages);
    }

    const bool allowNoteOn = juce::Process::isForegroundProcess();

    for (const auto& message : messages)
    {
        if (message.isNoteOn())
        {
            if (!allowNoteOn)
                continue;

            handleMidiKeyboardNoteOn(message.getNoteNumber(),
                                     juce::jlimit(1, 127, static_cast<int>(message.getVelocity())));
        }
        else if (message.isNoteOff())
        {
            handleMidiKeyboardNoteOff(message.getNoteNumber(),
                                      juce::jlimit(0, 127, static_cast<int>(message.getVelocity())));
        }
    }
}

void StudioShellComponent::handleMidiKeyboardNoteOn(int pitch, int velocity)
{
    if (documentState.project.tracks.empty())
        addTrack();

    if (getSelectedTrackIndex() < 0 && !documentState.project.tracks.empty())
        setSelectedTrackIndex(0);

    if (projectPreviewRunning && transportRecordEnabled)
    {
        handleRealtimeMidiRecordingNoteOn(pitch, velocity);
    }
    else if (midiInsertEnabled)
    {
        if (activeMidiInsertHeldPitches.contains(pitch))
            return;

        activeMidiInsertHeldPitches.add(pitch);
        insertLiveMidiNote(pitch, velocity, true, false);
    }
    else
        ignoreUnused(previewSelectedTrackMidiNoteOn(pitch, velocity));
}

void StudioShellComponent::handleMidiKeyboardNoteOff(int pitch, int velocity)
{
    previewSelectedTrackMidiNoteOff(pitch, velocity);

    if (projectPreviewRunning && transportRecordEnabled)
    {
        handleRealtimeMidiRecordingNoteOff(pitch);
    }
    else if (midiInsertEnabled)
    {
        activeMidiInsertHeldPitches.removeFirstMatchingValue(pitch);
        if (activeMidiInsertHeldPitches.isEmpty())
        {
            activeMidiInsertPatternId.clear();
            activeMidiInsertTrackIndex = -1;
            activeMidiInsertChordStartTick = -1;
        }
    }
}

void StudioShellComponent::insertLiveMidiNote(int pitch,
                                              int velocity,
                                              bool flashVirtualKey,
                                              bool autoStopPreview)
{
    if (documentState.project.tracks.empty())
        addTrack();

    if (getSelectedTrackIndex() < 0 && !documentState.project.tracks.empty())
        setSelectedTrackIndex(0);

    const auto trackIndex = getSelectedTrackIndex();
    if (!juce::isPositiveAndBelow(trackIndex, static_cast<int>(documentState.project.tracks.size())))
        return;

    auto sectionIndex = getSelectedMidiSectionIndex();
    if (!juce::isPositiveAndBelow(sectionIndex, static_cast<int>(documentState.project.midiSections.size()))
        || documentState.project.midiSections[static_cast<size_t>(sectionIndex)].trackIndex != trackIndex)
    {
        ensureSelectedMidiSectionForTrack(trackIndex);
        sectionIndex = getSelectedMidiSectionIndex();
    }

    auto createdSection = false;
    if (!juce::isPositiveAndBelow(sectionIndex, static_cast<int>(documentState.project.midiSections.size())))
    {
        auto updatedProject = documentState.project;
        const auto patternLength = defaultPatternLengthTicks(updatedProject);
        const auto snapTicks = arrangementSnapTickLength(updatedProject);

        MidiPattern pattern;
        pattern.id = juce::Uuid().toString();
        pattern.name = updatedProject.tracks[static_cast<size_t>(trackIndex)].name.trim().isNotEmpty()
            ? updatedProject.tracks[static_cast<size_t>(trackIndex)].name.trim() + " Pattern"
            : "Pattern " + juce::String(static_cast<int>(updatedProject.midiPatterns.size()) + 1);
        pattern.lengthTicks = patternLength;

        MidiSection section;
        section.trackIndex = trackIndex;
        section.startTick = snapTicks > 0
            ? (updatedProject.playheadTick / snapTicks) * snapTicks
            : updatedProject.playheadTick;
        section.lengthTicks = patternLength;
        section.name = pattern.name;
        section.patternId = pattern.id;

        updatedProject.midiPatterns.push_back(std::move(pattern));
        updatedProject.midiSections.push_back(std::move(section));
        updatedProject.recalculateTimeFields();
        sectionIndex = static_cast<int>(updatedProject.midiSections.size()) - 1;
        applyProjectStateEdit(updatedProject, "Create Pattern");
        setSelectedTrackIndex(trackIndex);
        setSelectedMidiSectionIndex(sectionIndex, true);
        createdSection = true;
    }

    auto updatedProject = documentState.project;
    auto* pattern = findMidiPattern(updatedProject,
                                    updatedProject.midiSections[static_cast<size_t>(sectionIndex)].patternId);
    if (pattern == nullptr)
        return;

    const auto clampedPitch = juce::jlimit(kEditableMidiPitchMin, kEditableMidiPitchMax, pitch);
    const auto clampedVelocity = juce::jlimit(1, 127, velocity);
    const auto durationTick = kTicksPerBeat / 2;
    const bool allowChordInsertGrouping = !autoStopPreview;
    auto cursorTick = 0;
    if (allowChordInsertGrouping
        && activeMidiInsertChordStartTick >= 0
        && activeMidiInsertTrackIndex == trackIndex
        && activeMidiInsertPatternId == pattern->id
        && !activeMidiInsertHeldPitches.isEmpty())
    {
        cursorTick = activeMidiInsertChordStartTick;
    }
    else
    {
        for (const auto& note : pattern->notes)
            cursorTick = juce::jmax(cursorTick, note.startTick + note.durationTick);

        if (allowChordInsertGrouping)
        {
            activeMidiInsertTrackIndex = trackIndex;
            activeMidiInsertPatternId = pattern->id;
            activeMidiInsertChordStartTick = cursorTick;
        }
    }

    for (const auto& existingNote : pattern->notes)
    {
        if (existingNote.startTick == cursorTick
            && existingNote.pitch == clampedPitch
            && allowChordInsertGrouping)
        {
            ignoreUnused(previewSelectedTrackMidiNoteOn(clampedPitch, clampedVelocity));
            return;
        }
    }

    MidiNote note;
    note.startTick = cursorTick;
    note.durationTick = durationTick;
    note.pitch = clampedPitch;
    note.velocity = clampedVelocity;
    pattern->notes.push_back(note);
    pattern->lengthTicks = juce::jmax(pattern->lengthTicks, note.startTick + note.durationTick);
    updatedProject.recalculateTimeFields();

    applyProjectStateEdit(updatedProject, "Insert Live Note");
    setSelectedTrackIndex(trackIndex);
    setSelectedMidiSectionIndex(sectionIndex, true);

    if (flashVirtualKey && virtualPianoWindowContent != nullptr)
        virtualPianoWindowContent->flashPitch(clampedPitch);

    ignoreUnused(previewSelectedTrackMidiNoteOn(clampedPitch, clampedVelocity));

    if (autoStopPreview)
    {
        juce::Timer::callAfterDelay(juce::jlimit(90,
                                                 600,
                                                 juce::roundToInt(tickToSeconds(durationTick, documentState.project.bpm) * 1000.0)),
                                    [safeThis = juce::Component::SafePointer<StudioShellComponent>(this),
                                     trackIndex,
                                     clampedPitch]
                                    {
                                        if (safeThis == nullptr
                                            || !safeThis->nativeVstHost.isReady()
                                            || !juce::isPositiveAndBelow(trackIndex,
                                                                        static_cast<int>(safeThis->documentState.project.tracks.size())))
                                            return;

                                        const auto midiChannel = juce::jlimit(1,
                                                                              16,
                                                                              safeThis->documentState.project.tracks[static_cast<size_t>(trackIndex)].midiChannel + 1);
                                        safeThis->nativeVstHost.noteOffAudioEngineTrack(trackIndex,
                                                                                       clampedPitch,
                                                                                       midiChannel,
                                                                                       0.0f);
                                    });
    }

    if (createdSection)
        statusLabel.setText("Created a new pattern and inserted " + noteNameLabel(clampedPitch) + ".", juce::dontSendNotification);
}

void StudioShellComponent::handleIncomingMidiMessage(juce::MidiInput* source, const juce::MidiMessage& message)
{
    juce::ignoreUnused(source);

    if (!message.isNoteOn() && !message.isNoteOff())
        return;

    {
        const juce::ScopedLock lock(midiInputQueueLock);
        pendingMidiInputMessages.push_back(message);
    }

    triggerAsyncUpdate();
}

void StudioShellComponent::handleAsyncUpdate()
{
    dispatchPendingMidiInputMessages();
}

void StudioShellComponent::appendActivityLog(const juce::String& title, const juce::String& body)
{
    const auto timestamp = juce::Time::getCurrentTime().formatted("[%H:%M:%S]");
    const auto header = title.trim().isNotEmpty() ? timestamp + " " + title.trim() : timestamp;
    const auto entry = body.trim().isNotEmpty() ? (header + "\n" + body.trim()) : header;

    {
        const juce::ScopedLock lock(activityLogLock);
        activityLogEntries.add(entry);
        constexpr int kMaxActivityLogEntries = 400;
        while (activityLogEntries.size() > kMaxActivityLogEntries)
            activityLogEntries.remove(0);
    }

    appendActivityLogToFile(entry);

    auto refreshUiNow = [safeThis = juce::Component::SafePointer<StudioShellComponent>(this)]
    {
        if (safeThis == nullptr)
            return;

        if (safeThis->activityLogWindowContent != nullptr)
            safeThis->activityLogWindowContent->refreshFromModel();
    };

    if (auto* messageManager = juce::MessageManager::getInstanceWithoutCreating();
        messageManager != nullptr && messageManager->isThisTheMessageThread())
    {
        refreshUiNow();
    }
    else
    {
        juce::MessageManager::callAsync([refreshUiNow]
        {
            refreshUiNow();
        });
    }
}

void StudioShellComponent::appendActivityLogToFile(const juce::String& entry) const
{
    if (activityLogFile == juce::File())
        return;

    const auto parent = activityLogFile.getParentDirectory();
    if (parent != juce::File())
        parent.createDirectory();

    if (!activityLogFile.existsAsFile())
        activityLogFile.replaceWithText("Mutagen Activity Log\n\n", false, false, "\n");

    activityLogFile.appendText(entry + "\n\n", false, false, "\n");
}

juce::String StudioShellComponent::activityLogTextSnapshot() const
{
    const juce::ScopedLock lock(activityLogLock);
    return activityLogEntries.joinIntoString("\n\n");
}

void StudioShellComponent::clearActivityLog()
{
    {
        const juce::ScopedLock lock(activityLogLock);
        activityLogEntries.clear();
    }

    if (activityLogWindowContent != nullptr)
        activityLogWindowContent->refreshFromModel();
}

StudioShellComponent::RackEditorSession* StudioShellComponent::findRackEditorSession(int trackIndex)
{
    for (auto& session : rackEditorSessions)
    {
        if (session != nullptr && session->trackIndex == trackIndex)
            return session.get();
    }

    return nullptr;
}

const StudioShellComponent::RackEditorSession* StudioShellComponent::findRackEditorSession(int trackIndex) const
{
    for (const auto& session : rackEditorSessions)
    {
        if (session != nullptr && session->trackIndex == trackIndex)
            return session.get();
    }

    return nullptr;
}

bool StudioShellComponent::hasOpenRackEditorSessions() const
{
    for (const auto& session : rackEditorSessions)
    {
        if (session != nullptr && session->editorOpen)
            return true;
    }

    return false;
}

bool StudioShellComponent::isTrackBeingLiveEdited(int trackIndex) const
{
    if (trackIndex < 0)
        return false;

    if (const auto* session = findRackEditorSession(trackIndex))
        return session->editorOpen;

    return false;
}

juce::Result StudioShellComponent::ensureRackEditorSessionReadyForTrack(int trackIndex,
                                                                        const TrackState& track,
                                                                        RackEditorSession*& outSession)
{
    outSession = nullptr;

    const auto pluginPath = resolveRackPluginPath(documentState.project, track);
    if (pluginPath.isEmpty())
        return juce::Result::fail("The selected track does not resolve to a native rack plugin. Assign a bundled rack instrument or point Rack VST to a plugin path.");

    const auto pluginFile = juce::File(pluginPath);
    if (!pluginFile.exists())
        return juce::Result::fail("The resolved Rack VST path does not exist:\n" + pluginPath);

    auto result = ensureNativeAudioEnginePrepared(projectPreviewRunning);
    if (result.failed())
        return result;

    if (auto* existingSession = findRackEditorSession(trackIndex))
    {
        bool editorOpen = false;
        existingSession->editorOpen = nativeVstHost.queryAudioEngineTrackEditorOpen(trackIndex, editorOpen).wasOk()
            && editorOpen;
        outSession = existingSession;
        return juce::Result::ok();
    }

    auto session = std::make_unique<RackEditorSession>();
    session->trackIndex = trackIndex;
    bool editorOpen = false;
    session->editorOpen = nativeVstHost.queryAudioEngineTrackEditorOpen(trackIndex, editorOpen).wasOk()
        && editorOpen;

    outSession = session.get();
    rackEditorSessions.push_back(std::move(session));
    refreshPollingTimerState();
    return juce::Result::ok();
}

void StudioShellComponent::syncOpenRackEditorSessions(bool duringPlayback)
{
    const auto nowMs = juce::Time::getMillisecondCounter();
    if (duringPlayback && (nowMs - lastRackEditorSessionSyncMs) < kPlaybackRackEditorSyncIntervalMs)
        return;

    if (duringPlayback)
        lastRackEditorSessionSyncMs = nowMs;

    juce::Array<int> sharedSessionsToClose;
    for (auto& session : rackEditorSessions)
    {
        if (session == nullptr)
            continue;

        if (!session->editorOpen)
            continue;

        bool editorOpen = false;
        if (nativeVstHost.queryAudioEngineTrackEditorOpen(session->trackIndex, editorOpen).failed())
            continue;

        if (editorOpen)
            continue;

        NativeVstHostSession::RackParameterSnapshot rackSnapshot;
        if (nativeVstHost.queryAudioEngineTrackParameterSnapshot(session->trackIndex, rackSnapshot).wasOk())
        {
            session->lastStateGeneration = rackSnapshot.stateGeneration;
            syncTrackRackParametersFromValues(session->trackIndex,
                                              rackSnapshot.parameterValues,
                                              true);
        }

        session->editorOpen = false;
        sharedSessionsToClose.addIfNotAlreadyThere(session->trackIndex);
    }

    for (const auto trackIndex : sharedSessionsToClose)
        closeRackEditorSession(trackIndex);

    if (!sharedSessionsToClose.isEmpty())
    {
        refreshPollingTimerState();
        updateEditorState();
        trackTable.repaint();
    }
}

void StudioShellComponent::closeRackEditorSession(int trackIndex)
{
    for (auto it = rackEditorSessions.begin(); it != rackEditorSessions.end(); ++it)
    {
        auto& session = *it;
        if (session == nullptr || session->trackIndex != trackIndex)
            continue;

        nativeVstHost.closeAudioEngineTrackEditor(trackIndex);

        rackEditorSessions.erase(it);
        break;
    }
}

void StudioShellComponent::closeAllRackEditorSessions()
{
    for (auto& session : rackEditorSessions)
    {
        if (session == nullptr)
            continue;

        nativeVstHost.closeAudioEngineTrackEditor(session->trackIndex);
    }

    rackEditorSessions.clear();
}

juce::Result StudioShellComponent::ensureNativeAudioEnginePrepared(bool preserveTransport)
{
    auto result = nativeVstHost.ensureCreated();
    if (result.failed())
        return result;

    if (!audioEngineStateValid || audioEngineStateDirty)
    {
        result = nativeVstHost.setAudioEngineState(documentState.project, preserveTransport);
        if (result.failed())
            return result;

        pendingLiveRackParameterEngineSyncTrack = -1;
        audioEngineStateValid = true;
        audioEngineStateDirty = false;
    }

    return juce::Result::ok();
}

void StudioShellComponent::showTrackContextMenu(int rowNumber, juce::Point<int> screenPosition)
{
    if (!juce::isPositiveAndBelow(rowNumber, static_cast<int>(documentState.project.tracks.size())))
        return;

    setSelectedTrackIndex(rowNumber);
    const auto& track = documentState.project.tracks[static_cast<size_t>(rowNumber)];
    const bool canAssignRack = !track.trackType.trim().equalsIgnoreCase("sample");
    const bool hasAssignedRack = track.rackVst.trim().isNotEmpty()
        || track.instrumentMode.trim().equalsIgnoreCase("VSTI Rack");

    juce::PopupMenu menu;
    constexpr int menuEditVst = 1;
    constexpr int menuAutoAssignRack = 2;
    constexpr int menuClearRack = 3;
    constexpr int menuAddTrack = 4;
    constexpr int menuAddSampleTrack = 5;
    constexpr int menuDuplicateTrack = 6;
    constexpr int menuRemoveTrack = 7;
    constexpr int menuShowAllAutomationLanes = 8;
    constexpr int menuHideAllAutomationLanes = 9;
    constexpr int menuGenerateAceStepAudio = 10;
    int nextRackMenuId = 100;
    int nextAutomationMenuId = 200;
    std::vector<std::pair<int, juce::String>> rackAssignments;
    std::vector<std::pair<int, juce::String>> automationLaneAssignments;
    const auto usedAutomationTargetsForMenu = usedAutomationTargets(track);
    const auto visibleAutomationTargetsForMenu = visibleArrangementAutomationTargets(track);

    juce::PopupMenu rackMenu;
    const auto currentRackIndex = findRackInstrumentIndexByReference(documentState.project, track.rackVst);
    for (int rackIndex = 0; rackIndex < static_cast<int>(documentState.project.vstRack.size()); ++rackIndex)
    {
        const auto& entry = documentState.project.vstRack[static_cast<size_t>(rackIndex)];
        if (!entry.hostSupported || !entry.isInstrument)
            continue;

        const auto reference = entry.path.isNotEmpty() ? entry.path : entry.name;
        if (reference.trim().isEmpty())
            continue;

        auto label = entry.name;
        if (label.trim().isEmpty())
            label = entry.pluginName;
        if (label.trim().isEmpty() && entry.path.isNotEmpty())
            label = juce::File(entry.path).getFileNameWithoutExtension();
        if (label.trim().isEmpty())
            label = "Rack " + juce::String(rackIndex + 1);

        rackMenu.addItem(nextRackMenuId,
                         label,
                         canAssignRack,
                         currentRackIndex == rackIndex);
        rackAssignments.emplace_back(nextRackMenuId, reference);
        ++nextRackMenuId;
    }

    if (rackAssignments.empty())
        rackMenu.addItem(nextRackMenuId, "No Rack Instruments Found", false);

    juce::PopupMenu automationLaneMenu;
    for (const auto& target : usedAutomationTargetsForMenu)
    {
        const auto menuItemId = nextAutomationMenuId++;
        automationLaneMenu.addItem(menuItemId,
                                   automationTargetLabel(track, target),
                                   true,
                                   visibleAutomationTargetsForMenu.contains(target, true));
        automationLaneAssignments.emplace_back(menuItemId, target);
    }

    if (!usedAutomationTargetsForMenu.isEmpty())
        automationLaneMenu.addSeparator();

    automationLaneMenu.addItem(menuShowAllAutomationLanes,
                               "Show All Used Lanes",
                               !usedAutomationTargetsForMenu.isEmpty(),
                               !usedAutomationTargetsForMenu.isEmpty()
                                   && visibleAutomationTargetsForMenu.size() == usedAutomationTargetsForMenu.size());
    automationLaneMenu.addItem(menuHideAllAutomationLanes,
                               "Hide All Automation Lanes",
                               !visibleAutomationTargetsForMenu.isEmpty());

    if (usedAutomationTargetsForMenu.isEmpty())
        automationLaneMenu.addItem(nextAutomationMenuId, "No Used Automation Lanes", false);

    menu.addItem(menuEditVst,
                 "Edit VST",
                 track.trackType != "sample" && track.instrumentMode.containsIgnoreCase("VST"));
    menu.addSubMenu("Assign Rack Instrument", rackMenu, canAssignRack && !rackAssignments.empty());
    menu.addSubMenu("Automation Lanes", automationLaneMenu, true);
    menu.addItem(menuAutoAssignRack, "Auto Assign Rack", canAssignRack);
    menu.addItem(menuClearRack, "Clear Rack Assignment", canAssignRack && hasAssignedRack);
    menu.addSeparator();
    menu.addItem(menuGenerateAceStepAudio,
                 "Generate Audio With ACE-Step",
                 track.trackType.equalsIgnoreCase("sample") && !aiComposeBusy && !aceStepGenerationBusy);
    menu.addSeparator();
    menu.addItem(menuAddTrack, "Add Instrument Track");
    menu.addItem(menuAddSampleTrack, "Add Sample Track");
    menu.addSeparator();
    menu.addItem(menuDuplicateTrack, "Duplicate Track");
    menu.addItem(menuRemoveTrack, "Remove Track");

    menu.showMenuAsync(juce::PopupMenu::Options().withTargetScreenArea({ screenPosition.x, screenPosition.y, 1, 1 }),
                       [safeThis = juce::Component::SafePointer<StudioShellComponent>(this),
                        rowNumber,
                        rackAssignments = std::move(rackAssignments),
                        automationLaneAssignments = std::move(automationLaneAssignments),
                        usedAutomationTargetsForMenu,
                        visibleAutomationTargetsForMenu](int result)
                       {
                           if (safeThis == nullptr)
                               return;

                           safeThis->setSelectedTrackIndex(rowNumber);
                           for (const auto& [itemId, reference] : rackAssignments)
                           {
                               if (result == itemId)
                               {
                                   safeThis->assignSelectedTrackRackByReference(reference);
                                   return;
                               }
                           }

                           for (const auto& [itemId, target] : automationLaneAssignments)
                           {
                               if (result != itemId)
                                   continue;

                               if (!juce::isPositiveAndBelow(rowNumber,
                                                             static_cast<int>(safeThis->documentState.project.tracks.size())))
                               {
                                   return;
                               }

                               auto updatedTrack = safeThis->documentState.project.tracks[static_cast<size_t>(rowNumber)];
                               auto visibleTargets = visibleArrangementAutomationTargets(updatedTrack);
                               const auto currentlyVisible = visibleTargets.contains(target, true);
                               if (currentlyVisible)
                                   visibleTargets.removeString(target, true);
                               else
                                   visibleTargets.addIfNotAlreadyThere(target);

                               updatedTrack.arrangementVisibleAutomationTargets = visibleTargets;
                               safeThis->applyTrackStateEdit(rowNumber,
                                                             updatedTrack,
                                                             currentlyVisible ? "Hide Automation Lane"
                                                                              : "Show Automation Lane");
                               return;
                           }

                           switch (result)
                           {
                               case menuEditVst: safeThis->openSelectedTrackRackEditor(); break;
                               case menuShowAllAutomationLanes:
                               {
                                   if (!juce::isPositiveAndBelow(rowNumber,
                                                                 static_cast<int>(safeThis->documentState.project.tracks.size())))
                                   {
                                       break;
                                   }

                                   auto updatedTrack = safeThis->documentState.project.tracks[static_cast<size_t>(rowNumber)];
                                   updatedTrack.arrangementVisibleAutomationTargets = usedAutomationTargetsForMenu;
                                   safeThis->applyTrackStateEdit(rowNumber,
                                                                 updatedTrack,
                                                                 "Show Automation Lanes");
                                   break;
                               }
                               case menuHideAllAutomationLanes:
                               {
                                   if (!juce::isPositiveAndBelow(rowNumber,
                                                                 static_cast<int>(safeThis->documentState.project.tracks.size())))
                                   {
                                       break;
                                   }

                                   auto updatedTrack = safeThis->documentState.project.tracks[static_cast<size_t>(rowNumber)];
                                   updatedTrack.arrangementVisibleAutomationTargets.clear();
                                   safeThis->applyTrackStateEdit(rowNumber,
                                                                 updatedTrack,
                                                                 "Hide Automation Lanes");
                                   break;
                               }
                                case menuAutoAssignRack: safeThis->materialiseSelectedTrackRackAssignment(); break;
                               case menuClearRack: safeThis->clearSelectedTrackRackAssignment(); break;
                               case menuGenerateAceStepAudio: safeThis->generateAudioWithAceStep(); break;
                               case menuAddTrack: safeThis->addTrack(); break;
                               case menuAddSampleTrack: safeThis->addSampleTrack(); break;
                               case menuDuplicateTrack: safeThis->duplicateSelectedTrack(); break;
                               case menuRemoveTrack: safeThis->removeSelectedTrack(); break;
                               default: break;
                           }
                       });
}

void StudioShellComponent::setupFloatingWindows()
{
    const auto displayArea = juce::Desktop::getInstance().getDisplays().getPrimaryDisplay() != nullptr
        ? juce::Desktop::getInstance().getDisplays().getPrimaryDisplay()->userArea
        : juce::Rectangle<int>(80, 80, 1600, 900);
    const auto attachVirtualPianoShortcutHandler = [this] (FloatingPanelWindow* window)
    {
        if (window == nullptr)
            return;

        window->onKeyPressed = [this] (const juce::KeyPress& key)
        {
            return keyPressed(key);
        };
    };

    transportWindow = std::make_unique<FloatingPanelWindow>("Transport", true);
    attachVirtualPianoShortcutHandler(transportWindow.get());
    transportWindow->onClosePressed = [this]
    {
        transportWindowVisible = false;
        persistWindowVisibilityState();
    };
    auto* newTransportPanel = new TransportPanelComponent([this] { jumpPlayheadToStart(); },
                                                          [this] { playFullProjectThroughNativeEngine(); },
                                                          [this] { playSelectedTrackThroughRack(); },
                                                          [this] { stopRackPreview(); },
                                                          [this] (int tick) { setTransportPlayheadTick(tick); },
                                                          [this] (int tick) { setTransportLeftLocatorTick(tick); },
                                                          [this] (int tick) { setTransportRightLocatorTick(tick); },
                                                          [this] (int bpm) { setTransportTempo(bpm); },
                                                          [this] (bool enabled) { setTransportLoopEnabled(enabled); },
                                                          [this] (bool enabled) { setTransportMetronomeEnabled(enabled); },
                                                          [this] (bool enabled) { setTransportRecordEnabled(enabled); });
    transportPanel = newTransportPanel;
    transportWindow->setContentOwned(newTransportPanel, true);
    transportWindow->setBounds(displayArea.getX() + juce::jmax(24, (displayArea.getWidth() - 980) / 2),
                               displayArea.getY() + 18,
                               juce::jmin(980, displayArea.getWidth() - 48),
                               164);
    transportWindow->setVisible(transportWindowVisible);

    audioWindow = std::make_unique<FloatingPanelWindow>("Audio");
    attachVirtualPianoShortcutHandler(audioWindow.get());
    audioWindow->onClosePressed = [this]
    {
        audioWindowVisible = false;
        persistWindowVisibilityState();
    };
    auto* newFloatingAudioWorkspace = new AudioWorkspaceWindowComponent([this] { jumpPlayheadToStart(); },
                                                                        [this] { playFullProjectThroughNativeEngine(); },
                                                                        [this] { playSelectedTrackThroughRack(); },
                                                                        [this] { stopRackPreview(); },
                                                                        [this] { showAudioSettingsWindow(); },
                                                                        [this] (int tick) { setTransportPlayheadTick(tick); },
                                                                        [this] (int tick) { setTransportLeftLocatorTick(tick); },
                                                                        [this] (int tick) { setTransportRightLocatorTick(tick); },
                                                                        [this] (int bpm) { setTransportTempo(bpm); },
                                                                        [this] (bool enabled) { setTransportLoopEnabled(enabled); },
                                                                        [this] (bool enabled) { setTransportMetronomeEnabled(enabled); },
                                                                        [this] (bool enabled) { setTransportRecordEnabled(enabled); },
                                                                        [this] { promptExportWav(); },
                                                                        [this] { promptExportProjectStems(); },
                                                                        [this] () -> const ProjectState& { return documentState.project; },
                                                                        [this] (int trackIndex,
                                                                                const TrackState& track,
                                                                                bool undoable,
                                                                                const juce::String& actionName)
                                                                        {
                                                                            if (undoable)
                                                                                applyTrackStateEdit(trackIndex, track, actionName);
                                                                            else
                                                                                replaceTrackStateNoUndo(trackIndex, track);
                                                                        },
                                                                        [this] (const ProjectState& project,
                                                                                bool undoable,
                                                                                const juce::String& actionName)
                                                                        {
                                                                            if (undoable)
                                                                                applyProjectStateEdit(project, actionName);
                                                                            else
                                                                                setProjectStateInternal(project);
                                                                        },
                                                                        [this] (int trackIndex) -> float
                                                                        {
                                                                            return juce::isPositiveAndBelow(trackIndex, static_cast<int>(trackMeterLevels.size()))
                                                                                ? trackMeterLevels[static_cast<size_t>(trackIndex)]
                                                                                : 0.0f;
                                                                        },
                                                                        [this] () -> std::pair<float, float>
                                                                        {
                                                                            return { transportMasterPeakLeft, transportMasterPeakRight };
                                                                        },
                                                                        [this] () -> juce::String
                                                                        {
                                                                            const auto createResult = nativeVstHost.ensureCreated();
                                                                            if (createResult.failed())
                                                                                return "Audio host unavailable.";

                                                                            NativeVstHostSession::AudioDeviceSnapshot snapshot;
                                                                            const auto snapshotResult = nativeVstHost.queryAudioDeviceSnapshot(snapshot);
                                                                            if (snapshotResult.failed())
                                                                                return "Audio device status unavailable.";

                                                                            const auto sampleRate = juce::roundToInt(snapshot.sampleRate);
                                                                            const auto bufferSize = snapshot.bufferSize;

                                                                            return "Driver: "
                                                                                + (snapshot.audioDeviceType.isNotEmpty() ? snapshot.audioDeviceType : "Default")
                                                                                + "  |  Device: "
                                                                                + (snapshot.audioDeviceName.isNotEmpty() ? snapshot.audioDeviceName : "Unavailable")
                                                                                + "  |  "
                                                                                + juce::String(juce::jmax(1, sampleRate))
                                                                                + " Hz  |  "
                                                                                + juce::String(juce::jmax(1, bufferSize))
                                                                                + " samples";
                                                                        });
    floatingAudioWorkspace = newFloatingAudioWorkspace;
    audioWindow->setContentOwned(newFloatingAudioWorkspace, true);
    audioWindow->setBounds(displayArea.getRight() - juce::jmin(1160, displayArea.getWidth() - 80) - 36,
                           displayArea.getY() + 154,
                           juce::jmin(1160, displayArea.getWidth() - 80),
                           juce::jmin(520, displayArea.getHeight() - 230));
    audioWindow->setVisible(audioWindowVisible);

    mixerWindow = std::make_unique<FloatingPanelWindow>("Mixer");
    attachVirtualPianoShortcutHandler(mixerWindow.get());
    mixerWindow->onClosePressed = [this]
    {
        mixerWindowVisible = false;
        persistWindowVisibilityState();
    };
    auto* newFloatingMixer = new MixerComponent([this] () -> const ProjectState& { return documentState.project; },
                                                [this] (int trackIndex,
                                                        const TrackState& track,
                                                        bool undoable,
                                                        const juce::String& actionName)
                                                {
                                                    if (undoable)
                                                        applyTrackStateEdit(trackIndex, track, actionName);
                                                    else
                                                        replaceTrackStateNoUndo(trackIndex, track);
                                                },
                                                [this] (const ProjectState& project,
                                                        bool undoable,
                                                        const juce::String& actionName)
                                                {
                                                    if (undoable)
                                                        applyProjectStateEdit(project, actionName);
                                                    else
                                                        setProjectStateInternal(project);
                                                },
                                                [this] (int trackIndex) -> float
                                                {
                                                    return juce::isPositiveAndBelow(trackIndex, static_cast<int>(trackMeterLevels.size()))
                                                        ? trackMeterLevels[static_cast<size_t>(trackIndex)]
                                                        : 0.0f;
                                                },
                                                [this] () -> std::pair<float, float>
                                                {
                                                    return { transportMasterPeakLeft, transportMasterPeakRight };
                                                },
                                                [this] (int trackIndex, int effectIndex)
                                                {
                                                    openTrackEffectEditorFromMixer(trackIndex, effectIndex);
                                                },
                                                [this] (int effectIndex)
                                                {
                                                    openMasterEffectEditorFromMixer(effectIndex);
                                                });
    floatingMixerComponent = newFloatingMixer;
    mixerWindow->setContentOwned(newFloatingMixer, true);
    mixerWindow->setBounds(displayArea.getX() + 30,
                           displayArea.getY() + 150,
                           juce::jmin(1180, displayArea.getWidth() - 60),
                           juce::jmin(420, displayArea.getHeight() - 220));
    mixerWindow->setVisible(mixerWindowVisible);

    tracksWindow = std::make_unique<FloatingPanelWindow>("Tracks");
    attachVirtualPianoShortcutHandler(tracksWindow.get());
    tracksWindow->onClosePressed = [this]
    {
        tracksWindowVisible = false;
        persistWindowVisibilityState();
    };
    auto* newFloatingTracksWorkspace = new TracksWorkspaceWindowComponent([this] () -> const ProjectState& { return documentState.project; },
                                                                          [this] { return getSelectedTrackIndex(); },
                                                                          [this] (int row) { setSelectedTrackIndex(row); },
                                                                          [this] (std::function<void(TrackState&)> mutation,
                                                                                  const juce::String& actionName)
                                                                          {
                                                                              applySelectedTrackMutation(std::move(mutation), actionName);
                                                                          },
                                                                          [this] (int trackIndex) -> juce::String
                                                                          {
                                                                              if (!juce::isPositiveAndBelow(trackIndex, static_cast<int>(documentState.project.tracks.size())))
                                                                                  return "No track selected.";
                                                                              return describeTrack(documentState.project,
                                                                                                   documentState.project.tracks[static_cast<size_t>(trackIndex)]);
                                                                          },
                                                                          [this] { addTrack(); },
                                                                          [this] { duplicateSelectedTrack(); },
                                                                          [this] { removeSelectedTrack(); },
                                                                          [this] { openSelectedTrackRackEditor(); },
                                                                          [this] { saveSelectedTrackRackState(); },
                                                                          [this] { playSelectedTrackThroughRack(); },
                                                                          [this] { stopRackPreview(); });
    floatingTracksWorkspace = newFloatingTracksWorkspace;
    tracksWindow->setContentOwned(newFloatingTracksWorkspace, true);
    tracksWindow->setBounds(displayArea.getX() + 36,
                            displayArea.getY() + 176,
                            juce::jmin(1220, displayArea.getWidth() - 72),
                            juce::jmin(720, displayArea.getHeight() - 230));
    tracksWindow->setVisible(tracksWindowVisible);

    
    rackBrowserWindow = std::make_unique<FloatingPanelWindow>("Rack Browser");
    attachVirtualPianoShortcutHandler(rackBrowserWindow.get());
    rackBrowserWindow->onClosePressed = [this]
    {
        rackBrowserWindowVisible = false;
        persistWindowVisibilityState();
    };
    auto* newFloatingRackBrowser = new RackBrowserWindowComponent([this] () -> const ProjectState& { return documentState.project; },
                                                                  [this] { return getSelectedTrackIndex(); },
                                                                  [this] (const juce::String& reference) { assignSelectedTrackRackByReference(reference); },
                                                                  [this] { clearSelectedTrackRackAssignment(); },
                                                                  [this] { materialiseSelectedTrackRackAssignment(); },
                                                                  [this] { promptImportRackPlugin(); },
                                                                  [this] { refreshRackCatalog(); },
                                                                  [this] { openSelectedTrackRackEditor(); },
                                                                  [this] { saveSelectedTrackRackState(); },
                                                                  [this] { playSelectedTrackThroughRack(); },
                                                                  [this] { stopRackPreview(); });
    floatingRackBrowser = newFloatingRackBrowser;
    rackBrowserWindow->setContentOwned(newFloatingRackBrowser, true);
    rackBrowserWindow->setBounds(displayArea.getX() + 82,
                                 displayArea.getY() + 162,
                                 juce::jmin(1180, displayArea.getWidth() - 100),
                                 juce::jmin(700, displayArea.getHeight() - 210));
    rackBrowserWindow->setVisible(rackBrowserWindowVisible);

    renderManagerWindow = std::make_unique<FloatingPanelWindow>("Render Manager");
    attachVirtualPianoShortcutHandler(renderManagerWindow.get());
    renderManagerWindow->onClosePressed = [this]
    {
        renderManagerWindowVisible = false;
        persistWindowVisibilityState();
    };
    auto* newFloatingRenderManager = new RenderManagerWindowComponent([this] () -> const ProjectState& { return documentState.project; },
                                                                      [this] { return getSelectedTrackIndex(); },
                                                                      [this] (int row) { setSelectedTrackIndex(row); },
                                                                      [this] { promptExportSelectedTrackWav(); },
                                                                      [this] { promptExportProjectStems(); },
                                                                      [this] { promptRelinkSelectedTrackRenderedAudio(); },
                                                                      [this] { clearSelectedTrackRenderedAudioPath(); },
                                                                      [this] { importSelectedTrackRenderToSampleLibrary(); },
                                                                      [this] { placeSelectedTrackRenderAtPlayhead(); });
    floatingRenderManager = newFloatingRenderManager;
    renderManagerWindow->setContentOwned(newFloatingRenderManager, true);
    renderManagerWindow->setBounds(displayArea.getX() + 110,
                                   displayArea.getY() + 188,
                                   juce::jmin(1120, displayArea.getWidth() - 120),
                                   juce::jmin(620, displayArea.getHeight() - 220));
    renderManagerWindow->setVisible(renderManagerWindowVisible);

    arrangementWindow = std::make_unique<FloatingPanelWindow>("Arrangement");
    attachVirtualPianoShortcutHandler(arrangementWindow.get());
    arrangementWindow->onClosePressed = [this]
    {
        arrangementWindowVisible = false;
        persistWindowVisibilityState();
    };
    auto* newFloatingArrangement = new ArrangementOverviewComponent([this] () -> const ProjectState& { return documentState.project; },
                                                                    [this] (const ProjectState& project,
                                                                            bool undoable,
                                                                            const juce::String& actionName)
                                                                    {
                                                                        if (undoable)
                                                                            applyProjectStateEdit(project, actionName);
                                                                        else
                                                                            setProjectStateInternal(project);
                                                                    },
                                                                    [this] () { return getSelectedMidiSectionIndex(); },
                                                                    [this] (int sectionIndex, bool focusEditor)
                                                                    {
                                                                        setSelectedMidiSectionIndex(sectionIndex, true);
                                                                        if (focusEditor)
                                                                            focusMidiSectionInPianoRoll(sectionIndex);
                                                                    },
                                                                    [this] () { return editorToolMode; },
                                                                    [this] (EditorToolMode mode)
                                                                    {
                                                                        setEditorToolMode(mode);
                                                                    });
    newFloatingArrangement->setZoomChangedCallback([this] (float pixelsPerBar)
                                                   {
                                                       if (std::abs(arrangementZoomPixelsPerBar - pixelsPerBar) < 0.01f)
                                                           return;

                                                       arrangementZoomPixelsPerBar = pixelsPerBar;
                                                       applyEditorViewScaleState();
                                                   });
    newFloatingArrangement->setLaneHeightChangedCallback([this] (float laneHeightPixels)
                                                         {
                                                             if (std::abs(arrangementLaneHeightPixels - laneHeightPixels) < 0.01f)
                                                                 return;

                                                             arrangementLaneHeightPixels = laneHeightPixels;
                                                             applyEditorViewScaleState();
                                                         });
    newFloatingArrangement->setSampleClipStemSeparationCallback([this] (int clipIndex)
                                                                {
                                                                    separateSampleClipToStems(clipIndex);
                                                                });
    floatingArrangementOverview = newFloatingArrangement;
    arrangementWindow->setContentOwned(newFloatingArrangement, true);
    arrangementWindow->setBounds(displayArea.getX() + 42,
                                 displayArea.getY() + 180,
                                 juce::jmin(1120, displayArea.getWidth() - 84),
                                 juce::jmin(360, displayArea.getHeight() - 260));
    arrangementWindow->setVisible(arrangementWindowVisible);

    automationWindow = std::make_unique<FloatingPanelWindow>("Automation");
    attachVirtualPianoShortcutHandler(automationWindow.get());
    automationWindow->onClosePressed = [this]
    {
        automationWindowVisible = false;
        persistWindowVisibilityState();
    };
    auto* newFloatingAutomation = new AutomationEditorComponent([this] () -> const ProjectState& { return documentState.project; },
                                                                [this] () -> int { return getSelectedTrackIndex(); },
                                                                [this] (int trackIndex,
                                                                        const TrackState& track,
                                                                        bool undoable,
                                                                        const juce::String& actionName)
                                                                {
                                                                    if (trackIndex < 0)
                                                                        return;

                                                                    if (undoable)
                                                                        applyTrackStateEdit(trackIndex, track, actionName);
                                                                    else
                                                                        replaceTrackStateNoUndo(trackIndex, track);
                                                                });
    floatingAutomationEditor = newFloatingAutomation;
    automationWindow->setContentOwned(newFloatingAutomation, true);
    automationWindow->setBounds(displayArea.getX() + 88,
                                displayArea.getY() + 206,
                                juce::jmin(980, displayArea.getWidth() - 176),
                                juce::jmin(560, displayArea.getHeight() - 260));
    automationWindow->setVisible(automationWindowVisible);

    samplesWindow = std::make_unique<FloatingPanelWindow>("Samples");
    attachVirtualPianoShortcutHandler(samplesWindow.get());
    samplesWindow->onClosePressed = [this]
    {
        samplesWindowVisible = false;
        persistWindowVisibilityState();
    };
    auto* newFloatingSampleWorkspace = new SampleWorkspaceWindowComponent([this] () -> const ProjectState& { return documentState.project; },
                                                                          [this] (const ProjectState& project,
                                                                                  bool undoable,
                                                                                  const juce::String& actionName)
                                                                          {
                                                                              if (undoable)
                                                                                  applyProjectStateEdit(project, actionName);
                                                                              else
                                                                                  setProjectStateInternal(project);
                                                                          },
                                                                          [this] { return getSelectedSampleAssetIndex(); },
                                                                          [this] (int row) { setSelectedSampleAssetIndex(row); },
                                                                          [this] { promptImportSample(); },
                                                                          [this] { placeSelectedSampleAtPlayhead(); },
                                                                          [this]
                                                                          {
                                                                              return getSelectedSampleAssetIndex() >= 0
                                                                                  && findPreferredSampleTrackIndex() >= 0;
                                                                          });
    floatingSampleWorkspace = newFloatingSampleWorkspace;
    samplesWindow->setContentOwned(newFloatingSampleWorkspace, true);
    samplesWindow->setBounds(displayArea.getX() + 120,
                             displayArea.getY() + 236,
                             juce::jmin(1180, displayArea.getWidth() - 180),
                             juce::jmin(680, displayArea.getHeight() - 260));
    samplesWindow->setVisible(samplesWindowVisible);

    pianoRollWindow = std::make_unique<FloatingPanelWindow>("Piano Roll");
    attachVirtualPianoShortcutHandler(pianoRollWindow.get());
    pianoRollWindow->onClosePressed = [this]
    {
        pianoRollWindowVisible = false;
        persistWindowVisibilityState();
    };
    auto* newFloatingPianoRoll = new PianoRollWindowComponent([this] () -> const ProjectState& { return documentState.project; },
                                                              [this] () { return getSelectedTrackIndex(); },
                                                              [this] () { return getSelectedMidiSectionIndex(); },
                                                              [this] (const ProjectState& project,
                                                                      bool undoable,
                                                                      const juce::String& actionName)
                                                              {
                                                                  if (undoable)
                                                                      applyProjectStateEdit(project, actionName);
                                                                  else
                                                                      setProjectStateInternal(project);
                                                               },
                                                               [this] { return editorToolMode; },
                                                               [this] (EditorToolMode mode) { setEditorToolMode(mode); },
                                                               pianoRollZoomPixelsPerBeat,
                                                               pianoRollRowHeightPixels,
                                                               [this] (float zoom)
                                                               {
                                                                   pianoRollZoomPixelsPerBeat = zoom;
                                                                   applyEditorViewScaleState();
                                                               },
                                                               [this] (float rowHeight)
                                                               {
                                                                   pianoRollRowHeightPixels = rowHeight;
                                                                   applyEditorViewScaleState();
                                                               });
    newFloatingPianoRoll->setNotePreviewCallbacks([this] (int pitch, int velocity)
                                                  {
                                                      previewSelectedTrackMidiNoteOn(pitch, velocity);
                                                  },
                                                  [this] (int pitch, int velocity)
                                                  {
                                                      previewSelectedTrackMidiNoteOff(pitch, velocity);
                                                  },
                                                  [this]
                                                  {
                                                      stopSelectedTrackMidiPreview();
                                                  });
    newFloatingPianoRoll->setKeyHandlerCallback([this] (const juce::KeyPress& key)
                                                {
                                                    return keyPressed(key);
                                                });
    floatingPianoRollWorkspace = newFloatingPianoRoll;
    pianoRollWindow->setContentOwned(newFloatingPianoRoll, true);
    pianoRollWindow->setBounds(displayArea.getX() + 146,
                               displayArea.getY() + 262,
                               juce::jmin(1140, displayArea.getWidth() - 200),
                               juce::jmin(720, displayArea.getHeight() - 260));
    pianoRollWindow->setVisible(pianoRollWindowVisible);

    virtualPianoWindow = std::make_unique<FloatingPanelWindow>("Virtual Piano");
    attachVirtualPianoShortcutHandler(virtualPianoWindow.get());
    virtualPianoWindow->onClosePressed = [this]
    {
        virtualPianoWindowVisible = false;
        persistWindowVisibilityState();
    };
    virtualPianoWindow->onKeyPressed = [this] (const juce::KeyPress& key)
    {
        return tryHandleVirtualPianoShortcut(key);
    };
    auto* newVirtualPianoWindowContent = new VirtualPianoWindowComponent([this] (int pitch)
                                                                         {
                                                                             insertVirtualKeyboardNote(pitch, false);
                                                                         },
                                                                         [this] (const juce::KeyPress& key)
                                                                         {
                                                                             return tryHandleVirtualPianoShortcut(key);
                                                                         });
    virtualPianoWindowContent = newVirtualPianoWindowContent;
    virtualPianoWindow->setContentOwned(newVirtualPianoWindowContent, true);
    virtualPianoWindow->setBounds(displayArea.getX() + 188,
                                  displayArea.getY() + 286,
                                  juce::jmin(920, displayArea.getWidth() - 240),
                                  290);
    virtualPianoWindow->setVisible(virtualPianoWindowVisible);

    activityLogWindow = std::make_unique<FloatingPanelWindow>("Activity Log");
    attachVirtualPianoShortcutHandler(activityLogWindow.get());
    activityLogWindow->onClosePressed = [this]
    {
        activityLogWindowVisible = false;
        persistWindowVisibilityState();
    };
    auto* newActivityLogWindowContent = new ActivityLogWindowComponent([this]
                                                                       {
                                                                           return activityLogTextSnapshot();
                                                                       },
                                                                       [this]
                                                                       {
                                                                           clearActivityLog();
                                                                       });
    activityLogWindowContent = newActivityLogWindowContent;
    activityLogWindow->setContentOwned(newActivityLogWindowContent, true);
    activityLogWindow->setBounds(displayArea.getX() + 168,
                                 displayArea.getY() + 108,
                                 juce::jmin(920, displayArea.getWidth() - 220),
                                 juce::jmin(680, displayArea.getHeight() - 180));
    activityLogWindow->setVisible(activityLogWindowVisible);

    panelsWindow = std::make_unique<FloatingPanelWindow>("Panels");
    attachVirtualPianoShortcutHandler(panelsWindow.get());
    panelsWindow->onClosePressed = [this]
    {
        panelsWindowVisible = false;
        persistWindowVisibilityState();
    };
    auto* newPanelsWindowContent = new PanelsWindowComponent([this] () -> const ProjectState& { return documentState.project; },
                                                             [this] () { return getSelectedMidiSectionIndex(); },
                                                             [this] (int sectionIndex, bool focusEditor)
                                                             {
                                                                 setSelectedMidiSectionIndex(sectionIndex, true);
                                                                 if (focusEditor)
                                                                     focusMidiSectionInPianoRoll(sectionIndex);
                                                             },
                                                             [this] { return getSelectedTrackIndex(); },
                                                             [this] (int trackIndex,
                                                                     const TrackState& track,
                                                                     bool undoable,
                                                                     const juce::String& actionName)
                                                             {
                                                                 if (undoable)
                                                                     applyTrackStateEdit(trackIndex, track, actionName);
                                                                 else
                                                                     replaceTrackStateNoUndo(trackIndex, track);
                                                             },
                                                             [this] (const ProjectState& project,
                                                                     bool undoable,
                                                                     const juce::String& actionName)
                                                             {
                                                                 if (undoable)
                                                                     applyProjectStateEdit(project, actionName);
                                                                 else
                                                                     setProjectStateInternal(project);
                                                              },
                                                              [this] { return editorToolMode; },
                                                              [this] (EditorToolMode mode) { setEditorToolMode(mode); });
    newPanelsWindowContent->setNotePreviewCallbacks([this] (int pitch, int velocity)
                                                    {
                                                        previewSelectedTrackMidiNoteOn(pitch, velocity);
                                                    },
                                                    [this] (int pitch, int velocity)
                                                    {
                                                        previewSelectedTrackMidiNoteOff(pitch, velocity);
                                                    },
                                                    [this]
                                                    {
                                                        stopSelectedTrackMidiPreview();
                                                    });
    newPanelsWindowContent->setKeyHandlerCallback([this] (const juce::KeyPress& key)
                                                  {
                                                      return keyPressed(key);
                                                  });
    newPanelsWindowContent->setArrangementZoomChangedCallback([this] (float pixelsPerBar)
                                                              {
                                                                  if (std::abs(arrangementZoomPixelsPerBar - pixelsPerBar) < 0.01f)
                                                                      return;

                                                                  arrangementZoomPixelsPerBar = pixelsPerBar;
                                                                  applyEditorViewScaleState();
                                                              });
    newPanelsWindowContent->setArrangementLaneHeightChangedCallback([this] (float laneHeightPixels)
                                                                    {
                                                                        if (std::abs(arrangementLaneHeightPixels - laneHeightPixels) < 0.01f)
                                                                            return;

                                                                        arrangementLaneHeightPixels = laneHeightPixels;
                                                                        applyEditorViewScaleState();
                                                                    });
    newPanelsWindowContent->setPianoRollZoomChangedCallback([this] (float pixelsPerBeat)
                                                            {
                                                                if (std::abs(pianoRollZoomPixelsPerBeat - pixelsPerBeat) < 0.01f)
                                                                    return;

                                                                pianoRollZoomPixelsPerBeat = pixelsPerBeat;
                                                                applyEditorViewScaleState();
                                                            });
    newPanelsWindowContent->setPianoRollRowHeightChangedCallback([this] (float rowHeightPixels)
                                                                 {
                                                                     if (std::abs(pianoRollRowHeightPixels - rowHeightPixels) < 0.01f)
                                                                         return;

                                                                     pianoRollRowHeightPixels = rowHeightPixels;
                                                                     applyEditorViewScaleState();
                                                                 });
    panelsWindowContent = newPanelsWindowContent;
    panelsWindow->setContentOwned(newPanelsWindowContent, true);
    panelsWindow->setBounds(displayArea.getX() + juce::jmax(40, displayArea.getWidth() / 8),
                            displayArea.getY() + juce::jmax(84, displayArea.getHeight() / 8),
                            juce::jmin(1120, displayArea.getWidth() - 80),
                            juce::jmin(760, displayArea.getHeight() - 120));
    panelsWindow->setVisible(panelsWindowVisible);

    audioSettingsWindow = std::make_unique<FloatingPanelWindow>("Audio Settings");
    attachVirtualPianoShortcutHandler(audioSettingsWindow.get());
    auto* newAudioSettingsPanel = new AudioSettingsPanelComponent([this] (const juce::String& audioDeviceType,
                                                                          const juce::String& audioDeviceName,
                                                                          int sampleRate,
                                                                          int bufferSize)
                                                                 {
                                                                     return applyAudioDeviceSettings(audioDeviceType,
                                                                                                     audioDeviceName,
                                                                                                     sampleRate,
                                                                                                     bufferSize);
                                                                 },
                                                                 [this] (const juce::String& midiInputIdentifier)
                                                                 {
                                                                     return setPreferredMidiInputDevice(midiInputIdentifier);
                                                                 },
                                                                 [this] { refreshAudioSettingsFromHost(true); });
    audioSettingsPanel = newAudioSettingsPanel;
    audioSettingsWindow->setContentOwned(newAudioSettingsPanel, true);
    audioSettingsWindow->setBounds(displayArea.getCentreX() - 260,
                                   displayArea.getCentreY() - 190,
                                   520,
                                   340);
    audioSettingsWindow->setVisible(false);

    vstFolderManagerWindow = std::make_unique<FloatingPanelWindow>("VST Folder Manager");
    attachVirtualPianoShortcutHandler(vstFolderManagerWindow.get());
    auto* newVstFolderManagerPanel = new VstFolderManagerComponent([this]
                                                                   {
                                                                       return defaultVstFolderPath();
                                                                   },
                                                                   [this]
                                                                   {
                                                                       return userManagedVstFolderPaths();
                                                                   },
                                                                   [this]
                                                                   {
                                                                       promptAddUserVstFolder();
                                                                   },
                                                                   [this] (const juce::String& folderPath)
                                                                   {
                                                                       removeUserVstFolder(folderPath);
                                                                   },
                                                                   [this]
                                                                   {
                                                                       refreshRackCatalog();
                                                                   });
    vstFolderManagerPanel = newVstFolderManagerPanel;
    vstFolderManagerWindow->setContentOwned(newVstFolderManagerPanel, true);
    vstFolderManagerWindow->setBounds(displayArea.getCentreX() - 320,
                                      displayArea.getCentreY() - 190,
                                      640,
                                      380);
    vstFolderManagerWindow->setVisible(false);

    refreshFloatingWindows();
}

void StudioShellComponent::refreshFloatingWindows(bool includeEditorRefresh)
{
    const ScopedAppProfileSample profileSample(AppProfileSection::refreshFloatingWindows);
    const bool deferHostUiQueries = (loadedRackEditorOpen || hasOpenRackEditorSessions())
        && !rackPreviewRunning
        && !projectPreviewRunning;

    if (transportPanel != nullptr && transportWindow != nullptr && transportWindow->isVisible())
    {
        transportPanel->refreshFromState(documentState.project,
                                         getSelectedTrack() != nullptr,
                                         rackPreviewRunning,
                                         projectPreviewRunning,
                                         transportRecordEnabled,
                                         statusLabel.getText(),
                                         transportCpuUsagePercent,
                                         transportMasterPeakLeft,
                                         transportMasterPeakRight);
    }

    if (floatingAudioWorkspace != nullptr && audioWindow != nullptr && audioWindow->isVisible())
    {
        floatingAudioWorkspace->refreshFromModel(documentState.project,
                                                 getSelectedTrack() != nullptr,
                                                 rackPreviewRunning,
                                                 projectPreviewRunning,
                                                 transportRecordEnabled,
                                                 statusLabel.getText(),
                                                 transportCpuUsagePercent,
                                                 transportMasterPeakLeft,
                                                 transportMasterPeakRight,
                                                 deferHostUiQueries);
    }

    if (floatingMixerComponent != nullptr && mixerWindow != nullptr && mixerWindow->isVisible())
    {
        if (includeEditorRefresh)
            floatingMixerComponent->refreshFromModel();
        floatingMixerComponent->refreshMeters();
    }

    if (includeEditorRefresh && floatingTracksWorkspace != nullptr && tracksWindow != nullptr && tracksWindow->isVisible())
        floatingTracksWorkspace->refreshFromModel();

    if (includeEditorRefresh && floatingModulationMatrix != nullptr && modulationMatrixWindow != nullptr && modulationMatrixWindow->isVisible())
        floatingModulationMatrix->refreshFromModel();

    if (includeEditorRefresh && floatingRackBrowser != nullptr && rackBrowserWindow != nullptr && rackBrowserWindow->isVisible())
        floatingRackBrowser->refreshFromModel();

    if (activityLogWindowContent != nullptr && activityLogWindow != nullptr && activityLogWindow->isVisible())
        activityLogWindowContent->refreshFromModel();

    if (includeEditorRefresh && floatingRenderManager != nullptr && renderManagerWindow != nullptr && renderManagerWindow->isVisible())
        floatingRenderManager->refreshFromModel();

    if (includeEditorRefresh && floatingArrangementOverview != nullptr && arrangementWindow != nullptr && arrangementWindow->isVisible())
        floatingArrangementOverview->refreshFromModel();

    if (includeEditorRefresh && floatingAutomationEditor != nullptr && automationWindow != nullptr && automationWindow->isVisible())
        floatingAutomationEditor->refreshFromModel();

    if (includeEditorRefresh && floatingSampleWorkspace != nullptr && samplesWindow != nullptr && samplesWindow->isVisible())
        floatingSampleWorkspace->refreshFromModel();

    if (includeEditorRefresh && floatingPianoRollWorkspace != nullptr && pianoRollWindow != nullptr && pianoRollWindow->isVisible())
    {
        if (projectPreviewRunning || rackPreviewRunning)
            floatingPianoRollWorkspace->refreshPlaybackState();
        else
            floatingPianoRollWorkspace->refreshFromModel();
    }

    if (includeEditorRefresh && panelsWindowContent != nullptr && panelsWindow != nullptr && panelsWindow->isVisible())
        panelsWindowContent->refreshFromModel();

    if (includeEditorRefresh && audioSettingsWindow != nullptr && audioSettingsWindow->isVisible() && !deferHostUiQueries)
        refreshAudioSettingsFromHost(false);

    if (includeEditorRefresh && vstFolderManagerPanel != nullptr && vstFolderManagerWindow != nullptr && vstFolderManagerWindow->isVisible())
        vstFolderManagerPanel->refreshFromModel();
}

void StudioShellComponent::refreshFloatingWindowsForMidiEdit()
{
    if (floatingArrangementOverview != nullptr && arrangementWindow != nullptr && arrangementWindow->isVisible())
        floatingArrangementOverview->refreshFromModel();

    if (floatingPianoRollWorkspace != nullptr && pianoRollWindow != nullptr && pianoRollWindow->isVisible())
    {
        if (projectPreviewRunning || rackPreviewRunning)
            floatingPianoRollWorkspace->refreshPlaybackState();
        else
            floatingPianoRollWorkspace->refreshFromModel();
    }

    if (panelsWindowContent != nullptr && panelsWindow != nullptr && panelsWindow->isVisible())
        panelsWindowContent->refreshMidiEditState();
}

void StudioShellComponent::jumpPlayheadToStart()
{
    if (documentState.project.playheadTick == 0)
        return;

    auto updatedProject = documentState.project;
    updatedProject.playheadTick = 0;
    updatedProject.recalculateTimeFields();
    applyProjectStateEdit(updatedProject, "Go To Start");
    statusLabel.setText("Moved playhead to start.", juce::dontSendNotification);
}

void StudioShellComponent::setTransportPlayheadTick(int tick)
{
    auto updatedProject = documentState.project;
    updatedProject.playheadTick = juce::jmax(0, tick);
    updatedProject.recalculateTimeFields();
    if (updatedProject.playheadTick == documentState.project.playheadTick)
        return;

    applyProjectStateEdit(updatedProject, "Move Playhead");
    statusLabel.setText("Moved playhead to tick " + juce::String(documentState.project.playheadTick) + ".", juce::dontSendNotification);
}

void StudioShellComponent::setTransportLeftLocatorTick(int tick)
{
    auto updatedProject = documentState.project;
    const auto minimumSpan = juce::jmax(1, ticksPerTimeSignatureBeat(updatedProject));
    updatedProject.leftLocatorTick = juce::jmax(0, tick);
    updatedProject.rightLocatorTick = juce::jmax(updatedProject.leftLocatorTick + minimumSpan, updatedProject.rightLocatorTick);
    updatedProject.recalculateTimeFields();
    if (updatedProject.leftLocatorTick == documentState.project.leftLocatorTick
        && updatedProject.rightLocatorTick == documentState.project.rightLocatorTick)
    {
        return;
    }

    applyProjectStateEdit(updatedProject, "Move Left Locator");
    statusLabel.setText("Moved left locator to tick " + juce::String(documentState.project.leftLocatorTick) + ".", juce::dontSendNotification);
}

void StudioShellComponent::setTransportRightLocatorTick(int tick)
{
    auto updatedProject = documentState.project;
    const auto minimumSpan = juce::jmax(1, ticksPerTimeSignatureBeat(updatedProject));
    updatedProject.rightLocatorTick = juce::jmax(updatedProject.leftLocatorTick + minimumSpan, tick);
    updatedProject.recalculateTimeFields();
    if (updatedProject.rightLocatorTick == documentState.project.rightLocatorTick)
        return;

    applyProjectStateEdit(updatedProject, "Move Right Locator");
    statusLabel.setText("Moved right locator to tick " + juce::String(documentState.project.rightLocatorTick) + ".", juce::dontSendNotification);
}

void StudioShellComponent::setTransportTempo(int bpm)
{
    const auto newTempo = juce::jlimit(20, 300, bpm);
    if (documentState.project.bpm == newTempo)
        return;

    auto updatedProject = documentState.project;
    updatedProject.bpm = newTempo;
    updatedProject.tempoMarkers.clear();
    updatedProject.tempoMarkers.push_back({ 0, newTempo });
    updatedProject.recalculateTimeFields();
    applyProjectStateEdit(updatedProject, "Change Tempo");
    statusLabel.setText("Updated tempo to " + juce::String(newTempo) + " BPM.", juce::dontSendNotification);
}

void StudioShellComponent::setTransportLoopEnabled(bool enabled)
{
    if (documentState.project.loopEnabled == enabled)
        return;

    auto updatedProject = documentState.project;
    updatedProject.loopEnabled = enabled;
    applyProjectStateEdit(updatedProject, "Toggle Loop");
    statusLabel.setText("Updated loop setting.", juce::dontSendNotification);
}

void StudioShellComponent::setTransportMetronomeEnabled(bool enabled)
{
    if (documentState.project.metronomeEnabled == enabled)
        return;

    auto updatedProject = documentState.project;
    updatedProject.metronomeEnabled = enabled;
    applyProjectStateEdit(updatedProject, "Toggle Metronome");
    statusLabel.setText("Updated metronome setting.", juce::dontSendNotification);
}

void StudioShellComponent::setTransportRecordEnabled(bool enabled)
{
    if (transportRecordEnabled == enabled)
        return;

    if (!enabled)
        finishActiveRealtimeRecordedNotes();

    transportRecordEnabled = enabled;
    refreshPlaybackToggleButton();
    refreshFloatingWindows(false);
    persistSessionState();
    statusLabel.setText(enabled
                            ? "Transport record armed for incoming MIDI on the selected track."
                            : "Transport record disarmed.",
                        juce::dontSendNotification);
}

int StudioShellComponent::currentAudioEngineTransportTick() const
{
    if (!projectPreviewRunning || !nativeVstHost.isReady())
        return juce::jmax(0, documentState.project.playheadTick);

    NativeVstHostSession::TransportSnapshot snapshot;
    if (nativeVstHost.queryTransportSnapshot(snapshot, false).failed())
        return juce::jmax(0, documentState.project.playheadTick);

    const auto sampleRate = juce::jmax(1.0, snapshot.sampleRate);
    return juce::jmax(0,
                      frameToTick(documentState.project,
                                  snapshot.audioEnginePositionFrame,
                                  sampleRate));
}

int StudioShellComponent::ensureMidiSectionForTrackAtTick(ProjectState& project,
                                                          int trackIndex,
                                                          int tick,
                                                          bool& createdSection)
{
    createdSection = false;
    if (!juce::isPositiveAndBelow(trackIndex, static_cast<int>(project.tracks.size())))
        return -1;

    const auto targetTick = juce::jmax(0, tick);

    for (int sectionIndex = 0; sectionIndex < static_cast<int>(project.midiSections.size()); ++sectionIndex)
    {
        const auto& section = project.midiSections[static_cast<size_t>(sectionIndex)];
        if (section.trackIndex != trackIndex)
            continue;

        const auto sectionEndTick = section.startTick + juce::jmax(kMinSequenceSnapTicks, section.lengthTicks);
        if (targetTick >= section.startTick && targetTick < sectionEndTick)
            return sectionIndex;
    }

    const auto patternLength = defaultPatternLengthTicks(project);
    MidiPattern pattern;
    pattern.id = juce::Uuid().toString();
    pattern.name = project.tracks[static_cast<size_t>(trackIndex)].name.trim().isNotEmpty()
        ? project.tracks[static_cast<size_t>(trackIndex)].name.trim() + " Pattern"
        : "Pattern " + juce::String(static_cast<int>(project.midiPatterns.size()) + 1);
    pattern.lengthTicks = juce::jmax(kMinSequenceSnapTicks, patternLength);

    MidiSection section;
    section.trackIndex = trackIndex;
    section.startTick = targetTick;
    section.lengthTicks = pattern.lengthTicks;
    section.name = pattern.name;
    section.patternId = pattern.id;

    project.midiPatterns.push_back(std::move(pattern));
    project.midiSections.push_back(std::move(section));
    project.recalculateTimeFields();
    createdSection = true;
    return static_cast<int>(project.midiSections.size()) - 1;
}

void StudioShellComponent::commitRealtimeRecordedProjectState(ProjectState updatedProject,
                                                              const juce::Array<int>& changedTrackIndices,
                                                              int preferredTrackIndex,
                                                              int preferredSectionIndex)
{
    if (changedTrackIndices.isEmpty())
        return;

    markDirty();
    documentState.project = std::move(updatedProject);
    normaliseProject(documentState.project);

    if (juce::isPositiveAndBelow(preferredTrackIndex, static_cast<int>(documentState.project.tracks.size())))
        setSelectedTrackIndex(preferredTrackIndex);
    if (juce::isPositiveAndBelow(preferredSectionIndex, static_cast<int>(documentState.project.midiSections.size())))
        setSelectedMidiSectionIndex(preferredSectionIndex, false);

    refreshProjectSummaryLabels();
    trackTable.repaint();
    if (preferredTrackIndex == getSelectedTrackIndex())
        refreshInspector();
    updateEditorState();
    if (arrangementOverview != nullptr && !arrangementViewport.getBounds().isEmpty())
        arrangementOverview->repaint();
    if (pianoRoll != nullptr && !pianoRollViewport.getBounds().isEmpty())
        pianoRoll->repaint();
    refreshFloatingWindows(false);

    if ((projectPreviewRunning || rackPreviewRunning) && nativeVstHost.isReady())
    {
        juce::Result result = juce::Result::ok();
        bool usedIncrementalUpdate = false;

        if (changedTrackIndices.size() == 1)
        {
            const auto trackIndex = changedTrackIndices.getFirst();
            if (juce::isPositiveAndBelow(trackIndex, static_cast<int>(documentState.project.tracks.size())))
            {
                result = nativeVstHost.setAudioEngineTrackNotes(trackIndex,
                                                                documentState.project,
                                                                documentState.project.tracks[static_cast<size_t>(trackIndex)]);
                usedIncrementalUpdate = result.wasOk();
            }
        }

        if (!usedIncrementalUpdate)
            result = nativeVstHost.setAudioEngineState(documentState.project, true);

        if (result.failed())
        {
            audioEngineStateValid = false;
            audioEngineStateDirty = true;
            statusLabel.setText("Live record sync failed: " + result.getErrorMessage(),
                                juce::dontSendNotification);
        }
        else
        {
            pendingLiveRackParameterEngineSyncTrack = -1;
            audioEngineStateValid = true;
            audioEngineStateDirty = false;
        }
    }
}

void StudioShellComponent::handleRealtimeMidiRecordingNoteOn(int pitch, int velocity)
{
    if (!projectPreviewRunning)
    {
        ignoreUnused(previewSelectedTrackMidiNoteOn(pitch, velocity));
        return;
    }

    if (documentState.project.tracks.empty())
        addTrack();

    if (getSelectedTrackIndex() < 0 && !documentState.project.tracks.empty())
        setSelectedTrackIndex(0);

    const auto trackIndex = getSelectedTrackIndex();
    if (!juce::isPositiveAndBelow(trackIndex, static_cast<int>(documentState.project.tracks.size())))
        return;

    const auto clampedPitch = juce::jlimit(kEditableMidiPitchMin, kEditableMidiPitchMax, pitch);
    const auto clampedVelocity = juce::jlimit(1, 127, velocity);

    for (const auto& activeNote : activeRealtimeRecordedNotes)
    {
        if (activeNote.trackIndex == trackIndex && activeNote.pitch == clampedPitch)
            return;
    }

    const auto recordTick = currentAudioEngineTransportTick();
    auto updatedProject = documentState.project;
    bool createdSection = false;
    const auto sectionIndex = ensureMidiSectionForTrackAtTick(updatedProject, trackIndex, recordTick, createdSection);
    if (!juce::isPositiveAndBelow(sectionIndex, static_cast<int>(updatedProject.midiSections.size())))
        return;

    const auto& section = updatedProject.midiSections[static_cast<size_t>(sectionIndex)];
    auto* pattern = findMidiPattern(updatedProject, section.patternId);
    if (pattern == nullptr)
        return;

    MidiNote note;
    note.startTick = juce::jmax(0, recordTick - section.startTick);
    note.durationTick = kMinSequenceSnapTicks;
    note.pitch = clampedPitch;
    note.velocity = clampedVelocity;
    pattern->notes.push_back(note);
    pattern->lengthTicks = juce::jmax(pattern->lengthTicks, note.startTick + note.durationTick);

    auto& mutableSection = updatedProject.midiSections[static_cast<size_t>(sectionIndex)];
    mutableSection.lengthTicks = juce::jmax(mutableSection.lengthTicks, pattern->lengthTicks);
    updatedProject.recalculateTimeFields();

    const auto patternId = pattern->id;
    const auto sectionStartTick = section.startTick;
    const auto patternNoteStartTick = note.startTick;
    juce::Array<int> changedTracks;
    changedTracks.add(trackIndex);
    commitRealtimeRecordedProjectState(std::move(updatedProject), changedTracks, trackIndex, sectionIndex);

    activeRealtimeRecordedNotes.push_back({ trackIndex,
                                            patternId,
                                            sectionStartTick,
                                            patternNoteStartTick,
                                            recordTick,
                                            clampedPitch });

    ignoreUnused(previewSelectedTrackMidiNoteOn(clampedPitch, clampedVelocity));
    statusLabel.setText(createdSection
                            ? "Recording MIDI into a new pattern on " + documentState.project.tracks[static_cast<size_t>(trackIndex)].name + "."
                            : "Recording MIDI into " + documentState.project.tracks[static_cast<size_t>(trackIndex)].name + ".",
                        juce::dontSendNotification);
}

void StudioShellComponent::handleRealtimeMidiRecordingNoteOff(int pitch)
{
    const auto clampedPitch = juce::jlimit(kEditableMidiPitchMin, kEditableMidiPitchMax, pitch);
    auto activeIt = std::find_if(activeRealtimeRecordedNotes.rbegin(),
                                 activeRealtimeRecordedNotes.rend(),
                                 [clampedPitch] (const ActiveRealtimeRecordedNote& note)
                                 {
                                     return note.pitch == clampedPitch;
                                 });
    if (activeIt == activeRealtimeRecordedNotes.rend())
        return;

    const auto activeNote = *activeIt;
    activeRealtimeRecordedNotes.erase(std::next(activeIt).base());

    auto updatedProject = documentState.project;
    auto* pattern = findMidiPattern(updatedProject, activeNote.patternId);
    if (pattern == nullptr)
        return;

    auto endTick = currentAudioEngineTransportTick();
    const auto loopSpan = updatedProject.loopEnabled
        ? juce::jmax(0, updatedProject.rightLocatorTick - updatedProject.leftLocatorTick)
        : 0;
    while (loopSpan > 0 && endTick < activeNote.absoluteStartTick)
        endTick += loopSpan;

    const auto relativeEndTick = juce::jmax(activeNote.patternNoteStartTick + kMinSequenceSnapTicks,
                                            endTick - activeNote.sectionStartTick);

    for (auto& note : pattern->notes)
    {
        if (note.pitch != activeNote.pitch || note.startTick != activeNote.patternNoteStartTick)
            continue;

        note.durationTick = juce::jmax(kMinSequenceSnapTicks, relativeEndTick - note.startTick);
        pattern->lengthTicks = juce::jmax(pattern->lengthTicks, note.startTick + note.durationTick);
        break;
    }

    for (auto& section : updatedProject.midiSections)
    {
        if (section.trackIndex != activeNote.trackIndex
            || section.patternId != activeNote.patternId
            || section.startTick != activeNote.sectionStartTick)
            continue;

        section.lengthTicks = juce::jmax(section.lengthTicks, pattern->lengthTicks);
        break;
    }

    updatedProject.recalculateTimeFields();
    juce::Array<int> changedTracks;
    changedTracks.add(activeNote.trackIndex);
    commitRealtimeRecordedProjectState(std::move(updatedProject), changedTracks, activeNote.trackIndex, -1);
}

void StudioShellComponent::finishActiveRealtimeRecordedNotes(int endTick)
{
    if (activeRealtimeRecordedNotes.empty())
        return;

    auto updatedProject = documentState.project;
    const auto resolvedEndTick = endTick >= 0 ? endTick : currentAudioEngineTransportTick();
    const auto loopSpan = updatedProject.loopEnabled
        ? juce::jmax(0, updatedProject.rightLocatorTick - updatedProject.leftLocatorTick)
        : 0;

    juce::Array<int> changedTracks;
    int preferredTrackIndex = -1;

    for (const auto& activeNote : activeRealtimeRecordedNotes)
    {
        auto* pattern = findMidiPattern(updatedProject, activeNote.patternId);
        if (pattern == nullptr)
            continue;

        auto noteEndTick = resolvedEndTick;
        while (loopSpan > 0 && noteEndTick < activeNote.absoluteStartTick)
            noteEndTick += loopSpan;

        const auto relativeEndTick = juce::jmax(activeNote.patternNoteStartTick + kMinSequenceSnapTicks,
                                                noteEndTick - activeNote.sectionStartTick);

        for (auto& note : pattern->notes)
        {
            if (note.pitch != activeNote.pitch || note.startTick != activeNote.patternNoteStartTick)
                continue;

            note.durationTick = juce::jmax(kMinSequenceSnapTicks, relativeEndTick - note.startTick);
            pattern->lengthTicks = juce::jmax(pattern->lengthTicks, note.startTick + note.durationTick);
            break;
        }

        for (auto& section : updatedProject.midiSections)
        {
            if (section.trackIndex != activeNote.trackIndex
                || section.patternId != activeNote.patternId
                || section.startTick != activeNote.sectionStartTick)
                continue;

            section.lengthTicks = juce::jmax(section.lengthTicks, pattern->lengthTicks);
            break;
        }

        changedTracks.addIfNotAlreadyThere(activeNote.trackIndex);
        if (preferredTrackIndex < 0)
            preferredTrackIndex = activeNote.trackIndex;
    }

    activeRealtimeRecordedNotes.clear();
    updatedProject.recalculateTimeFields();
    commitRealtimeRecordedProjectState(std::move(updatedProject), changedTracks, preferredTrackIndex, -1);
}

void StudioShellComponent::selectAllNotesFromMenu()
{
    juce::String statusText;
    if (selectAllFromFocusedEditor(&statusText) && statusText.isNotEmpty())
        statusLabel.setText(statusText, juce::dontSendNotification);
}

bool StudioShellComponent::selectAllFromFocusedEditor(juce::String* statusTextOut)
{
    const auto setStatus = [statusTextOut] (const juce::String& statusText)
    {
        if (statusTextOut != nullptr)
            *statusTextOut = statusText;
    };

    if (floatingArrangementOverview != nullptr
        && arrangementWindow != nullptr
        && arrangementWindow->isVisible()
        && floatingArrangementOverview->hasKeyboardFocus(true)
        && floatingArrangementOverview->selectAllSections())
    {
        setStatus("Selected all patterns.");
        return true;
    }

    if (panelsWindowContent != nullptr
        && panelsWindow != nullptr
        && panelsWindow->isVisible()
        && panelsWindowContent->hasArrangementKeyboardFocus()
        && panelsWindowContent->selectAllSections())
    {
        setStatus("Selected all patterns.");
        return true;
    }

    if (arrangementOverview != nullptr
        && arrangementOverview->hasKeyboardFocus(true)
        && arrangementOverview->selectAllSections())
    {
        setStatus("Selected all patterns.");
        return true;
    }

    if (floatingPianoRollWorkspace != nullptr
        && pianoRollWindow != nullptr
        && pianoRollWindow->isVisible()
        && floatingPianoRollWorkspace->hasKeyboardFocus(true)
        && floatingPianoRollWorkspace->selectAllNotes())
    {
        setStatus("Selected all notes.");
        return true;
    }

    if (panelsWindowContent != nullptr
        && panelsWindow != nullptr
        && panelsWindow->isVisible()
        && panelsWindowContent->hasPianoRollKeyboardFocus()
        && panelsWindowContent->selectAllNotes())
    {
        setStatus("Selected all notes.");
        return true;
    }

    if (pianoRoll != nullptr
        && pianoRoll->hasKeyboardFocus(true)
        && pianoRoll->selectAllNotes())
    {
        setStatus("Selected all notes.");
        return true;
    }

    if (floatingPianoRollWorkspace != nullptr
        && pianoRollWindow != nullptr
        && pianoRollWindow->isVisible()
        && floatingPianoRollWorkspace->selectAllNotes())
    {
        setStatus("Selected all notes.");
        return true;
    }

    if (panelsWindowContent != nullptr
        && panelsWindow != nullptr
        && panelsWindow->isVisible()
        && panelsWindowContent->selectAllNotes())
    {
        setStatus("Selected all notes.");
        return true;
    }

    if (pianoRoll != nullptr && pianoRoll->selectAllNotes())
    {
        setStatus("Selected all notes.");
        return true;
    }

    return false;
}

void StudioShellComponent::copyNotesFromMenu()
{
    if (pianoRoll != nullptr && pianoRoll->copySelected())
        statusLabel.setText("Copied selected notes.", juce::dontSendNotification);
}

void StudioShellComponent::cutNotesFromMenu()
{
    if (pianoRoll != nullptr && pianoRoll->cutSelected())
        statusLabel.setText("Cut selected notes.", juce::dontSendNotification);
}

void StudioShellComponent::deleteNotesFromMenu()
{
    if (pianoRoll != nullptr && pianoRoll->deleteSelected())
        statusLabel.setText("Deleted selected notes.", juce::dontSendNotification);
}

void StudioShellComponent::duplicateNotesFromMenu()
{
    if (pianoRoll != nullptr && pianoRoll->duplicateSelectedByGrid())
        statusLabel.setText("Duplicated selected notes.", juce::dontSendNotification);
}

void StudioShellComponent::pasteNotesFromMenu()
{
    if (pianoRoll != nullptr && pianoRoll->pasteClipboard())
        statusLabel.setText("Pasted notes.", juce::dontSendNotification);
}

void StudioShellComponent::focusArrangementPanel()
{
    if (arrangementWindow != nullptr)
    {
        setArrangementWindowVisible(true);
        if (floatingArrangementOverview != nullptr)
            floatingArrangementOverview->grabKeyboardFocus();
        return;
    }

    if (panelsWindowContent != nullptr)
    {
        setPanelsWindowVisible(true);
        panelsWindowContent->showArrangementTab();
        return;
    }

    if (arrangementOverview != nullptr)
        arrangementOverview->grabKeyboardFocus();
}

void StudioShellComponent::focusAutomationPanel()
{
    if (automationWindow != nullptr)
    {
        setAutomationWindowVisible(true);
        if (floatingAutomationEditor != nullptr)
            floatingAutomationEditor->grabKeyboardFocus();
        return;
    }

    if (panelsWindowContent != nullptr)
    {
        setPanelsWindowVisible(true);
        panelsWindowContent->showAutomationTab();
        return;
    }

    if (automationEditor != nullptr)
        automationEditor->grabKeyboardFocus();
}

void StudioShellComponent::focusSamplesPanel()
{
    if (samplesWindow != nullptr)
    {
        setSamplesWindowVisible(true);
        if (floatingSampleWorkspace != nullptr)
            floatingSampleWorkspace->focusLibrary();
        return;
    }

    if (panelsWindowContent != nullptr)
    {
        setPanelsWindowVisible(true);
        panelsWindowContent->showSamplesTab();
        return;
    }

    sampleAssetList.grabKeyboardFocus();
}

void StudioShellComponent::focusTracksPanel()
{
    if (tracksWindow != nullptr)
    {
        setTracksWindowVisible(true);
        if (floatingTracksWorkspace != nullptr)
            floatingTracksWorkspace->focusTrackList();
        return;
    }

    trackTable.grabKeyboardFocus();
}

void StudioShellComponent::ensureModulationMatrixWindowCreated()
{
    if (modulationMatrixWindow != nullptr)
        return;

    const auto displayArea = juce::Desktop::getInstance().getDisplays().getPrimaryDisplay() != nullptr
        ? juce::Desktop::getInstance().getDisplays().getPrimaryDisplay()->userArea
        : juce::Rectangle<int>(80, 80, 1600, 900);
    const auto attachVirtualPianoShortcutHandler = [this] (FloatingPanelWindow* window)
    {
        if (window == nullptr)
            return;

        window->onKeyPressed = [this] (const juce::KeyPress& key)
        {
            return keyPressed(key);
        };
    };

    modulationMatrixWindow = std::make_unique<FloatingPanelWindow>("Modulation Matrix");
    attachVirtualPianoShortcutHandler(modulationMatrixWindow.get());
    modulationMatrixWindow->onClosePressed = [this]
    {
        modulationMatrixWindowVisible = false;
        persistWindowVisibilityState();
    };

    auto* newFloatingModulationMatrix = new ModulationMatrixWindowComponent([this] () -> const ProjectState& { return documentState.project; },
                                                                            [this] (const juce::String& reference)
                                                                            {
                                                                                addInstrumentTrackFromReference(reference);
                                                                            },
                                                                            [this] (const juce::String& reference, int inputTrackIndex)
                                                                            {
                                                                                addSharedEffectBusFromReference(reference, inputTrackIndex);
                                                                            },
                                                                            [this] (const juce::String& busId, const juce::String& reference)
                                                                            {
                                                                                replaceSharedEffectBusReference(busId, reference);
                                                                            },
                                                                            [this] (const juce::String& busId)
                                                                            {
                                                                                removeSharedEffectBus(busId);
                                                                            },
                                                                            [this] (int trackIndex, const juce::String& targetId)
                                                                            {
                                                                                routeTrackToTarget(trackIndex, targetId);
                                                                            },
                                                                            [this] (int trackIndex)
                                                                            {
                                                                                openTrackRackEditor(trackIndex);
                                                                            },
                                                                            [this] (const juce::String& busId)
                                                                            {
                                                                                openSharedEffectBusEditor(busId);
                                                                            },
                                                                            [this] (const juce::String& busId)
                                                                            {
                                                                                clearSharedEffectBusOutputTargets(busId);
                                                                            },
                                                                            [this] (const juce::String& busId,
                                                                                     const juce::String& targetId,
                                                                                     bool enabled)
                                                                            {
                                                                                setSharedEffectBusOutputTargetEnabled(busId, targetId, enabled);
                                                                            });
    floatingModulationMatrix = newFloatingModulationMatrix;
    modulationMatrixWindow->setContentOwned(newFloatingModulationMatrix, true);
    modulationMatrixWindow->setBounds(displayArea.getX() + 120,
                                      displayArea.getY() + 132,
                                      juce::jmin(1240, displayArea.getWidth() - 120),
                                      juce::jmin(760, displayArea.getHeight() - 160));
    modulationMatrixWindow->sendLookAndFeelChange();
    modulationMatrixWindow->setColour(juce::ResizableWindow::backgroundColourId,
                                      themeSpecForIndex(currentThemeIndex).mainBackground);
    if (auto* content = modulationMatrixWindow->getContentComponent())
    {
        content->sendLookAndFeelChange();
        refreshExplicitUiFontsInComponentTree(*content, 1.0f);
        applyThemeToComponentTree(*content, themeSpecForIndex(currentThemeIndex));
    }
    modulationMatrixWindow->setVisible(false);
}

void StudioShellComponent::focusModulationMatrixPanel()
{
    setModulationMatrixWindowVisible(true);
}

void StudioShellComponent::focusRackBrowserPanel()
{
    if (rackBrowserWindow != nullptr)
    {
        setRackBrowserWindowVisible(true);
        if (floatingRackBrowser != nullptr)
        {
            floatingRackBrowser->selectAssignedRack();
            floatingRackBrowser->focusRackList();
        }
        return;
    }
}

void StudioShellComponent::focusRenderManagerPanel()
{
    if (renderManagerWindow != nullptr)
    {
        setRenderManagerWindowVisible(true);
        if (floatingRenderManager != nullptr)
            floatingRenderManager->focusTrackList();
        return;
    }
}

void StudioShellComponent::focusAudioPanel()
{
    if (audioWindow != nullptr)
    {
        setAudioWindowVisible(true);
        if (floatingAudioWorkspace != nullptr)
            floatingAudioWorkspace->focusWorkspace();
        return;
    }

    showAudioSettingsWindow();
}

void StudioShellComponent::focusMixerPanel()
{
    if (mixerWindow != nullptr)
    {
        setMixerWindowVisible(true);
        if (floatingMixerComponent != nullptr)
            floatingMixerComponent->grabKeyboardFocus();
        return;
    }

    if (mixerComponent != nullptr)
        mixerComponent->grabKeyboardFocus();
}

void StudioShellComponent::focusPianoRollPanel()
{
    if (pianoRollWindow != nullptr)
    {
        setPianoRollWindowVisible(true);
        if (floatingPianoRollWorkspace != nullptr)
            floatingPianoRollWorkspace->focusEditor();
        return;
    }

    if (panelsWindowContent != nullptr)
    {
        setPanelsWindowVisible(true);
        panelsWindowContent->showPianoRollTab();
        return;
    }

    if (pianoRoll != nullptr)
        pianoRoll->grabKeyboardFocus();
}

void StudioShellComponent::focusVirtualPianoPanel()
{
    if (virtualPianoWindow == nullptr)
        return;

    setVirtualPianoWindowVisible(true);
    if (virtualPianoWindowContent != nullptr)
        virtualPianoWindowContent->focusKeyboard();
}

void StudioShellComponent::showAudioSettingsWindow()
{
    if (audioSettingsWindow == nullptr)
        return;

    audioSettingsWindow->setVisible(true);
    audioSettingsWindow->toFront(true);
    refreshAudioSettingsFromHost(true);
}

void StudioShellComponent::showVstFolderManagerWindow()
{
    if (vstFolderManagerWindow == nullptr)
        return;

    if (vstFolderManagerPanel != nullptr)
        vstFolderManagerPanel->refreshFromModel();

    vstFolderManagerWindow->setVisible(true);
    vstFolderManagerWindow->toFront(false);
}

void StudioShellComponent::refreshAudioSettingsFromHost(bool forceCreate)
{
    if (audioSettingsPanel == nullptr)
        return;
    if (!forceCreate && (audioSettingsWindow == nullptr || !audioSettingsWindow->isVisible()))
        return;

    audioSettingsPanel->applyMidiInputSnapshot(juce::MidiInput::getAvailableDevices(),
                                               preferredMidiInputIdentifier);

    const auto createResult = nativeVstHost.ensureCreated();
    if (createResult.failed())
    {
        audioSettingsPanel->setStatusMessage(createResult.getErrorMessage(), true);
        return;
    }

    NativeVstHostSession::AudioDeviceSnapshot snapshot;
    const auto snapshotResult = nativeVstHost.queryAudioDeviceSnapshot(snapshot);
    if (snapshotResult.failed())
    {
        audioSettingsPanel->setStatusMessage(snapshotResult.getErrorMessage(), true);
        return;
    }

    audioSettingsPanel->applyAudioDeviceSnapshot(snapshot);
}

juce::Result StudioShellComponent::applyAudioDeviceSettings(const juce::String& audioDeviceType,
                                                            const juce::String& audioDeviceName,
                                                            int sampleRate,
                                                            int bufferSize)
{
    const auto createResult = nativeVstHost.ensureCreated();
    if (createResult.failed())
        return createResult;

    if (rackPreviewRunning || projectPreviewRunning)
        stopRackPreview();

    const auto result = nativeVstHost.configureAudioDevice(audioDeviceType,
                                                           audioDeviceName,
                                                           static_cast<double>(sampleRate),
                                                           bufferSize);
    if (result.failed())
    {
        refreshAudioSettingsFromHost(true);
        return result;
    }

    refreshAudioSettingsFromHost(true);

    audioEngineStateValid = false;
    audioEngineStateDirty = true;
    statusLabel.setText("Updated native audio device settings.", juce::dontSendNotification);
    refreshFloatingWindows();
    return juce::Result::ok();
}

juce::Result StudioShellComponent::setPreferredMidiInputDevice(const juce::String& deviceIdentifier)
{
    const auto requestedIdentifier = deviceIdentifier.trim();
    const auto availableDevices = juce::MidiInput::getAvailableDevices();

    if (requestedIdentifier != kMidiInputSelectionDisabled && requestedIdentifier.isNotEmpty())
    {
        bool found = false;
        for (const auto& device : availableDevices)
        {
            if (device.identifier == requestedIdentifier)
            {
                found = true;
                break;
            }
        }

        if (!found)
            return juce::Result::fail("The selected MIDI input device is unavailable.");
    }

    preferredMidiInputIdentifier = requestedIdentifier;
    refreshMidiInputDevices();
    persistSessionState();

    statusLabel.setText(describeMidiInputSelection(preferredMidiInputIdentifier,
                                                   availableDevices,
                                                   activeMidiInputNames),
                        juce::dontSendNotification);
    if (audioSettingsPanel != nullptr)
        audioSettingsPanel->applyMidiInputSnapshot(availableDevices, preferredMidiInputIdentifier);

    return juce::Result::ok();
}

void StudioShellComponent::setTransportWindowVisible(bool shouldBeVisible)
{
    transportWindowVisible = shouldBeVisible;
    persistWindowVisibilityState();
    if (transportWindow == nullptr)
        return;

    transportWindow->setVisible(shouldBeVisible);
    if (shouldBeVisible)
    {
        refreshFloatingWindows();
        transportWindow->toFront(true);
    }
}

bool StudioShellComponent::isTransportWindowVisible() const noexcept
{
    return transportWindowVisible && transportWindow != nullptr && transportWindow->isVisible();
}

void StudioShellComponent::setMixerWindowVisible(bool shouldBeVisible)
{
    mixerWindowVisible = shouldBeVisible;
    persistWindowVisibilityState();
    if (mixerWindow == nullptr)
        return;

    mixerWindow->setVisible(shouldBeVisible);
    if (shouldBeVisible)
    {
        refreshFloatingWindows();
        mixerWindow->toFront(true);
    }
}

bool StudioShellComponent::isMixerWindowVisible() const noexcept
{
    return mixerWindowVisible && mixerWindow != nullptr && mixerWindow->isVisible();
}

void StudioShellComponent::setAudioWindowVisible(bool shouldBeVisible)
{
    audioWindowVisible = shouldBeVisible;
    persistWindowVisibilityState();
    if (audioWindow == nullptr)
        return;

    audioWindow->setVisible(shouldBeVisible);
    if (shouldBeVisible)
    {
        refreshFloatingWindows();
        audioWindow->toFront(true);
    }
}

bool StudioShellComponent::isAudioWindowVisible() const noexcept
{
    return audioWindowVisible && audioWindow != nullptr && audioWindow->isVisible();
}

void StudioShellComponent::setPanelsWindowVisible(bool shouldBeVisible)
{
    panelsWindowVisible = shouldBeVisible;
    persistWindowVisibilityState();
    if (panelsWindow == nullptr)
        return;

    panelsWindow->setVisible(shouldBeVisible);
    if (shouldBeVisible)
    {
        refreshFloatingWindows();
        panelsWindow->toFront(true);
    }
}

bool StudioShellComponent::isPanelsWindowVisible() const noexcept
{
    return panelsWindowVisible && panelsWindow != nullptr && panelsWindow->isVisible();
}

void StudioShellComponent::setTracksWindowVisible(bool shouldBeVisible)
{
    tracksWindowVisible = shouldBeVisible;
    persistWindowVisibilityState();
    if (tracksWindow == nullptr)
        return;

    tracksWindow->setVisible(shouldBeVisible);
    if (shouldBeVisible)
    {
        refreshFloatingWindows();
        tracksWindow->toFront(true);
    }
}

bool StudioShellComponent::isTracksWindowVisible() const noexcept
{
    return tracksWindowVisible && tracksWindow != nullptr && tracksWindow->isVisible();
}

void StudioShellComponent::setModulationMatrixWindowVisible(bool shouldBeVisible)
{
    modulationMatrixWindowVisible = shouldBeVisible;
    persistWindowVisibilityState();
    if (shouldBeVisible)
        ensureModulationMatrixWindowCreated();
    if (modulationMatrixWindow == nullptr)
        return;

    modulationMatrixWindow->setVisible(shouldBeVisible);
    if (shouldBeVisible)
    {
        refreshFloatingWindows();
        modulationMatrixWindow->toFront(true);
    }
}

bool StudioShellComponent::isModulationMatrixWindowVisible() const noexcept
{
    return modulationMatrixWindowVisible
        && modulationMatrixWindow != nullptr
        && modulationMatrixWindow->isVisible();
}

void StudioShellComponent::setRackBrowserWindowVisible(bool shouldBeVisible)
{
    rackBrowserWindowVisible = shouldBeVisible;
    persistWindowVisibilityState();
    if (rackBrowserWindow == nullptr)
        return;

    rackBrowserWindow->setVisible(shouldBeVisible);
    if (shouldBeVisible)
    {
        refreshFloatingWindows();
        rackBrowserWindow->toFront(true);
    }
}

bool StudioShellComponent::isRackBrowserWindowVisible() const noexcept
{
    return rackBrowserWindowVisible && rackBrowserWindow != nullptr && rackBrowserWindow->isVisible();
}

void StudioShellComponent::setRenderManagerWindowVisible(bool shouldBeVisible)
{
    renderManagerWindowVisible = shouldBeVisible;
    persistWindowVisibilityState();
    if (renderManagerWindow == nullptr)
        return;

    renderManagerWindow->setVisible(shouldBeVisible);
    if (shouldBeVisible)
    {
        refreshFloatingWindows();
        renderManagerWindow->toFront(true);
    }
}

bool StudioShellComponent::isRenderManagerWindowVisible() const noexcept
{
    return renderManagerWindowVisible && renderManagerWindow != nullptr && renderManagerWindow->isVisible();
}

void StudioShellComponent::setArrangementWindowVisible(bool shouldBeVisible)
{
    arrangementWindowVisible = shouldBeVisible;
    persistWindowVisibilityState();
    if (arrangementWindow == nullptr)
        return;

    arrangementWindow->setVisible(shouldBeVisible);
    if (shouldBeVisible)
    {
        refreshFloatingWindows();
        arrangementWindow->toFront(true);
    }
}

bool StudioShellComponent::isArrangementWindowVisible() const noexcept
{
    return arrangementWindowVisible && arrangementWindow != nullptr && arrangementWindow->isVisible();
}

void StudioShellComponent::setAutomationWindowVisible(bool shouldBeVisible)
{
    automationWindowVisible = shouldBeVisible;
    persistWindowVisibilityState();
    if (automationWindow == nullptr)
        return;

    automationWindow->setVisible(shouldBeVisible);
    if (shouldBeVisible)
    {
        refreshFloatingWindows();
        automationWindow->toFront(true);
    }
}

bool StudioShellComponent::isAutomationWindowVisible() const noexcept
{
    return automationWindowVisible && automationWindow != nullptr && automationWindow->isVisible();
}

void StudioShellComponent::setSamplesWindowVisible(bool shouldBeVisible)
{
    samplesWindowVisible = shouldBeVisible;
    persistWindowVisibilityState();
    if (samplesWindow == nullptr)
        return;

    samplesWindow->setVisible(shouldBeVisible);
    if (shouldBeVisible)
    {
        refreshFloatingWindows();
        samplesWindow->toFront(true);
    }
}

bool StudioShellComponent::isSamplesWindowVisible() const noexcept
{
    return samplesWindowVisible && samplesWindow != nullptr && samplesWindow->isVisible();
}

void StudioShellComponent::setPianoRollWindowVisible(bool shouldBeVisible)
{
    pianoRollWindowVisible = shouldBeVisible;
    persistWindowVisibilityState();
    if (pianoRollWindow == nullptr)
        return;

    pianoRollWindow->setVisible(shouldBeVisible);
    if (shouldBeVisible)
    {
        refreshFloatingWindows();
        pianoRollWindow->toFront(true);
    }
}

bool StudioShellComponent::isPianoRollWindowVisible() const noexcept
{
    return pianoRollWindowVisible && pianoRollWindow != nullptr && pianoRollWindow->isVisible();
}

void StudioShellComponent::setVirtualPianoWindowVisible(bool shouldBeVisible)
{
    virtualPianoWindowVisible = shouldBeVisible;
    persistWindowVisibilityState();
    if (virtualPianoWindow == nullptr)
        return;

    virtualPianoWindow->setVisible(shouldBeVisible);
    if (shouldBeVisible)
    {
        virtualPianoWindow->toFront(true);
        if (virtualPianoWindowContent != nullptr)
            virtualPianoWindowContent->focusKeyboard();
    }
}

bool StudioShellComponent::isVirtualPianoWindowVisible() const noexcept
{
    return virtualPianoWindowVisible && virtualPianoWindow != nullptr && virtualPianoWindow->isVisible();
}

void StudioShellComponent::setActivityLogWindowVisible(bool shouldBeVisible)
{
    activityLogWindowVisible = shouldBeVisible;
    persistWindowVisibilityState();
    if (activityLogWindow == nullptr)
        return;

    activityLogWindow->setVisible(shouldBeVisible);
    if (shouldBeVisible)
    {
        refreshFloatingWindows();
        activityLogWindow->toFront(true);
    }
}

bool StudioShellComponent::isActivityLogWindowVisible() const noexcept
{
    return activityLogWindowVisible && activityLogWindow != nullptr && activityLogWindow->isVisible();
}

void StudioShellComponent::createNewProject()
{
    if (rackPreviewRunning && nativeVstHost.isReady())
        nativeVstHost.stopAudioEngine();
    if (projectPreviewRunning && nativeVstHost.isReady())
        nativeVstHost.stopAudioEngine();
    closeAllRackEditorSessions();
    resetRackHostTracking();
    rackPreviewRunning = false;
    projectPreviewRunning = false;
    playbackUiTickCounter = 0;
    pendingLiveRackParameterEngineSyncTrack = -1;
    audioEngineStateValid = false;
    audioEngineStateDirty = true;

    documentState = makeDefaultProjectFile();
    const auto defaultTemplateId = currentDefaultTemplateIdentifier();
    if (defaultTemplateId != kBuiltInDefaultTemplateId)
    {
        ProjectFileData templateProject;
        const auto templateResult = loadProjectFile(juce::File(defaultTemplateId), templateProject);
        if (templateResult.wasOk())
            documentState = std::move(templateProject);
        else if (windowStateSettings != nullptr)
        {
            windowStateSettings->setValue("project_default_template_id", kBuiltInDefaultTemplateId);
            windowStateSettings->saveIfNeeded();
        }
    }

    syncBundledRackCatalogInProject();
    currentProjectFile = juce::File();
    clearDirty();
    undoManager.clearUndoHistory();
    refreshUi();
    trackTable.selectRow(0);
    scheduleSelectedTrackRackPreviewWarmup(0);
    statusLabel.setText("Created new native project from template.", juce::dontSendNotification);
    appendActivityLog("Project", "Created new native project from template.");
}

void StudioShellComponent::promptOpenProject()
{
    activeFileChooser = std::make_unique<juce::FileChooser>("Open Mutagen project",
                                                            currentProjectFile.existsAsFile() ? currentProjectFile : juce::File(),
                                                            kProjectFileWildcard);
    activeFileChooser->launchAsync(juce::FileBrowserComponent::openMode | juce::FileBrowserComponent::canSelectFiles,
                                   [safeThis = juce::Component::SafePointer<StudioShellComponent>(this)](const juce::FileChooser& chooser)
                                   {
                                       const auto selected = chooser.getResult();
                                       if (safeThis == nullptr)
                                           return;

                                       safeThis->activeFileChooser.reset();
                                       if (selected == juce::File())
                                           return;

                                       const auto result = safeThis->openProjectFile(selected);
                                       if (result.failed())
                                       {
                                           juce::AlertWindow::showMessageBoxAsync(juce::AlertWindow::WarningIcon,
                                                                                  "Open Project Failed",
                                                                                  result.getErrorMessage());
                                       }
                                   });
}

void StudioShellComponent::saveProject()
{
    if (currentProjectFile == juce::File())
    {
        saveProjectAs();
        return;
    }

    for (const auto& session : rackEditorSessions)
    {
        if (session == nullptr || !session->editorOpen)
            continue;

        NativeVstHostSession::RackParameterSnapshot rackSnapshot;
        if (nativeVstHost.queryAudioEngineTrackParameterSnapshot(session->trackIndex, rackSnapshot).wasOk())
        {
            syncTrackRackParametersFromValues(session->trackIndex,
                                              rackSnapshot.parameterValues,
                                              true);
        }
    }

    const auto result = saveProjectFile(currentProjectFile, documentState);
    if (result.failed())
    {
        juce::AlertWindow::showMessageBoxAsync(juce::AlertWindow::WarningIcon,
                                               "Save Project Failed",
                                               result.getErrorMessage());
        return;
    }

    clearDirty();
    refreshUi();
    statusLabel.setText("Saved project: " + currentProjectFile.getFileName(), juce::dontSendNotification);
    appendActivityLog("Project", "Saved project\n" + currentProjectFile.getFullPathName());
}

void StudioShellComponent::saveProjectAs()
{
    activeFileChooser = std::make_unique<juce::FileChooser>("Save Mutagen project",
                                                            suggestProjectFile(),
                                                            kProjectFileWildcard);
    activeFileChooser->launchAsync(juce::FileBrowserComponent::saveMode
                                       | juce::FileBrowserComponent::canSelectFiles
                                       | juce::FileBrowserComponent::warnAboutOverwriting,
                                   [safeThis = juce::Component::SafePointer<StudioShellComponent>(this)](const juce::FileChooser& chooser)
                                   {
                                       const auto selected = chooser.getResult();
                                       if (safeThis == nullptr)
                                           return;

                                       safeThis->activeFileChooser.reset();
                                       if (selected == juce::File())
                                           return;

                                       safeThis->currentProjectFile = safeThis->ensureProjectSuffix(selected);
                                       safeThis->saveProject();
                                   });
}

void StudioShellComponent::addTrack()
{
    auto updatedProject = documentState.project;
    TrackState track;
    track.name = "Track " + juce::String(static_cast<int>(updatedProject.tracks.size()) + 1);
    track.midiChannel = static_cast<int>(updatedProject.tracks.size()) % 16;
    track.followThemeTrackColour = true;
    track.themeColourSlot = static_cast<int>(updatedProject.tracks.size());
    track.colorHex = defaultTrackColour(track.themeColourSlot).toDisplayString(false);
    updatedProject.tracks.push_back(std::move(track));
    applyProjectStateEdit(updatedProject, "Add Track");
    trackTable.selectRow(static_cast<int>(documentState.project.tracks.size()) - 1);
    statusLabel.setText("Added native track.", juce::dontSendNotification);
}

void StudioShellComponent::addSampleTrack()
{
    auto updatedProject = documentState.project;
    TrackState track;
    track.name = "Track " + juce::String(static_cast<int>(updatedProject.tracks.size()) + 1);
    track.instrument = "Sample";
    track.instrumentMode = "Audio Clip";
    track.trackType = "sample";
    track.midiChannel = static_cast<int>(updatedProject.tracks.size()) % 16;
    track.followThemeTrackColour = true;
    track.themeColourSlot = static_cast<int>(updatedProject.tracks.size());
    track.colorHex = defaultTrackColour(track.themeColourSlot).toDisplayString(false);
    updatedProject.tracks.push_back(std::move(track));
    applyProjectStateEdit(updatedProject, "Add Sample Track");
    trackTable.selectRow(static_cast<int>(documentState.project.tracks.size()) - 1);
    statusLabel.setText("Added sample track.", juce::dontSendNotification);
}

void StudioShellComponent::addInstrumentTrackFromReference(const juce::String& reference)
{
    auto updatedProject = documentState.project;
    const auto* entry = findRackEntryByReference(updatedProject, reference, false);
    if (entry == nullptr || !entry->isInstrument)
        return;

    TrackState track;
    const auto nextTrackIndex = static_cast<int>(updatedProject.tracks.size());
    const auto entryLabel = rackEntryDisplayName(*entry);
    track.name = "Track " + juce::String(nextTrackIndex + 1);
    track.instrument = entryLabel.isNotEmpty() ? entryLabel : "Instrument";
    track.instrumentMode = "VSTI Rack";
    track.rackVst = entry->path.isNotEmpty() ? entry->path : entryLabel;
    track.synthProfile = "vst_instrument";
    track.midiProgram = defaultMidiProgramForInstrumentName(track.instrument);
    track.midiChannel = nextTrackIndex % 16;
    track.followThemeTrackColour = true;
    track.themeColourSlot = nextTrackIndex;
    track.colorHex = defaultTrackColour(track.themeColourSlot).toDisplayString(false);

    updatedProject.tracks.push_back(std::move(track));
    updatedProject.recalculateTimeFields();
    applyProjectStateEdit(updatedProject, "Add Instrument Track");
    trackTable.selectRow(static_cast<int>(documentState.project.tracks.size()) - 1);
    statusLabel.setText("Added instrument track: " + (entryLabel.isNotEmpty() ? entryLabel : "Instrument") + ".", juce::dontSendNotification);
}

void StudioShellComponent::addSharedEffectBusFromReference(const juce::String& reference, int inputTrackIndex)
{
    auto updatedProject = documentState.project;
    const auto* entry = findRackEntryByReference(updatedProject, reference, true);
    if (entry == nullptr || !entry->isEffect)
        return;

    SharedEffectBusState bus;
    bus.id = juce::Uuid().toString();
    bus.name = rackEntryDisplayName(*entry);
    bus.effect = entry->path.isNotEmpty() ? entry->path : bus.name;
    updatedProject.sharedFxBuses.push_back(bus);

    if (juce::isPositiveAndBelow(inputTrackIndex, static_cast<int>(updatedProject.tracks.size())))
        updatedProject.tracks[static_cast<size_t>(inputTrackIndex)].routingTarget = bus.id;

    updatedProject.recalculateTimeFields();
    applyProjectStateEdit(updatedProject,
                          juce::isPositiveAndBelow(inputTrackIndex, static_cast<int>(updatedProject.tracks.size()))
                              ? "Insert Shared Effect"
                              : "Add Shared Effect Bus");
    statusLabel.setText("Added shared effect: " + (bus.name.isNotEmpty() ? bus.name : "FX Bus") + ".", juce::dontSendNotification);
}

void StudioShellComponent::replaceSharedEffectBusReference(const juce::String& busId, const juce::String& reference)
{
    auto updatedProject = documentState.project;
    const auto* entry = findRackEntryByReference(updatedProject, reference, true);
    if (entry == nullptr || !entry->isEffect)
        return;

    const auto trimmedBusId = busId.trim();
    if (trimmedBusId.isEmpty())
        return;

    bool changed = false;
    for (auto& bus : updatedProject.sharedFxBuses)
    {
        if (!bus.id.equalsIgnoreCase(trimmedBusId))
            continue;

        bus.effect = entry->path.isNotEmpty() ? entry->path : rackEntryDisplayName(*entry);
        bus.name = rackEntryDisplayName(*entry);
        bus.parameters.clear();
        bus.statePath.clear();
        bus.bypassed = false;
        changed = true;
        break;
    }

    if (!changed)
        return;

    updatedProject.recalculateTimeFields();
    applyProjectStateEdit(updatedProject, "Replace Shared Effect");
    statusLabel.setText("Updated shared effect bus.", juce::dontSendNotification);
}

void StudioShellComponent::removeSharedEffectBus(const juce::String& busId)
{
    auto updatedProject = documentState.project;
    const auto trimmedBusId = busId.trim();
    if (trimmedBusId.isEmpty())
        return;

    const auto originalSize = updatedProject.sharedFxBuses.size();
    updatedProject.sharedFxBuses.erase(std::remove_if(updatedProject.sharedFxBuses.begin(),
                                                      updatedProject.sharedFxBuses.end(),
                                                      [trimmedBusId] (const SharedEffectBusState& bus)
                                                      {
                                                          return bus.id.equalsIgnoreCase(trimmedBusId);
                                                      }),
                                       updatedProject.sharedFxBuses.end());
    if (updatedProject.sharedFxBuses.size() == originalSize)
        return;

    for (auto& track : updatedProject.tracks)
    {
        if (track.routingTarget.equalsIgnoreCase(trimmedBusId))
            track.routingTarget = "master";
    }
    for (auto& bus : updatedProject.sharedFxBuses)
    {
        bus.outputTargets.removeString(trimmedBusId);
        if (bus.outputTargets.isEmpty())
            bus.outputTargets.add("master");
    }

    updatedProject.recalculateTimeFields();
    applyProjectStateEdit(updatedProject, "Remove Shared Effect");
    statusLabel.setText("Removed shared effect bus.", juce::dontSendNotification);
}

void StudioShellComponent::routeTrackToTarget(int trackIndex, const juce::String& targetId)
{
    auto updatedProject = documentState.project;
    if (!juce::isPositiveAndBelow(trackIndex, static_cast<int>(updatedProject.tracks.size())))
        return;

    auto resolvedTarget = targetId.trim();
    if (resolvedTarget.isEmpty() || resolvedTarget.equalsIgnoreCase("none"))
    {
        resolvedTarget = "none";
    }
    else if (resolvedTarget.equalsIgnoreCase("master"))
    {
        resolvedTarget = "master";
    }
    else
    {
        const auto busIt = std::find_if(updatedProject.sharedFxBuses.begin(),
                                        updatedProject.sharedFxBuses.end(),
                                        [resolvedTarget] (const SharedEffectBusState& bus)
                                        {
                                            return bus.id.equalsIgnoreCase(resolvedTarget);
                                        });
        if (busIt == updatedProject.sharedFxBuses.end())
            return;
    }

    auto& track = updatedProject.tracks[static_cast<size_t>(trackIndex)];
    if (track.routingTarget.equalsIgnoreCase(resolvedTarget))
        return;

    track.routingTarget = resolvedTarget;
    updatedProject.recalculateTimeFields();
    applyProjectStateEdit(updatedProject, "Route Track");
    statusLabel.setText("Updated routing for " + track.name + ".", juce::dontSendNotification);
}

void StudioShellComponent::clearSharedEffectBusOutputTargets(const juce::String& busId)
{
    const auto trimmedBusId = busId.trim();
    if (trimmedBusId.isEmpty())
        return;

    auto updatedProject = documentState.project;
    const auto busIt = std::find_if(updatedProject.sharedFxBuses.begin(),
                                    updatedProject.sharedFxBuses.end(),
                                    [trimmedBusId] (const SharedEffectBusState& bus)
                                    {
                                        return bus.id.equalsIgnoreCase(trimmedBusId);
                                    });
    if (busIt == updatedProject.sharedFxBuses.end() || busIt->outputTargets.isEmpty())
        return;

    busIt->outputTargets.clear();
    updatedProject.recalculateTimeFields();
    applyProjectStateEdit(updatedProject, "Disconnect Shared Effect Outputs");
    statusLabel.setText("Disconnected shared effect outputs.", juce::dontSendNotification);
}

void StudioShellComponent::setSharedEffectBusOutputTargetEnabled(const juce::String& busId,
                                                                 const juce::String& targetId,
                                                                 bool enabled)
{
    const auto trimmedBusId = busId.trim();
    auto trimmedTargetId = targetId.trim();
    if (trimmedBusId.isEmpty())
        return;

    if (trimmedTargetId.isEmpty())
        return;

    auto updatedProject = documentState.project;
    const auto busIt = std::find_if(updatedProject.sharedFxBuses.begin(),
                                    updatedProject.sharedFxBuses.end(),
                                    [trimmedBusId] (const SharedEffectBusState& bus)
                                    {
                                        return bus.id.equalsIgnoreCase(trimmedBusId);
                                    });
    if (busIt == updatedProject.sharedFxBuses.end())
        return;

    if (!trimmedTargetId.equalsIgnoreCase("master"))
    {
        const auto targetBusIt = std::find_if(updatedProject.sharedFxBuses.begin(),
                                              updatedProject.sharedFxBuses.end(),
                                              [trimmedTargetId] (const SharedEffectBusState& bus)
                                              {
                                                  return bus.id.equalsIgnoreCase(trimmedTargetId);
                                              });
        if (targetBusIt == updatedProject.sharedFxBuses.end()
            || trimmedTargetId.equalsIgnoreCase(trimmedBusId))
        {
            return;
        }

        juce::StringArray visitedBusIds;
        std::function<bool(const juce::String&)> reachesSource = [&] (const juce::String& currentBusId) -> bool
        {
            if (currentBusId.equalsIgnoreCase(trimmedBusId))
                return true;
            if (visitedBusIds.contains(currentBusId, true))
                return false;
            visitedBusIds.add(currentBusId);

            const auto currentBusIt = std::find_if(updatedProject.sharedFxBuses.begin(),
                                                   updatedProject.sharedFxBuses.end(),
                                                   [currentBusId] (const SharedEffectBusState& bus)
                                                   {
                                                       return bus.id.equalsIgnoreCase(currentBusId);
                                                   });
            if (currentBusIt == updatedProject.sharedFxBuses.end())
                return false;

            for (const auto& nextTarget : currentBusIt->outputTargets)
            {
                const auto trimmedNextTarget = nextTarget.trim();
                if (trimmedNextTarget.isEmpty() || trimmedNextTarget.equalsIgnoreCase("master"))
                    continue;
                if (reachesSource(trimmedNextTarget))
                    return true;
            }

            return false;
        };

        if (enabled && reachesSource(trimmedTargetId))
        {
            statusLabel.setText("Routing blocked: effect chains cannot loop back into themselves.",
                                juce::dontSendNotification);
            return;
        }
    }

    auto& bus = *busIt;
    const auto alreadyEnabled = bus.outputTargets.contains(trimmedTargetId, true);
    if (enabled == alreadyEnabled)
        return;

    if (enabled)
    {
        bus.outputTargets.addIfNotAlreadyThere(trimmedTargetId);
    }
    else
    {
        bus.outputTargets.removeString(trimmedTargetId);
    }

    updatedProject.recalculateTimeFields();
    applyProjectStateEdit(updatedProject, "Update Shared Effect Routing");
    statusLabel.setText("Updated routing for shared effect bus.", juce::dontSendNotification);
}

void StudioShellComponent::duplicateSelectedTrack()
{
    const auto selected = getSelectedTrackIndex();
    if (selected < 0)
        return;

    auto updatedProject = documentState.project;
    auto copy = updatedProject.tracks[static_cast<size_t>(selected)];
    copy.name = copy.name + " Copy";
    if (copy.followThemeTrackColour)
    {
        copy.themeColourSlot = selected + 1;
        copy.colorHex = defaultTrackColour(copy.themeColourSlot).toDisplayString(false);
    }
    else if (copy.colorHex.trim().isEmpty())
    {
        copy.colorHex = defaultTrackColour(selected + 1).toDisplayString(false);
    }
    updatedProject.tracks.insert(updatedProject.tracks.begin() + selected + 1, std::move(copy));

    for (auto& section : updatedProject.midiSections)
    {
        if (section.trackIndex > selected)
            ++section.trackIndex;
    }
    for (auto& clip : updatedProject.sampleClips)
    {
        if (clip.trackIndex > selected)
            ++clip.trackIndex;
    }

    std::vector<MidiSection> duplicatedSections;
    for (const auto& section : documentState.project.midiSections)
    {
        if (section.trackIndex != selected)
            continue;

        auto duplicateSection = section;
        duplicateSection.trackIndex = selected + 1;
        duplicatedSections.push_back(std::move(duplicateSection));
    }

    updatedProject.midiSections.insert(updatedProject.midiSections.end(),
                                       duplicatedSections.begin(),
                                       duplicatedSections.end());

    std::vector<SampleClip> duplicatedClips;
    for (const auto& clip : documentState.project.sampleClips)
    {
        if (clip.trackIndex != selected)
            continue;

        auto duplicateClip = clip;
        duplicateClip.trackIndex = selected + 1;
        duplicatedClips.push_back(std::move(duplicateClip));
    }

    updatedProject.sampleClips.insert(updatedProject.sampleClips.end(),
                                      duplicatedClips.begin(),
                                      duplicatedClips.end());
    applyProjectStateEdit(updatedProject, "Duplicate Track");
    trackTable.selectRow(selected + 1);
    statusLabel.setText("Duplicated track.", juce::dontSendNotification);
}

void StudioShellComponent::removeSelectedTrack()
{
    const auto selected = getSelectedTrackIndex();
    if (selected < 0)
        return;

    auto updatedProject = documentState.project;
    updatedProject.tracks.erase(updatedProject.tracks.begin() + selected);
    if (updatedProject.tracks.empty())
        updatedProject.tracks.push_back(makeFallbackDefaultTrack());

    updatedProject.midiSections.erase(std::remove_if(updatedProject.midiSections.begin(),
                                                     updatedProject.midiSections.end(),
                                                     [selected] (const MidiSection& section)
                                                     {
                                                         return section.trackIndex == selected;
                                                     }),
                                     updatedProject.midiSections.end());
    updatedProject.sampleClips.erase(std::remove_if(updatedProject.sampleClips.begin(),
                                                    updatedProject.sampleClips.end(),
                                                    [selected] (const SampleClip& clip)
                                                    {
                                                        return clip.trackIndex == selected;
                                                    }),
                                     updatedProject.sampleClips.end());
    for (auto& section : updatedProject.midiSections)
    {
        if (section.trackIndex > selected)
            --section.trackIndex;
    }
    for (auto& clip : updatedProject.sampleClips)
    {
        if (clip.trackIndex > selected)
            --clip.trackIndex;
    }

    applyProjectStateEdit(updatedProject, "Remove Track");
    trackTable.selectRow(juce::jlimit(0, static_cast<int>(documentState.project.tracks.size()) - 1, selected));
    statusLabel.setText("Removed track.", juce::dontSendNotification);
}

void StudioShellComponent::handleTempoChanged()
{
    setTransportTempo(juce::roundToInt(tempoSlider.getValue()));
}

void StudioShellComponent::handleTimeSignatureChanged()
{
    const auto numerator = timeSignatureNumeratorBox.getSelectedId() > 0
        ? timeSignatureNumeratorBox.getSelectedId()
        : documentState.project.timeSigNumerator;
    const auto denominator = timeSignatureDenominatorBox.getSelectedId() > 0
        ? timeSignatureDenominatorBox.getSelectedId()
        : documentState.project.timeSigDenominator;

    auto updatedProject = documentState.project;
    updatedProject.timeSigNumerator = normaliseTimeSignatureNumerator(numerator);
    updatedProject.timeSigDenominator = normaliseTimeSignatureDenominator(denominator);

    if (updatedProject.timeSigNumerator == documentState.project.timeSigNumerator
        && updatedProject.timeSigDenominator == documentState.project.timeSigDenominator)
    {
        return;
    }

    updatedProject.recalculateTimeFields();
    applyProjectStateEdit(updatedProject, "Change Time Signature");
    statusLabel.setText("Updated time signature to " + timeSignatureDisplayName(updatedProject) + ".",
                        juce::dontSendNotification);
}

void StudioShellComponent::handlePatternBarsChanged()
{
    const auto tickLength = normaliseSequenceTickLength(patternBarsBox.getSelectedId(),
                                                        ticksPerBar(documentState.project));
    auto updatedProject = documentState.project;
    updatedProject.defaultPatternTicks = tickLength;
    if (juce::isPositiveAndBelow(selectedMidiSectionIndex, static_cast<int>(updatedProject.midiSections.size())))
    {
        const auto patternId = updatedProject.midiSections[static_cast<size_t>(selectedMidiSectionIndex)].patternId;
        if (auto* pattern = findMidiPattern(updatedProject, patternId))
        {
            if (pattern->lengthTicks != tickLength)
                pattern->lengthTicks = tickLength;
        }
    }

    if (fingerprint(updatedProject) == fingerprint(documentState.project))
        return;

    updatedProject.recalculateTimeFields();
    applyProjectStateEdit(updatedProject, "Change Pattern Size");
}

void StudioShellComponent::handleKeyQuantizeChanged()
{
    const auto* option = findKeyQuantizeOptionById(keyQuantizeBox.getSelectedId());
    if (option == nullptr)
        return;

    auto updatedProject = documentState.project;
    updatedProject.keyQuantizeRoot = option->root;
    updatedProject.keyQuantizeScale = option->scaleId;

    if (fingerprint(updatedProject) == fingerprint(documentState.project))
        return;

    updatedProject.recalculateTimeFields();
    applyProjectStateEdit(updatedProject, "Change Key Quantize");
}

void StudioShellComponent::handleArrangementSnapChanged()
{
    const auto tickLength = normaliseSequenceTickLength(arrangementSnapBox.getSelectedId(),
                                                        ticksPerBar(documentState.project));
    auto updatedProject = documentState.project;
    updatedProject.arrangementSnapTicks = tickLength;

    if (fingerprint(updatedProject) == fingerprint(documentState.project))
        return;

    updatedProject.recalculateTimeFields();
    applyProjectStateEdit(updatedProject, "Change Sequencer Snap");
}

void StudioShellComponent::handleArrangementZoomChanged()
{
    arrangementZoomPixelsPerBar = static_cast<float>(arrangementZoomSlider.getValue());
    applyEditorViewScaleState();
}

void StudioShellComponent::handleArrangementLaneHeightChanged()
{
    arrangementLaneHeightPixels = static_cast<float>(arrangementLaneHeightSlider.getValue());
    applyEditorViewScaleState();
}

void StudioShellComponent::handlePianoRollZoomChanged()
{
    pianoRollZoomPixelsPerBeat = static_cast<float>(pianoRollZoomSlider.getValue());
    applyEditorViewScaleState();
}

void StudioShellComponent::handlePianoRollRowHeightChanged()
{
    pianoRollRowHeightPixels = static_cast<float>(pianoRollRowHeightSlider.getValue());
    applyEditorViewScaleState();
}

void StudioShellComponent::promptImportJson()
{
    activeFileChooser = std::make_unique<juce::FileChooser>("Import Mutagen JSON Project",
                                                            juce::File(),
                                                            kJsonProjectWildcard);
    activeFileChooser->launchAsync(juce::FileBrowserComponent::openMode | juce::FileBrowserComponent::canSelectFiles,
                                   [safeThis = juce::Component::SafePointer<StudioShellComponent>(this)](const juce::FileChooser& chooser)
                                   {
                                       const auto selected = chooser.getResult();
                                       if (safeThis == nullptr)
                                           return;

                                       safeThis->activeFileChooser.reset();
                                       if (selected == juce::File())
                                           return;

                                       const auto result = safeThis->openProjectFile(selected);
                                       if (result.failed())
                                       {
                                           juce::AlertWindow::showMessageBoxAsync(juce::AlertWindow::WarningIcon,
                                                                                  "Import JSON Failed",
                                                                                  result.getErrorMessage());
                                           return;
                                       }

                                       safeThis->statusLabel.setText("Imported JSON project: " + selected.getFileName(),
                                                                     juce::dontSendNotification);
                                   });
}

void StudioShellComponent::promptImportMidi()
{
    activeFileChooser = std::make_unique<juce::FileChooser>("Import MIDI",
                                                            juce::File(),
                                                            kMidiFileWildcard);
    activeFileChooser->launchAsync(juce::FileBrowserComponent::openMode | juce::FileBrowserComponent::canSelectFiles,
                                   [safeThis = juce::Component::SafePointer<StudioShellComponent>(this)](const juce::FileChooser& chooser)
                                   {
                                       const auto selected = chooser.getResult();
                                       if (safeThis == nullptr)
                                           return;

                                       safeThis->activeFileChooser.reset();
                                       if (selected == juce::File())
                                           return;

                                       auto* dialog = new juce::AlertWindow("Import MIDI",
                                                                            "Choose how Mutagen should assign instruments for the imported MIDI tracks.",
                                                                            juce::AlertWindow::NoIcon);
                                       dialog->addComboBox("mode",
                                                           { "General MIDI (As Is)",
                                                             "Try Native VST Instruments (fallback to GM)" },
                                                           "Instrument Assignment");
                                       dialog->addTextBlock("General MIDI keeps the file as standard GM tracks with GM program changes.\n\n"
                                                            "Native VST mode tries to map GM sounds like piano, strings, organ, flute, violin, bass and drums "
                                                            "onto your native rack instruments. If a track does not match anything suitable, it stays General MIDI.");
                                       if (auto* modeBox = dialog->getComboBoxComponent("mode"))
                                           modeBox->setSelectedId(kMidiImportModeNativeRackId, juce::dontSendNotification);
                                       dialog->setSize(560, 260);
                                       dialog->addButton("Import", 1, juce::KeyPress(juce::KeyPress::returnKey));
                                       dialog->addButton("Cancel", 0, juce::KeyPress(juce::KeyPress::escapeKey));

                                       auto safeDialog = juce::Component::SafePointer<juce::AlertWindow>(dialog);
                                       dialog->enterModalState(true,
                                                               juce::ModalCallbackFunction::create([safeThis, safeDialog, selected] (int result)
                                                               {
                                                                   if (safeThis == nullptr || safeDialog == nullptr || result != 1)
                                                                       return;

                                                                   const auto importMode = midiImportAssignmentModeFromComboId(
                                                                       safeDialog->getComboBoxComponent("mode") != nullptr
                                                                           ? safeDialog->getComboBoxComponent("mode")->getSelectedId()
                                                                           : kMidiImportModeGeneralMidiId);

                                                                   auto importedProject = safeThis->documentState.project;
                                                                   const auto importResult = importMidiFileToProject(selected, importedProject);
                                                                   if (importResult.failed())
                                                                   {
                                                                       juce::AlertWindow::showMessageBoxAsync(juce::AlertWindow::WarningIcon,
                                                                                                              "Import MIDI Failed",
                                                                                                              importResult.getErrorMessage());
                                                                       return;
                                                                   }

                                                                   int nativeAssignedCount = 0;
                                                                   int gmFallbackCount = 0;
                                                                   if (importMode == MidiImportAssignmentMode::tryNativeRack)
                                                                   {
                                                                       for (auto& track : importedProject.tracks)
                                                                       {
                                                                           if (!track.trackType.trim().equalsIgnoreCase("instrument"))
                                                                               continue;

                                                                           const auto suggestedIndex = suggestedNativeInstrumentIndexForTrack(importedProject, track);
                                                                           if (suggestedIndex < 0)
                                                                           {
                                                                               track.instrumentMode = "General MIDI";
                                                                               track.rackVst.clear();
                                                                               ++gmFallbackCount;
                                                                               continue;
                                                                           }

                                                                           const auto& entry = importedProject.vstRack[static_cast<size_t>(suggestedIndex)];
                                                                           const auto entryLabel = entry.name.isNotEmpty() ? entry.name
                                                                               : (entry.pluginName.isNotEmpty() ? entry.pluginName
                                                                                                                : juce::File(entry.path).getFileNameWithoutExtension());

                                                                           track.instrumentMode = "VSTI Rack";
                                                                           track.rackVst = entryLabel.isNotEmpty() ? entryLabel : entry.path.trim();
                                                                           track.vstiStatePath.clear();
                                                                           track.vstiStateBase64.clear();
                                                                           ++nativeAssignedCount;
                                                                       }
                                                                   }

                                                                   safeThis->applyProjectStateEdit(importedProject, "Import MIDI");
                                                                   safeThis->trackTable.selectRow(0);

                                                                   auto statusText = "Imported MIDI: "
                                                                       + selected.getFileName()
                                                                       + "   "
                                                                       + juce::String(importedProject.bpm)
                                                                       + " BPM   "
                                                                       + timeSignatureDisplayName(importedProject);
                                                                   if (importMode == MidiImportAssignmentMode::tryNativeRack)
                                                                   {
                                                                       statusText << "   "
                                                                                  << juce::String(nativeAssignedCount)
                                                                                  << " native / "
                                                                                  << juce::String(gmFallbackCount)
                                                                                  << " GM";
                                                                   }
                                                                   else
                                                                   {
                                                                       statusText << "   General MIDI";
                                                                   }

                                                                   safeThis->statusLabel.setText(statusText, juce::dontSendNotification);
                                                               }),
                                                               true);
                                   });
}

void StudioShellComponent::promptExportJson()
{
    auto defaultTarget = juce::File::getSpecialLocation(juce::File::userDocumentsDirectory)
        .getChildFile("ai-music-studio-native.json");
    if (currentProjectFile != juce::File())
        defaultTarget = currentProjectFile.withFileExtension(".json");

    activeFileChooser = std::make_unique<juce::FileChooser>("Export Project as JSON",
                                                            defaultTarget,
                                                            kJsonProjectWildcard);
    activeFileChooser->launchAsync(juce::FileBrowserComponent::saveMode
                                       | juce::FileBrowserComponent::canSelectFiles
                                       | juce::FileBrowserComponent::warnAboutOverwriting,
                                   [safeThis = juce::Component::SafePointer<StudioShellComponent>(this)](const juce::FileChooser& chooser)
                                   {
                                       const auto selected = chooser.getResult();
                                       if (safeThis == nullptr)
                                           return;

                                       safeThis->activeFileChooser.reset();
                                       if (selected == juce::File())
                                           return;

                                       auto target = selected;
                                       if (!target.hasFileExtension(".json"))
                                           target = target.withFileExtension(".json");

                                       const auto result = saveProjectFile(target, safeThis->documentState);
                                       if (result.failed())
                                       {
                                           juce::AlertWindow::showMessageBoxAsync(juce::AlertWindow::WarningIcon,
                                                                                  "Export JSON Failed",
                                                                                  result.getErrorMessage());
                                           return;
                                       }

                                       safeThis->statusLabel.setText("Exported JSON: " + target.getFileName(),
                                                                     juce::dontSendNotification);
                                   });
}

void StudioShellComponent::promptExportMidi()
{
    activeFileChooser = std::make_unique<juce::FileChooser>("Export MIDI",
                                                            juce::File::getSpecialLocation(juce::File::userDocumentsDirectory).getChildFile("ai-music-studio-native.mid"),
                                                            kMidiFileWildcard);
    activeFileChooser->launchAsync(juce::FileBrowserComponent::saveMode
                                       | juce::FileBrowserComponent::canSelectFiles
                                       | juce::FileBrowserComponent::warnAboutOverwriting,
                                   [safeThis = juce::Component::SafePointer<StudioShellComponent>(this)](const juce::FileChooser& chooser)
                                   {
                                       const auto selected = chooser.getResult();
                                       if (safeThis == nullptr)
                                           return;

                                       safeThis->activeFileChooser.reset();
                                       if (selected == juce::File())
                                           return;

                                       auto target = selected;
                                       if (!target.hasFileExtension(".mid") && !target.hasFileExtension(".midi"))
                                           target = target.withFileExtension(".mid");

                                       const auto result = exportProjectToMidiFile(target, safeThis->documentState.project);
                                       if (result.failed())
                                       {
                                           juce::AlertWindow::showMessageBoxAsync(juce::AlertWindow::WarningIcon,
                                                                                  "Export MIDI Failed",
                                                                                  result.getErrorMessage());
                                           return;
                                       }

                                       safeThis->statusLabel.setText("Exported MIDI: " + target.getFileName(), juce::dontSendNotification);
                                   });
}

void StudioShellComponent::promptExportMp3()
{
    if (documentState.project.rightLocatorSec <= documentState.project.leftLocatorSec)
    {
        juce::AlertWindow::showMessageBoxAsync(juce::AlertWindow::WarningIcon,
                                               "Invalid Locators",
                                               "Right locator must be greater than left locator for MP3 export.");
        return;
    }

    if (rackPreviewRunning || projectPreviewRunning)
        stopRackPreview();

    activeFileChooser = std::make_unique<juce::FileChooser>("Export Sequence as MP3",
                                                            juce::File::getSpecialLocation(juce::File::userDocumentsDirectory)
                                                                .getChildFile("ai-music-studio-native-sequence.mp3"),
                                                            "*.mp3");
    activeFileChooser->launchAsync(juce::FileBrowserComponent::saveMode
                                       | juce::FileBrowserComponent::canSelectFiles
                                       | juce::FileBrowserComponent::warnAboutOverwriting,
                                   [safeThis = juce::Component::SafePointer<StudioShellComponent>(this)](const juce::FileChooser& chooser)
                                   {
                                       const auto selected = chooser.getResult();
                                       if (safeThis == nullptr)
                                           return;

                                       safeThis->activeFileChooser.reset();
                                       if (selected == juce::File())
                                           return;

                                       auto target = selected;
                                       if (!target.hasFileExtension(".mp3"))
                                           target = target.withFileExtension(".mp3");

                                       AudioExportSummary summary;
                                       const auto result = exportProjectRangeToAudioFile(target,
                                                                                         safeThis->documentState.project,
                                                                                         safeThis->nativeVstHost,
                                                                                         summary);
                                       if (result.failed())
                                       {
                                           juce::AlertWindow::showMessageBoxAsync(juce::AlertWindow::WarningIcon,
                                                                                  "Export MP3 Failed",
                                                                                  result.getErrorMessage());
                                           return;
                                       }

                                       auto statusMessage = "Exported MP3: " + target.getFileName()
                                           + "  (" + juce::String(summary.renderedInstrumentTrackCount) + " instrument track"
                                           + (summary.renderedInstrumentTrackCount == 1 ? "" : "s")
                                           + ", " + juce::String(summary.mixedAudioTrackCount) + " audio track"
                                           + (summary.mixedAudioTrackCount == 1 ? "" : "s")
                                           + ")";
                                       if (summary.warnings.size() > 0)
                                           statusMessage << "  Warnings: " << juce::String(summary.warnings.size());
                                       safeThis->statusLabel.setText(statusMessage, juce::dontSendNotification);

                                       if (summary.warnings.size() > 0)
                                       {
                                           juce::StringArray previewWarnings;
                                           const auto previewCount = juce::jmin(6, summary.warnings.size());
                                           for (int index = 0; index < previewCount; ++index)
                                               previewWarnings.add(summary.warnings[index]);
                                           if (summary.warnings.size() > previewCount)
                                               previewWarnings.add("...");

                                           juce::AlertWindow::showMessageBoxAsync(juce::AlertWindow::InfoIcon,
                                                                                  "Export MP3 Completed With Warnings",
                                                                                  previewWarnings.joinIntoString("\n"));
                                       }
                                   });
}

void StudioShellComponent::promptExportWav()
{
    if (documentState.project.rightLocatorSec <= documentState.project.leftLocatorSec)
    {
        juce::AlertWindow::showMessageBoxAsync(juce::AlertWindow::WarningIcon,
                                               "Invalid Locators",
                                               "Right locator must be greater than left locator for WAV export.");
        return;
    }

    if (rackPreviewRunning || projectPreviewRunning)
        stopRackPreview();

    activeFileChooser = std::make_unique<juce::FileChooser>("Export Sequence as WAV",
                                                            juce::File::getSpecialLocation(juce::File::userDocumentsDirectory)
                                                                .getChildFile("ai-music-studio-native-sequence.wav"),
                                                            "*.wav");
    activeFileChooser->launchAsync(juce::FileBrowserComponent::saveMode
                                       | juce::FileBrowserComponent::canSelectFiles
                                       | juce::FileBrowserComponent::warnAboutOverwriting,
                                   [safeThis = juce::Component::SafePointer<StudioShellComponent>(this)](const juce::FileChooser& chooser)
                                   {
                                       const auto selected = chooser.getResult();
                                       if (safeThis == nullptr)
                                           return;

                                       safeThis->activeFileChooser.reset();
                                       if (selected == juce::File())
                                           return;

                                       auto target = selected;
                                       if (!target.hasFileExtension(".wav"))
                                           target = target.withFileExtension(".wav");

                                       AudioExportSummary summary;
                                       const auto result = exportProjectRangeToWavFile(target,
                                                                                       safeThis->documentState.project,
                                                                                       safeThis->nativeVstHost,
                                                                                       summary);
                                       if (result.failed())
                                       {
                                           juce::AlertWindow::showMessageBoxAsync(juce::AlertWindow::WarningIcon,
                                                                                  "Export WAV Failed",
                                                                                  result.getErrorMessage());
                                           return;
                                       }

                                       auto statusMessage = "Exported WAV: " + target.getFileName()
                                           + "  (" + juce::String(summary.renderedInstrumentTrackCount) + " instrument track"
                                           + (summary.renderedInstrumentTrackCount == 1 ? "" : "s")
                                           + ", " + juce::String(summary.mixedAudioTrackCount) + " audio track"
                                           + (summary.mixedAudioTrackCount == 1 ? "" : "s")
                                           + ")";
                                       if (summary.warnings.size() > 0)
                                           statusMessage << "  Warnings: " << juce::String(summary.warnings.size());
                                       safeThis->statusLabel.setText(statusMessage, juce::dontSendNotification);

                                       if (summary.warnings.size() > 0)
                                       {
                                           juce::StringArray previewWarnings;
                                           const auto previewCount = juce::jmin(6, summary.warnings.size());
                                           for (int index = 0; index < previewCount; ++index)
                                               previewWarnings.add(summary.warnings[index]);
                                           if (summary.warnings.size() > previewCount)
                                               previewWarnings.add("...");

                                           juce::AlertWindow::showMessageBoxAsync(juce::AlertWindow::InfoIcon,
                                                                                  "Export WAV Completed With Warnings",
                                                                                  previewWarnings.joinIntoString("\n"));
                                       }
                                   });
}

void StudioShellComponent::promptExportSelectedTrackMp3()
{
    const auto selected = getSelectedTrackIndex();
    if (!juce::isPositiveAndBelow(selected, static_cast<int>(documentState.project.tracks.size())))
    {
        juce::AlertWindow::showMessageBoxAsync(juce::AlertWindow::InfoIcon,
                                               "No Track Selected",
                                               "Select a track first before exporting a native track MP3.");
        return;
    }

    if (documentState.project.rightLocatorSec <= documentState.project.leftLocatorSec)
    {
        juce::AlertWindow::showMessageBoxAsync(juce::AlertWindow::WarningIcon,
                                               "Invalid Locators",
                                               "Right locator must be greater than left locator for MP3 export.");
        return;
    }

    if (rackPreviewRunning || projectPreviewRunning)
        stopRackPreview();

    const auto& track = documentState.project.tracks[static_cast<size_t>(selected)];
    const auto suggestedTrackName = track.name.trim().isNotEmpty() ? track.name.trim() : ("Track " + juce::String(selected + 1));
    const auto defaultStemName = juce::File::createLegalFileName(suggestedTrackName.replaceCharacter(' ', '_'));
    const auto defaultTarget = juce::File::getSpecialLocation(juce::File::userDocumentsDirectory)
        .getChildFile((defaultStemName.isNotEmpty() ? defaultStemName : "track") + ".mp3");

    activeFileChooser = std::make_unique<juce::FileChooser>("Export Selected Track as MP3",
                                                            defaultTarget,
                                                            "*.mp3");
    activeFileChooser->launchAsync(juce::FileBrowserComponent::saveMode
                                       | juce::FileBrowserComponent::canSelectFiles
                                       | juce::FileBrowserComponent::warnAboutOverwriting,
                                   [safeThis = juce::Component::SafePointer<StudioShellComponent>(this),
                                    trackIndex = selected](const juce::FileChooser& chooser)
                                   {
                                       const auto selectedFile = chooser.getResult();
                                       if (safeThis == nullptr)
                                           return;

                                       safeThis->activeFileChooser.reset();
                                       if (selectedFile == juce::File())
                                           return;

                                       if (!juce::isPositiveAndBelow(trackIndex, static_cast<int>(safeThis->documentState.project.tracks.size())))
                                           return;

                                       auto target = selectedFile;
                                       if (!target.hasFileExtension(".mp3"))
                                           target = target.withFileExtension(".mp3");

                                       AudioExportSummary summary;
                                       const auto result = exportTrackRangeToAudioFile(target,
                                                                                       safeThis->documentState.project,
                                                                                       trackIndex,
                                                                                       safeThis->nativeVstHost,
                                                                                       summary);
                                       if (result.failed())
                                       {
                                           juce::AlertWindow::showMessageBoxAsync(juce::AlertWindow::WarningIcon,
                                                                                  "Track Export Failed",
                                                                                  result.getErrorMessage());
                                           return;
                                       }

                                       auto statusMessage = "Exported track MP3: " + target.getFileName();
                                       if (summary.warnings.size() > 0)
                                           statusMessage << "  Warnings: " << juce::String(summary.warnings.size());
                                       safeThis->statusLabel.setText(statusMessage, juce::dontSendNotification);

                                       if (summary.warnings.size() > 0)
                                       {
                                           juce::StringArray previewWarnings;
                                           const auto previewCount = juce::jmin(6, summary.warnings.size());
                                           for (int index = 0; index < previewCount; ++index)
                                               previewWarnings.add(summary.warnings[index]);
                                           if (summary.warnings.size() > previewCount)
                                               previewWarnings.add("...");

                                           juce::AlertWindow::showMessageBoxAsync(juce::AlertWindow::InfoIcon,
                                                                                  "Track Export Completed With Warnings",
                                                                                  previewWarnings.joinIntoString("\n"));
                                       }
                                   });
}

void StudioShellComponent::promptExportSelectedTrackWav()
{
    const auto selected = getSelectedTrackIndex();
    if (!juce::isPositiveAndBelow(selected, static_cast<int>(documentState.project.tracks.size())))
    {
        juce::AlertWindow::showMessageBoxAsync(juce::AlertWindow::InfoIcon,
                                               "No Track Selected",
                                               "Select a track first before exporting a native track WAV.");
        return;
    }

    if (documentState.project.rightLocatorSec <= documentState.project.leftLocatorSec)
    {
        juce::AlertWindow::showMessageBoxAsync(juce::AlertWindow::WarningIcon,
                                               "Invalid Locators",
                                               "Right locator must be greater than left locator for WAV export.");
        return;
    }

    if (rackPreviewRunning || projectPreviewRunning)
        stopRackPreview();

    const auto& track = documentState.project.tracks[static_cast<size_t>(selected)];
    const auto suggestedTrackName = track.name.trim().isNotEmpty() ? track.name.trim() : ("Track " + juce::String(selected + 1));
    const auto defaultStemName = juce::File::createLegalFileName(suggestedTrackName.replaceCharacter(' ', '_'));
    const auto defaultTarget = juce::File::getSpecialLocation(juce::File::userDocumentsDirectory)
        .getChildFile((defaultStemName.isNotEmpty() ? defaultStemName : "track") + ".wav");

    activeFileChooser = std::make_unique<juce::FileChooser>("Export Selected Track as WAV",
                                                            defaultTarget,
                                                            "*.wav");
    activeFileChooser->launchAsync(juce::FileBrowserComponent::saveMode
                                       | juce::FileBrowserComponent::canSelectFiles
                                       | juce::FileBrowserComponent::warnAboutOverwriting,
                                   [safeThis = juce::Component::SafePointer<StudioShellComponent>(this),
                                    trackIndex = selected](const juce::FileChooser& chooser)
                                   {
                                       const auto selectedFile = chooser.getResult();
                                       if (safeThis == nullptr)
                                           return;

                                       safeThis->activeFileChooser.reset();
                                       if (selectedFile == juce::File())
                                           return;

                                       if (!juce::isPositiveAndBelow(trackIndex, static_cast<int>(safeThis->documentState.project.tracks.size())))
                                           return;

                                       auto target = selectedFile;
                                       if (!target.hasFileExtension(".wav"))
                                           target = target.withFileExtension(".wav");

                                       AudioExportSummary summary;
                                       const auto result = exportTrackRangeToWavFile(target,
                                                                                     safeThis->documentState.project,
                                                                                     trackIndex,
                                                                                     safeThis->nativeVstHost,
                                                                                     summary);
                                       if (result.failed())
                                       {
                                           juce::AlertWindow::showMessageBoxAsync(juce::AlertWindow::WarningIcon,
                                                                                  "Track Export Failed",
                                                                                  result.getErrorMessage());
                                           return;
                                       }

                                       auto updatedTrack = safeThis->documentState.project.tracks[static_cast<size_t>(trackIndex)];
                                       updatedTrack.renderedAudioPath = target.getFullPathName();
                                       safeThis->applyTrackStateEdit(trackIndex, updatedTrack, "Export Track WAV");

                                       auto statusMessage = "Exported track WAV: " + target.getFileName();
                                       if (summary.warnings.size() > 0)
                                           statusMessage << "  Warnings: " << juce::String(summary.warnings.size());
                                       safeThis->statusLabel.setText(statusMessage, juce::dontSendNotification);

                                       if (summary.warnings.size() > 0)
                                       {
                                           juce::StringArray previewWarnings;
                                           const auto previewCount = juce::jmin(6, summary.warnings.size());
                                           for (int index = 0; index < previewCount; ++index)
                                               previewWarnings.add(summary.warnings[index]);
                                           if (summary.warnings.size() > previewCount)
                                               previewWarnings.add("...");

                                           juce::AlertWindow::showMessageBoxAsync(juce::AlertWindow::InfoIcon,
                                                                                  "Track Export Completed With Warnings",
                                                                                  previewWarnings.joinIntoString("\n"));
                                       }
                                   });
}

void StudioShellComponent::promptExportProjectStems()
{
    if (documentState.project.rightLocatorSec <= documentState.project.leftLocatorSec)
    {
        juce::AlertWindow::showMessageBoxAsync(juce::AlertWindow::WarningIcon,
                                               "Invalid Locators",
                                               "Right locator must be greater than left locator for stem export.");
        return;
    }

    if (rackPreviewRunning || projectPreviewRunning)
        stopRackPreview();

    activeFileChooser = std::make_unique<juce::FileChooser>("Export Native Stems",
                                                            juce::File::getSpecialLocation(juce::File::userDocumentsDirectory)
                                                                .getChildFile("ai-music-studio-native-stems"));
    activeFileChooser->launchAsync(juce::FileBrowserComponent::openMode
                                       | juce::FileBrowserComponent::canSelectDirectories,
                                   [safeThis = juce::Component::SafePointer<StudioShellComponent>(this)](const juce::FileChooser& chooser)
                                   {
                                       const auto selectedFolder = chooser.getResult();
                                       if (safeThis == nullptr)
                                           return;

                                       safeThis->activeFileChooser.reset();
                                       if (selectedFolder == juce::File())
                                           return;

                                       AudioExportBatchSummary summary;
                                       const auto result = exportProjectStemsToFolder(selectedFolder,
                                                                                      safeThis->documentState.project,
                                                                                      safeThis->nativeVstHost,
                                                                                      summary);
                                       if (result.failed())
                                       {
                                           juce::AlertWindow::showMessageBoxAsync(juce::AlertWindow::WarningIcon,
                                                                                  "Stem Export Failed",
                                                                                  result.getErrorMessage());
                                           return;
                                       }

                                       auto updatedProject = safeThis->documentState.project;
                                       for (const auto& exportedFile : summary.exportedFiles)
                                       {
                                           if (!juce::isPositiveAndBelow(exportedFile.trackIndex, static_cast<int>(updatedProject.tracks.size())))
                                               continue;
                                           updatedProject.tracks[static_cast<size_t>(exportedFile.trackIndex)].renderedAudioPath = exportedFile.filePath;
                                       }
                                       safeThis->applyProjectStateEdit(updatedProject, "Export Stems");

                                       auto statusMessage = "Exported " + juce::String(summary.exportedTrackCount)
                                           + " stem" + (summary.exportedTrackCount == 1 ? "" : "s")
                                           + " to " + selectedFolder.getFullPathName();
                                       if (summary.warnings.size() > 0)
                                           statusMessage << "  Warnings: " << juce::String(summary.warnings.size());
                                       safeThis->statusLabel.setText(statusMessage, juce::dontSendNotification);

                                       if (summary.warnings.size() > 0)
                                       {
                                           juce::StringArray previewWarnings;
                                           const auto previewCount = juce::jmin(8, summary.warnings.size());
                                           for (int index = 0; index < previewCount; ++index)
                                               previewWarnings.add(summary.warnings[index]);
                                           if (summary.warnings.size() > previewCount)
                                               previewWarnings.add("...");

                                           juce::AlertWindow::showMessageBoxAsync(juce::AlertWindow::InfoIcon,
                                                                                  "Stem Export Completed With Warnings",
                                                                                  previewWarnings.joinIntoString("\n"));
                                       }
                                   });
}

juce::Result StudioShellComponent::ensureSampleAssetForFile(const juce::File& file, int& outAssetIndex)
{
    outAssetIndex = -1;
    const auto targetPath = file.getFullPathName();
    for (int index = 0; index < static_cast<int>(documentState.project.sampleAssets.size()); ++index)
    {
        if (documentState.project.sampleAssets[static_cast<size_t>(index)].path.equalsIgnoreCase(targetPath))
        {
            outAssetIndex = index;
            setSelectedSampleAssetIndex(index);
            return juce::Result::ok();
        }
    }

    SampleAsset asset;
    const auto loadResult = loadSampleAssetFromFile(file, asset);
    if (loadResult.failed())
        return loadResult;

    auto updatedProject = documentState.project;
    updatedProject.sampleAssets.push_back(std::move(asset));
    updatedProject.samplePaths.addIfNotAlreadyThere(targetPath);
    const auto insertedIndex = static_cast<int>(updatedProject.sampleAssets.size()) - 1;
    applyProjectStateEdit(updatedProject, "Import Sample");
    outAssetIndex = insertedIndex;
    setSelectedSampleAssetIndex(insertedIndex);
    return juce::Result::ok();
}

juce::Result StudioShellComponent::placeSampleAssetOnTrackAtTick(int assetIndex,
                                                                 int trackIndex,
                                                                 int startTick,
                                                                 const juce::String& actionName,
                                                                 juce::String& outTrackName)
{
    if (!juce::isPositiveAndBelow(assetIndex, static_cast<int>(documentState.project.sampleAssets.size())))
        return juce::Result::fail("Select a sample from the native sample library first.");

    if (!juce::isPositiveAndBelow(trackIndex, static_cast<int>(documentState.project.tracks.size())))
        return juce::Result::fail("Choose a valid track before placing audio clips.");

    const auto& asset = documentState.project.sampleAssets[static_cast<size_t>(assetIndex)];
    auto updatedProject = documentState.project;
    SampleClip clip;
    clip.path = asset.path;
    clip.trackIndex = trackIndex;
    clip.startSec = tickToSeconds(updatedProject, juce::jmax(0, startTick));
    clip.durationSec = asset.durationSec;
    clip.sourceOffsetSec = 0.0;
    clip.sourceFileDurationSec = asset.durationSec;
    clip.sampleRate = asset.sampleRate;
    clip.waveformPreview = asset.waveformPreview;
    updatedProject.sampleClips.push_back(std::move(clip));
    outTrackName = updatedProject.tracks[static_cast<size_t>(trackIndex)].name;
    applyProjectStateEdit(updatedProject, actionName);
    return juce::Result::ok();
}

juce::Result StudioShellComponent::placeSampleFileOnTrackAtTick(const juce::File& file,
                                                                int trackIndex,
                                                                int startTick,
                                                                const juce::String& actionName,
                                                                juce::String& outTrackName)
{
    int assetIndex = -1;
    const auto importResult = ensureSampleAssetForFile(file, assetIndex);
    if (importResult.failed())
        return importResult;

    return placeSampleAssetOnTrackAtTick(assetIndex, trackIndex, startTick, actionName, outTrackName);
}

juce::Result StudioShellComponent::placeSampleAssetAtPlayhead(int assetIndex,
                                                              const juce::String& actionName,
                                                              juce::String& outTrackName)
{
    const auto trackIndex = findPreferredSampleTrackIndex();
    if (trackIndex < 0)
        return juce::Result::fail("Set a track type to sample before placing audio clips.");

    return placeSampleAssetOnTrackAtTick(assetIndex,
                                         trackIndex,
                                         documentState.project.playheadTick,
                                         actionName,
                                         outTrackName);
}

void StudioShellComponent::promptImportSample()
{
    activeFileChooser = std::make_unique<juce::FileChooser>("Import Sample",
                                                            juce::File(),
                                                            "*.wav;*.aiff;*.aif;*.flac;*.ogg;*.mp3");
    activeFileChooser->launchAsync(juce::FileBrowserComponent::openMode | juce::FileBrowserComponent::canSelectFiles,
                                   [safeThis = juce::Component::SafePointer<StudioShellComponent>(this)](const juce::FileChooser& chooser)
                                   {
                                       const auto selected = chooser.getResult();
                                       if (safeThis == nullptr)
                                           return;

                                       safeThis->activeFileChooser.reset();
                                       if (selected == juce::File())
                                           return;

                                       int assetIndex = -1;
                                       const auto result = safeThis->ensureSampleAssetForFile(selected, assetIndex);
                                       if (result.failed())
                                       {
                                           juce::AlertWindow::showMessageBoxAsync(juce::AlertWindow::WarningIcon,
                                                                                  "Import Sample Failed",
                                                                                  result.getErrorMessage());
                                           return;
                                       }

                                        safeThis->statusLabel.setText("Imported sample: " + selected.getFileName(), juce::dontSendNotification);
                                    });
}

void StudioShellComponent::separateSampleClipToStems(int clipIndex)
{
    if (stemSeparationBusy)
    {
        statusLabel.setText("Audio stem extraction is already running.", juce::dontSendNotification);
        return;
    }

    if (!juce::isPositiveAndBelow(clipIndex, static_cast<int>(documentState.project.sampleClips.size())))
    {
        juce::AlertWindow::showMessageBoxAsync(juce::AlertWindow::InfoIcon,
                                               "Extract Audio Stems",
                                               "Select an audio clip first.");
        return;
    }

    juce::StringArray demucsCommandPrefix;
    if (!tryDetectDemucsCommand(demucsCommandPrefix))
    {
        juce::AlertWindow::showMessageBoxAsync(juce::AlertWindow::InfoIcon,
                                               "Audio Stem Extraction Unavailable",
                                               "Audio stem extraction requires Demucs.\n\nInstall it with:\npy -m pip install demucs\n\nMore info: "
                                                   + juce::String(kDemucsInstallUrl));
        return;
    }

    const auto sourceClip = documentState.project.sampleClips[static_cast<size_t>(clipIndex)];
    const juce::File sourceFile(sourceClip.path);
    if (!sourceFile.existsAsFile())
    {
        juce::AlertWindow::showMessageBoxAsync(juce::AlertWindow::WarningIcon,
                                               "Extract Audio Stems Failed",
                                               "The source audio file for this clip could not be found.");
        return;
    }

    const auto sourceTrackName = juce::isPositiveAndBelow(sourceClip.trackIndex, static_cast<int>(documentState.project.tracks.size()))
        ? documentState.project.tracks[static_cast<size_t>(sourceClip.trackIndex)].name
        : juce::String("Track ") + juce::String(sourceClip.trackIndex + 1);

    stemSeparationBusy = true;
    refreshPollingTimerState();
    updateEditorState();
    statusLabel.setText("Extracting stems for " + sourceFile.getFileName() + "...", juce::dontSendNotification);
    appendActivityLog("Audio Stem Extraction",
                      "Started audio stem extraction\nSource: "
                          + sourceFile.getFullPathName()
                          + "\nTrack: "
                          + sourceTrackName);

    stemSeparationFuture = std::async(std::launch::async,
                                      [demucsCommandPrefix, sourceClip, sourceTrackName]() -> StemSeparationResult
                                      {
                                          StemSeparationResult result;
                                          result.sourceClip = sourceClip;
                                          result.sourceTrackName = sourceTrackName;

                                          const juce::File inputFile(sourceClip.path);
                                          if (!inputFile.existsAsFile())
                                          {
                                              result.errorMessage = "The source audio file could not be found.";
                                              return result;
                                          }

                                          auto outputRoot = nativeStemSeparationDirectory();
                                          if (!outputRoot.exists() && !outputRoot.createDirectory())
                                          {
                                              result.errorMessage = "Could not create the stem separation folder.";
                                              return result;
                                          }

                                          const auto baseName = juce::File::createLegalFileName(inputFile.getFileNameWithoutExtension());
                                          const auto jobFolderName = (baseName.isNotEmpty() ? baseName : juce::String("stems"))
                                              + "-"
                                              + juce::String::toHexString(static_cast<juce::int64>(juce::Time::getCurrentTime().toMilliseconds()));
                                          const auto jobFolder = outputRoot.getChildFile(jobFolderName);
                                          if (!jobFolder.createDirectory())
                                          {
                                              result.errorMessage = "Could not create the stem separation job folder.";
                                              return result;
                                          }

                                          juce::StringArray command(demucsCommandPrefix);
                                          command.add("-d");
                                          command.add("cpu");
                                          command.add("--out");
                                          command.add(jobFolder.getFullPathName());
                                          command.add(inputFile.getFullPathName());

                                          juce::ChildProcess process;
                                          if (!process.start(command))
                                          {
                                              result.errorMessage = "Could not launch Demucs. Install it with: py -m pip install demucs";
                                              return result;
                                          }

                                          process.waitForProcessToFinish(-1);
                                          const auto processOutput = process.readAllProcessOutput().trim();
                                          if (process.getExitCode() != 0)
                                          {
                                              result.errorMessage = "Demucs failed to separate the stems."
                                                  + (processOutput.isNotEmpty() ? "\n\n" + processOutput : juce::String());
                                              return result;
                                          }

                                          auto stemFiles = jobFolder.findChildFiles(juce::File::findFiles, true, "*.wav");
                                          if (stemFiles.isEmpty())
                                          {
                                              stemFiles = jobFolder.findChildFiles(juce::File::findFiles, true, "*.flac");
                                          }
                                          if (stemFiles.isEmpty())
                                          {
                                              result.errorMessage = "No stem audio files were produced.";
                                              return result;
                                          }

                                          std::sort(stemFiles.begin(),
                                                    stemFiles.end(),
                                                    [] (const juce::File& lhs, const juce::File& rhs)
                                                    {
                                                        const auto lhsName = stemDisplayName(lhs);
                                                        const auto rhsName = stemDisplayName(rhs);
                                                        const auto lhsRank = stemSortRank(lhsName);
                                                        const auto rhsRank = stemSortRank(rhsName);
                                                        if (lhsRank != rhsRank)
                                                            return lhsRank < rhsRank;
                                                        return lhsName.compareNatural(rhsName) < 0;
                                                    });

                                          for (const auto& stemFile : stemFiles)
                                          {
                                              SampleAsset asset;
                                              const auto loadResult = loadSampleAssetFromFile(stemFile, asset);
                                              if (loadResult.failed())
                                                  continue;

                                              StemSeparationStem stem;
                                              stem.name = stemDisplayName(stemFile);
                                              stem.asset = std::move(asset);
                                              result.stems.push_back(std::move(stem));
                                          }

                                          if (result.stems.empty())
                                          {
                                              result.errorMessage = "The separated stem files could not be loaded.";
                                              return result;
                                          }

                                          result.outputDirectory = jobFolder;
                                          result.success = true;
                                          return result;
                                      });
}

juce::String StudioShellComponent::defaultVstFolderPath() const
{
    const auto systemDirectory = defaultSystemVstDirectory();
    if (systemDirectory != juce::File())
        return systemDirectory.getFullPathName();

    const auto bundledDirectory = findBundledVstDirectory();
    return bundledDirectory != juce::File() ? bundledDirectory.getFullPathName() : juce::String();
}

juce::StringArray StudioShellComponent::userManagedVstFolderPaths() const
{
    juce::StringArray folders;
    const auto defaultFolder = defaultVstFolderPath().trim();

    for (const auto& rawFolder : documentState.project.vstiFolderPaths)
    {
        auto normalisedFolder = rawFolder.trim();
        if (normalisedFolder.isEmpty())
            continue;

        normalisedFolder = juce::File(normalisedFolder).getFullPathName();
        if (defaultFolder.isNotEmpty() && normalisedFolder.equalsIgnoreCase(defaultFolder))
            continue;

        bool alreadyAdded = false;
        for (const auto& existing : folders)
        {
            if (existing.equalsIgnoreCase(normalisedFolder))
            {
                alreadyAdded = true;
                break;
            }
        }

        if (!alreadyAdded)
            folders.add(normalisedFolder);
    }

    return folders;
}

void StudioShellComponent::promptAddUserVstFolder()
{
    activeFileChooser = std::make_unique<juce::FileChooser>("Select User VST Folder",
                                                            juce::File(defaultVstFolderPath()),
                                                            "*");
    activeFileChooser->launchAsync(juce::FileBrowserComponent::openMode | juce::FileBrowserComponent::canSelectDirectories,
                                   [safeThis = juce::Component::SafePointer<StudioShellComponent>(this)](const juce::FileChooser& chooser)
                                   {
                                       const auto selected = chooser.getResult();
                                       if (safeThis == nullptr)
                                           return;

                                       safeThis->activeFileChooser.reset();
                                       if (selected == juce::File())
                                           return;

                                       if (!selected.isDirectory())
                                       {
                                           juce::AlertWindow::showMessageBoxAsync(juce::AlertWindow::WarningIcon,
                                                                                  "Add VST Folder Failed",
                                                                                  "Choose a VST folder directory.");
                                           return;
                                       }

                                       const auto selectedPath = selected.getFullPathName();
                                       const auto defaultFolder = safeThis->defaultVstFolderPath().trim();
                                       if (defaultFolder.isNotEmpty() && selectedPath.equalsIgnoreCase(defaultFolder))
                                       {
                                           safeThis->statusLabel.setText("The default VST folder is already scanned automatically.",
                                                                         juce::dontSendNotification);
                                           return;
                                       }

                                       auto updatedProject = safeThis->documentState.project;
                                       bool alreadyTracked = false;
                                       for (const auto& existing : updatedProject.vstiFolderPaths)
                                       {
                                           if (juce::File(existing.trim()).getFullPathName().equalsIgnoreCase(selectedPath))
                                           {
                                               alreadyTracked = true;
                                               break;
                                           }
                                       }

                                       if (alreadyTracked)
                                       {
                                           safeThis->statusLabel.setText("User VST folder already added: " + selectedPath,
                                                                         juce::dontSendNotification);
                                           return;
                                       }

                                       updatedProject.vstiFolderPaths.add(selectedPath);
                                       safeThis->applyProjectStateEdit(updatedProject, "Add VST Folder");
                                       safeThis->refreshRackCatalog();
                                       safeThis->statusLabel.setText("Added user VST folder: " + selectedPath,
                                                                     juce::dontSendNotification);
                                   });
}

void StudioShellComponent::removeUserVstFolder(const juce::String& folderPath)
{
    const auto trimmedPath = folderPath.trim();
    if (trimmedPath.isEmpty())
        return;

    auto updatedProject = documentState.project;
    bool removed = false;
    for (int index = updatedProject.vstiFolderPaths.size(); --index >= 0;)
    {
        if (juce::File(updatedProject.vstiFolderPaths[index].trim()).getFullPathName().equalsIgnoreCase(trimmedPath))
        {
            updatedProject.vstiFolderPaths.remove(index);
            removed = true;
        }
    }

    if (!removed)
        return;

    applyProjectStateEdit(updatedProject, "Remove VST Folder");
    statusLabel.setText("Removed user VST folder: " + trimmedPath, juce::dontSendNotification);
}

void StudioShellComponent::promptImportRackPlugin()
{
    activeFileChooser = std::make_unique<juce::FileChooser>("Import Rack Plugin",
                                                            juce::File(),
                                                            "*.vst3;*.dll;*.so");
    activeFileChooser->launchAsync(juce::FileBrowserComponent::openMode
                                       | juce::FileBrowserComponent::canSelectFiles
                                       | juce::FileBrowserComponent::canSelectDirectories,
                                   [safeThis = juce::Component::SafePointer<StudioShellComponent>(this)](const juce::FileChooser& chooser)
                                   {
                                       const auto selected = chooser.getResult();
                                       if (safeThis == nullptr)
                                           return;

                                       safeThis->activeFileChooser.reset();
                                       if (selected == juce::File())
                                           return;

                                       const auto selectedPath = selected.getFullPathName();
                                       const auto hasSupportedExtension = selected.hasFileExtension(".vst3;.dll;.so");
                                       if (!hasSupportedExtension)
                                       {
                                           juce::AlertWindow::showMessageBoxAsync(juce::AlertWindow::WarningIcon,
                                                                                  "Import Rack Plugin Failed",
                                                                                  "Choose a `.vst3`, `.dll`, or `.so` plugin path.");
                                           return;
                                       }

                                       auto updatedProject = safeThis->documentState.project;
                                       updatedProject.vstiPaths.addIfNotAlreadyThere(selectedPath);
                                       updatedProject.vstiFolderPaths.addIfNotAlreadyThere(selected.getParentDirectory().getFullPathName());

                                       VstInstrument entry;
                                       entry = makeRackPluginEntry(selected);

                                       bool changed = false;
                                       if (const auto existingIndex = findRackInstrumentIndexByReference(updatedProject, selectedPath); existingIndex >= 0)
                                       {
                                           auto& existing = updatedProject.vstRack[static_cast<size_t>(existingIndex)];
                                           if (existing.path != entry.path
                                               || existing.name != entry.name
                                               || existing.pluginName != entry.pluginName
                                               || existing.isInstrument != entry.isInstrument
                                               || existing.isEffect != entry.isEffect
                                               || existing.category != entry.category
                                               || existing.hostSupported != entry.hostSupported
                                               || existing.hostError != entry.hostError)
                                           {
                                               existing.path = entry.path;
                                               existing.name = entry.name;
                                               existing.pluginName = entry.pluginName;
                                               existing.isInstrument = entry.isInstrument;
                                               existing.isEffect = entry.isEffect;
                                               existing.category = entry.category;
                                               existing.hostSupported = entry.hostSupported;
                                               existing.hostError = entry.hostError;
                                               changed = true;
                                           }
                                       }
                                       else
                                       {
                                           updatedProject.vstRack.push_back(entry);
                                           changed = true;
                                       }

                                       if (!changed)
                                       {
                                           safeThis->statusLabel.setText("Rack plugin already available: " + selected.getFileName(),
                                                                         juce::dontSendNotification);
                                           if (safeThis->floatingRackBrowser != nullptr)
                                               safeThis->floatingRackBrowser->selectRackByReference(selectedPath);
                                           return;
                                       }

                                       safeThis->applyProjectStateEdit(updatedProject, "Import Rack Plugin");
                                       if (safeThis->floatingRackBrowser != nullptr)
                                           safeThis->floatingRackBrowser->selectRackByReference(selectedPath);
                                       const juce::String pluginKind = entry.isEffect ? "FX" : (entry.isInstrument ? "instrument" : "plugin");
                                       safeThis->statusLabel.setText("Imported rack " + pluginKind + ": " + selected.getFileName(),
                                                                     juce::dontSendNotification);
                                   });
}

void StudioShellComponent::promptRelinkSelectedTrackRenderedAudio()
{
    const auto selected = getSelectedTrackIndex();
    if (!juce::isPositiveAndBelow(selected, static_cast<int>(documentState.project.tracks.size())))
        return;

    activeFileChooser = std::make_unique<juce::FileChooser>("Relink Rendered Audio",
                                                            juce::File(),
                                                            "*.wav;*.aiff;*.aif;*.flac;*.ogg;*.mp3");
    activeFileChooser->launchAsync(juce::FileBrowserComponent::openMode | juce::FileBrowserComponent::canSelectFiles,
                                   [safeThis = juce::Component::SafePointer<StudioShellComponent>(this),
                                    trackIndex = selected](const juce::FileChooser& chooser)
                                   {
                                       const auto selectedFile = chooser.getResult();
                                       if (safeThis == nullptr)
                                           return;

                                       safeThis->activeFileChooser.reset();
                                       if (selectedFile == juce::File())
                                           return;

                                       if (!juce::isPositiveAndBelow(trackIndex, static_cast<int>(safeThis->documentState.project.tracks.size())))
                                           return;

                                       auto updatedTrack = safeThis->documentState.project.tracks[static_cast<size_t>(trackIndex)];
                                       updatedTrack.renderedAudioPath = selectedFile.getFullPathName();
                                       safeThis->applyTrackStateEdit(trackIndex, updatedTrack, "Relink Rendered Audio");
                                       safeThis->statusLabel.setText("Relinked rendered audio for " + updatedTrack.name + ".",
                                                                     juce::dontSendNotification);
                                   });
}

void StudioShellComponent::placeSelectedSampleAtPlayhead()
{
    const auto assetRow = sampleAssetList.getSelectedRow();
    juce::String trackName;
    const auto result = placeSampleAssetAtPlayhead(assetRow, "Place Sample Clip", trackName);
    if (result.failed())
    {
        juce::AlertWindow::showMessageBoxAsync(juce::AlertWindow::InfoIcon,
                                               "Place Sample Failed",
                                               result.getErrorMessage());
        return;
    }

    statusLabel.setText("Placed sample clip on " + trackName + ".", juce::dontSendNotification);
}

void StudioShellComponent::importSelectedTrackRenderToSampleLibrary()
{
    const auto* track = getSelectedTrack();
    if (track == nullptr)
        return;

    const auto renderFile = juce::File(track->renderedAudioPath.trim());
    if (!renderFile.existsAsFile())
    {
        juce::AlertWindow::showMessageBoxAsync(juce::AlertWindow::InfoIcon,
                                               "Missing Render File",
                                               "Export or relink a rendered audio file for the selected track first.");
        return;
    }

    int assetIndex = -1;
    const auto result = ensureSampleAssetForFile(renderFile, assetIndex);
    if (result.failed())
    {
        juce::AlertWindow::showMessageBoxAsync(juce::AlertWindow::WarningIcon,
                                               "Import Render Failed",
                                               result.getErrorMessage());
        return;
    }

    statusLabel.setText("Added rendered audio for " + track->name + " to the native sample library.",
                        juce::dontSendNotification);
}

void StudioShellComponent::placeSelectedTrackRenderAtPlayhead()
{
    const auto* track = getSelectedTrack();
    if (track == nullptr)
        return;

    const auto renderFile = juce::File(track->renderedAudioPath.trim());
    if (!renderFile.existsAsFile())
    {
        juce::AlertWindow::showMessageBoxAsync(juce::AlertWindow::InfoIcon,
                                               "Missing Render File",
                                               "Export or relink a rendered audio file for the selected track first.");
        return;
    }

    int assetIndex = -1;
    auto result = ensureSampleAssetForFile(renderFile, assetIndex);
    if (result.failed())
    {
        juce::AlertWindow::showMessageBoxAsync(juce::AlertWindow::WarningIcon,
                                               "Place Render Failed",
                                               result.getErrorMessage());
        return;
    }

    juce::String trackName;
    result = placeSampleAssetAtPlayhead(assetIndex, "Place Render Clip", trackName);
    if (result.failed())
    {
        juce::AlertWindow::showMessageBoxAsync(juce::AlertWindow::InfoIcon,
                                               "Place Render Failed",
                                               result.getErrorMessage());
        return;
    }

    statusLabel.setText("Placed rendered audio on " + trackName + ".", juce::dontSendNotification);
}

void StudioShellComponent::showTemplateSettingsDialog()
{
    auto templateOptions = availableProjectTemplateOptions();
    const auto currentTemplateId = currentDefaultTemplateIdentifier();

    juce::StringArray templateLabels;
    int selectedTemplateIndex = 0;
    for (int index = 0; index < static_cast<int>(templateOptions.size()); ++index)
    {
        auto label = templateOptions[static_cast<size_t>(index)].name;
        if (templateOptions[static_cast<size_t>(index)].builtIn)
            label << " (Built-in)";

        templateLabels.add(label);
        if (templateOptions[static_cast<size_t>(index)].identifier == currentTemplateId)
            selectedTemplateIndex = index;
    }

    auto* dialog = new juce::AlertWindow("Project Templates",
                                         "Choose which template seeds new projects. The built-in default is one AI Piano track.",
                                         juce::AlertWindow::NoIcon);
    dialog->addComboBox("defaultTemplate", templateLabels, "Default template");
    if (auto* templateBox = dialog->getComboBoxComponent("defaultTemplate"))
        templateBox->setSelectedItemIndex(selectedTemplateIndex);
    dialog->addTextBlock("Templates folder:\n" + nativeTemplatesDirectory().getFullPathName());
    dialog->addButton("Set Default", 1, juce::KeyPress(juce::KeyPress::returnKey));
    dialog->addButton("Save Current As Template", 2);
    dialog->addButton("Import Project As Template", 3);
    dialog->addButton("Open Folder", 4);
    dialog->addButton("Close", 0, juce::KeyPress(juce::KeyPress::escapeKey));

    auto safeThis = juce::Component::SafePointer<StudioShellComponent>(this);
    auto safeDialog = juce::Component::SafePointer<juce::AlertWindow>(dialog);
    dialog->enterModalState(true,
                            juce::ModalCallbackFunction::create([safeThis, safeDialog, templateOptions = std::move(templateOptions)] (int result) mutable
                            {
                                if (safeThis == nullptr || safeDialog == nullptr || result == 0)
                                    return;

                                const auto templateRoot = nativeTemplatesDirectory();

                                if (result == 4)
                                {
                                    ignoreUnused(templateRoot.createDirectory());
                                    templateRoot.startAsProcess();
                                    safeThis->statusLabel.setText("Opened template folder.", juce::dontSendNotification);
                                    safeThis->appendActivityLog("Templates", "Opened template folder\n" + templateRoot.getFullPathName());
                                    return;
                                }

                                if (result == 2)
                                {
                                    ignoreUnused(templateRoot.createDirectory());
                                    const auto suggestedTemplate = ensureTemplateSuffix(templateRoot.getChildFile(suggestTemplateBaseName(safeThis->currentProjectFile,
                                                                                                                                     safeThis->documentState)));
                                    safeThis->activeFileChooser = std::make_unique<juce::FileChooser>("Save Project Template",
                                                                                                      suggestedTemplate,
                                                                                                      kProjectTemplateWildcard);
                                    safeThis->activeFileChooser->launchAsync(juce::FileBrowserComponent::saveMode
                                                                                 | juce::FileBrowserComponent::canSelectFiles
                                                                                 | juce::FileBrowserComponent::warnAboutOverwriting,
                                                                             [safeThis](const juce::FileChooser& chooser)
                                                                             {
                                                                                 if (safeThis == nullptr)
                                                                                     return;

                                                                                 const auto selected = chooser.getResult();
                                                                                 safeThis->activeFileChooser.reset();
                                                                                 if (selected == juce::File())
                                                                                     return;

                                                                                 auto templateProject = safeThis->documentState;
                                                                                 templateProject.savedAtUnix = juce::Time::getCurrentTime().toMilliseconds() / 1000;
                                                                                 const auto targetFile = ensureTemplateSuffix(selected);
                                                                                 const auto saveResult = saveProjectFile(targetFile, templateProject);
                                                                                 if (saveResult.failed())
                                                                                 {
                                                                                     juce::AlertWindow::showMessageBoxAsync(juce::AlertWindow::WarningIcon,
                                                                                                                            "Save Template Failed",
                                                                                                                            saveResult.getErrorMessage());
                                                                                     return;
                                                                                 }

                                                                                 safeThis->statusLabel.setText("Saved template: " + targetFile.getFileNameWithoutExtension(),
                                                                                                               juce::dontSendNotification);
                                                                                 safeThis->appendActivityLog("Templates",
                                                                                                             "Saved project template\n" + targetFile.getFullPathName());
                                                                             });
                                    return;
                                }

                                if (result == 3)
                                {
                                    ignoreUnused(templateRoot.createDirectory());
                                    safeThis->activeFileChooser = std::make_unique<juce::FileChooser>("Import Project As Template",
                                                                                                      safeThis->currentProjectFile.existsAsFile()
                                                                                                          ? safeThis->currentProjectFile
                                                                                                          : juce::File::getSpecialLocation(juce::File::userDocumentsDirectory),
                                                                                                      kProjectFileWildcard);
                                    safeThis->activeFileChooser->launchAsync(juce::FileBrowserComponent::openMode
                                                                                 | juce::FileBrowserComponent::canSelectFiles,
                                                                             [safeThis, templateRoot](const juce::FileChooser& chooser)
                                                                             {
                                                                                 if (safeThis == nullptr)
                                                                                     return;

                                                                                 const auto selected = chooser.getResult();
                                                                                 safeThis->activeFileChooser.reset();
                                                                                 if (selected == juce::File())
                                                                                     return;

                                                                                 ProjectFileData importedProject;
                                                                                 const auto loadResult = loadProjectFile(selected, importedProject);
                                                                                 if (loadResult.failed())
                                                                                 {
                                                                                     juce::AlertWindow::showMessageBoxAsync(juce::AlertWindow::WarningIcon,
                                                                                                                            "Import Template Failed",
                                                                                                                            loadResult.getErrorMessage());
                                                                                     return;
                                                                                 }

                                                                                 importedProject.savedAtUnix = juce::Time::getCurrentTime().toMilliseconds() / 1000;
                                                                                 auto templateName = juce::File::createLegalFileName(selected.getFileNameWithoutExtension());
                                                                                 if (templateName.isEmpty())
                                                                                     templateName = "Imported Template";
                                                                                 const auto targetFile = ensureTemplateSuffix(templateRoot.getChildFile(templateName));
                                                                                 const auto saveResult = saveProjectFile(targetFile, importedProject);
                                                                                 if (saveResult.failed())
                                                                                 {
                                                                                     juce::AlertWindow::showMessageBoxAsync(juce::AlertWindow::WarningIcon,
                                                                                                                            "Import Template Failed",
                                                                                                                            saveResult.getErrorMessage());
                                                                                     return;
                                                                                 }

                                                                                 safeThis->statusLabel.setText("Imported template: " + targetFile.getFileNameWithoutExtension(),
                                                                                                               juce::dontSendNotification);
                                                                                 safeThis->appendActivityLog("Templates",
                                                                                                             "Imported project template\nSource: "
                                                                                                                 + selected.getFullPathName()
                                                                                                                 + "\nTemplate: "
                                                                                                                 + targetFile.getFullPathName());
                                                                             });
                                    return;
                                }

                                if (result == 1)
                                {
                                    const auto* templateBox = safeDialog->getComboBoxComponent("defaultTemplate");
                                    const auto templateIndex = templateBox != nullptr
                                        ? juce::jlimit(0, static_cast<int>(templateOptions.size()) - 1, templateBox->getSelectedItemIndex())
                                        : 0;
                                    const auto& selectedTemplate = templateOptions[static_cast<size_t>(templateIndex)];

                                    if (!selectedTemplate.builtIn && !selectedTemplate.file.existsAsFile())
                                    {
                                        juce::AlertWindow::showMessageBoxAsync(juce::AlertWindow::WarningIcon,
                                                                               "Template Missing",
                                                                               "That template file could not be found.");
                                        return;
                                    }

                                    if (safeThis->windowStateSettings != nullptr)
                                    {
                                        safeThis->windowStateSettings->setValue("project_default_template_id", selectedTemplate.identifier);
                                        safeThis->windowStateSettings->saveIfNeeded();
                                    }

                                    safeThis->statusLabel.setText("Default template set to " + selectedTemplate.name + ".", juce::dontSendNotification);
                                    safeThis->appendActivityLog("Templates",
                                                                "Set default template\n" + selectedTemplate.name);
                                }
                            }),
                            true);
}

void StudioShellComponent::showAiSettingsDialog()
{
    const auto installedOllama = detectedOllamaExecutable();
    const bool ollamaDetected = installedOllama.existsAsFile();
    const auto detectedAceStep = detectedAceStepInstallDirectory();
    auto safeThis = juce::Component::SafePointer<StudioShellComponent>(this);
    auto dialogContent = std::make_unique<AiSettingsDialogContainerComponent>(aiClient,
                                                                              aceStepClient,
                                                                              ollamaDetected,
                                                                              installedOllama,
                                                                              detectedAceStep,
                                                                              [safeThis] (int result,
                                                                                          const TabbedAiSettingsContentComponent& content)
                                                                              {
                                                                                  if (safeThis == nullptr || result == 0)
                                                                                      return;

                                                                                  if (result == 2)
                                                                                  {
                                                                                      safeThis->aiClient.clearAuth();
                                                                                      safeThis->aceStepClient.clearApiKey();
                                                                                      safeThis->updateAiStatusSummary();
                                                                                      safeThis->statusLabel.setText("Cleared saved AI credentials.", juce::dontSendNotification);
                                                                                      safeThis->appendActivityLog("AI Settings", "Cleared saved AI credentials.");
                                                                                      return;
                                                                                  }

                                                                                  const auto provider = content.selectedProvider();
                                                                                  safeThis->aiClient.setProvider(provider);
                                                                                  safeThis->aiClient.setRemoteModel(content.remoteModel());
                                                                                  safeThis->aiClient.setRemoteBaseUrl(content.remoteBaseUrl());
                                                                                  safeThis->aiClient.setOllamaConnection(content.ollamaBaseUrl(), content.ollamaModel());
                                                                                  safeThis->aiClient.setRequestTimeoutSeconds(content.timeoutSeconds());
                                                                                  safeThis->aiClient.saveSettings();

                                                                                  safeThis->aceStepClient.setBaseUrl(content.aceStepBaseUrl());
                                                                                  safeThis->aceStepClient.setInstallDirectory(content.aceStepInstallDirectory());
                                                                                  safeThis->aceStepClient.setAutoStartServer(content.aceStepAutoStartEnabled());
                                                                                  safeThis->aceStepClient.setDefaultModel(content.aceStepDefaultModel());
                                                                                  safeThis->aceStepClient.setDefaultAudioFormat(content.aceStepAudioFormat());
                                                                                  safeThis->aceStepClient.setDefaultQualityPreset(content.aceStepQualityPreset());
                                                                                  safeThis->aceStepClient.setDefaultVocalLanguage(content.aceStepVocalLanguage());
                                                                                  safeThis->aceStepClient.setDefaultThinking(content.aceStepThinkingEnabled());
                                                                                  safeThis->aceStepClient.setDefaultUseRandomSeed(content.aceStepUseRandomSeed());
                                                                                  safeThis->aceStepClient.setDefaultSeed(content.aceStepSeed());
                                                                                  safeThis->aceStepClient.setDefaultInferenceSteps(content.aceStepInferenceSteps());
                                                                                  safeThis->aceStepClient.setDefaultGuidanceScale(content.aceStepGuidanceScale());
                                                                                  safeThis->aceStepClient.setDefaultInferMethod(content.aceStepInferMethod());
                                                                                  safeThis->aceStepClient.setStartupTimeoutSeconds(content.aceStepStartupTimeoutSeconds());
                                                                                  safeThis->aceStepClient.setJobTimeoutSeconds(content.aceStepJobTimeoutSeconds());
                                                                                  safeThis->aceStepClient.saveSettings();

                                                                                  const auto apiKey = content.remoteApiKey();
                                                                                  if (apiKey.isNotEmpty())
                                                                                      safeThis->aiClient.setApiKey(apiKey);

                                                                                  const auto aceStepApiKey = content.aceStepApiKey();
                                                                                  if (aceStepApiKey.isNotEmpty())
                                                                                      safeThis->aceStepClient.setApiKey(aceStepApiKey);

                                                                                  safeThis->updateAiStatusSummary();
                                                                                  safeThis->statusLabel.setText("Updated AI settings.", juce::dontSendNotification);
                                                                                  safeThis->appendActivityLog("AI Settings",
                                                                                                              "Updated AI settings\nProvider: "
                                                                                                                  + AIClient::providerDisplayName(provider)
                                                                                                                  + "\nStatus: "
                                                                                                                  + safeThis->aiClient.authStatus()
                                                                                                                  + "\nACE-Step: "
                                                                                                                  + safeThis->aceStepClient.statusSummary());
                                                                              });

    juce::DialogWindow::LaunchOptions options;
    options.dialogTitle = "AI Settings";
    options.dialogBackgroundColour = juce::Colour::fromRGB(52, 63, 72);
    options.componentToCentreAround = this;
    options.escapeKeyTriggersCloseButton = true;
    options.useNativeTitleBar = true;
    options.resizable = true;
    options.useBottomRightCornerResizer = true;
    options.content.setOwned(dialogContent.release());
    auto* dialog = options.launchAsync();
    if (dialog != nullptr)
    {
        dialog->setResizable(true, true);
        dialog->setResizeLimits(760, 540, 1200, 900);
    }
}

juce::String StudioShellComponent::buildAboutDetails() const
{
    auto* app = juce::JUCEApplicationBase::getInstance();
    const auto appVersion = app != nullptr ? app->getApplicationVersion() : juce::String("Unknown");
    const auto executable = juce::File::getSpecialLocation(juce::File::currentApplicationFile);
    const auto installTime = executable.getCreationTime();
    const auto modifiedTime = executable.getLastModificationTime();

    juce::StringArray lines;
    lines.add("Release: Mutagen " + appVersion);
    lines.add("Built: " + juce::String(__DATE__) + " " + juce::String(__TIME__));
    lines.add("Installed: " + installTime.toString(true, true, false, true));
    lines.add("Updated: " + modifiedTime.toString(true, true, false, true));
    lines.add("Executable: " + executable.getFullPathName());
    lines.add({});
    lines.add("Tracks: " + juce::String(static_cast<int>(documentState.project.tracks.size())));
    lines.add("Pattern clips: " + juce::String(static_cast<int>(documentState.project.midiSections.size())));
    lines.add("Rack items: " + juce::String(static_cast<int>(documentState.project.vstRack.size())));
    lines.add("Tracked VST paths: " + juce::String(documentState.project.vstiPaths.size()));
    lines.add("Tracked VST folders: " + juce::String(documentState.project.vstiFolderPaths.size()));
    lines.add({});
    lines.add("OS: " + juce::SystemStats::getOperatingSystemName()
              + (juce::SystemStats::isOperatingSystem64Bit() ? " (64-bit)" : ""));
    lines.add("CPU: " + juce::SystemStats::getCpuVendor() + " " + juce::SystemStats::getCpuModel());
    lines.add("Logical cores: " + juce::String(juce::SystemStats::getNumCpus()));
    lines.add("Memory: " + juce::String(juce::SystemStats::getMemorySizeInMegabytes()) + " MB");
    return lines.joinIntoString("\n");
}

void StudioShellComponent::showAboutDialog()
{
    juce::AlertWindow::showMessageBoxAsync(juce::AlertWindow::InfoIcon,
                                           "About Mutagen",
                                           buildAboutDetails());
}

juce::File StudioShellComponent::suggestAceStepGeneratedAudioFile(int trackIndex, const juce::String& audioFormat) const
{
    auto outputDirectory = currentProjectFile.existsAsFile()
        ? currentProjectFile.getParentDirectory().getChildFile(currentProjectFile.getFileNameWithoutExtension() + "_generated_audio")
        : nativeGeneratedAudioDirectory();

    auto baseName = juce::String("sample-track");
    if (juce::isPositiveAndBelow(trackIndex, static_cast<int>(documentState.project.tracks.size())))
        baseName = documentState.project.tracks[static_cast<size_t>(trackIndex)].name.trim();
    baseName = juce::File::createLegalFileName(baseName);
    if (baseName.isEmpty())
        baseName = "sample-track";

    const auto safeExtension = audioFormat.trim().isNotEmpty()
        ? audioFormat.trim().toLowerCase()
        : juce::String("wav");
    const auto timestamp = juce::Time::getCurrentTime().formatted("%Y%m%d-%H%M%S");
    auto candidate = outputDirectory.getChildFile("ace-step-" + timestamp + "-" + baseName)
        .withFileExtension("." + (safeExtension.isNotEmpty() ? safeExtension : juce::String("wav")));

    int suffix = 2;
    while (candidate.existsAsFile())
    {
        candidate = outputDirectory.getChildFile("ace-step-" + timestamp + "-" + baseName + "-" + juce::String(suffix++))
            .withFileExtension("." + (safeExtension.isNotEmpty() ? safeExtension : juce::String("wav")));
    }

    return candidate;
}

juce::Result StudioShellComponent::ensureAceStepServerLaunchRequested()
{
    if (aceStepClient.isServerReachable(800))
        return juce::Result::ok();

    if (aceStepClient.isServerPortAcceptingConnections(250))
    {
        appendActivityLog("ACE-Step",
                          "ACE-Step is already bound to "
                              + aceStepClient.getBaseUrl()
                              + "\nWaiting for the health endpoint instead of launching a second server.");
        return juce::Result::ok();
    }

    if (!aceStepClient.getAutoStartServer())
    {
        return juce::Result::fail("ACE-Step is not reachable at "
                                  + aceStepClient.getBaseUrl()
                                  + ". Start the API server manually or enable auto-start in AI Settings.");
    }

    if (aceStepServerProcess != nullptr && aceStepServerProcess->isRunning())
        return juce::Result::ok();

    const auto launchScript = aceStepClient.launchScriptFile();
    if (!launchScript.existsAsFile())
    {
        return juce::Result::fail("Mutagen could not find the ACE-Step launch script. Configure the ACE-Step install folder in AI Settings.");
    }

    aceStepServerProcess = std::make_unique<juce::ChildProcess>();
    aceStepServerLogFile = nativeAceStepServerLogFile();
    aceStepServerLogFile.getParentDirectory().createDirectory();
    aceStepServerLogFile.deleteFile();
    aceStepServerLogReadPosition = 0;
    aceStepServerOutputCarry.clear();
    aceStepLastProgressLogLine.clear();
    aceStepLastProgressLogMs = 0;

   #if JUCE_WINDOWS
    const juce::StringArray command { "cmd", "/c", launchScript.getFullPathName() };
   #else
    const juce::StringArray command { "sh", launchScript.getFullPathName() };
   #endif

    if (!aceStepServerProcess->start(command, 0))
    {
        aceStepServerProcess.reset();
        return juce::Result::fail("Mutagen could not launch ACE-Step from " + launchScript.getFullPathName());
    }

    appendActivityLog("ACE-Step", "Launched local ACE-Step API server\nScript: " + launchScript.getFullPathName());
    return juce::Result::ok();
}

void StudioShellComponent::generateAudioWithAceStep()
{
    if (aiComposeBusy)
    {
        statusLabel.setText("Finish the current AI composition before starting ACE-Step audio generation.",
                            juce::dontSendNotification);
        return;
    }

    if (aceStepGenerationBusy)
    {
        statusLabel.setText("ACE-Step generation is already running in the background.", juce::dontSendNotification);
        return;
    }

    const auto trackIndex = getSelectedTrackIndex();
    if (!juce::isPositiveAndBelow(trackIndex, static_cast<int>(documentState.project.tracks.size()))
        || !documentState.project.tracks[static_cast<size_t>(trackIndex)].trackType.equalsIgnoreCase("sample"))
    {
        juce::AlertWindow::showMessageBoxAsync(juce::AlertWindow::InfoIcon,
                                               "Generate Audio",
                                               "Select a sample track first. Add a sample track when you want ACE-Step to create audio directly in the DAW.");
        return;
    }

    const auto& project = documentState.project;
    auto* dialog = new juce::AlertWindow("Generate Audio With ACE-Step",
                                         "Describe the audio you want on the selected sample track. Mutagen will render it in the background and insert the clip at the playhead.",
                                         juce::AlertWindow::NoIcon);
    dialog->addTextEditor("prompt", aceStepDefaultPrompt, "Prompt");
    if (auto* promptEditor = dialog->getTextEditor("prompt"))
    {
        promptEditor->setMultiLine(true, true);
        promptEditor->setReturnKeyStartsNewLine(true);
        promptEditor->setSize(promptEditor->getWidth(), 140);
    }

    dialog->addTextEditor("lyrics", aceStepDefaultLyrics, "Lyrics (optional)");
    if (auto* lyricsEditor = dialog->getTextEditor("lyrics"))
    {
        lyricsEditor->setMultiLine(true, true);
        lyricsEditor->setReturnKeyStartsNewLine(true);
        lyricsEditor->setSize(lyricsEditor->getWidth(), 120);
    }

    dialog->addTextEditor("bars", juce::String(aceStepDefaultBars), "Bars");
    dialog->addTextBlock("Project context\nTempo: "
                         + juce::String(project.bpm)
                         + " BPM\nTime signature: "
                         + timeSignatureDisplayName(project)
                         + "\nKey: "
                         + keyQuantizeDisplayName(project.keyQuantizeRoot, project.keyQuantizeScale)
                         + "\nTrack: "
                         + documentState.project.tracks[static_cast<size_t>(trackIndex)].name);
    dialog->addTextBlock("ACE-Step defaults\nQuality: "
                         + aceStepClient.getDefaultQualityPreset()
                         + "\nModel: "
                         + (aceStepClient.getDefaultModel().trim().isNotEmpty() ? aceStepClient.getDefaultModel().trim() : juce::String("Server default"))
                         + "\nFormat: "
                         + aceStepClient.getDefaultAudioFormat()
                         + "\nLanguage: "
                         + aceStepClient.getDefaultVocalLanguage()
                         + "\nSeed: "
                         + (aceStepClient.getDefaultUseRandomSeed()
                                ? juce::String("Random")
                                : juce::String("Fixed ") + juce::String(aceStepClient.getDefaultSeed()))
                         + "\nThinking: "
                         + (aceStepClient.getDefaultThinking() ? "Enabled" : "Disabled")
                         + "\nInference steps: "
                         + juce::String(aceStepClient.getDefaultInferenceSteps())
                         + "\nInfer method: "
                         + aceStepClient.getDefaultInferMethod());
    dialog->setSize(620, 720);
    dialog->addButton("Generate", 1, juce::KeyPress(juce::KeyPress::returnKey));
    dialog->addButton("Cancel", 0, juce::KeyPress(juce::KeyPress::escapeKey));

    auto safeThis = juce::Component::SafePointer<StudioShellComponent>(this);
    auto safeDialog = juce::Component::SafePointer<juce::AlertWindow>(dialog);
    dialog->enterModalState(true,
                            juce::ModalCallbackFunction::create([safeThis, safeDialog, trackIndex] (int result)
                            {
                                if (safeThis == nullptr || safeDialog == nullptr || result != 1)
                                    return;

                                if (!juce::isPositiveAndBelow(trackIndex, static_cast<int>(safeThis->documentState.project.tracks.size()))
                                    || !safeThis->documentState.project.tracks[static_cast<size_t>(trackIndex)].trackType.equalsIgnoreCase("sample"))
                                {
                                    juce::AlertWindow::showMessageBoxAsync(juce::AlertWindow::InfoIcon,
                                                                           "Generate Audio",
                                                                           "The selected sample track is no longer available.");
                                    return;
                                }

                                const auto prompt = safeDialog->getTextEditorContents("prompt").trim();
                                const auto lyrics = safeDialog->getTextEditorContents("lyrics").trim();
                                if (prompt.isEmpty() && lyrics.isEmpty())
                                {
                                    juce::AlertWindow::showMessageBoxAsync(juce::AlertWindow::InfoIcon,
                                                                           "Generate Audio",
                                                                           "Enter a prompt or lyrics for ACE-Step first.");
                                    return;
                                }

                                const auto bars = juce::jlimit(1, 128, safeDialog->getTextEditorContents("bars").getIntValue());
                                const auto launchResult = safeThis->ensureAceStepServerLaunchRequested();
                                if (launchResult.failed())
                                {
                                    juce::AlertWindow::showMessageBoxAsync(juce::AlertWindow::WarningIcon,
                                                                           "ACE-Step Unavailable",
                                                                           launchResult.getErrorMessage());
                                    return;
                                }

                                safeThis->aceStepDefaultPrompt = prompt;
                                safeThis->aceStepDefaultLyrics = lyrics;
                                safeThis->aceStepDefaultBars = bars;

                                AceStepGenerationRequest request;
                                request.prompt = prompt;
                                request.lyrics = lyrics;
                                request.model = safeThis->aceStepClient.getDefaultModel();
                                request.bpm = safeThis->documentState.project.bpm;
                                request.keyScale = aceStepKeyScaleValue(safeThis->documentState.project);
                                request.timeSignature = aceStepTimeSignatureValue(safeThis->documentState.project);
                                request.durationSeconds = juce::jmax(10.0,
                                                                     tickToSeconds(safeThis->documentState.project,
                                                                                   bars * ticksPerBar(safeThis->documentState.project)));
                                request.thinking = safeThis->aceStepClient.getDefaultThinking();
                                request.audioFormat = safeThis->aceStepClient.getDefaultAudioFormat();
                                request.vocalLanguage = safeThis->aceStepClient.getDefaultVocalLanguage();
                                request.useRandomSeed = safeThis->aceStepClient.getDefaultUseRandomSeed();
                                request.seed = safeThis->aceStepClient.getDefaultSeed();
                                request.inferenceSteps = safeThis->aceStepClient.getDefaultInferenceSteps();
                                request.guidanceScale = safeThis->aceStepClient.getDefaultGuidanceScale();
                                request.inferMethod = safeThis->aceStepClient.getDefaultInferMethod();

                                const auto insertTick = safeThis->documentState.project.playheadTick;
                                const auto targetFile = safeThis->suggestAceStepGeneratedAudioFile(trackIndex, request.audioFormat);
                                const auto trackName = safeThis->documentState.project.tracks[static_cast<size_t>(trackIndex)].name;

                                safeThis->setAceStepGenerationBusy(true, "ACE-Step generating audio...");
                                safeThis->statusLabel.setText("ACE-Step is generating audio for " + trackName + ".",
                                                              juce::dontSendNotification);
                                safeThis->appendActivityLog("ACE-Step",
                                                            "Audio generation requested\nTrack: "
                                                                + trackName
                                                                + "\nBars: "
                                                                + juce::String(bars)
                                                                + "\nDuration: "
                                                                + juce::String(request.durationSeconds, 1)
                                                                + " sec\nPrompt: "
                                                                + prompt
                                                                + "\nQuality: "
                                                                + safeThis->aceStepClient.getDefaultQualityPreset()
                                                                + "\nModel: "
                                                                + (request.model.isNotEmpty() ? request.model : juce::String("Server default"))
                                                                + "\nFormat: "
                                                                + request.audioFormat
                                                                + "\nLanguage: "
                                                                + request.vocalLanguage
                                                                + "\nSeed: "
                                                                + (request.useRandomSeed ? juce::String("Random") : juce::String(request.seed))
                                                                + "\nThinking: "
                                                                + (request.thinking ? "Enabled" : "Disabled")
                                                                + "\nInference steps: "
                                                                + juce::String(request.inferenceSteps)
                                                                + (lyrics.isNotEmpty() ? "\nLyrics:\n" + lyrics : juce::String()));

                                auto clientCopy = safeThis->aceStepClient;
                                safeThis->aceStepGenerationFuture = std::async(std::launch::async,
                                                                               [client = std::move(clientCopy),
                                                                                request,
                                                                                targetFile,
                                                                                trackIndex,
                                                                                insertTick] () mutable
                                                                               {
                                                                                   AceStepTrackGenerationResult generationResult;
                                                                                   generationResult.trackIndex = trackIndex;
                                                                                   generationResult.insertTick = insertTick;
                                                                                   try
                                                                                   {
                                                                                       generationResult.generation = client.generateToFile(request, targetFile);
                                                                                       generationResult.success = true;
                                                                                   }
                                                                                   catch (const std::exception& exc)
                                                                                   {
                                                                                       generationResult.errorMessage = juce::String::fromUTF8(exc.what());
                                                                                   }
                                                                                   return generationResult;
                                                                               });
                            }),
                            true);
}

void StudioShellComponent::composeWithAi()
{
    if (aiComposeBusy)
    {
        statusLabel.setText("AI composition is already running in the background.", juce::dontSendNotification);
        return;
    }

    if (!aiClient.isEnabled())
    {
        juce::AlertWindow::showMessageBoxAsync(juce::AlertWindow::InfoIcon,
                                               "AI Not Configured",
                                               "Open AI Settings first and connect a remote provider or choose a local Ollama model.");
        return;
    }

    auto* dialog = new juce::AlertWindow("AI Compose",
                                         "Describe the arrangement you want the native C++ composer to generate.",
                                         juce::AlertWindow::NoIcon);
    dialog->addTextEditor("prompt", aiComposeDefaultPrompt, "Prompt");
    if (auto* promptEditor = dialog->getTextEditor("prompt"))
    {
        promptEditor->setMultiLine(true, true);
        promptEditor->setReturnKeyStartsNewLine(true);
        promptEditor->setSize(promptEditor->getWidth(), 180);
    }

    const auto styleChoices = aiComposeStyleChoices();
    const auto energyChoices = aiComposeEnergyChoices();
    const auto densityChoices = aiComposeDensityChoices();
    const auto variationChoices = aiComposeVariationChoices();
    const auto registerChoices = aiComposeRegisterChoices();

    dialog->addComboBox("targetMode",
                        { "Replace Current Track", "Replace All Tracks", "Add To Current Track", "Add To All Tracks" },
                        "Target");
    dialog->addTextEditor("bars", juce::String(aiComposeDefaultBars), "Bars");
    dialog->addComboBox("style", styleChoices, "Style");
    dialog->addComboBox("energy", energyChoices, "Energy");
    dialog->addComboBox("density", densityChoices, "Density");
    dialog->addComboBox("variation", variationChoices, "Variation");
    dialog->addComboBox("register", registerChoices, "Register");
    if (auto* targetModeBox = dialog->getComboBoxComponent("targetMode"))
        targetModeBox->setSelectedId(aiComposeModeComboId(aiComposeDefaultMode), juce::dontSendNotification);
    if (auto* styleBox = dialog->getComboBoxComponent("style"))
        styleBox->setSelectedItemIndex(juce::jmax(0, styleChoices.indexOf(aiComposeDefaultStyle)), juce::dontSendNotification);
    if (auto* energyBox = dialog->getComboBoxComponent("energy"))
        energyBox->setSelectedItemIndex(juce::jmax(0, energyChoices.indexOf(aiComposeDefaultEnergy)), juce::dontSendNotification);
    if (auto* densityBox = dialog->getComboBoxComponent("density"))
        densityBox->setSelectedItemIndex(juce::jmax(0, densityChoices.indexOf(aiComposeDefaultDensity)), juce::dontSendNotification);
    if (auto* variationBox = dialog->getComboBoxComponent("variation"))
        variationBox->setSelectedItemIndex(juce::jmax(0, variationChoices.indexOf(aiComposeDefaultVariation)), juce::dontSendNotification);
    if (auto* registerBox = dialog->getComboBoxComponent("register"))
        registerBox->setSelectedItemIndex(juce::jmax(0, registerChoices.indexOf(aiComposeDefaultRegister)), juce::dontSendNotification);
    dialog->setSize(620, 640);
    dialog->addButton("Compose", 1, juce::KeyPress(juce::KeyPress::returnKey));
    dialog->addButton("Cancel", 0, juce::KeyPress(juce::KeyPress::escapeKey));

    auto safeThis = juce::Component::SafePointer<StudioShellComponent>(this);
    auto safeDialog = juce::Component::SafePointer<juce::AlertWindow>(dialog);
    dialog->enterModalState(true,
                            juce::ModalCallbackFunction::create([safeThis, safeDialog] (int result)
                            {
                                if (safeThis == nullptr || safeDialog == nullptr || result != 1)
                                    return;

                                const auto prompt = safeDialog->getTextEditorContents("prompt").trim();
                                const auto bars = juce::jlimit(1, 128, safeDialog->getTextEditorContents("bars").getIntValue());
                                const auto targetMode = aiComposeModeFromComboId(safeDialog->getComboBoxComponent("targetMode") != nullptr
                                                                                     ? safeDialog->getComboBoxComponent("targetMode")->getSelectedId()
                                                                                     : aiComposeModeComboId(safeThis->aiComposeDefaultMode));
                                const auto style = safeDialog->getComboBoxComponent("style") != nullptr
                                    ? safeDialog->getComboBoxComponent("style")->getText().trim()
                                    : safeThis->aiComposeDefaultStyle;
                                const auto energy = safeDialog->getComboBoxComponent("energy") != nullptr
                                    ? safeDialog->getComboBoxComponent("energy")->getText().trim()
                                    : safeThis->aiComposeDefaultEnergy;
                                const auto density = safeDialog->getComboBoxComponent("density") != nullptr
                                    ? safeDialog->getComboBoxComponent("density")->getText().trim()
                                    : safeThis->aiComposeDefaultDensity;
                                const auto variation = safeDialog->getComboBoxComponent("variation") != nullptr
                                    ? safeDialog->getComboBoxComponent("variation")->getText().trim()
                                    : safeThis->aiComposeDefaultVariation;
                                const auto registerFocus = safeDialog->getComboBoxComponent("register") != nullptr
                                    ? safeDialog->getComboBoxComponent("register")->getText().trim()
                                    : safeThis->aiComposeDefaultRegister;
                                if (prompt.isEmpty())
                                {
                                    juce::AlertWindow::showMessageBoxAsync(juce::AlertWindow::InfoIcon,
                                                                           "AI Compose",
                                                                           "Enter a prompt for the composition first.");
                                    return;
                                }

                                const auto& project = safeThis->documentState.project;
                                const auto selectedTrackIndex = safeThis->getSelectedTrackIndex();
                                const auto collectInstrumentTrackIndices = [&project]()
                                {
                                    std::vector<int> indices;
                                    for (int trackIndex = 0; trackIndex < static_cast<int>(project.tracks.size()); ++trackIndex)
                                    {
                                        if (project.tracks[static_cast<size_t>(trackIndex)].trackType.equalsIgnoreCase("instrument"))
                                            indices.push_back(trackIndex);
                                    }
                                    return indices;
                                };

                                auto targetTrackIndices = collectInstrumentTrackIndices();
                                auto requestedTrackCount = 4;

                                if (targetMode == AIComposeTargetMode::replaceCurrentTrack
                                    || targetMode == AIComposeTargetMode::addToCurrentTrack)
                                {
                                    if (!juce::isPositiveAndBelow(selectedTrackIndex, static_cast<int>(project.tracks.size()))
                                        || !project.tracks[static_cast<size_t>(selectedTrackIndex)].trackType.equalsIgnoreCase("instrument"))
                                    {
                                        juce::AlertWindow::showMessageBoxAsync(juce::AlertWindow::InfoIcon,
                                                                               "AI Compose",
                                                                               "Select an instrument track first. Current-track AI Compose does not target sample tracks.");
                                        return;
                                    }

                                    targetTrackIndices = { selectedTrackIndex };
                                    requestedTrackCount = 1;
                                }
                                else if (targetMode == AIComposeTargetMode::addToAllTracks)
                                {
                                    if (targetTrackIndices.empty())
                                    {
                                        juce::AlertWindow::showMessageBoxAsync(juce::AlertWindow::InfoIcon,
                                                                               "AI Compose",
                                                                               "There are no instrument tracks available to add MIDI to.");
                                        return;
                                    }

                                    if (targetTrackIndices.size() > 16)
                                        targetTrackIndices.resize(16);
                                    requestedTrackCount = static_cast<int>(targetTrackIndices.size());
                                }
                                else
                                {
                                    const auto existingInstrumentTrackCount = static_cast<int>(targetTrackIndices.size());
                                    requestedTrackCount = existingInstrumentTrackCount >= 2
                                        ? juce::jmin(8, existingInstrumentTrackCount)
                                        : 4;
                                }

                                const auto insertStartTick = (targetMode == AIComposeTargetMode::addToCurrentTrack
                                                              || targetMode == AIComposeTargetMode::addToAllTracks)
                                    ? aiComposeRoundToGrid(project.playheadTick, aiComposeGridTick(project))
                                    : 0;

                                juce::StringArray promptLines;
                                promptLines.add("User brief: " + prompt);
                                promptLines.add("Project tempo: " + juce::String(project.bpm) + " BPM.");
                                promptLines.add("Project time signature: " + timeSignatureDisplayName(project) + ".");
                                promptLines.add("Project key quantize: " + keyQuantizeDisplayName(project.keyQuantizeRoot, project.keyQuantizeScale) + ".");
                                promptLines.add("Project rhythmic grid: " + aiComposeGridLabel(project)
                                                + " (" + juce::String(aiComposeGridTick(project)) + " ticks).");
                                promptLines.add("Length to generate: " + juce::String(bars) + " bars.");
                                promptLines.add("Creative direction: style " + style
                                                + ", energy " + energy
                                                + ", density " + density
                                                + ", variation " + variation
                                                + ", register " + registerFocus + ".");

                                switch (targetMode)
                                {
                                    case AIComposeTargetMode::replaceCurrentTrack:
                                        promptLines.add("Operation: replace the current track only.");
                                        promptLines.add("Generate exactly 1 track in the JSON response.");
                                        promptLines.add("The DAW will preserve the selected track's existing instrument and rack assignment.");
                                        promptLines.add("Target track context:");
                                        promptLines.add(aiComposeTrackContextLine(project, targetTrackIndices.front()));
                                        break;

                                    case AIComposeTargetMode::replaceAllTracks:
                                        promptLines.add("Operation: replace the full arrangement with a fresh set of tracks.");
                                        promptLines.add("Generate exactly " + juce::String(requestedTrackCount) + " tracks in the JSON response.");
                                        promptLines.add("Aim for a complete arrangement with clear musical roles across the generated tracks.");
                                        break;

                                    case AIComposeTargetMode::addToCurrentTrack:
                                        promptLines.add("Operation: add notes to the current existing track.");
                                        promptLines.add("Generate exactly 1 track in the JSON response.");
                                        promptLines.add("Write complementary notes for the existing track rather than changing its role.");
                                        promptLines.add("Returned notes should start relative to beat 0; the DAW will insert them at "
                                                        + aiComposeTickLocationLabel(project, insertStartTick) + ".");
                                        promptLines.add("Target track context:");
                                        promptLines.add(aiComposeTrackContextLine(project, targetTrackIndices.front()));
                                        break;

                                    case AIComposeTargetMode::addToAllTracks:
                                        promptLines.add("Operation: add notes to every existing instrument track.");
                                        promptLines.add("Generate exactly " + juce::String(requestedTrackCount)
                                                        + " tracks in the JSON response, one per target track, in the same order listed below.");
                                        promptLines.add("Write complementary material for each target track rather than replacing it.");
                                        promptLines.add("Returned notes should start relative to beat 0; the DAW will insert them at "
                                                        + aiComposeTickLocationLabel(project, insertStartTick) + ".");
                                        promptLines.add("Target track order:");
                                        for (int index = 0; index < static_cast<int>(targetTrackIndices.size()); ++index)
                                            promptLines.add(juce::String(index + 1) + ". "
                                                            + aiComposeTrackContextLine(project, targetTrackIndices[static_cast<size_t>(index)]));
                                        break;
                                }

                                promptLines.add("All returned note starts, durations, and pitches should make sense after DAW quantization to the current grid and key.");
                                promptLines.add("Avoid empty tracks.");
                                const auto expandedPrompt = promptLines.joinIntoString("\n");

                                safeThis->aiComposeDefaultPrompt = prompt;
                                safeThis->aiComposeDefaultBars = bars;
                                safeThis->aiComposeDefaultMode = targetMode;
                                safeThis->aiComposeDefaultStyle = style;
                                safeThis->aiComposeDefaultEnergy = energy;
                                safeThis->aiComposeDefaultDensity = density;
                                safeThis->aiComposeDefaultVariation = variation;
                                safeThis->aiComposeDefaultRegister = registerFocus;
                                safeThis->aiComposeRequestedBars = bars;
                                safeThis->aiComposeRequestedMode = targetMode;
                                safeThis->aiComposeRequestedTargetTracks = targetTrackIndices;
                                safeThis->aiComposeRequestedInsertTick = insertStartTick;
                                safeThis->setAiComposeBusy(true, "AI composing " + aiComposeModeLabel(targetMode).toLowerCase() + "...");
                                safeThis->statusLabel.setText("AI processing " + aiComposeModeLabel(targetMode).toLowerCase()
                                                                  + " via " + safeThis->aiClient.authStatus() + ".",
                                                              juce::dontSendNotification);
                                safeThis->appendActivityLog("AI Compose",
                                                            "Compose requested\nMode: "
                                                                + aiComposeModeLabel(targetMode)
                                                                + "\nBars: "
                                                                + juce::String(bars)
                                                                + "\nBPM: "
                                                                + juce::String(project.bpm)
                                                                + "\nKey: "
                                                                + keyQuantizeDisplayName(project.keyQuantizeRoot, project.keyQuantizeScale)
                                                                + "\nGrid: "
                                                                + aiComposeGridLabel(project)
                                                                + "\nProvider: "
                                                                + safeThis->aiClient.authStatus()
                                                                + "\nStyle: "
                                                                + style
                                                                + "\nEnergy: "
                                                                + energy
                                                                + "\nDensity: "
                                                                + density
                                                                + "\nVariation: "
                                                                + variation
                                                                + "\nRegister: "
                                                                + registerFocus
                                                                + "\n\nPrompt\n"
                                                                + expandedPrompt);

                                auto clientCopy = safeThis->aiClient;
                                safeThis->aiComposeFuture = std::async(std::launch::async,
                                                                       [client = std::move(clientCopy),
                                                                        expandedPrompt,
                                                                        bars,
                                                                        bpm = project.bpm,
                                                                        requestedTrackCount] () mutable
                                                                       {
                                                                           AIComposer composer(std::move(client));
                                                                           return composer.compose(expandedPrompt,
                                                                                                   bars,
                                                                                                   bpm,
                                                                                                   requestedTrackCount,
                                                                                                   requestedTrackCount);
                                                                       });
                            }),
                            true);
}

void StudioShellComponent::setAiComposeBusy(bool busy, const juce::String& detail)
{
    aiComposeBusy = busy;
    aiComposeBusyDetail = busy ? detail : juce::String();
    refreshPollingTimerState();
    updateEditorState();
    updateAiStatusSummary();
}

void StudioShellComponent::setAceStepGenerationBusy(bool busy, const juce::String& detail)
{
    aceStepGenerationBusy = busy;
    aceStepBusyDetail = busy ? detail : juce::String();
    aceStepBootstrapNoticeShown = false;
    refreshPollingTimerState();
    updateEditorState();
    updateAiStatusSummary();
}

void StudioShellComponent::pollAceStepServerOutput()
{
    if (aceStepServerLogFile == juce::File() || !aceStepServerLogFile.existsAsFile())
        return;

    const auto fileSize = aceStepServerLogFile.getSize();
    if (fileSize <= aceStepServerLogReadPosition)
        return;

    constexpr int maxBytesPerPoll = 8 * 1024;
    const auto bytesToRead = static_cast<int> (juce::jmin<int64_t> (fileSize - aceStepServerLogReadPosition,
                                                                     static_cast<int64_t> (maxBytesPerPoll)));
    if (bytesToRead <= 0)
        return;

    juce::FileInputStream input(aceStepServerLogFile);
    if (!input.openedOk())
        return;

    if (!input.setPosition(aceStepServerLogReadPosition))
        return;

    juce::MemoryBlock block;
    block.setSize(static_cast<size_t> (bytesToRead), false);
    const auto numRead = input.read(block.getData(), bytesToRead);
    if (numRead <= 0)
        return;

    aceStepServerLogReadPosition += numRead;

    aceStepServerOutputCarry << sanitiseAceStepServerOutput(juce::String::fromUTF8(static_cast<const char*> (block.getData()),
                                                                                   numRead));
    constexpr int maxCarryCharacters = 65536;
    if (aceStepServerOutputCarry.length() > maxCarryCharacters)
        aceStepServerOutputCarry = aceStepServerOutputCarry.substring(aceStepServerOutputCarry.length() - maxCarryCharacters);

    const auto endedWithNewline = aceStepServerOutputCarry.endsWithChar('\n');
    juce::StringArray lines;
    lines.addLines(aceStepServerOutputCarry);

    aceStepServerOutputCarry.clear();
    if (!endedWithNewline && lines.size() > 0)
    {
        aceStepServerOutputCarry = lines[lines.size() - 1];
        lines.remove(lines.size() - 1);
    }

    juce::String latestProgressLine;
    juce::StringArray informationalLines;
    for (const auto& rawLine : lines)
    {
        const auto line = rawLine.trim();
        if (line.isEmpty())
            continue;

        if (isAceStepProgressLine(line))
        {
            latestProgressLine = line;
            continue;
        }

        if (isAceStepInformationalLine(line))
            informationalLines.add(line);
    }

    if (!informationalLines.isEmpty())
    {
        constexpr int maxInformationalLinesPerPoll = 6;
        if (informationalLines.size() > maxInformationalLinesPerPoll)
            informationalLines.removeRange(maxInformationalLinesPerPoll,
                                           informationalLines.size() - maxInformationalLinesPerPoll);

        appendActivityLog("ACE-Step Server", informationalLines.joinIntoString("\n"));
    }

    if (latestProgressLine.isNotEmpty())
    {
        const auto nowMs = juce::Time::getMillisecondCounter();
        const bool shouldLogProgress = latestProgressLine != aceStepLastProgressLogLine
            && (aceStepLastProgressLogMs == 0 || nowMs - aceStepLastProgressLogMs >= 2500);

        aceStepLastProgressLogLine = latestProgressLine;
        if (shouldLogProgress)
        {
            aceStepLastProgressLogMs = nowMs;
            appendActivityLog("ACE-Step Download", latestProgressLine);
        }
    }
}

void StudioShellComponent::refreshPollingTimerState()
{
    const bool hasOpenRackEditors = hasOpenRackEditorSessions();
    const bool hasVisibleFloatingPianoRoll = floatingPianoRollWorkspace != nullptr
        && pianoRollWindow != nullptr
        && pianoRollWindow->isVisible();
    const bool shouldPauseForRackEditor = loadedRackEditorOpen
        && !hasOpenRackEditors
        && !rackPreviewRunning
        && !projectPreviewRunning
        && !aiComposeBusy
        && !aceStepGenerationBusy
        && !stemSeparationBusy;

    int targetHz = 0;
    if (shouldPauseForRackEditor)
    {
        targetHz = 0;
    }
    else if (projectPreviewRunning || rackPreviewRunning)
    {
        targetHz = playbackRefreshRateForComponent(*this);
        if (loadedRackEditorOpen || hasOpenRackEditors)
            targetHz = juce::jmin(targetHz, kPlaybackRefreshRateWithOpenEditorHz);
        else if (hasVisibleFloatingPianoRoll)
            targetHz = juce::jmin(targetHz, kPlaybackRefreshRateWithOpenPianoRollHz);
    }
    else if (aiComposeBusy || aceStepGenerationBusy || stemSeparationBusy)
    {
        targetHz = 6;
    }
    else if (loadedRackEditorOpen || hasOpenRackEditors)
    {
        targetHz = kIdleRefreshRateWithOpenEditorHz;
    }

    if (targetHz <= 0)
    {
        if (pollingTimerHz != 0 || isTimerRunning())
        {
            stopTimer();
            pollingTimerHz = 0;
        }
        return;
    }

    if (pollingTimerHz != targetHz || !isTimerRunning())
    {
        startTimerHz(targetHz);
        pollingTimerHz = targetHz;
    }
}

void StudioShellComponent::pollAiComposeFuture()
{
    if (!aiComposeBusy || !aiComposeFuture.valid())
        return;

    if (aiComposeFuture.wait_for(std::chrono::seconds(0)) != std::future_status::ready)
        return;

    try
    {
        auto result = aiComposeFuture.get();
        applyAiComposeResult(result, aiComposeRequestedBars);
        juce::String completionText;
        switch (aiComposeRequestedMode)
        {
            case AIComposeTargetMode::replaceCurrentTrack:
                completionText = "AI replaced the current track.";
                break;
            case AIComposeTargetMode::replaceAllTracks:
                completionText = "AI replaced the arrangement with "
                    + juce::String(static_cast<int>(result.tracks.size())) + " track(s).";
                break;
            case AIComposeTargetMode::addToCurrentTrack:
                completionText = "AI added notes to the current track.";
                break;
            case AIComposeTargetMode::addToAllTracks:
                completionText = "AI added notes to "
                    + juce::String(static_cast<int>(juce::jmin(result.tracks.size(),
                                                               aiComposeRequestedTargetTracks.size())))
                    + " track(s).";
                break;
        }

        statusLabel.setText(completionText, juce::dontSendNotification);
        appendActivityLog("AI Compose",
                          "Compose completed successfully\nMode: "
                              + aiComposeModeLabel(aiComposeRequestedMode)
                              + "\nGenerated tracks: "
                              + juce::String(static_cast<int>(result.tracks.size())));
    }
    catch (const std::exception& exc)
    {
        juce::AlertWindow::showMessageBoxAsync(juce::AlertWindow::WarningIcon,
                                               "AI Composition Failed",
                                               juce::String::fromUTF8(exc.what()));
        statusLabel.setText("AI composition failed.", juce::dontSendNotification);
        appendActivityLog("AI Compose Error", juce::String::fromUTF8(exc.what()));
    }

    setAiComposeBusy(false);
}

void StudioShellComponent::pollAceStepGenerationFuture()
{
    if (!aceStepGenerationBusy || !aceStepGenerationFuture.valid())
        return;

    if (aceStepGenerationFuture.wait_for(std::chrono::seconds(0)) != std::future_status::ready)
        return;

    try
    {
        const auto result = aceStepGenerationFuture.get();
        if (!result.success)
        {
            const auto errorText = result.errorMessage.isNotEmpty()
                ? result.errorMessage
                : juce::String("ACE-Step generation failed.");
            const bool bootstrapMessage = aceStepBootstrapMessageMatches(errorText);

            juce::AlertWindow::showMessageBoxAsync(bootstrapMessage ? juce::AlertWindow::InfoIcon
                                                                    : juce::AlertWindow::WarningIcon,
                                                   bootstrapMessage ? "ACE-Step Setup In Progress"
                                                                    : "ACE-Step Generation Failed",
                                                   bootstrapMessage ? aceStepBootstrapUserMessage()
                                                                    : errorText);
            statusLabel.setText(bootstrapMessage ? aceStepBootstrapStatusText()
                                                 : juce::String("ACE-Step generation failed."),
                                juce::dontSendNotification);
            appendActivityLog(bootstrapMessage ? "ACE-Step Setup" : "ACE-Step Error", errorText);
            setAceStepGenerationBusy(false);
            return;
        }

        juce::String trackName;
        const auto placeResult = placeSampleFileOnTrackAtTick(result.generation.outputFile,
                                                              result.trackIndex,
                                                              result.insertTick,
                                                              "Generate ACE-Step Audio",
                                                              trackName);
        if (placeResult.failed())
            throw std::runtime_error(placeResult.getErrorMessage().toStdString());

        setSelectedTrackIndex(result.trackIndex);
        statusLabel.setText("Placed ACE-Step audio on " + trackName + ".", juce::dontSendNotification);
        appendActivityLog("ACE-Step",
                          "Inserted generated audio clip\nTrack: "
                              + trackName
                              + "\nFile: "
                              + result.generation.outputFile.getFullPathName()
                              + (result.generation.modelName.isNotEmpty()
                                     ? "\nModel: " + result.generation.modelName
                                     : juce::String()));
    }
    catch (const std::exception& exc)
    {
        const auto errorText = juce::String::fromUTF8(exc.what());
        const bool bootstrapMessage = aceStepBootstrapMessageMatches(errorText);
        juce::AlertWindow::showMessageBoxAsync(bootstrapMessage ? juce::AlertWindow::InfoIcon
                                                                : juce::AlertWindow::WarningIcon,
                                               bootstrapMessage ? "ACE-Step Setup In Progress"
                                                                : "ACE-Step Generation Failed",
                                               bootstrapMessage ? aceStepBootstrapUserMessage()
                                                                : errorText);
        statusLabel.setText(bootstrapMessage ? aceStepBootstrapStatusText()
                                             : juce::String("ACE-Step generation failed."),
                            juce::dontSendNotification);
        appendActivityLog(bootstrapMessage ? "ACE-Step Setup" : "ACE-Step Error", errorText);
    }

    setAceStepGenerationBusy(false);
}

void StudioShellComponent::pollStemSeparationFuture()
{
    if (!stemSeparationBusy || !stemSeparationFuture.valid())
        return;

    if (stemSeparationFuture.wait_for(std::chrono::seconds(0)) != std::future_status::ready)
        return;

    auto finishBusyState = [this]()
    {
        stemSeparationBusy = false;
        refreshPollingTimerState();
        updateEditorState();
    };

    try
    {
        const auto result = stemSeparationFuture.get();
        if (!result.success)
        {
            juce::AlertWindow::showMessageBoxAsync(juce::AlertWindow::WarningIcon,
                                                   "Stem Separation Failed",
                                                   result.errorMessage.isNotEmpty()
                                                       ? result.errorMessage
                                                       : juce::String("Stem separation failed."));
            statusLabel.setText("Stem separation failed.", juce::dontSendNotification);
            appendActivityLog("Stem Separation Error",
                              result.errorMessage.isNotEmpty() ? result.errorMessage
                                                               : juce::String("Stem separation failed."));
            finishBusyState();
            return;
        }

        auto updatedProject = documentState.project;
        auto sameClip = [] (const SampleClip& lhs, const SampleClip& rhs)
        {
            return lhs.path.equalsIgnoreCase(rhs.path)
                && lhs.trackIndex == rhs.trackIndex
                && std::abs(lhs.startSec - rhs.startSec) < 0.0005
                && std::abs(lhs.durationSec - rhs.durationSec) < 0.0005
                && std::abs(lhs.sourceOffsetSec - rhs.sourceOffsetSec) < 0.0005;
        };

        int sourceTrackIndex = juce::jlimit(0,
                                            juce::jmax(0, static_cast<int>(updatedProject.tracks.size()) - 1),
                                            result.sourceClip.trackIndex);
        for (const auto& clip : updatedProject.sampleClips)
        {
            if (sameClip(clip, result.sourceClip))
            {
                sourceTrackIndex = juce::jlimit(0,
                                                juce::jmax(0, static_cast<int>(updatedProject.tracks.size()) - 1),
                                                clip.trackIndex);
                break;
            }
        }

        const auto insertionTrackIndex = sourceTrackIndex + 1;
        const auto stemCount = static_cast<int>(result.stems.size());

        for (auto& section : updatedProject.midiSections)
        {
            if (section.trackIndex > sourceTrackIndex)
                section.trackIndex += stemCount;
        }
        for (auto& clip : updatedProject.sampleClips)
        {
            if (clip.trackIndex > sourceTrackIndex)
                clip.trackIndex += stemCount;
        }

        for (int stemIndex = 0; stemIndex < stemCount; ++stemIndex)
        {
            const auto& stem = result.stems[static_cast<size_t>(stemIndex)];

            TrackState stemTrack;
            stemTrack.name = result.sourceTrackName.trim().isNotEmpty()
                ? result.sourceTrackName.trim() + " " + stem.name
                : stem.name;
            stemTrack.trackType = "sample";
            stemTrack.instrument = stem.name;
            stemTrack.instrumentMode = "Audio Clip";
            stemTrack.midiChannel = (insertionTrackIndex + stemIndex) % 16;
            stemTrack.followThemeTrackColour = true;
            stemTrack.themeColourSlot = insertionTrackIndex + stemIndex;
            stemTrack.colorHex = defaultTrackColour(stemTrack.themeColourSlot).toDisplayString(false);

            updatedProject.tracks.insert(updatedProject.tracks.begin() + insertionTrackIndex + stemIndex,
                                         std::move(stemTrack));

            const auto stemPath = stem.asset.path.trim();
            bool haveAsset = false;
            for (const auto& existing : updatedProject.sampleAssets)
            {
                if (existing.path.equalsIgnoreCase(stemPath))
                {
                    haveAsset = true;
                    break;
                }
            }

            if (!haveAsset)
                updatedProject.sampleAssets.push_back(stem.asset);
            updatedProject.samplePaths.addIfNotAlreadyThere(stemPath);

            SampleClip stemClip;
            stemClip.path = stem.asset.path;
            stemClip.trackIndex = insertionTrackIndex + stemIndex;
            stemClip.startSec = result.sourceClip.startSec;
            stemClip.sourceOffsetSec = result.sourceClip.sourceOffsetSec;
            stemClip.sourceFileDurationSec = stem.asset.durationSec;
            stemClip.sampleRate = stem.asset.sampleRate;
            stemClip.waveformPreview = stem.asset.waveformPreview;

            const auto availableDuration = juce::jmax(0.0, stem.asset.durationSec - stemClip.sourceOffsetSec);
            stemClip.durationSec = result.sourceClip.durationSec > 0.0
                ? juce::jmin(result.sourceClip.durationSec, availableDuration > 0.0 ? availableDuration : result.sourceClip.durationSec)
                : availableDuration;
            if (stemClip.durationSec <= 0.0)
                stemClip.durationSec = stem.asset.durationSec;

            updatedProject.sampleClips.push_back(std::move(stemClip));
        }

        applyProjectStateEdit(updatedProject, "Separate Sample Clip To Stems");
        trackTable.selectRow(juce::jlimit(0,
                                          juce::jmax(0, static_cast<int>(documentState.project.tracks.size()) - 1),
                                          insertionTrackIndex));

        statusLabel.setText("Separated stems into " + juce::String(stemCount) + " audio track"
                                + (stemCount == 1 ? "" : "s") + ".",
                            juce::dontSendNotification);
        appendActivityLog("Stem Separation",
                          "Separated audio clip into "
                              + juce::String(stemCount)
                              + " stem(s)\nOutput: "
                              + result.outputDirectory.getFullPathName());
    }
    catch (const std::exception& exc)
    {
        juce::AlertWindow::showMessageBoxAsync(juce::AlertWindow::WarningIcon,
                                               "Stem Separation Failed",
                                               juce::String::fromUTF8(exc.what()));
        statusLabel.setText("Stem separation failed.", juce::dontSendNotification);
        appendActivityLog("Stem Separation Error", juce::String::fromUTF8(exc.what()));
    }

    finishBusyState();
}

void StudioShellComponent::updateAiStatusSummary()
{
    const auto& theme = themeSpecForIndex(currentThemeIndex);
    auto text = aiClient.authStatus();
    if (aiComposeBusy)
        text = (aiComposeBusyDetail.isNotEmpty() ? aiComposeBusyDetail : juce::String("AI processing...")) + "   " + text;
    else if (aceStepGenerationBusy)
        text = (aceStepBusyDetail.isNotEmpty() ? aceStepBusyDetail : juce::String("ACE-Step processing..."))
            + "   "
            + aceStepClient.statusSummary();

    aiStatusSummaryLabel.setColour(juce::Label::textColourId,
                                   (aiComposeBusy || aceStepGenerationBusy) ? theme.infoText
                                                                            : (aiClient.isEnabled() ? theme.successText
                                                                                                    : theme.warningText));
    aiStatusSummaryLabel.setText(text, juce::dontSendNotification);
}

void StudioShellComponent::applyAiComposeResult(const AIComposeResult& result, int requestedBars)
{
    if (result.tracks.empty())
        throw std::runtime_error("AI returned no tracks.");

    syncBundledRackCatalogInProject();

    auto updatedProject = documentState.project;
    const auto gridTick = aiComposeGridTick(updatedProject);
    const auto projectBarTicks = ticksPerBar(updatedProject);
    const auto requestedBarTicks = juce::jmax(projectBarTicks, juce::jmax(1, requestedBars) * projectBarTicks);
    auto maxEndTick = juce::jmax(projectBarTicks, juce::jmax(updatedProject.rightLocatorTick, requestedBarTicks));
    auto preferredTrackSelection = getSelectedTrackIndex();

    const auto buildPatternFromSource = [&updatedProject, gridTick, projectBarTicks] (const AIComposeTrack& sourceTrack,
                                                                                       const juce::String& fallbackName,
                                                                                       MidiPattern& patternOut)
    {
        MidiPattern pattern;
        pattern.id = juce::Uuid().toString();
        pattern.name = sourceTrack.name.trim().isNotEmpty() ? sourceTrack.name.trim() : fallbackName;
        auto patternEndTick = 0;
        pattern.notes.reserve(sourceTrack.notes.size());
        for (const auto& sourceNote : sourceTrack.notes)
        {
            const auto rawStartTick = juce::jmax(0, static_cast<int>(std::llround(sourceNote.startBeat * kTicksPerBeat)));
            const auto rawDurationTick = juce::jmax(1, static_cast<int>(std::llround(sourceNote.durationBeat * kTicksPerBeat)));
            auto quantizedStartTick = aiComposeRoundToGrid(rawStartTick, gridTick);
            auto quantizedEndTick = aiComposeRoundToGrid(rawStartTick + rawDurationTick, gridTick);
            if (quantizedEndTick <= quantizedStartTick)
                quantizedEndTick = quantizedStartTick + juce::jmax(1, gridTick);

            MidiNote note;
            note.startTick = quantizedStartTick;
            note.durationTick = juce::jmax(1, quantizedEndTick - quantizedStartTick);
            note.pitch = aiComposeQuantizedPitch(updatedProject, sourceNote.pitch);
            note.velocity = juce::jlimit(1, 127, sourceNote.velocity);
            patternEndTick = juce::jmax(patternEndTick, note.startTick + note.durationTick);
            pattern.notes.push_back(note);
        }

        if (pattern.notes.empty())
            return false;

        pattern.lengthTicks = juce::jmax(projectBarTicks, patternEndTick);
        patternOut = std::move(pattern);
        return true;
    };

    const auto appendPatternSection = [&updatedProject, &maxEndTick] (int trackIndex, MidiPattern pattern, int sectionStartTick)
    {
        updatedProject.midiPatterns.push_back(std::move(pattern));
        const auto& storedPattern = updatedProject.midiPatterns.back();

        MidiSection section;
        section.trackIndex = trackIndex;
        section.startTick = juce::jmax(0, sectionStartTick);
        section.lengthTicks = storedPattern.lengthTicks;
        section.name = storedPattern.name;
        section.patternId = storedPattern.id;
        updatedProject.midiSections.push_back(std::move(section));

        maxEndTick = juce::jmax(maxEndTick,
                                updatedProject.midiSections.back().startTick + updatedProject.midiSections.back().lengthTicks);
    };

    const auto trackIsEligible = [&updatedProject] (int trackIndex)
    {
        return juce::isPositiveAndBelow(trackIndex, static_cast<int>(updatedProject.tracks.size()))
            && updatedProject.tracks[static_cast<size_t>(trackIndex)].trackType.equalsIgnoreCase("instrument");
    };

    switch (aiComposeRequestedMode)
    {
        case AIComposeTargetMode::replaceAllTracks:
        {
            updatedProject.tracks.clear();
            updatedProject.midiPatterns.clear();
            updatedProject.midiSections.clear();
            updatedProject.sampleClips.clear();
            updatedProject.leftLocatorTick = 0;
            updatedProject.playheadTick = 0;
            maxEndTick = requestedBarTicks;

            for (const auto& sourceTrack : result.tracks)
            {
                TrackState track;
                track.name = sourceTrack.name.trim().isNotEmpty()
                    ? sourceTrack.name.trim()
                    : "AI Track " + juce::String(static_cast<int>(updatedProject.tracks.size()) + 1);
                track.trackType = "instrument";
                track.instrument = sourceTrack.instrument.trim().isNotEmpty() ? sourceTrack.instrument.trim() : track.name;
                track.midiProgram = defaultMidiProgramForInstrumentName(track.instrument);
                track.synthProfile = inferSynthProfile(track.instrument, track.midiProgram);

                const auto instrumentName = track.instrument.toLowerCase();
                if (track.synthProfile == "noise_kit" || instrumentName.contains("drum") || instrumentName.contains("kit"))
                {
                    track.midiChannel = 9;
                    if (track.midiProgram == 0)
                        track.midiProgram = 112;
                }
                else
                {
                    track.midiChannel = static_cast<int>(updatedProject.tracks.size()) % 16;
                    if (track.midiChannel == 9)
                        track.midiChannel = 10;
                }

                MidiPattern pattern;
                if (!buildPatternFromSource(sourceTrack, track.name, pattern))
                    continue;

                materialiseNativeInstrumentTrack(updatedProject, track);
                updatedProject.tracks.push_back(track);
                appendPatternSection(static_cast<int>(updatedProject.tracks.size()) - 1, std::move(pattern), 0);
            }

            if (updatedProject.tracks.empty())
                throw std::runtime_error("AI returned invalid track data.");

            preferredTrackSelection = 0;
            break;
        }

        case AIComposeTargetMode::replaceCurrentTrack:
        {
            if (aiComposeRequestedTargetTracks.empty() || !trackIsEligible(aiComposeRequestedTargetTracks.front()))
                throw std::runtime_error("Selected track is no longer available for AI replacement.");

            const auto targetTrackIndex = aiComposeRequestedTargetTracks.front();
            updatedProject.midiSections.erase(std::remove_if(updatedProject.midiSections.begin(),
                                                             updatedProject.midiSections.end(),
                                                             [targetTrackIndex] (const MidiSection& section)
                                                             {
                                                                 return section.trackIndex == targetTrackIndex;
                                                             }),
                                              updatedProject.midiSections.end());
            updatedProject.sampleClips.erase(std::remove_if(updatedProject.sampleClips.begin(),
                                                            updatedProject.sampleClips.end(),
                                                            [targetTrackIndex] (const SampleClip& clip)
                                                            {
                                                                return clip.trackIndex == targetTrackIndex;
                                                            }),
                                             updatedProject.sampleClips.end());

            MidiPattern pattern;
            if (!buildPatternFromSource(result.tracks.front(),
                                        updatedProject.tracks[static_cast<size_t>(targetTrackIndex)].name + " AI",
                                        pattern))
            {
                throw std::runtime_error("AI returned no usable notes for the selected track.");
            }

            appendPatternSection(targetTrackIndex, std::move(pattern), 0);
            preferredTrackSelection = targetTrackIndex;
            break;
        }

        case AIComposeTargetMode::addToCurrentTrack:
        {
            if (aiComposeRequestedTargetTracks.empty() || !trackIsEligible(aiComposeRequestedTargetTracks.front()))
                throw std::runtime_error("Selected track is no longer available for AI insertion.");

            const auto targetTrackIndex = aiComposeRequestedTargetTracks.front();
            MidiPattern pattern;
            if (!buildPatternFromSource(result.tracks.front(),
                                        updatedProject.tracks[static_cast<size_t>(targetTrackIndex)].name + " AI",
                                        pattern))
            {
                throw std::runtime_error("AI returned no usable notes for the selected track.");
            }

            appendPatternSection(targetTrackIndex, std::move(pattern), aiComposeRequestedInsertTick);
            preferredTrackSelection = targetTrackIndex;
            break;
        }

        case AIComposeTargetMode::addToAllTracks:
        {
            auto insertedTrackCount = 0;
            const auto trackCount = juce::jmin(static_cast<int>(aiComposeRequestedTargetTracks.size()),
                                               static_cast<int>(result.tracks.size()));
            for (int index = 0; index < trackCount; ++index)
            {
                const auto targetTrackIndex = aiComposeRequestedTargetTracks[static_cast<size_t>(index)];
                if (!trackIsEligible(targetTrackIndex))
                    continue;

                MidiPattern pattern;
                if (!buildPatternFromSource(result.tracks[static_cast<size_t>(index)],
                                            updatedProject.tracks[static_cast<size_t>(targetTrackIndex)].name + " AI",
                                            pattern))
                {
                    continue;
                }

                appendPatternSection(targetTrackIndex, std::move(pattern), aiComposeRequestedInsertTick);
                ++insertedTrackCount;
            }

            if (insertedTrackCount == 0)
                throw std::runtime_error("AI returned no usable notes for the existing tracks.");

            if (juce::isPositiveAndBelow(preferredTrackSelection, static_cast<int>(updatedProject.tracks.size()))
                && !updatedProject.tracks[static_cast<size_t>(preferredTrackSelection)].trackType.equalsIgnoreCase("instrument"))
            {
                preferredTrackSelection = aiComposeRequestedTargetTracks.empty() ? -1 : aiComposeRequestedTargetTracks.front();
            }
            break;
        }
    }

    updatedProject.rightLocatorTick = juce::jmax(projectBarTicks, maxEndTick);
    updatedProject.recalculateTimeFields();
    applyProjectStateEdit(updatedProject, "AI Compose");
    if (juce::isPositiveAndBelow(preferredTrackSelection, static_cast<int>(documentState.project.tracks.size())))
        setSelectedTrackIndex(preferredTrackSelection);
    else if (!documentState.project.tracks.empty())
        trackTable.selectRow(0);
    if (pianoRoll != nullptr)
        pianoRoll->grabKeyboardFocus();
}

void StudioShellComponent::syncBundledRackCatalogInProject()
{
    juce::StringArray scanFolderPaths;
    const auto addScanFolder = [&scanFolderPaths] (const juce::String& rawFolder)
    {
        auto normalisedFolder = rawFolder.trim();
        if (normalisedFolder.isEmpty())
            return;

        normalisedFolder = juce::File(normalisedFolder).getFullPathName();
        for (const auto& existing : scanFolderPaths)
        {
            if (existing.equalsIgnoreCase(normalisedFolder))
                return;
        }

        scanFolderPaths.add(normalisedFolder);
    };

    addScanFolder(defaultVstFolderPath());
    for (const auto& folderPath : userManagedVstFolderPaths())
        addScanFolder(folderPath);

    std::vector<VstInstrument> discoveredEntries;
    for (const auto& folderPath : scanFolderPaths)
    {
        const auto folderEntries = discoverVstCatalogInDirectory(juce::File(folderPath));
        discoveredEntries.insert(discoveredEntries.end(), folderEntries.begin(), folderEntries.end());
    }

    auto dedupeEntriesByPath = [] (std::vector<VstInstrument>& entries)
    {
        std::sort(entries.begin(),
                  entries.end(),
                  [] (const VstInstrument& lhs, const VstInstrument& rhs)
                  {
                      return lhs.path.compareIgnoreCase(rhs.path) < 0;
                  });

        entries.erase(std::unique(entries.begin(),
                                  entries.end(),
                                  [] (const VstInstrument& lhs, const VstInstrument& rhs)
                                  {
                                      return lhs.path.equalsIgnoreCase(rhs.path);
                                  }),
                      entries.end());
    };

    dedupeEntriesByPath(discoveredEntries);

    std::vector<VstInstrument> mergedEntries = discoveredEntries;
    std::sort(mergedEntries.begin(),
              mergedEntries.end(),
              [] (const VstInstrument& lhs, const VstInstrument& rhs)
              {
                  return lhs.name.compareIgnoreCase(rhs.name) < 0;
              });

    juce::StringArray updatedPaths;
    for (const auto& entry : mergedEntries)
    {
        const auto entryPath = entry.path.trim();
        if (entryPath.isNotEmpty())
            updatedPaths.addIfNotAlreadyThere(entryPath);
    }

    bool changed = documentState.project.vstRack.size() != mergedEntries.size()
        || documentState.project.vstiPaths.size() != updatedPaths.size();

    if (!changed)
    {
        for (int index = 0; index < static_cast<int>(mergedEntries.size()); ++index)
        {
            const auto& existing = documentState.project.vstRack[static_cast<size_t>(index)];
            const auto& updated = mergedEntries[static_cast<size_t>(index)];
            if (!existing.path.equalsIgnoreCase(updated.path)
                || existing.name != updated.name
                || existing.pluginName != updated.pluginName
                || existing.isInstrument != updated.isInstrument
                || existing.isEffect != updated.isEffect
                || existing.category != updated.category
                || existing.hostSupported != updated.hostSupported
                || existing.hostError != updated.hostError)
            {
                changed = true;
                break;
            }
        }
    }

    if (!changed)
    {
        for (int index = 0; index < updatedPaths.size(); ++index)
        {
            if (!documentState.project.vstiPaths[index].equalsIgnoreCase(updatedPaths[index]))
            {
                changed = true;
                break;
            }
        }
    }

    if (!changed)
        return;

    documentState.project.vstRack = std::move(mergedEntries);
    documentState.project.vstiPaths = updatedPaths;
    normaliseProject(documentState.project);
}

void StudioShellComponent::refreshRackCatalog()
{
    const auto before = fingerprint(documentState.project);
    syncBundledRackCatalogInProject();
    const auto after = fingerprint(documentState.project);

    if (before != after)
    {
        markDirty();
        refreshUi();
        statusLabel.setText("Refreshed native VST catalog.", juce::dontSendNotification);
        return;
    }

    refreshFloatingWindows();
    statusLabel.setText("Native VST catalog is already up to date.", juce::dontSendNotification);
}

void StudioShellComponent::assignSelectedTrackRackByReference(const juce::String& reference)
{
    const auto trimmedReference = reference.trim();
    if (trimmedReference.isEmpty())
        return;

    const auto selected = getSelectedTrackIndex();
    if (!juce::isPositiveAndBelow(selected, static_cast<int>(documentState.project.tracks.size())))
        return;

    const auto rackIndex = findRackInstrumentIndexByReference(documentState.project, trimmedReference);
    const auto entryLabel = rackIndex >= 0
        ? [&project = documentState.project, rackIndex]
        {
            const auto& entry = project.vstRack[static_cast<size_t>(rackIndex)];
            if (entry.name.isNotEmpty())
                return entry.name;
            if (entry.pluginName.isNotEmpty())
                return entry.pluginName;
            if (entry.path.isNotEmpty())
                return juce::File(entry.path).getFileNameWithoutExtension();
            return juce::String();
        }()
        : trimmedReference;

    auto updatedTrack = documentState.project.tracks[static_cast<size_t>(selected)];
    updatedTrack.instrumentMode = "VSTI Rack";
    updatedTrack.rackVst = entryLabel;
    if (updatedTrack.instrument.trim().isEmpty()
        || updatedTrack.instrument.trim().equalsIgnoreCase(displayRackName(documentState.project,
                                                                          documentState.project.tracks[static_cast<size_t>(selected)])))
    {
        updatedTrack.instrument = entryLabel;
    }
    updatedTrack.vstiStatePath.clear();
    updatedTrack.vstiStateBase64.clear();
    updatedTrack.vstiParameters.clear();
    updatedTrack.synthProfile = "vst_instrument";

    applyTrackStateEdit(selected, updatedTrack, "Assign Rack Plugin");
    trackTable.updateContent();
    trackTable.repaint(trackTable.getRowPosition(selected, true));
    refreshInspector();
    refreshFloatingWindows(false);
    statusLabel.setText("Assigned native rack: " + entryLabel + ".", juce::dontSendNotification);
}

void StudioShellComponent::clearSelectedTrackRackAssignment()
{
    const auto selected = getSelectedTrackIndex();
    if (!juce::isPositiveAndBelow(selected, static_cast<int>(documentState.project.tracks.size())))
        return;

    auto updatedTrack = documentState.project.tracks[static_cast<size_t>(selected)];
    updatedTrack.rackVst.clear();
    if (updatedTrack.instrumentMode.trim().equalsIgnoreCase("VSTI Rack"))
        updatedTrack.instrumentMode = "General MIDI";
    updatedTrack.vstiStatePath.clear();
    updatedTrack.vstiStateBase64.clear();
    updatedTrack.vstiParameters.clear();
    updatedTrack.synthProfile = inferSynthProfile(updatedTrack.instrument, updatedTrack.midiProgram);
    applyTrackStateEdit(selected, updatedTrack, "Clear Rack Assignment");
    statusLabel.setText("Cleared native rack assignment for " + updatedTrack.name + ".", juce::dontSendNotification);
}

void StudioShellComponent::materialiseSelectedTrackRackAssignment()
{
    const auto selected = getSelectedTrackIndex();
    if (!juce::isPositiveAndBelow(selected, static_cast<int>(documentState.project.tracks.size())))
        return;

    auto updatedTrack = documentState.project.tracks[static_cast<size_t>(selected)];
    if (!materialiseNativeInstrumentTrack(documentState.project, updatedTrack))
    {
        juce::AlertWindow::showMessageBoxAsync(juce::AlertWindow::InfoIcon,
                                               "No Native Rack Available",
                                               "Could not find a suitable native rack instrument for the selected track.");
        return;
    }

    applyTrackStateEdit(selected, updatedTrack, "Auto Assign Rack");
    statusLabel.setText("Auto-assigned native rack for " + updatedTrack.name + ".", juce::dontSendNotification);
}

void StudioShellComponent::clearSelectedTrackRenderedAudioPath()
{
    const auto selected = getSelectedTrackIndex();
    if (!juce::isPositiveAndBelow(selected, static_cast<int>(documentState.project.tracks.size())))
        return;

    auto updatedTrack = documentState.project.tracks[static_cast<size_t>(selected)];
    if (updatedTrack.renderedAudioPath.trim().isEmpty())
        return;

    updatedTrack.renderedAudioPath.clear();
    applyTrackStateEdit(selected, updatedTrack, "Clear Render Path");
    statusLabel.setText("Cleared rendered audio path for " + updatedTrack.name + ".", juce::dontSendNotification);
}

void StudioShellComponent::scheduleSelectedTrackRackPreviewWarmup(int delayMs)
{
    if (rackPreviewWarmupPending || projectPreviewRunning || rackPreviewRunning)
        return;

    const auto selected = getSelectedTrackIndex();
    const auto* track = getSelectedTrack();
    if (selected < 0 || track == nullptr)
        return;

    if (resolveRackPluginPath(documentState.project, *track).isEmpty())
        return;

    rackPreviewWarmupPending = true;
    juce::Timer::callAfterDelay(juce::jmax(0, delayMs),
                                [safeThis = juce::Component::SafePointer<StudioShellComponent>(this), selected]
                                {
                                    if (safeThis == nullptr)
                                        return;

                                    safeThis->rackPreviewWarmupPending = false;

                                    if (safeThis->projectPreviewRunning || safeThis->rackPreviewRunning)
                                        return;

                                    const auto currentIndex = safeThis->getSelectedTrackIndex();
                                    const auto* currentTrack = safeThis->getSelectedTrack();
                                    if (currentTrack == nullptr || currentIndex != selected)
                                        return;

                                    const auto result = safeThis->ensureNativeAudioEnginePrepared(false);
                                    if (result.failed())
                                    {
                                        safeThis->statusLabel.setText("Preview warmup failed: " + result.getErrorMessage(),
                                                                      juce::dontSendNotification);
                                    }
                                });
}

bool StudioShellComponent::virtualPianoShortcutsEnabled() const
{
    for (auto* focused = juce::Component::getCurrentlyFocusedComponent();
         focused != nullptr;
         focused = focused->getParentComponent())
    {
        if (dynamic_cast<juce::TextEditor*>(focused) != nullptr
            || dynamic_cast<juce::ComboBox*>(focused) != nullptr)
        {
            return false;
        }
    }

    return true;
}

bool StudioShellComponent::tryHandleVirtualPianoShortcut(const juce::KeyPress& key)
{
    const auto modifiers = key.getModifiers();
    if (modifiers.isAltDown() || modifiers.isCtrlDown() || modifiers.isCommandDown())
        return false;

    if (!virtualPianoShortcutsEnabled())
        return false;

    const auto shortcut = normaliseVirtualPianoShortcutKey(key);
    if (shortcut.isEmpty())
        return false;

    for (const auto& spec : virtualPianoKeySpecs())
    {
        if (shortcut.equalsIgnoreCase(spec.primary))
        {
            insertVirtualKeyboardNote(spec.pitch, true);
            return true;
        }

        for (const auto* alias : spec.aliases)
        {
            if (shortcut.equalsIgnoreCase(alias))
            {
                insertVirtualKeyboardNote(spec.pitch, true);
                return true;
            }
        }
    }

    return false;
}

void StudioShellComponent::insertVirtualKeyboardNote(int pitch, bool fromShortcut)
{
    insertLiveMidiNote(pitch, 100, fromShortcut, true);
}

juce::Result StudioShellComponent::previewSelectedTrackMidiNoteOn(int pitch, int velocity)
{
    const auto selected = getSelectedTrackIndex();
    const auto* track = getSelectedTrack();
    if (selected < 0 || track == nullptr)
        return juce::Result::fail("No track selected.");

    auto result = ensureNativeAudioEnginePrepared(projectPreviewRunning || rackPreviewRunning);
    if (result.failed())
    {
        scheduleSelectedTrackRackPreviewWarmup(0);
        return result;
    }

    return nativeVstHost.noteOnAudioEngineTrack(selected,
                                                juce::jlimit(0, 127, pitch),
                                                juce::jlimit(1, 16, track->midiChannel + 1),
                                                static_cast<float>(juce::jlimit(1, 127, velocity)) / 127.0f);
}

void StudioShellComponent::previewSelectedTrackMidiNoteOff(int pitch, int velocity)
{
    const auto selected = getSelectedTrackIndex();
    const auto* track = getSelectedTrack();
    if (selected < 0 || track == nullptr || !nativeVstHost.isReady())
        return;

    nativeVstHost.noteOffAudioEngineTrack(selected,
                                          juce::jlimit(0, 127, pitch),
                                          juce::jlimit(1, 16, track->midiChannel + 1),
                                          static_cast<float>(juce::jlimit(0, 127, velocity)) / 127.0f);
}

void StudioShellComponent::stopSelectedTrackMidiPreview()
{
    const auto selected = getSelectedTrackIndex();
    if (selected < 0 || !nativeVstHost.isReady())
        return;

    nativeVstHost.allNotesOffAudioEngineTrack(selected);
}

void StudioShellComponent::openSelectedTrackRackEditor()
{
    openTrackRackEditor(getSelectedTrackIndex());
}

void StudioShellComponent::openTrackRackEditor(int trackIndex)
{
    if (!juce::isPositiveAndBelow(trackIndex, static_cast<int>(documentState.project.tracks.size())))
        return;

    const auto* track = &documentState.project.tracks[static_cast<size_t>(trackIndex)];
    RackEditorSession* session = nullptr;
    const auto result = ensureRackEditorSessionReadyForTrack(trackIndex, *track, session);
    if (result.failed())
    {
        juce::AlertWindow::showMessageBoxAsync(juce::AlertWindow::WarningIcon,
                                               "Native Host Error",
                                               result.getErrorMessage());
        return;
    }

    jassert(session != nullptr);
    const auto openResult = session != nullptr
        ? nativeVstHost.openAudioEngineTrackEditor(trackIndex)
        : juce::Result::fail("The native editor session could not be prepared.");
    if (openResult.failed())
    {
        juce::AlertWindow::showMessageBoxAsync(juce::AlertWindow::WarningIcon,
                                               "Editor Open Failed",
                                               openResult.getErrorMessage());
        return;
    }

    if (session != nullptr)
        session->editorOpen = true;
    refreshPollingTimerState();
    updateEditorState();
    trackTable.repaint();
    statusLabel.setText("Opened native rack editor for " + track->name + ".", juce::dontSendNotification);
    appendActivityLog("VST Editor", "Opened editor for track: " + track->name);
}

void StudioShellComponent::openTrackEffectEditorFromMixer(int trackIndex, int effectIndex)
{
    if (!juce::isPositiveAndBelow(trackIndex, static_cast<int>(documentState.project.tracks.size())))
        return;

    const auto& track = documentState.project.tracks[static_cast<size_t>(trackIndex)];
    if (!juce::isPositiveAndBelow(effectIndex, track.vstFxChain.size()))
        return;

    auto effectName = track.vstFxChain[effectIndex];
    for (const auto& entry : documentState.project.vstRack)
    {
        if (!entry.isEffect)
            continue;

        if (entry.path.equalsIgnoreCase(effectName)
            || entry.name.equalsIgnoreCase(effectName)
            || entry.pluginName.equalsIgnoreCase(effectName))
        {
            if (entry.name.trim().isNotEmpty())
                effectName = entry.name.trim();
            else if (entry.pluginName.trim().isNotEmpty())
                effectName = entry.pluginName.trim();
            break;
        }
    }

    if (effectName.containsAnyOf("\\/"))
        effectName = juce::File(effectName).getFileNameWithoutExtension();

    auto result = ensureNativeAudioEnginePrepared(projectPreviewRunning || rackPreviewRunning);
    if (result.failed())
    {
        juce::AlertWindow::showMessageBoxAsync(juce::AlertWindow::WarningIcon,
                                               "Native Host Error",
                                               result.getErrorMessage());
        return;
    }

    result = nativeVstHost.openAudioEngineTrackEffectEditor(trackIndex, effectIndex);
    if (result.failed())
    {
        juce::AlertWindow::showMessageBoxAsync(juce::AlertWindow::WarningIcon,
                                               "FX Editor Open Failed",
                                               result.getErrorMessage());
        return;
    }

    refreshPollingTimerState();
    updateEditorState();
    statusLabel.setText("Opened FX editor for " + effectName + " on " + track.name + ".", juce::dontSendNotification);
    appendActivityLog("FX Editor",
                      "Opened track FX editor\nTrack: " + track.name + "\nFX: " + effectName);
}

void StudioShellComponent::openMasterEffectEditorFromMixer(int effectIndex)
{
    if (!juce::isPositiveAndBelow(effectIndex, documentState.project.masterFxChain.size()))
        return;

    auto effectName = documentState.project.masterFxChain[effectIndex];
    for (const auto& entry : documentState.project.vstRack)
    {
        if (!entry.isEffect)
            continue;

        if (entry.path.equalsIgnoreCase(effectName)
            || entry.name.equalsIgnoreCase(effectName)
            || entry.pluginName.equalsIgnoreCase(effectName))
        {
            if (entry.name.trim().isNotEmpty())
                effectName = entry.name.trim();
            else if (entry.pluginName.trim().isNotEmpty())
                effectName = entry.pluginName.trim();
            break;
        }
    }

    if (effectName.containsAnyOf("\\/"))
        effectName = juce::File(effectName).getFileNameWithoutExtension();

    auto result = ensureNativeAudioEnginePrepared(projectPreviewRunning || rackPreviewRunning);
    if (result.failed())
    {
        juce::AlertWindow::showMessageBoxAsync(juce::AlertWindow::WarningIcon,
                                               "Native Host Error",
                                               result.getErrorMessage());
        return;
    }

    result = nativeVstHost.openAudioEngineMasterEffectEditor(effectIndex);
    if (result.failed())
    {
        juce::AlertWindow::showMessageBoxAsync(juce::AlertWindow::WarningIcon,
                                               "FX Editor Open Failed",
                                               result.getErrorMessage());
        return;
    }

    refreshPollingTimerState();
    updateEditorState();
    statusLabel.setText("Opened master FX editor for " + effectName + ".", juce::dontSendNotification);
    appendActivityLog("FX Editor",
                      "Opened master FX editor\nFX: " + effectName);
}

void StudioShellComponent::openSharedEffectBusEditor(const juce::String& busId)
{
    const auto trimmedBusId = busId.trim();
    if (trimmedBusId.isEmpty())
        return;

    const auto busIt = std::find_if(documentState.project.sharedFxBuses.begin(),
                                    documentState.project.sharedFxBuses.end(),
                                    [trimmedBusId] (const SharedEffectBusState& bus)
                                    {
                                        return bus.id.equalsIgnoreCase(trimmedBusId);
                                    });
    if (busIt == documentState.project.sharedFxBuses.end())
        return;

    auto effectName = busIt->name.trim().isNotEmpty() ? busIt->name.trim() : "FX Bus";
    auto result = ensureNativeAudioEnginePrepared(projectPreviewRunning || rackPreviewRunning);
    if (result.failed())
    {
        juce::AlertWindow::showMessageBoxAsync(juce::AlertWindow::WarningIcon,
                                               "Native Host Error",
                                               result.getErrorMessage());
        return;
    }

    result = nativeVstHost.openAudioEngineSharedEffectBusEditor(trimmedBusId);
    if (result.failed())
    {
        juce::AlertWindow::showMessageBoxAsync(juce::AlertWindow::WarningIcon,
                                               "FX Editor Open Failed",
                                               result.getErrorMessage());
        return;
    }

    refreshPollingTimerState();
    updateEditorState();
    statusLabel.setText("Opened shared FX editor for " + effectName + ".", juce::dontSendNotification);
    appendActivityLog("FX Editor",
                      "Opened shared FX editor\nBus: " + effectName);
}

void StudioShellComponent::saveSelectedTrackRackState()
{
    const auto selected = getSelectedTrackIndex();
    if (selected < 0)
        return;

    const auto* track = getSelectedTrack();
    if (track == nullptr)
        return;

    auto* rackEditorSession = findRackEditorSession(selected);
    auto result = ensureNativeAudioEnginePrepared(false);
    if (result.failed())
    {
        juce::AlertWindow::showMessageBoxAsync(juce::AlertWindow::WarningIcon,
                                               "Native Host Error",
                                               result.getErrorMessage());
        return;
    }

    NativeVstHostSession::RackParameterSnapshot rackSnapshot;
    if (nativeVstHost.queryAudioEngineTrackParameterSnapshot(selected, rackSnapshot).wasOk())
    {
        if (rackEditorSession != nullptr)
            rackEditorSession->lastStateGeneration = rackSnapshot.stateGeneration;
        syncTrackRackParametersFromValues(selected, rackSnapshot.parameterValues, true);
    }

    const auto stateFile = suggestTrackStateFile(selected, *track);
    if (!stateFile.getParentDirectory().createDirectory())
    {
        juce::AlertWindow::showMessageBoxAsync(juce::AlertWindow::WarningIcon,
                                               "State Save Failed",
                                               "Could not create the native state folder:\n" + stateFile.getParentDirectory().getFullPathName());
        return;
    }

    result = nativeVstHost.saveAudioEngineTrackPluginState(selected, stateFile.getFullPathName());
    if (result.failed())
    {
        juce::AlertWindow::showMessageBoxAsync(juce::AlertWindow::WarningIcon,
                                               "State Save Failed",
                                               result.getErrorMessage());
        return;
    }

    auto updatedTrack = *track;
    updatedTrack.vstiStatePath = stateFile.getFullPathName();
    applyTrackStateEdit(selected, updatedTrack, "Save Rack State");
    statusLabel.setText("Saved native rack state for " + track->name + ".", juce::dontSendNotification);
}

void StudioShellComponent::playSelectedTrackThroughRack()
{
    const auto* track = getSelectedTrack();
    if (track == nullptr)
        return;

    if (track->notes.empty())
    {
        juce::AlertWindow::showMessageBoxAsync(juce::AlertWindow::WarningIcon,
                                               "No Notes To Play",
                                               "The selected track does not have any MIDI notes yet.");
        return;
    }

    const auto selected = getSelectedTrackIndex();
    if (projectPreviewRunning)
        stopRackPreview();

    auto previewProject = documentState.project;
    for (int trackIndex = 0; trackIndex < static_cast<int>(previewProject.tracks.size()); ++trackIndex)
    {
        if (trackIndex == selected)
            continue;

        previewProject.tracks[static_cast<size_t>(trackIndex)].mute = true;
        previewProject.tracks[static_cast<size_t>(trackIndex)].solo = false;
    }
    previewProject.tracks[static_cast<size_t>(selected)].mute = false;
    previewProject.tracks[static_cast<size_t>(selected)].solo = true;
    previewProject.recalculateTimeFields();

    auto result = nativeVstHost.ensureCreated();
    if (result.failed())
    {
        juce::AlertWindow::showMessageBoxAsync(juce::AlertWindow::WarningIcon,
                                               "Project Playback Setup Failed",
                                               result.getErrorMessage());
        return;
    }

    result = nativeVstHost.setAudioEngineState(previewProject);
    if (result.failed())
    {
        juce::AlertWindow::showMessageBoxAsync(juce::AlertWindow::WarningIcon,
                                               "Project Playback Setup Failed",
                                               result.getErrorMessage());
        return;
    }

    result = nativeVstHost.startAudioEngine(documentState.project.playheadTick);
    if (result.failed())
    {
        juce::AlertWindow::showMessageBoxAsync(juce::AlertWindow::WarningIcon,
                                               "Playback Failed",
                                               result.getErrorMessage());
        return;
    }

    rackPreviewRunning = true;
    projectPreviewRunning = false;
    pendingLiveRackParameterEngineSyncTrack = -1;
    audioEngineStateValid = false;
    audioEngineStateDirty = true;
    refreshPollingTimerState();
    refreshPlaybackToggleButton();
    updateEditorState();
    statusLabel.setText("Playing selected rack track from tick " + juce::String(documentState.project.playheadTick) + ".", juce::dontSendNotification);
    appendActivityLog("Playback",
                      "Started selected-track preview\nTrack: "
                          + track->name
                          + "\nTick: "
                          + juce::String(documentState.project.playheadTick));
}

void StudioShellComponent::playFullProjectThroughNativeEngine()
{
    if (documentState.project.tracks.empty() && !documentState.project.metronomeEnabled)
        return;

    if (rackPreviewRunning)
        stopRackPreview();

    auto result = nativeVstHost.ensureCreated();
    if (result.failed())
    {
        juce::AlertWindow::showMessageBoxAsync(juce::AlertWindow::WarningIcon,
                                               "Native Host Error",
                                               result.getErrorMessage());
        return;
    }

    if (!audioEngineStateValid || audioEngineStateDirty)
    {
        result = nativeVstHost.setAudioEngineState(documentState.project);
        if (result.failed())
        {
            juce::AlertWindow::showMessageBoxAsync(juce::AlertWindow::WarningIcon,
                                                   "Project Playback Setup Failed",
                                                   result.getErrorMessage());
            return;
        }

        audioEngineStateValid = true;
        audioEngineStateDirty = false;
    }

    result = nativeVstHost.startAudioEngine(documentState.project.playheadTick);
    if (result.failed())
    {
        juce::AlertWindow::showMessageBoxAsync(juce::AlertWindow::WarningIcon,
                                               "Project Playback Failed",
                                               result.getErrorMessage());
        return;
    }

    rackPreviewRunning = false;
    projectPreviewRunning = true;
    playbackUiTickCounter = 0;
    pendingLiveRackParameterEngineSyncTrack = -1;
    refreshPollingTimerState();
    refreshPlaybackToggleButton();
    updateEditorState();
    statusLabel.setText("Playing native project preview from tick " + juce::String(documentState.project.playheadTick) + ".", juce::dontSendNotification);
    appendActivityLog("Playback", "Started project playback\nTick: " + juce::String(documentState.project.playheadTick));
}

void StudioShellComponent::stopRackPreview()
{
    if (projectPreviewRunning && !activeRealtimeRecordedNotes.empty())
        finishActiveRealtimeRecordedNotes(currentAudioEngineTransportTick());

    juce::Result result = juce::Result::ok();
    if (projectPreviewRunning || rackPreviewRunning)
        result = nativeVstHost.stopAudioEngine(true);

    if (result.failed())
    {
        juce::AlertWindow::showMessageBoxAsync(juce::AlertWindow::WarningIcon,
                                               "Stop Failed",
                                               result.getErrorMessage());
        return;
    }

    rackPreviewRunning = false;
    projectPreviewRunning = false;
    playbackUiTickCounter = 0;
    pendingLiveRackParameterEngineSyncTrack = -1;
    activeRealtimeRecordedNotes.clear();
    activeMidiInsertHeldPitches.clear();
    activeMidiInsertPatternId.clear();
    activeMidiInsertTrackIndex = -1;
    activeMidiInsertChordStartTick = -1;
    audioEngineStateValid = nativeVstHost.isReady();
    audioEngineStateDirty = false;
    refreshPollingTimerState();
    std::fill(trackMeterLevels.begin(), trackMeterLevels.end(), 0.0f);
    if (mixerComponent != nullptr)
        mixerComponent->refreshMeters();
    repaintTrackVolumeMeters();
    refreshPlaybackToggleButton();
    updateEditorState();
    statusLabel.setText("Stopped native preview playback.", juce::dontSendNotification);
    appendActivityLog("Playback", "Stopped playback.");
}

void StudioShellComponent::undo()
{
    if (!undoManager.canUndo())
        return;

    undoManager.undo();
    markDirty();
    refreshUi();
    statusLabel.setText("Undo.", juce::dontSendNotification);
}

void StudioShellComponent::redo()
{
    if (!undoManager.canRedo())
        return;

    undoManager.redo();
    markDirty();
    refreshUi();
    statusLabel.setText("Redo.", juce::dontSendNotification);
}

void StudioShellComponent::quantizeSelectedNotes()
{
    if (pianoRoll != nullptr && pianoRoll->quantizeSelected())
    {
        statusLabel.setText("Quantized selected notes.", juce::dontSendNotification);
        return;
    }

    const auto selectedSectionIndices = [this]
    {
        if (floatingArrangementOverview != nullptr
            && arrangementWindow != nullptr
            && arrangementWindow->isVisible())
        {
            auto indices = floatingArrangementOverview->getSelectedSectionIndices();
            if (!indices.empty())
                return indices;
        }

        if (panelsWindowContent != nullptr
            && panelsWindow != nullptr
            && panelsWindow->isVisible())
        {
            auto indices = panelsWindowContent->getSelectedSectionIndices();
            if (!indices.empty())
                return indices;
        }

        return arrangementOverview != nullptr
            ? arrangementOverview->getSelectedSectionIndices()
            : std::vector<int>{};
    }();

    if (selectedSectionIndices.empty())
        return;

    auto updatedProject = documentState.project;
    const auto quantizeStep = aiComposeGridTick(updatedProject);
    juce::StringArray processedPatternIds;
    bool changed = false;

    for (const auto sectionIndex : selectedSectionIndices)
    {
        if (!juce::isPositiveAndBelow(sectionIndex, static_cast<int>(updatedProject.midiSections.size())))
            continue;

        const auto& section = updatedProject.midiSections[static_cast<size_t>(sectionIndex)];
        if (section.patternId.trim().isEmpty() || processedPatternIds.contains(section.patternId))
            continue;

        processedPatternIds.add(section.patternId);
        auto* pattern = findMidiPattern(updatedProject, section.patternId);
        if (pattern == nullptr)
            continue;

        for (auto& note : pattern->notes)
        {
            const auto quantizedStartTick = aiComposeRoundToGrid(note.startTick, quantizeStep);
            const auto quantizedDurationTick = juce::jmax(quantizeStep,
                                                          aiComposeRoundToGrid(note.durationTick, quantizeStep));
            if (note.startTick != quantizedStartTick || note.durationTick != quantizedDurationTick)
            {
                note.startTick = quantizedStartTick;
                note.durationTick = quantizedDurationTick;
                changed = true;
            }
        }
    }

    if (!changed)
        return;

    applyProjectStateEdit(updatedProject,
                          processedPatternIds.size() > 1 ? "Quantize Pattern Clips" : "Quantize Pattern Clip");
    statusLabel.setText("Quantized notes in selected patterns.", juce::dontSendNotification);
}

void StudioShellComponent::setEditorToolMode(EditorToolMode mode)
{
    if (editorToolMode == mode)
        return;

    editorToolMode = mode;
    if (arrangementOverview != nullptr)
        arrangementOverview->refreshFromModel();
    if (floatingArrangementOverview != nullptr)
        floatingArrangementOverview->refreshFromModel();
    if (pianoRoll != nullptr)
        pianoRoll->setToolMode(mode);
    if (floatingPianoRollWorkspace != nullptr)
        floatingPianoRollWorkspace->refreshFromModel();
    if (panelsWindowContent != nullptr)
        panelsWindowContent->refreshFromModel();
    statusLabel.setText("Editor tool: " + editorToolModeLabel(mode) + ".", juce::dontSendNotification);
    if (pianoRoll != nullptr)
        pianoRoll->grabKeyboardFocus();
}

void StudioShellComponent::resetRackHostTracking()
{
    loadedRackEditorOpen = false;
    rackPreviewWarmupPending = false;
    pendingLiveRackParameterEngineSyncTrack = -1;
    refreshPollingTimerState();
}

bool StudioShellComponent::syncTrackRackParametersFromValues(int trackIndex,
                                                             const juce::NamedValueSet& hostParameters,
                                                             bool skipParameterEnginePush)
{
    if (!juce::isPositiveAndBelow(trackIndex, static_cast<int>(documentState.project.tracks.size())))
        return false;

    if (hostParameters.size() <= 0)
        return false;

    auto updatedTrack = documentState.project.tracks[static_cast<size_t>(trackIndex)];
    if (parameterSetSignature(updatedTrack.vstiParameters) == parameterSetSignature(hostParameters))
        return false;

    updatedTrack.vstiParameters = hostParameters;
    applyTrackStateSyncNoUndo(trackIndex,
                              updatedTrack,
                              isTrackBeingLiveEdited(trackIndex) && projectPreviewRunning && !skipParameterEnginePush,
                              skipParameterEnginePush);
    return true;
}

void StudioShellComponent::replaceTrackStateNoUndo(int trackIndex, const TrackState& updatedTrack)
{
    setTrackStateInternal(trackIndex, updatedTrack);
}

void StudioShellComponent::applyTrackStateSyncNoUndo(int trackIndex,
                                                     const TrackState& updatedTrack,
                                                     bool deferParameterEnginePush,
                                                     bool skipParameterEnginePush)
{
    if (trackIndex < 0 || trackIndex >= static_cast<int>(documentState.project.tracks.size()))
        return;

    auto before = documentState.project.tracks[static_cast<size_t>(trackIndex)];
    auto after = updatedTrack;
    normaliseTrack(before);
    normaliseTrack(after);

    if (fingerprint(before) == fingerprint(after))
        return;

    markDirty();
    setTrackStateInternal(trackIndex, after, deferParameterEnginePush, skipParameterEnginePush);
}

void StudioShellComponent::applyTrackStateEdit(int trackIndex, const TrackState& updatedTrack, const juce::String& actionName)
{
    if (trackIndex < 0 || trackIndex >= static_cast<int>(documentState.project.tracks.size()))
        return;

    auto before = documentState.project.tracks[static_cast<size_t>(trackIndex)];
    auto after = updatedTrack;
    normaliseTrack(before);
    normaliseTrack(after);

    if (fingerprint(before) == fingerprint(after))
        return;

    markDirty();
    if (actionName.isNotEmpty())
        undoManager.beginNewTransaction(actionName);
    else
        undoManager.beginNewTransaction();
    undoManager.perform(new TrackEditAction([this, trackIndex] (const TrackState& track)
                                            {
                                                setTrackStateInternal(trackIndex, track);
                                            },
                                            before,
                                            after),
                        actionName);
}

void StudioShellComponent::applySelectedTrackMutation(std::function<void(TrackState&)> mutation,
                                                      const juce::String& actionName)
{
    const auto selected = getSelectedTrackIndex();
    if (selected < 0)
        return;

    auto updatedTrack = documentState.project.tracks[static_cast<size_t>(selected)];
    mutation(updatedTrack);
    applyTrackStateEdit(selected, updatedTrack, actionName);
}

void StudioShellComponent::applyProjectStateEdit(const ProjectState& updatedProject, const juce::String& actionName)
{
    auto before = documentState.project;
    auto after = updatedProject;
    normaliseProject(before);
    normaliseProject(after);

    if (fingerprint(before) == fingerprint(after))
        return;

    markDirty();
    if (actionName.isNotEmpty())
        undoManager.beginNewTransaction(actionName);
    else
        undoManager.beginNewTransaction();
    undoManager.perform(new ProjectEditAction([this] (const ProjectState& project)
                                              {
                                                  setProjectStateInternal(project);
                                              },
                                              before,
                                              after),
                        actionName);
}

void StudioShellComponent::setTrackStateInternal(int trackIndex,
                                                 const TrackState& updatedTrack,
                                                 bool deferParameterEnginePush,
                                                 bool skipParameterEnginePush)
{
    if (trackIndex < 0 || trackIndex >= static_cast<int>(documentState.project.tracks.size()))
        return;

    const auto previousProject = documentState.project;
    const auto previousTrack = documentState.project.tracks[static_cast<size_t>(trackIndex)];
    documentState.project.tracks[static_cast<size_t>(trackIndex)] = updatedTrack;
    normaliseProject(documentState.project);
    audioEngineStateDirty = true;

    const auto& currentTrack = documentState.project.tracks[static_cast<size_t>(trackIndex)];
    const bool rackBindingChanged = !previousTrack.rackVst.equalsIgnoreCase(currentTrack.rackVst)
        || !previousTrack.instrumentMode.equalsIgnoreCase(currentTrack.instrumentMode)
        || !previousTrack.trackType.equalsIgnoreCase(currentTrack.trackType)
        || !previousTrack.vstiStatePath.equalsIgnoreCase(currentTrack.vstiStatePath);
    const bool noteContentChanged = noteTransportSignature(previousTrack.notes) != noteTransportSignature(currentTrack.notes);
    const bool parameterContentChanged = parameterSetSignature(previousTrack.vstiParameters) != parameterSetSignature(currentTrack.vstiParameters);
    const auto mixValueChanged = [] (double lhs, double rhs)
    {
        return std::abs(lhs - rhs) > 1.0e-6;
    };
    const bool fxStateChanged = previousTrack.vstFxBypassed != currentTrack.vstFxBypassed
        || previousTrack.vstFxChain != currentTrack.vstFxChain;
    const bool routingChanged = !previousTrack.routingTarget.equalsIgnoreCase(currentTrack.routingTarget);
    const bool mixStateChanged = mixValueChanged(previousTrack.volume, currentTrack.volume)
        || mixValueChanged(previousTrack.pan, currentTrack.pan)
        || mixValueChanged(previousTrack.vstiOutputGainDb, currentTrack.vstiOutputGainDb)
        || previousTrack.mute != currentTrack.mute
        || previousTrack.solo != currentTrack.solo
        || projectTrackIsAudible(previousProject, trackIndex) != projectTrackIsAudible(documentState.project, trackIndex);
    const auto previousRackPath = resolveRackPluginPath(previousProject, previousTrack);
    const auto currentRackPath = resolveRackPluginPath(documentState.project, currentTrack);
    const bool instrumentActivationChanged = (!previousRackPath.isEmpty() || !currentRackPath.isEmpty())
        && (previousTrack.notes.empty() != currentTrack.notes.empty());
    const bool requiresFullEngineState = rackBindingChanged
        || routingChanged
        || fxStateChanged
        || !previousTrack.renderedAudioPath.equalsIgnoreCase(currentTrack.renderedAudioPath)
        || instrumentActivationChanged;
    const bool parameterOnlyContentChanged = parameterContentChanged
        && !rackBindingChanged
        && !fxStateChanged
        && !noteContentChanged
        && !mixStateChanged
        && !instrumentActivationChanged
        && previousTrack.renderedAudioPath.equalsIgnoreCase(currentTrack.renderedAudioPath);
    const bool lightweightTrackUiRefresh = !parameterOnlyContentChanged
        && !rackBindingChanged
        && !fxStateChanged
        && !mixStateChanged
        && !instrumentActivationChanged
        && previousTrack.renderedAudioPath.equalsIgnoreCase(currentTrack.renderedAudioPath);

    if (rackBindingChanged)
        closeRackEditorSession(trackIndex);

    if (parameterOnlyContentChanged)
    {
        if (!isTrackBeingLiveEdited(trackIndex))
        {
            refreshProjectSummaryLabels();
            refreshFloatingWindows(false);
        }
    }
    else if (lightweightTrackUiRefresh)
    {
        refreshProjectSummaryLabels();
        trackTable.repaint();
        if (trackIndex == getSelectedTrackIndex())
            refreshInspector();
        updateEditorState();
    }
    else
    {
        refreshUi();
    }

    if ((projectPreviewRunning || rackPreviewRunning) && nativeVstHost.isReady())
    {
        auto result = juce::Result::ok();
        bool usedIncrementalUpdate = false;
        bool deferredParameterSync = false;

        if (requiresFullEngineState)
        {
            result = nativeVstHost.setAudioEngineState(documentState.project, true);
        }
        else
        {
            if (mixStateChanged)
            {
                result = nativeVstHost.setAudioEngineTrackMixState(documentState.project, trackIndex, currentTrack);
                usedIncrementalUpdate = true;
            }

            if (result.wasOk() && noteContentChanged)
            {
                result = nativeVstHost.setAudioEngineTrackNotes(trackIndex, documentState.project, currentTrack);
                usedIncrementalUpdate = true;
            }

            if (result.wasOk() && parameterContentChanged)
            {
                if (skipParameterEnginePush)
                {
                    usedIncrementalUpdate = true;
                }
                else if (deferParameterEnginePush)
                {
                    pendingLiveRackParameterEngineSyncTrack = trackIndex;
                    deferredParameterSync = true;
                }
                else
                {
                    result = nativeVstHost.setAudioEngineTrackParameters(trackIndex, currentTrack);
                    usedIncrementalUpdate = true;
                }
            }

            if (result.failed() || (!usedIncrementalUpdate && !deferredParameterSync))
                result = result.wasOk() && !usedIncrementalUpdate
                    ? juce::Result::ok()
                    : nativeVstHost.setAudioEngineState(documentState.project, true);
        }

        if (result.failed())
        {
            statusLabel.setText("Native preview update failed: " + result.getErrorMessage(), juce::dontSendNotification);
        }
        else if (requiresFullEngineState || usedIncrementalUpdate)
        {
            if (requiresFullEngineState || (usedIncrementalUpdate && parameterContentChanged))
                pendingLiveRackParameterEngineSyncTrack = -1;
            audioEngineStateValid = true;
            audioEngineStateDirty = false;
        }
    }
}

void StudioShellComponent::setProjectStateInternal(const ProjectState& updatedProject)
{
    const auto previousProject = documentState.project;
    const bool trackCountChanged = updatedProject.tracks.size() != previousProject.tracks.size();

    if (trackCountChanged)
    {
        resetRackHostTracking();
        closeAllRackEditorSessions();
    }

    documentState.project = updatedProject;
    normaliseProject(documentState.project);
    audioEngineStateDirty = true;

    const bool projectHeaderStateChanged = previousProject.bpm != documentState.project.bpm
        || previousProject.timeSigNumerator != documentState.project.timeSigNumerator
        || previousProject.timeSigDenominator != documentState.project.timeSigDenominator
        || previousProject.defaultPatternTicks != documentState.project.defaultPatternTicks
        || previousProject.keyQuantizeRoot != documentState.project.keyQuantizeRoot
        || !previousProject.keyQuantizeScale.equalsIgnoreCase(documentState.project.keyQuantizeScale)
        || previousProject.arrangementSnapTicks != documentState.project.arrangementSnapTicks
        || previousProject.loopEnabled != documentState.project.loopEnabled
        || previousProject.metronomeEnabled != documentState.project.metronomeEnabled;
    const bool projectAssetCatalogChanged = previousProject.sampleAssets.size() != documentState.project.sampleAssets.size()
        || previousProject.vstRack.size() != documentState.project.vstRack.size()
        || !sameStringArray(previousProject.vstiPaths, documentState.project.vstiPaths)
        || !sameStringArray(previousProject.vstiFolderPaths, documentState.project.vstiFolderPaths)
        || !sameStringArray(previousProject.samplePaths, documentState.project.samplePaths);
    const bool projectSharedFxStateChanged = !sameSharedEffectBuses(previousProject.sharedFxBuses,
                                                                    documentState.project.sharedFxBuses);
    const bool projectTempoMapChanged = !sameTempoMarkers(previousProject.tempoMarkers, documentState.project.tempoMarkers);
    const bool projectTransportPositionChanged = previousProject.playheadTick != documentState.project.playheadTick
        || previousProject.leftLocatorTick != documentState.project.leftLocatorTick
        || previousProject.rightLocatorTick != documentState.project.rightLocatorTick;
    const bool projectMasterStateChanged = std::abs(previousProject.masterVolume - documentState.project.masterVolume) > 1.0e-6
        || previousProject.masterFxBypassed != documentState.project.masterFxBypassed
        || !sameStringArray(previousProject.masterFxChain, documentState.project.masterFxChain)
        || !sameBoolVector(previousProject.masterFxSlotBypassed, documentState.project.masterFxSlotBypassed);
    const bool projectClipStateChanged = !sameSampleClips(previousProject.sampleClips, documentState.project.sampleClips);
    const bool projectMidiPatternStateChanged = !sameMidiPatterns(previousProject.midiPatterns, documentState.project.midiPatterns);
    const bool projectMidiSectionStateChanged = !sameMidiSections(previousProject.midiSections, documentState.project.midiSections);
    const bool requiresFullUiRefresh = trackCountChanged
        || projectHeaderStateChanged
        || projectAssetCatalogChanged
        || projectSharedFxStateChanged;
    std::vector<TrackEngineDiff> trackDiffs;
    trackDiffs.reserve(documentState.project.tracks.size());
    bool sawMidiTrackContentChange = false;
    bool nonMidiTrackStateChanged = false;

    if (!trackCountChanged)
    {
        for (int trackIndex = 0; trackIndex < static_cast<int>(documentState.project.tracks.size()); ++trackIndex)
        {
            auto diff = analyseTrackEngineDiff(previousProject, documentState.project, trackIndex);
            sawMidiTrackContentChange = sawMidiTrackContentChange || diff.noteContentChanged || diff.controllerContentChanged;
            nonMidiTrackStateChanged = nonMidiTrackStateChanged
                || diff.rackBindingChanged
                || diff.parameterContentChanged
                || diff.mixStateChanged
                || diff.fxStateChanged
                || diff.automationContentChanged
                || diff.renderedAudioChanged
                || diff.instrumentActivationChanged
                || diff.requiresFullEngineState;
            trackDiffs.push_back(std::move(diff));
        }
    }

    const bool canUseMidiEditUiRefresh = !requiresFullUiRefresh
        && !projectTempoMapChanged
        && !projectTransportPositionChanged
        && !projectMasterStateChanged
        && !projectClipStateChanged
        && !nonMidiTrackStateChanged
        && (projectMidiPatternStateChanged || projectMidiSectionStateChanged || sawMidiTrackContentChange);

    if (requiresFullUiRefresh)
    {
        refreshUi();
    }
    else if (canUseMidiEditUiRefresh)
    {
        refreshProjectSummaryLabels();
        trackTable.repaint();
        ensureSelectedMidiSectionForTrack(getSelectedTrackIndex());
        refreshInspector();
        refreshMidiEditState();
    }
    else
    {
        refreshProjectSummaryLabels();
        trackTable.repaint();
        ensureSelectedMidiSectionForTrack(getSelectedTrackIndex());
        refreshInspector();
        updateEditorState();
    }

    if ((projectPreviewRunning || rackPreviewRunning) && nativeVstHost.isReady())
    {
        const bool projectTransportSettingsChanged = previousProject.bpm != documentState.project.bpm
            || previousProject.leftLocatorTick != documentState.project.leftLocatorTick
            || previousProject.rightLocatorTick != documentState.project.rightLocatorTick
            || previousProject.loopEnabled != documentState.project.loopEnabled
            || previousProject.metronomeEnabled != documentState.project.metronomeEnabled;
        const bool projectPlayheadChanged = previousProject.playheadTick != documentState.project.playheadTick;
        const bool projectTimelineStateChanged = previousProject.timeSigNumerator != documentState.project.timeSigNumerator
            || previousProject.timeSigDenominator != documentState.project.timeSigDenominator
            || projectTempoMapChanged;

        bool requiresFullEngineState = trackCountChanged
            || projectTimelineStateChanged
            || projectMasterStateChanged
            || projectClipStateChanged
            || projectSharedFxStateChanged;

        if (!requiresFullEngineState)
        {
            for (const auto& diff : trackDiffs)
            {
                if (diff.requiresFullEngineState)
                    requiresFullEngineState = true;
            }
        }

        auto result = juce::Result::ok();
        bool usedIncrementalUpdate = false;
        if (requiresFullEngineState)
        {
            result = nativeVstHost.setAudioEngineState(documentState.project, true);
        }
        else
        {
            for (int trackIndex = 0; trackIndex < static_cast<int>(trackDiffs.size()) && result.wasOk(); ++trackIndex)
            {
                const auto& diff = trackDiffs[static_cast<size_t>(trackIndex)];
                const auto& track = documentState.project.tracks[static_cast<size_t>(trackIndex)];

                if (diff.mixStateChanged)
                {
                    result = nativeVstHost.setAudioEngineTrackMixState(documentState.project, trackIndex, track);
                    usedIncrementalUpdate = true;
                }

                if (result.wasOk() && (diff.noteContentChanged || diff.controllerContentChanged))
                {
                    result = nativeVstHost.setAudioEngineTrackNotes(trackIndex, documentState.project, track);
                    usedIncrementalUpdate = true;
                }

                if (result.wasOk() && diff.parameterContentChanged)
                {
                    result = nativeVstHost.setAudioEngineTrackParameters(trackIndex, track);
                    usedIncrementalUpdate = true;
                }
            }
        }

        if (result.wasOk() && projectTransportSettingsChanged)
        {
            result = nativeVstHost.updateAudioEngineTransport(documentState.project);
            usedIncrementalUpdate = true;
        }

        if (result.wasOk() && projectPlayheadChanged)
        {
            result = nativeVstHost.seekAudioEngine(documentState.project.playheadTick);
            usedIncrementalUpdate = true;
        }

        if (result.failed())
            statusLabel.setText("Native preview update failed: " + result.getErrorMessage(), juce::dontSendNotification);
        else
        {
            pendingLiveRackParameterEngineSyncTrack = -1;
            audioEngineStateValid = true;
            audioEngineStateDirty = false;
        }
    }
}

juce::File StudioShellComponent::suggestProjectFile() const
{
    if (currentProjectFile != juce::File())
        return currentProjectFile;
    return juce::File::getSpecialLocation(juce::File::userDocumentsDirectory)
        .getChildFile("ai-music-studio-native.aims");
}

juce::File StudioShellComponent::suggestTrackStateFile(int trackIndex, const TrackState& track) const
{
    auto stateRoot = currentProjectFile.existsAsFile()
        ? currentProjectFile.getParentDirectory().getChildFile(currentProjectFile.getFileNameWithoutExtension() + "_native_states")
        : juce::File::getSpecialLocation(juce::File::userDocumentsDirectory).getChildFile("Mutagen Native States");

    auto safeTrackName = juce::File::createLegalFileName(track.name.trim());
    if (safeTrackName.isEmpty())
        safeTrackName = "track";

    return stateRoot.getChildFile(juce::String(trackIndex + 1) + "_" + safeTrackName).withFileExtension(".vststate");
}

juce::File StudioShellComponent::ensureProjectSuffix(const juce::File& file) const
{
    if (file.hasFileExtension(".aims") || file.hasFileExtension(".json"))
        return file;
    return file.withFileExtension(".aims");
}

int StudioShellComponent::findPreferredSampleTrackIndex() const
{
    const auto selected = getSelectedTrackIndex();
    if (juce::isPositiveAndBelow(selected, static_cast<int>(documentState.project.tracks.size()))
        && documentState.project.tracks[static_cast<size_t>(selected)].trackType.equalsIgnoreCase("sample"))
    {
        return selected;
    }

    for (int trackIndex = 0; trackIndex < static_cast<int>(documentState.project.tracks.size()); ++trackIndex)
    {
        if (documentState.project.tracks[static_cast<size_t>(trackIndex)].trackType.equalsIgnoreCase("sample"))
            return trackIndex;
    }

    return -1;
}

int StudioShellComponent::getSelectedTrackIndex() const
{
    const auto row = trackTable.getSelectedRow();
    return juce::isPositiveAndBelow(row, static_cast<int>(documentState.project.tracks.size()))
        ? row
        : -1;
}

void StudioShellComponent::setSelectedTrackIndex(int row)
{
    if (row < 0 || row >= static_cast<int>(documentState.project.tracks.size()))
    {
        if (trackTable.getSelectedRow() >= 0)
            trackTable.deselectAllRows();
        return;
    }

    if (trackTable.getSelectedRow() == row)
        return;

    trackTable.selectRow(row);
}

int StudioShellComponent::getSelectedMidiSectionIndex() const noexcept
{
    return selectedMidiSectionIndex;
}

void StudioShellComponent::setSelectedMidiSectionIndex(int sectionIndex, bool bringTrackIntoFocus)
{
    const auto validSection = juce::isPositiveAndBelow(sectionIndex, static_cast<int>(documentState.project.midiSections.size()))
        ? sectionIndex
        : -1;
    const auto previousSection = selectedMidiSectionIndex;
    selectedMidiSectionIndex = validSection;

    if (selectedMidiSectionIndex >= 0 && bringTrackIntoFocus)
    {
        const auto trackIndex = documentState.project.midiSections[static_cast<size_t>(selectedMidiSectionIndex)].trackIndex;
        if (trackIndex != getSelectedTrackIndex())
            setSelectedTrackIndex(trackIndex);
    }

    if (previousSection != selectedMidiSectionIndex)
        updateEditorState();
}

void StudioShellComponent::ensureSelectedMidiSectionForTrack(int trackIndex)
{
    if (!juce::isPositiveAndBelow(trackIndex, static_cast<int>(documentState.project.tracks.size())))
    {
        selectedMidiSectionIndex = -1;
        return;
    }

    if (juce::isPositiveAndBelow(selectedMidiSectionIndex, static_cast<int>(documentState.project.midiSections.size())))
    {
        const auto& section = documentState.project.midiSections[static_cast<size_t>(selectedMidiSectionIndex)];
        if (section.trackIndex == trackIndex)
            return;
    }

    selectedMidiSectionIndex = -1;
    for (int sectionIndex = 0; sectionIndex < static_cast<int>(documentState.project.midiSections.size()); ++sectionIndex)
    {
        if (documentState.project.midiSections[static_cast<size_t>(sectionIndex)].trackIndex == trackIndex)
        {
            selectedMidiSectionIndex = sectionIndex;
            break;
        }
    }
}

void StudioShellComponent::focusMidiSectionInPianoRoll(int sectionIndex)
{
    setSelectedMidiSectionIndex(sectionIndex, true);
    focusPianoRollPanel();
}

int StudioShellComponent::getSelectedSampleAssetIndex() const
{
    return sampleAssetList.getSelectedRow();
}

void StudioShellComponent::setSelectedSampleAssetIndex(int row)
{
    if (row < 0 || row >= static_cast<int>(documentState.project.sampleAssets.size()))
    {
        if (sampleAssetList.getSelectedRow() >= 0)
            sampleAssetList.deselectAllRows();
        return;
    }

    if (sampleAssetList.getSelectedRow() == row)
        return;

    sampleAssetList.selectRow(row);
}

TrackState* StudioShellComponent::getSelectedTrack()
{
    const auto selected = getSelectedTrackIndex();
    if (selected < 0 || selected >= static_cast<int>(documentState.project.tracks.size()))
        return nullptr;
    return &documentState.project.tracks[static_cast<size_t>(selected)];
}

const TrackState* StudioShellComponent::getSelectedTrack() const
{
    const auto selected = getSelectedTrackIndex();
    if (selected < 0 || selected >= static_cast<int>(documentState.project.tracks.size()))
        return nullptr;
    return &documentState.project.tracks[static_cast<size_t>(selected)];
}

MainWindow::MainWindow(const juce::File& startupProject)
    : juce::DocumentWindow("Mutagen",
                           juce::Colour::fromRGB(11, 13, 17),
                           juce::DocumentWindow::allButtons)
{
    setUsingNativeTitleBar(true);
    setResizable(true, true);
    setResizeLimits(1080, 720, 2400, 1800);
    setIcon(loadMutagenLogoBinaryData(true));
    setMenuBar(this);

    shell = new StudioShellComponent();
    setContentOwned(shell, true);
    centreWithSize(kDefaultWindowWidth, kDefaultWindowHeight);
    setVisible(true);

    if (startupProject.existsAsFile())
        shell->openProjectFile(startupProject);
}

MainWindow::~MainWindow()
{
    setMenuBar(nullptr);
}

juce::StringArray MainWindow::getMenuBarNames()
{
    return { "File", "Edit", "Settings", "Windows", "Help" };
}

juce::PopupMenu MainWindow::getMenuForIndex(int topLevelMenuIndex, const juce::String& menuName)
{
    juce::ignoreUnused(menuName);

    juce::PopupMenu menu;
    if (shell == nullptr)
        return menu;

    switch (topLevelMenuIndex)
    {
        case 0:
            menu.addItem(menuFileNew, "New Project");
            menu.addItem(menuFileOpen, "Open Project...");
            menu.addItem(menuFileSave, "Save Project");
            menu.addItem(menuFileSaveAs, "Save Project As...");
            menu.addSeparator();
            menu.addItem(menuFileAddTrack, "Add Instrument Track");
            menu.addItem(menuFileAddSampleTrack, "Add Sample Track");
            menu.addSeparator();
            menu.addItem(menuFileImportJson, "Import JSON Project...");
            menu.addItem(menuFileImportMidi, "Import MIDI...");
            menu.addItem(menuFileImportSample, "Import Sample...");
            menu.addSeparator();
            menu.addItem(menuFileExportJson, "Export Project as JSON...");
            menu.addItem(menuFileExportMp3, "Export Sequence as MP3...");
            menu.addItem(menuFileExportWav, "Export Sequence as WAV...");
            menu.addItem(menuFileExportTrackMp3, "Export Selected Track as MP3...");
            menu.addItem(menuFileExportTrackWav, "Export Selected Track as WAV...");
            menu.addItem(menuFileExportStems, "Export Stems...");
            menu.addItem(menuFileExportMidi, "Export MIDI...");
            break;

        case 1:
            menu.addItem(menuEditUndo, "Undo", shell->undoManager.canUndo(), false);
            menu.addItem(menuEditRedo, "Redo", shell->undoManager.canRedo(), false);
            menu.addSeparator();
            menu.addItem(menuEditQuantize, "Quantize Selected Notes");
            menu.addSeparator();
            menu.addItem(menuEditSelectAll, "Select All");
            menu.addItem(menuEditCopy, "Copy Notes");
            menu.addItem(menuEditCut, "Cut Notes");
            menu.addItem(menuEditDelete, "Delete Notes");
            menu.addItem(menuEditDuplicate, "Duplicate Notes");
            menu.addItem(menuEditPaste, "Paste Notes");
            break;

        case 2:
            menu.addItem(menuSettingsAudio, "Audio Settings...");
            menu.addItem(menuSettingsVstFolders, "VST Folder Manager...");
            menu.addItem(menuSettingsTemplates, "Template Settings...");
            menu.addItem(menuSettingsAi, "AI Settings...");
            {
                juce::PopupMenu themesMenu;
                for (int themeIndex = 0; themeIndex < shell->getThemeCount(); ++themeIndex)
                {
                    themesMenu.addItem(menuThemeBase + themeIndex,
                                       shell->getThemeName(themeIndex),
                                       true,
                                       themeIndex == shell->getCurrentThemeIndex());
                }
                menu.addSeparator();
                menu.addSubMenu("Themes", themesMenu);
            }
            {
                juce::PopupMenu fontsMenu;
                for (int fontIndex = 0; fontIndex < shell->getFontCount(); ++fontIndex)
                {
                    fontsMenu.addItem(menuFontChoiceBase + fontIndex,
                                      shell->getFontName(fontIndex),
                                      true,
                                      fontIndex == shell->getCurrentFontIndex());
                }
                juce::PopupMenu fontSizesMenu;
                for (int sizeIndex = 0; sizeIndex < shell->getFontSizeCount(); ++sizeIndex)
                {
                    fontSizesMenu.addItem(menuFontSizeBase + sizeIndex,
                                          shell->getFontSizeLabel(sizeIndex),
                                          true,
                                          sizeIndex == shell->getCurrentFontSizeIndex());
                }
                fontsMenu.addSeparator();
                fontsMenu.addSubMenu("Size", fontSizesMenu);
                menu.addSubMenu("Fonts", fontsMenu);
            }
            break;

        case 3:
            menu.addItem(menuWindowsPanels, "Show Panels Window", true, shell->isPanelsWindowVisible());
            menu.addItem(menuWindowsTransport, "Show Transport Window", true, shell->isTransportWindowVisible());
            menu.addItem(menuWindowsMixer, "Show Mixer Window", true, shell->isMixerWindowVisible());
            menu.addItem(menuWindowsAudio, "Show Audio Window", true, shell->isAudioWindowVisible());
            menu.addItem(menuWindowsTracks, "Show Tracks Window", true, shell->isTracksWindowVisible());
            menu.addItem(menuWindowsModulationMatrix, "Show Modulation Matrix", true, shell->isModulationMatrixWindowVisible());
            menu.addItem(menuWindowsRackBrowser, "Show Rack Browser Window", true, shell->isRackBrowserWindowVisible());
            menu.addItem(menuWindowsRenderManager, "Show Render Manager Window", true, shell->isRenderManagerWindowVisible());
            menu.addSeparator();
            menu.addItem(menuWindowsArrangement, "Show Arrangement Window", true, shell->isArrangementWindowVisible());
            menu.addItem(menuWindowsAutomation, "Show Automation Window", true, shell->isAutomationWindowVisible());
            menu.addItem(menuWindowsSamples, "Show Samples Window", true, shell->isSamplesWindowVisible());
            menu.addItem(menuWindowsPianoRoll, "Show Piano Roll Window", true, shell->isPianoRollWindowVisible());
            menu.addItem(menuWindowsVirtualPiano, "Show Virtual Piano Window", true, shell->isVirtualPianoWindowVisible());
            menu.addItem(menuWindowsActivityLog, "Show Activity Log Window", true, shell->isActivityLogWindowVisible());
            menu.addSeparator();
            menu.addItem(menuWindowsFullscreen, "Borderless Fullscreen\tF11", true, isBorderlessFullscreenActive());
            break;

        case 4:
            menu.addItem(menuHelpSite, "Open Mutagen Site");
            menu.addItem(menuHelpOllama, "Open Ollama Site");
            menu.addSeparator();
            menu.addItem(menuHelpAbout, "About Mutagen");
            break;

        default:
            break;
    }

    return menu;
}

void MainWindow::menuItemSelected(int menuItemID, int topLevelMenuIndex)
{
    juce::ignoreUnused(topLevelMenuIndex);
    if (shell == nullptr)
        return;

    if (menuItemID >= menuThemeBase && menuItemID < menuThemeBase + shell->getThemeCount())
    {
        shell->setThemeIndex(menuItemID - menuThemeBase);
        return;
    }

    if (menuItemID >= menuFontChoiceBase && menuItemID < menuFontChoiceBase + shell->getFontCount())
    {
        shell->setFontIndex(menuItemID - menuFontChoiceBase);
        return;
    }

    if (menuItemID >= menuFontSizeBase && menuItemID < menuFontSizeBase + shell->getFontSizeCount())
    {
        shell->setFontSizeIndex(menuItemID - menuFontSizeBase);
        return;
    }

    switch (menuItemID)
    {
        case menuFileNew: shell->createNewProject(); break;
        case menuFileOpen: shell->promptOpenProject(); break;
        case menuFileSave: shell->saveProject(); break;
        case menuFileSaveAs: shell->saveProjectAs(); break;
        case menuFileAddTrack: shell->addTrack(); break;
        case menuFileAddSampleTrack: shell->addSampleTrack(); break;
        case menuFileImportJson: shell->promptImportJson(); break;
        case menuFileImportMidi: shell->promptImportMidi(); break;
        case menuFileImportSample: shell->promptImportSample(); break;
        case menuFileExportJson: shell->promptExportJson(); break;
        case menuFileExportMp3: shell->promptExportMp3(); break;
        case menuFileExportWav: shell->promptExportWav(); break;
        case menuFileExportTrackMp3: shell->promptExportSelectedTrackMp3(); break;
        case menuFileExportTrackWav: shell->promptExportSelectedTrackWav(); break;
        case menuFileExportStems: shell->promptExportProjectStems(); break;
        case menuFileExportMidi: shell->promptExportMidi(); break;
        case menuEditUndo: shell->undo(); break;
        case menuEditRedo: shell->redo(); break;
        case menuEditQuantize: shell->quantizeSelectedNotes(); break;
        case menuEditSelectAll: shell->selectAllNotesFromMenu(); break;
        case menuEditCopy: shell->copyNotesFromMenu(); break;
        case menuEditCut: shell->cutNotesFromMenu(); break;
        case menuEditDelete: shell->deleteNotesFromMenu(); break;
        case menuEditDuplicate: shell->duplicateNotesFromMenu(); break;
        case menuEditPaste: shell->pasteNotesFromMenu(); break;
        case menuSettingsAudio: shell->showAudioSettingsWindow(); break;
        case menuSettingsVstFolders: shell->showVstFolderManagerWindow(); break;
        case menuSettingsTemplates: shell->showTemplateSettingsDialog(); break;
        case menuSettingsAi: shell->showAiSettingsDialog(); break;
        case menuWindowsPanels: shell->setPanelsWindowVisible(!shell->isPanelsWindowVisible()); break;
        case menuWindowsTransport: shell->setTransportWindowVisible(!shell->isTransportWindowVisible()); break;
        case menuWindowsMixer: shell->setMixerWindowVisible(!shell->isMixerWindowVisible()); break;
        case menuWindowsAudio:
            if (shell->isAudioWindowVisible())
                shell->setAudioWindowVisible(false);
            else
                shell->focusAudioPanel();
            break;
        case menuWindowsTracks:
            if (shell->isTracksWindowVisible())
                shell->setTracksWindowVisible(false);
            else
                shell->focusTracksPanel();
            break;
        case menuWindowsModulationMatrix:
            if (shell->isModulationMatrixWindowVisible())
                shell->setModulationMatrixWindowVisible(false);
            else
                shell->focusModulationMatrixPanel();
            break;
        case menuWindowsRackBrowser:
            if (shell->isRackBrowserWindowVisible())
                shell->setRackBrowserWindowVisible(false);
            else
                shell->focusRackBrowserPanel();
            break;
        case menuWindowsRenderManager:
            if (shell->isRenderManagerWindowVisible())
                shell->setRenderManagerWindowVisible(false);
            else
                shell->focusRenderManagerPanel();
            break;
        case menuWindowsArrangement:
            if (shell->isArrangementWindowVisible())
                shell->setArrangementWindowVisible(false);
            else
                shell->focusArrangementPanel();
            break;
        case menuWindowsAutomation:
            if (shell->isAutomationWindowVisible())
                shell->setAutomationWindowVisible(false);
            else
                shell->focusAutomationPanel();
            break;
        case menuWindowsSamples:
            if (shell->isSamplesWindowVisible())
                shell->setSamplesWindowVisible(false);
            else
                shell->focusSamplesPanel();
            break;
        case menuWindowsPianoRoll:
            if (shell->isPianoRollWindowVisible())
                shell->setPianoRollWindowVisible(false);
            else
                shell->focusPianoRollPanel();
            break;
        case menuWindowsVirtualPiano:
            if (shell->isVirtualPianoWindowVisible())
                shell->setVirtualPianoWindowVisible(false);
            else
                shell->focusVirtualPianoPanel();
            break;
        case menuWindowsActivityLog:
            shell->setActivityLogWindowVisible(!shell->isActivityLogWindowVisible());
            break;
        case menuWindowsFullscreen:
            toggleBorderlessFullscreen();
            break;
        case menuHelpSite:
            juce::URL("https://mysticalg.github.io/AI-Music-Studio/").launchInDefaultBrowser();
            break;
        case menuHelpOllama:
            juce::URL(kOllamaSiteUrl).launchInDefaultBrowser();
            break;
        case menuHelpAbout:
            shell->showAboutDialog();
            break;
        default: break;
    }
}

void MainWindow::closeButtonPressed()
{
    juce::JUCEApplication::getInstance()->systemRequestedQuit();
}

void MainWindow::toggleBorderlessFullscreen()
{
    auto& desktop = juce::Desktop::getInstance();

    if (isBorderlessFullscreenActive())
    {
        desktop.setKioskModeComponent(nullptr);
        toFront(true);
        return;
    }

    if (auto* currentKiosk = desktop.getKioskModeComponent(); currentKiosk != nullptr && currentKiosk != this)
        desktop.setKioskModeComponent(nullptr);

    if (isFullScreen())
        setFullScreen(false);

    desktop.setKioskModeComponent(this, false);
    toFront(true);
}

bool MainWindow::isBorderlessFullscreenActive() const
{
    return isKioskMode() || juce::Desktop::getInstance().getKioskModeComponent() == this;
}

bool MainWindow::keyPressed(const juce::KeyPress& key)
{
    if (key.getKeyCode() == juce::KeyPress::F11Key && !key.getModifiers().isAnyModifierKeyDown())
    {
        toggleBorderlessFullscreen();
        return true;
    }

    return juce::DocumentWindow::keyPressed(key);
}

} // namespace aims
