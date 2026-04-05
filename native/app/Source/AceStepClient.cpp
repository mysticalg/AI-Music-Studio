#include "AceStepClient.h"

#include <chrono>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <thread>

namespace aims
{
namespace
{
constexpr const char* kAppName = "Mutagen";
constexpr const char* kDefaultBaseUrl = "http://127.0.0.1:8001";

juce::var makeObjectVar()
{
    return juce::var(new juce::DynamicObject());
}

void setObjectProperty(juce::var& objectVar, const juce::Identifier& name, const juce::var& value)
{
    if (auto* object = objectVar.getDynamicObject())
        object->setProperty(name, value);
}

juce::String getStringProperty(const juce::var& objectVar,
                               const juce::Identifier& name,
                               const juce::String& fallback = {})
{
    if (auto* object = objectVar.getDynamicObject())
    {
        const auto value = object->getProperty(name);
        return value.isVoid() ? fallback : value.toString();
    }

    return fallback;
}

int getIntProperty(const juce::var& objectVar,
                   const juce::Identifier& name,
                   int fallback,
                   int minimum,
                   int maximum)
{
    if (auto* object = objectVar.getDynamicObject())
    {
        const auto value = object->getProperty(name);
        if (!value.isVoid())
            return juce::jlimit(minimum, maximum, static_cast<int>(value));
    }

    return fallback;
}

bool getBoolProperty(const juce::var& objectVar,
                     const juce::Identifier& name,
                     bool fallback)
{
    if (auto* object = objectVar.getDynamicObject())
    {
        const auto value = object->getProperty(name);
        if (!value.isVoid())
            return static_cast<bool>(value);
    }

    return fallback;
}

double getDoubleProperty(const juce::var& objectVar,
                         const juce::Identifier& name,
                         double fallback)
{
    if (auto* object = objectVar.getDynamicObject())
    {
        const auto value = object->getProperty(name);
        if (!value.isVoid())
            return static_cast<double>(value);
    }

    return fallback;
}

juce::String truncateActivityLogText(juce::String text, int maxCharacters = 24000)
{
    if (text.length() <= maxCharacters)
        return text;

    return text.substring(0, maxCharacters)
        + "\n\n...[truncated "
        + juce::String(text.length() - maxCharacters)
        + " chars]";
}

juce::String prettyJsonForActivityLog(const juce::var& value)
{
    return truncateActivityLogText(juce::JSON::toString(value, false));
}

juce::String prettyRawJsonForActivityLog(const juce::String& raw)
{
    juce::var parsed;
    if (juce::JSON::parse(raw, parsed).wasOk())
        return prettyJsonForActivityLog(parsed);

    return truncateActivityLogText(raw);
}

juce::String prepareDebugLogText(juce::String text, int maxCharacters = 200000)
{
    if (text.length() <= maxCharacters)
        return text;

    return text.substring(0, maxCharacters)
        + "\n\n...[truncated "
        + juce::String(text.length() - maxCharacters)
        + " chars for debug log safety]";
}

juce::var loadJsonObject(const juce::File& file)
{
    if (!file.existsAsFile())
        return makeObjectVar();

    juce::var parsed;
    if (juce::JSON::parse(file.loadFileAsString(), parsed).failed() || !parsed.isObject())
        return makeObjectVar();

    return parsed;
}

void saveJsonObject(const juce::File& file, const juce::var& objectVar)
{
    if (auto parent = file.getParentDirectory(); parent != juce::File())
        parent.createDirectory();

    file.replaceWithText(juce::JSON::toString(objectVar, true), false, false, "\n");
}

juce::String resolvedAudioUrl(const juce::String& baseUrl, const juce::String& urlOrPath)
{
    const auto trimmed = urlOrPath.trim();
    if (trimmed.startsWithIgnoreCase("http://") || trimmed.startsWithIgnoreCase("https://"))
        return trimmed;

    if (trimmed.startsWithChar('/'))
        return baseUrl + trimmed;

    return baseUrl + "/" + trimmed;
}

juce::String inferReturnedModelName(const juce::var& resultEntry)
{
    auto modelName = getStringProperty(resultEntry, "dit_model").trim();
    if (modelName.isEmpty())
        modelName = getStringProperty(resultEntry, "model").trim();
    return modelName;
}

struct EndpointAddress
{
    juce::String host;
    int port = 0;
};

EndpointAddress parseEndpointAddress(juce::String baseUrl)
{
    auto text = baseUrl.trim();
    const auto schemeIndex = text.indexOf("://");
    if (schemeIndex >= 0)
        text = text.substring(schemeIndex + 3);

    const auto slashIndex = text.indexOfChar('/');
    if (slashIndex >= 0)
        text = text.substring(0, slashIndex);

    EndpointAddress endpoint;

    if (text.startsWithChar('['))
    {
        const auto closingBracket = text.indexOfChar(']');
        if (closingBracket > 1)
        {
            endpoint.host = text.substring(1, closingBracket);
            if (closingBracket + 1 < text.length() && text[closingBracket + 1] == ':')
                endpoint.port = text.substring(closingBracket + 2).getIntValue();
            return endpoint;
        }
    }

    const auto colonIndex = text.lastIndexOfChar(':');
    if (colonIndex > 0 && text.indexOfChar(':') == colonIndex)
    {
        endpoint.host = text.substring(0, colonIndex);
        endpoint.port = text.substring(colonIndex + 1).getIntValue();
    }
    else
    {
        endpoint.host = text;
    }

    return endpoint;
}
} // namespace

AceStepClient::AceStepClient()
    : baseUrl(kDefaultBaseUrl),
      defaultAudioFormat("wav")
{
    loadSavedSettings();
    loadSavedAuth();
}

juce::String AceStepClient::getBaseUrl() const
{
    return baseUrl;
}

juce::String AceStepClient::getInstallDirectory() const
{
    return installDirectory;
}

juce::String AceStepClient::getDefaultModel() const
{
    return defaultModel;
}

juce::String AceStepClient::getDefaultAudioFormat() const
{
    return defaultAudioFormat;
}

juce::String AceStepClient::getDefaultQualityPreset() const
{
    return defaultQualityPreset;
}

juce::String AceStepClient::getDefaultVocalLanguage() const
{
    return defaultVocalLanguage;
}

bool AceStepClient::getAutoStartServer() const noexcept
{
    return autoStartServer;
}

bool AceStepClient::getDefaultThinking() const noexcept
{
    return defaultThinking;
}

bool AceStepClient::getDefaultUseRandomSeed() const noexcept
{
    return defaultUseRandomSeed;
}

int AceStepClient::getDefaultSeed() const noexcept
{
    return defaultSeed;
}

int AceStepClient::getDefaultInferenceSteps() const noexcept
{
    return defaultInferenceSteps;
}

double AceStepClient::getDefaultGuidanceScale() const noexcept
{
    return defaultGuidanceScale;
}

juce::String AceStepClient::getDefaultInferMethod() const
{
    return defaultInferMethod;
}

int AceStepClient::getStartupTimeoutSeconds() const noexcept
{
    return startupTimeoutSeconds;
}

int AceStepClient::getJobTimeoutSeconds() const noexcept
{
    return jobTimeoutSeconds;
}

void AceStepClient::setBaseUrl(const juce::String& newBaseUrl)
{
    baseUrl = normaliseBaseUrl(newBaseUrl);
}

void AceStepClient::setApiKey(const juce::String& newApiKey)
{
    apiKey = newApiKey.trim();
    saveAuth();
}

void AceStepClient::clearApiKey()
{
    apiKey.clear();
    saveAuth();
}

void AceStepClient::setInstallDirectory(const juce::String& newInstallDirectory)
{
    auto directory = juce::File(newInstallDirectory.trim());
    if (directory.existsAsFile())
        directory = directory.getParentDirectory();
    installDirectory = directory.getFullPathName().trim();
}

void AceStepClient::setDefaultModel(const juce::String& newDefaultModel)
{
    defaultModel = newDefaultModel.trim();
}

void AceStepClient::setDefaultAudioFormat(const juce::String& newAudioFormat)
{
    defaultAudioFormat = normaliseAudioFormat(newAudioFormat);
}

void AceStepClient::setDefaultQualityPreset(const juce::String& newQualityPreset)
{
    defaultQualityPreset = normaliseQualityPreset(newQualityPreset);
}

void AceStepClient::setDefaultVocalLanguage(const juce::String& newVocalLanguage)
{
    defaultVocalLanguage = normaliseVocalLanguage(newVocalLanguage);
}

void AceStepClient::setAutoStartServer(bool shouldAutoStart) noexcept
{
    autoStartServer = shouldAutoStart;
}

void AceStepClient::setDefaultThinking(bool shouldUseThinking) noexcept
{
    defaultThinking = shouldUseThinking;
}

void AceStepClient::setDefaultUseRandomSeed(bool shouldUseRandomSeed) noexcept
{
    defaultUseRandomSeed = shouldUseRandomSeed;
}

void AceStepClient::setDefaultSeed(int newSeed) noexcept
{
    defaultSeed = newSeed;
}

void AceStepClient::setDefaultInferenceSteps(int newInferenceSteps)
{
    defaultInferenceSteps = normaliseInferenceSteps(newInferenceSteps);
}

void AceStepClient::setDefaultGuidanceScale(double newGuidanceScale) noexcept
{
    defaultGuidanceScale = normaliseGuidanceScale(newGuidanceScale);
}

void AceStepClient::setDefaultInferMethod(const juce::String& newInferMethod)
{
    defaultInferMethod = normaliseInferMethod(newInferMethod);
}

void AceStepClient::setStartupTimeoutSeconds(int timeoutSeconds)
{
    startupTimeoutSeconds = normaliseTimeoutSeconds(timeoutSeconds);
}

void AceStepClient::setJobTimeoutSeconds(int timeoutSeconds)
{
    jobTimeoutSeconds = normaliseTimeoutSeconds(timeoutSeconds);
}

void AceStepClient::saveSettings() const
{
    auto preferences = makeObjectVar();
    setObjectProperty(preferences, "base_url", baseUrl);
    setObjectProperty(preferences, "install_directory", installDirectory);
    setObjectProperty(preferences, "default_model", defaultModel);
    setObjectProperty(preferences, "default_audio_format", defaultAudioFormat);
    setObjectProperty(preferences, "default_quality_preset", defaultQualityPreset);
    setObjectProperty(preferences, "default_vocal_language", defaultVocalLanguage);
    setObjectProperty(preferences, "auto_start_server", autoStartServer);
    setObjectProperty(preferences, "default_thinking", defaultThinking);
    setObjectProperty(preferences, "default_use_random_seed", defaultUseRandomSeed);
    setObjectProperty(preferences, "default_seed", defaultSeed);
    setObjectProperty(preferences, "default_inference_steps", defaultInferenceSteps);
    setObjectProperty(preferences, "default_guidance_scale", defaultGuidanceScale);
    setObjectProperty(preferences, "default_infer_method", defaultInferMethod);
    setObjectProperty(preferences, "startup_timeout_seconds", startupTimeoutSeconds);
    setObjectProperty(preferences, "job_timeout_seconds", jobTimeoutSeconds);
    saveJsonObject(preferencesFile(), preferences);
}

juce::StringArray AceStepClient::availableModels(int timeoutMs) const
{
    juce::StringPairArray headers;
    if (apiKey.trim().isNotEmpty())
        headers.set("Authorization", authorizationHeaderValue());

    const auto timeoutSeconds = juce::jlimit(1, 10, timeoutMs / 1000);
    const auto response = requestJson(endpointUrl("v1/models"),
                                      nullptr,
                                      headers,
                                      "GET",
                                      timeoutSeconds,
                                      "ACE-Step Models",
                                      false);

    juce::StringArray models;
    auto payload = response;
    if (auto* responseObject = response.getDynamicObject())
    {
        const auto data = responseObject->getProperty("data");
        if (data.isObject())
            payload = data;
    }

    if (auto* payloadObject = payload.getDynamicObject())
    {
        const auto entries = payloadObject->getProperty("models");
        if (auto* array = entries.getArray())
        {
            for (const auto& entry : *array)
            {
                const auto name = getStringProperty(entry, "name").trim();
                if (name.isNotEmpty() && !models.contains(name))
                    models.add(name);
            }
        }

        const auto configuredDefault = getStringProperty(payload, "default_model").trim();
        if (configuredDefault.isNotEmpty() && !models.contains(configuredDefault))
            models.insert(0, configuredDefault);
    }

    return models;
}

bool AceStepClient::isServerReachable(int timeoutMs) const
{
    const auto timeoutSeconds = juce::jmax(1, timeoutMs / 1000);

    int statusCode = 0;
    juce::StringPairArray responseHeaders;
    auto inputStream = juce::URL(endpointUrl("health")).createInputStream(juce::URL::InputStreamOptions(juce::URL::ParameterHandling::inAddress)
                                                                              .withConnectionTimeoutMs(timeoutMs)
                                                                              .withResponseHeaders(&responseHeaders)
                                                                              .withStatusCode(&statusCode)
                                                                              .withNumRedirectsToFollow(2)
                                                                              .withHttpRequestCmd("GET"));
    juce::ignoreUnused(timeoutSeconds);
    if (inputStream == nullptr || statusCode >= 400)
        return false;

    juce::var parsed;
    return juce::JSON::parse(inputStream->readEntireStreamAsString(), parsed).wasOk();
}

bool AceStepClient::isServerPortAcceptingConnections(int timeoutMs) const
{
    const auto endpoint = parseEndpointAddress(baseUrl);
    if (endpoint.host.isEmpty() || endpoint.port <= 0)
        return false;

    juce::StreamingSocket socket;
    return socket.connect(endpoint.host, endpoint.port, timeoutMs);
}

juce::Result AceStepClient::waitForServerReady() const
{
    const auto initialDeadline = std::chrono::steady_clock::now() + std::chrono::seconds(startupTimeoutSeconds);
    const auto extendedDeadline = std::chrono::steady_clock::now() + std::chrono::seconds(jobTimeoutSeconds);
    bool sawListeningPort = false;
    bool emittedWarmupLog = false;

    while (std::chrono::steady_clock::now() < extendedDeadline)
    {
        if (isServerReachable(1500))
            return juce::Result::ok();

        sawListeningPort = sawListeningPort || isServerPortAcceptingConnections(250);
        if (sawListeningPort && !emittedWarmupLog)
        {
            emitActivityLog("ACE-Step",
                            "ACE-Step is listening on "
                                + baseUrl
                                + " but is still initializing models. "
                                  "Mutagen will keep waiting while first-run downloads or model loading complete.");
            emittedWarmupLog = true;
        }

        if (std::chrono::steady_clock::now() >= initialDeadline && !sawListeningPort)
            break;

        std::this_thread::sleep_for(std::chrono::milliseconds(1200));
    }

    if (sawListeningPort)
    {
        return juce::Result::fail("ACE-Step is still initializing or downloading models at "
                                  + baseUrl
                                  + ". Leave the server running until initialization completes, then try again.");
    }

    return juce::Result::fail("ACE-Step did not become ready at " + baseUrl
                              + ". Start the API server manually or configure the install folder for auto-start.");
}

bool AceStepClient::hasLaunchConfiguration() const
{
    return launchScriptFile().existsAsFile();
}

juce::File AceStepClient::launchScriptFile() const
{
    const auto installRoot = juce::File(installDirectory.trim());
    if (!installRoot.isDirectory())
        return {};

   #if JUCE_WINDOWS
    const auto mutagenLauncher = installRoot.getChildFile("mutagen-start-api-server.bat");
    if (mutagenLauncher.existsAsFile())
        return mutagenLauncher;
    return installRoot.getChildFile("start_api_server.bat");
   #elif JUCE_MAC
    return installRoot.getChildFile("start_api_server_macos.sh");
   #else
    return installRoot.getChildFile("start_api_server.sh");
   #endif
}

juce::String AceStepClient::statusSummary() const
{
    auto summary = "ACE-Step at " + baseUrl;
    if (autoStartServer && hasLaunchConfiguration())
        summary << " (auto-start enabled)";
    return summary;
}

AceStepGenerationResult AceStepClient::generateToFile(const AceStepGenerationRequest& request,
                                                      const juce::File& destinationFile) const
{
    if (request.prompt.trim().isEmpty() && request.lyrics.trim().isEmpty())
        throw std::runtime_error("Enter a prompt or lyrics for ACE-Step generation.");

    const auto readyResult = waitForServerReady();
    if (readyResult.failed())
        throw std::runtime_error(readyResult.getErrorMessage().toStdString());

    auto payload = makeObjectVar();
    setObjectProperty(payload, "prompt", request.prompt.trim());
    setObjectProperty(payload, "lyrics", request.lyrics.trim());
    setObjectProperty(payload, "task_type", "text2music");
    setObjectProperty(payload, "thinking", request.thinking);
    setObjectProperty(payload, "audio_format", normaliseAudioFormat(request.audioFormat));
    setObjectProperty(payload, "batch_size", 1);
    setObjectProperty(payload, "duration", juce::jlimit(10.0, 600.0, request.durationSeconds));
    setObjectProperty(payload, "bpm", juce::jlimit(30, 300, request.bpm));
    if (request.vocalLanguage.trim().isNotEmpty())
        setObjectProperty(payload, "vocal_language", normaliseVocalLanguage(request.vocalLanguage));
    setObjectProperty(payload, "use_random_seed", request.useRandomSeed);
    if (!request.useRandomSeed)
        setObjectProperty(payload, "seed", request.seed);
    setObjectProperty(payload, "inference_steps", normaliseInferenceSteps(request.inferenceSteps));
    setObjectProperty(payload, "guidance_scale", normaliseGuidanceScale(request.guidanceScale));
    setObjectProperty(payload, "infer_method", normaliseInferMethod(request.inferMethod));
    if (request.model.trim().isNotEmpty())
        setObjectProperty(payload, "model", request.model.trim());
    else if (defaultModel.trim().isNotEmpty())
        setObjectProperty(payload, "model", defaultModel.trim());
    if (request.keyScale.trim().isNotEmpty())
        setObjectProperty(payload, "key_scale", request.keyScale.trim());
    if (request.timeSignature.trim().isNotEmpty())
        setObjectProperty(payload, "time_signature", request.timeSignature.trim());

    juce::StringPairArray headers;
    if (apiKey.trim().isNotEmpty())
        headers.set("Authorization", authorizationHeaderValue());

    bool modelsInitialized = true;
    try
    {
        const auto healthResponse = requestJson(endpointUrl("health"),
                                                nullptr,
                                                headers,
                                                "GET",
                                                8,
                                                "ACE-Step Health",
                                                false);
        const auto healthData = healthResponse.getProperty("data", {});
        modelsInitialized = getBoolProperty(healthData, "models_initialized", true);
        if (!modelsInitialized)
        {
            emitActivityLog("ACE-Step",
                            "First-run model setup in progress\n"
                            "ACE-Step is still downloading or loading required models. "
                            "The first request can take several minutes before audio generation starts.");
        }
    }
    catch (...)
    {
        // Best-effort hint only. If health parsing fails here, fall back to the normal request flow.
    }

    const auto submitTimeoutSeconds = modelsInitialized
        ? juce::jlimit(10, maxTimeoutSeconds, juce::jmax(60, startupTimeoutSeconds))
        : jobTimeoutSeconds;

    juce::var taskResponse;
    try
    {
        taskResponse = requestJson(endpointUrl("release_task"),
                                   &payload,
                                   headers,
                                   "POST",
                                   submitTimeoutSeconds,
                                   "ACE-Step");
    }
    catch (const std::exception& exception)
    {
        if (!modelsInitialized)
        {
            throw std::runtime_error(("ACE-Step is still preparing first-run models. "
                                      "Leave the ACE-Step server running until the initial downloads and model loading complete, then try again. "
                                      "Details: "
                                      + juce::String(exception.what()))
                                         .toStdString());
        }

        throw;
    }

    const auto taskData = taskResponse.getProperty("data", {});
    const auto taskId = getStringProperty(taskData, "task_id").trim();
    if (taskId.isEmpty())
        throw std::runtime_error("ACE-Step did not return a task id.");

    emitActivityLog("ACE-Step",
                    "Queued audio generation\nTask: "
                        + taskId
                        + "\nPrompt: "
                        + request.prompt.trim()
                        + "\nDuration: "
                        + juce::String(request.durationSeconds, 1)
                        + " sec");

    auto queryPayload = makeObjectVar();
    juce::Array<juce::var> taskIds;
    taskIds.add(taskId);
    setObjectProperty(queryPayload, "task_id_list", juce::var(taskIds));

    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(jobTimeoutSeconds);
    while (std::chrono::steady_clock::now() < deadline)
    {
        const auto queryResponse = requestJson(endpointUrl("query_result"),
                                               &queryPayload,
                                               headers,
                                               "POST",
                                               20,
                                               "ACE-Step Query",
                                               false);
        auto* entries = queryResponse.getProperty("data", {}).getArray();
        if (entries == nullptr || entries->isEmpty())
        {
            std::this_thread::sleep_for(std::chrono::milliseconds(1200));
            continue;
        }

        auto* entryObject = (*entries)[0].getDynamicObject();
        if (entryObject == nullptr)
        {
            std::this_thread::sleep_for(std::chrono::milliseconds(1200));
            continue;
        }

        const auto status = static_cast<int>(entryObject->getProperty("status"));
        if (status == 0)
        {
            std::this_thread::sleep_for(std::chrono::milliseconds(1200));
            continue;
        }

        if (status == 2)
        {
            const auto rawFailure = entryObject->getProperty("result").toString().trim();
            throw std::runtime_error((rawFailure.isNotEmpty() ? rawFailure
                                                              : juce::String("ACE-Step generation failed.")).toStdString());
        }

        const auto rawResult = entryObject->getProperty("result").toString().trim();
        juce::var parsedResult;
        if (juce::JSON::parse(rawResult, parsedResult).failed() || !parsedResult.isArray())
            throw std::runtime_error("ACE-Step returned an invalid generation result.");

        auto* resultArray = parsedResult.getArray();
        if (resultArray == nullptr || resultArray->isEmpty())
            throw std::runtime_error("ACE-Step returned an empty generation result.");

        const auto& resultEntry = resultArray->getReference(0);
        const auto outputUrl = getStringProperty(resultEntry, "file").trim();
        if (outputUrl.isEmpty())
            throw std::runtime_error("ACE-Step did not return an audio download URL.");

        downloadToFile(outputUrl, destinationFile, 120);

        AceStepGenerationResult result;
        result.outputFile = destinationFile;
        result.taskId = taskId;
        result.outputUrl = outputUrl;
        result.prompt = getStringProperty(resultEntry, "prompt").trim();
        result.lyrics = getStringProperty(resultEntry, "lyrics").trim();
        result.generationInfo = getStringProperty(resultEntry, "generation_info").trim();
        result.modelName = inferReturnedModelName(resultEntry);
        emitActivityLog("ACE-Step",
                        "Audio generation completed\nTask: "
                            + taskId
                            + "\nFile: "
                            + destinationFile.getFullPathName()
                            + (result.modelName.isNotEmpty() ? "\nModel: " + result.modelName : juce::String()));
        return result;
    }

    throw std::runtime_error(("ACE-Step generation timed out after "
                              + juce::String(jobTimeoutSeconds)
                              + " seconds.").toStdString());
}

void AceStepClient::setActivityLogCallback(std::function<void(const juce::String& title, const juce::String& body)> callback)
{
    activityLogCallback = std::move(callback);
}

juce::String AceStepClient::normaliseBaseUrl(const juce::String& rawBaseUrl)
{
    auto normalised = rawBaseUrl.trim();
    if (normalised.isEmpty())
        normalised = kDefaultBaseUrl;
    return normalised.trimEnd().trimCharactersAtEnd("/");
}

juce::String AceStepClient::normaliseAudioFormat(const juce::String& rawAudioFormat)
{
    const auto format = rawAudioFormat.trim().toLowerCase();
    if (format == "wav" || format == "flac" || format == "mp3" || format == "opus" || format == "aac" || format == "wav32")
        return format;
    return "wav";
}

juce::String AceStepClient::normaliseQualityPreset(const juce::String& rawQualityPreset)
{
    const auto preset = rawQualityPreset.trim();
    if (preset.equalsIgnoreCase("fast"))
        return "Fast";
    if (preset.equalsIgnoreCase("high"))
        return "High";
    if (preset.equalsIgnoreCase("custom"))
        return "Custom";
    return "Balanced";
}

juce::String AceStepClient::normaliseVocalLanguage(const juce::String& rawVocalLanguage)
{
    const auto language = rawVocalLanguage.trim().toLowerCase();
    return language.isNotEmpty() ? language : juce::String("en");
}

juce::String AceStepClient::normaliseInferMethod(const juce::String& rawInferMethod)
{
    const auto method = rawInferMethod.trim().toLowerCase();
    if (method == "sde")
        return "sde";
    return "ode";
}

int AceStepClient::normaliseInferenceSteps(int inferenceSteps)
{
    return juce::jlimit(1, 200, inferenceSteps);
}

double AceStepClient::normaliseGuidanceScale(double guidanceScale) noexcept
{
    return juce::jlimit(0.0, 20.0, guidanceScale);
}

int AceStepClient::normaliseTimeoutSeconds(int timeoutSeconds)
{
    return juce::jlimit(minTimeoutSeconds, maxTimeoutSeconds, timeoutSeconds);
}

juce::File AceStepClient::appDataDirectory() const
{
    return juce::File::getSpecialLocation(juce::File::userApplicationDataDirectory).getChildFile(kAppName);
}

juce::File AceStepClient::logsDirectory() const
{
    return appDataDirectory().getChildFile("logs");
}

juce::File AceStepClient::preferencesFile() const
{
    return appDataDirectory().getChildFile("ace_step_settings.json");
}

juce::File AceStepClient::authFile() const
{
    return appDataDirectory().getChildFile("ace_step_auth.json");
}

juce::File AceStepClient::debugLogFile() const
{
    return logsDirectory().getChildFile("acestep-http-debug.log");
}

void AceStepClient::loadSavedSettings()
{
    const auto preferences = loadJsonObject(preferencesFile());
    setBaseUrl(getStringProperty(preferences, "base_url", baseUrl));
    setInstallDirectory(getStringProperty(preferences, "install_directory", installDirectory));
    setDefaultModel(getStringProperty(preferences, "default_model", defaultModel));
    setDefaultAudioFormat(getStringProperty(preferences, "default_audio_format", defaultAudioFormat));
    setDefaultQualityPreset(getStringProperty(preferences, "default_quality_preset", defaultQualityPreset));
    setDefaultVocalLanguage(getStringProperty(preferences, "default_vocal_language", defaultVocalLanguage));
    setAutoStartServer(getBoolProperty(preferences, "auto_start_server", autoStartServer));
    setDefaultThinking(getBoolProperty(preferences, "default_thinking", defaultThinking));
    setDefaultUseRandomSeed(getBoolProperty(preferences, "default_use_random_seed", defaultUseRandomSeed));
    setDefaultSeed(getIntProperty(preferences,
                                  "default_seed",
                                  defaultSeed,
                                  std::numeric_limits<int>::min(),
                                  std::numeric_limits<int>::max()));
    setDefaultInferenceSteps(getIntProperty(preferences,
                                            "default_inference_steps",
                                            defaultInferenceSteps,
                                            1,
                                            200));
    setDefaultGuidanceScale(getDoubleProperty(preferences, "default_guidance_scale", defaultGuidanceScale));
    setDefaultInferMethod(getStringProperty(preferences, "default_infer_method", defaultInferMethod));
    setStartupTimeoutSeconds(getIntProperty(preferences,
                                            "startup_timeout_seconds",
                                            startupTimeoutSeconds,
                                            minTimeoutSeconds,
                                            maxTimeoutSeconds));
    setJobTimeoutSeconds(getIntProperty(preferences,
                                        "job_timeout_seconds",
                                        jobTimeoutSeconds,
                                        minTimeoutSeconds,
                                        maxTimeoutSeconds));
}

void AceStepClient::loadSavedAuth()
{
    const auto auth = loadJsonObject(authFile());
    apiKey = getStringProperty(auth, "api_key", apiKey);
}

void AceStepClient::saveAuth() const
{
    auto auth = makeObjectVar();
    setObjectProperty(auth, "api_key", apiKey);
    saveJsonObject(authFile(), auth);
}

juce::String AceStepClient::endpointUrl(const juce::String& endpoint) const
{
    auto cleanEndpoint = endpoint.trim().trimCharactersAtStart("/");
    if (cleanEndpoint.isEmpty())
        cleanEndpoint = "health";
    return baseUrl + "/" + cleanEndpoint;
}

juce::String AceStepClient::authorizationHeaderValue() const
{
    return "Bearer " + apiKey.trim();
}

juce::var AceStepClient::requestJson(const juce::String& url,
                                     const juce::var* payload,
                                     const juce::StringPairArray& headers,
                                     const juce::String& method,
                                     int timeoutSeconds,
                                     const juce::String& label,
                                     bool emitActivityLogs) const
{
    juce::String requestSummary = method + " " + url;
    if (payload != nullptr)
        requestSummary << "\n\nPayload\n" << prettyJsonForActivityLog(*payload);
    if (emitActivityLogs)
        emitActivityLog(label + " Request", requestSummary);

    juce::String debugRequestSummary = method + " " + url;
    if (payload != nullptr)
        debugRequestSummary << "\n\nPayload\n" << juce::JSON::toString(*payload, false);
    appendDebugLog(label + " Request", debugRequestSummary);

    juce::String extraHeaders = "Content-Type: application/json\r\n";
    for (int index = 0; index < headers.size(); ++index)
        extraHeaders << headers.getAllKeys()[index] << ": " << headers.getAllValues()[index] << "\r\n";

    int statusCode = 0;
    juce::StringPairArray responseHeaders;

    auto requestUrl = juce::URL(url);
    if (payload != nullptr)
        requestUrl = requestUrl.withPOSTData(juce::JSON::toString(*payload, false));

    auto inputStream = requestUrl.createInputStream(juce::URL::InputStreamOptions(juce::URL::ParameterHandling::inAddress)
                                                        .withExtraHeaders(extraHeaders)
                                                        .withConnectionTimeoutMs(timeoutSeconds * 1000)
                                                        .withResponseHeaders(&responseHeaders)
                                                        .withStatusCode(&statusCode)
                                                        .withNumRedirectsToFollow(4)
                                                        .withHttpRequestCmd(method));
    if (inputStream == nullptr)
    {
        const auto message = statusCode != 0
            ? "HTTP " + juce::String(statusCode) + "\n\nNo response stream was returned."
            : "Network error: could not connect.";
        appendDebugLog(label + " Error", message);
        if (emitActivityLogs)
            emitActivityLog(label + " Error", message);
        throw std::runtime_error((label + " request failed.").toStdString());
    }

    const auto raw = inputStream->readEntireStreamAsString();
    appendDebugLog(label + " Response",
                   "HTTP " + juce::String(statusCode) + "\n\nRaw Response\n" + raw);
    if (emitActivityLogs)
        emitActivityLog(label + " Response",
                        "HTTP " + juce::String(statusCode) + "\n\n" + prettyRawJsonForActivityLog(raw));

    if (statusCode >= 400)
        throw std::runtime_error((label + " API error: " + juce::String(statusCode) + " " + raw).toStdString());

    const auto trimmedRaw = raw.trim();
    if (trimmedRaw == "null")
    {
        throw std::runtime_error((label
                                  + " returned a null response. ACE-Step may still be starting or the local server may be stuck."
                                  + " Restart ACE-Step and try again.")
                                     .toStdString());
    }

    if (trimmedRaw.isEmpty())
        throw std::runtime_error((label + " returned an empty response.").toStdString());

    juce::var parsed;
    if (juce::JSON::parse(trimmedRaw, parsed).failed() || !parsed.isObject())
        throw std::runtime_error((label + " returned invalid JSON: " + trimmedRaw.substring(0, 300)).toStdString());

    return parsed;
}

void AceStepClient::downloadToFile(const juce::String& urlOrPath,
                                   const juce::File& destinationFile,
                                   int timeoutSeconds) const
{
    const auto resolvedUrl = resolvedAudioUrl(baseUrl, urlOrPath);
    juce::String extraHeaders;
    if (apiKey.trim().isNotEmpty())
        extraHeaders << "Authorization: " << authorizationHeaderValue() << "\r\n";

    int statusCode = 0;
    juce::StringPairArray responseHeaders;
    auto inputStream = juce::URL(resolvedUrl).createInputStream(juce::URL::InputStreamOptions(juce::URL::ParameterHandling::inAddress)
                                                                    .withExtraHeaders(extraHeaders)
                                                                    .withConnectionTimeoutMs(timeoutSeconds * 1000)
                                                                    .withResponseHeaders(&responseHeaders)
                                                                    .withStatusCode(&statusCode)
                                                                    .withNumRedirectsToFollow(4)
                                                                    .withHttpRequestCmd("GET"));
    if (inputStream == nullptr || statusCode >= 400)
        throw std::runtime_error(("ACE-Step audio download failed from " + resolvedUrl).toStdString());

    destinationFile.getParentDirectory().createDirectory();
    juce::FileOutputStream output(destinationFile);
    if (!output.openedOk())
        throw std::runtime_error(("Could not write ACE-Step output to " + destinationFile.getFullPathName()).toStdString());

    if (output.writeFromInputStream(*inputStream, -1) <= 0)
        throw std::runtime_error("ACE-Step output download produced no audio data.");
}

void AceStepClient::emitActivityLog(const juce::String& title, const juce::String& body) const
{
    if (activityLogCallback == nullptr)
        return;

    activityLogCallback(title, body);
}

void AceStepClient::appendDebugLog(const juce::String& title, const juce::String& body) const
{
    const auto file = debugLogFile();
    file.getParentDirectory().createDirectory();

    if (!file.existsAsFile())
        file.replaceWithText("Mutagen ACE-Step Debug Log\n\n", false, false, "\n");

    const auto timestamp = juce::Time::getCurrentTime().formatted("[%Y-%m-%d %H:%M:%S]");
    const auto entry = timestamp
        + " "
        + title.trim()
        + "\n"
        + prepareDebugLogText(body)
        + "\n\n";
    file.appendText(entry, false, false, "\n");
}

} // namespace aims
