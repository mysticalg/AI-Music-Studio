#include "AIClient.h"
#include "ProjectModel.h"

#include <cmath>
#include <cstdlib>
#include <stdexcept>

namespace aims
{
namespace
{
constexpr const char* kAppName = "Mutagen";
constexpr const char* kOpenAiApiUrl = "https://api.openai.com/v1/responses";
constexpr const char* kDefaultOpenAiModel = "gpt-5-codex";
constexpr const char* kDefaultOllamaBaseUrl = "http://127.0.0.1:11434";

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

double getDoubleProperty(const juce::var& objectVar, const juce::Identifier& name, double fallback)
{
    if (auto* object = objectVar.getDynamicObject())
    {
        const auto value = object->getProperty(name);
        if (!value.isVoid())
        {
            const auto parsed = static_cast<double>(value);
            if (std::isfinite(parsed))
                return parsed;
        }
    }

    return fallback;
}

juce::String getenvString(const char* name)
{
    if (name == nullptr)
        return {};

#if defined(_MSC_VER)
    char* valueBuffer = nullptr;
    size_t valueLength = 0;
    if (_dupenv_s(&valueBuffer, &valueLength, name) == 0 && valueBuffer != nullptr)
    {
        const auto value = juce::String::fromUTF8(valueBuffer).trim();
        std::free(valueBuffer);
        return value;
    }
    return {};
#else
    if (const auto* value = std::getenv(name))
        return juce::String::fromUTF8(value).trim();

    return {};
#endif
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

AIComposeTrack parseComposeTrack(const juce::var& trackVar)
{
    AIComposeTrack track;
    auto* object = trackVar.getDynamicObject();
    if (object == nullptr)
        return track;

    track.name = object->getProperty("name").toString().trim();
    track.instrument = object->getProperty("instrument").toString().trim();

    auto* notesArray = object->getProperty("notes").getArray();
    if (notesArray == nullptr)
        return track;

    track.notes.reserve(static_cast<size_t>(notesArray->size()));
    for (const auto& noteVar : *notesArray)
    {
        auto* noteObject = noteVar.getDynamicObject();
        if (noteObject == nullptr)
            continue;

        AIComposeNote note;
        note.startBeat = juce::jmax(0.0, static_cast<double>(noteObject->getProperty("start_beat")));
        note.durationBeat = juce::jmax(0.125, static_cast<double>(noteObject->getProperty("duration_beat")));
        note.pitch = juce::jlimit(kEditableMidiPitchMin,
                                  kEditableMidiPitchMax,
                                  static_cast<int>(noteObject->getProperty("pitch")));
        note.velocity = juce::jlimit(1, 127, static_cast<int>(noteObject->getProperty("velocity")));
        track.notes.push_back(note);
    }

    return track;
}
} // namespace

AIClient::AIClient()
{
    remoteModel = getenvString("OPENAI_MODEL");
    if (remoteModel.isEmpty())
        remoteModel = kDefaultOpenAiModel;

    auto configuredOllamaBase = getenvString("OLLAMA_BASE_URL");
    if (configuredOllamaBase.isEmpty())
        configuredOllamaBase = getenvString("OLLAMA_HOST");
    ollamaBaseUrl = normaliseOllamaBaseUrl(configuredOllamaBase);

    apiKey = getenvString("OPENAI_API_KEY");

    loadSavedSettings();
    loadSavedAuth();
}

AIClient::Provider AIClient::getProvider() const noexcept
{
    return provider;
}

juce::String AIClient::getProviderKey() const
{
    return provider == Provider::ollama ? "ollama" : "openai";
}

juce::String AIClient::getRemoteModel() const
{
    return remoteModel;
}

juce::String AIClient::getOllamaBaseUrl() const
{
    return ollamaBaseUrl;
}

juce::String AIClient::getOllamaModel() const
{
    return ollamaModel;
}

int AIClient::getRequestTimeoutSeconds() const noexcept
{
    return requestTimeoutSeconds;
}

bool AIClient::hasSavedCredential() const noexcept
{
    return apiKey.isNotEmpty() || oauthAccessToken.isNotEmpty();
}

bool AIClient::isEnabled() const noexcept
{
    if (provider == Provider::ollama)
        return ollamaBaseUrl.isNotEmpty() && ollamaModel.isNotEmpty();

    return hasSavedCredential();
}

juce::String AIClient::authStatus() const
{
    if (provider == Provider::ollama)
    {
        if (ollamaModel.isNotEmpty())
            return "AI: Local Ollama (" + ollamaModel + ")";
        return "AI: Local Ollama at " + ollamaBaseUrl + " (no model selected)";
    }

    if (oauthAccessToken.isNotEmpty())
    {
        const auto nowSeconds = juce::Time::getCurrentTime().toMilliseconds() / 1000.0;
        if (oauthExpiresAt > nowSeconds)
        {
            const auto minsRemaining = juce::jmax(0, static_cast<int>((oauthExpiresAt - nowSeconds) / 60.0));
            return "AI: OpenAI via OAuth (expires in ~" + juce::String(minsRemaining) + " min)";
        }

        return "AI: OpenAI via OAuth";
    }

    if (apiKey.isNotEmpty())
        return "AI: OpenAI via API key";

    return "AI: OpenAI not connected";
}

juce::StringArray AIClient::remoteModelChoices() const
{
    juce::StringArray choices;
    for (const auto& value : { remoteModel, getenvString("OPENAI_MODEL"), juce::String(kDefaultOpenAiModel) })
    {
        const auto trimmed = value.trim();
        if (trimmed.isNotEmpty() && !choices.contains(trimmed))
            choices.add(trimmed);
    }

    return choices;
}

juce::StringArray AIClient::availableOllamaModels(const juce::String& baseUrl) const
{
    const auto response = requestJson(ollamaEndpoint("api/tags", baseUrl),
                                      nullptr,
                                      {},
                                      "GET",
                                      8,
                                      "Ollama");

    juce::StringArray models;
    if (auto* object = response.getDynamicObject())
    {
        if (auto* modelArray = object->getProperty("models").getArray())
        {
            for (const auto& rawModel : *modelArray)
            {
                auto* modelObject = rawModel.getDynamicObject();
                if (modelObject == nullptr)
                    continue;

                const auto name = modelObject->hasProperty("name")
                    ? modelObject->getProperty("name").toString().trim()
                    : modelObject->getProperty("model").toString().trim();
                if (name.isNotEmpty() && !models.contains(name))
                    models.add(name);
            }
        }
    }

    return models;
}

void AIClient::setProvider(Provider newProvider)
{
    provider = newProvider;
}

void AIClient::setRemoteModel(const juce::String& newModel)
{
    remoteModel = newModel.trim();
    if (remoteModel.isEmpty())
        remoteModel = getenvString("OPENAI_MODEL");
    if (remoteModel.isEmpty())
        remoteModel = kDefaultOpenAiModel;
}

void AIClient::setOllamaConnection(const juce::String& newBaseUrl, const juce::String& newModel)
{
    ollamaBaseUrl = normaliseOllamaBaseUrl(newBaseUrl);
    if (newModel.isNotEmpty())
        ollamaModel = newModel.trim();
}

void AIClient::setRequestTimeoutSeconds(int timeoutSeconds)
{
    requestTimeoutSeconds = normaliseRequestTimeoutSeconds(timeoutSeconds);
}

void AIClient::setActivityLogCallback(std::function<void(const juce::String& title, const juce::String& body)> callback)
{
    activityLogCallback = std::move(callback);
}

void AIClient::saveSettings() const
{
    auto preferences = loadJsonObject(preferencesFile());
    setObjectProperty(preferences, "ai_provider", getProviderKey());
    setObjectProperty(preferences, "ai_remote_model", remoteModel);
    setObjectProperty(preferences, "ai_ollama_base_url", ollamaBaseUrl);
    setObjectProperty(preferences, "ai_ollama_model", ollamaModel);
    setObjectProperty(preferences, "ai_request_timeout_seconds", requestTimeoutSeconds);
    saveJsonObject(preferencesFile(), preferences);
}

void AIClient::setApiKey(const juce::String& newApiKey)
{
    apiKey = newApiKey.trim();
    oauthAccessToken.clear();
    oauthRefreshToken.clear();
    oauthExpiresAt = 0.0;
    saveAuth();
}

void AIClient::clearAuth()
{
    apiKey.clear();
    oauthAccessToken.clear();
    oauthRefreshToken.clear();
    oauthExpiresAt = 0.0;
    saveAuth();
}

juce::var AIClient::runJsonPrompt(const juce::String& systemInstruction, const juce::String& userInstruction) const
{
    if (provider == Provider::ollama)
        return runOllamaJsonPrompt(systemInstruction, userInstruction);

    return runOpenAiJsonPrompt(systemInstruction, userInstruction);
}

AIClient::Provider AIClient::normaliseProvider(const juce::String& rawProvider)
{
    const auto value = rawProvider.trim().toLowerCase();
    if (value == "ollama" || value == "local" || value == "ollama_local")
        return Provider::ollama;

    return Provider::openAI;
}

juce::String AIClient::normaliseOllamaBaseUrl(const juce::String& baseUrl)
{
    auto normalized = baseUrl.trim();
    if (normalized.isEmpty())
        normalized = kDefaultOllamaBaseUrl;

    return normalized.trimEnd().trimCharactersAtEnd("/");
}

int AIClient::normaliseRequestTimeoutSeconds(int timeoutSeconds)
{
    return juce::jlimit(minRequestTimeoutSeconds, maxRequestTimeoutSeconds, timeoutSeconds);
}

juce::File AIClient::appDataDirectory() const
{
    return juce::File::getSpecialLocation(juce::File::userApplicationDataDirectory).getChildFile(kAppName);
}

juce::File AIClient::logsDirectory() const
{
    return appDataDirectory().getChildFile("logs");
}

juce::File AIClient::preferencesFile() const
{
    return appDataDirectory().getChildFile("preferences.json");
}

juce::File AIClient::authFile() const
{
    return appDataDirectory().getChildFile("ai_auth.json");
}

juce::File AIClient::debugLogFile() const
{
    return logsDirectory().getChildFile("ai-http-debug.log");
}

void AIClient::loadSavedSettings()
{
    const auto preferences = loadJsonObject(preferencesFile());
    provider = normaliseProvider(getStringProperty(preferences, "ai_provider", getProviderKey()));
    setRemoteModel(getStringProperty(preferences, "ai_remote_model", remoteModel));
    setOllamaConnection(getStringProperty(preferences, "ai_ollama_base_url", ollamaBaseUrl),
                        getStringProperty(preferences, "ai_ollama_model", ollamaModel));
    setRequestTimeoutSeconds(getIntProperty(preferences,
                                            "ai_request_timeout_seconds",
                                            requestTimeoutSeconds,
                                            minRequestTimeoutSeconds,
                                            maxRequestTimeoutSeconds));
}

void AIClient::loadSavedAuth()
{
    const auto auth = loadJsonObject(authFile());
    if (apiKey.isEmpty())
        apiKey = getStringProperty(auth, "api_key", apiKey);
    oauthAccessToken = getStringProperty(auth, "oauth_access_token", oauthAccessToken);
    oauthRefreshToken = getStringProperty(auth, "oauth_refresh_token", oauthRefreshToken);
    oauthExpiresAt = getDoubleProperty(auth, "oauth_expires_at", oauthExpiresAt);
}

void AIClient::saveAuth() const
{
    auto auth = makeObjectVar();
    setObjectProperty(auth, "api_key", apiKey);
    setObjectProperty(auth, "oauth_access_token", oauthAccessToken);
    setObjectProperty(auth, "oauth_refresh_token", oauthRefreshToken);
    setObjectProperty(auth, "oauth_expires_at", oauthExpiresAt);
    saveJsonObject(authFile(), auth);
}

juce::String AIClient::authorizationHeaderValue() const
{
    const auto token = oauthAccessToken.isNotEmpty() ? oauthAccessToken : apiKey;
    return "Bearer " + token;
}

juce::String AIClient::ollamaEndpoint(const juce::String& endpoint, const juce::String& overrideBaseUrl) const
{
    const auto baseUrl = normaliseOllamaBaseUrl(overrideBaseUrl.isNotEmpty() ? overrideBaseUrl : ollamaBaseUrl);
    auto cleanEndpoint = endpoint.trim().trimCharactersAtStart("/");
    if (cleanEndpoint.isEmpty())
        cleanEndpoint = "api/tags";

    if (baseUrl.endsWithIgnoreCase("/api"))
        cleanEndpoint = cleanEndpoint.fromFirstOccurrenceOf("api/", false, false);

    return baseUrl + "/" + cleanEndpoint;
}

juce::var AIClient::requestJson(const juce::String& url,
                                const juce::var* payload,
                                const juce::StringPairArray& headers,
                                const juce::String& method,
                                int timeoutSeconds,
                                const juce::String& errorPrefix) const
{
    juce::String requestSummary = method + " " + url;
    if (payload != nullptr)
        requestSummary << "\n\nPayload\n" << prettyJsonForActivityLog(*payload);
    emitActivityLog(errorPrefix + " Request", requestSummary);

    juce::String debugRequestSummary = method + " " + url;
    if (payload != nullptr)
        debugRequestSummary << "\n\nPayload\n" << juce::JSON::toString(*payload, false);
    appendDebugLog(errorPrefix + " Request", debugRequestSummary);

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
        if (statusCode != 0)
        {
            appendDebugLog(errorPrefix + " Error",
                           "HTTP " + juce::String(statusCode) + "\n\nNo response stream was returned.");
            emitActivityLog(errorPrefix + " Error",
                            "HTTP " + juce::String(statusCode) + "\n\nNo response stream was returned.");
            throw std::runtime_error((errorPrefix + " API error: " + juce::String(statusCode)).toStdString());
        }

        appendDebugLog(errorPrefix + " Error", "Network error: could not connect.");
        emitActivityLog(errorPrefix + " Error", "Network error: could not connect.");
        throw std::runtime_error((errorPrefix + " network error: could not connect.").toStdString());
    }

    const auto raw = inputStream->readEntireStreamAsString();
    appendDebugLog(errorPrefix + " Response",
                   "HTTP " + juce::String(statusCode) + "\n\nRaw Response\n" + raw);
    emitActivityLog(errorPrefix + " Response",
                    "HTTP " + juce::String(statusCode) + "\n\n" + prettyRawJsonForActivityLog(raw));
    if (statusCode >= 400)
    {
        appendDebugLog(errorPrefix + " Error",
                       "HTTP " + juce::String(statusCode) + "\n\nRaw Response\n" + raw);
        emitActivityLog(errorPrefix + " Error",
                        "HTTP " + juce::String(statusCode) + "\n\n" + prettyRawJsonForActivityLog(raw));
        throw std::runtime_error((errorPrefix + " API error: " + juce::String(statusCode) + " " + raw).toStdString());
    }

    juce::var parsed;
    if (juce::JSON::parse(raw, parsed).failed() || !parsed.isObject())
    {
        appendDebugLog(errorPrefix + " Error", "Invalid JSON\n\n" + raw);
        emitActivityLog(errorPrefix + " Error",
                        "Invalid JSON\n\n" + truncateActivityLogText(raw));
        throw std::runtime_error((errorPrefix + " returned invalid JSON: " + raw.substring(0, 300)).toStdString());
    }

    return parsed;
}

juce::var AIClient::runOpenAiJsonPrompt(const juce::String& systemInstruction, const juce::String& userInstruction) const
{
    if (!hasSavedCredential())
        throw std::runtime_error("Remote OpenAI is not connected. Open AI Settings and add an API key or use saved OAuth credentials.");

    auto payload = makeObjectVar();
    setObjectProperty(payload, "model", remoteModel.isNotEmpty() ? remoteModel : juce::String(kDefaultOpenAiModel));

    juce::Array<juce::var> inputItems;
    auto systemObject = makeObjectVar();
    setObjectProperty(systemObject, "role", "system");
    setObjectProperty(systemObject, "content", systemInstruction);
    inputItems.add(systemObject);

    auto userObject = makeObjectVar();
    setObjectProperty(userObject, "role", "user");
    setObjectProperty(userObject, "content", userInstruction);
    inputItems.add(userObject);

    setObjectProperty(payload, "input", juce::var(inputItems));

    juce::StringPairArray headers;
    headers.set("Authorization", authorizationHeaderValue());
    const auto response = requestJson(kOpenAiApiUrl,
                                      &payload,
                                      headers,
                                      "POST",
                                      requestTimeoutSeconds,
                                      "OpenAI");

    auto text = getStringProperty(response, "output_text").trim();
    if (text.isEmpty())
    {
        if (auto* object = response.getDynamicObject())
        {
            if (auto* outputArray = object->getProperty("output").getArray())
            {
                for (const auto& item : *outputArray)
                {
                    auto* itemObject = item.getDynamicObject();
                    if (itemObject == nullptr)
                        continue;

                    auto* contentArray = itemObject->getProperty("content").getArray();
                    if (contentArray == nullptr)
                        continue;

                    for (const auto& content : *contentArray)
                    {
                        auto* contentObject = content.getDynamicObject();
                        if (contentObject != nullptr
                            && contentObject->getProperty("type").toString() == "output_text")
                        {
                            text += contentObject->getProperty("text").toString();
                        }
                    }
                }
            }
        }
    }

    if (text.trim().isEmpty())
        throw std::runtime_error("No text returned by OpenAI.");

    appendDebugLog("OpenAI Output Text", text);

    juce::var parsed;
    if (juce::JSON::parse(text, parsed).failed() || !parsed.isObject())
        throw std::runtime_error(("Model response was not valid JSON: " + text.substring(0, 300)).toStdString());

    appendDebugLog("OpenAI Parsed Output", juce::JSON::toString(parsed, false));
    emitActivityLog("OpenAI Parsed Output", prettyJsonForActivityLog(parsed));
    return parsed;
}

juce::var AIClient::runOllamaJsonPrompt(const juce::String& systemInstruction, const juce::String& userInstruction) const
{
    if (ollamaModel.isEmpty())
        throw std::runtime_error("Local Ollama model is not selected. Open AI Settings and choose an Ollama model.");

    auto payload = makeObjectVar();
    setObjectProperty(payload, "model", ollamaModel);

    juce::Array<juce::var> messages;
    auto systemObject = makeObjectVar();
    setObjectProperty(systemObject, "role", "system");
    setObjectProperty(systemObject, "content", systemInstruction);
    messages.add(systemObject);

    auto userObject = makeObjectVar();
    setObjectProperty(userObject, "role", "user");
    setObjectProperty(userObject, "content", userInstruction);
    messages.add(userObject);

    setObjectProperty(payload, "messages", juce::var(messages));
    setObjectProperty(payload, "format", "json");
    setObjectProperty(payload, "stream", false);

    const auto response = requestJson(ollamaEndpoint("api/chat"),
                                      &payload,
                                      {},
                                      "POST",
                                      requestTimeoutSeconds,
                                      "Ollama");

    auto* responseObject = response.getDynamicObject();
    auto* messageObject = responseObject != nullptr
        ? responseObject->getProperty("message").getDynamicObject()
        : nullptr;
    const auto text = messageObject != nullptr ? messageObject->getProperty("content").toString().trim()
                                               : juce::String();
    if (text.isEmpty())
        throw std::runtime_error("No text returned by Ollama.");

    appendDebugLog("Ollama Output Text", text);

    juce::var parsed;
    if (juce::JSON::parse(text, parsed).failed() || !parsed.isObject())
        throw std::runtime_error(("Model response was not valid JSON: " + text.substring(0, 300)).toStdString());

    appendDebugLog("Ollama Parsed Output", juce::JSON::toString(parsed, false));
    emitActivityLog("Ollama Parsed Output", prettyJsonForActivityLog(parsed));
    return parsed;
}

void AIClient::emitActivityLog(const juce::String& title, const juce::String& body) const
{
    if (activityLogCallback == nullptr)
        return;

    activityLogCallback(title, body);
}

void AIClient::appendDebugLog(const juce::String& title, const juce::String& body) const
{
    const auto file = debugLogFile();
    file.getParentDirectory().createDirectory();

    if (!file.existsAsFile())
    {
        file.replaceWithText("Mutagen AI Debug Log\n\n", false, false, "\n");
    }

    const auto timestamp = juce::Time::getCurrentTime().formatted("[%Y-%m-%d %H:%M:%S]");
    const auto entry = timestamp
        + " "
        + title.trim()
        + "\n"
        + prepareDebugLogText(body)
        + "\n\n";
    file.appendText(entry, false, false, "\n");
}

AIComposer::AIComposer(AIClient clientIn)
    : client(std::move(clientIn))
{
}

AIComposeResult AIComposer::compose(const juce::String& prompt,
                                    int bars,
                                    int bpm,
                                    int minTracks,
                                    int maxTracks) const
{
    const auto safeMinTracks = juce::jlimit(1, 16, minTracks);
    const auto safeMaxTracks = juce::jlimit(safeMinTracks, 16, maxTracks);
    const juce::String systemInstruction =
        "You are a MIDI composer for a DAW. Return strict JSON only with the schema: "
        "{\"tracks\": [{\"name\": str, \"instrument\": str, \"notes\": "
        "[{\"start_beat\": number, \"duration_beat\": number, \"pitch\": int, \"velocity\": int}]}]}. "
        "Keep pitches in MIDI range " + juce::String(kEditableMidiPitchMin) + ".."
        + juce::String(kEditableMidiPitchMax) + " and fit inside requested bars.";

    const juce::String userInstruction =
        "Create a musically coherent MIDI composition. Prompt: " + prompt + ". Bars: " + juce::String(juce::jmax(1, bars))
        + ". BPM: " + juce::String(juce::jmax(20, bpm))
        + ". Use "
        + (safeMinTracks == safeMaxTracks
               ? ("exactly " + juce::String(safeMinTracks) + " track(s)")
               : ("between " + juce::String(safeMinTracks) + " and " + juce::String(safeMaxTracks) + " tracks"))
        + " and musically coherent note timing.";

    const auto rawResult = client.runJsonPrompt(systemInstruction, userInstruction);
    auto* object = rawResult.getDynamicObject();
    if (object == nullptr)
        throw std::runtime_error("AI returned an unexpected response.");

    auto* tracksArray = object->getProperty("tracks").getArray();
    if (tracksArray == nullptr || tracksArray->isEmpty())
        throw std::runtime_error("AI returned no tracks.");

    AIComposeResult result;
    result.tracks.reserve(static_cast<size_t>(tracksArray->size()));
    for (const auto& trackVar : *tracksArray)
    {
        auto track = parseComposeTrack(trackVar);
        if (track.name.isEmpty())
            track.name = "AI Track " + juce::String(static_cast<int>(result.tracks.size()) + 1);
        if (track.instrument.isEmpty())
            track.instrument = "AI Instrument";
        if (!track.notes.empty())
            result.tracks.push_back(std::move(track));
    }

    if (result.tracks.empty())
        throw std::runtime_error("AI returned invalid track data.");

    return result;
}

} // namespace aims
