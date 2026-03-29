#pragma once

#include <juce_core/juce_core.h>

#include <functional>
#include <vector>

namespace aims
{

struct AIComposeNote
{
    double startBeat = 0.0;
    double durationBeat = 0.5;
    int pitch = 60;
    int velocity = 100;
};

struct AIComposeTrack
{
    juce::String name;
    juce::String instrument;
    std::vector<AIComposeNote> notes;
};

struct AIComposeResult
{
    std::vector<AIComposeTrack> tracks;
};

class AIClient
{
public:
    enum class Provider
    {
        openAI,
        ollama
    };

    static constexpr int defaultRequestTimeoutSeconds = 120;
    static constexpr int minRequestTimeoutSeconds = 15;
    static constexpr int maxRequestTimeoutSeconds = 1800;

    AIClient();

    Provider getProvider() const noexcept;
    juce::String getProviderKey() const;
    juce::String getRemoteModel() const;
    juce::String getOllamaBaseUrl() const;
    juce::String getOllamaModel() const;
    int getRequestTimeoutSeconds() const noexcept;
    bool hasSavedCredential() const noexcept;
    bool isEnabled() const noexcept;
    juce::String authStatus() const;
    juce::StringArray remoteModelChoices() const;
    juce::StringArray availableOllamaModels(const juce::String& baseUrl = {}) const;

    void setProvider(Provider newProvider);
    void setRemoteModel(const juce::String& newModel);
    void setOllamaConnection(const juce::String& newBaseUrl, const juce::String& newModel = {});
    void setRequestTimeoutSeconds(int timeoutSeconds);
    void saveSettings() const;
    void setActivityLogCallback(std::function<void(const juce::String& title, const juce::String& body)> callback);

    void setApiKey(const juce::String& newApiKey);
    void clearAuth();

    juce::var runJsonPrompt(const juce::String& systemInstruction, const juce::String& userInstruction) const;

private:
    static Provider normaliseProvider(const juce::String& rawProvider);
    static juce::String normaliseOllamaBaseUrl(const juce::String& baseUrl);
    static int normaliseRequestTimeoutSeconds(int timeoutSeconds);

    juce::File appDataDirectory() const;
    juce::File logsDirectory() const;
    juce::File preferencesFile() const;
    juce::File authFile() const;
    juce::File debugLogFile() const;
    void loadSavedSettings();
    void loadSavedAuth();
    void saveAuth() const;
    juce::String authorizationHeaderValue() const;
    juce::String ollamaEndpoint(const juce::String& endpoint, const juce::String& overrideBaseUrl = {}) const;
    juce::var requestJson(const juce::String& url,
                          const juce::var* payload,
                          const juce::StringPairArray& headers,
                          const juce::String& method,
                          int timeoutSeconds,
                          const juce::String& errorPrefix) const;
    juce::var runOpenAiJsonPrompt(const juce::String& systemInstruction, const juce::String& userInstruction) const;
    juce::var runOllamaJsonPrompt(const juce::String& systemInstruction, const juce::String& userInstruction) const;
    void emitActivityLog(const juce::String& title, const juce::String& body) const;
    void appendDebugLog(const juce::String& title, const juce::String& body) const;

    Provider provider = Provider::openAI;
    juce::String remoteModel;
    juce::String ollamaBaseUrl;
    juce::String ollamaModel;
    int requestTimeoutSeconds = defaultRequestTimeoutSeconds;
    juce::String apiKey;
    juce::String oauthAccessToken;
    juce::String oauthRefreshToken;
    double oauthExpiresAt = 0.0;
    std::function<void(const juce::String&, const juce::String&)> activityLogCallback;
};

class AIComposer
{
public:
    explicit AIComposer(AIClient clientIn);

    AIComposeResult compose(const juce::String& prompt, int bars, int bpm) const;

private:
    AIClient client;
};

} // namespace aims
