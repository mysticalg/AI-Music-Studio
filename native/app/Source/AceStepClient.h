#pragma once

#include <juce_core/juce_core.h>

#include <functional>

namespace aims
{

struct AceStepGenerationRequest
{
    juce::String prompt;
    juce::String lyrics;
    juce::String model;
    int bpm = 120;
    juce::String keyScale;
    juce::String timeSignature;
    double durationSeconds = 30.0;
    bool thinking = true;
    juce::String audioFormat = "wav";
    juce::String vocalLanguage = "en";
    bool useRandomSeed = true;
    int seed = -1;
    int inferenceSteps = 8;
    double guidanceScale = 7.0;
    juce::String inferMethod = "ode";
};

struct AceStepGenerationResult
{
    juce::File outputFile;
    juce::String taskId;
    juce::String outputUrl;
    juce::String prompt;
    juce::String lyrics;
    juce::String generationInfo;
    juce::String modelName;
};

class AceStepClient
{
public:
    static constexpr int defaultStartupTimeoutSeconds = 120;
    static constexpr int defaultJobTimeoutSeconds = 1800;
    static constexpr int minTimeoutSeconds = 5;
    static constexpr int maxTimeoutSeconds = 7200;

    AceStepClient();

    juce::String getBaseUrl() const;
    juce::String getInstallDirectory() const;
    juce::String getDefaultModel() const;
    juce::String getDefaultAudioFormat() const;
    juce::String getDefaultQualityPreset() const;
    juce::String getDefaultVocalLanguage() const;
    bool getAutoStartServer() const noexcept;
    bool getDefaultThinking() const noexcept;
    bool getDefaultUseRandomSeed() const noexcept;
    int getDefaultSeed() const noexcept;
    int getDefaultInferenceSteps() const noexcept;
    double getDefaultGuidanceScale() const noexcept;
    juce::String getDefaultInferMethod() const;
    int getStartupTimeoutSeconds() const noexcept;
    int getJobTimeoutSeconds() const noexcept;

    void setBaseUrl(const juce::String& newBaseUrl);
    void setApiKey(const juce::String& newApiKey);
    void clearApiKey();
    void setInstallDirectory(const juce::String& newInstallDirectory);
    void setDefaultModel(const juce::String& newDefaultModel);
    void setDefaultAudioFormat(const juce::String& newAudioFormat);
    void setDefaultQualityPreset(const juce::String& newQualityPreset);
    void setDefaultVocalLanguage(const juce::String& newVocalLanguage);
    void setAutoStartServer(bool shouldAutoStart) noexcept;
    void setDefaultThinking(bool shouldUseThinking) noexcept;
    void setDefaultUseRandomSeed(bool shouldUseRandomSeed) noexcept;
    void setDefaultSeed(int newSeed) noexcept;
    void setDefaultInferenceSteps(int newInferenceSteps);
    void setDefaultGuidanceScale(double newGuidanceScale) noexcept;
    void setDefaultInferMethod(const juce::String& newInferMethod);
    void setStartupTimeoutSeconds(int timeoutSeconds);
    void setJobTimeoutSeconds(int timeoutSeconds);
    void saveSettings() const;

    juce::StringArray availableModels(int timeoutMs = 2000) const;
    bool isServerReachable(int timeoutMs = 1500) const;
    bool isServerPortAcceptingConnections(int timeoutMs = 500) const;
    juce::Result waitForServerReady() const;
    bool hasLaunchConfiguration() const;
    juce::File launchScriptFile() const;
    juce::String statusSummary() const;
    AceStepGenerationResult generateToFile(const AceStepGenerationRequest& request,
                                           const juce::File& destinationFile) const;

    void setActivityLogCallback(std::function<void(const juce::String& title, const juce::String& body)> callback);

private:
    static juce::String normaliseBaseUrl(const juce::String& baseUrl);
    static juce::String normaliseAudioFormat(const juce::String& audioFormat);
    static juce::String normaliseQualityPreset(const juce::String& qualityPreset);
    static juce::String normaliseVocalLanguage(const juce::String& vocalLanguage);
    static juce::String normaliseInferMethod(const juce::String& inferMethod);
    static int normaliseInferenceSteps(int inferenceSteps);
    static double normaliseGuidanceScale(double guidanceScale) noexcept;
    static int normaliseTimeoutSeconds(int timeoutSeconds);

    juce::File appDataDirectory() const;
    juce::File logsDirectory() const;
    juce::File preferencesFile() const;
    juce::File authFile() const;
    juce::File debugLogFile() const;
    void loadSavedSettings();
    void loadSavedAuth();
    void saveAuth() const;
    juce::String endpointUrl(const juce::String& endpoint) const;
    juce::String authorizationHeaderValue() const;
    juce::var requestJson(const juce::String& url,
                          const juce::var* payload,
                          const juce::StringPairArray& headers,
                          const juce::String& method,
                          int timeoutSeconds,
                          const juce::String& label,
                          bool emitActivityLogs = true) const;
    void downloadToFile(const juce::String& urlOrPath,
                        const juce::File& destinationFile,
                        int timeoutSeconds) const;
    void emitActivityLog(const juce::String& title, const juce::String& body) const;
    void appendDebugLog(const juce::String& title, const juce::String& body) const;

    juce::String baseUrl;
    juce::String apiKey;
    juce::String installDirectory;
    juce::String defaultModel;
    juce::String defaultAudioFormat;
    juce::String defaultQualityPreset = "Balanced";
    juce::String defaultVocalLanguage = "en";
    bool autoStartServer = true;
    bool defaultThinking = true;
    bool defaultUseRandomSeed = true;
    int defaultSeed = -1;
    int defaultInferenceSteps = 8;
    double defaultGuidanceScale = 7.0;
    juce::String defaultInferMethod = "ode";
    int startupTimeoutSeconds = defaultStartupTimeoutSeconds;
    int jobTimeoutSeconds = defaultJobTimeoutSeconds;
    std::function<void(const juce::String&, const juce::String&)> activityLogCallback;
};

} // namespace aims
