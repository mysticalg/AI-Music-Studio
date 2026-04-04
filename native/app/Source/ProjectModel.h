#pragma once

#include <juce_core/juce_core.h>
#include <juce_graphics/juce_graphics.h>
#include <utility>
#include <vector>

namespace aims
{
constexpr int kTicksPerBeat = 480;
constexpr int kTicksPerBar = kTicksPerBeat * 4;
constexpr int kDefaultBpm = 120;
constexpr int kProjectFileVersion = 12;
constexpr int kMinSequenceSnapTicks = kTicksPerBeat / 16;
constexpr int kEditableMidiPitchMin = 12;
constexpr int kEditableMidiPitchMax = 120;
inline constexpr const char* kAutomationTargetVolume = "volume";
inline constexpr const char* kAutomationTargetPan = "pan";
inline constexpr const char* kAutomationTargetVstOutputGainDb = "vsti_output_gain_db";
inline constexpr const char* kAutomationTargetVstParameterPrefix = "vsti_parameter:";

enum class EditorToolMode
{
    pencil,
    selection,
    scissors,
    glue,
    eraser
};

enum class PianoRollInsertMode
{
    singleNote,
    majorTriad,
    minorTriad,
    majorFifth,
    minorFifth,
    majorSeventh,
    minorSeventh,
    dominantSeventh,
    suspended2,
    suspended4,
    resolvedMajor,
    resolvedMinor
};

struct MidiNote
{
    int startTick = 0;
    int durationTick = kTicksPerBeat / 2;
    int pitch = 60;
    int velocity = 100;
    bool selected = false;
};

struct AutomationPoint
{
    int tick = 0;
    double value = 0.0;
};

struct AutomationLane
{
    juce::String target;
    bool enabled = true;
    std::vector<AutomationPoint> points;
};

struct TempoMarker
{
    int tick = 0;
    int bpm = kDefaultBpm;
};

struct MidiPattern
{
    juce::String id;
    juce::String name = "Pattern";
    int lengthTicks = kTicksPerBar;
    std::vector<MidiNote> notes;
    std::vector<AutomationLane> controllerLanes;
};

struct MidiControllerEvent
{
    int tick = 0;
    int controller = 0;
    int value = 0;
    int channel = 1;
};

struct TrackState
{
    juce::String name = "Track";
    juce::String trackType = "instrument";
    std::vector<MidiNote> notes;
    double volume = 0.8;
    double pan = 0.0;
    juce::String instrument = "Piano";
    juce::String instrumentMode = "General MIDI";
    juce::String rackVst;
    juce::StringArray plugins;
    juce::NamedValueSet vstiParameters;
    juce::String vstiStatePath;
    juce::String vstiStateBase64;
    double vstiOutputGainDb = 0.0;
    double vstiWetMix = 100.0;
    juce::Array<juce::var> vstiOutputBusRoutes;
    juce::Array<juce::var> vstiInputBusRoutes;
    juce::StringArray vstFxChain;
    bool vstFxBypassed = false;
    std::vector<bool> vstFxSlotBypassed;
    int midiProgram = 0;
    int midiChannel = 0;
    juce::String synthProfile = "synth";
    juce::String renderedAudioPath;
    bool mute = false;
    bool solo = false;
    bool liveArmed = false;
    juce::String colorHex;
    bool followThemeTrackColour = true;
    int themeColourSlot = -1;
    std::vector<AutomationLane> automationLanes;
    juce::StringArray arrangementVisibleAutomationTargets;
    juce::String routingTarget = "master";
};

struct SharedEffectBusState
{
    juce::String id;
    juce::String name = "FX Bus";
    juce::String effect;
    juce::StringArray outputTargets { "master" };
    juce::NamedValueSet parameters;
    juce::String statePath;
    bool bypassed = false;
};

struct VstInstrument
{
    juce::String name;
    juce::String path;
    juce::String pluginName;
    bool isInstrument = false;
    bool isEffect = false;
    juce::String category;
    bool hostSupported = true;
    juce::String hostError;
};

struct SampleAsset
{
    juce::String path;
    double durationSec = 0.0;
    int sampleRate = 44100;
    std::vector<float> waveformPreview;
};

struct SampleClip
{
    juce::String path;
    int trackIndex = 0;
    double startSec = 0.0;
    double durationSec = 0.0;
    double sourceOffsetSec = 0.0;
    double sourceFileDurationSec = 0.0;
    int sampleRate = 44100;
    std::vector<float> waveformPreview;
};

struct MidiSection
{
    int trackIndex = 0;
    int startTick = 0;
    int lengthTicks = kTicksPerBar;
    double startSec = 0.0;
    double durationSec = 0.0;
    juce::String name = "MIDI Part";
    juce::String patternId;
};

struct ProjectState
{
    int bpm = kDefaultBpm;
    int timeSigNumerator = 4;
    int timeSigDenominator = 4;
    int defaultPatternTicks = kTicksPerBar;
    bool quantizeEnabled = true;
    int quantizeDiv = 8;
    bool quantizeTriplet = false;
    int keyQuantizeRoot = 0;
    juce::String keyQuantizeScale = "chromatic";
    int pianoRollVisiblePitchMin = kEditableMidiPitchMin;
    int pianoRollVisiblePitchMax = kEditableMidiPitchMax;
    int arrangementSnapTicks = kTicksPerBeat;
    bool loopEnabled = true;
    bool metronomeEnabled = false;
    int leftLocatorTick = 0;
    int rightLocatorTick = kTicksPerBar;
    int playheadTick = 0;
    double leftLocatorSec = 0.0;
    double rightLocatorSec = 0.0;
    double playheadSec = 0.0;
    double masterVolume = 1.0;
    juce::StringArray masterFxChain;
    bool masterFxBypassed = false;
    std::vector<bool> masterFxSlotBypassed;
    std::vector<SharedEffectBusState> sharedFxBuses;
    std::vector<TempoMarker> tempoMarkers;
    std::vector<TrackState> tracks;
    juce::StringArray vstiPaths;
    juce::StringArray vstiFolderPaths;
    juce::StringArray samplePaths;
    std::vector<VstInstrument> vstRack;
    std::vector<SampleAsset> sampleAssets;
    std::vector<SampleClip> sampleClips;
    std::vector<MidiPattern> midiPatterns;
    std::vector<MidiSection> midiSections;

    void recalculateTimeFields();
    int getTotalNoteCount() const;
};

struct ProjectFileData
{
    juce::String format = "ai_music_studio_project";
    int version = kProjectFileVersion;
    juce::int64 savedAtUnix = 0;
    ProjectState project;
    juce::var uiState;
};

ProjectFileData makeDefaultProjectFile();
double tickToSeconds(int tick, int bpm);
int secondsToTick(double seconds, int bpm);
int normaliseTimeSignatureNumerator(int numerator);
int normaliseTimeSignatureDenominator(int denominator);
int ticksPerTimeSignatureBeat(int denominator);
int ticksPerTimeSignatureBeat(const ProjectState& project);
int ticksPerBar(int numerator, int denominator);
int ticksPerBar(const ProjectState& project);
juce::String timeSignatureDisplayName(int numerator, int denominator);
juce::String timeSignatureDisplayName(const ProjectState& project);
void sanitiseTempoMarkers(std::vector<TempoMarker>& tempoMarkers, int fallbackBpm = kDefaultBpm);
double tickToSeconds(const ProjectState& project, int tick);
int secondsToTick(const ProjectState& project, double seconds);
int64_t tickToFrame(const ProjectState& project, int tick, double sampleRate);
double frameToTickDouble(const ProjectState& project, int64_t frame, double sampleRate);
int frameToTick(const ProjectState& project, int64_t frame, double sampleRate);
const juce::StringArray& trackColorPalette();
void setTrackColourTheme(const juce::Colour& accent,
                         const juce::Colour& surface,
                         const juce::Colour& outline,
                         const juce::Colour& success = juce::Colours::transparentBlack,
                         const juce::Colour& info = juce::Colours::transparentBlack,
                         const juce::Colour& warning = juce::Colours::transparentBlack);
juce::Colour defaultTrackColour(int index);
juce::Colour trackDisplayColour(const TrackState& track, int index = 0);
juce::Colour trackTextColour(const juce::Colour& colour);
juce::String automationTargetForVstParameter(const juce::String& name);
bool automationTargetIsVstParameter(const juce::String& target);
juce::String automationTargetParameterName(const juce::String& target);
int normaliseSequenceTickLength(int tickLength, int barTicks = kTicksPerBar);
int defaultPatternLengthTicks(const ProjectState& project);
int arrangementSnapTickLength(const ProjectState& project);
juce::String editorToolModeLabel(EditorToolMode mode);
juce::String pianoRollInsertModeLabel(PianoRollInsertMode mode);
juce::StringArray availableKeyQuantizeScales();
juce::String normaliseKeyQuantizeScale(const juce::String& scaleId);
juce::String pitchClassDisplayName(int pitchClass);
juce::String keyQuantizeDisplayName(int rootPitchClass, const juce::String& scaleId);
juce::String midiControllerTarget(int controllerNumber);
bool controllerTargetIsMidiCc(const juce::String& target);
int midiControllerNumberFromTarget(const juce::String& target);
juce::String midiControllerTargetLabel(const juce::String& target);
juce::StringArray defaultMidiControllerTargets();
bool projectUsesKeyQuantize(const ProjectState& project);
bool pitchInKeyQuantizeScale(const ProjectState& project, int pitch);
std::vector<int> visiblePitchesForProjectScale(const ProjectState& project, int minPitch, int maxPitch);
double automationTargetDefaultValue(const TrackState& track, const juce::String& target);
std::pair<double, double> automationTargetBounds(const TrackState& track, const juce::String& target);
juce::String automationTargetLabel(const TrackState& track, const juce::String& target);
juce::StringArray availableAutomationTargets(const TrackState& track);
juce::StringArray usedAutomationTargets(const TrackState& track);
juce::StringArray visibleArrangementAutomationTargets(const TrackState& track);
void sanitiseAutomationLanes(TrackState& track);
void sanitiseArrangementAutomationLaneVisibility(TrackState& track);
void sanitisePatternControllerLanes(MidiPattern& pattern);
int patternLengthTicks(const MidiPattern& pattern);
MidiPattern* findMidiPattern(ProjectState& project, const juce::String& patternId);
const MidiPattern* findMidiPattern(const ProjectState& project, const juce::String& patternId);
std::vector<MidiControllerEvent> collectTrackControllerEvents(const ProjectState& project, int trackIndex);
void synchronisePatternClipsToTrackNotes(ProjectState& project);
int defaultMidiProgramForInstrumentName(const juce::String& instrumentName);
juce::String inferSynthProfile(const juce::String& instrumentName, int midiProgram);
int findRackInstrumentIndexByReference(const ProjectState& project, const juce::String& reference);
juce::String resolveRackPluginPath(const ProjectState& project, const TrackState& track);
juce::String displayRackName(const ProjectState& project, const TrackState& track);
int preferredInstrumentIndexByNames(const ProjectState& project, const juce::StringArray& preferredNames);
int defaultInstrumentIndex(const ProjectState& project);
int suggestedNativeInstrumentIndexForTrack(const ProjectState& project, const TrackState& track);
int nativeInstrumentIndexForTrack(const ProjectState& project, const TrackState& track);
bool materialiseNativeInstrumentTrack(const ProjectState& project, TrackState& track);
juce::var toVar(const TrackState& track);
juce::var toVar(const ProjectState& project);
juce::String fingerprint(const TrackState& track);
juce::String fingerprint(const ProjectState& project);
juce::Result loadProjectFile(const juce::File& file, ProjectFileData& outProject);
juce::Result saveProjectFile(const juce::File& file, const ProjectFileData& projectFile);

} // namespace aims
