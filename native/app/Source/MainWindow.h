#pragma once

#include "AIClient.h"
#include "AutomationEditorComponent.h"
#include "ArrangementOverviewComponent.h"
#include "MidiFileIO.h"
#include "MixerComponent.h"
#include "NativeVstHostSession.h"
#include "PianoRollComponent.h"
#include "ProjectModel.h"
#include "SampleFileIO.h"
#include "SampleTimelineComponent.h"

#include <juce_gui_extra/juce_gui_extra.h>

#include <future>

namespace aims
{

enum class AIComposeTargetMode
{
    replaceCurrentTrack,
    replaceAllTracks,
    addToCurrentTrack,
    addToAllTracks
};

class FloatingPanelWindow;
class TransportPanelComponent;
class AudioSettingsPanelComponent;
class AudioWorkspaceWindowComponent;
class ActivityLogWindowComponent;
class VstFolderManagerComponent;
class PanelsWindowComponent;
class SampleWorkspaceWindowComponent;
class PianoRollWindowComponent;
class VirtualPianoWindowComponent;
class TracksWorkspaceWindowComponent;
class RackBrowserWindowComponent;
class RenderManagerWindowComponent;
class HeaderLcdDisplay;

class StudioShellComponent final : public juce::Component,
                                   private juce::Timer
{
public:
    StudioShellComponent();
    ~StudioShellComponent() override;

    void paint(juce::Graphics& g) override;
    void resized() override;
    bool keyPressed(const juce::KeyPress& key) override;
    void mouseMove(const juce::MouseEvent& event) override;
    void mouseExit(const juce::MouseEvent& event) override;
    void mouseDown(const juce::MouseEvent& event) override;
    void mouseDrag(const juce::MouseEvent& event) override;
    void mouseUp(const juce::MouseEvent& event) override;

    juce::Result openProjectFile(const juce::File& file);
    void showAboutDialog();

private:
    class TrackTableModel final : public juce::TableListBoxModel
    {
    public:
        explicit TrackTableModel(StudioShellComponent& ownerIn);

        int getNumRows() override;
        void paintRowBackground(juce::Graphics& g, int rowNumber, int width, int height, bool rowIsSelected) override;
        void paintCell(juce::Graphics& g, int rowNumber, int columnId, int width, int height, bool rowIsSelected) override;
        void cellClicked(int rowNumber, int columnId, const juce::MouseEvent& event) override;
        void backgroundClicked(const juce::MouseEvent& event) override;
        void selectedRowsChanged(int lastRowSelected) override;

    private:
        StudioShellComponent& owner;
    };

    class SampleAssetListModel final : public juce::ListBoxModel
    {
    public:
        explicit SampleAssetListModel(StudioShellComponent& ownerIn);

        int getNumRows() override;
        void paintListBoxItem(int rowNumber,
                              juce::Graphics& g,
                              int width,
                              int height,
                              bool rowIsSelected) override;
        void selectedRowsChanged(int lastRowSelected) override;

    private:
        StudioShellComponent& owner;
    };

    struct RackEditorSession
    {
        int trackIndex = -1;
        int lastStateGeneration = -1;
        bool editorOpen = false;
    };

    void createToolbarButton(juce::TextButton& button, const juce::String& text);
    void configureInspectorTextEditor(juce::TextEditor& editor, const juce::String& placeholder);
    void configureInspectorSlider(juce::Slider& slider, double minimum, double maximum, double step, const juce::String& suffix);
    void refreshUi();
    void refreshInspector();
    void updateEditorState();
    void markDirty();
    void clearDirty();

    void createNewProject();
    void promptOpenProject();
    void saveProject();
    void saveProjectAs();
    void jumpPlayheadToStart();
    void setTransportPlayheadTick(int tick);
    void setTransportLeftLocatorTick(int tick);
    void setTransportRightLocatorTick(int tick);
    void setTransportTempo(int bpm);
    void setTransportLoopEnabled(bool enabled);
    void setTransportMetronomeEnabled(bool enabled);
    void addTrack();
    void duplicateSelectedTrack();
    void removeSelectedTrack();
    void handleTempoChanged();
    void handleTimeSignatureChanged();
    void handlePatternBarsChanged();
    void handleKeyQuantizeChanged();
    void handleArrangementSnapChanged();
    void handleArrangementZoomChanged();
    void handleArrangementLaneHeightChanged();
    void handlePianoRollZoomChanged();
    void handlePianoRollRowHeightChanged();
    void promptImportMidi();
    void promptExportMidi();
    void promptExportWav();
    void promptExportSelectedTrackWav();
    void promptExportProjectStems();
    void promptImportSample();
    void promptImportRackPlugin();
    void promptRelinkSelectedTrackRenderedAudio();
    void placeSelectedSampleAtPlayhead();
    void importSelectedTrackRenderToSampleLibrary();
    void placeSelectedTrackRenderAtPlayhead();
    void showAiSettingsDialog();
    void composeWithAi();
    void openSelectedTrackRackEditor();
    void openTrackEffectEditorFromMixer(int trackIndex, int effectIndex);
    void openMasterEffectEditorFromMixer(int effectIndex);
    void saveSelectedTrackRackState();
    void playSelectedTrackThroughRack();
    void playFullProjectThroughNativeEngine();
    void stopRackPreview();
    void undo();
    void redo();
    void selectAllNotesFromMenu();
    void copyNotesFromMenu();
    void cutNotesFromMenu();
    void deleteNotesFromMenu();
    void duplicateNotesFromMenu();
    void pasteNotesFromMenu();
    void quantizeSelectedNotes();
    void setEditorToolMode(EditorToolMode mode);
    void focusArrangementPanel();
    void focusAutomationPanel();
    void focusSamplesPanel();
    void focusTracksPanel();
    void focusRackBrowserPanel();
    void focusRenderManagerPanel();
    void focusAudioPanel();
    void focusMixerPanel();
    void focusPianoRollPanel();
    void focusVirtualPianoPanel();
    void showAudioSettingsWindow();
    void showVstFolderManagerWindow();
    void refreshAudioSettingsFromHost(bool forceCreate);
    juce::Result applyAudioDeviceSettings(const juce::String& audioDeviceType,
                                          const juce::String& audioDeviceName,
                                          int sampleRate,
                                          int bufferSize);
    void setTransportWindowVisible(bool shouldBeVisible);
    bool isTransportWindowVisible() const noexcept;
    void setMixerWindowVisible(bool shouldBeVisible);
    bool isMixerWindowVisible() const noexcept;
    void setAudioWindowVisible(bool shouldBeVisible);
    bool isAudioWindowVisible() const noexcept;
    void setPanelsWindowVisible(bool shouldBeVisible);
    bool isPanelsWindowVisible() const noexcept;
    void setTracksWindowVisible(bool shouldBeVisible);
    bool isTracksWindowVisible() const noexcept;
    void setRackBrowserWindowVisible(bool shouldBeVisible);
    bool isRackBrowserWindowVisible() const noexcept;
    void setRenderManagerWindowVisible(bool shouldBeVisible);
    bool isRenderManagerWindowVisible() const noexcept;
    void setArrangementWindowVisible(bool shouldBeVisible);
    bool isArrangementWindowVisible() const noexcept;
    void setAutomationWindowVisible(bool shouldBeVisible);
    bool isAutomationWindowVisible() const noexcept;
    void setSamplesWindowVisible(bool shouldBeVisible);
    bool isSamplesWindowVisible() const noexcept;
    void setPianoRollWindowVisible(bool shouldBeVisible);
    bool isPianoRollWindowVisible() const noexcept;
    void setVirtualPianoWindowVisible(bool shouldBeVisible);
    bool isVirtualPianoWindowVisible() const noexcept;
    void setActivityLogWindowVisible(bool shouldBeVisible);
    bool isActivityLogWindowVisible() const noexcept;

    void replaceTrackStateNoUndo(int trackIndex, const TrackState& updatedTrack);
    void applyTrackStateSyncNoUndo(int trackIndex,
                                   const TrackState& updatedTrack,
                                   bool deferParameterEnginePush = false,
                                   bool skipParameterEnginePush = false);
    void applyTrackStateEdit(int trackIndex, const TrackState& updatedTrack, const juce::String& actionName);
    void applyProjectStateEdit(const ProjectState& updatedProject, const juce::String& actionName);
    void applySelectedTrackMutation(std::function<void(TrackState&)> mutation, const juce::String& actionName);
    void setAiComposeBusy(bool busy, const juce::String& detail = {});
    void pollAiComposeFuture();
    void updateAiStatusSummary();
    void applyAiComposeResult(const AIComposeResult& result, int requestedBars);
    void syncBundledRackCatalogInProject();
    void refreshRackCatalog();
    juce::String defaultVstFolderPath() const;
    juce::StringArray userManagedVstFolderPaths() const;
    void promptAddUserVstFolder();
    void removeUserVstFolder(const juce::String& folderPath);
    void assignSelectedTrackRackByReference(const juce::String& reference);
    void clearSelectedTrackRackAssignment();
    void materialiseSelectedTrackRackAssignment();
    void clearSelectedTrackRenderedAudioPath();
    void refreshPollingTimerState();
    void scheduleSelectedTrackRackPreviewWarmup(int delayMs = 180);
    void setupFloatingWindows();
    void refreshFloatingWindows(bool includeEditorRefresh = true);
    void refreshProjectSummaryLabels();
    void applyEditorViewScaleState();
    void refreshPlaybackToggleButton();
    void repaintTrackVolumeMeters();
    void restorePersistedSessionState();
    void persistSessionState() const;
    void restorePersistedWindowVisibility();
    void persistWindowVisibilityState() const;
    void restorePersistedThemeSelection();
    void restorePersistedFontSelection();
    juce::String buildAboutDetails() const;
    void persistThemeSelection() const;
    void persistFontSelection() const;
    void applyTheme();
    void applyUiFont();
    void appendActivityLog(const juce::String& title, const juce::String& body);
    void appendActivityLogToFile(const juce::String& entry) const;
    juce::String activityLogTextSnapshot() const;
    void clearActivityLog();
    void showTrackContextMenu(int rowNumber, juce::Point<int> screenPosition);
    RackEditorSession* findRackEditorSession(int trackIndex);
    const RackEditorSession* findRackEditorSession(int trackIndex) const;
    bool hasOpenRackEditorSessions() const;
    bool isTrackBeingLiveEdited(int trackIndex) const;
    juce::Result ensureRackEditorSessionReadyForTrack(int trackIndex,
                                                      const TrackState& track,
                                                      RackEditorSession*& outSession);
    void syncOpenRackEditorSessions(bool duringPlayback);
    void closeRackEditorSession(int trackIndex);
    void closeAllRackEditorSessions();
    juce::Result ensureNativeAudioEnginePrepared(bool preserveTransport = false);
    juce::Result previewSelectedTrackMidiNoteOn(int pitch, int velocity = 100);
    void previewSelectedTrackMidiNoteOff(int pitch, int velocity = 0);
    void stopSelectedTrackMidiPreview();
    void insertVirtualKeyboardNote(int pitch, bool fromShortcut = false);
    bool tryHandleVirtualPianoShortcut(const juce::KeyPress& key);
    bool virtualPianoShortcutsEnabled() const;
    juce::Result ensureSampleAssetForFile(const juce::File& file, int& outAssetIndex);
    juce::Result placeSampleAssetAtPlayhead(int assetIndex,
                                            const juce::String& actionName,
                                            juce::String& outTrackName);
    void resetRackHostTracking();
    bool syncTrackRackParametersFromValues(int trackIndex,
                                           const juce::NamedValueSet& hostParameters,
                                           bool skipParameterEnginePush = false);
    void timerCallback() override;
    void setTrackStateInternal(int trackIndex,
                               const TrackState& updatedTrack,
                               bool deferParameterEnginePush = false,
                               bool skipParameterEnginePush = false);
    void setProjectStateInternal(const ProjectState& updatedProject);

    juce::File suggestProjectFile() const;
    juce::File suggestTrackStateFile(int trackIndex, const TrackState& track) const;
    juce::File ensureProjectSuffix(const juce::File& file) const;
    int findPreferredSampleTrackIndex() const;
    int getSelectedTrackIndex() const;
    void setSelectedTrackIndex(int row);
    int getSelectedMidiSectionIndex() const noexcept;
    void setSelectedMidiSectionIndex(int sectionIndex, bool bringTrackIntoFocus = true);
    void ensureSelectedMidiSectionForTrack(int trackIndex);
    void focusMidiSectionInPianoRoll(int sectionIndex);
    int getSelectedSampleAssetIndex() const;
    void setSelectedSampleAssetIndex(int row);
    int getCurrentThemeIndex() const noexcept;
    int getThemeCount() const noexcept;
    juce::String getThemeName(int index) const;
    void setThemeIndex(int index);
    int getCurrentFontIndex() const noexcept;
    int getFontCount() const noexcept;
    juce::String getFontName(int index) const;
    void setFontIndex(int index);
    TrackState* getSelectedTrack();
    const TrackState* getSelectedTrack() const;

    ProjectFileData documentState;
    juce::File currentProjectFile;
    bool dirty = false;
    juce::UndoManager undoManager;

    TrackTableModel tableModel;
    juce::TableListBox trackTable;
    SampleAssetListModel sampleAssetListModel;
    juce::ListBox sampleAssetList;

    juce::Image headerLogoImage;
    juce::ImageComponent headerLogo;
    juce::Label headerLabel;
    juce::Label fileLabel;
    std::unique_ptr<HeaderLcdDisplay> headerTimecodeDisplay;
    juce::Label statsLabel;
    juce::Label statusLabel;
    juce::Label aiStatusSummaryLabel;
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
    juce::TextEditor inspectorEditor;
    juce::Label mixerLabel;
    juce::Viewport mixerViewport;
    std::unique_ptr<MixerComponent> mixerComponent;
    juce::Label samplesLabel;
    juce::Label sampleLibraryLabel;
    juce::Label sampleTimelineLabel;
    juce::Viewport sampleTimelineViewport;
    std::unique_ptr<SampleTimelineComponent> sampleTimeline;
    juce::Label automationLabel;
    std::unique_ptr<AutomationEditorComponent> automationEditor;
    juce::Label pianoRollLabel;
    juce::Label arrangementLabel;
    juce::Viewport arrangementViewport;
    std::unique_ptr<ArrangementOverviewComponent> arrangementOverview;
    juce::Viewport pianoRollViewport;
    std::unique_ptr<PianoRollComponent> pianoRoll;

    juce::TextButton newButton;
    juce::TextButton openButton;
    juce::TextButton saveButton;
    juce::TextButton saveAsButton;
    juce::TextButton importMidiButton;
    juce::TextButton exportWavButton;
    juce::TextButton exportMidiButton;
    juce::TextButton addTrackButton;
    juce::TextButton duplicateTrackButton;
    juce::TextButton removeTrackButton;
    juce::TextButton importSampleButton;
    juce::TextButton placeSampleButton;
    juce::TextButton aiSettingsButton;
    juce::TextButton aiComposeButton;
    juce::TextButton openRackEditorButton;
    juce::TextButton saveRackStateButton;
    juce::TextButton playProjectButton;
    juce::TextButton undoButton;
    juce::TextButton redoButton;
    juce::Slider tempoSlider;
    juce::Label tempoLabel;
    juce::Label timeSignatureLabel;
    juce::ComboBox timeSignatureNumeratorBox;
    juce::Label timeSignatureSlashLabel;
    juce::ComboBox timeSignatureDenominatorBox;
    juce::Label patternBarsLabel;
    juce::ComboBox patternBarsBox;
    juce::Label keyQuantizeLabel;
    juce::ComboBox keyQuantizeBox;
    juce::Label arrangementSnapLabel;
    juce::ComboBox arrangementSnapBox;
    juce::Label arrangementZoomLabel;
    juce::Slider arrangementZoomSlider;
    juce::Label arrangementLaneHeightLabel;
    juce::Slider arrangementLaneHeightSlider;
    juce::Label pianoRollZoomLabel;
    juce::Slider pianoRollZoomSlider;
    juce::Label pianoRollRowHeightLabel;
    juce::Slider pianoRollRowHeightSlider;
    juce::TextButton loopToggle;
    juce::TextButton metronomeToggle;
    std::unique_ptr<juce::FileChooser> activeFileChooser;
    AIClient aiClient;
    NativeVstHostSession nativeVstHost;
    std::vector<float> trackMeterLevels;
    double transportCpuUsagePercent = 0.0;
    float transportMasterPeakLeft = 0.0f;
    float transportMasterPeakRight = 0.0f;
    std::future<AIComposeResult> aiComposeFuture;
    juce::String aiComposeBusyDetail;
    juce::String aiComposeDefaultPrompt;
    int aiComposeDefaultBars = 8;
    int aiComposeRequestedBars = 8;
    AIComposeTargetMode aiComposeDefaultMode = AIComposeTargetMode::replaceAllTracks;
    AIComposeTargetMode aiComposeRequestedMode = AIComposeTargetMode::replaceAllTracks;
    juce::String aiComposeDefaultStyle = "Balanced";
    juce::String aiComposeDefaultEnergy = "Medium";
    juce::String aiComposeDefaultDensity = "Balanced";
    juce::String aiComposeDefaultVariation = "Moderate";
    juce::String aiComposeDefaultRegister = "Natural";
    std::vector<int> aiComposeRequestedTargetTracks;
    int aiComposeRequestedInsertTick = 0;
    bool aiComposeBusy = false;
    std::vector<std::unique_ptr<RackEditorSession>> rackEditorSessions;
    bool loadedRackEditorOpen = false;
    bool rackPreviewWarmupPending = false;
    bool rackPreviewRunning = false;
    bool projectPreviewRunning = false;
    bool syncingInspectorControls = false;
    EditorToolMode editorToolMode = EditorToolMode::pencil;
    int selectedMidiSectionIndex = -1;
    bool audioEngineStateDirty = true;
    bool audioEngineStateValid = false;
    int pendingLiveRackParameterEngineSyncTrack = -1;
    float arrangementZoomPixelsPerBar = 108.0f;
    float arrangementLaneHeightPixels = 34.0f;
    float pianoRollZoomPixelsPerBeat = 28.0f;
    float pianoRollRowHeightPixels = 16.0f;
    int currentThemeIndex = 0;
    int currentFontIndex = 0;
    juce::StringArray availableUiFonts;
    std::unique_ptr<juce::LookAndFeel_V4> compactHeaderLookAndFeel;
    juce::uint32 lastSharedRackParameterSyncMs = 0;
    juce::uint32 lastRackEditorSessionSyncMs = 0;
    juce::uint32 lastDeferredParameterFlushMs = 0;
    int pollingTimerHz = 0;
    int playbackUiTickCounter = 0;
    int leftPaneWidthPixels = 0;
    int leftPaneDragStartWidth = 0;
    bool leftPaneResizeDragging = false;
    juce::Rectangle<int> leftPaneResizeHandleBounds;
    std::unique_ptr<juce::PropertiesFile> windowStateSettings;
    mutable juce::CriticalSection activityLogLock;
    juce::StringArray activityLogEntries;
    juce::File activityLogFile;
    bool transportWindowVisible = true;
    bool mixerWindowVisible = true;
    bool audioWindowVisible = true;
    bool panelsWindowVisible = false;
    bool tracksWindowVisible = false;
    bool rackBrowserWindowVisible = false;
    bool renderManagerWindowVisible = false;
    bool arrangementWindowVisible = true;
    bool automationWindowVisible = false;
    bool samplesWindowVisible = false;
    bool pianoRollWindowVisible = false;
    bool virtualPianoWindowVisible = false;
    bool activityLogWindowVisible = false;
    std::unique_ptr<FloatingPanelWindow> transportWindow;
    TransportPanelComponent* transportPanel = nullptr;
    std::unique_ptr<FloatingPanelWindow> mixerWindow;
    MixerComponent* floatingMixerComponent = nullptr;
    std::unique_ptr<FloatingPanelWindow> audioWindow;
    AudioWorkspaceWindowComponent* floatingAudioWorkspace = nullptr;
    std::unique_ptr<FloatingPanelWindow> panelsWindow;
    PanelsWindowComponent* panelsWindowContent = nullptr;
    std::unique_ptr<FloatingPanelWindow> tracksWindow;
    TracksWorkspaceWindowComponent* floatingTracksWorkspace = nullptr;
    std::unique_ptr<FloatingPanelWindow> rackBrowserWindow;
    RackBrowserWindowComponent* floatingRackBrowser = nullptr;
    std::unique_ptr<FloatingPanelWindow> renderManagerWindow;
    RenderManagerWindowComponent* floatingRenderManager = nullptr;
    std::unique_ptr<FloatingPanelWindow> arrangementWindow;
    ArrangementOverviewComponent* floatingArrangementOverview = nullptr;
    std::unique_ptr<FloatingPanelWindow> automationWindow;
    AutomationEditorComponent* floatingAutomationEditor = nullptr;
    std::unique_ptr<FloatingPanelWindow> samplesWindow;
    SampleWorkspaceWindowComponent* floatingSampleWorkspace = nullptr;
    std::unique_ptr<FloatingPanelWindow> pianoRollWindow;
    PianoRollWindowComponent* floatingPianoRollWorkspace = nullptr;
    std::unique_ptr<FloatingPanelWindow> virtualPianoWindow;
    VirtualPianoWindowComponent* virtualPianoWindowContent = nullptr;
    std::unique_ptr<FloatingPanelWindow> activityLogWindow;
    ActivityLogWindowComponent* activityLogWindowContent = nullptr;
    std::unique_ptr<FloatingPanelWindow> audioSettingsWindow;
    AudioSettingsPanelComponent* audioSettingsPanel = nullptr;
    std::unique_ptr<FloatingPanelWindow> vstFolderManagerWindow;
    VstFolderManagerComponent* vstFolderManagerPanel = nullptr;

    friend class MainWindow;
    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR(StudioShellComponent)
};

class MainWindow final : public juce::DocumentWindow,
                         private juce::MenuBarModel
{
public:
    explicit MainWindow(const juce::File& startupProject = {});
    ~MainWindow() override;
    void toggleBorderlessFullscreen();
    bool isBorderlessFullscreenActive() const;

private:
    enum MenuItemId
    {
        menuFileNew = 1,
        menuFileOpen,
        menuFileSave,
        menuFileSaveAs,
        menuFileImportMidi,
        menuFileImportSample,
        menuFileExportWav,
        menuFileExportTrackWav,
        menuFileExportStems,
        menuFileExportMidi,
        menuEditUndo,
        menuEditRedo,
        menuEditQuantize,
        menuEditSelectAll,
        menuEditCopy,
        menuEditCut,
        menuEditDelete,
        menuEditDuplicate,
        menuEditPaste,
        menuSettingsAudio,
        menuSettingsVstFolders,
        menuSettingsAi,
        menuWindowsPanels = 2100,
        menuWindowsTransport,
        menuWindowsMixer,
        menuWindowsAudio,
        menuWindowsTracks,
        menuWindowsRackBrowser,
        menuWindowsRenderManager,
        menuWindowsArrangement,
        menuWindowsAutomation,
        menuWindowsSamples,
        menuWindowsPianoRoll,
        menuWindowsVirtualPiano,
        menuWindowsActivityLog,
        menuWindowsFullscreen,
        menuHelpSite = 3000,
        menuHelpAbout,
        menuThemeBase = 4000,
        menuFontChoiceBase = 5000
    };

    juce::StringArray getMenuBarNames() override;
    juce::PopupMenu getMenuForIndex(int topLevelMenuIndex, const juce::String& menuName) override;
    void menuItemSelected(int menuItemID, int topLevelMenuIndex) override;
    void closeButtonPressed() override;
    bool keyPressed(const juce::KeyPress& key) override;

    StudioShellComponent* shell = nullptr;

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR(MainWindow)
};

} // namespace aims
