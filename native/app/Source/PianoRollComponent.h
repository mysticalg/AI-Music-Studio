#pragma once

#include "ProjectModel.h"

#include <juce_gui_basics/juce_gui_basics.h>

namespace aims
{

class PianoRollComponent final : public juce::Component
{
public:
    enum class SurfaceMode
    {
        full,
        notesOnly,
        controllerOnly
    };

    enum class PencilDrawMode
    {
        step,
        brush,
        line,
        box,
        sine,
        square,
        saw,
        triangle,
        circle,
        noise
    };

    using ProjectGetter = std::function<const ProjectState&()>;
    using TrackIndexGetter = std::function<int()>;
    using SelectedSectionIndexGetter = std::function<int()>;
    using ProjectWriter = std::function<void(const ProjectState&, bool undoable, const juce::String&)>;
    using NotePreviewCallback = std::function<void(int pitch, int velocity)>;
    using PreviewStopCallback = std::function<void()>;
    using ToolModeChangeCallback = std::function<void(EditorToolMode mode)>;
    using KeyHandlerCallback = std::function<bool(const juce::KeyPress&)>;
    using ZoomChangedCallback = std::function<void(float pixelsPerBeat)>;
    using RowHeightChangedCallback = std::function<void(float rowHeightPixels)>;

    PianoRollComponent(ProjectGetter projectGetterIn,
                       TrackIndexGetter trackIndexGetterIn,
                       SelectedSectionIndexGetter selectedSectionIndexGetterIn,
                       ProjectWriter projectWriterIn);
    ~PianoRollComponent() override;

    void refreshFromModel();
    void setToolMode(EditorToolMode mode);
    void setSurfaceMode(SurfaceMode mode);
    void setHorizontalZoom(float pixelsPerBeat);
    float getHorizontalZoom() const noexcept;
    void setNoteRowHeight(float height);
    float getNoteRowHeight() const noexcept;
    void setNotePreviewCallbacks(NotePreviewCallback noteOnCallbackIn,
                                 NotePreviewCallback noteOffCallbackIn,
                                 PreviewStopCallback stopPreviewCallbackIn = {});
    void setToolModeChangeCallback(ToolModeChangeCallback toolModeChangeCallbackIn);
    void setKeyHandlerCallback(KeyHandlerCallback keyHandlerCallbackIn);
    void setZoomChangedCallback(ZoomChangedCallback zoomChangedCallbackIn);
    void setRowHeightChangedCallback(RowHeightChangedCallback rowHeightChangedCallbackIn);
    int viewPositionYForPitch(int pitch, int viewportHeight) const;

    bool copySelected() const;
    bool cutSelected();
    bool deleteSelected();
    bool pasteClipboard();
    bool quantizeSelected();
    bool duplicateSelectedByGrid();
    bool selectAllNotes();

    void paint(juce::Graphics& g) override;
    void resized() override;
    void mouseMove(const juce::MouseEvent& event) override;
    void mouseDown(const juce::MouseEvent& event) override;
    void mouseDrag(const juce::MouseEvent& event) override;
    void mouseUp(const juce::MouseEvent& event) override;
    void mouseExit(const juce::MouseEvent& event) override;
    void mouseWheelMove(const juce::MouseEvent& event, const juce::MouseWheelDetails& wheel) override;
    bool keyPressed(const juce::KeyPress& key) override;

private:
    enum class Interaction
    {
        none,
        drag,
        resizeLeft,
        resizeRight,
        create,
        marqueeSelect,
        drawShape,
        erase,
        moveLeftLocator,
        moveRightLocator,
        movePlayhead
    };

    struct SelectedSnapshot
    {
        int index = -1;
        int startTick = 0;
        int pitch = 60;
        int durationTick = kTicksPerBeat / 2;
    };

    const ProjectState& project() const;
    const ProjectState& displayedProject() const;
    int currentTrackIndex() const;
    int currentSectionIndex() const;
    const MidiSection* currentSection() const;
    const MidiPattern* currentPattern() const;
    const TrackState* currentTrack() const;
    const MidiPattern& drawingPattern() const;
    MidiPattern* patternForSection(ProjectState& state, int sectionIndex) const;
    const MidiPattern* patternForSection(const ProjectState& state, int sectionIndex) const;

    void updateContentSize();
    int contentTickLength() const;
    bool showsNoteEditor() const noexcept;
    bool showsControllerEditor() const noexcept;
    float noteRowsHeight() const;
    float pitchLaneWidth() const;
    float rulerHeight() const;
    float controllerHeaderHeight() const;
    float controllerLaneHeight() const;
    float tickToX(int tick) const;
    float durationToWidth(int durationTick) const;
    std::vector<int> visiblePitches() const;
    int pitchToRow(int pitch) const;
    int nearestVisibleRowForPitch(int pitch) const;
    int rowToPitch(int row) const;
    int pitchRowForY(float y) const;
    int pitchForY(float y) const;
    int transposePitchByVisibleRows(int pitch, int deltaRows) const;
    int gridTick() const;
    int noteLengthTick() const;
    float gridWidthPixels() const;
    juce::Rectangle<float> transportHandleBounds(int absoluteTick) const;
    Interaction hitTestTransportMarker(juce::Point<float> position) const;
    int xToGridStartTick(float x) const;
    int xToGridEndTick(float x) const;
    juce::Rectangle<float> noteRect(const MidiNote& note) const;
    juce::Rectangle<float> controllerBarRect(const MidiNote& note) const;
    juce::Rectangle<float> noteGridBounds() const;
    juce::Rectangle<float> controllerHeaderBounds() const;
    juce::Rectangle<float> controllerLaneBounds() const;
    bool isControllerLanePosition(juce::Point<float> position) const;
    void refreshControllerTargetChoices();
    juce::String currentControllerTarget() const;
    bool controllerTargetEditsVelocity() const;
    bool controllerTargetEditsPatternLane() const;
    int controllerStoredTickFromLocalTick(int localTick) const;
    int controllerLocalTickFromStoredTick(int storedTick) const;
    std::pair<double, double> controllerTargetBounds() const;
    double controllerDefaultValue() const;
    double controllerValueForY(float y) const;
    float yForControllerValue(double value) const;
    const AutomationLane* currentControllerLane() const;
    AutomationLane* editableControllerLane(ProjectState& state, bool createIfMissing);
    std::vector<AutomationPoint> displayedControllerPoints() const;
    void rebuildControllerPreview(juce::Point<float> currentPosition, bool adjustFrequency);
    juce::String controllerTargetDisplayText(const juce::String& target) const;
    void showControllerTargetMenu();

    void clearSelection(MidiPattern& pattern) const;
    void selectSingleNote(MidiPattern& pattern, int noteIndex) const;
    bool hasSelectedNotes(const MidiPattern& pattern) const;
    bool eraseNotesAtPosition(MidiPattern& pattern, juce::Point<float> position) const;
    bool eraseNotesAlongPath(MidiPattern& pattern,
                             juce::Point<float> startPosition,
                             juce::Point<float> endPosition) const;
    bool splitClickedNoteAtTick(int noteIndex, int splitTick);
    bool glueSelectedOrClickedNotes(int noteIndex);
    std::vector<int> insertIntervals() const;
    void insertNotesForCurrentMode(MidiPattern& pattern,
                                   int startTick,
                                   int basePitch,
                                   int durationTick,
                                   int velocity,
                                   bool select,
                                   std::vector<int>* insertedIndices = nullptr) const;
    void rebuildShapePreview(juce::Point<float> currentPosition, bool adjustFrequency);
    int rowForProgress(float progress, int targetRow) const;
    double valueForProgress(float progress, double targetValue) const;

    int hitTestNote(juce::Point<float> position, juce::String& edgeOut) const;
    void updateCursorForPosition(juce::Point<float> position);
    void showContextMenu(juce::Point<int> screenPosition);

    void applyPreviewNoUndo();
    void commitPreviewUndoable(const juce::String& actionName);
    void startPreviewNote(int pitch, int velocity = 100);
    void updatePreviewNotePitch(int pitch, int velocity = 100);
    void stopPreviewNote();

    ProjectGetter projectGetter;
    TrackIndexGetter trackIndexGetter;
    SelectedSectionIndexGetter selectedSectionIndexGetter;
    ProjectWriter projectWriter;
    NotePreviewCallback notePreviewOn;
    NotePreviewCallback notePreviewOff;
    PreviewStopCallback stopPreviewCallback;
    ToolModeChangeCallback toolModeChangeCallback;
    KeyHandlerCallback keyHandlerCallback;
    ZoomChangedCallback zoomChangedCallback;
    RowHeightChangedCallback rowHeightChangedCallback;

    ProjectState previewProject;
    ProjectState beforeProject;
    bool previewActive = false;
    bool previewDirty = false;
    bool pitchLanePreviewActive = false;
    int activeNoteIndex = -1;
    int activePreviewPitch = -1;
    int activePreviewVelocity = 0;
    int anchorGridTick = 0;
    int anchorPitchRow = 0;
    int anchorNoteStartTick = 0;
    int anchorNoteEndTick = 0;
    int shapeAmplitudeRows = 6;
    int shapeFrequencyCycles = 1;
    double anchorControllerValue = 100.0;
    juce::Point<float> marqueeStart;
    juce::Rectangle<float> marqueeRect;
    std::vector<juce::Point<float>> drawPathPoints;
    std::vector<SelectedSnapshot> selectedSnapshots;
    Interaction interaction = Interaction::none;
    bool controllerLaneInteraction = false;

    float cellWidth = 28.0f;
    float cellHeight = 16.0f;
    int minimumBeats = 32;
    SurfaceMode surfaceMode = SurfaceMode::full;
    EditorToolMode toolMode = EditorToolMode::pencil;
    PianoRollInsertMode insertMode = PianoRollInsertMode::singleNote;
    PencilDrawMode pencilDrawMode = PencilDrawMode::step;
    juce::Label controllerTargetLabel;
    juce::TextButton controllerTargetButton;
    juce::String selectedControllerTarget = "velocity";

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR(PianoRollComponent)
};

} // namespace aims
