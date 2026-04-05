#pragma once

#include "ProjectModel.h"

#include <juce_gui_basics/juce_gui_basics.h>

#include <vector>

namespace aims
{
inline constexpr float kArrangementMinPixelsPerBar = 1.0f;
inline constexpr float kArrangementMaxPixelsPerBar = 640.0f;
inline constexpr int kArrangementMinimumVisibleSeconds = 600;

class ArrangementOverviewComponent final : public juce::Component,
                                           public juce::FileDragAndDropTarget
{
public:
    using ProjectGetter = std::function<const ProjectState&()>;
    using ProjectWriter = std::function<void(const ProjectState&, bool undoable, const juce::String&)>;
    using SelectedSectionGetter = std::function<int()>;
    using SectionSelectCallback = std::function<void(int sectionIndex, bool focusEditor)>;
    using ToolGetter = std::function<EditorToolMode()>;
    using ToolModeChangeCallback = std::function<void(EditorToolMode mode)>;
    using ZoomChangedCallback = std::function<void(float pixelsPerBar)>;
    using LaneHeightChangedCallback = std::function<void(float laneHeightPixels)>;
    using SampleClipStemSeparationCallback = std::function<void(int clipIndex)>;
    using SampleFileDropCallback = std::function<juce::Result(const juce::StringArray& files,
                                                              int targetTrackIndex,
                                                              int targetStartTick)>;

    ArrangementOverviewComponent(ProjectGetter projectGetterIn,
                                 ProjectWriter projectWriterIn,
                                 SelectedSectionGetter selectedSectionGetterIn,
                                 SectionSelectCallback sectionSelectCallbackIn,
                                 ToolGetter toolGetterIn,
                                 ToolModeChangeCallback toolModeChangeCallbackIn);
    ~ArrangementOverviewComponent() override = default;

    void refreshFromModel();
    void setHorizontalZoom(float pixelsPerBarIn);
    float getHorizontalZoom() const noexcept;
    void setLaneHeight(float laneHeightIn);
    float getLaneHeight() const noexcept;
    void setZoomChangedCallback(ZoomChangedCallback zoomChangedCallbackIn);
    void setLaneHeightChangedCallback(LaneHeightChangedCallback laneHeightChangedCallbackIn);
    void setSampleClipStemSeparationCallback(SampleClipStemSeparationCallback callbackIn);
    void setSampleFileDropCallback(SampleFileDropCallback callbackIn);
    bool selectAllSections();
    std::vector<int> getSelectedSectionIndices() const;

    void paint(juce::Graphics& g) override;
    void resized() override;
    void mouseMove(const juce::MouseEvent& event) override;
    void mouseDown(const juce::MouseEvent& event) override;
    void mouseDrag(const juce::MouseEvent& event) override;
    void mouseUp(const juce::MouseEvent& event) override;
    void mouseExit(const juce::MouseEvent& event) override;
    void mouseWheelMove(const juce::MouseEvent& event, const juce::MouseWheelDetails& wheel) override;
    bool keyPressed(const juce::KeyPress& key) override;
    bool isInterestedInFileDrag(const juce::StringArray& files) override;
    void fileDragEnter(const juce::StringArray& files, int x, int y) override;
    void fileDragMove(const juce::StringArray& files, int x, int y) override;
    void fileDragExit(const juce::StringArray& files) override;
    void filesDropped(const juce::StringArray& files, int x, int y) override;

private:
    enum class DragMode
    {
        none,
        moveSection,
        moveSampleClip,
        createSection,
        marqueeSelect,
        resizeLeft,
        resizeRight,
        resizeSampleLeft,
        resizeSampleRight,
        eraseSections,
        moveLeftLocator,
        moveRightLocator,
        movePlayhead
    };

    const ProjectState& project() const;
    const ProjectState& displayedProject() const;
    void updateContentSize();
    float laneHeaderWidth() const;
    float rulerHeight() const;
    float laneHeight() const;
    float trackClipTop(int trackIndex, const ProjectState& state) const;
    float trackTotalHeight(int trackIndex, const ProjectState& state) const;
    bool hitTestTrackLane(float y, int& outTrackIndex, juce::String& outAutomationTarget) const;
    juce::Rectangle<float> transportHandleBounds(int tick) const;
    DragMode hitTestTransportMarker(juce::Point<float> position) const;
    float tickToX(int tick) const;
    int xToSnapStartTick(float x) const;
    int xToSnapEndTick(float x) const;
    int yToTrackIndex(float y) const;
    int displayedBarCount() const;
    int clipLengthTicks(const MidiSection& section, const ProjectState& state) const;
    int clipLengthTicks(const SampleClip& clip, const ProjectState& state) const;
    juce::Rectangle<float> sectionRect(const MidiSection& section, const ProjectState& state) const;
    juce::Rectangle<float> clipRect(const SampleClip& clip) const;
    int hitTestSampleClip(const ProjectState& state, juce::Point<float> position, juce::String& edgeOut) const;
    int hitTestSampleClip(juce::Point<float> position, juce::String& edgeOut) const;
    int hitTestMidiSection(const ProjectState& state, juce::Point<float> position, juce::String& edgeOut) const;
    int hitTestMidiSection(juce::Point<float> position, juce::String& edgeOut) const;
    void updateCursorForPosition(juce::Point<float> position);
    void showSectionContextMenu(int sectionIndex, int sampleClipIndex, juce::Point<int> screenPosition, int contextTrackIndex, int contextTick);
    bool copySectionToClipboard(int sectionIndex) const;
    bool pasteClipboardAt(ProjectState& updatedProject, int targetTrackIndex, int targetStartTick, int& outSectionIndex) const;
    bool splitSectionAtTick(int sectionIndex, int splitTick);
    bool splitSampleClipAtTick(int clipIndex, int splitTick);
    bool glueSectionsAtIndex(int sectionIndex);
    bool duplicateSelectedSections();
    bool markSectionForErase(int sectionIndex);
    bool markSampleClipForErase(int clipIndex);
    void rebuildErasePreview();

    ProjectGetter projectGetter;
    ProjectWriter projectWriter;
    SelectedSectionGetter selectedSectionGetter;
    SectionSelectCallback sectionSelectCallback;
    ToolGetter toolGetter;
    ToolModeChangeCallback toolModeChangeCallback;
    ZoomChangedCallback zoomChangedCallback;
    LaneHeightChangedCallback laneHeightChangedCallback;
    SampleClipStemSeparationCallback sampleClipStemSeparationCallback;
    SampleFileDropCallback sampleFileDropCallback;

    ProjectState previewProject;
    ProjectState beforeProject;
    bool previewActive = false;
    bool previewDirty = false;
    DragMode dragMode = DragMode::none;
    juce::Point<float> marqueeStart;
    juce::Rectangle<float> marqueeRect;
    std::vector<int> selectedSectionIndices;
    int selectedSampleClipIndex = -1;
    std::vector<int> draggedSectionIndices;
    std::vector<int> draggedSectionBaseStartTicks;
    std::vector<int> draggedSectionBaseTrackIndices;
    int draggedSectionIndex = -1;
    int draggedSampleClipIndex = -1;
    SampleClip draggedSampleClipBase;
    int dragOffsetTick = 0;
    int dragAnchorStartTick = 0;
    int dragAnchorTrackIndex = 0;
    int createStartTick = 0;
    int resizeAnchorStartTick = 0;
    int resizeAnchorLengthTicks = kTicksPerBar;
    bool dragCreatesCopy = false;
    std::vector<int> erasedSectionIndices;
    std::vector<int> erasedSampleClipIndices;
    int contextMenuTrackIndex = -1;
    int contextMenuTick = 0;
    bool sampleFileDragActive = false;
    int sampleFileDragTrackIndex = -1;
    int sampleFileDragStartTick = 0;

    float pixelsPerBar = 108.0f;
    float laneHeightPixels = 34.0f;
    int minimumBars = 16;

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR(ArrangementOverviewComponent)
};

} // namespace aims
