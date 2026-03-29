#pragma once

#include "ProjectModel.h"

#include <juce_gui_basics/juce_gui_basics.h>

#include <vector>

namespace aims
{

class ArrangementOverviewComponent final : public juce::Component
{
public:
    using ProjectGetter = std::function<const ProjectState&()>;
    using ProjectWriter = std::function<void(const ProjectState&, bool undoable, const juce::String&)>;
    using SelectedSectionGetter = std::function<int()>;
    using SectionSelectCallback = std::function<void(int sectionIndex, bool focusEditor)>;
    using ToolGetter = std::function<EditorToolMode()>;
    using ToolModeChangeCallback = std::function<void(EditorToolMode mode)>;

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

    void paint(juce::Graphics& g) override;
    void resized() override;
    void mouseMove(const juce::MouseEvent& event) override;
    void mouseDown(const juce::MouseEvent& event) override;
    void mouseDrag(const juce::MouseEvent& event) override;
    void mouseUp(const juce::MouseEvent& event) override;
    void mouseExit(const juce::MouseEvent& event) override;

private:
    enum class DragMode
    {
        none,
        moveSection,
        createSection,
        resizeLeft,
        resizeRight,
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
    juce::Rectangle<float> transportHandleBounds(int tick) const;
    DragMode hitTestTransportMarker(juce::Point<float> position) const;
    float tickToX(int tick) const;
    int xToSnapStartTick(float x) const;
    int xToSnapEndTick(float x) const;
    int yToTrackIndex(float y) const;
    int displayedBarCount() const;
    int clipLengthTicks(const MidiSection& section, const ProjectState& state) const;
    juce::Rectangle<float> sectionRect(const MidiSection& section, const ProjectState& state) const;
    juce::Rectangle<float> clipRect(const SampleClip& clip) const;
    int hitTestMidiSection(const ProjectState& state, juce::Point<float> position, juce::String& edgeOut) const;
    int hitTestMidiSection(juce::Point<float> position, juce::String& edgeOut) const;
    void updateCursorForPosition(juce::Point<float> position);
    void showSectionContextMenu(int sectionIndex, juce::Point<int> screenPosition, int contextTrackIndex, int contextTick);
    bool copySectionToClipboard(int sectionIndex) const;
    bool pasteClipboardAt(ProjectState& updatedProject, int targetTrackIndex, int targetStartTick, int& outSectionIndex) const;
    bool glueSectionsAtIndex(int sectionIndex);
    bool markSectionForErase(int sectionIndex);
    void rebuildErasePreview();

    ProjectGetter projectGetter;
    ProjectWriter projectWriter;
    SelectedSectionGetter selectedSectionGetter;
    SectionSelectCallback sectionSelectCallback;
    ToolGetter toolGetter;
    ToolModeChangeCallback toolModeChangeCallback;

    ProjectState previewProject;
    ProjectState beforeProject;
    bool previewActive = false;
    bool previewDirty = false;
    DragMode dragMode = DragMode::none;
    int draggedSectionIndex = -1;
    int dragOffsetTick = 0;
    int createStartTick = 0;
    int resizeAnchorStartTick = 0;
    int resizeAnchorLengthTicks = kTicksPerBar;
    std::vector<int> erasedSectionIndices;
    int contextMenuTrackIndex = -1;
    int contextMenuTick = 0;

    float pixelsPerBar = 108.0f;
    float laneHeightPixels = 34.0f;
    int minimumBars = 16;

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR(ArrangementOverviewComponent)
};

} // namespace aims
