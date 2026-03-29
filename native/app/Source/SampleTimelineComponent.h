#pragma once

#include "ProjectModel.h"

#include <juce_gui_basics/juce_gui_basics.h>

namespace aims
{

class SampleTimelineComponent final : public juce::Component
{
public:
    using ProjectGetter = std::function<const ProjectState&()>;
    using ProjectWriter = std::function<void(const ProjectState&, bool undoable, const juce::String&)>;

    SampleTimelineComponent(ProjectGetter projectGetterIn,
                            ProjectWriter projectWriterIn);
    ~SampleTimelineComponent() override = default;

    void refreshFromModel();

    void paint(juce::Graphics& g) override;
    void resized() override;
    void mouseDown(const juce::MouseEvent& event) override;
    void mouseDrag(const juce::MouseEvent& event) override;
    void mouseUp(const juce::MouseEvent& event) override;
    void mouseWheelMove(const juce::MouseEvent& event, const juce::MouseWheelDetails& wheel) override;
    bool keyPressed(const juce::KeyPress& key) override;

private:
    const ProjectState& project() const;
    std::vector<int> sampleTrackIndices(const ProjectState& state) const;
    int sampleTrackForLane(int laneIndex, const ProjectState& state) const;
    int laneForTrackIndex(int trackIndex, const ProjectState& state) const;
    void updateContentSize();
    float laneHeaderWidth() const;
    float rulerHeight() const;
    float laneHeight() const;
    float secondsToX(double seconds) const;
    double xToSeconds(float x) const;
    int yToLaneIndex(float y) const;
    juce::Rectangle<float> clipRect(const SampleClip& clip, const ProjectState& state) const;
    void clearSelectionIfInvalid();

    ProjectGetter projectGetter;
    ProjectWriter projectWriter;

    ProjectState previewProject;
    bool previewActive = false;
    bool previewDirty = false;
    int selectedClipIndex = -1;
    int draggedClipIndex = -1;
    double dragOffsetSeconds = 0.0;

    float pixelsPerSecond = 80.0f;
    int minimumSeconds = 16;

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR(SampleTimelineComponent)
};

} // namespace aims
