#pragma once

#include "ProjectModel.h"

#include <juce_gui_extra/juce_gui_extra.h>
#include <functional>
#include <map>

namespace aims
{

class AutomationEditorComponent final : public juce::Component
{
public:
    using ProjectGetter = std::function<const ProjectState&()>;
    using SelectedTrackIndexGetter = std::function<int()>;
    using TrackWriter = std::function<void(int trackIndex,
                                           const TrackState& track,
                                           bool undoable,
                                           const juce::String& actionName)>;

    AutomationEditorComponent(ProjectGetter projectGetterIn,
                              SelectedTrackIndexGetter selectedTrackIndexGetterIn,
                              TrackWriter trackWriterIn);
    ~AutomationEditorComponent() override;

    void paint(juce::Graphics& g) override;
    void resized() override;

    void refreshFromModel();
    void refreshViewState();

private:
    class LaneListModel;
    class CurveComponent;

    const TrackState* getSelectedTrack() const;
    int getSelectedTrackIndex() const;
    const AutomationLane* getSelectedLane() const;
    juce::String getSelectedTarget() const;
    void setSelectedTarget(const juce::String& target);

    void populateTargetCombo();
    void updateLaneControls();
    void syncCurveFromSelection();
    int computeCurveMaxTick() const;
    void handleLaneSelectionChanged(int row);

    void addSelectedLane();
    void removeSelectedLane();
    void toggleSelectedLaneEnabled();
    void resetSelectedLane();
    void commitCurrentLanePoints(const std::vector<AutomationPoint>& points,
                                 const juce::String& actionName);

    ProjectGetter projectGetter;
    SelectedTrackIndexGetter selectedTrackIndexGetter;
    TrackWriter trackWriter;

    bool updatingUi = false;
    bool hasTrackSnapshot = false;
    TrackState currentTrackSnapshot;
    std::vector<AutomationLane> currentLanes;
    std::map<int, juce::String> selectedTargetByTrack;
    std::vector<juce::String> comboTargets;

    juce::Label headerLabel;
    juce::Label trackSummaryLabel;
    juce::Label laneSummaryLabel;
    juce::Label targetLabel;
    juce::ComboBox addLaneCombo;
    juce::TextButton addLaneButton;
    juce::TextButton removeLaneButton;
    juce::ToggleButton enableLaneToggle;
    juce::TextButton resetLaneButton;
    juce::ListBox laneList;
    std::unique_ptr<LaneListModel> laneListModel;
    std::unique_ptr<CurveComponent> curveComponent;

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR(AutomationEditorComponent)
};

} // namespace aims
