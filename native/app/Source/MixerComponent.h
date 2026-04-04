#pragma once

#include "ProjectModel.h"

#include <juce_gui_basics/juce_gui_basics.h>

#include <functional>
#include <utility>

namespace aims
{

class MixerComponent final : public juce::Component
{
public:
    using ProjectGetter = std::function<const ProjectState&()>;
    using TrackWriter = std::function<void(int, const TrackState&, bool, const juce::String&)>;
    using ProjectWriter = std::function<void(const ProjectState&, bool, const juce::String&)>;
    using MeterGetter = std::function<float(int)>;
    using MasterMeterGetter = std::function<std::pair<float, float>()>;
    using TrackEffectEditorOpener = std::function<void(int, int)>;
    using MasterEffectEditorOpener = std::function<void(int)>;

    MixerComponent(ProjectGetter projectGetterIn,
                   TrackWriter trackWriterIn,
                   ProjectWriter projectWriterIn,
                   MeterGetter meterGetterIn,
                   MasterMeterGetter masterMeterGetterIn,
                   TrackEffectEditorOpener trackEffectEditorOpenerIn = {},
                   MasterEffectEditorOpener masterEffectEditorOpenerIn = {});
    ~MixerComponent() override = default;

    void refreshFromModel();
    void refreshMeters();
    void paint(juce::Graphics& g) override;
    void resized() override;

private:
    class MeterComponent final : public juce::Component,
                                 public juce::SettableTooltipClient
    {
    public:
        void setLevel(float newLevel);
        float getLevel() const noexcept { return level; }
        void paint(juce::Graphics& g) override;

    private:
        float level = 0.0f;
    };

    class MeterScaleComponent final : public juce::Component,
                                      public juce::SettableTooltipClient
    {
    public:
        void paint(juce::Graphics& g) override;
    };

    class ChannelStrip final : public juce::Component
    {
    public:
        ChannelStrip(int trackIndexIn,
                     ProjectGetter projectGetterIn,
                     TrackWriter trackWriterIn,
                     MeterGetter meterGetterIn,
                     TrackEffectEditorOpener trackEffectEditorOpenerIn);

        void applyTrack(const TrackState& track);
        void refreshMeter();
        void paint(juce::Graphics& g) override;
        void resized() override;
        void mouseWheelMove(const juce::MouseEvent& event, const juce::MouseWheelDetails& wheel) override;

    private:
        struct FxRowWidgets
        {
            int effectIndex = -1;
            std::unique_ptr<juce::Label> nameLabel;
            std::unique_ptr<juce::TextButton> viewButton;
            std::unique_ptr<juce::TextButton> bypassButton;
            std::unique_ptr<juce::TextButton> removeButton;
            juce::Rectangle<int> rowBounds;
        };

        void showFxMenu();
        void showAddFxMenu(juce::Component* target);
        void syncFxRows();
        void openEffectEditor(int effectIndex);
        void removeEffectAt(int effectIndex);
        int findFxRowIndexForComponent(const juce::Component* component) const;
        int findFxInsertIndexForY(int y) const;
        int fxInsertionLineY(int insertIndex) const;
        void finishFxDrag(bool commitChanges);
        void moveEffectToInsertIndex(int sourceIndex, int insertIndex);
        void updateMeterTooltip();
        void clampFxScrollOffset();
        void updateButtonColours(const juce::Colour& accentColour);
        void commitTrackEdit(const juce::String& actionName);
        void mouseDown(const juce::MouseEvent& event) override;
        void mouseDrag(const juce::MouseEvent& event) override;
        void mouseUp(const juce::MouseEvent& event) override;

        const int trackIndex;
        ProjectGetter projectGetter;
        TrackWriter trackWriter;
        MeterGetter meterGetter;
        TrackEffectEditorOpener trackEffectEditorOpener;
        TrackState currentTrack;
        bool syncing = false;
        bool volumeDragging = false;
        bool panDragging = false;

        juce::Label nameLabel;
        juce::Label fxSummaryLabel;
        juce::Label meterLabel;
        juce::TextButton fxButton;
        juce::TextButton fxAddButton;
        juce::TextButton fxBypassButton;
        juce::TextButton fxUpButton;
        juce::TextButton fxDownButton;
        MeterScaleComponent meterScale;
        MeterComponent meter;
        juce::Slider volumeSlider;
        juce::Label volumeLabel;
        juce::Slider panSlider;
        juce::TextButton muteButton;
        juce::TextButton soloButton;
        std::vector<std::unique_ptr<FxRowWidgets>> fxRows;
        float currentMeterLevel = 0.0f;
        int fxScrollOffset = 0;
        int fxVisibleRowCount = 0;
        juce::Rectangle<int> fxListBounds;
        bool fxDragActive = false;
        int fxDragSourceIndex = -1;
        int fxDropInsertIndex = -1;

        JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR(ChannelStrip)
    };

    class MasterStrip final : public juce::Component
    {
    public:
        MasterStrip(ProjectGetter projectGetterIn,
                    ProjectWriter projectWriterIn,
                    MasterMeterGetter masterMeterGetterIn,
                    MasterEffectEditorOpener masterEffectEditorOpenerIn);

        void applyProject(const ProjectState& project);
        void refreshMeter();
        void paint(juce::Graphics& g) override;
        void resized() override;
        void mouseWheelMove(const juce::MouseEvent& event, const juce::MouseWheelDetails& wheel) override;

    private:
        struct FxRowWidgets
        {
            int effectIndex = -1;
            std::unique_ptr<juce::Label> nameLabel;
            std::unique_ptr<juce::TextButton> viewButton;
            std::unique_ptr<juce::TextButton> bypassButton;
            std::unique_ptr<juce::TextButton> removeButton;
            juce::Rectangle<int> rowBounds;
        };

        void showFxMenu();
        void showAddFxMenu(juce::Component* target);
        void syncFxRows();
        void openEffectEditor(int effectIndex);
        void removeEffectAt(int effectIndex);
        int findFxRowIndexForComponent(const juce::Component* component) const;
        int findFxInsertIndexForY(int y) const;
        int fxInsertionLineY(int insertIndex) const;
        void finishFxDrag(bool commitChanges);
        void moveEffectToInsertIndex(int sourceIndex, int insertIndex);
        void updateMeterTooltip();
        void clampFxScrollOffset();
        void commitProjectEdit(const juce::String& actionName);
        void mouseDown(const juce::MouseEvent& event) override;
        void mouseDrag(const juce::MouseEvent& event) override;
        void mouseUp(const juce::MouseEvent& event) override;

        ProjectGetter projectGetter;
        ProjectWriter projectWriter;
        MasterMeterGetter masterMeterGetter;
        MasterEffectEditorOpener masterEffectEditorOpener;
        ProjectState currentProject;
        bool syncing = false;
        bool volumeDragging = false;

        juce::Label nameLabel;
        juce::Label fxSummaryLabel;
        juce::Label meterLabel;
        juce::TextButton fxButton;
        juce::TextButton fxAddButton;
        juce::TextButton fxBypassButton;
        juce::TextButton fxUpButton;
        juce::TextButton fxDownButton;
        MeterScaleComponent meterScale;
        MeterComponent leftMeter;
        MeterComponent rightMeter;
        juce::Slider volumeSlider;
        juce::Label volumeLabel;
        std::vector<std::unique_ptr<FxRowWidgets>> fxRows;
        std::pair<float, float> currentMeterLevels { 0.0f, 0.0f };
        int fxScrollOffset = 0;
        int fxVisibleRowCount = 0;
        juce::Rectangle<int> fxListBounds;
        bool fxDragActive = false;
        int fxDragSourceIndex = -1;
        int fxDropInsertIndex = -1;

        JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR(MasterStrip)
    };

    void ensureStripCount(int trackCount);

    ProjectGetter projectGetter;
    TrackWriter trackWriter;
    ProjectWriter projectWriter;
    MeterGetter meterGetter;
    MasterMeterGetter masterMeterGetter;
    TrackEffectEditorOpener trackEffectEditorOpener;
    MasterEffectEditorOpener masterEffectEditorOpener;
    juce::OwnedArray<ChannelStrip> strips;
    std::unique_ptr<MasterStrip> masterStrip;

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR(MixerComponent)
};

} // namespace aims
