#pragma once

#include "ProjectModel.h"

#include <juce_gui_basics/juce_gui_basics.h>

namespace aims
{

class MixerComponent final : public juce::Component
{
public:
    using ProjectGetter = std::function<const ProjectState&()>;
    using TrackWriter = std::function<void(int, const TrackState&, bool, const juce::String&)>;
    using MeterGetter = std::function<float(int)>;

    MixerComponent(ProjectGetter projectGetterIn,
                   TrackWriter trackWriterIn,
                   MeterGetter meterGetterIn);
    ~MixerComponent() override = default;

    void refreshFromModel();
    void refreshMeters();
    void paint(juce::Graphics& g) override;
    void resized() override;

private:
    class MeterComponent final : public juce::Component
    {
    public:
        void setLevel(float newLevel);
        void paint(juce::Graphics& g) override;

    private:
        float level = 0.0f;
    };

    class ChannelStrip final : public juce::Component
    {
    public:
        ChannelStrip(int trackIndexIn, TrackWriter trackWriterIn, MeterGetter meterGetterIn);

        void applyTrack(const TrackState& track);
        void refreshMeter();
        void paint(juce::Graphics& g) override;
        void resized() override;

    private:
        void commitTrackEdit(const juce::String& actionName);

        const int trackIndex;
        TrackWriter trackWriter;
        MeterGetter meterGetter;
        TrackState currentTrack;
        bool syncing = false;

        juce::Label nameLabel;
        MeterComponent meter;
        juce::Slider volumeSlider;
        juce::Label volumeLabel;
        juce::Slider panSlider;
        juce::ToggleButton muteButton;
        juce::ToggleButton soloButton;

        JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR(ChannelStrip)
    };

    void ensureStripCount(int trackCount);

    ProjectGetter projectGetter;
    TrackWriter trackWriter;
    MeterGetter meterGetter;
    juce::OwnedArray<ChannelStrip> strips;

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR(MixerComponent)
};

} // namespace aims
