#pragma once

#include <JuceHeader.h>
#include "PluginProcessor.h"

class AdvancedVSTiAudioProcessorEditor final : public juce::AudioProcessorEditor
{
public:
    explicit AdvancedVSTiAudioProcessorEditor (AdvancedVSTiAudioProcessor&);
    ~AdvancedVSTiAudioProcessorEditor() override = default;

    void paint (juce::Graphics&) override;
    void resized() override;

private:
    struct Theme
    {
        juce::String title;
        juce::String subtitle;
        juce::Colour accent;
        juce::Colour accentGlow;
        juce::Colour background;
        juce::Colour panel;
        juce::Colour panelEdge;
        juce::Colour text;
        juce::Colour hint;
        juce::Colour faceplate = juce::Colours::transparentBlack;
        juce::Colour trim = juce::Colours::transparentBlack;
        juce::Colour legend = juce::Colours::transparentBlack;
        juce::Colour knobBody = juce::Colours::transparentBlack;
        juce::Colour knobCap = juce::Colours::transparentBlack;
        bool tribute303 = false;
    };

    class AccentLookAndFeel final : public juce::LookAndFeel_V4
    {
    public:
        explicit AccentLookAndFeel (Theme themeToUse);

        void drawRotarySlider (juce::Graphics&, int x, int y, int width, int height, float sliderPosProportional,
                               float rotaryStartAngle, float rotaryEndAngle, juce::Slider&) override;

        Theme theme;
    };

    class KnobCard final : public juce::Component
    {
    public:
        KnobCard (AccentLookAndFeel&, const juce::String& titleText, const juce::String& hintText);
        ~KnobCard() override = default;

        void resized() override;
        void paint (juce::Graphics&) override;

        juce::Slider slider;

    private:
        juce::Label titleLabel;
        juce::Label hintLabel;
        AccentLookAndFeel& lookAndFeel;

        JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR (KnobCard)
    };

    class ChoiceCard final : public juce::Component
    {
    public:
        ChoiceCard (Theme themeToUse, const juce::String& titleText, const juce::String& hintText);
        ~ChoiceCard() override = default;

        void resized() override;
        void paint (juce::Graphics&) override;

        juce::ComboBox combo;

    private:
        Theme theme;
        juce::Label titleLabel;
        juce::Label hintLabel;

        JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR (ChoiceCard)
    };

    struct KnobSpec
    {
        juce::String paramId;
        juce::String title;
        juce::String hint;
    };

    struct ChoiceSpec
    {
        juce::String paramId;
        juce::String title;
        juce::String hint;
        juce::StringArray values;
    };

    [[nodiscard]] Theme buildTheme() const;
    [[nodiscard]] std::vector<ChoiceSpec> buildChoiceSpecs() const;
    [[nodiscard]] std::vector<KnobSpec> buildKnobSpecs() const;
    [[nodiscard]] bool isTribute303() const noexcept;
    void buildEditor();

    AdvancedVSTiAudioProcessor& audioProcessor;
    Theme theme;
    AccentLookAndFeel lookAndFeel;

    juce::Label badgeLabel;
    juce::Label titleLabel;
    juce::Label subtitleLabel;
    juce::OwnedArray<KnobCard> knobCards;
    juce::OwnedArray<ChoiceCard> choiceCards;
    std::vector<std::unique_ptr<juce::AudioProcessorValueTreeState::SliderAttachment>> sliderAttachments;
    std::vector<std::unique_ptr<juce::AudioProcessorValueTreeState::ComboBoxAttachment>> comboAttachments;

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR (AdvancedVSTiAudioProcessorEditor)
};
