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
        bool tribute909 = false;
        bool tributeVirus = false;
    };

    class AccentLookAndFeel final : public juce::LookAndFeel_V4
    {
    public:
        explicit AccentLookAndFeel (Theme themeToUse);

        void drawRotarySlider (juce::Graphics&, int x, int y, int width, int height, float sliderPosProportional,
                               float rotaryStartAngle, float rotaryEndAngle, juce::Slider&) override;

        Theme theme;
    };

    class LedToggleButton final : public juce::ToggleButton
    {
    public:
        LedToggleButton (Theme themeToUse, const juce::String& text, bool latching = true);
        void setScale (float scaleFactor);
        void paintButton (juce::Graphics&, bool shouldDrawButtonAsHighlighted, bool shouldDrawButtonAsDown) override;

    private:
        Theme theme;
        float scale = 1.0f;
    };

    class KnobCard final : public juce::Component
    {
    public:
        class PreviewSlider final : public juce::Slider
        {
        public:
            std::function<void()> onPreviewRequested;

            void mouseDown (const juce::MouseEvent& event) override
            {
                if (event.mods.isLeftButtonDown() && onPreviewRequested != nullptr)
                    onPreviewRequested();
                juce::Slider::mouseDown (event);
            }
        };

        KnobCard (AccentLookAndFeel&, const juce::String& titleText, const juce::String& hintText, bool compactLayout = false,
                  const juce::String& toggleText = {});
        ~KnobCard() override = default;

        void setScale (float scaleFactor);
        void resized() override;
        void paint (juce::Graphics&) override;
        [[nodiscard]] juce::Button* getToggleButton() noexcept;

        PreviewSlider slider;

    private:
        void updateMetrics();

        juce::Label titleLabel;
        juce::Label hintLabel;
        AccentLookAndFeel& lookAndFeel;
        std::unique_ptr<LedToggleButton> toggleButton;
        bool compact = false;
        float scale = 1.0f;

        JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR (KnobCard)
    };

    struct ChoiceSpec;

    class ChoiceCard final : public juce::Component
    {
    public:
        explicit ChoiceCard (Theme themeToUse, const ChoiceSpec& spec);
        ~ChoiceCard() override = default;

        void setScale (float scaleFactor);
        void resized() override;
        void paint (juce::Graphics&) override;

        juce::ComboBox combo;

    private:
        void updateMetrics();
        void syncButtonsFromCombo();

        Theme theme;
        juce::Label titleLabel;
        juce::Label hintLabel;
        float scale = 1.0f;
        bool usesLedButtons = false;
        int ledButtonColumns = 0;
        juce::OwnedArray<LedToggleButton> optionButtons;

        JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR (ChoiceCard)
    };

    class DrumPad final : public juce::Component
    {
    public:
        class PreviewSlider final : public juce::Slider
        {
        public:
            std::function<void()> onPreviewRequested;

            void mouseDown (const juce::MouseEvent& event) override
            {
                if (event.mods.isLeftButtonDown() && onPreviewRequested != nullptr)
                    onPreviewRequested();
                juce::Slider::mouseDown (event);
            }
        };

        DrumPad (AccentLookAndFeel&, const juce::String& titleText, const juce::String& noteText);
        ~DrumPad() override = default;

        void mouseDown (const juce::MouseEvent&) override;
        void resized() override;
        void paint (juce::Graphics&) override;

        std::function<void()> onPreviewRequested;
        PreviewSlider slider;

    private:
        juce::Label titleLabel;
        juce::Label noteLabel;
        AccentLookAndFeel& lookAndFeel;

        JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR (DrumPad)
    };

    struct KnobSpec
    {
        juce::String paramId;
        juce::String title;
        juce::String hint;
        int previewMidiNote = -1;
        juce::String toggleParamId;
        juce::String toggleText;
    };

    struct ChoiceSpec
    {
        juce::String paramId;
        juce::String title;
        juce::String hint;
        juce::StringArray values;
        juce::StringArray buttonLabels;
        bool ledButtons = false;
        int buttonColumns = 0;
    };

    struct DrumPadSpec
    {
        juce::String paramId;
        juce::String title;
        juce::String note;
        int midiNote = 36;
    };

    [[nodiscard]] Theme buildTheme() const;
    [[nodiscard]] std::vector<ChoiceSpec> buildChoiceSpecs() const;
    [[nodiscard]] std::vector<KnobSpec> buildKnobSpecs() const;
    [[nodiscard]] std::vector<DrumPadSpec> buildDrumPadSpecs() const;
    [[nodiscard]] bool isTribute303() const noexcept;
    [[nodiscard]] bool isTribute909() const noexcept;
    [[nodiscard]] bool isTributeVirus() const noexcept;
    [[nodiscard]] bool usesDrumPadLayout() const noexcept;
    void buildEditor();

    AdvancedVSTiAudioProcessor& audioProcessor;
    Theme theme;
    AccentLookAndFeel lookAndFeel;

    juce::Label badgeLabel;
    juce::Label titleLabel;
    juce::Label subtitleLabel;
    juce::OwnedArray<KnobCard> knobCards;
    juce::OwnedArray<ChoiceCard> choiceCards;
    juce::OwnedArray<DrumPad> drumPads;
    std::vector<std::unique_ptr<juce::AudioProcessorValueTreeState::SliderAttachment>> sliderAttachments;
    std::vector<std::unique_ptr<juce::AudioProcessorValueTreeState::ComboBoxAttachment>> comboAttachments;
    std::vector<std::unique_ptr<juce::AudioProcessorValueTreeState::ButtonAttachment>> buttonAttachments;

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR (AdvancedVSTiAudioProcessorEditor)
};
