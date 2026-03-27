#pragma once

#include <array>
#include <JuceHeader.h>
#include "PluginProcessor.h"

class AdvancedVSTiAudioProcessorEditor final : public juce::AudioProcessorEditor,
                                               private juce::Timer
{
public:
    explicit AdvancedVSTiAudioProcessorEditor (AdvancedVSTiAudioProcessor&);
    ~AdvancedVSTiAudioProcessorEditor() override = default;

    void paint (juce::Graphics&) override;
    void paintOverChildren (juce::Graphics&) override;
    void resized() override;

private:
    void timerCallback() override;

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
        bool vecPadMachine = false;
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
        void setVirusIndicatorOnly (bool shouldBeIndicatorOnly);
        void setVirusTopLedVisible (bool shouldShowTopLed);
        void paintButton (juce::Graphics&, bool shouldDrawButtonAsHighlighted, bool shouldDrawButtonAsDown) override;
        void mouseDown (const juce::MouseEvent&) override;
        void mouseUp (const juce::MouseEvent&) override;

    private:
        Theme theme;
        float scale = 1.0f;
        bool virusIndicatorOnly = false;
        bool virusTopLedVisible = false;
        double pressVisualUntilMs = 0.0;
    };

    class KnobCard final : public juce::Component,
                           public juce::SettableTooltipClient
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

    class ChoiceCard final : public juce::Component,
                             public juce::SettableTooltipClient
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
        void setSampleInfo (const juce::String& presetText, const juce::String& sampleText);
        void setStepperVisible (bool shouldShowStepper);
        void setStepperEnabled (bool canStepLeft, bool canStepRight);
        void setEnvelopeControlsVisible (bool shouldShowEnvelopeControls);

        std::function<void()> onPreviewRequested;
        std::function<void()> onStepLeftRequested;
        std::function<void()> onStepRightRequested;
        PreviewSlider slider;
        PreviewSlider sustainSlider;
        PreviewSlider releaseSlider;

    private:
        juce::Label titleLabel;
        juce::Label noteLabel;
        juce::Label presetLabel;
        juce::Label sampleLabel;
        juce::Label levelLabel;
        juce::Label sustainLabel;
        juce::Label releaseLabel;
        juce::TextButton leftButton { "<" };
        juce::TextButton rightButton { ">" };
        AccentLookAndFeel& lookAndFeel;
        bool showStepper = false;
        bool showEnvelopeControls = false;

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
        juce::String levelParamId;
        juce::String sustainParamId;
        juce::String releaseParamId;
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
    void syncExternalPadDisplays();
    void updateVirusOscillatorBindings();
    void updateVirusModulatorBindings();
    void refreshVirusValueKnobBindings();
    void refreshVirusMatrixMenuOsd();
    void refreshVirusLfoMenuOsd();
    void refreshVirusOscMenuOsd();
    void refreshVirusFilterMenuOsd();
    void refreshVirusFxMenuOsd (bool lowerSection);
    void refreshVirusArpMenuOsd();
    void refreshVirusPresetOsd();
    void bindVirusKnobToParam (int knobIndex,
                               const juce::String& paramId,
                               const juce::String& title,
                               const juce::String& hint,
                               std::function<void()> onValueChange = {});
    void refreshVirusShiftAwareKnobBindings();
    void clearVirusLfoMenu (bool clearOsd = false);
    void showVirusOsdMessage (const juce::String& title,
                              const juce::String& value,
                              const juce::String& detail = {},
                              bool pinned = false,
                              double lifetimeMs = 2200.0);
    void showVirusOsdForParam (const juce::String& paramId,
                               const juce::String& titleOverride = {},
                               const juce::String& detail = {},
                               bool pinned = false,
                               double lifetimeMs = 2200.0);
    void syncVirusPanelButtons();
    void buildVirusPanelButtons();
    LedToggleButton* addVirusPanelButton (const juce::String& key, const juce::String& text, const juce::String& tooltip, bool latching = true);
    [[nodiscard]] LedToggleButton* findVirusPanelButton (const juce::String& key) const;
    void buildEditor();

    AdvancedVSTiAudioProcessor& audioProcessor;
    Theme theme;
    AccentLookAndFeel lookAndFeel;
    juce::TooltipWindow tooltipWindow;

    juce::Label badgeLabel;
    juce::Label titleLabel;
    juce::Label subtitleLabel;
    juce::OwnedArray<KnobCard> knobCards;
    juce::OwnedArray<ChoiceCard> choiceCards;
    juce::OwnedArray<DrumPad> drumPads;
    juce::OwnedArray<LedToggleButton> virusPanelButtons;
    std::vector<juce::String> virusPanelButtonKeys;
    std::unique_ptr<LedToggleButton> virusBackgroundToggle;
    std::unique_ptr<LedToggleButton> virusKeyboardToggle;
    std::unique_ptr<juce::MidiKeyboardComponent> virusKeyboard;
    juce::Image virusTemplateImage;
    juce::Image backgroundImage;
    bool virusShowBackground = true;
    bool virusKeyboardVisible = false;
    int virusMatrixSlotIndex = 0;
    int virusMatrixTargetIndex = 0;
    int virusModulatorIndex = 0;
    int virusOscillatorIndex = 0;
    int virusFilterEditIndex = 0;
    int virusUpperFxLegendIndex = 0;
    int virusLowerFxLegendIndex = 0;
    int virusArpPageIndex = 0;
    int virusPanelModeIndex = 2;
    std::array<int, 3> virusModLeftTargetIndices { 0, 2, 4 };
    std::array<int, 3> virusModRightTargetIndices { -1, -1, -1 };
    juce::String virusOsdTitle;
    juce::String virusOsdValue;
    juce::String virusOsdDetail;
    double virusOsdUntilMs = 0.0;
    bool virusOsdPinned = false;
    int virusActiveLfoMenu = -1;
    int virusActiveMatrixMenu = -1;
    int virusActiveOscMenu = -1;
    int virusActiveFilterMenu = -1;
    int virusActiveFxMenu = -1;
    bool virusActivePresetMenu = false;
    bool virusActiveArpMenu = false;
    bool virusShiftMode = false;
    double virusPresetNavigationGuardUntilMs = 0.0;
    std::vector<std::unique_ptr<juce::AudioProcessorValueTreeState::SliderAttachment>> sliderAttachments;
    std::vector<std::unique_ptr<juce::AudioProcessorValueTreeState::ComboBoxAttachment>> comboAttachments;
    std::vector<std::unique_ptr<juce::AudioProcessorValueTreeState::ButtonAttachment>> buttonAttachments;

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR (AdvancedVSTiAudioProcessorEditor)
};
