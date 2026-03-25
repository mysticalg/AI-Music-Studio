#include "PluginEditor.h"

namespace
{
juce::String normalizedPluginName (const AdvancedVSTiAudioProcessor& processor)
{
    return processor.getName().trim().toLowerCase();
}

int scaledInt (float value, float scale)
{
    return juce::roundToInt (value * scale);
}

float scaledFloat (float value, float scale)
{
    return value * scale;
}
} // namespace

AdvancedVSTiAudioProcessorEditor::AccentLookAndFeel::AccentLookAndFeel (Theme themeToUse)
    : theme (std::move (themeToUse))
{
    setColour (juce::Slider::thumbColourId, theme.accent);
    setColour (juce::Slider::rotarySliderOutlineColourId, theme.panelEdge);
    setColour (juce::Slider::rotarySliderFillColourId, theme.accent);
    setColour (juce::Slider::trackColourId, theme.accent);
    setColour (juce::Slider::backgroundColourId, theme.panelEdge.withAlpha (0.85f));
    setColour (juce::ComboBox::backgroundColourId, theme.panel);
    setColour (juce::ComboBox::outlineColourId, theme.panelEdge);
    setColour (juce::ComboBox::textColourId, theme.text);
}

AdvancedVSTiAudioProcessorEditor::LedToggleButton::LedToggleButton (Theme themeToUse, const juce::String& text, bool latching)
    : theme (std::move (themeToUse))
{
    setButtonText (text);
    setClickingTogglesState (latching);
}

void AdvancedVSTiAudioProcessorEditor::LedToggleButton::setScale (float scaleFactor)
{
    scale = juce::jlimit (0.7f, 1.6f, scaleFactor);
    repaint();
}

void AdvancedVSTiAudioProcessorEditor::LedToggleButton::paintButton (juce::Graphics& g, bool shouldDrawButtonAsHighlighted, bool shouldDrawButtonAsDown)
{
    auto area = getLocalBounds().toFloat().reduced (0.5f);
    const auto radius = scaledFloat (4.5f, scale);
    const auto corner = scaledFloat (4.5f, scale);
    const auto outline = juce::jmax (1.0f, scaledFloat (1.0f, scale));
    const auto fill = shouldDrawButtonAsDown ? theme.panel.brighter (0.12f)
                                             : (shouldDrawButtonAsHighlighted ? theme.panel.brighter (0.08f) : theme.panel.brighter (0.04f));

    g.setColour (fill);
    g.fillRoundedRectangle (area, corner);
    g.setColour (theme.trim.withAlpha (0.95f));
    g.drawRoundedRectangle (area, corner, outline);

    auto ledBounds = juce::Rectangle<float> (radius * 2.0f, radius * 2.0f)
        .withCentre ({ area.getX() + scaledFloat (10.0f, scale), area.getCentreY() });
    g.setColour (getToggleState() ? theme.accent.brighter (0.16f) : theme.panelEdge.darker (0.6f));
    g.fillEllipse (ledBounds);
    g.setColour (juce::Colours::black.withAlpha (0.45f));
    g.drawEllipse (ledBounds, outline);

    g.setColour (getToggleState() ? theme.legend : theme.hint.brighter (0.1f));
    g.setFont (juce::Font (juce::FontOptions { 8.2f * scale, juce::Font::bold }));
    g.drawFittedText (getButtonText(),
                      getLocalBounds().withTrimmedLeft (scaledInt (18.0f, scale)).reduced (scaledInt (4.0f, scale), 0),
                      juce::Justification::centred,
                      1);
}

void AdvancedVSTiAudioProcessorEditor::AccentLookAndFeel::drawRotarySlider (
    juce::Graphics& g,
    int x,
    int y,
    int width,
    int height,
    float sliderPosProportional,
    float rotaryStartAngle,
    float rotaryEndAngle,
    juce::Slider&)
{
    if (theme.tribute303)
    {
        const auto rawBounds = juce::Rectangle<float> (static_cast<float> (x), static_cast<float> (y), static_cast<float> (width), static_cast<float> (height)).reduced (10.0f);
        const auto size = juce::jmin (rawBounds.getWidth(), rawBounds.getHeight());
        const auto bounds = juce::Rectangle<float> (size, size).withCentre (rawBounds.getCentre());
        const auto centre = bounds.getCentre();
        const auto radius = juce::jmin (bounds.getWidth(), bounds.getHeight()) * 0.5f;
        const auto angle = rotaryStartAngle + (sliderPosProportional * (rotaryEndAngle - rotaryStartAngle));

        g.setColour (juce::Colours::black.withAlpha (0.16f));
        g.fillEllipse (bounds.translated (0.0f, 2.0f));

        g.setColour (theme.knobBody.isTransparent() ? juce::Colour::fromRGB (233, 232, 225) : theme.knobBody);
        g.fillEllipse (bounds);
        g.setColour (juce::Colours::white.withAlpha (0.84f));
        g.drawEllipse (bounds.reduced (1.0f), 1.0f);
        g.setColour (juce::Colour::fromRGB (108, 108, 104));
        g.drawEllipse (bounds, 1.6f);

        for (int index = 0; index < 22; ++index)
        {
            const auto proportion = static_cast<float> (index) / 21.0f;
            const auto tickAngle = rotaryStartAngle + (proportion * (rotaryEndAngle - rotaryStartAngle));
            const auto innerRadius = radius * 0.76f;
            const auto outerRadius = radius * 0.92f;
            const auto innerPoint = centre.getPointOnCircumference (innerRadius, tickAngle);
            const auto outerPoint = centre.getPointOnCircumference (outerRadius, tickAngle);
            g.setColour (theme.trim.isTransparent() ? juce::Colour::fromRGB (126, 72, 46) : theme.trim.withAlpha (0.7f));
            g.drawLine ({ innerPoint, outerPoint }, 1.2f);
        }

        const auto capRadius = radius * 0.28f;
        g.setColour (theme.knobCap.isTransparent() ? juce::Colour::fromRGB (64, 64, 66) : theme.knobCap);
        g.fillEllipse (juce::Rectangle<float> (capRadius * 2.0f, capRadius * 2.0f).withCentre (centre));
        g.setColour (juce::Colours::black.withAlpha (0.25f));
        g.drawEllipse (juce::Rectangle<float> (capRadius * 2.0f, capRadius * 2.0f).withCentre (centre), 1.0f);

        juce::Path indicator;
        indicator.addRoundedRectangle (-2.0f, -radius * 0.72f, 4.0f, radius * 0.36f, 2.0f);
        g.setColour (theme.accent);
        g.fillPath (indicator, juce::AffineTransform::rotation (angle).translated (centre.x, centre.y));
        return;
    }

    if (theme.tribute909)
    {
        const auto rawBounds = juce::Rectangle<float> (static_cast<float> (x), static_cast<float> (y), static_cast<float> (width), static_cast<float> (height)).reduced (10.0f);
        const auto size = juce::jmin (rawBounds.getWidth(), rawBounds.getHeight());
        const auto bounds = juce::Rectangle<float> (size, size).withCentre (rawBounds.getCentre());
        const auto centre = bounds.getCentre();
        const auto radius = juce::jmin (bounds.getWidth(), bounds.getHeight()) * 0.5f;
        const auto angle = rotaryStartAngle + (sliderPosProportional * (rotaryEndAngle - rotaryStartAngle));

        g.setColour (juce::Colours::black.withAlpha (0.24f));
        g.fillEllipse (bounds.translated (0.0f, 2.5f));

        g.setColour (theme.knobBody.isTransparent() ? juce::Colour::fromRGB (48, 49, 49) : theme.knobBody);
        g.fillEllipse (bounds);
        g.setColour (juce::Colours::white.withAlpha (0.18f));
        g.drawEllipse (bounds.reduced (1.2f), 1.0f);
        g.setColour (juce::Colours::black.withAlpha (0.45f));
        g.drawEllipse (bounds, 1.2f);

        for (int index = 0; index < 19; ++index)
        {
            const auto proportion = static_cast<float> (index) / 18.0f;
            const auto tickAngle = rotaryStartAngle + (proportion * (rotaryEndAngle - rotaryStartAngle));
            const auto innerPoint = centre.getPointOnCircumference (radius * 0.84f, tickAngle);
            const auto outerPoint = centre.getPointOnCircumference (radius * 1.08f, tickAngle);
            g.setColour (theme.panelEdge.darker (0.4f).withAlpha (0.9f));
            g.drawLine ({ innerPoint, outerPoint }, index % 3 == 0 ? 1.4f : 0.9f);
        }

        juce::Path indicator;
        indicator.addRoundedRectangle (-1.8f, -radius * 0.72f, 3.6f, radius * 0.34f, 1.8f);
        g.setColour (theme.accent);
        g.fillPath (indicator, juce::AffineTransform::rotation (angle).translated (centre.x, centre.y));

        const auto capRadius = radius * 0.22f;
        g.setColour (theme.knobCap.isTransparent() ? juce::Colour::fromRGB (78, 78, 76) : theme.knobCap);
        g.fillEllipse (juce::Rectangle<float> (capRadius * 2.0f, capRadius * 2.0f).withCentre (centre));
        return;
    }

    if (theme.tributeVirus)
    {
        const auto rawBounds = juce::Rectangle<float> (static_cast<float> (x), static_cast<float> (y), static_cast<float> (width), static_cast<float> (height)).reduced (12.0f);
        const auto size = juce::jmin (rawBounds.getWidth(), rawBounds.getHeight());
        const auto bounds = juce::Rectangle<float> (size, size).withCentre (rawBounds.getCentre());
        const auto centre = bounds.getCentre();
        const auto radius = juce::jmin (bounds.getWidth(), bounds.getHeight()) * 0.5f;
        const auto angle = rotaryStartAngle + (sliderPosProportional * (rotaryEndAngle - rotaryStartAngle));

        g.setColour (juce::Colours::black.withAlpha (0.28f));
        g.fillEllipse (bounds.translated (0.0f, 2.5f));

        juce::ColourGradient knobFill (theme.knobBody.brighter (0.2f), bounds.getCentreX(), bounds.getY(),
                                       theme.knobBody.darker (0.55f), bounds.getCentreX(), bounds.getBottom(), false);
        g.setGradientFill (knobFill);
        g.fillEllipse (bounds);
        g.setColour (juce::Colours::white.withAlpha (0.22f));
        g.drawEllipse (bounds.reduced (1.0f), 1.0f);
        g.setColour (theme.trim.withAlpha (0.9f));
        g.drawEllipse (bounds, 1.3f);

        for (int index = 0; index < 21; ++index)
        {
            const auto proportion = static_cast<float> (index) / 20.0f;
            const auto tickAngle = rotaryStartAngle + (proportion * (rotaryEndAngle - rotaryStartAngle));
            const auto innerPoint = centre.getPointOnCircumference (radius * 0.84f, tickAngle);
            const auto outerPoint = centre.getPointOnCircumference (radius * 1.08f, tickAngle);
            g.setColour ((index % 5 == 0) ? theme.legend.withAlpha (0.72f) : theme.trim.withAlpha (0.6f));
            g.drawLine ({ innerPoint, outerPoint }, index % 5 == 0 ? 1.3f : 0.9f);
        }

        juce::Path indicator;
        indicator.addRoundedRectangle (-1.9f, -radius * 0.74f, 3.8f, radius * 0.4f, 1.9f);
        g.setColour (theme.knobCap.isTransparent() ? theme.text : theme.knobCap);
        g.fillPath (indicator, juce::AffineTransform::rotation (angle).translated (centre.x, centre.y));

        const auto capRadius = radius * 0.22f;
        g.setColour (theme.faceplate.brighter (0.2f));
        g.fillEllipse (juce::Rectangle<float> (capRadius * 2.0f, capRadius * 2.0f).withCentre (centre));
        g.setColour (juce::Colours::black.withAlpha (0.35f));
        g.drawEllipse (juce::Rectangle<float> (capRadius * 2.0f, capRadius * 2.0f).withCentre (centre), 1.0f);
        return;
    }

    const auto rawBounds = juce::Rectangle<float> (static_cast<float> (x), static_cast<float> (y), static_cast<float> (width), static_cast<float> (height)).reduced (8.0f);
    const auto size = juce::jmin (rawBounds.getWidth(), rawBounds.getHeight());
    const auto bounds = juce::Rectangle<float> (size, size).withCentre (rawBounds.getCentre());
    const auto radius = juce::jmin (bounds.getWidth(), bounds.getHeight()) * 0.5f;
    const auto lineW = juce::jmin (7.0f, radius * 0.14f);
    const auto angle = rotaryStartAngle + (sliderPosProportional * (rotaryEndAngle - rotaryStartAngle));
    const auto centre = bounds.getCentre();
    const auto arcRadius = radius - (lineW * 0.5f);

    g.setColour (theme.accentGlow.withAlpha (0.18f));
    g.fillEllipse (bounds.expanded (4.0f));

    g.setColour (theme.panel.brighter (0.12f));
    g.fillEllipse (bounds);

    g.setColour (theme.panelEdge);
    g.drawEllipse (bounds, 1.4f);

    juce::Path backgroundArc;
    backgroundArc.addCentredArc (centre.x, centre.y, arcRadius, arcRadius, 0.0f, rotaryStartAngle, rotaryEndAngle, true);
    g.setColour (theme.panelEdge.brighter (0.2f));
    g.strokePath (backgroundArc, juce::PathStrokeType (lineW, juce::PathStrokeType::curved, juce::PathStrokeType::rounded));

    juce::Path valueArc;
    valueArc.addCentredArc (centre.x, centre.y, arcRadius, arcRadius, 0.0f, rotaryStartAngle, angle, true);
    g.setColour (theme.accent);
    g.strokePath (valueArc, juce::PathStrokeType (lineW, juce::PathStrokeType::curved, juce::PathStrokeType::rounded));

    const auto pointerLength = arcRadius * 0.72f;
    const auto pointerThickness = juce::jmax (2.0f, radius * 0.08f);
    juce::Path pointer;
    pointer.addRoundedRectangle (-pointerThickness * 0.5f, -pointerLength, pointerThickness, pointerLength, pointerThickness * 0.5f);
    g.setColour (theme.text);
    g.fillPath (pointer, juce::AffineTransform::rotation (angle).translated (centre.x, centre.y));

    g.setColour (theme.panelEdge.brighter (0.6f));
    g.fillEllipse (juce::Rectangle<float> (pointerThickness * 1.8f, pointerThickness * 1.8f).withCentre (centre));
}

AdvancedVSTiAudioProcessorEditor::KnobCard::KnobCard (
    AccentLookAndFeel& lf,
    const juce::String& titleText,
    const juce::String& hintText,
    bool compactLayout,
    const juce::String& toggleText)
    : lookAndFeel (lf),
      compact (compactLayout)
{
    titleLabel.setText (titleText, juce::dontSendNotification);
    titleLabel.setJustificationType (juce::Justification::centred);
    titleLabel.setColour (juce::Label::textColourId,
                          (lf.theme.tributeVirus || lf.theme.tribute303 || lf.theme.tribute909) ? lf.theme.legend : lf.theme.text);
    titleLabel.setMinimumHorizontalScale (0.72f);
    addAndMakeVisible (titleLabel);

    hintLabel.setText (hintText, juce::dontSendNotification);
    hintLabel.setJustificationType (juce::Justification::centred);
    hintLabel.setColour (juce::Label::textColourId,
                         (lf.theme.tribute909 || lf.theme.tributeVirus) ? lf.theme.panelEdge.brighter (0.18f) : lf.theme.hint);
    hintLabel.setMinimumHorizontalScale (0.75f);
    addAndMakeVisible (hintLabel);

    if (toggleText.isNotEmpty())
    {
        toggleButton = std::make_unique<LedToggleButton> (lf.theme, toggleText);
        addAndMakeVisible (*toggleButton);
    }

    slider.setSliderStyle (juce::Slider::RotaryHorizontalVerticalDrag);
    slider.setLookAndFeel (&lookAndFeel);
    slider.setColour (juce::Slider::textBoxTextColourId, lf.theme.text);
    slider.setColour (juce::Slider::textBoxBackgroundColourId,
                      lf.theme.tribute303 ? lf.theme.faceplate.brighter (0.05f)
                                         : (lf.theme.tributeVirus ? lf.theme.panel.brighter (0.04f)
                                                                  : (lf.theme.tribute909 ? lf.theme.faceplate.brighter (0.1f) : lf.theme.panel.darker (0.2f))));
    slider.setColour (juce::Slider::textBoxOutlineColourId, (lf.theme.tribute303 || lf.theme.tribute909 || lf.theme.tributeVirus) ? lf.theme.trim : lf.theme.panelEdge);
    slider.setColour (juce::Slider::textBoxHighlightColourId, lf.theme.accent.withAlpha (0.18f));
    addAndMakeVisible (slider);

    updateMetrics();
}

void AdvancedVSTiAudioProcessorEditor::KnobCard::setScale (float scaleFactor)
{
    scale = juce::jlimit (0.7f, 1.8f, scaleFactor);
    if (toggleButton != nullptr)
        toggleButton->setScale (scale);
    updateMetrics();
    resized();
}

void AdvancedVSTiAudioProcessorEditor::KnobCard::updateMetrics()
{
    const auto titleSize = (lookAndFeel.theme.tributeVirus ? 9.4f
                            : (lookAndFeel.theme.tribute909 && compact ? 8.4f
                               : ((lookAndFeel.theme.tribute303 || compact) ? 10.5f : 14.0f))) * scale;
    const auto hintSize = (lookAndFeel.theme.tributeVirus ? 7.4f
                           : (lookAndFeel.theme.tribute909 && compact ? 0.1f
                              : ((lookAndFeel.theme.tribute303 || compact) ? 8.4f : 11.0f))) * scale;
    titleLabel.setFont (juce::Font (titleSize, juce::Font::bold));
    hintLabel.setFont (juce::Font (hintSize, juce::Font::plain));

    slider.setTextBoxStyle (juce::Slider::TextBoxBelow, false,
                            juce::roundToInt ((lookAndFeel.theme.tributeVirus ? 44.0f : (lookAndFeel.theme.tribute909 ? (compact ? 42.0f : 54.0f) : (lookAndFeel.theme.tribute303 ? 56.0f : (compact ? 62.0f : 72.0f)))) * scale),
                            juce::roundToInt ((lookAndFeel.theme.tributeVirus ? 15.0f : (lookAndFeel.theme.tribute909 ? (compact ? 16.0f : 18.0f) : (lookAndFeel.theme.tribute303 ? 18.0f : 22.0f))) * scale));
    slider.setMouseDragSensitivity (juce::roundToInt ((lookAndFeel.theme.tributeVirus ? 185.0f : (compact ? 170.0f : 140.0f)) / juce::jmax (0.75f, scale)));
}

juce::Button* AdvancedVSTiAudioProcessorEditor::KnobCard::getToggleButton() noexcept
{
    return toggleButton.get();
}

void AdvancedVSTiAudioProcessorEditor::KnobCard::paint (juce::Graphics& g)
{
    auto area = getLocalBounds().toFloat().reduced (1.0f);
    if (lookAndFeel.theme.tribute303)
    {
        const auto radius = scaledFloat (12.0f, scale);
        g.setColour (lookAndFeel.theme.faceplate.withAlpha (0.92f));
        g.fillRoundedRectangle (area, radius);
        g.setColour (lookAndFeel.theme.trim.withAlpha (0.85f));
        g.drawRoundedRectangle (area, radius, juce::jmax (1.0f, scaledFloat (1.1f, scale)));
        g.setColour (lookAndFeel.theme.accent.withAlpha (0.18f));
        g.fillRoundedRectangle (area.removeFromBottom (scaledFloat (3.0f, scale)), scaledFloat (1.5f, scale));
        return;
    }

    if (lookAndFeel.theme.tributeVirus)
    {
        const auto radius = scaledFloat (8.0f, scale);
        juce::ColourGradient fill (lookAndFeel.theme.panel.brighter (0.08f), area.getCentreX(), area.getY(),
                                   lookAndFeel.theme.panel.darker (0.18f), area.getCentreX(), area.getBottom(), false);
        g.setGradientFill (fill);
        g.fillRoundedRectangle (area, radius);
        g.setColour (lookAndFeel.theme.panelEdge.withAlpha (0.95f));
        g.drawRoundedRectangle (area, radius, juce::jmax (1.0f, scaledFloat (1.0f, scale)));
        g.setColour (lookAndFeel.theme.trim.withAlpha (0.75f));
        g.drawLine (area.getX() + scaledFloat (8.0f, scale),
                    area.getY() + scaledFloat (20.0f, scale),
                    area.getRight() - scaledFloat (8.0f, scale),
                    area.getY() + scaledFloat (20.0f, scale),
                    juce::jmax (1.0f, scaledFloat (1.0f, scale)));
        return;
    }

    if (lookAndFeel.theme.tribute909)
    {
        if (compact)
            return;

        const auto radius = scaledFloat (8.0f, scale);
        g.setColour (lookAndFeel.theme.faceplate.withAlpha (0.98f));
        g.fillRoundedRectangle (area, radius);
        g.setColour (lookAndFeel.theme.panelEdge);
        g.drawRoundedRectangle (area, radius, juce::jmax (1.0f, scaledFloat (1.0f, scale)));

        auto header = area.removeFromTop (scaledFloat (24.0f, scale));
        g.setColour (lookAndFeel.theme.panelEdge);
        g.fillRoundedRectangle (header, scaledFloat (6.0f, scale));
        g.setColour (lookAndFeel.theme.accent.withAlpha (0.9f));
        g.fillRect (header.removeFromBottom (scaledFloat (2.0f, scale)));
        return;
    }

    g.setColour (lookAndFeel.theme.panel.withAlpha (0.92f));
    g.fillRoundedRectangle (area, scaledFloat (18.0f, scale));
    g.setColour (lookAndFeel.theme.panelEdge);
    g.drawRoundedRectangle (area, scaledFloat (18.0f, scale), juce::jmax (1.0f, scaledFloat (1.2f, scale)));
}

void AdvancedVSTiAudioProcessorEditor::KnobCard::resized()
{
    if (lookAndFeel.theme.tributeVirus)
    {
        auto area = getLocalBounds().reduced (scaledInt (7.0f, scale));
        auto header = area.removeFromTop (scaledInt (18.0f, scale));
        if (toggleButton != nullptr)
        {
            const auto toggleWidth = scaledInt (48.0f, scale);
            const auto toggleHeight = scaledInt (18.0f, scale);
            auto toggleArea = header.removeFromRight (toggleWidth);
            toggleButton->setBounds (juce::Rectangle<int> (toggleWidth, toggleHeight).withCentre (toggleArea.getCentre()));
            header.removeFromRight (scaledInt (4.0f, scale));
        }

        titleLabel.setBounds (header);
        hintLabel.setBounds (area.removeFromBottom (scaledInt (12.0f, scale)));

        const auto sliderWidth = scaledInt (74.0f, scale);
        const auto sliderHeight = scaledInt (88.0f, scale);
        auto sliderArea = area.removeFromTop (juce::jmin (area.getHeight(), sliderHeight));
        slider.setBounds (juce::Rectangle<int> (sliderWidth, juce::jmin (sliderHeight, sliderArea.getHeight())).withCentre (sliderArea.getCentre()));
        return;
    }

    if (lookAndFeel.theme.tribute909)
    {
        auto area = getLocalBounds().reduced (scaledInt (compact ? 2.0f : 8.0f, scale));
        titleLabel.setBounds (area.removeFromTop (scaledInt (compact ? 12.0f : 18.0f, scale)));
        slider.setBounds (area.removeFromTop (compact
                                                  ? juce::jmin (scaledInt (54.0f, scale),
                                                                juce::jmax (scaledInt (44.0f, scale), area.getHeight() - scaledInt (24.0f, scale)))
                                                  : juce::jmin (scaledInt (78.0f, scale),
                                                                juce::jmax (scaledInt (60.0f, scale), area.getHeight() - scaledInt (30.0f, scale)))));
        if (compact)
            hintLabel.setBounds ({});
        else
            hintLabel.setBounds (area.removeFromTop (scaledInt (14.0f, scale)));
        return;
    }

    auto area = getLocalBounds().reduced (scaledInt (lookAndFeel.theme.tribute303 ? 8.0f : (compact ? 9.0f : 12.0f), scale));
    if (toggleButton != nullptr)
        toggleButton->setBounds ({});
    titleLabel.setBounds (area.removeFromTop (scaledInt ((lookAndFeel.theme.tribute303 || compact) ? 18.0f : 22.0f, scale)));
    hintLabel.setBounds (area.removeFromBottom (scaledInt (lookAndFeel.theme.tribute303 ? 16.0f : (compact ? 16.0f : 20.0f), scale)));
    slider.setBounds (area.reduced (0, scaledInt (lookAndFeel.theme.tribute303 ? 0.0f : (compact ? 4.0f : 2.0f), scale)));
}

AdvancedVSTiAudioProcessorEditor::ChoiceCard::ChoiceCard (Theme themeToUse, const ChoiceSpec& spec)
    : theme (std::move (themeToUse))
{
    titleLabel.setText (spec.title, juce::dontSendNotification);
    titleLabel.setJustificationType (theme.tributeVirus ? juce::Justification::centred : juce::Justification::centredLeft);
    titleLabel.setColour (juce::Label::textColourId, (theme.tribute303 || theme.tribute909 || theme.tributeVirus) ? theme.legend : theme.text);
    titleLabel.setMinimumHorizontalScale (0.72f);
    addAndMakeVisible (titleLabel);

    hintLabel.setText (spec.hint, juce::dontSendNotification);
    hintLabel.setJustificationType (theme.tributeVirus ? juce::Justification::centred : juce::Justification::centredLeft);
    hintLabel.setColour (juce::Label::textColourId, (theme.tribute909 || theme.tributeVirus) ? theme.panelEdge.brighter (0.18f) : theme.hint);
    hintLabel.setMinimumHorizontalScale (0.72f);
    addAndMakeVisible (hintLabel);

    usesLedButtons = theme.tributeVirus && spec.ledButtons && spec.values.size() > 0;
    ledButtonColumns = juce::jmax (1, spec.buttonColumns > 0 ? spec.buttonColumns : spec.values.size());

    combo.setColour (juce::ComboBox::backgroundColourId,
                     theme.tribute303 ? theme.faceplate.brighter (0.06f)
                                      : (theme.tributeVirus ? theme.panel.brighter (0.04f)
                                                             : (theme.tribute909 ? theme.faceplate.brighter (0.08f) : theme.panel.brighter (0.08f))));
    combo.setColour (juce::ComboBox::outlineColourId, (theme.tribute303 || theme.tribute909 || theme.tributeVirus) ? theme.trim : theme.panelEdge);
    combo.setColour (juce::ComboBox::textColourId, theme.text);
    combo.setColour (juce::ComboBox::arrowColourId, theme.accent);
    addAndMakeVisible (combo);

    if (usesLedButtons)
    {
        const auto labels = spec.buttonLabels.isEmpty() ? spec.values : spec.buttonLabels;
        for (int index = 0; index < labels.size(); ++index)
        {
            auto* button = optionButtons.add (new LedToggleButton (theme, labels[index], false));
            button->onClick = [this, index]
            {
                combo.setSelectedId (index + 1, juce::sendNotificationSync);
                syncButtonsFromCombo();
            };
            addAndMakeVisible (button);
        }

        combo.onChange = [this]
        {
            syncButtonsFromCombo();
        };
    }

    updateMetrics();
}

void AdvancedVSTiAudioProcessorEditor::ChoiceCard::setScale (float scaleFactor)
{
    scale = juce::jlimit (0.7f, 1.8f, scaleFactor);
    updateMetrics();
    resized();
}

void AdvancedVSTiAudioProcessorEditor::ChoiceCard::updateMetrics()
{
    titleLabel.setFont (juce::Font (((theme.tribute303 || theme.tribute909 || theme.tributeVirus) ? 10.5f : 13.0f) * scale, juce::Font::bold));
    hintLabel.setFont (juce::Font (((theme.tribute303 || theme.tribute909 || theme.tributeVirus) ? 8.4f : 11.0f) * scale, juce::Font::plain));
    for (auto* button : optionButtons)
        button->setScale (theme.tributeVirus ? scale * 0.92f : scale);
}

void AdvancedVSTiAudioProcessorEditor::ChoiceCard::paint (juce::Graphics& g)
{
    auto area = getLocalBounds().toFloat().reduced (1.0f);
    if (theme.tribute303)
    {
        const auto radius = scaledFloat (10.0f, scale);
        g.setColour (theme.faceplate.withAlpha (0.96f));
        g.fillRoundedRectangle (area, radius);
        g.setColour (theme.trim);
        g.drawRoundedRectangle (area, radius, juce::jmax (1.0f, scaledFloat (1.0f, scale)));
        g.setColour (theme.accent.withAlpha (0.88f));
        g.fillRoundedRectangle (area.removeFromLeft (scaledFloat (6.0f, scale)), scaledFloat (6.0f, scale));
        return;
    }

    if (theme.tributeVirus)
    {
        const auto radius = scaledFloat (8.0f, scale);
        juce::ColourGradient fill (theme.panel.brighter (0.06f), area.getCentreX(), area.getY(),
                                   theme.panel.darker (0.18f), area.getCentreX(), area.getBottom(), false);
        g.setGradientFill (fill);
        g.fillRoundedRectangle (area, radius);
        g.setColour (theme.panelEdge);
        g.drawRoundedRectangle (area, radius, juce::jmax (1.0f, scaledFloat (1.0f, scale)));
        g.setColour (theme.trim.withAlpha (0.75f));
        g.fillRect (area.removeFromTop (scaledFloat (3.0f, scale)));
        return;
    }

    if (theme.tribute909)
    {
        const auto radius = scaledFloat (8.0f, scale);
        g.setColour (theme.faceplate.withAlpha (0.98f));
        g.fillRoundedRectangle (area, radius);
        g.setColour (theme.panelEdge);
        g.drawRoundedRectangle (area, radius, juce::jmax (1.0f, scaledFloat (1.0f, scale)));

        auto header = area.removeFromTop (scaledFloat (24.0f, scale));
        g.setColour (theme.panelEdge);
        g.fillRoundedRectangle (header, scaledFloat (6.0f, scale));
        g.setColour (theme.accent.withAlpha (0.95f));
        g.fillRect (header.removeFromBottom (scaledFloat (2.0f, scale)));
        return;
    }

    g.setColour (theme.panel.withAlpha (0.92f));
    g.fillRoundedRectangle (area, scaledFloat (16.0f, scale));
    g.setColour (theme.panelEdge);
    g.drawRoundedRectangle (area, scaledFloat (16.0f, scale), juce::jmax (1.0f, scaledFloat (1.2f, scale)));
}

void AdvancedVSTiAudioProcessorEditor::ChoiceCard::resized()
{
    if (theme.tributeVirus)
    {
        auto area = getLocalBounds().reduced (scaledInt (8.0f, scale));
        titleLabel.setBounds (area.removeFromTop (scaledInt (14.0f, scale)));
        hintLabel.setBounds (area.removeFromTop (scaledInt (11.0f, scale)));
        area.removeFromTop (scaledInt (3.0f, scale));

        if (usesLedButtons && optionButtons.size() > 0)
        {
            combo.setVisible (false);
            combo.setBounds ({});

            const int spacing = scaledInt (5.0f, scale);
            const int columns = juce::jmax (1, ledButtonColumns);
            const int rows = juce::jmax (1, (optionButtons.size() + columns - 1) / columns);
            const int cellWidth = (area.getWidth() - spacing * (columns - 1)) / columns;
            const int cellHeight = (area.getHeight() - spacing * (rows - 1)) / rows;

            for (int index = 0; index < optionButtons.size(); ++index)
            {
                const int row = index / columns;
                const int column = index % columns;
                optionButtons[index]->setBounds (area.getX() + column * (cellWidth + spacing),
                                                 area.getY() + row * (cellHeight + spacing),
                                                 cellWidth,
                                                 cellHeight);
            }
        }
        else
        {
            combo.setVisible (true);
            combo.setBounds (area.removeFromTop (scaledInt (24.0f, scale)));
        }
        return;
    }

    auto area = getLocalBounds().reduced (scaledInt (theme.tribute909 ? 8.0f : 12.0f, scale));
    titleLabel.setBounds (area.removeFromTop (scaledInt (theme.tribute909 ? 18.0f : 20.0f, scale)));
    hintLabel.setBounds (area.removeFromTop (scaledInt (theme.tribute909 ? 14.0f : 18.0f, scale)));
    combo.setBounds (area.removeFromTop (scaledInt (theme.tribute909 ? 28.0f : 32.0f, scale)));
}

void AdvancedVSTiAudioProcessorEditor::ChoiceCard::syncButtonsFromCombo()
{
    const auto selected = combo.getSelectedId();
    for (int index = 0; index < optionButtons.size(); ++index)
        optionButtons[index]->setToggleState (selected == index + 1, juce::dontSendNotification);
}

AdvancedVSTiAudioProcessorEditor::DrumPad::DrumPad (
    AccentLookAndFeel& lf,
    const juce::String& titleText,
    const juce::String& noteText)
    : lookAndFeel (lf)
{
    titleLabel.setText (titleText, juce::dontSendNotification);
    titleLabel.setJustificationType (juce::Justification::centred);
    titleLabel.setFont (juce::Font (juce::FontOptions { lf.theme.tribute909 ? 12.0f : 13.0f, juce::Font::bold }));
    titleLabel.setColour (juce::Label::textColourId, lf.theme.tribute909 ? lf.theme.legend : lf.theme.text);
    titleLabel.setMinimumHorizontalScale (0.7f);
    titleLabel.setInterceptsMouseClicks (false, false);
    addAndMakeVisible (titleLabel);

    noteLabel.setText (noteText, juce::dontSendNotification);
    noteLabel.setJustificationType (juce::Justification::centred);
    noteLabel.setFont (juce::Font (juce::FontOptions { lf.theme.tribute909 ? 9.5f : 10.5f, juce::Font::plain }));
    noteLabel.setColour (juce::Label::textColourId, lf.theme.tribute909 ? lf.theme.panelEdge.darker (0.42f) : lf.theme.hint);
    noteLabel.setInterceptsMouseClicks (false, false);
    addAndMakeVisible (noteLabel);

    slider.setSliderStyle (juce::Slider::RotaryHorizontalVerticalDrag);
    slider.setTextBoxStyle (juce::Slider::TextBoxBelow, false, lf.theme.tribute909 ? 52 : 58, 20);
    slider.setLookAndFeel (&lookAndFeel);
    slider.setColour (juce::Slider::textBoxTextColourId, lf.theme.text);
    slider.setColour (juce::Slider::textBoxBackgroundColourId, lf.theme.tribute909 ? lf.theme.faceplate.brighter (0.08f) : lf.theme.panel.brighter (0.08f));
    slider.setColour (juce::Slider::textBoxOutlineColourId, lf.theme.tribute909 ? lf.theme.trim : lf.theme.panelEdge);
    slider.setColour (juce::Slider::textBoxHighlightColourId, lf.theme.accent.withAlpha (0.16f));
    slider.setMouseDragSensitivity (160);
    addAndMakeVisible (slider);
}

void AdvancedVSTiAudioProcessorEditor::DrumPad::mouseDown (const juce::MouseEvent& event)
{
    if (event.mods.isLeftButtonDown() && onPreviewRequested != nullptr)
        onPreviewRequested();
}

void AdvancedVSTiAudioProcessorEditor::DrumPad::paint (juce::Graphics& g)
{
    auto area = getLocalBounds().toFloat().reduced (1.0f);

    if (lookAndFeel.theme.tribute909)
    {
        g.setColour (lookAndFeel.theme.faceplate.withAlpha (0.98f));
        g.fillRoundedRectangle (area, 8.0f);
        g.setColour (lookAndFeel.theme.panelEdge);
        g.drawRoundedRectangle (area, 8.0f, 1.0f);

        auto header = area.removeFromTop (34.0f);
        g.setColour (lookAndFeel.theme.panelEdge);
        g.fillRoundedRectangle (header, 6.0f);
        g.setColour (lookAndFeel.theme.accent.withAlpha (0.92f));
        g.fillRect (header.removeFromBottom (2.0f));
        return;
    }

    juce::ColourGradient padFill (lookAndFeel.theme.panel.brighter (0.18f), area.getTopLeft(),
                                  lookAndFeel.theme.panel.darker (0.16f), area.getBottomRight(), false);
    g.setGradientFill (padFill);
    g.fillRoundedRectangle (area, 20.0f);

    g.setColour (lookAndFeel.theme.accentGlow.withAlpha (0.18f));
    g.fillRoundedRectangle (area.reduced (10.0f, 8.0f).removeFromTop (40.0f), 12.0f);

    g.setColour (lookAndFeel.theme.panelEdge);
    g.drawRoundedRectangle (area, 20.0f, 1.2f);

    g.setColour (lookAndFeel.theme.accent.withAlpha (0.6f));
    g.fillRoundedRectangle (area.reduced (14.0f).removeFromBottom (4.0f), 2.0f);
}

void AdvancedVSTiAudioProcessorEditor::DrumPad::resized()
{
    auto area = getLocalBounds().reduced (lookAndFeel.theme.tribute909 ? 8 : 12);
    titleLabel.setBounds (area.removeFromTop (lookAndFeel.theme.tribute909 ? 18 : 20));
    noteLabel.setBounds (area.removeFromTop (lookAndFeel.theme.tribute909 ? 12 : 16));
    area.removeFromTop (lookAndFeel.theme.tribute909 ? 0 : 2);
    slider.setBounds (area.reduced (lookAndFeel.theme.tribute909 ? 2 : 6, 0));
}

AdvancedVSTiAudioProcessorEditor::AdvancedVSTiAudioProcessorEditor (AdvancedVSTiAudioProcessor& p)
    : AudioProcessorEditor (&p),
      audioProcessor (p),
      theme (buildTheme()),
      lookAndFeel (theme)
{
    buildEditor();

    int defaultWidth = 1040;
    int defaultHeight = 640;
    int minWidth = 820;
    int minHeight = 520;
    int maxWidth = 2200;
    int maxHeight = 1600;

    if (isTribute303())
    {
        defaultWidth = 1020;
        defaultHeight = 430;
        minWidth = 860;
        minHeight = 380;
        maxWidth = 1800;
        maxHeight = 980;
    }
    else if (isTributeVirus())
    {
        defaultWidth = 1500;
        defaultHeight = 820;
        minWidth = 1240;
        minHeight = 700;
        maxWidth = 2800;
        maxHeight = 1800;
    }
    else if (usesDrumPadLayout())
    {
        defaultWidth = isTribute909() ? 1120 : 1180;
        defaultHeight = isTribute909() ? 640 : 860;
        minWidth = isTribute909() ? 940 : 960;
        minHeight = isTribute909() ? 540 : 700;
        maxWidth = 2400;
        maxHeight = 1800;
    }

    setResizable (true, false);
    setResizeLimits (minWidth, minHeight, maxWidth, maxHeight);
    setSize (defaultWidth, defaultHeight);
}

void AdvancedVSTiAudioProcessorEditor::buildEditor()
{
    badgeLabel.setText (isTribute303() ? "ACID BASS LINE EMULATOR TRIBUTE"
                                       : (isTribute909() ? "RHYTHM COMPOSER TRIBUTE"
                                                         : (isTributeVirus() ? "ACCESS-STYLE MAIN SYNTH" : audioProcessor.getName())),
                        juce::dontSendNotification);
    badgeLabel.setJustificationType (juce::Justification::centredLeft);
    badgeLabel.setFont (juce::Font ((isTribute303() || isTribute909() || isTributeVirus()) ? 11.0f : 12.0f, juce::Font::bold));
    badgeLabel.setColour (juce::Label::textColourId, (isTribute303() || isTribute909() || isTributeVirus()) ? theme.legend : theme.accent);
    addAndMakeVisible (badgeLabel);

    titleLabel.setText (theme.title, juce::dontSendNotification);
    titleLabel.setJustificationType (juce::Justification::centredLeft);
    titleLabel.setFont (juce::Font (isTribute303() ? 30.0f : (isTribute909() ? 28.0f : (isTributeVirus() ? 32.0f : 28.0f)), juce::Font::bold));
    titleLabel.setColour (juce::Label::textColourId, (isTribute303() || isTribute909()) ? theme.panelEdge : theme.text);
    addAndMakeVisible (titleLabel);

    subtitleLabel.setText (theme.subtitle, juce::dontSendNotification);
    subtitleLabel.setJustificationType (juce::Justification::centredLeft);
    subtitleLabel.setFont (juce::Font (13.0f, juce::Font::plain));
    subtitleLabel.setColour (juce::Label::textColourId,
                             isTribute303() ? theme.trim.darker (0.2f)
                                            : ((isTribute909() || isTributeVirus()) ? theme.panelEdge.brighter (0.18f) : theme.hint));
    addAndMakeVisible (subtitleLabel);

    for (const auto& spec : buildChoiceSpecs())
    {
        auto* card = choiceCards.add (new ChoiceCard (theme, spec));
        for (int index = 0; index < spec.values.size(); ++index)
            card->combo.addItem (spec.values[index], index + 1);
        comboAttachments.push_back (std::make_unique<juce::AudioProcessorValueTreeState::ComboBoxAttachment> (audioProcessor.apvts, spec.paramId, card->combo));
        addAndMakeVisible (card);
    }

    for (const auto& spec : buildKnobSpecs())
    {
        auto* card = knobCards.add (new KnobCard (lookAndFeel, spec.title, spec.hint,
                                                  usesDrumPadLayout() || isTributeVirus(),
                                                  spec.toggleText));
        if (spec.previewMidiNote >= 0)
        {
            card->slider.onPreviewRequested = [this, midiNote = spec.previewMidiNote]
            {
                audioProcessor.previewDrumPad (midiNote);
            };
        }
        sliderAttachments.push_back (std::make_unique<juce::AudioProcessorValueTreeState::SliderAttachment> (audioProcessor.apvts, spec.paramId, card->slider));
        if (spec.toggleParamId.isNotEmpty())
        {
            if (auto* toggleButton = card->getToggleButton())
                buttonAttachments.push_back (std::make_unique<juce::AudioProcessorValueTreeState::ButtonAttachment> (audioProcessor.apvts, spec.toggleParamId, *toggleButton));
        }
        addAndMakeVisible (card);
    }

    if (usesDrumPadLayout() && ! isTribute909())
    {
        for (const auto& spec : buildDrumPadSpecs())
        {
            auto* pad = drumPads.add (new DrumPad (lookAndFeel, spec.title, spec.note));
            pad->onPreviewRequested = [this, midiNote = spec.midiNote]
            {
                audioProcessor.previewDrumPad (midiNote);
            };
            pad->slider.onPreviewRequested = pad->onPreviewRequested;
            sliderAttachments.push_back (std::make_unique<juce::AudioProcessorValueTreeState::SliderAttachment> (audioProcessor.apvts, spec.paramId, pad->slider));
            addAndMakeVisible (pad);
        }
    }
}

bool AdvancedVSTiAudioProcessorEditor::isTribute303() const noexcept
{
    return theme.tribute303;
}

bool AdvancedVSTiAudioProcessorEditor::isTribute909() const noexcept
{
    return theme.tribute909;
}

bool AdvancedVSTiAudioProcessorEditor::isTributeVirus() const noexcept
{
    return theme.tributeVirus;
}

bool AdvancedVSTiAudioProcessorEditor::usesDrumPadLayout() const noexcept
{
    const auto name = normalizedPluginName (audioProcessor);
    return name.contains ("drum") || name.contains ("808");
}

AdvancedVSTiAudioProcessorEditor::Theme AdvancedVSTiAudioProcessorEditor::buildTheme() const
{
    const auto name = normalizedPluginName (audioProcessor);

    if (name.contains ("808"))
    {
        Theme tribute {
            "AI 808 Machine",
            "Retro drum macros on a lighter rhythm-composer faceplate with longer tails and deeper low-end.",
            juce::Colour::fromRGB (245, 138, 57), juce::Colour::fromRGB (255, 197, 136),
            juce::Colour::fromRGB (233, 231, 224), juce::Colour::fromRGB (227, 225, 216), juce::Colour::fromRGB (84, 88, 86),
            juce::Colour::fromRGB (37, 40, 41), juce::Colour::fromRGB (117, 118, 112)
        };
        tribute.faceplate = juce::Colour::fromRGB (236, 234, 227);
        tribute.trim = juce::Colour::fromRGB (123, 127, 122);
        tribute.legend = juce::Colour::fromRGB (235, 132, 48);
        tribute.knobBody = juce::Colour::fromRGB (54, 56, 58);
        tribute.knobCap = juce::Colour::fromRGB (82, 82, 79);
        tribute.tribute909 = true;
        return tribute;
    }

    if (name.contains ("303"))
    {
        Theme tribute {
            "AI TB303 Tribute",
            "A respectful nod to the silver box: squelchy cutoff, snappy decay, and that unmistakable acid front panel.",
            juce::Colour::fromRGB (255, 116, 38), juce::Colour::fromRGB (255, 190, 86),
            juce::Colour::fromRGB (44, 46, 48), juce::Colour::fromRGB (221, 217, 204), juce::Colour::fromRGB (145, 87, 58),
            juce::Colour::fromRGB (36, 34, 32), juce::Colour::fromRGB (119, 94, 73)
        };
        tribute.faceplate = juce::Colour::fromRGB (220, 216, 202);
        tribute.trim = juce::Colour::fromRGB (133, 82, 56);
        tribute.legend = juce::Colour::fromRGB (195, 83, 37);
        tribute.knobBody = juce::Colour::fromRGB (235, 232, 224);
        tribute.knobCap = juce::Colour::fromRGB (56, 56, 58);
        tribute.tribute303 = true;
        return tribute;
    }

    if (name.contains ("drum"))
    {
        Theme tribute {
            "AI Drum Machine",
            "909-style punch, clap and hats on a brighter rhythm-composer tribute panel.",
            juce::Colour::fromRGB (244, 126, 42), juce::Colour::fromRGB (255, 211, 167),
            juce::Colour::fromRGB (234, 232, 225), juce::Colour::fromRGB (229, 227, 218), juce::Colour::fromRGB (83, 88, 86),
            juce::Colour::fromRGB (38, 42, 43), juce::Colour::fromRGB (118, 120, 114)
        };
        tribute.faceplate = juce::Colour::fromRGB (239, 237, 229);
        tribute.trim = juce::Colour::fromRGB (124, 128, 123);
        tribute.legend = juce::Colour::fromRGB (237, 128, 34);
        tribute.knobBody = juce::Colour::fromRGB (52, 54, 56);
        tribute.knobCap = juce::Colour::fromRGB (80, 80, 77);
        tribute.tribute909 = true;
        return tribute;
    }

    if (name.contains ("bass"))
        return { "AI Bass Synth", "Heavier subs, firmer harmonics, and a darker panel built for low-end shaping.",
                 juce::Colour::fromRGB (82, 231, 166), juce::Colour::fromRGB (24, 153, 111),
                 juce::Colour::fromRGB (14, 18, 18), juce::Colour::fromRGB (18, 28, 27), juce::Colour::fromRGB (30, 87, 71),
                 juce::Colours::white, juce::Colour::fromRGB (175, 212, 202) };

    if (name.contains ("string"))
        return { "AI String Synth", "Softer ensemble motion, longer blooms, and warmer control cards for layered pads.",
                 juce::Colour::fromRGB (255, 210, 98), juce::Colour::fromRGB (198, 142, 49),
                 juce::Colour::fromRGB (18, 18, 23), juce::Colour::fromRGB (28, 26, 23), juce::Colour::fromRGB (102, 75, 34),
                 juce::Colours::white, juce::Colour::fromRGB (216, 203, 176) };

    if (name.contains ("lead"))
        return { "AI Lead Synth", "Brighter attack, extra edge, and a cleaner lead workflow with performance-focused knobs.",
                 juce::Colour::fromRGB (255, 117, 124), juce::Colour::fromRGB (190, 62, 84),
                 juce::Colour::fromRGB (19, 17, 22), juce::Colour::fromRGB (32, 23, 28), juce::Colour::fromRGB (92, 47, 60),
                 juce::Colours::white, juce::Colour::fromRGB (214, 184, 192) };

    if (name.contains ("pad"))
        return { "AI Pad Synth", "Wider blooms, slower drift, and a softer ambient panel for beds and swells.",
                 juce::Colour::fromRGB (128, 205, 255), juce::Colour::fromRGB (49, 132, 190),
                 juce::Colour::fromRGB (16, 19, 24), juce::Colour::fromRGB (22, 28, 34), juce::Colour::fromRGB (42, 77, 101),
                 juce::Colours::white, juce::Colour::fromRGB (184, 207, 222) };

    if (name.contains ("pluck"))
        return { "AI Pluck Synth", "Shorter decay, crisp bite, and a compact front panel tuned for rhythmic hooks.",
                 juce::Colour::fromRGB (255, 184, 98), juce::Colour::fromRGB (204, 112, 38),
                 juce::Colour::fromRGB (21, 18, 18), juce::Colour::fromRGB (34, 27, 22), juce::Colour::fromRGB (104, 66, 36),
                 juce::Colours::white, juce::Colour::fromRGB (221, 199, 174) };

    if (name.contains ("sampler"))
        return { "AI Sampler", "Trim the playback window and shape the tone from a cleaner sample-first performance deck.",
                 juce::Colour::fromRGB (191, 144, 255), juce::Colour::fromRGB (126, 84, 205),
                 juce::Colour::fromRGB (16, 17, 24), juce::Colour::fromRGB (27, 23, 34), juce::Colour::fromRGB (78, 58, 114),
                 juce::Colours::white, juce::Colour::fromRGB (205, 193, 228) };

    Theme tribute {
        "AI Virus Synth",
        "Dual oscillators, dual filters, modulation, arp, and onboard FX in a darker flagship virtual-analog panel.",
        juce::Colour::fromRGB (145, 179, 214), juce::Colour::fromRGB (78, 112, 150),
        juce::Colour::fromRGB (8, 10, 14), juce::Colour::fromRGB (14, 18, 24), juce::Colour::fromRGB (58, 86, 112),
        juce::Colour::fromRGB (235, 238, 244), juce::Colour::fromRGB (156, 168, 189)
    };
    tribute.faceplate = juce::Colour::fromRGB (18, 21, 26);
    tribute.trim = juce::Colour::fromRGB (79, 96, 121);
    tribute.legend = juce::Colour::fromRGB (176, 194, 219);
    tribute.knobBody = juce::Colour::fromRGB (30, 33, 38);
    tribute.knobCap = juce::Colour::fromRGB (226, 229, 234);
    tribute.tributeVirus = true;
    return tribute;
}

std::vector<AdvancedVSTiAudioProcessorEditor::ChoiceSpec> AdvancedVSTiAudioProcessorEditor::buildChoiceSpecs() const
{
    const auto name = normalizedPluginName (audioProcessor);
    std::vector<ChoiceSpec> specs;

    const auto presetNames = audioProcessor.presetNames();
    if (! presetNames.isEmpty())
        specs.push_back ({ "PRESET", "Preset", isTributeVirus() ? "Performance memory" : "Classic bundled patch", presetNames, {}, false, 0 });

    if (isTributeVirus())
    {
        specs.push_back ({ "OSCTYPE", "Osc 1 Wave", "Primary waveform", { "Sine", "Saw", "Square", "Noise", "Sample" }, { "SIN", "SAW", "SQR", "NOI", "SMP" }, true, 5 });
        specs.push_back ({ "OSC2TYPE", "Osc 2 Wave", "Secondary waveform", { "Sine", "Saw", "Square", "Noise", "Sample" }, { "SIN", "SAW", "SQR", "NOI", "SMP" }, true, 5 });
        specs.push_back ({ "FILTERTYPE", "Filter 1 Mode", "Main response", { "LP", "BP", "HP", "Notch" }, { "LP", "BP", "HP", "NCH" }, true, 2 });
        specs.push_back ({ "FILTERSLOPE", "Filter 1 Slope", "Roll-off steepness", { "12 dB", "16 dB", "24 dB" }, { "12", "16", "24" }, true, 3 });
        specs.push_back ({ "FILTER2TYPE", "Filter 2 Mode", "Secondary response", { "LP", "BP", "HP", "Notch" }, { "LP", "BP", "HP", "NCH" }, true, 2 });
        specs.push_back ({ "FILTER2SLOPE", "Filter 2 Slope", "Secondary roll-off", { "12 dB", "16 dB", "24 dB" }, { "12", "16", "24" }, true, 3 });
        specs.push_back ({ "FXTYPE", "Effects", "Insert algorithm", { "Off", "Dist", "Phaser", "Chorus" }, { "OFF", "DST", "PHA", "CHO" }, true, 2 });
        specs.push_back ({ "LFO1SHAPE", "LFO 1 Shape", "Primary motion", { "Sine", "Saw", "Square" }, { "SIN", "SAW", "SQR" }, true, 3 });
        specs.push_back ({ "LFO2SHAPE", "LFO 2 Shape", "Secondary motion", { "Sine", "Saw", "Square" }, { "SIN", "SAW", "SQR" }, true, 3 });
        specs.push_back ({ "ARPMODE", "Arp Mode", "Pattern motion", { "Up", "Down", "UpDown", "Random" }, { "FWD", "REV", "ALT", "RND" }, true, 2 });
        return specs;
    }

    if (name.contains ("303"))
    {
        specs.push_back ({ "OSCTYPE", "Waveform", "Switch the acid core source", { "Sine", "Saw", "Square", "Noise", "Sample" }, {}, false, 0 });
        specs.push_back ({ "FILTERTYPE", "Filter Mode", "Tribute panel with modern filter options", { "LP", "BP", "HP", "Notch" }, {}, false, 0 });
        specs.push_back ({ "FILTERSLOPE", "Slope", "Acid roll-off steepness", { "12 dB", "16 dB", "24 dB" }, {}, false, 0 });
        return specs;
    }

    if (name.contains ("sampler"))
    {
        specs.push_back ({ "SAMPLEBANK", "Source", "Generated sample bank", { "Dusty Keys", "Tape Choir", "Velvet Pluck", "Vox Chop", "Sub Stab", "Glass Bell" }, {}, false, 0 });
        specs.push_back ({ "FILTERTYPE", "Filter", "Playback filter curve", { "LP", "BP", "HP", "Notch" }, {}, false, 0 });
        specs.push_back ({ "FILTERSLOPE", "Slope", "Playback roll-off", { "12 dB", "16 dB", "24 dB" }, {}, false, 0 });
        return specs;
    }

    if (name.contains ("drum") || name.contains ("808"))
    {
        specs.push_back ({ "FILTERTYPE", "Tone Filter", "Top-end response", { "Off", "LP", "BP", "HP", "Notch" }, {}, false, 0 });
        specs.push_back ({ "FILTERSLOPE", "Slope", "Filter steepness", { "12 dB", "16 dB", "24 dB" }, {}, false, 0 });
        return specs;
    }

    specs.push_back ({ "OSCTYPE", "Oscillator", "Core harmonic source", { "Sine", "Saw", "Square", "Noise", "Sample" }, {}, false, 0 });
    specs.push_back ({ "FILTERTYPE", "Filter", "Main timbre curve", { "LP", "BP", "HP", "Notch" }, {}, false, 0 });
    specs.push_back ({ "FILTERSLOPE", "Slope", "Filter roll-off", { "12 dB", "16 dB", "24 dB" }, {}, false, 0 });
    return specs;
}

std::vector<AdvancedVSTiAudioProcessorEditor::KnobSpec> AdvancedVSTiAudioProcessorEditor::buildKnobSpecs() const
{
    const auto name = normalizedPluginName (audioProcessor);

    if (isTributeVirus())
        return {
            { "MASTERLEVEL", "Master", "Output level" },
            { "UNISON", "Unison", "Voice stack" },
            { "DETUNE", "Spread", "Unison width" },
            { "LFO1RATE", "LFO 1 Rate", "Pitch motion" },
            { "LFO1PITCH", "LFO 1 Amt", "Pitch depth" },
            { "LFO2RATE", "LFO 2 Rate", "Filter motion" },
            { "LFO2FILTER", "LFO 2 Amt", "Filter depth" },

            { "OSC2SEMITONE", "Semitone", "Osc 2 tuning" },
            { "OSC2DETUNE", "Detune 2", "Fine tuning" },
            { "OSC2MIX", "Osc Mix", "Blend osc 2" },
            { "SUBOSCLEVEL", "Osc 3 / Sub", "Third osc support", -1, "OSC3ENABLE", "On" },
            { "NOISELEVEL", "Noise", "Noise bed" },
            { "RINGMOD", "Ring Mod", "Metallic edge", -1, "RINGMODENABLE", "On" },
            { "FMAMOUNT", "FM Amt", "Cross-mod bite", -1, "FMENABLE", "On" },
            { "SYNC", "Sync", "Hard sync", -1, "SYNCENABLE", "On" },
            { "OSCGATE", "Gate", "Note length" },
            { "ENVCURVE", "Env Curve", "Contour shape" },

            { "CUTOFF", "Cutoff 1", "Filter 1 freq" },
            { "RESONANCE", "Resonance", "Filter focus" },
            { "FILTERENVAMOUNT", "Env Amt", "Env to filter" },
            { "CUTOFF2", "Cutoff 2", "Filter 2 freq" },
            { "FILTERBALANCE", "Filt Bal", "1 to 2 mix" },
            { "FILTATTACK", "Flt Attack", "Envelope rise" },
            { "FILTDECAY", "Flt Decay", "Envelope fall" },
            { "FILTSUSTAIN", "Flt Sustain", "Envelope hold" },
            { "FILTRELEASE", "Flt Release", "Envelope tail" },

            { "AMPATTACK", "Amp Attack", "Level rise" },
            { "AMPDECAY", "Amp Decay", "Level fall" },
            { "AMPSUSTAIN", "Amp Sustain", "Level hold" },
            { "AMPRELEASE", "Amp Release", "Level tail" },

            { "ARPRATE", "Arp Rate", "Pattern speed" },
            { "RHYTHMGATE_RATE", "Gate Rate", "Rhythm speed" },
            { "RHYTHMGATE_DEPTH", "Gate Depth", "Rhythm amount" },

            { "FXMIX", "FX Mix", "Insert blend" },
            { "FXINTENSITY", "FX Int", "Insert depth" },
            { "DELAYSEND", "Delay Send", "Send level" },
            { "DELAYTIME", "Delay Time", "Echo space" },
            { "DELAYFEEDBACK", "Feedback", "Echo repeats" },
            { "REVERBMIX", "Reverb", "Room blend" },
        };

    if (name.contains ("808"))
        return {
            { "FMAMOUNT", "Accent", "", -1 },
            { "DRUMMASTERLEVEL", "Volume", "", -1 },
            { "CUTOFF", "Tone", "", -1 },
            { "FILTERENVAMOUNT", "Noise", "", -1 },

            { "DRUMTUNE_KICK", "Tune", "", 36 },
            { "DRUMLEVEL_KICK", "Level", "", 36 },
            { "DRUM_KICK_ATTACK", "Attack", "", 36 },
            { "DRUMDECAY_KICK", "Decay", "", 36 },

            { "DRUMTUNE_SNARE", "Tune", "", 38 },
            { "DRUMLEVEL_SNARE", "Level", "", 38 },
            { "DRUM_SNARE_TONE", "Tone", "", 38 },
            { "DRUM_SNARE_SNAPPY", "Snappy", "", 38 },

            { "DRUMTUNE_LOW_TOM", "Tune", "", 42 },
            { "DRUMLEVEL_LOW_TOM", "Level", "", 42 },
            { "DRUMDECAY_LOW_TOM", "Decay", "", 42 },

            { "DRUMTUNE_MID_TOM", "Tune", "", 43 },
            { "DRUMLEVEL_MID_TOM", "Level", "", 43 },
            { "DRUMDECAY_MID_TOM", "Decay", "", 43 },

            { "DRUMTUNE_HIGH_TOM", "Tune", "", 44 },
            { "DRUMLEVEL_HIGH_TOM", "Level", "", 44 },
            { "DRUMDECAY_HIGH_TOM", "Decay", "", 44 },

            { "DRUMLEVEL_RIM", "Rim", "", 37 },
            { "DRUMLEVEL_CLAP", "Clap", "", 39 },

            { "DRUMLEVEL_CLOSED_HAT", "CH Level", "", 40 },
            { "DRUMLEVEL_OPEN_HAT", "OH Level", "", 41 },
            { "DRUMDECAY_CLOSED_HAT", "CH Dec", "", 40 },
            { "DRUMDECAY_OPEN_HAT", "OH Dec", "", 41 },

            { "DRUMLEVEL_CRASH", "Crash Lv", "", 45 },
            { "DRUMLEVEL_RIDE", "Ride Lv", "", 46 },
            { "DRUMTUNE_CRASH", "Crash Tun", "", 45 },
            { "DRUMTUNE_RIDE", "Ride Tun", "", 46 },

            { "DRUMLEVEL_COWBELL", "Cowbell", "", 47 },
            { "DRUMLEVEL_CLAVE", "Clave", "", 48 },
            { "DRUMLEVEL_MARACA", "Maraca", "", 49 },
            { "DRUMLEVEL_PERC", "Perc", "", 50 },
        };

    if (name.contains ("303"))
        return {
            { "DETUNE", "Tune", "Fine tuning around the sweet spot" },
            { "CUTOFF", "Cutoff", "Main acid filter sweep" },
            { "RESONANCE", "Resonance", "Squelch and focus" },
            { "FILTERENVAMOUNT", "Env Mod", "Envelope push into the filter" },
            { "AMPDECAY", "Decay", "Short acidic note length" },
            { "FMAMOUNT", "Accent", "Extra bite and hit emphasis" },
            { "OSCGATE", "Slide", "Held length and glide-style overlap" },
        };

    if (name.contains ("drum"))
        return {
            { "FMAMOUNT", "Accent", "", -1 },
            { "DRUMMASTERLEVEL", "Volume", "", -1 },
            { "CUTOFF", "Tone", "", -1 },
            { "FILTERENVAMOUNT", "Noise", "", -1 },

            { "DRUMTUNE_KICK", "Tune", "", 36 },
            { "DRUMLEVEL_KICK", "Level", "", 36 },
            { "DRUM_KICK_ATTACK", "Attack", "", 36 },
            { "DRUMDECAY_KICK", "Decay", "", 36 },

            { "DRUMTUNE_SNARE", "Tune", "", 38 },
            { "DRUMLEVEL_SNARE", "Level", "", 38 },
            { "DRUM_SNARE_TONE", "Tone", "", 38 },
            { "DRUM_SNARE_SNAPPY", "Snappy", "", 38 },

            { "DRUMTUNE_LOW_TOM", "Tune", "", 42 },
            { "DRUMLEVEL_LOW_TOM", "Level", "", 42 },
            { "DRUMDECAY_LOW_TOM", "Decay", "", 42 },

            { "DRUMTUNE_MID_TOM", "Tune", "", 43 },
            { "DRUMLEVEL_MID_TOM", "Level", "", 43 },
            { "DRUMDECAY_MID_TOM", "Decay", "", 43 },

            { "DRUMTUNE_HIGH_TOM", "Tune", "", 44 },
            { "DRUMLEVEL_HIGH_TOM", "Level", "", 44 },
            { "DRUMDECAY_HIGH_TOM", "Decay", "", 44 },

            { "DRUMLEVEL_RIM", "Rim", "", 37 },
            { "DRUMLEVEL_CLAP", "Clap", "", 39 },

            { "DRUMLEVEL_CLOSED_HAT", "CH Level", "", 40 },
            { "DRUMLEVEL_OPEN_HAT", "OH Level", "", 41 },
            { "DRUMDECAY_CLOSED_HAT", "CH Dec", "", 40 },
            { "DRUMDECAY_OPEN_HAT", "OH Dec", "", 41 },

            { "DRUMLEVEL_CRASH", "Crash Lv", "", 45 },
            { "DRUMLEVEL_RIDE", "Ride Lv", "", 46 },
            { "DRUMTUNE_CRASH", "Crash Tun", "", 45 },
            { "DRUMTUNE_RIDE", "Ride Tun", "", 46 },

            { "DRUMLEVEL_COWBELL", "Cowbell", "", 47 },
            { "DRUMLEVEL_CLAVE", "Clave", "", 48 },
            { "DRUMLEVEL_MARACA", "Maraca", "", 49 },
            { "DRUMLEVEL_PERC", "Perc", "", 50 },
        };

    if (name.contains ("bass"))
        return {
            { "CUTOFF", "Cutoff", "Low-end brightness" },
            { "RESONANCE", "Resonance", "Focused filter edge" },
            { "FILTERENVAMOUNT", "Env Mod", "Envelope filter push" },
            { "FMAMOUNT", "Growl", "Extra harmonic bite" },
            { "AMPRELEASE", "Release", "Tail and sustain feel" },
        };

    if (name.contains ("string") || name.contains ("pad"))
        return {
            { "CUTOFF", "Cutoff", "Brightness and air" },
            { "AMPATTACK", "Attack", "Slow onset" },
            { "AMPRELEASE", "Release", "Tail length" },
            { "DETUNE", "Detune", "Ensemble spread" },
            { "FILTERENVAMOUNT", "Bloom", "Envelope swell depth" },
        };

    if (name.contains ("lead") || name.contains ("pluck"))
        return {
            { "CUTOFF", "Cutoff", "Top-end bite" },
            { "RESONANCE", "Resonance", "Focused edge" },
            { "FMAMOUNT", "Bite", "Extra harmonic attack" },
            { "AMPDECAY", "Decay", "Hook length" },
            { "FILTERENVAMOUNT", "Env Mod", "Filter snap" },
        };

    if (name.contains ("sampler"))
        return {
            { "SAMPLESTART", "Start", "Playback start point" },
            { "SAMPLEEND", "End", "Playback end point" },
            { "CUTOFF", "Cutoff", "Tone shaping" },
            { "AMPRELEASE", "Release", "Tail length" },
        };

    return {
        { "DETUNE", "Detune", "Unison spread" },
        { "FMAMOUNT", "FM", "Frequency modulation amount" },
        { "SYNC", "Sync", "Oscillator sync intensity" },
        { "CUTOFF", "Cutoff", "Filter brightness" },
        { "RESONANCE", "Resonance", "Filter focus" },
        { "FILTERENVAMOUNT", "Env Mod", "Envelope filter amount" },
        { "LFO1RATE", "LFO 1", "Primary movement speed" },
        { "LFO2RATE", "LFO 2", "Secondary movement speed" },
    };
}

std::vector<AdvancedVSTiAudioProcessorEditor::DrumPadSpec> AdvancedVSTiAudioProcessorEditor::buildDrumPadSpecs() const
{
    return {
        { "DRUMLEVEL_KICK", "Bass Drum", "C1", 36 },
        { "DRUMLEVEL_RIM", "Rim Shot", "C#1", 37 },
        { "DRUMLEVEL_SNARE", "Snare", "D1", 38 },
        { "DRUMLEVEL_CLAP", "Clap", "D#1", 39 },
        { "DRUMLEVEL_CLOSED_HAT", "Closed Hat", "E1", 40 },
        { "DRUMLEVEL_OPEN_HAT", "Open Hat", "F1", 41 },
        { "DRUMLEVEL_LOW_TOM", "Low Tom", "F#1", 42 },
        { "DRUMLEVEL_MID_TOM", "Mid Tom", "G1", 43 },
        { "DRUMLEVEL_HIGH_TOM", "High Tom", "G#1", 44 },
        { "DRUMLEVEL_CRASH", "Crash", "A1", 45 },
        { "DRUMLEVEL_RIDE", "Ride", "A#1", 46 },
        { "DRUMLEVEL_COWBELL", "Cowbell", "B1", 47 },
        { "DRUMLEVEL_CLAVE", "Clave", "C2", 48 },
        { "DRUMLEVEL_MARACA", "Maraca", "C#2", 49 },
        { "DRUMLEVEL_PERC", "Perc", "D2", 50 },
    };
}

void AdvancedVSTiAudioProcessorEditor::paint (juce::Graphics& g)
{
    if (isTribute303())
    {
        const auto uiScale = juce::jlimit (0.75f, 1.6f, juce::jmin (getWidth() / 1020.0f, getHeight() / 430.0f));
        const auto s = [uiScale] (float value) { return scaledFloat (value, uiScale); };
        const auto si = [uiScale] (float value) { return scaledInt (value, uiScale); };
        auto bounds = getLocalBounds().toFloat();
        g.fillAll (theme.background);

        auto unit = bounds.reduced (s (14.0f));
        g.setColour (juce::Colours::black.withAlpha (0.28f));
        g.fillRoundedRectangle (unit.translated (0.0f, s (5.0f)), s (22.0f));

        g.setColour (theme.faceplate);
        g.fillRoundedRectangle (unit, s (22.0f));
        g.setColour (juce::Colours::white.withAlpha (0.65f));
        g.drawRoundedRectangle (unit.reduced (s (1.0f)), s (22.0f), juce::jmax (1.0f, s (1.0f)));
        g.setColour (theme.trim);
        g.drawRoundedRectangle (unit, s (22.0f), juce::jmax (1.0f, s (1.4f)));

        auto topStrip = unit.removeFromTop (s (52.0f));
        g.setColour (theme.accent);
        g.fillRoundedRectangle (topStrip.reduced (s (12.0f), s (10.0f)), s (10.0f));
        g.setColour (juce::Colours::black.withAlpha (0.22f));
        g.drawRoundedRectangle (topStrip.reduced (s (12.0f), s (10.0f)), s (10.0f), juce::jmax (1.0f, s (1.0f)));

        g.setColour (theme.legend);
        g.setFont (juce::Font (11.5f * uiScale, juce::Font::bold));
        g.drawFittedText ("COMPUTER CONTROLLED - ACID BASS LINE SYNTHESIZER",
                          getLocalBounds().reduced (si (32.0f), si (20.0f)).removeFromTop (si (22.0f)),
                          juce::Justification::centredRight,
                          1);

        auto buttonRow = getLocalBounds().reduced (si (32.0f), si (26.0f)).removeFromBottom (si (30.0f)).toFloat();
        const float buttonWidth = s (28.0f);
        const float gap = s (7.0f);
        const float totalWidth = (buttonWidth * 16.0f) + (gap * 15.0f);
        float x = buttonRow.getCentreX() - (totalWidth * 0.5f);
        for (int index = 0; index < 16; ++index)
        {
            juce::Rectangle<float> step (x, buttonRow.getY(), buttonWidth, s (18.0f));
            g.setColour ((index % 4 == 0) ? theme.accent.withAlpha (0.88f) : theme.trim.withAlpha (0.65f));
            g.fillRoundedRectangle (step, s (5.0f));
            g.setColour (juce::Colours::black.withAlpha (0.25f));
            g.drawRoundedRectangle (step, s (5.0f), juce::jmax (1.0f, s (1.0f)));
            x += buttonWidth + gap;
        }
        return;
    }

    if (isTribute909())
    {
        const auto uiScale = juce::jlimit (0.72f, 1.45f, juce::jmin (getWidth() / 1120.0f, getHeight() / 640.0f));
        const auto s = [uiScale] (float value) { return scaledFloat (value, uiScale); };
        const auto si = [uiScale] (float value) { return scaledInt (value, uiScale); };
        auto bounds = getLocalBounds().toFloat();
        juce::ColourGradient faceplate (theme.faceplate.brighter (0.03f), bounds.getTopLeft(),
                                        theme.faceplate.darker (0.08f), bounds.getBottomRight(), false);
        g.setGradientFill (faceplate);
        g.fillAll();

        g.setColour (juce::Colours::white.withAlpha (0.38f));
        g.drawRect (getLocalBounds().reduced (1), 1);

        auto frame = bounds.reduced (s (12.0f));
        g.setColour (theme.trim.withAlpha (0.8f));
        g.drawRoundedRectangle (frame, s (10.0f), juce::jmax (1.0f, s (1.0f)));

        auto hero = frame.removeFromTop (s (82.0f));
        auto titleStrip = hero.removeFromTop (s (38.0f));
        g.setColour (theme.legend);
        g.setFont (juce::Font (28.0f * uiScale, juce::Font::bold));
        g.drawFittedText ("TR-909", titleStrip.toNearestInt().removeFromLeft (si (220.0f)), juce::Justification::centredLeft, 1);
        g.setColour (theme.panelEdge);
        g.setFont (juce::Font (14.0f * uiScale, juce::Font::bold));
        g.drawFittedText ("RHYTHM COMPOSER", titleStrip.toNearestInt().removeFromRight (si (280.0f)), juce::Justification::centredRight, 1);

        auto drawSection = [&] (juce::Rectangle<float> area, const juce::String& title)
        {
            g.setColour (theme.faceplate.withAlpha (0.95f));
            g.fillRoundedRectangle (area, s (7.0f));
            g.setColour (theme.trim.withAlpha (0.9f));
            g.drawRoundedRectangle (area, s (7.0f), juce::jmax (1.0f, s (1.0f)));
            const auto titleBounds = area.toNearestInt().reduced (si (8.0f), si (2.0f)).removeFromTop (si (14.0f));
            auto header = area.removeFromTop (s (18.0f));
            g.setColour (theme.panelEdge);
            g.fillRoundedRectangle (header, s (5.0f));
            g.setColour (theme.accent.withAlpha (0.92f));
            g.fillRect (header.removeFromBottom (s (2.0f)));
            g.setColour (theme.legend);
            g.setFont (juce::Font (8.2f * uiScale, juce::Font::bold));
            g.drawFittedText (title, titleBounds, juce::Justification::centredLeft, 1);
        };

        auto surface = frame.reduced (s (10.0f));
        surface.removeFromTop (s (70.0f));
        auto bottomDecor = surface.removeFromBottom (s (82.0f));
        auto topControls = surface;
        auto leftRail = topControls.removeFromLeft (s (180.0f));
        topControls.removeFromLeft (s (6.0f));

        const float gap = s (6.0f);
        const float ratioSum = 1.55f + 1.55f + 1.15f + 1.15f + 1.15f + 0.95f + 1.55f + 1.55f;
        const float unit = (topControls.getWidth() - gap * 7.0f) / ratioSum;
        auto kickArea = topControls.removeFromLeft (unit * 1.55f);
        topControls.removeFromLeft (gap);
        auto snareArea = topControls.removeFromLeft (unit * 1.55f);
        topControls.removeFromLeft (gap);
        auto lowTomArea = topControls.removeFromLeft (unit * 1.15f);
        topControls.removeFromLeft (gap);
        auto midTomArea = topControls.removeFromLeft (unit * 1.15f);
        topControls.removeFromLeft (gap);
        auto highTomArea = topControls.removeFromLeft (unit * 1.15f);
        topControls.removeFromLeft (gap);
        auto rimClapArea = topControls.removeFromLeft (unit * 0.95f);
        topControls.removeFromLeft (gap);
        auto hatArea = topControls.removeFromLeft (unit * 1.55f);
        topControls.removeFromLeft (gap);
        auto cymbalArea = topControls;

        drawSection (leftRail, "MASTER");
        drawSection (kickArea, "BASS DRUM");
        drawSection (snareArea, "SNARE DRUM");
        drawSection (lowTomArea, "LOW TOM");
        drawSection (midTomArea, "MID TOM");
        drawSection (highTomArea, "HI TOM");
        drawSection (rimClapArea, "RIM / CLAP");
        drawSection (hatArea, "HI HAT");
        drawSection (cymbalArea, "CYMBAL");
        drawSection (bottomDecor.removeFromLeft (juce::jmin (s (290.0f), bottomDecor.getWidth() * 0.31f)), "PRESET / FILTER");
        drawSection (bottomDecor, "PERCUSSION LEVELS");

        auto stepStrip = frame.removeFromBottom (s (34.0f)).reduced (s (10.0f), s (5.0f));
        const float buttonWidth = (stepStrip.getWidth() - 15.0f * s (5.0f)) / 16.0f;
        float stepX = stepStrip.getX();
        for (int index = 0; index < 16; ++index)
        {
            juce::Rectangle<float> step (stepX, stepStrip.getY(), buttonWidth, stepStrip.getHeight());
            g.setColour (theme.faceplate.brighter (0.06f));
            g.fillRoundedRectangle (step, s (4.0f));
            g.setColour (theme.trim.withAlpha (0.9f));
            g.drawRoundedRectangle (step, s (4.0f), juce::jmax (1.0f, s (1.0f)));
            g.setColour (theme.panelEdge);
            g.setFont (juce::Font (8.6f * uiScale, juce::Font::bold));
            g.drawFittedText (juce::String (index + 1), step.toNearestInt(), juce::Justification::centred, 1);
            stepX += buttonWidth + s (5.0f);
        }
        return;
    }

    if (isTributeVirus())
    {
        const auto uiScale = juce::jlimit (0.78f, 1.18f, juce::jmin (getWidth() / 1500.0f, getHeight() / 820.0f));
        const auto s = [uiScale] (float value) { return scaledFloat (value, uiScale); };
        const auto si = [uiScale] (float value) { return scaledInt (value, uiScale); };
        auto bounds = getLocalBounds().toFloat();
        juce::ColourGradient background (theme.background.brighter (0.08f), bounds.getTopLeft(),
                                         theme.background.darker (0.2f), bounds.getBottomRight(), false);
        g.setGradientFill (background);
        g.fillAll();

        auto chassis = bounds.reduced (s (16.0f));
        g.setColour (juce::Colours::black.withAlpha (0.3f));
        g.fillRoundedRectangle (chassis.translated (0.0f, s (5.0f)), s (18.0f));

        juce::ColourGradient faceplate (theme.faceplate.brighter (0.06f), chassis.getX(), chassis.getY(),
                                        theme.faceplate.darker (0.12f), chassis.getRight(), chassis.getBottom(), false);
        g.setGradientFill (faceplate);
        g.fillRoundedRectangle (chassis, s (18.0f));
        g.setColour (theme.trim.withAlpha (0.85f));
        g.drawRoundedRectangle (chassis, s (18.0f), juce::jmax (1.0f, s (1.2f)));
        g.setColour (juce::Colours::white.withAlpha (0.12f));
        g.drawRoundedRectangle (chassis.reduced (s (1.0f)), s (18.0f), juce::jmax (1.0f, s (1.0f)));

        auto content = chassis.reduced (s (18.0f));
        auto ioStrip = content.removeFromTop (s (22.0f));
        g.setColour (theme.legend);
        g.setFont (juce::Font (14.0f * uiScale, juce::Font::bold));
        g.drawFittedText ("ACCESS  VIRUS C", ioStrip.toNearestInt().removeFromLeft (si (280.0f)), juce::Justification::centredLeft, 1);
        g.setFont (juce::Font (8.8f * uiScale, juce::Font::bold));
        g.drawFittedText ("MIDI THRU   MIDI OUT   MIDI IN   L/PHONES   R/MONO   OUT 2   OUT 3   EXT IN   ON/OFF",
                          ioStrip.toNearestInt().reduced (si (4.0f), 0),
                          juce::Justification::centredRight,
                          1);

        content.removeFromTop (s (8.0f));
        auto layout = content;
        auto leftColumn = layout.removeFromLeft (juce::jmin<float> (s (330.0f), layout.getWidth() * 0.25f));
        layout.removeFromLeft (s (10.0f));
        auto rightColumn = layout.removeFromRight (juce::jmin<float> (s (470.0f), layout.getWidth() * 0.41f));
        layout.removeFromRight (s (10.0f));
        auto centreColumn = layout;

        auto heroArea = leftColumn.removeFromTop (s (108.0f));
        leftColumn.removeFromTop (s (10.0f));
        auto modArea = leftColumn.removeFromTop (juce::jmax<float> (s (248.0f), leftColumn.getHeight() * 0.49f));
        leftColumn.removeFromTop (s (10.0f));
        auto fxArea = leftColumn;

        auto drawSection = [&] (juce::Rectangle<float> area, const juce::String& title)
        {
            g.setColour (theme.panel.withAlpha (0.9f));
            g.fillRoundedRectangle (area, s (12.0f));
            g.setColour (theme.panelEdge.withAlpha (0.95f));
            g.drawRoundedRectangle (area, s (12.0f), juce::jmax (1.0f, s (1.0f)));
            g.setColour (theme.trim.withAlpha (0.8f));
            g.drawLine (area.getX() + s (12.0f), area.getY() + s (24.0f), area.getRight() - s (12.0f), area.getY() + s (24.0f), juce::jmax (1.0f, s (1.0f)));
            g.setColour (theme.legend);
            g.setFont (juce::Font (10.0f * uiScale, juce::Font::bold));
            g.drawFittedText (title, area.toNearestInt().reduced (si (12.0f), si (4.0f)).removeFromTop (si (18.0f)), juce::Justification::centredLeft, 1);
        };

        auto oscArea = centreColumn.removeFromTop (juce::jmax<float> (s (246.0f), centreColumn.getHeight() * 0.5f));
        centreColumn.removeFromTop (s (10.0f));
        auto displayArea = centreColumn.removeFromTop (s (128.0f));
        centreColumn.removeFromTop (s (10.0f));
        auto arpArea = centreColumn;

        auto mixFilterArea = rightColumn.removeFromTop (juce::jmax<float> (s (308.0f), rightColumn.getHeight() * 0.58f));
        rightColumn.removeFromTop (s (10.0f));
        auto ampArea = rightColumn;
        auto mixArea = mixFilterArea.removeFromLeft (juce::jmin<float> (s (130.0f), mixFilterArea.getWidth() * 0.24f));
        mixFilterArea.removeFromLeft (s (10.0f));
        auto filterArea = mixFilterArea;

        drawSection (heroArea, "MAIN / MASTER");
        drawSection (modArea, "LFOS / MOD");
        drawSection (fxArea, "EFFECTS / DELAY");
        drawSection (oscArea, "OSCILLATORS");
        drawSection (displayArea, "PROGRAM / EDIT");
        drawSection (arpArea, "ARPEGGIATOR");
        drawSection (mixArea, "MIX");
        drawSection (filterArea, "FILTERS");
        drawSection (ampArea, "AMPLIFIER");

        auto displayScreen = displayArea.reduced (s (14.0f), s (16.0f));
        displayScreen.removeFromTop (s (18.0f));
        auto lcd = displayScreen.removeFromLeft (juce::jmin (s (250.0f), displayScreen.getWidth() * 0.62f)).reduced (s (6.0f), s (4.0f));
        g.setColour (juce::Colours::black.withAlpha (0.62f));
        g.fillRoundedRectangle (lcd, s (8.0f));
        g.setColour (theme.panelEdge);
        g.drawRoundedRectangle (lcd, s (8.0f), juce::jmax (1.0f, s (1.0f)));
        g.setColour (theme.accent.withAlpha (0.18f));
        g.fillRoundedRectangle (lcd.reduced (s (7.0f)), s (7.0f));
        g.setColour (theme.legend);
        g.setFont (juce::Font (11.0f * uiScale, juce::Font::bold));
        g.drawFittedText ("AI VIRUS ENGINE", lcd.toNearestInt().removeFromTop (si (18.0f)), juce::Justification::centred, 1);
        g.setColour (theme.text.withAlpha (0.78f));
        g.setFont (juce::Font (8.8f * uiScale, juce::Font::plain));
        g.drawFittedText ("OSC  |  FILTER  |  AMP  |  FX", lcd.toNearestInt().reduced (si (10.0f), si (20.0f)), juce::Justification::centred, 2);

        auto softSection = displayScreen.reduced (s (10.0f), s (10.0f));
        softSection.removeFromLeft (juce::jmin (s (258.0f), softSection.getWidth() * 0.6f));
        g.setColour (theme.legend);
        g.setFont (juce::Font (8.2f * uiScale, juce::Font::bold));
        g.drawFittedText ("SOFT KNOB 1", softSection.toNearestInt().removeFromTop (si (14.0f)).removeFromLeft (si (92.0f)), juce::Justification::centred, 1);
        g.drawFittedText ("SOFT KNOB 2", softSection.toNearestInt().removeFromTop (0).removeFromLeft (si (188.0f)).withTrimmedLeft (si (96.0f)), juce::Justification::centred, 1);

        auto bottomLegends = chassis.reduced (s (24.0f), s (18.0f)).removeFromBottom (s (18.0f));
        g.setColour (theme.panelEdge.brighter (0.15f));
        g.setFont (juce::Font (8.2f * uiScale, juce::Font::bold));
        g.drawFittedText ("ADVANCED SIMULATED ANALOG SYNTHESIZER", bottomLegends.toNearestInt(), juce::Justification::centredRight, 1);
        return;
    }

    auto bounds = getLocalBounds().toFloat();
    juce::ColourGradient background (theme.background.brighter (0.06f), bounds.getTopLeft(),
                                     theme.background.darker (0.24f), bounds.getBottomRight(), false);
    g.setGradientFill (background);
    g.fillAll();

    auto hero = bounds.removeFromTop (126.0f);
    juce::ColourGradient heroGlow (theme.accentGlow.withAlpha (0.34f), hero.getX() + 120.0f, hero.getY() + 18.0f,
                                   theme.background.withAlpha (0.0f), hero.getRight(), hero.getBottom(), true);
    g.setGradientFill (heroGlow);
    g.fillRoundedRectangle (hero.reduced (16.0f, 12.0f), 24.0f);

    g.setColour (theme.panelEdge);
    g.drawRoundedRectangle (hero.reduced (16.0f, 12.0f), 24.0f, 1.3f);

    g.setColour (theme.accent.withAlpha (0.6f));
    g.fillRoundedRectangle (juce::Rectangle<float> (24.0f, 108.0f, static_cast<float> (getWidth()) - 48.0f, 2.5f), 1.25f);
}

void AdvancedVSTiAudioProcessorEditor::resized()
{
    if (isTribute303())
    {
        const auto uiScale = juce::jlimit (0.75f, 1.6f, juce::jmin (getWidth() / 1020.0f, getHeight() / 430.0f));
        const auto s = [uiScale] (float value) { return scaledInt (value, uiScale); };
        badgeLabel.setFont (juce::Font (11.0f * uiScale, juce::Font::bold));
        titleLabel.setFont (juce::Font (30.0f * uiScale, juce::Font::bold));
        subtitleLabel.setFont (juce::Font (13.0f * uiScale, juce::Font::plain));
        for (auto* card : knobCards)
            card->setScale (uiScale);
        for (auto* card : choiceCards)
            card->setScale (uiScale);

        auto area = getLocalBounds().reduced (s (28.0f));
        auto hero = area.removeFromTop (s (84.0f));

        auto titleArea = hero.removeFromLeft (juce::jmin (s (460.0f), juce::jmax (s (340.0f), hero.getWidth() - s (220.0f))));
        badgeLabel.setBounds (titleArea.removeFromTop (s (18.0f)));
        titleLabel.setBounds (titleArea.removeFromTop (s (38.0f)));
        subtitleLabel.setBounds (titleArea.removeFromTop (s (28.0f)));

        if (! choiceCards.isEmpty())
        {
            auto choiceArea = hero.reduced (0, s (6.0f));
            const int spacing = s (10.0f);
            const int columns = juce::jmax (1, choiceCards.size());
            const int cardWidth = (choiceArea.getWidth() - (spacing * (columns - 1))) / columns;
            for (int index = 0; index < choiceCards.size(); ++index)
            {
                choiceCards[index]->setBounds (choiceArea.removeFromLeft (cardWidth));
                if (index + 1 < choiceCards.size())
                    choiceArea.removeFromLeft (spacing);
            }
        }

        area.removeFromTop (s (8.0f));
        const int spacing = s (10.0f);
        const int knobCount = knobCards.size();
        const int cardWidth = knobCount > 0 ? (area.getWidth() - ((knobCount - 1) * spacing)) / knobCount : area.getWidth();
        const int cardHeight = s (188.0f);
        int x = area.getX();
        for (auto* card : knobCards)
        {
            card->setBounds (x, area.getY(), cardWidth, cardHeight);
            x += cardWidth + spacing;
        }
        return;
    }

    if (isTribute909())
    {
        const auto uiScale = juce::jlimit (0.72f, 1.45f, juce::jmin (getWidth() / 1120.0f, getHeight() / 640.0f));
        const auto s = [uiScale] (float value) { return scaledInt (value, uiScale); };
        badgeLabel.setFont (juce::Font (11.0f * uiScale, juce::Font::bold));
        titleLabel.setFont (juce::Font (28.0f * uiScale, juce::Font::bold));
        subtitleLabel.setFont (juce::Font (13.0f * uiScale, juce::Font::plain));
        for (auto* card : knobCards)
            card->setScale (uiScale);
        for (auto* card : choiceCards)
            card->setScale (uiScale);

        auto frame = getLocalBounds().reduced (s (20.0f));
        auto hero = frame.removeFromTop (s (58.0f));
        badgeLabel.setBounds (hero.removeFromTop (s (12.0f)));
        titleLabel.setBounds (hero.removeFromTop (s (30.0f)));
        subtitleLabel.setBounds (hero.removeFromTop (s (18.0f)));

        auto surface = frame;
        auto stepStrip = surface.removeFromBottom (s (30.0f));
        juce::ignoreUnused (stepStrip);
        auto topControls = surface.removeFromTop (juce::jmax (s (210.0f), surface.getHeight() - s (74.0f)));
        surface.removeFromTop (s (6.0f));
        auto bottomDecor = surface;

        auto leftRail = topControls.removeFromLeft (s (180.0f));
        topControls.removeFromLeft (s (6.0f));

        const int gap = s (6.0f);
        const float ratioSum = 1.55f + 1.55f + 1.15f + 1.15f + 1.15f + 0.95f + 1.55f + 1.55f;
        const int unit = juce::roundToInt ((topControls.getWidth() - gap * 7) / ratioSum);
        auto kickArea = topControls.removeFromLeft (juce::roundToInt (unit * 1.55f));
        topControls.removeFromLeft (gap);
        auto snareArea = topControls.removeFromLeft (juce::roundToInt (unit * 1.55f));
        topControls.removeFromLeft (gap);
        auto lowTomArea = topControls.removeFromLeft (juce::roundToInt (unit * 1.15f));
        topControls.removeFromLeft (gap);
        auto midTomArea = topControls.removeFromLeft (juce::roundToInt (unit * 1.15f));
        topControls.removeFromLeft (gap);
        auto highTomArea = topControls.removeFromLeft (juce::roundToInt (unit * 1.15f));
        topControls.removeFromLeft (gap);
        auto rimClapArea = topControls.removeFromLeft (juce::roundToInt (unit * 0.95f));
        topControls.removeFromLeft (gap);
        auto hatArea = topControls.removeFromLeft (juce::roundToInt (unit * 1.55f));
        topControls.removeFromLeft (gap);
        auto cymbalArea = topControls;

        auto presetArea = bottomDecor.removeFromLeft (juce::jmin (s (290.0f), juce::roundToInt (bottomDecor.getWidth() * 0.31f)));
        bottomDecor.removeFromLeft (s (6.0f));
        auto extrasArea = bottomDecor;

        auto layoutGroup = [&] (juce::Rectangle<int> bounds, std::initializer_list<int> indices, int columns)
        {
            auto inner = bounds.reduced (s (4.0f));
            inner.removeFromTop (s (18.0f));
            const int count = static_cast<int> (indices.size());
            if (count <= 0 || columns <= 0)
                return;

            const int spacing = s (3.0f);
            const int rows = (count + columns - 1) / columns;
            const int cellWidth = (inner.getWidth() - spacing * (columns - 1)) / columns;
            const int cellHeight = (inner.getHeight() - spacing * (rows - 1)) / juce::jmax (1, rows);

            int localIndex = 0;
            for (int knobIndex : indices)
            {
                if (! juce::isPositiveAndBelow (knobIndex, knobCards.size()))
                    continue;

                const int row = localIndex / columns;
                const int column = localIndex % columns;
                int x = inner.getX() + column * (cellWidth + spacing);
                const int y = inner.getY() + row * (cellHeight + spacing);
                int width = cellWidth;

                const bool singleInLastRow = (count % columns) == 1 && row == rows - 1 && column == 0;
                if (singleInLastRow)
                    x = inner.getX() + (inner.getWidth() - cellWidth) / 2;

                knobCards[knobIndex]->setBounds (x, y, width, cellHeight);
                ++localIndex;
            }
        };

        layoutGroup (leftRail, { 0, 1, 2, 3 }, 2);
        layoutGroup (kickArea, { 4, 5, 6, 7 }, 2);
        layoutGroup (snareArea, { 8, 9, 10, 11 }, 2);
        layoutGroup (lowTomArea, { 12, 13, 14 }, 2);
        layoutGroup (midTomArea, { 15, 16, 17 }, 2);
        layoutGroup (highTomArea, { 18, 19, 20 }, 2);
        layoutGroup (rimClapArea, { 21, 22 }, 2);
        layoutGroup (hatArea, { 23, 24, 25, 26 }, 2);
        layoutGroup (cymbalArea, { 27, 28, 29, 30 }, 2);
        layoutGroup (extrasArea, { 31, 32, 33, 34 }, 4);

        if (choiceCards.size() >= 1)
        {
            auto choices = presetArea.reduced (s (6.0f));
            const int spacing = s (6.0f);
            const int cardHeight = (choices.getHeight() - spacing) / 2;
            choiceCards[0]->setBounds (choices.removeFromTop (cardHeight));
            if (choiceCards.size() >= 2)
            {
                choices.removeFromTop (spacing);
                choiceCards[1]->setBounds (choices.removeFromTop (cardHeight));
            }
        }

        for (auto* pad : drumPads)
            pad->setBounds ({});

        return;
    }

    if (isTributeVirus())
    {
        const auto uiScale = juce::jlimit (0.78f, 1.18f, juce::jmin (getWidth() / 1500.0f, getHeight() / 820.0f));
        const auto s = [uiScale] (float value) { return scaledInt (value, uiScale); };
        badgeLabel.setFont (juce::Font (11.0f * uiScale, juce::Font::bold));
        titleLabel.setFont (juce::Font (32.0f * uiScale, juce::Font::bold));
        subtitleLabel.setFont (juce::Font (13.0f * uiScale, juce::Font::plain));
        for (auto* card : knobCards)
            card->setScale (uiScale);
        for (auto* card : choiceCards)
            card->setScale (uiScale);

        auto area = getLocalBounds().reduced (s (32.0f));
        area.removeFromTop (s (30.0f));

        auto layout = area;
        auto leftColumn = layout.removeFromLeft (juce::jmin<int> (s (330.0f), juce::roundToInt (layout.getWidth() * 0.25f)));
        layout.removeFromLeft (s (10.0f));
        auto rightColumn = layout.removeFromRight (juce::jmin<int> (s (470.0f), juce::roundToInt (layout.getWidth() * 0.41f)));
        layout.removeFromRight (s (10.0f));
        auto centreColumn = layout;

        auto heroArea = leftColumn.removeFromTop (s (108.0f));
        leftColumn.removeFromTop (s (10.0f));
        auto modArea = leftColumn.removeFromTop (juce::jmax<int> (s (248.0f), static_cast<int> (leftColumn.getHeight() * 0.49f)));
        leftColumn.removeFromTop (s (10.0f));
        auto fxArea = leftColumn;

        auto oscArea = centreColumn.removeFromTop (juce::jmax<int> (s (246.0f), static_cast<int> (centreColumn.getHeight() * 0.5f)));
        centreColumn.removeFromTop (s (10.0f));
        auto displayArea = centreColumn.removeFromTop (s (128.0f));
        centreColumn.removeFromTop (s (10.0f));
        auto arpArea = centreColumn;

        auto mixFilterArea = rightColumn.removeFromTop (juce::jmax<int> (s (308.0f), static_cast<int> (rightColumn.getHeight() * 0.58f)));
        rightColumn.removeFromTop (s (10.0f));
        auto ampArea = rightColumn;
        auto mixArea = mixFilterArea.removeFromLeft (juce::jmin<int> (s (130.0f), juce::roundToInt (mixFilterArea.getWidth() * 0.24f)));
        mixFilterArea.removeFromLeft (s (10.0f));
        auto filterArea = mixFilterArea;

        auto masterArea = heroArea.removeFromRight (s (128.0f)).reduced (s (8.0f), s (8.0f));
        auto titleArea = heroArea.reduced (s (12.0f), s (10.0f));
        badgeLabel.setBounds (titleArea.removeFromTop (s (18.0f)));
        titleLabel.setBounds (titleArea.removeFromTop (s (38.0f)));
        subtitleLabel.setBounds (titleArea.removeFromTop (s (26.0f)));

        auto placeChoice = [&] (int index, juce::Rectangle<int> bounds)
        {
            if (juce::isPositiveAndBelow (index, choiceCards.size()))
                choiceCards[index]->setBounds (bounds);
        };

        auto clearUnused = [&]
        {
            for (int index = 0; index < choiceCards.size(); ++index)
                choiceCards[index]->setBounds ({});
            for (int index = 0; index < knobCards.size(); ++index)
                knobCards[index]->setBounds ({});
        };
        clearUnused();

        auto layoutGrid = [&] (juce::Rectangle<int> bounds, std::initializer_list<int> indices, int columns, int maxWidth, int maxHeight)
        {
            if (indices.size() == 0 || columns <= 0)
                return;

            auto inner = bounds.reduced (s (10.0f));
            const int spacing = s (8.0f);
            const int count = static_cast<int> (indices.size());
            const int rows = juce::jmax (1, (count + columns - 1) / columns);
            const int cellWidth = juce::jmin ((inner.getWidth() - spacing * (columns - 1)) / columns, s (static_cast<float> (maxWidth)));
            const int cellHeight = juce::jmin ((inner.getHeight() - spacing * (rows - 1)) / rows, s (static_cast<float> (maxHeight)));
            const int totalWidth = cellWidth * columns + spacing * (columns - 1);
            const int totalHeight = cellHeight * rows + spacing * (rows - 1);
            auto grid = juce::Rectangle<int> (totalWidth, totalHeight).withCentre (inner.getCentre());

            int localIndex = 0;
            for (int knobIndex : indices)
            {
                if (! juce::isPositiveAndBelow (knobIndex, knobCards.size()))
                    continue;

                const int row = localIndex / columns;
                const int column = localIndex % columns;
                knobCards[knobIndex]->setBounds (grid.getX() + column * (cellWidth + spacing),
                                                 grid.getY() + row * (cellHeight + spacing),
                                                 cellWidth,
                                                 cellHeight);
                ++localIndex;
            }
        };

        if (juce::isPositiveAndBelow (0, knobCards.size()))
            knobCards[0]->setBounds (masterArea);

        auto modChoices = modArea.reduced (s (10.0f));
        auto modChoiceHeight = s (68.0f);
        auto modChoiceWidth = (modChoices.getWidth() - s (8.0f)) / 2;
        placeChoice (8, modChoices.removeFromLeft (modChoiceWidth).removeFromTop (modChoiceHeight));
        modChoices.removeFromLeft (s (8.0f));
        placeChoice (9, modChoices.removeFromLeft (modChoiceWidth).removeFromTop (modChoiceHeight));
        modArea.removeFromTop (s (82.0f));
        layoutGrid (modArea, { 1, 2, 3, 4, 5, 6 }, 3, 108, 132);

        auto fxChoices = fxArea.reduced (s (10.0f));
        placeChoice (7, fxChoices.removeFromTop (s (84.0f)));
        fxArea.removeFromTop (s (92.0f));
        layoutGrid (fxArea, { 33, 34, 35, 36, 37, 38 }, 3, 108, 130);

        auto oscChoices = oscArea.reduced (s (10.0f));
        auto oscChoiceHeight = s (70.0f);
        auto oscChoiceWidth = (oscChoices.getWidth() - s (8.0f)) / 2;
        placeChoice (1, oscChoices.removeFromLeft (oscChoiceWidth).removeFromTop (oscChoiceHeight));
        oscChoices.removeFromLeft (s (8.0f));
        placeChoice (2, oscChoices.removeFromLeft (oscChoiceWidth).removeFromTop (oscChoiceHeight));
        oscArea.removeFromTop (s (84.0f));
        auto oscMain = oscArea;
        auto oscControls = oscMain.removeFromLeft (juce::jmax<int> (s (250.0f), static_cast<int> (oscMain.getWidth() * 0.58f)));
        oscMain.removeFromLeft (s (8.0f));
        auto oscModControls = oscMain;
        layoutGrid (oscControls, { 7, 8, 15, 16 }, 2, 112, 128);
        layoutGrid (oscModControls, { 13, 14 }, 2, 112, 128);

        layoutGrid (mixArea, { 9, 10, 11, 12 }, 1, 104, 104);

        auto filterChoices = filterArea.reduced (s (10.0f));
        auto filterChoiceHeight = s (76.0f);
        auto filterChoiceWidth = (filterChoices.getWidth() - s (8.0f)) / 2;
        placeChoice (3, filterChoices.removeFromLeft (filterChoiceWidth).removeFromTop (filterChoiceHeight));
        filterChoices.removeFromLeft (s (8.0f));
        placeChoice (5, filterChoices.removeFromLeft (filterChoiceWidth).removeFromTop (filterChoiceHeight));
        filterChoices.removeFromTop (s (8.0f));
        auto filterSlopeRow = filterChoices.removeFromTop (s (70.0f));
        placeChoice (4, filterSlopeRow.removeFromLeft (filterChoiceWidth));
        filterSlopeRow.removeFromLeft (s (8.0f));
        placeChoice (6, filterSlopeRow.removeFromLeft (filterChoiceWidth));
        filterArea.removeFromTop (s (162.0f));
        layoutGrid (filterArea, { 17, 18, 19, 20, 21, 22, 23, 24, 25 }, 3, 110, 120);

        layoutGrid (ampArea, { 26, 27, 28, 29 }, 4, 100, 126);

        auto presetArea = displayArea.reduced (s (12.0f));
        placeChoice (0, presetArea.removeFromLeft (juce::jmin<int> (s (260.0f), static_cast<int> (presetArea.getWidth() * 0.55f))));

        auto arpChoiceArea = arpArea.reduced (s (10.0f));
        placeChoice (10, arpChoiceArea.removeFromTop (s (84.0f)).removeFromLeft (juce::jmin<int> (s (190.0f), static_cast<int> (arpChoiceArea.getWidth() * 0.44f))));
        arpArea.removeFromTop (s (96.0f));
        layoutGrid (arpArea, { 30, 31, 32 }, 3, 112, 128);

        for (auto* pad : drumPads)
            pad->setBounds ({});
        return;
    }

    if (usesDrumPadLayout())
    {
        const auto uiScale = juce::jlimit (0.8f, 1.4f, juce::jmin (getWidth() / 1180.0f, getHeight() / 860.0f));
        badgeLabel.setFont (juce::Font (11.0f * uiScale, juce::Font::bold));
        titleLabel.setFont (juce::Font (28.0f * uiScale, juce::Font::bold));
        subtitleLabel.setFont (juce::Font (13.0f * uiScale, juce::Font::plain));
        for (auto* card : knobCards)
            card->setScale (uiScale);
        for (auto* card : choiceCards)
            card->setScale (uiScale);

        auto area = getLocalBounds().reduced (isTribute909() ? 26 : 22);
        auto hero = area.removeFromTop (isTribute909() ? 88 : 96);
        badgeLabel.setBounds (hero.removeFromTop (18));
        titleLabel.setBounds (hero.removeFromTop (isTribute909() ? 40 : 36));
        subtitleLabel.setBounds (hero.removeFromTop (36));
        area.removeFromTop (isTribute909() ? 18 : 10);

        auto controlBand = area.removeFromTop (isTribute909() ? 164 : 204);
        if (! choiceCards.isEmpty())
        {
            auto choiceArea = controlBand.removeFromLeft (isTribute909() ? juce::jmin (258, juce::jmax (228, getWidth() / 5))
                                                                         : juce::jmin (300, juce::jmax (250, getWidth() / 4)));
            const int spacing = isTribute909() ? 10 : 12;
            const int cardHeight = (choiceArea.getHeight() - (spacing * (choiceCards.size() - 1))) / juce::jmax (1, choiceCards.size());
            for (int index = 0; index < choiceCards.size(); ++index)
            {
                choiceCards[index]->setBounds (choiceArea.removeFromTop (cardHeight));
                if (index + 1 < choiceCards.size())
                    choiceArea.removeFromTop (spacing);
            }
            controlBand.removeFromLeft (isTribute909() ? 12 : 18);
        }

        if (! knobCards.isEmpty())
        {
            const int spacing = isTribute909() ? 10 : 14;
            const int cardWidth = (controlBand.getWidth() - (spacing * (knobCards.size() - 1))) / juce::jmax (1, knobCards.size());
            for (int index = 0; index < knobCards.size(); ++index)
            {
                knobCards[index]->setBounds (controlBand.removeFromLeft (cardWidth));
                if (index + 1 < knobCards.size())
                    controlBand.removeFromLeft (spacing);
            }
        }

        area.removeFromTop (isTribute909() ? 14 : 18);

        auto padArea = area;
        const int spacing = isTribute909() ? 10 : 14;
        const int columns = isTribute909() ? 5 : 4;
        const int rows = isTribute909() ? 3 : 4;
        const int padWidth = (padArea.getWidth() - ((columns - 1) * spacing)) / columns;
        const int padHeight = (padArea.getHeight() - ((rows - 1) * spacing)) / rows;

        int x = padArea.getX();
        int y = padArea.getY();
        int column = 0;
        for (auto* pad : drumPads)
        {
            pad->setBounds (x, y, padWidth, padHeight);
            ++column;
            if (column >= columns)
            {
                column = 0;
                x = padArea.getX();
                y += padHeight + spacing;
            }
            else
            {
                x += padWidth + spacing;
            }
        }
        return;
    }

    auto area = getLocalBounds().reduced (22);
    const auto uiScale = juce::jlimit (0.8f, 1.4f, juce::jmin (getWidth() / 1040.0f, getHeight() / 640.0f));
    badgeLabel.setFont (juce::Font (12.0f * uiScale, juce::Font::bold));
    titleLabel.setFont (juce::Font (28.0f * uiScale, juce::Font::bold));
    subtitleLabel.setFont (juce::Font (13.0f * uiScale, juce::Font::plain));
    for (auto* card : knobCards)
        card->setScale (uiScale);
    for (auto* card : choiceCards)
        card->setScale (uiScale);
    auto hero = area.removeFromTop (96);
    badgeLabel.setBounds (hero.removeFromTop (18));
    titleLabel.setBounds (hero.removeFromTop (36));
    subtitleLabel.setBounds (hero.removeFromTop (36));
    area.removeFromTop (8);

    const int choiceHeight = choiceCards.isEmpty() ? 0 : 96;
    if (! choiceCards.isEmpty())
    {
        auto choiceArea = area.removeFromTop (choiceHeight);
        const int spacing = 14;
        const int columns = juce::jmax (1, choiceCards.size());
        const int cardWidth = (choiceArea.getWidth() - (spacing * (columns - 1))) / columns;
        for (int index = 0; index < choiceCards.size(); ++index)
        {
            choiceCards[index]->setBounds (choiceArea.removeFromLeft (cardWidth));
            if (index + 1 < choiceCards.size())
                choiceArea.removeFromLeft (spacing);
        }
        area.removeFromTop (14);
    }

    const int spacing = 14;
    const int columnCount = area.getWidth() > 940 ? 4 : (area.getWidth() > 680 ? 3 : 2);
    const int cardWidth = (area.getWidth() - ((columnCount - 1) * spacing)) / columnCount;
    const int cardHeight = 220;
    int x = area.getX();
    int y = area.getY();
    int column = 0;

    for (auto* card : knobCards)
    {
        card->setBounds (x, y, cardWidth, cardHeight);
        ++column;
        if (column >= columnCount)
        {
            column = 0;
            x = area.getX();
            y += cardHeight + spacing;
        }
        else
        {
            x += cardWidth + spacing;
        }
    }
}
