#include "PluginEditor.h"

namespace
{
juce::String normalizedPluginName (const AdvancedVSTiAudioProcessor& processor)
{
    return processor.getName().trim().toLowerCase();
}
} // namespace

AdvancedVSTiAudioProcessorEditor::AccentLookAndFeel::AccentLookAndFeel (Theme themeToUse)
    : theme (std::move (themeToUse))
{
    setColour (juce::Slider::thumbColourId, theme.accent);
    setColour (juce::Slider::rotarySliderOutlineColourId, theme.panelEdge);
    setColour (juce::Slider::rotarySliderFillColourId, theme.accent);
    setColour (juce::ComboBox::backgroundColourId, theme.panel);
    setColour (juce::ComboBox::outlineColourId, theme.panelEdge);
    setColour (juce::ComboBox::textColourId, theme.text);
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
        const auto bounds = juce::Rectangle<float> (static_cast<float> (x), static_cast<float> (y), static_cast<float> (width), static_cast<float> (height)).reduced (10.0f);
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

    const auto bounds = juce::Rectangle<float> (static_cast<float> (x), static_cast<float> (y), static_cast<float> (width), static_cast<float> (height)).reduced (8.0f);
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
    const juce::String& hintText)
    : lookAndFeel (lf)
{
    titleLabel.setText (titleText, juce::dontSendNotification);
    titleLabel.setJustificationType (juce::Justification::centred);
    titleLabel.setFont (juce::Font (lf.theme.tribute303 ? 12.0f : 14.0f, juce::Font::bold));
    titleLabel.setColour (juce::Label::textColourId, lf.theme.tribute303 ? lf.theme.legend : lf.theme.text);
    addAndMakeVisible (titleLabel);

    hintLabel.setText (hintText, juce::dontSendNotification);
    hintLabel.setJustificationType (juce::Justification::centred);
    hintLabel.setFont (juce::Font (lf.theme.tribute303 ? 9.5f : 11.0f, juce::Font::plain));
    hintLabel.setColour (juce::Label::textColourId, lf.theme.hint);
    addAndMakeVisible (hintLabel);

    slider.setSliderStyle (juce::Slider::RotaryHorizontalVerticalDrag);
    slider.setTextBoxStyle (juce::Slider::TextBoxBelow, false, lf.theme.tribute303 ? 56 : 72, lf.theme.tribute303 ? 18 : 22);
    slider.setLookAndFeel (&lookAndFeel);
    slider.setColour (juce::Slider::textBoxTextColourId, lf.theme.text);
    slider.setColour (juce::Slider::textBoxBackgroundColourId, lf.theme.tribute303 ? lf.theme.faceplate.brighter (0.05f) : lf.theme.panel.darker (0.2f));
    slider.setColour (juce::Slider::textBoxOutlineColourId, lf.theme.tribute303 ? lf.theme.trim : lf.theme.panelEdge);
    slider.setColour (juce::Slider::textBoxHighlightColourId, lf.theme.accent.withAlpha (0.18f));
    addAndMakeVisible (slider);
}

void AdvancedVSTiAudioProcessorEditor::KnobCard::paint (juce::Graphics& g)
{
    auto area = getLocalBounds().toFloat().reduced (1.0f);
    if (lookAndFeel.theme.tribute303)
    {
        g.setColour (lookAndFeel.theme.faceplate.withAlpha (0.92f));
        g.fillRoundedRectangle (area, 12.0f);
        g.setColour (lookAndFeel.theme.trim.withAlpha (0.85f));
        g.drawRoundedRectangle (area, 12.0f, 1.1f);
        g.setColour (lookAndFeel.theme.accent.withAlpha (0.18f));
        g.fillRoundedRectangle (area.removeFromBottom (3.0f), 1.5f);
        return;
    }

    g.setColour (lookAndFeel.theme.panel.withAlpha (0.92f));
    g.fillRoundedRectangle (area, 18.0f);
    g.setColour (lookAndFeel.theme.panelEdge);
    g.drawRoundedRectangle (area, 18.0f, 1.2f);
}

void AdvancedVSTiAudioProcessorEditor::KnobCard::resized()
{
    auto area = getLocalBounds().reduced (lookAndFeel.theme.tribute303 ? 8 : 12);
    titleLabel.setBounds (area.removeFromTop (lookAndFeel.theme.tribute303 ? 18 : 22));
    hintLabel.setBounds (area.removeFromBottom (lookAndFeel.theme.tribute303 ? 16 : 20));
    slider.setBounds (area.reduced (0, lookAndFeel.theme.tribute303 ? 0 : 2));
}

AdvancedVSTiAudioProcessorEditor::ChoiceCard::ChoiceCard (
    Theme themeToUse,
    const juce::String& titleText,
    const juce::String& hintText)
    : theme (std::move (themeToUse))
{
    titleLabel.setText (titleText, juce::dontSendNotification);
    titleLabel.setJustificationType (juce::Justification::centredLeft);
    titleLabel.setFont (juce::Font (theme.tribute303 ? 12.0f : 13.0f, juce::Font::bold));
    titleLabel.setColour (juce::Label::textColourId, theme.tribute303 ? theme.legend : theme.text);
    addAndMakeVisible (titleLabel);

    hintLabel.setText (hintText, juce::dontSendNotification);
    hintLabel.setJustificationType (juce::Justification::centredLeft);
    hintLabel.setFont (juce::Font (theme.tribute303 ? 9.5f : 11.0f, juce::Font::plain));
    hintLabel.setColour (juce::Label::textColourId, theme.hint);
    addAndMakeVisible (hintLabel);

    combo.setColour (juce::ComboBox::backgroundColourId, theme.tribute303 ? theme.faceplate.brighter (0.06f) : theme.panel.brighter (0.08f));
    combo.setColour (juce::ComboBox::outlineColourId, theme.tribute303 ? theme.trim : theme.panelEdge);
    combo.setColour (juce::ComboBox::textColourId, theme.text);
    combo.setColour (juce::ComboBox::arrowColourId, theme.accent);
    addAndMakeVisible (combo);
}

void AdvancedVSTiAudioProcessorEditor::ChoiceCard::paint (juce::Graphics& g)
{
    auto area = getLocalBounds().toFloat().reduced (1.0f);
    if (theme.tribute303)
    {
        g.setColour (theme.faceplate.withAlpha (0.96f));
        g.fillRoundedRectangle (area, 10.0f);
        g.setColour (theme.trim);
        g.drawRoundedRectangle (area, 10.0f, 1.0f);
        g.setColour (theme.accent.withAlpha (0.88f));
        g.fillRoundedRectangle (area.removeFromLeft (6.0f), 6.0f);
        return;
    }

    g.setColour (theme.panel.withAlpha (0.92f));
    g.fillRoundedRectangle (area, 16.0f);
    g.setColour (theme.panelEdge);
    g.drawRoundedRectangle (area, 16.0f, 1.2f);
}

void AdvancedVSTiAudioProcessorEditor::ChoiceCard::resized()
{
    auto area = getLocalBounds().reduced (12);
    titleLabel.setBounds (area.removeFromTop (20));
    hintLabel.setBounds (area.removeFromTop (18));
    combo.setBounds (area.removeFromTop (32));
}

AdvancedVSTiAudioProcessorEditor::AdvancedVSTiAudioProcessorEditor (AdvancedVSTiAudioProcessor& p)
    : AudioProcessorEditor (&p),
      audioProcessor (p),
      theme (buildTheme()),
      lookAndFeel (theme)
{
    buildEditor();
    setSize (isTribute303() ? 1020 : 1040, isTribute303() ? 430 : 640);
}

void AdvancedVSTiAudioProcessorEditor::buildEditor()
{
    badgeLabel.setText (isTribute303() ? "ACID BASS LINE EMULATOR TRIBUTE" : audioProcessor.getName(), juce::dontSendNotification);
    badgeLabel.setJustificationType (juce::Justification::centredLeft);
    badgeLabel.setFont (juce::Font (isTribute303() ? 11.0f : 12.0f, juce::Font::bold));
    badgeLabel.setColour (juce::Label::textColourId, isTribute303() ? theme.legend : theme.accent);
    addAndMakeVisible (badgeLabel);

    titleLabel.setText (theme.title, juce::dontSendNotification);
    titleLabel.setJustificationType (juce::Justification::centredLeft);
    titleLabel.setFont (juce::Font (isTribute303() ? 30.0f : 28.0f, juce::Font::bold));
    titleLabel.setColour (juce::Label::textColourId, isTribute303() ? theme.trim : theme.text);
    addAndMakeVisible (titleLabel);

    subtitleLabel.setText (theme.subtitle, juce::dontSendNotification);
    subtitleLabel.setJustificationType (juce::Justification::centredLeft);
    subtitleLabel.setFont (juce::Font (13.0f, juce::Font::plain));
    subtitleLabel.setColour (juce::Label::textColourId, isTribute303() ? theme.trim.darker (0.2f) : theme.hint);
    addAndMakeVisible (subtitleLabel);

    for (const auto& spec : buildChoiceSpecs())
    {
        auto* card = choiceCards.add (new ChoiceCard (theme, spec.title, spec.hint));
        for (int index = 0; index < spec.values.size(); ++index)
            card->combo.addItem (spec.values[index], index + 1);
        comboAttachments.push_back (std::make_unique<juce::AudioProcessorValueTreeState::ComboBoxAttachment> (audioProcessor.apvts, spec.paramId, card->combo));
        addAndMakeVisible (card);
    }

    for (const auto& spec : buildKnobSpecs())
    {
        auto* card = knobCards.add (new KnobCard (lookAndFeel, spec.title, spec.hint));
        sliderAttachments.push_back (std::make_unique<juce::AudioProcessorValueTreeState::SliderAttachment> (audioProcessor.apvts, spec.paramId, card->slider));
        addAndMakeVisible (card);
    }
}

bool AdvancedVSTiAudioProcessorEditor::isTribute303() const noexcept
{
    return theme.tribute303;
}

AdvancedVSTiAudioProcessorEditor::Theme AdvancedVSTiAudioProcessorEditor::buildTheme() const
{
    const auto name = normalizedPluginName (audioProcessor);

    if (name.contains ("808"))
        return { "AI 808 Machine", "Subby analogue drums, long kicks, and softer metallic hats in a cleaner retro panel.",
                 juce::Colour::fromRGB (255, 164, 69), juce::Colour::fromRGB (255, 120, 36),
                 juce::Colour::fromRGB (17, 18, 21), juce::Colour::fromRGB (30, 24, 20), juce::Colour::fromRGB (96, 61, 34),
                 juce::Colours::white, juce::Colour::fromRGB (205, 184, 160) };

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
        return { "AI Drum Machine", "909-inspired punch, clap and hats with faster transient control and a sturdier drum front end.",
                 juce::Colour::fromRGB (255, 110, 72), juce::Colour::fromRGB (192, 58, 39),
                 juce::Colour::fromRGB (18, 18, 22), juce::Colour::fromRGB (31, 24, 26), juce::Colour::fromRGB (98, 52, 46),
                 juce::Colours::white, juce::Colour::fromRGB (214, 190, 185) };

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

    return { "AdvancedVSTi", "Modern bundled instrument controls with a cleaner rotary layout and faster parameter access.",
             juce::Colour::fromRGB (108, 184, 255), juce::Colour::fromRGB (63, 124, 192),
             juce::Colour::fromRGB (16, 18, 24), juce::Colour::fromRGB (23, 29, 36), juce::Colour::fromRGB (42, 77, 101),
             juce::Colours::white, juce::Colour::fromRGB (183, 199, 214) };
}

std::vector<AdvancedVSTiAudioProcessorEditor::ChoiceSpec> AdvancedVSTiAudioProcessorEditor::buildChoiceSpecs() const
{
    const auto name = normalizedPluginName (audioProcessor);

    if (name.contains ("303"))
        return {
            { "OSCTYPE", "Waveform", "Switch the acid core source", { "Sine", "Saw", "Square", "Noise", "Sample" } },
            { "FILTERTYPE", "Filter Mode", "Tribute panel with modern filter options", { "LP", "BP", "HP", "Notch" } },
        };

    if (name.contains ("sampler"))
        return {
            { "SAMPLEBANK", "Source", "Generated sample bank", { "Dusty Keys", "Tape Choir", "Velvet Pluck", "Vox Chop", "Sub Stab", "Glass Bell" } },
            { "FILTERTYPE", "Filter", "Playback filter curve", { "LP", "BP", "HP", "Notch" } },
        };

    if (name.contains ("drum") || name.contains ("808"))
        return {
            { "FILTERTYPE", "Tone Filter", "Top-end response", { "LP", "BP", "HP", "Notch" } },
        };

    return {
        { "OSCTYPE", "Oscillator", "Core harmonic source", { "Sine", "Saw", "Square", "Noise", "Sample" } },
        { "FILTERTYPE", "Filter", "Main timbre curve", { "LP", "BP", "HP", "Notch" } },
    };
}

std::vector<AdvancedVSTiAudioProcessorEditor::KnobSpec> AdvancedVSTiAudioProcessorEditor::buildKnobSpecs() const
{
    const auto name = normalizedPluginName (audioProcessor);

    if (name.contains ("808"))
        return {
            { "FMAMOUNT", "Punch", "Kick weight and beater intensity" },
            { "OSCGATE", "Decay", "Overall drum tail length" },
            { "CUTOFF", "Tone", "Brightness across hats and snare" },
            { "FILTERENVAMOUNT", "Noise", "Noise and grit balance" },
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
            { "FMAMOUNT", "Punch", "Kick hit and transient weight" },
            { "OSCGATE", "Decay", "Short or longer drum bodies" },
            { "CUTOFF", "Tone", "Brightness and hat sheen" },
            { "FILTERENVAMOUNT", "Noise", "Air and grit content" },
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

void AdvancedVSTiAudioProcessorEditor::paint (juce::Graphics& g)
{
    if (isTribute303())
    {
        auto bounds = getLocalBounds().toFloat();
        g.fillAll (theme.background);

        auto unit = bounds.reduced (14.0f);
        g.setColour (juce::Colours::black.withAlpha (0.28f));
        g.fillRoundedRectangle (unit.translated (0.0f, 5.0f), 22.0f);

        g.setColour (theme.faceplate);
        g.fillRoundedRectangle (unit, 22.0f);
        g.setColour (juce::Colours::white.withAlpha (0.65f));
        g.drawRoundedRectangle (unit.reduced (1.0f), 22.0f, 1.0f);
        g.setColour (theme.trim);
        g.drawRoundedRectangle (unit, 22.0f, 1.4f);

        auto topStrip = unit.removeFromTop (52.0f);
        g.setColour (theme.accent);
        g.fillRoundedRectangle (topStrip.reduced (12.0f, 10.0f), 10.0f);
        g.setColour (juce::Colours::black.withAlpha (0.22f));
        g.drawRoundedRectangle (topStrip.reduced (12.0f, 10.0f), 10.0f, 1.0f);

        g.setColour (theme.legend);
        g.setFont (juce::Font (11.5f, juce::Font::bold));
        g.drawFittedText ("COMPUTER CONTROLLED • ACID BASS LINE SYNTHESIZER", getLocalBounds().reduced (32, 20).removeFromTop (22), juce::Justification::centredRight, 1);

        auto buttonRow = getLocalBounds().reduced (32, 26).removeFromBottom (30).toFloat();
        const float buttonWidth = 28.0f;
        const float gap = 7.0f;
        const float totalWidth = (buttonWidth * 16.0f) + (gap * 15.0f);
        float x = buttonRow.getCentreX() - (totalWidth * 0.5f);
        for (int index = 0; index < 16; ++index)
        {
            juce::Rectangle<float> step (x, buttonRow.getY(), buttonWidth, 18.0f);
            g.setColour ((index % 4 == 0) ? theme.accent.withAlpha (0.88f) : theme.trim.withAlpha (0.65f));
            g.fillRoundedRectangle (step, 5.0f);
            g.setColour (juce::Colours::black.withAlpha (0.25f));
            g.drawRoundedRectangle (step, 5.0f, 1.0f);
            x += buttonWidth + gap;
        }
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
        auto area = getLocalBounds().reduced (28);
        auto hero = area.removeFromTop (84);

        auto titleArea = hero.removeFromLeft (juce::jmin (460, hero.getWidth() - 220));
        badgeLabel.setBounds (titleArea.removeFromTop (18));
        titleLabel.setBounds (titleArea.removeFromTop (38));
        subtitleLabel.setBounds (titleArea.removeFromTop (28));

        if (! choiceCards.isEmpty())
        {
            auto choiceArea = hero.reduced (0, 6);
            const int spacing = 10;
            const int cardWidth = (choiceArea.getWidth() - spacing) / 2;
            for (int index = 0; index < choiceCards.size(); ++index)
            {
                choiceCards[index]->setBounds (choiceArea.removeFromLeft (cardWidth));
                if (index + 1 < choiceCards.size())
                    choiceArea.removeFromLeft (spacing);
            }
        }

        area.removeFromTop (8);
        const int spacing = 10;
        const int knobCount = knobCards.size();
        const int cardWidth = knobCount > 0 ? (area.getWidth() - ((knobCount - 1) * spacing)) / knobCount : area.getWidth();
        const int cardHeight = 188;
        int x = area.getX();
        for (auto* card : knobCards)
        {
            card->setBounds (x, area.getY(), cardWidth, cardHeight);
            x += cardWidth + spacing;
        }
        return;
    }

    auto area = getLocalBounds().reduced (22);
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
