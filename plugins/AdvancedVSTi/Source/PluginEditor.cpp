#include "PluginEditor.h"
#include <BinaryData.h>

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

juce::String buildControlTooltip (const juce::String& title, const juce::String& hint)
{
    if (hint.isEmpty())
        return title;
    return title + "\n" + hint;
}

juce::String humanizeVirusControlKey (const juce::String& key)
{
    juce::String result;
    juce::juce_wchar previous = 0;

    for (int index = 0; index < key.length(); ++index)
    {
        const auto current = key[index];
        const bool currentUpper = juce::CharacterFunctions::isUpperCase (current);
        const bool currentDigit = juce::CharacterFunctions::isDigit (current);
        const bool previousLower = previous != 0 && juce::CharacterFunctions::isLowerCase (previous);
        const bool previousDigit = previous != 0 && juce::CharacterFunctions::isDigit (previous);

        if (index > 0 && ((currentUpper && (previousLower || previousDigit)) || (currentDigit && ! previousDigit)))
            result << ' ';

        result << current;
        previous = current;
    }

    const std::array<std::pair<const char*, const char*>, 12> replacements {{
        { "Arp", "ARP" },
        { "Osc", "OSC" },
        { "Lfo", "LFO" },
        { "Fx", "FX" },
        { "Fm", "FM" },
        { "Eq", "EQ" },
        { "Lp", "LP" },
        { "Hp", "HP" },
        { "Bp", "BP" },
        { "Bs", "BS" },
        { "Sync Panel", "Sync" },
        { "Mode Lp", "Mode LP" }
    }};

    for (const auto& [from, to] : replacements)
        result = result.replace (from, to, true);

    return result.trim();
}

constexpr int kVirusTemplateWidth = 1045;
constexpr int kVirusTemplateHeight = 452;
constexpr int kVirusKnobWidth = 50;
constexpr int kVirusKnobHeight = 62;
constexpr int kVirusChoiceHeight = 22;
constexpr int kVirusToggleSize = 16;
constexpr int kVirusSquareButtonSize = 33;
constexpr int kVirusKeyboardExtraHeight = 72;

const std::array<juce::String, 5> kVirusModLeftTargetLabels { "OSC 1", "OSC 2/3", "PW", "RESO", "FLT GAIN" };
const std::array<juce::String, 6> kVirusModRightTargetLabels { "CUTOFF 1", "CUTOFF 2", "SHAPE", "FM AMT", "PAN", "ASSIGN" };
const std::array<juce::String, 11> kVirusModDestinationLabels {
    "OSC 1", "OSC 2/3", "PW", "RESO", "FLT GAIN",
    "CUTOFF 1", "CUTOFF 2", "SHAPE", "FM AMT", "PAN", "ASSIGN"
};
const std::array<juce::String, 5> kVirusUpperFxLabels { "DELAY", "REVERB", "LOW EQ", "MID EQ", "HIGH EQ" };
const std::array<juce::String, 5> kVirusLowerFxLabels { "DISTORTION", "CHARACTER", "CHORUS", "PHASER", "OTHERS" };

struct VirusPresetDisplayInfo
{
    juce::String bankLabel;
    juce::String slotLabel;
    juce::String categoryCode;
    juce::String descriptorA;
    juce::String descriptorB;
    juce::String descriptorC;
};

VirusPresetDisplayInfo virusPresetDisplayInfo (const juce::String& presetName, int presetIndex, int totalPrograms)
{
    const auto normalized = presetName.toLowerCase();
    const auto bankLetter = static_cast<juce_wchar> ('A' + juce::jlimit (0, 25, presetIndex / 128));
    juce::ignoreUnused (totalPrograms);

    VirusPresetDisplayInfo info;
    info.bankLabel = "RAM-" + juce::String::charToString (bankLetter);
    info.slotLabel = juce::String ((presetIndex % 128) + 1).paddedLeft ('0', 3);

    if (normalized.contains ("pad"))
        return { info.bankLabel, info.slotLabel, "PAD", "ATMOS", "WARM", "WIDE" };
    if (normalized.contains ("lead"))
        return { info.bankLabel, info.slotLabel, "LEAD", "SOLO", "EDGE", "FOCUS" };
    if (normalized.contains ("bass"))
        return { info.bankLabel, info.slotLabel, "BASS", "SUB", "PUNCH", "DRIVE" };
    if (normalized.contains ("arp") || normalized.contains ("seq"))
        return { info.bankLabel, info.slotLabel, "SEQ", "MOTION", "RHYTHM", "STEP" };
    if (normalized.contains ("sweep") || normalized.contains ("ufo"))
        return { info.bankLabel, info.slotLabel, "FX", "SWEEP", "SPACE", "MOTION" };
    if (normalized.contains ("pluck") || normalized.contains ("zipp"))
        return { info.bankLabel, info.slotLabel, "PLUCK", "ATTACK", "BRIGHT", "SHORT" };
    if (normalized.contains ("square"))
        return { info.bankLabel, info.slotLabel, "CLASSIC", "SQUARE", "BRIGHT", "VINTAGE" };
    if (normalized.contains ("saw"))
        return { info.bankLabel, info.slotLabel, "CLASSIC", "SAW", "BRIGHT", "STACK" };
    if (normalized.contains ("fm"))
        return { info.bankLabel, info.slotLabel, "DIGI", "FM", "GLASS", "METAL" };
    if (normalized.contains ("poly"))
        return { info.bankLabel, info.slotLabel, "POLY", "STACK", "GLOSS", "WIDE" };
    if (normalized.contains ("mono"))
        return { info.bankLabel, info.slotLabel, "MONO", "FOCUS", "DRIVE", "LEGATO" };

    return { info.bankLabel, info.slotLabel, "SYNTH", "CUSTOM", "VIRUS", "USER" };
}

juce::String virusLfoEnableParamId (int index)
{
    switch (juce::jlimit (0, 2, index))
    {
        case 1: return "LFO2ENABLE";
        case 2: return "LFO3ENABLE";
        default: return "LFO1ENABLE";
    }
}

juce::String virusLfoShapeParamId (int index)
{
    switch (juce::jlimit (0, 2, index))
    {
        case 1: return "LFO2SHAPE";
        case 2: return "LFO3SHAPE";
        default: return "LFO1SHAPE";
    }
}

juce::String virusLfoEnvModeParamId (int index)
{
    switch (juce::jlimit (0, 2, index))
    {
        case 1: return "LFO2ENVMODE";
        case 2: return "LFO3ENVMODE";
        default: return "LFO1ENVMODE";
    }
}

juce::String virusLfoRateParamId (int index)
{
    switch (juce::jlimit (0, 2, index))
    {
        case 1: return "LFO2RATE";
        case 2: return "RHYTHMGATE_RATE";
        default: return "LFO1RATE";
    }
}

juce::String virusLfoAmountParamId (int index)
{
    switch (juce::jlimit (0, 2, index))
    {
        case 1: return "LFO2AMOUNT";
        case 2: return "LFO3AMOUNT";
        default: return "LFO1AMOUNT";
    }
}

juce::String virusLfoDestinationParamId (int index)
{
    switch (juce::jlimit (0, 2, index))
    {
        case 1: return "LFO2DEST";
        case 2: return "LFO3DEST";
        default: return "LFO1DEST";
    }
}

juce::String virusLfoAssignDestinationParamId (int index)
{
    switch (juce::jlimit (0, 2, index))
    {
        case 1: return "LFO2ASSIGNDEST";
        case 2: return "LFO3ASSIGNDEST";
        default: return "LFO1ASSIGNDEST";
    }
}

juce::String virusMatrixSourceParamId (int slotIndex)
{
    return "MATRIX" + juce::String (juce::jlimit (0, 5, slotIndex) + 1) + "SOURCE";
}

juce::String virusMatrixAmountParamId (int slotIndex, int targetIndex)
{
    return "MATRIX" + juce::String (juce::jlimit (0, 5, slotIndex) + 1)
           + "AMOUNT" + juce::String (juce::jlimit (0, 2, targetIndex) + 1);
}

juce::String virusMatrixDestinationParamId (int slotIndex, int targetIndex)
{
    return "MATRIX" + juce::String (juce::jlimit (0, 5, slotIndex) + 1)
           + "DEST" + juce::String (juce::jlimit (0, 2, targetIndex) + 1);
}

int virusCombinedDestinationIndex (const std::array<int, 3>& leftIndices,
                                   const std::array<int, 3>& rightIndices,
                                   int lfoIndex)
{
    const auto clampedIndex = juce::jlimit (0, 2, lfoIndex);
    const auto left = leftIndices[static_cast<size_t> (clampedIndex)];
    if (left >= 0)
        return juce::jlimit (0, static_cast<int> (kVirusModLeftTargetLabels.size()) - 1, left);

    const auto right = rightIndices[static_cast<size_t> (clampedIndex)];
    if (right >= 0)
        return static_cast<int> (kVirusModLeftTargetLabels.size())
               + juce::jlimit (0, static_cast<int> (kVirusModRightTargetLabels.size()) - 1, right);

    return 0;
}

void setVirusCombinedDestinationIndex (std::array<int, 3>& leftIndices,
                                       std::array<int, 3>& rightIndices,
                                       int lfoIndex,
                                       int destinationIndex)
{
    const auto clampedLfo = juce::jlimit (0, 2, lfoIndex);
    const auto clampedDestination = juce::jlimit (0, static_cast<int> (kVirusModDestinationLabels.size()) - 1, destinationIndex);

    if (clampedDestination < static_cast<int> (kVirusModLeftTargetLabels.size()))
    {
        leftIndices[static_cast<size_t> (clampedLfo)] = clampedDestination;
        rightIndices[static_cast<size_t> (clampedLfo)] = -1;
    }
    else
    {
        leftIndices[static_cast<size_t> (clampedLfo)] = -1;
        rightIndices[static_cast<size_t> (clampedLfo)]
            = clampedDestination - static_cast<int> (kVirusModLeftTargetLabels.size());
    }
}

juce::String parameterValueText (juce::AudioProcessorValueTreeState& apvts, const juce::String& paramId)
{
    if (auto* parameter = dynamic_cast<juce::RangedAudioParameter*> (apvts.getParameter (paramId)))
    {
        if (auto* choice = dynamic_cast<juce::AudioParameterChoice*> (parameter))
        {
            const auto choiceIndex = juce::jlimit (0, choice->choices.size() - 1, choice->getIndex());
            return choice->choices[choiceIndex];
        }

        if (auto* toggle = dynamic_cast<juce::AudioParameterBool*> (parameter))
            return toggle->get() ? "ON" : "OFF";

        auto text = parameter->getCurrentValueAsText();
        if (text.isEmpty())
            text = parameter->getText (parameter->getValue(), 24);
        return text;
    }

    return "--";
}

void setParameterActualValue (juce::AudioProcessorValueTreeState& apvts, const juce::String& paramId, float actualValue)
{
    if (auto* parameter = dynamic_cast<juce::RangedAudioParameter*> (apvts.getParameter (paramId)))
    {
        const auto legalValue = parameter->getNormalisableRange().snapToLegalValue (actualValue);
        parameter->beginChangeGesture();
        parameter->setValueNotifyingHost (parameter->convertTo0to1 (legalValue));
        parameter->endChangeGesture();
    }
}

struct VirusLegend
{
    juce::String primary;
    juce::String secondary;
    bool accentSecondary = false;
    int widthPad = 18;
    int yOffset = -2;
};

struct VirusGroupChrome
{
    juce::String title;
    juce::Rectangle<float> frame;
    juce::Rectangle<float> tab;
};

VirusLegend virusKnobLegend (int index)
{
    switch (index)
    {
        case 0:  return { "MASTER VOLUME", {}, false, 36, 6 };
        case 3:  return { "RATE", "LFO CONTOUR", true, 26, 5 };
        case 7:  return { "WAVE", "SHAPE", false, 12, 5 };
        case 15: return { "WAVE SELECT/PW", {}, false, 34, 5 };
        case 8:  return { "SEMITONE", "PORTAMENTO", true, 24, 5 };
        case 16: return { "DETUNE 2 / 3", "UNISON DETUNE", true, 30, 5 };
        case 13: return { "FM AMOUNT", "FM MODE", true, 22, -2 };
        case 9:  return { "OSC BALANCE", "PANORAMA", true, 26, 5 };
        case 10: return { "SUB OSC VOLUME", "OSC3 VOLUME", true, 30, -2 };
        case 12: return { "OSC VOLUME", "SATURATION TYPE", true, 30, -2 };
        case 11: return { "NOISE VOLUME", "RING MODULATOR", true, 30, -2 };
        case 17: return { "CUTOFF", {}, false, 16, 5 };
        case 18: return { "RESONANCE", "RESONANCE 2", true, 26, 5 };
        case 19: return { "ENV AMOUNT", "KEYFOLLOW", true, 24, 5 };
        case 21: return { "FILTER BALANCE", {}, false, 30, 5 };
        case 20: return { "CUTOFF 2", {}, false, 18, -2 };
        case 22: return { "ATTACK", {}, false, 16, -2 };
        case 23: return { "DECAY", {}, false, 16, -2 };
        case 24: return { "SUSTAIN", "SUSTAIN SLOPE", true, 18, -2 };
        case 25: return { "RELEASE", {}, false, 18, -2 };
        case 26: return { "ATTACK", "PATCH VOLUME", true, 18, -2 };
        case 27: return { "DECAY", {}, false, 16, -2 };
        case 28: return { "SUSTAIN", "SUSTAIN SLOPE", true, 18, -2 };
        case 29: return { "RELEASE", "TEMPO", true, 18, -2 };
        case 30: return { "VALUE 1", {}, false, 18, -2 };
        case 31: return { "VALUE 2", {}, false, 18, -2 };
        case 32: return { "VALUE 3", {}, false, 18, -2 };
        case 35: return { "SEND", "EQ GAIN", false, 10, -2 };
        case 36: return { "TIME/COLOR", "EQ FREQ", false, 18, -2 };
        case 37: return { "FEEDBACK/DAMPING", "EQ Q-FACTOR", false, 18, -2 };
        case 33: return { "TYPE/MIX", {}, false, 18, -2 };
        case 34: return { "INTENSITY", {}, false, 18, -2 };
        default: return {};
    }
}

VirusLegend virusButtonLegend (const juce::String& key)
{
    if (key == "arpEdit")         return { "EDIT", {}, false, 12, 1 };
    if (key == "arpOn")           return { "ARP ON", "HOLD", true, 16, 1 };
    if (key == "matrixSelect")    return { "SELECT", {}, false, 14, 1 };
    if (key == "modEdit")         return { "EDIT", {}, false, 12, 1 };
    if (key == "modEnvMode")      return { "ENV MODE", {}, false, 20, 1 };
    if (key == "modShape")        return { "SHAPE", {}, false, 16, 1 };
    if (key == "modSelect")       return { "SELECT", {}, false, 14, 1 };
    if (key == "modAssign")       return { "SELECT", {}, false, 14, 1 };
    if (key == "modSelectRight")  return { "SELECT", {}, false, 14, 1 };
    if (key == "oscSelect")       return { "SELECT", {}, false, 14, 1 };
    if (key == "oscEdit")         return { "EDIT", {}, false, 12, 1 };
    if (key == "osc3On")          return { "OSC3 ON", {}, false, 18, 1 };
    if (key == "mono")            return { "MONO", "PANIC", true, 14, 1 };
    if (key == "syncPanel")       return { "SYNC", {}, false, 14, 1 };
    if (key == "fxSelectA" || key == "fxSelectB") return { "SELECT", {}, false, 14, 1 };
    if (key == "fxEditA" || key == "fxEditB")     return { "EDIT", {}, false, 12, 1 };
    if (key == "displayExit")     return { "EXIT", {}, false, 12, 1 };
    if (key == "displayTap")      return { "TAP", {}, false, 12, 1 };
    if (key == "displayEdit")     return { "EDIT", "MULTI EDIT", true, 14, 1 };
    if (key == "displayConfig")   return { "CONFIG", "REMOTE", true, 18, 1 };
    if (key == "displayStore")    return { "STORE", "RANDOM", true, 16, 1 };
    if (key == "displayUndo")     return { "UNDO", "HELP", true, 14, 1 };
    if (key == "displayShift")    return { "SHIFT", {}, false, 14, 1 };
    if (key == "displaySearch")   return { "SEARCH", {}, false, 18, 1 };
    if (key == "part")            return {};
    if (key == "partLeft")        return {};
    if (key == "multi")           return { "MULTI", "SEQ MODE", true, 14, 1 };
    if (key == "single")          return { "SINGLE", "SEQ MODE", true, 16, 1 };
    if (key == "parameters")      return {};
    if (key == "parametersNext")  return {};
    if (key == "valueProgram")    return {};
    if (key == "valueProgramNext") return {};
    if (key == "filterEdit")      return { "EDIT", {}, false, 12, 1 };
    if (key == "filter1Focus")    return { "FLT 1", {}, false, 12, 1 };
    if (key == "filterMode")      return { "FLT 2", {}, false, 12, 1 };
    if (key == "filter2Focus")    return {};
    if (key == "filterSelect")    return {};
    return {};
}

std::array<VirusGroupChrome, 9> virusGroupChrome()
{
    return {{
        { "MATRIX",             {  90.0f,  61.0f,  80.0f, 158.0f }, { 103.0f,  49.0f,  57.0f, 14.0f } },
        { "MODULATORS",         { 171.0f,  61.0f, 220.0f, 158.0f }, { 183.0f,  49.0f,  88.0f, 14.0f } },
        { "OSCILLATORS",        { 392.0f,  61.0f, 281.0f, 158.0f }, { 404.0f,  49.0f,  96.0f, 14.0f } },
        { "MIXER",              { 690.0f,  61.0f,  64.0f, 358.0f }, { 696.0f,  49.0f,  52.0f, 14.0f } },
        { "FILTERS",            { 770.0f,  61.0f, 271.0f, 158.0f }, { 783.0f,  49.0f,  60.0f, 14.0f } },
        { "FILTER ENVELOPE",    { 770.0f, 239.0f, 270.0f,  80.0f }, { 782.0f, 227.0f, 104.0f, 14.0f } },
        { "AMPLIFIER ENVELOPE", { 770.0f, 335.0f, 270.0f,  80.0f }, { 782.0f, 323.0f, 126.0f, 14.0f } },
        { "EFFECTS",            {   8.0f, 239.0f, 300.0f, 172.0f }, {  19.0f, 227.0f,  60.0f, 14.0f } },
        { "ARP",                {   7.0f, 146.0f,  83.0f,  73.0f }, {  15.0f, 134.0f,  28.0f, 14.0f } }
    }};
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

void AdvancedVSTiAudioProcessorEditor::LedToggleButton::setVirusIndicatorOnly (bool shouldBeIndicatorOnly)
{
    virusIndicatorOnly = shouldBeIndicatorOnly;
    repaint();
}

void AdvancedVSTiAudioProcessorEditor::LedToggleButton::setVirusTopLedVisible (bool shouldShowTopLed)
{
    virusTopLedVisible = shouldShowTopLed;
    repaint();
}

void AdvancedVSTiAudioProcessorEditor::LedToggleButton::mouseDown (const juce::MouseEvent& event)
{
    pressVisualUntilMs = juce::Time::getMillisecondCounterHiRes() + 130.0;
    repaint();
    juce::Component::SafePointer<LedToggleButton> safeThis (this);
    juce::Timer::callAfterDelay (150, [safeThis]
    {
        if (safeThis != nullptr)
            safeThis->repaint();
    });

    juce::ToggleButton::mouseDown (event);
}

void AdvancedVSTiAudioProcessorEditor::LedToggleButton::mouseUp (const juce::MouseEvent& event)
{
    pressVisualUntilMs = juce::Time::getMillisecondCounterHiRes() + 85.0;
    repaint();
    juce::Component::SafePointer<LedToggleButton> safeThis (this);
    juce::Timer::callAfterDelay (100, [safeThis]
    {
        if (safeThis != nullptr)
            safeThis->repaint();
    });

    juce::ToggleButton::mouseUp (event);
}

void AdvancedVSTiAudioProcessorEditor::LedToggleButton::paintButton (juce::Graphics& g, bool shouldDrawButtonAsHighlighted, bool shouldDrawButtonAsDown)
{
    auto area = getLocalBounds().toFloat().reduced (0.5f);
    const bool pulsePressed = juce::Time::getMillisecondCounterHiRes() < pressVisualUntilMs;
    const bool pressedLook = shouldDrawButtonAsDown || pulsePressed;
    const auto radius = scaledFloat (4.5f, scale);
    const auto corner = scaledFloat (4.5f, scale);
    const auto outline = juce::jmax (1.0f, scaledFloat (1.0f, scale));
    const auto fill = pressedLook ? theme.panel.brighter (0.12f)
                                  : (shouldDrawButtonAsHighlighted ? theme.panel.brighter (0.08f) : theme.panel.brighter (0.04f));

    if (theme.tributeVirus)
    {
        const bool iconOnly = getButtonText().isEmpty();
        const bool compactVirusToggle = virusIndicatorOnly || getWidth() <= scaledInt (22.0f, scale) || getHeight() <= scaledInt (22.0f, scale);
        const auto ledDiameter = scaledFloat ((compactVirusToggle || virusIndicatorOnly) ? 5.2f : 4.6f, scale);
        const bool ledStripOnly = virusIndicatorOnly || (iconOnly && ! virusTopLedVisible && (area.getWidth() <= scaledFloat (26.0f, scale) || area.getHeight() <= scaledFloat (16.0f, scale)));
        auto ledBounds = juce::Rectangle<float> (ledDiameter, ledDiameter)
            .withCentre (ledStripOnly
                             ? area.getCentre()
                             : juce::Point<float> (area.getCentreX(), area.getY() + ledDiameter + scaledFloat (1.0f, scale)));

        auto paintVirusLed = [&] (juce::Rectangle<float> bounds, bool ledOn, juce::Colour offColour)
        {
            const auto glowColour = ledOn ? juce::Colour::fromRGB (255, 74, 78) : offColour.brighter (0.25f);
            g.setColour (glowColour.withAlpha (ledOn ? 0.18f : 0.045f));
            g.fillEllipse (bounds.expanded (scaledFloat (3.0f, scale), scaledFloat (3.0f, scale)));
            g.setColour (glowColour.withAlpha (ledOn ? 0.10f : 0.03f));
            g.fillEllipse (bounds.expanded (scaledFloat (5.0f, scale), scaledFloat (5.0f, scale)));
            g.setColour (ledOn ? juce::Colour::fromRGB (255, 66, 70) : offColour);
            g.fillEllipse (bounds);
            g.setColour ((ledOn ? juce::Colour::fromRGB (255, 196, 196) : offColour.brighter (0.55f)).withAlpha (0.9f));
            g.drawEllipse (bounds, outline);
        };

        if (virusIndicatorOnly)
        {
            if (shouldDrawButtonAsHighlighted || pressedLook)
            {
                g.setColour (juce::Colours::white.withAlpha (pressedLook ? 0.12f : 0.07f));
                g.drawRoundedRectangle (area.reduced (1.0f), scaledFloat (3.2f, scale), outline);
            }

            const auto ledOn = getToggleState();
            paintVirusLed (ledBounds, ledOn, juce::Colour::fromRGB (72, 26, 28));
            return;
        }

        auto buttonArea = area.reduced (scaledFloat (1.4f, scale), scaledFloat (1.4f, scale));
        if (virusTopLedVisible)
            buttonArea = buttonArea.withTrimmedTop ((ledBounds.getBottom() - area.getY()) + scaledFloat (2.4f, scale));
        buttonArea = buttonArea.reduced (scaledFloat (0.9f, scale), scaledFloat (0.9f, scale));

        const auto buttonSize = juce::jmin (buttonArea.getWidth(), buttonArea.getHeight());
        buttonArea = juce::Rectangle<float> (buttonSize, buttonSize)
            .withCentre ({ buttonArea.getCentreX(), buttonArea.getCentreY() + scaledFloat (0.2f, scale) });

        if (pressedLook)
            buttonArea = buttonArea.translated (scaledFloat (0.25f, scale), scaledFloat (0.85f, scale));

        auto bezelArea = buttonArea.expanded (scaledFloat (1.1f, scale), scaledFloat (1.1f, scale));
        if (pressedLook)
            bezelArea = bezelArea.translated (scaledFloat (-0.15f, scale), scaledFloat (-0.15f, scale));
        g.setColour (juce::Colours::black.withAlpha (0.55f));
        g.fillRoundedRectangle (bezelArea, scaledFloat (2.2f, scale));
        g.setColour (juce::Colours::black.withAlpha (0.85f));
        g.drawRoundedRectangle (bezelArea, scaledFloat (2.2f, scale), juce::jmax (1.0f, scaledFloat (1.0f, scale)));

        const auto bodyTop = pressedLook ? juce::Colour::fromRGB (182, 188, 199) : juce::Colour::fromRGB (232, 236, 243);
        const auto bodyBottom = pressedLook ? juce::Colour::fromRGB (44, 49, 59) : juce::Colour::fromRGB (62, 68, 80);
        juce::ColourGradient bodyFill (bodyTop, buttonArea.getCentreX(), buttonArea.getY(),
                                       bodyBottom, buttonArea.getCentreX(), buttonArea.getBottom(), false);
        g.setGradientFill (bodyFill);
        g.fillRoundedRectangle (buttonArea, scaledFloat (2.6f, scale));
        g.setColour (juce::Colours::white.withAlpha (pressedLook ? 0.26f : 0.48f));
        g.drawRoundedRectangle (buttonArea.reduced (0.5f), scaledFloat (2.6f, scale), outline);
        g.setColour (juce::Colours::black.withAlpha (0.62f));
        g.drawRoundedRectangle (buttonArea, scaledFloat (2.6f, scale), outline);

        if (pressedLook)
        {
            auto pressedShadow = buttonArea;
            g.setColour (juce::Colours::black.withAlpha (0.18f));
            g.fillRoundedRectangle (pressedShadow.removeFromTop (scaledFloat (2.2f, scale)), scaledFloat (2.0f, scale));
        }

        if (virusTopLedVisible)
        {
            const auto ledOn = getToggleState();
            paintVirusLed (ledBounds, ledOn, juce::Colour::fromRGB (64, 24, 24));
        }

        if (! iconOnly)
        {
            g.setColour (juce::Colour::fromRGB (31, 34, 39));
            g.setFont (juce::Font (juce::FontOptions { 7.1f * scale, juce::Font::bold }));
            g.drawFittedText (getButtonText(), buttonArea.toNearestInt().reduced (scaledInt (2.0f, scale), 0), juce::Justification::centred, 1);
        }

        return;
    }

    g.setColour (fill);
    g.fillRoundedRectangle (area, corner);
    g.setColour (theme.trim.withAlpha (0.95f));
    g.drawRoundedRectangle (area, corner, outline);

    const bool compactVirusToggle = theme.tributeVirus && getWidth() <= scaledInt (22.0f, scale);
    if (compactVirusToggle)
    {
        auto ledBounds = area.reduced (scaledFloat (4.0f, scale));
        g.setColour (getToggleState() ? theme.accent.brighter (0.16f) : theme.panelEdge.darker (0.6f));
        g.fillEllipse (ledBounds);
        g.setColour (juce::Colours::black.withAlpha (0.45f));
        g.drawEllipse (ledBounds, outline);
        return;
    }

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
        const auto rawBounds = juce::Rectangle<float> (static_cast<float> (x), static_cast<float> (y), static_cast<float> (width), static_cast<float> (height)).reduced (6.0f);
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
        g.setColour (juce::Colours::white.withAlpha (0.26f));
        g.drawEllipse (bounds.reduced (1.0f), 3.0f);
        g.setColour (theme.trim.withAlpha (0.9f));
        g.drawEllipse (bounds, 1.5f);

        const float dotRingRadius = (radius * 1.06f) + 6.0f;
        for (int index = 0; index < 29; ++index)
        {
            const auto proportion = static_cast<float> (index) / 28.0f;
            const auto tickAngle = rotaryStartAngle + (proportion * (rotaryEndAngle - rotaryStartAngle));
            const auto dotCentre = centre.getPointOnCircumference (dotRingRadius, tickAngle);
            const auto dotRadius = index % 7 == 0 ? radius * 0.065f : radius * 0.052f;
            const auto dotBounds = juce::Rectangle<float> (dotRadius * 2.0f, dotRadius * 2.0f).withCentre (dotCentre);
            g.setColour ((index % 7 == 0) ? theme.legend.withAlpha (0.88f) : theme.trim.withAlpha (0.74f));
            g.fillEllipse (dotBounds);
        }

        const float indicatorLength = (radius * 0.76f) + 1.0f;
        juce::Path indicator;
        indicator.addRoundedRectangle (-1.05f, -indicatorLength, 2.1f, indicatorLength, 1.0f);
        g.setColour ((theme.knobCap.isTransparent() ? theme.text : theme.knobCap).brighter (0.14f));
        g.fillPath (indicator, juce::AffineTransform::rotation (angle).translated (centre.x, centre.y));
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
    const auto tooltip = buildControlTooltip (titleText, hintText);
    setTooltip (tooltip);
    titleLabel.setText (titleText, juce::dontSendNotification);
    titleLabel.setJustificationType (juce::Justification::centred);
    titleLabel.setColour (juce::Label::textColourId,
                          (lf.theme.tributeVirus || lf.theme.tribute303 || lf.theme.tribute909) ? lf.theme.legend : lf.theme.text);
    titleLabel.setMinimumHorizontalScale (0.72f);
    titleLabel.setTooltip (tooltip);
    addAndMakeVisible (titleLabel);

    hintLabel.setText (hintText, juce::dontSendNotification);
    hintLabel.setJustificationType (juce::Justification::centred);
    hintLabel.setColour (juce::Label::textColourId,
                         (lf.theme.tribute909 || lf.theme.tributeVirus) ? lf.theme.panelEdge.brighter (0.18f) : lf.theme.hint);
    hintLabel.setMinimumHorizontalScale (0.75f);
    hintLabel.setTooltip (tooltip);
    addAndMakeVisible (hintLabel);

    if (toggleText.isNotEmpty())
    {
        toggleButton = std::make_unique<LedToggleButton> (lf.theme, toggleText);
        toggleButton->setTooltip (titleText + " toggle");
        addAndMakeVisible (*toggleButton);
    }

    slider.setSliderStyle (juce::Slider::RotaryHorizontalVerticalDrag);
    slider.setLookAndFeel (&lookAndFeel);
    slider.setTooltip (tooltip);
    slider.setColour (juce::Slider::textBoxTextColourId, lf.theme.text);
    slider.setColour (juce::Slider::textBoxBackgroundColourId,
                      lf.theme.tribute303 ? lf.theme.faceplate.brighter (0.05f)
                                         : (lf.theme.tributeVirus ? lf.theme.panel.brighter (0.04f)
                                                                  : (lf.theme.tribute909 ? lf.theme.faceplate.brighter (0.1f) : lf.theme.panel.darker (0.2f))));
    slider.setColour (juce::Slider::textBoxOutlineColourId, (lf.theme.tribute303 || lf.theme.tribute909 || lf.theme.tributeVirus) ? lf.theme.trim : lf.theme.panelEdge);
    slider.setColour (juce::Slider::textBoxHighlightColourId, lf.theme.accent.withAlpha (0.18f));
    addAndMakeVisible (slider);

    if (lf.theme.tributeVirus)
    {
        titleLabel.setVisible (false);
        hintLabel.setVisible (false);
    }

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
    const auto& skin = lookAndFeel.theme;
    const auto titleSize = (lookAndFeel.theme.tributeVirus ? 9.4f
                            : (lookAndFeel.theme.tribute909 && compact ? 8.4f
                               : ((lookAndFeel.theme.tribute303 || compact) ? 10.5f : 14.0f))) * scale;
    const auto hintSize = (lookAndFeel.theme.tributeVirus ? 7.4f
                           : (lookAndFeel.theme.tribute909 && compact ? 0.1f
                              : ((lookAndFeel.theme.tribute303 || compact) ? 8.4f : 11.0f))) * scale;
    titleLabel.setFont (juce::Font (titleSize, juce::Font::bold));
    hintLabel.setFont (juce::Font (hintSize, juce::Font::plain));

    slider.setColour (juce::Slider::textBoxTextColourId, skin.text);
    slider.setColour (juce::Slider::textBoxBackgroundColourId,
                      skin.tribute303 ? skin.faceplate.brighter (0.05f)
                                      : (skin.tributeVirus ? juce::Colours::transparentBlack
                                                           : (skin.tribute909 ? skin.faceplate.brighter (0.1f)
                                                                              : skin.panel.darker (0.2f))));
    slider.setColour (juce::Slider::textBoxOutlineColourId,
                      skin.tributeVirus ? juce::Colours::transparentBlack
                                        : ((skin.tribute303 || skin.tribute909) ? skin.trim : skin.panelEdge));
    slider.setColour (juce::Slider::textBoxHighlightColourId,
                      skin.tributeVirus ? juce::Colours::transparentBlack : skin.accent.withAlpha (0.18f));

    if (skin.tributeVirus)
    {
        slider.setTextBoxStyle (juce::Slider::NoTextBox, false, 0, 0);
        slider.setColour (juce::Slider::textBoxTextColourId, juce::Colours::transparentBlack);
    }
    else
    {
        slider.setTextBoxStyle (juce::Slider::TextBoxBelow, false,
                                juce::roundToInt (((lookAndFeel.theme.tribute909 ? (compact ? 42.0f : 54.0f) : (lookAndFeel.theme.tribute303 ? 56.0f : (compact ? 62.0f : 72.0f)))) * scale),
                                juce::roundToInt ((lookAndFeel.theme.tribute909 ? (compact ? 16.0f : 18.0f) : (lookAndFeel.theme.tribute303 ? 18.0f : 22.0f)) * scale));
    }
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
        return;

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
        auto area = getLocalBounds();
        titleLabel.setBounds ({});
        hintLabel.setBounds ({});
        if (toggleButton != nullptr)
        {
            const auto toggleSize = scaledInt (static_cast<float> (kVirusToggleSize), scale);
            toggleButton->setBounds (area.getRight() - toggleSize, area.getY(), toggleSize, toggleSize);
        }
        slider.setBounds (area.withTrimmedBottom (scaledInt (13.0f, scale)));
        return;
    }

    if (lookAndFeel.theme.tribute909)
    {
        if (compact)
        {
            titleLabel.setBounds ({});
            hintLabel.setBounds ({});
            slider.setTextBoxStyle (juce::Slider::NoTextBox, false, 0, 0);
            slider.setBounds (getLocalBounds());
            return;
        }
        auto area = getLocalBounds().reduced (scaledInt (8.0f, scale));
        titleLabel.setBounds (area.removeFromTop (scaledInt (18.0f, scale)));
        slider.setBounds (area.removeFromTop (juce::jmin (scaledInt (78.0f, scale),
                                                          juce::jmax (scaledInt (60.0f, scale), area.getHeight() - scaledInt (30.0f, scale)))));
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
    const auto tooltip = buildControlTooltip (spec.title, spec.hint);
    setTooltip (tooltip);
    titleLabel.setText (spec.title, juce::dontSendNotification);
    titleLabel.setJustificationType (theme.tributeVirus ? juce::Justification::centred : juce::Justification::centredLeft);
    titleLabel.setColour (juce::Label::textColourId, (theme.tribute303 || theme.tribute909 || theme.tributeVirus) ? theme.legend : theme.text);
    titleLabel.setMinimumHorizontalScale (0.72f);
    titleLabel.setTooltip (tooltip);
    addAndMakeVisible (titleLabel);

    hintLabel.setText (spec.hint, juce::dontSendNotification);
    hintLabel.setJustificationType (theme.tributeVirus ? juce::Justification::centred : juce::Justification::centredLeft);
    hintLabel.setColour (juce::Label::textColourId, (theme.tribute909 || theme.tributeVirus) ? theme.panelEdge.brighter (0.18f) : theme.hint);
    hintLabel.setMinimumHorizontalScale (0.72f);
    hintLabel.setTooltip (tooltip);
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
    combo.setTooltip (tooltip);
    addAndMakeVisible (combo);

    if (usesLedButtons)
    {
        const auto labels = spec.buttonLabels.isEmpty() ? spec.values : spec.buttonLabels;
        for (int index = 0; index < labels.size(); ++index)
        {
            auto* button = optionButtons.add (new LedToggleButton (theme, labels[index], false));
            button->setTooltip (tooltip);
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

    if (theme.tributeVirus)
    {
        titleLabel.setVisible (false);
        hintLabel.setVisible (false);
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
        return;

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
        auto area = getLocalBounds();
        titleLabel.setBounds ({});
        hintLabel.setBounds ({});

        if (usesLedButtons && optionButtons.size() > 0)
        {
            combo.setVisible (false);
            combo.setBounds ({});

            const int spacing = scaledInt (4.0f, scale);
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
            combo.setBounds (area);
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
    const bool compactVecPad = lf.theme.vecPadMachine;
    auto configurePadSlider = [this, compactVecPad, &lf] (PreviewSlider& targetSlider, const juce::String& tooltip)
    {
        targetSlider.setSliderStyle (juce::Slider::RotaryHorizontalVerticalDrag);
        targetSlider.setTextBoxStyle (juce::Slider::TextBoxBelow, false, compactVecPad ? 42 : (lf.theme.tribute909 ? 52 : 58), compactVecPad ? 16 : 20);
        targetSlider.setLookAndFeel (&lookAndFeel);
        targetSlider.setColour (juce::Slider::textBoxTextColourId, lf.theme.text);
        targetSlider.setColour (juce::Slider::textBoxBackgroundColourId, lf.theme.tribute909 ? lf.theme.faceplate.brighter (0.08f) : lf.theme.panel.brighter (0.08f));
        targetSlider.setColour (juce::Slider::textBoxOutlineColourId, lf.theme.tribute909 ? lf.theme.trim : lf.theme.panelEdge);
        targetSlider.setColour (juce::Slider::textBoxHighlightColourId, lf.theme.accent.withAlpha (0.16f));
        targetSlider.setMouseDragSensitivity (compactVecPad ? 190 : 160);
        targetSlider.setTooltip (tooltip);
        addAndMakeVisible (targetSlider);
    };
    auto configureFooterLabel = [this, compactVecPad, &lf] (juce::Label& label, const juce::String& text)
    {
        label.setText (text, juce::dontSendNotification);
        label.setJustificationType (juce::Justification::centred);
        label.setFont (juce::Font (juce::FontOptions { compactVecPad ? 8.0f : 8.8f, juce::Font::bold }));
        label.setColour (juce::Label::textColourId, lf.theme.hint.brighter (0.4f));
        label.setInterceptsMouseClicks (false, false);
        label.setVisible (false);
        addAndMakeVisible (label);
    };

    titleLabel.setText (titleText, juce::dontSendNotification);
    titleLabel.setJustificationType (juce::Justification::centred);
    titleLabel.setFont (juce::Font (juce::FontOptions { compactVecPad ? 11.5f : (lf.theme.tribute909 ? 12.0f : 13.0f), juce::Font::bold }));
    titleLabel.setColour (juce::Label::textColourId, lf.theme.tribute909 ? lf.theme.legend : lf.theme.text);
    titleLabel.setMinimumHorizontalScale (0.7f);
    titleLabel.setInterceptsMouseClicks (false, false);
    addAndMakeVisible (titleLabel);

    noteLabel.setText (noteText, juce::dontSendNotification);
    noteLabel.setJustificationType (juce::Justification::centred);
    noteLabel.setFont (juce::Font (juce::FontOptions { compactVecPad ? 8.8f : (lf.theme.tribute909 ? 9.5f : 10.5f), juce::Font::plain }));
    noteLabel.setColour (juce::Label::textColourId, lf.theme.tribute909 ? lf.theme.panelEdge.darker (0.42f) : lf.theme.hint);
    noteLabel.setInterceptsMouseClicks (false, false);
    addAndMakeVisible (noteLabel);

    presetLabel.setJustificationType (juce::Justification::centred);
    presetLabel.setFont (juce::Font (juce::FontOptions { compactVecPad ? 8.3f : 9.0f, juce::Font::bold }));
    presetLabel.setColour (juce::Label::textColourId, lf.theme.accent.brighter (0.25f));
    presetLabel.setInterceptsMouseClicks (false, false);
    presetLabel.setVisible (false);
    addAndMakeVisible (presetLabel);

    sampleLabel.setJustificationType (juce::Justification::centred);
    sampleLabel.setFont (juce::Font (juce::FontOptions { compactVecPad ? 10.8f : 10.5f, juce::Font::bold }));
    sampleLabel.setColour (juce::Label::textColourId, lf.theme.text);
    sampleLabel.setMinimumHorizontalScale (compactVecPad ? 0.58f : 0.66f);
    sampleLabel.setInterceptsMouseClicks (false, false);
    sampleLabel.setVisible (false);
    addAndMakeVisible (sampleLabel);

    configurePadSlider (slider, compactVecPad ? "Click to audition, drag to set level" : "Drag to set level");
    configurePadSlider (sustainSlider, "Click to audition, drag to set sustain time");
    configurePadSlider (releaseSlider, "Click to audition, drag to set release time");
    sustainSlider.setVisible (false);
    releaseSlider.setVisible (false);
    configureFooterLabel (levelLabel, "LVL");
    configureFooterLabel (sustainLabel, "SUS");
    configureFooterLabel (releaseLabel, "REL");

    auto configureButton = [this] (juce::TextButton& button)
    {
        button.setColour (juce::TextButton::buttonColourId, lookAndFeel.theme.panel.brighter (lookAndFeel.theme.vecPadMachine ? 0.16f : 0.08f));
        button.setColour (juce::TextButton::buttonOnColourId, lookAndFeel.theme.accent.withAlpha (0.75f));
        button.setColour (juce::TextButton::textColourOffId, lookAndFeel.theme.text);
        button.setColour (juce::TextButton::textColourOnId, lookAndFeel.theme.background);
        button.setVisible (false);
        addAndMakeVisible (button);
    };

    configureButton (leftButton);
    configureButton (rightButton);
    leftButton.onClick = [this]
    {
        if (onStepLeftRequested != nullptr)
            onStepLeftRequested();
    };
    rightButton.onClick = [this]
    {
        if (onStepRightRequested != nullptr)
            onStepRightRequested();
    };
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

    if (lookAndFeel.theme.vecPadMachine)
    {
        g.setColour (lookAndFeel.theme.background.darker (0.08f));
        g.fillRoundedRectangle (area, 18.0f);

        auto shell = area.reduced (8.0f);
        juce::ColourGradient shellFill (lookAndFeel.theme.panel.brighter (0.08f), shell.getTopLeft(),
                                        lookAndFeel.theme.panel.darker (0.22f), shell.getBottomRight(), false);
        g.setGradientFill (shellFill);
        g.fillRoundedRectangle (shell, 16.0f);
        g.setColour (lookAndFeel.theme.panelEdge.withAlpha (0.9f));
        g.drawRoundedRectangle (shell, 16.0f, 1.2f);

        auto padFace = shell.reduced (10.0f, 12.0f).withTrimmedTop (30.0f).withTrimmedBottom (showEnvelopeControls ? 82.0f : 50.0f);
        juce::ColourGradient faceFill (lookAndFeel.theme.panel.brighter (0.24f), padFace.getTopLeft(),
                                       lookAndFeel.theme.panel.darker (0.28f), padFace.getBottomRight(), false);
        g.setGradientFill (faceFill);
        g.fillRoundedRectangle (padFace, 14.0f);
        g.setColour (lookAndFeel.theme.accent.withAlpha (0.18f));
        g.fillRoundedRectangle (padFace.reduced (8.0f, 8.0f).removeFromTop (padFace.getHeight() * 0.32f), 10.0f);
        g.setColour (lookAndFeel.theme.panelEdge.brighter (0.2f));
        g.drawRoundedRectangle (padFace, 14.0f, 1.0f);
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

void AdvancedVSTiAudioProcessorEditor::DrumPad::setSampleInfo (const juce::String& presetText, const juce::String& sampleText)
{
    presetLabel.setText (presetText, juce::dontSendNotification);
    presetLabel.setVisible (presetText.isNotEmpty());
    sampleLabel.setText (sampleText, juce::dontSendNotification);
    sampleLabel.setVisible (sampleText.isNotEmpty());
    resized();
}

void AdvancedVSTiAudioProcessorEditor::DrumPad::setStepperVisible (bool shouldShowStepper)
{
    showStepper = shouldShowStepper;
    leftButton.setVisible (shouldShowStepper);
    rightButton.setVisible (shouldShowStepper);
    presetLabel.setVisible (shouldShowStepper && presetLabel.getText().isNotEmpty());
    sampleLabel.setVisible (shouldShowStepper && sampleLabel.getText().isNotEmpty());
    resized();
}

void AdvancedVSTiAudioProcessorEditor::DrumPad::setStepperEnabled (bool canStepLeft, bool canStepRight)
{
    leftButton.setEnabled (canStepLeft);
    rightButton.setEnabled (canStepRight);
}

void AdvancedVSTiAudioProcessorEditor::DrumPad::setEnvelopeControlsVisible (bool shouldShowEnvelopeControls)
{
    showEnvelopeControls = shouldShowEnvelopeControls;
    sustainSlider.setVisible (shouldShowEnvelopeControls);
    releaseSlider.setVisible (shouldShowEnvelopeControls);
    levelLabel.setVisible (shouldShowEnvelopeControls);
    sustainLabel.setVisible (shouldShowEnvelopeControls);
    releaseLabel.setVisible (shouldShowEnvelopeControls);
    resized();
}

void AdvancedVSTiAudioProcessorEditor::DrumPad::resized()
{
    const bool compactVecPad = lookAndFeel.theme.vecPadMachine;
    auto area = getLocalBounds().reduced (lookAndFeel.theme.tribute909 ? 8 : (compactVecPad ? 8 : 12));
    titleLabel.setBounds (area.removeFromTop (lookAndFeel.theme.tribute909 ? 18 : (compactVecPad ? 18 : 20)));
    noteLabel.setBounds (area.removeFromTop (lookAndFeel.theme.tribute909 ? 12 : (compactVecPad ? 12 : 16)));

    if (showStepper)
    {
        area.removeFromTop (compactVecPad ? 2 : 4);
        presetLabel.setBounds (area.removeFromTop (compactVecPad ? 12 : 14));
        sampleLabel.setBounds (area.removeFromTop (compactVecPad ? 18 : 24));
        auto footer = area.removeFromBottom (showEnvelopeControls ? (compactVecPad ? 86 : 96) : (compactVecPad ? 46 : 54));

        if (showEnvelopeControls)
        {
            auto stepperRow = footer.removeFromTop (compactVecPad ? 18 : 22);
            auto buttonArea = juce::Rectangle<int> (compactVecPad ? 52 : 64, stepperRow.getHeight()).withCentre (stepperRow.getCentre());
            leftButton.setBounds (buttonArea.removeFromLeft (compactVecPad ? 22 : 28));
            buttonArea.removeFromLeft (compactVecPad ? 6 : 8);
            rightButton.setBounds (buttonArea.removeFromLeft (compactVecPad ? 22 : 28));

            footer.removeFromTop (compactVecPad ? 10 : 12);
            auto sliderArea = footer.reduced (compactVecPad ? 0 : 2, 0);
            const int spacing = compactVecPad ? 4 : 6;
            const int sliderWidth = (sliderArea.getWidth() - (spacing * 2)) / 3;
            auto layoutSliderWithLabel = [] (juce::Rectangle<int> bounds, juce::Label& label, PreviewSlider& targetSlider)
            {
                auto labelBounds = bounds.removeFromTop (10);
                label.setBounds (labelBounds);
                targetSlider.setBounds (bounds);
            };

            auto levelArea = sliderArea.removeFromLeft (sliderWidth);
            layoutSliderWithLabel (levelArea, levelLabel, slider);
            sliderArea.removeFromLeft (spacing);
            auto sustainArea = sliderArea.removeFromLeft (sliderWidth);
            layoutSliderWithLabel (sustainArea, sustainLabel, sustainSlider);
            sliderArea.removeFromLeft (spacing);
            layoutSliderWithLabel (sliderArea, releaseLabel, releaseSlider);
        }
        else
        {
            auto buttonArea = footer.removeFromLeft (compactVecPad ? 52 : 64);
            leftButton.setBounds (buttonArea.removeFromLeft (compactVecPad ? 22 : 28));
            buttonArea.removeFromLeft (compactVecPad ? 6 : 8);
            rightButton.setBounds (buttonArea.removeFromLeft (compactVecPad ? 22 : 28));
            slider.setBounds (footer.removeFromRight (compactVecPad ? 60 : 70).reduced (0, compactVecPad ? 1 : 2));
        }
        return;
    }

    presetLabel.setBounds ({});
    sampleLabel.setBounds ({});
    leftButton.setBounds ({});
    rightButton.setBounds ({});
    levelLabel.setBounds ({});
    sustainLabel.setBounds ({});
    releaseLabel.setBounds ({});
    sustainSlider.setBounds ({});
    releaseSlider.setBounds ({});
    area.removeFromTop (lookAndFeel.theme.tribute909 ? 0 : (compactVecPad ? 1 : 2));
    slider.setBounds (area.reduced (lookAndFeel.theme.tribute909 ? 2 : (compactVecPad ? 4 : 6), 0));
}

AdvancedVSTiAudioProcessorEditor::AdvancedVSTiAudioProcessorEditor (AdvancedVSTiAudioProcessor& p)
    : AudioProcessorEditor (&p),
      audioProcessor (p),
      theme (buildTheme()),
      lookAndFeel (theme),
      tooltipWindow (this, 250)
{
    if (isTributeVirus())
    {
        virusTemplateImage = juce::ImageCache::getFromMemory (BinaryData::virus_jpg, BinaryData::virus_jpgSize);
        virusShowBackground = virusTemplateImage.isValid();
        startTimerHz (15);
    }
    else if (audioProcessor.isVec1DrumPadFlavor())
    {
        startTimerHz (12);
    }

    if (isTribute909() && normalizedPluginName (audioProcessor).contains ("drum"))
    {
#if AIMS_HAS_TR909_IMAGE
        backgroundImage = juce::ImageCache::getFromMemory (BinaryData::tr9092_jpg, BinaryData::tr9092_jpgSize);
#endif
    }

    buildEditor();
    if (isTributeVirus())
    {
        updateVirusOscillatorBindings();
        syncVirusPanelButtons();
    }

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
        defaultWidth = kVirusTemplateWidth;
        defaultHeight = kVirusTemplateHeight;
        minWidth = kVirusTemplateWidth;
        minHeight = kVirusTemplateHeight;
        maxWidth = kVirusTemplateWidth;
        maxHeight = kVirusTemplateHeight;
    }
    else if (normalizedPluginName (audioProcessor).contains ("string"))
    {
        defaultWidth = 1380;
        defaultHeight = 900;
        minWidth = 1060;
        minHeight = 760;
        maxWidth = 2600;
        maxHeight = 1800;
    }
    else if (usesDrumPadLayout())
    {
        if (audioProcessor.isVec1DrumPadFlavor())
        {
            defaultWidth = 1180;
            defaultHeight = 820;
            minWidth = 940;
            minHeight = 700;
            maxWidth = 2200;
            maxHeight = 1600;
        }
        else if (backgroundImage.isValid())
        {
            defaultWidth = 1120;
            defaultHeight = 460;
            minWidth = 1120;
            minHeight = 460;
            maxWidth = 1120;
            maxHeight = 460;
        }
        else
        {
            defaultWidth = isTribute909() ? 1120 : 1180;
            defaultHeight = isTribute909() ? 640 : 860;
            minWidth = isTribute909() ? 940 : 960;
            minHeight = isTribute909() ? 540 : 700;
            maxWidth = 2400;
            maxHeight = 1800;
        }
    }

    setResizable (! isTributeVirus() && ! backgroundImage.isValid(), false);
    setResizeLimits (minWidth, minHeight, maxWidth, maxHeight);
    setSize (defaultWidth, defaultHeight);
}

void AdvancedVSTiAudioProcessorEditor::buildEditor()
{
    const auto choiceSpecs = buildChoiceSpecs();
    const auto knobSpecs = buildKnobSpecs();
    const auto drumPadSpecs = buildDrumPadSpecs();

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

    if (isTributeVirus())
    {
        badgeLabel.setVisible (false);
        titleLabel.setVisible (false);
        subtitleLabel.setVisible (false);
    }

    for (const auto& spec : choiceSpecs)
    {
        auto* card = choiceCards.add (new ChoiceCard (theme, spec));
        for (int index = 0; index < spec.values.size(); ++index)
            card->combo.addItem (spec.values[index], index + 1);
        comboAttachments.push_back (std::make_unique<juce::AudioProcessorValueTreeState::ComboBoxAttachment> (audioProcessor.apvts, spec.paramId, card->combo));
        if (isTributeVirus())
        {
            const auto priorOnChange = card->combo.onChange;
            card->combo.onChange = [this, priorOnChange, paramId = spec.paramId, title = spec.title]
            {
                if (priorOnChange != nullptr)
                    priorOnChange();
                showVirusOsdForParam (paramId, title);
            };
        }
        addAndMakeVisible (card);
    }

    for (const auto& spec : knobSpecs)
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
        if (isTributeVirus())
        {
            card->slider.onValueChange = [this, paramId = spec.paramId, title = spec.title, hint = spec.hint]
            {
                showVirusOsdForParam (paramId, title, hint);
            };
        }
        if (spec.toggleParamId.isNotEmpty())
        {
            if (auto* toggleButton = card->getToggleButton())
                buttonAttachments.push_back (std::make_unique<juce::AudioProcessorValueTreeState::ButtonAttachment> (audioProcessor.apvts, spec.toggleParamId, *toggleButton));
        }
        addAndMakeVisible (card);
    }

    if (usesDrumPadLayout() && ! isTribute909())
    {
        int padIndex = 0;
        for (const auto& spec : drumPadSpecs)
        {
            auto* pad = drumPads.add (new DrumPad (lookAndFeel, spec.title, spec.note));
            pad->onPreviewRequested = [this, midiNote = spec.midiNote]
            {
                audioProcessor.previewDrumPad (midiNote);
            };
            pad->slider.onPreviewRequested = pad->onPreviewRequested;
            pad->sustainSlider.onPreviewRequested = pad->onPreviewRequested;
            pad->releaseSlider.onPreviewRequested = pad->onPreviewRequested;
            if (audioProcessor.isVec1DrumPadFlavor())
            {
                pad->setStepperVisible (true);
                pad->setEnvelopeControlsVisible (true);
                pad->onStepLeftRequested = [this, padIndex, midiNote = spec.midiNote]
                {
                    audioProcessor.stepExternalPadSample (padIndex, -1);
                    audioProcessor.previewDrumPad (midiNote);
                    syncExternalPadDisplays();
                };
                pad->onStepRightRequested = [this, padIndex, midiNote = spec.midiNote]
                {
                    audioProcessor.stepExternalPadSample (padIndex, 1);
                    audioProcessor.previewDrumPad (midiNote);
                    syncExternalPadDisplays();
                };
            }
            else
            {
                pad->setEnvelopeControlsVisible (false);
            }

            sliderAttachments.push_back (std::make_unique<juce::AudioProcessorValueTreeState::SliderAttachment> (audioProcessor.apvts, spec.levelParamId, pad->slider));
            if (spec.sustainParamId.isNotEmpty())
                sliderAttachments.push_back (std::make_unique<juce::AudioProcessorValueTreeState::SliderAttachment> (audioProcessor.apvts, spec.sustainParamId, pad->sustainSlider));
            if (spec.releaseParamId.isNotEmpty())
                sliderAttachments.push_back (std::make_unique<juce::AudioProcessorValueTreeState::SliderAttachment> (audioProcessor.apvts, spec.releaseParamId, pad->releaseSlider));
            addAndMakeVisible (pad);
            ++padIndex;
        }

        if (audioProcessor.isVec1DrumPadFlavor())
            syncExternalPadDisplays();
    }

    if (isTributeVirus())
        buildVirusPanelButtons();

    if (isTributeVirus())
    {
        if (virusBackgroundToggle == nullptr)
        {
            virusBackgroundToggle = std::make_unique<LedToggleButton> (theme, "BG", true);
            virusBackgroundToggle->setScale (0.82f);
            virusBackgroundToggle->setTooltip ("Toggle the Virus background photo");
            virusBackgroundToggle->onClick = [this]
            {
                if (virusBackgroundToggle != nullptr)
                {
                    virusShowBackground = virusBackgroundToggle->getToggleState();
                    repaint();
                }
            };
            addAndMakeVisible (*virusBackgroundToggle);
        }

        if (virusKeyboardToggle == nullptr)
        {
            virusKeyboardToggle = std::make_unique<LedToggleButton> (theme, "KB", true);
            virusKeyboardToggle->setScale (0.82f);
            virusKeyboardToggle->setTooltip ("Show or hide the preview keyboard");
            virusKeyboardToggle->onClick = [this]
            {
                if (virusKeyboardToggle != nullptr)
                {
                    virusKeyboardVisible = virusKeyboardToggle->getToggleState();
                    if (virusKeyboard != nullptr)
                        virusKeyboard->setVisible (virusKeyboardVisible);
                    setResizeLimits (kVirusTemplateWidth,
                                     kVirusTemplateHeight + (virusKeyboardVisible ? kVirusKeyboardExtraHeight : 0),
                                     kVirusTemplateWidth,
                                     kVirusTemplateHeight + (virusKeyboardVisible ? kVirusKeyboardExtraHeight : 0));
                    setSize (kVirusTemplateWidth, kVirusTemplateHeight + (virusKeyboardVisible ? kVirusKeyboardExtraHeight : 0));
                    resized();
                    repaint();
                }
            };
            addAndMakeVisible (*virusKeyboardToggle);
        }

        if (virusKeyboard == nullptr)
        {
            virusKeyboard = std::make_unique<juce::MidiKeyboardComponent> (audioProcessor.getKeyboardState(),
                                                                           juce::MidiKeyboardComponent::horizontalKeyboard);
            virusKeyboard->setKeyWidth (18.0f);
            virusKeyboard->setAvailableRange (24, 96);
            virusKeyboard->setWantsKeyboardFocus (false);
            virusKeyboard->setScrollButtonsVisible (false);
            virusKeyboard->setColour (juce::MidiKeyboardComponent::whiteNoteColourId, juce::Colour::fromRGB (232, 236, 242));
            virusKeyboard->setColour (juce::MidiKeyboardComponent::blackNoteColourId, juce::Colour::fromRGB (34, 38, 44));
            virusKeyboard->setColour (juce::MidiKeyboardComponent::keySeparatorLineColourId, juce::Colour::fromRGB (92, 102, 118));
            virusKeyboard->setColour (juce::MidiKeyboardComponent::mouseOverKeyOverlayColourId, theme.accent.withAlpha (0.22f));
            virusKeyboard->setColour (juce::MidiKeyboardComponent::keyDownOverlayColourId, theme.accent.withAlpha (0.52f));
            addAndMakeVisible (*virusKeyboard);
        }

        virusBackgroundToggle->setToggleState (virusShowBackground, juce::dontSendNotification);
        virusKeyboardToggle->setToggleState (virusKeyboardVisible, juce::dontSendNotification);
        virusBackgroundToggle->setVisible (virusTemplateImage.isValid());
        virusKeyboardToggle->setVisible (true);
        virusKeyboard->setVisible (virusKeyboardVisible);
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
    return audioProcessor.isVec1DrumPadFlavor() || name.contains ("drum") || name.contains ("808");
}

void AdvancedVSTiAudioProcessorEditor::timerCallback()
{
    if (isTributeVirus())
    {
        syncVirusPanelButtons();
        if (! virusOsdPinned
            && virusOsdUntilMs > 0.0
            && juce::Time::getMillisecondCounterHiRes() > virusOsdUntilMs)
        {
            virusOsdUntilMs = 0.0;
            virusOsdTitle.clear();
            virusOsdValue.clear();
            virusOsdDetail.clear();
            repaint();
        }
    }

    if (audioProcessor.isVec1DrumPadFlavor())
        syncExternalPadDisplays();
}

void AdvancedVSTiAudioProcessorEditor::syncExternalPadDisplays()
{
    if (! audioProcessor.isVec1DrumPadFlavor())
        return;

    const auto padCount = juce::jmin (drumPads.size(), audioProcessor.externalPadCount());
    for (int index = 0; index < padCount; ++index)
    {
        const auto state = audioProcessor.getExternalPadState (index);
        drumPads[index]->setSampleInfo (state.preset, state.sample);
        drumPads[index]->setStepperEnabled (state.canStepLeft, state.canStepRight);
    }
}

void AdvancedVSTiAudioProcessorEditor::showVirusOsdMessage (const juce::String& title,
                                                            const juce::String& value,
                                                            const juce::String& detail,
                                                            bool pinned,
                                                            double lifetimeMs)
{
    if (! isTributeVirus())
        return;

    virusOsdTitle = title;
    virusOsdValue = value;
    virusOsdDetail = detail;
    virusOsdPinned = pinned;
    virusOsdUntilMs = juce::Time::getMillisecondCounterHiRes() + lifetimeMs;
    repaint();
}

void AdvancedVSTiAudioProcessorEditor::showVirusOsdForParam (const juce::String& paramId,
                                                             const juce::String& titleOverride,
                                                             const juce::String& detail,
                                                             bool pinned,
                                                             double lifetimeMs)
{
    if (! isTributeVirus())
        return;

    if (paramId == "PRESET")
    {
        virusActivePresetMenu = true;
        refreshVirusValueKnobBindings();
        refreshVirusPresetOsd();
        return;
    }

    if (auto* parameter = dynamic_cast<juce::RangedAudioParameter*> (audioProcessor.apvts.getParameter (paramId)))
    {
        juce::String valueText;
        if (auto* choice = dynamic_cast<juce::AudioParameterChoice*> (parameter))
        {
            const auto choiceIndex = juce::jlimit (0, choice->choices.size() - 1, choice->getIndex());
            valueText = choice->choices[choiceIndex];
        }
        else if (auto* toggle = dynamic_cast<juce::AudioParameterBool*> (parameter))
        {
            valueText = toggle->get() ? "ON" : "OFF";
        }
        else
        {
            valueText = parameter->getCurrentValueAsText();
            if (valueText.isEmpty())
                valueText = parameter->getText (parameter->getValue(), 24);
        }

        const auto title = titleOverride.isNotEmpty() ? titleOverride : parameter->getName (36);
        showVirusOsdMessage (title, valueText, detail, pinned, lifetimeMs);
    }
}

void AdvancedVSTiAudioProcessorEditor::refreshVirusMatrixMenuOsd()
{
    if (! isTributeVirus())
        return;

    const auto slotIndex = juce::jlimit (0, 5, virusActiveMatrixMenu >= 0 ? virusActiveMatrixMenu : virusMatrixSlotIndex);
    const auto targetIndex = juce::jlimit (0, 2, virusMatrixTargetIndex);

    const auto value = "SRC "
                       + parameterValueText (audioProcessor.apvts, virusMatrixSourceParamId (slotIndex))
                       + "   AMT "
                       + parameterValueText (audioProcessor.apvts, virusMatrixAmountParamId (slotIndex, targetIndex))
                       + "   DEST "
                       + parameterValueText (audioProcessor.apvts, virusMatrixDestinationParamId (slotIndex, targetIndex));

    showVirusOsdMessage ("MATRIX SLOT " + juce::String (slotIndex + 1),
                         value,
                         "V1 SOURCE   V2 AMOUNT   V3 DEST   DESTINATION selects row "
                             + juce::String (targetIndex + 1),
                         true,
                         60000.0);
}

void AdvancedVSTiAudioProcessorEditor::refreshVirusLfoMenuOsd()
{
    if (! isTributeVirus())
        return;

    const auto lfoIndex = juce::jlimit (0, 2, virusActiveLfoMenu >= 0 ? virusActiveLfoMenu : virusModulatorIndex);

    juce::String amountText = "0";
    juce::String rateText = "0";
    auto destinationText = parameterValueText (audioProcessor.apvts, virusLfoDestinationParamId (lfoIndex));
    bool envMode = false;
    if (auto* amountParam = dynamic_cast<juce::RangedAudioParameter*> (audioProcessor.apvts.getParameter (virusLfoAmountParamId (lfoIndex))))
    {
        amountText = amountParam->getCurrentValueAsText();
        if (amountText.isEmpty())
            amountText = amountParam->getText (amountParam->getValue(), 24);
    }
    if (auto* rateParam = dynamic_cast<juce::RangedAudioParameter*> (audioProcessor.apvts.getParameter (virusLfoRateParamId (lfoIndex))))
    {
        rateText = rateParam->getCurrentValueAsText();
        if (rateText.isEmpty())
            rateText = rateParam->getText (rateParam->getValue(), 24);
    }
    if (auto* envModeParam = dynamic_cast<juce::AudioParameterBool*> (audioProcessor.apvts.getParameter (virusLfoEnvModeParamId (lfoIndex))))
        envMode = envModeParam->get();
    if (destinationText.equalsIgnoreCase ("ASSIGN"))
        destinationText = parameterValueText (audioProcessor.apvts, virusLfoAssignDestinationParamId (lfoIndex));

    const auto value = juce::String ("AMT ")
                       + amountText
                       + "   RATE "
                       + rateText
                       + "   DEST "
                       + destinationText;
    const auto detail = juce::String ("V1 AMOUNT   V2 RATE   V3 DESTINATION   ENV ")
                        + (envMode ? "NOTE" : "FREE");

    showVirusOsdMessage ("LFO " + juce::String (lfoIndex + 1), value, detail, true, 60000.0);
}

void AdvancedVSTiAudioProcessorEditor::refreshVirusOscMenuOsd()
{
    if (! isTributeVirus())
        return;

    const auto oscIndex = juce::jlimit (0, 2, virusActiveOscMenu >= 0 ? virusActiveOscMenu : virusOscillatorIndex);

    auto oscWaveParamId = [oscIndex] () -> juce::String
    {
        switch (oscIndex)
        {
            case 1: return "OSC2TYPE";
            case 2: return "OSC3TYPE";
            default: return "OSCTYPE";
        }
    }();

    auto oscShapeParamId = [oscIndex] () -> juce::String
    {
        switch (oscIndex)
        {
            case 1: return "OSC2PW";
            case 2: return "OSC3PW";
            default: return "OSC1PW";
        }
    }();

    auto oscSemiParamId = [oscIndex] () -> juce::String
    {
        switch (oscIndex)
        {
            case 1: return "OSC2SEMITONE";
            case 2: return "OSC3SEMITONE";
            default: return "OSC1SEMITONE";
        }
    }();

    const auto value = "WAVE "
                       + parameterValueText (audioProcessor.apvts, oscWaveParamId)
                       + "   SHAPE "
                       + parameterValueText (audioProcessor.apvts, oscShapeParamId)
                       + "   SEMI "
                       + parameterValueText (audioProcessor.apvts, oscSemiParamId);

    showVirusOsdMessage ("OSC " + juce::String (oscIndex + 1),
                         value,
                         "V1 WAVE   V2 SHAPE / PW   V3 SEMITONE",
                         true,
                         60000.0);
}

void AdvancedVSTiAudioProcessorEditor::refreshVirusFilterMenuOsd()
{
    if (! isTributeVirus())
        return;

    const auto pageIndex = juce::jlimit (0, 2, virusActiveFilterMenu >= 0 ? virusActiveFilterMenu : virusFilterEditIndex);

    if (pageIndex < 2)
    {
        const auto typeParamId = pageIndex == 0 ? "FILTERTYPE" : "FILTER2TYPE";
        const auto slopeParamId = pageIndex == 0 ? "FILTERSLOPE" : "FILTER2SLOPE";
        const auto enableParamId = pageIndex == 0 ? "FILTER1ENABLE" : "FILTER2ENABLE";
        const auto value = "TYPE "
                           + parameterValueText (audioProcessor.apvts, typeParamId)
                           + "   SLOPE "
                           + parameterValueText (audioProcessor.apvts, slopeParamId)
                           + "   ENABLE "
                           + parameterValueText (audioProcessor.apvts, enableParamId);

        showVirusOsdMessage ("FILTER " + juce::String (pageIndex + 1),
                             value,
                             "V1 TYPE   V2 SLOPE   V3 ENABLE",
                             true,
                             60000.0);
        return;
    }

    const auto value = "BAL "
                       + parameterValueText (audioProcessor.apvts, "FILTERBALANCE")
                       + "   ENV "
                       + parameterValueText (audioProcessor.apvts, "FILTERENVAMOUNT")
                       + "   CUT2 "
                       + parameterValueText (audioProcessor.apvts, "CUTOFF2");

    showVirusOsdMessage ("FILTER COMMON",
                         value,
                         "V1 BALANCE   V2 ENV AMT   V3 CUTOFF 2",
                         true,
                         60000.0);
}

void AdvancedVSTiAudioProcessorEditor::refreshVirusFxMenuOsd (bool lowerSection)
{
    if (! isTributeVirus())
        return;

    if (lowerSection)
    {
        const auto effectIndex = juce::jlimit (0, static_cast<int> (kVirusLowerFxLabels.size()) - 1, virusLowerFxLegendIndex);
        const auto value = "TYPE "
                           + parameterValueText (audioProcessor.apvts, "FXTYPE")
                           + "   MIX "
                           + parameterValueText (audioProcessor.apvts, "FXMIX")
                           + "   INT "
                           + parameterValueText (audioProcessor.apvts, "FXINTENSITY");

        showVirusOsdMessage (kVirusLowerFxLabels[static_cast<size_t> (effectIndex)],
                             value,
                             "V1 TYPE   V2 MIX   V3 INTENSITY",
                             true,
                             60000.0);
        return;
    }

    const auto effectIndex = juce::jlimit (0, static_cast<int> (kVirusUpperFxLabels.size()) - 1, virusUpperFxLegendIndex);
    const auto effectLabel = kVirusUpperFxLabels[static_cast<size_t> (effectIndex)];

    if (effectIndex == 0)
    {
        const auto value = "SEND "
                           + parameterValueText (audioProcessor.apvts, "DELAYSEND")
                           + "   TIME "
                           + parameterValueText (audioProcessor.apvts, "DELAYTIME")
                           + "   FEEDBACK "
                           + parameterValueText (audioProcessor.apvts, "DELAYFEEDBACK");
        showVirusOsdMessage (effectLabel,
                             value,
                             "V1 SEND   V2 TIME   V3 FEEDBACK",
                             true,
                             60000.0);
        return;
    }

    if (effectIndex == 1)
    {
        const auto value = "MIX "
                           + parameterValueText (audioProcessor.apvts, "REVERBMIX")
                           + "   TIME "
                           + parameterValueText (audioProcessor.apvts, "REVERBTIME")
                           + "   DAMP "
                           + parameterValueText (audioProcessor.apvts, "REVERBDAMP");
        showVirusOsdMessage (effectLabel,
                             value,
                             "V1 MIX   V2 TIME   V3 DAMP",
                             true,
                             60000.0);
        return;
    }

    if (effectIndex == 2)
    {
        const auto value = "GAIN "
                           + parameterValueText (audioProcessor.apvts, "LOWEQGAIN")
                           + "   FREQ "
                           + parameterValueText (audioProcessor.apvts, "LOWEQFREQ")
                           + "   Q "
                           + parameterValueText (audioProcessor.apvts, "LOWEQQ");
        showVirusOsdMessage (effectLabel, value, "V1 GAIN   V2 FREQ   V3 Q", true, 60000.0);
        return;
    }

    if (effectIndex == 3)
    {
        const auto value = "GAIN "
                           + parameterValueText (audioProcessor.apvts, "MIDEQGAIN")
                           + "   FREQ "
                           + parameterValueText (audioProcessor.apvts, "MIDEQFREQ")
                           + "   Q "
                           + parameterValueText (audioProcessor.apvts, "MIDEQQ");
        showVirusOsdMessage (effectLabel, value, "V1 GAIN   V2 FREQ   V3 Q", true, 60000.0);
        return;
    }

    const auto value = "GAIN "
                       + parameterValueText (audioProcessor.apvts, "HIGHEQGAIN")
                       + "   FREQ "
                       + parameterValueText (audioProcessor.apvts, "HIGHEQFREQ")
                       + "   Q "
                       + parameterValueText (audioProcessor.apvts, "HIGHEQQ");
    showVirusOsdMessage (effectLabel, value, "V1 GAIN   V2 FREQ   V3 Q", true, 60000.0);
}

void AdvancedVSTiAudioProcessorEditor::refreshVirusArpMenuOsd()
{
    if (! isTributeVirus())
        return;

    const auto value = "ON "
                       + parameterValueText (audioProcessor.apvts, "ARPENABLE")
                       + "   MODE "
                       + parameterValueText (audioProcessor.apvts, "ARPMODE")
                       + "   RATE "
                       + parameterValueText (audioProcessor.apvts, "ARPRATE");

    showVirusOsdMessage ("ARPEGGIATOR",
                         value,
                         "V1 ON / OFF   V2 MODE   V3 RATE",
                         true,
                         60000.0);
}

void AdvancedVSTiAudioProcessorEditor::refreshVirusPresetOsd()
{
    if (! isTributeVirus())
        return;

    const auto presetIndex = audioProcessor.getCurrentProgram();
    auto presetName = audioProcessor.getProgramName (presetIndex);
    if (presetName.isEmpty())
        presetName = "Init Preset";
    const auto presetInfo = virusPresetDisplayInfo (presetName, presetIndex, audioProcessor.getNumPrograms());

    showVirusOsdMessage (virusPanelModeIndex == 1 ? "MULTI" : "SINGLE",
                         presetName,
                         presetInfo.descriptorA + "|" + presetInfo.descriptorB + "|" + presetInfo.descriptorC,
                         true,
                         60000.0);
}

void AdvancedVSTiAudioProcessorEditor::clearVirusLfoMenu (bool clearOsd)
{
    virusActiveMatrixMenu = -1;
    virusActiveLfoMenu = -1;
    virusActiveOscMenu = -1;
    virusActiveFilterMenu = -1;
    virusActiveFxMenu = -1;
    virusActivePresetMenu = false;
    virusActiveArpMenu = false;
    virusOsdPinned = false;
    refreshVirusValueKnobBindings();

    if (clearOsd)
    {
        virusOsdUntilMs = 0.0;
        virusOsdTitle.clear();
        virusOsdValue.clear();
        virusOsdDetail.clear();
        repaint();
    }
}

void AdvancedVSTiAudioProcessorEditor::refreshVirusValueKnobBindings()
{
    if (! isTributeVirus() || knobCards.size() <= 32 || sliderAttachments.size() <= 32)
        return;

    auto rangedParameter = [this] (const juce::String& paramId) -> juce::RangedAudioParameter*
    {
        return dynamic_cast<juce::RangedAudioParameter*> (audioProcessor.apvts.getParameter (paramId));
    };

    auto bindParamKnob = [this, rangedParameter] (int knobIndex,
                                                  const juce::String& paramId,
                                                  const juce::String& title,
                                                  const juce::String& hint)
    {
        if (! juce::isPositiveAndBelow (knobIndex, knobCards.size()) || ! juce::isPositiveAndBelow (knobIndex, static_cast<int> (sliderAttachments.size())))
            return;

        if (auto* parameter = rangedParameter (paramId))
            knobCards[knobIndex]->slider.setValue (parameter->convertFrom0to1 (parameter->getValue()), juce::dontSendNotification);

        sliderAttachments[static_cast<size_t> (knobIndex)] =
            std::make_unique<juce::AudioProcessorValueTreeState::SliderAttachment> (audioProcessor.apvts,
                                                                                     paramId,
                                                                                     knobCards[knobIndex]->slider);
        knobCards[knobIndex]->slider.setTooltip (buildControlTooltip (title, hint));
        knobCards[knobIndex]->slider.onValueChange = [this, paramId, title, hint]
        {
            showVirusOsdForParam (paramId, title, hint);
        };
    };

    auto bindMenuKnob = [this] (int knobIndex,
                                double minValue,
                                double maxValue,
                                double interval,
                                double currentValue,
                                const juce::String& tooltip,
                                std::function<void (double)> onChange)
    {
        if (! juce::isPositiveAndBelow (knobIndex, knobCards.size()) || ! juce::isPositiveAndBelow (knobIndex, static_cast<int> (sliderAttachments.size())))
            return;

        sliderAttachments[static_cast<size_t> (knobIndex)].reset();
        auto& slider = knobCards[knobIndex]->slider;
        slider.onValueChange = nullptr;
        slider.setRange (minValue, maxValue, interval);
        slider.setValue (currentValue, juce::dontSendNotification);
        slider.setTooltip (tooltip);
        slider.onValueChange = [safeThis = juce::Component::SafePointer<AdvancedVSTiAudioProcessorEditor> (this),
                                knobIndex,
                                onChange = std::move (onChange)]() mutable
        {
            if (safeThis == nullptr || onChange == nullptr)
                return;

            onChange (safeThis->knobCards[knobIndex]->slider.getValue());
        };
    };

    if (virusActiveMatrixMenu >= 0)
    {
        const auto slotIndex = juce::jlimit (0, 5, virusActiveMatrixMenu);
        const auto targetIndex = juce::jlimit (0, 2, virusMatrixTargetIndex);

        bindParamKnob (30, virusMatrixSourceParamId (slotIndex), "Value 1", "Selected matrix source");
        bindParamKnob (31, virusMatrixAmountParamId (slotIndex, targetIndex), "Value 2", "Selected matrix amount");
        bindParamKnob (32, virusMatrixDestinationParamId (slotIndex, targetIndex), "Value 3", "Selected matrix destination");

        knobCards[30]->slider.onValueChange = [this] { refreshVirusMatrixMenuOsd(); };
        knobCards[31]->slider.onValueChange = [this] { refreshVirusMatrixMenuOsd(); };
        knobCards[32]->slider.onValueChange = [this] { refreshVirusMatrixMenuOsd(); };
        return;
    }

    if (virusActiveLfoMenu >= 0)
    {
        const auto lfoIndex = juce::jlimit (0, 2, virusActiveLfoMenu);
        const auto amountParamId = virusLfoAmountParamId (lfoIndex);
        const auto rateParamId = virusLfoRateParamId (lfoIndex);

        double amountStart = 0.0;
        double amountEnd = 1.0;
        double amountInterval = 0.0;
        double amountValue = 0.0;
        if (auto* amountParam = dynamic_cast<juce::RangedAudioParameter*> (audioProcessor.apvts.getParameter (amountParamId)))
        {
            const auto range = amountParam->getNormalisableRange();
            amountStart = range.start;
            amountEnd = range.end;
            amountInterval = range.interval;
            amountValue = amountParam->convertFrom0to1 (amountParam->getValue());
        }

        double rateStart = 0.0;
        double rateEnd = 1.0;
        double rateInterval = 0.0;
        double rateValue = 0.0;
        if (auto* rateParam = dynamic_cast<juce::RangedAudioParameter*> (audioProcessor.apvts.getParameter (rateParamId)))
        {
            const auto range = rateParam->getNormalisableRange();
            rateStart = range.start;
            rateEnd = range.end;
            rateInterval = range.interval;
            rateValue = rateParam->convertFrom0to1 (rateParam->getValue());
        }

        double destinationValue = 0.0;
        if (auto* destinationParam = dynamic_cast<juce::AudioParameterChoice*> (audioProcessor.apvts.getParameter (virusLfoDestinationParamId (lfoIndex))))
            destinationValue = static_cast<double> (destinationParam->getIndex());

        bindMenuKnob (30, amountStart, amountEnd, amountInterval, amountValue,
                      buildControlTooltip ("Value 1", "Set LFO " + juce::String (lfoIndex + 1) + " amount"),
                      [this, lfoIndex] (double rawValue)
                      {
                          setParameterActualValue (audioProcessor.apvts, virusLfoAmountParamId (lfoIndex), static_cast<float> (rawValue));
                          setParameterActualValue (audioProcessor.apvts, virusLfoEnableParamId (lfoIndex), std::abs (rawValue) > 0.0001 ? 1.0f : 0.0f);
                          syncVirusPanelButtons();
                          refreshVirusLfoMenuOsd();
                      });

        bindMenuKnob (31, rateStart, rateEnd, rateInterval, rateValue,
                      buildControlTooltip ("Value 2", "Set LFO " + juce::String (lfoIndex + 1) + " rate"),
                      [this, lfoIndex] (double rawValue)
                      {
                          setParameterActualValue (audioProcessor.apvts, virusLfoRateParamId (lfoIndex), static_cast<float> (rawValue));
                          refreshVirusLfoMenuOsd();
                      });

        bindMenuKnob (32, 0.0, static_cast<double> (kVirusModDestinationLabels.size() - 1), 1.0,
                      destinationValue,
                      buildControlTooltip ("Value 3", "Cycle LFO " + juce::String (lfoIndex + 1) + " destination"),
                      [this, lfoIndex] (double rawValue)
                      {
                          setParameterActualValue (audioProcessor.apvts,
                                                   virusLfoDestinationParamId (lfoIndex),
                                                   static_cast<float> (juce::roundToInt (rawValue)));
                          syncVirusPanelButtons();
                          refreshVirusLfoMenuOsd();
                          repaint();
                      });
        return;
    }

    if (virusActiveOscMenu >= 0)
    {
        const auto oscIndex = juce::jlimit (0, 2, virusActiveOscMenu);
        const auto waveParamId = oscIndex == 0 ? "OSCTYPE" : (oscIndex == 1 ? "OSC2TYPE" : "OSC3TYPE");
        const auto shapeParamId = oscIndex == 0 ? "OSC1PW" : (oscIndex == 1 ? "OSC2PW" : "OSC3PW");
        const auto semiParamId = oscIndex == 0 ? "OSC1SEMITONE" : (oscIndex == 1 ? "OSC2SEMITONE" : "OSC3SEMITONE");

        bindParamKnob (30, waveParamId, "Value 1", "Selected oscillator wave");
        bindParamKnob (31, shapeParamId, "Value 2", "Selected oscillator shape / pulse width");
        bindParamKnob (32, semiParamId, "Value 3", "Selected oscillator semitone");

        knobCards[30]->slider.onValueChange = [this] { refreshVirusOscMenuOsd(); };
        knobCards[31]->slider.onValueChange = [this] { refreshVirusOscMenuOsd(); };
        knobCards[32]->slider.onValueChange = [this] { refreshVirusOscMenuOsd(); };
        return;
    }

    if (virusActiveFilterMenu >= 0)
    {
        const auto filterPage = juce::jlimit (0, 2, virusActiveFilterMenu);
        if (filterPage < 2)
        {
            const auto typeParamId = filterPage == 0 ? "FILTERTYPE" : "FILTER2TYPE";
            const auto slopeParamId = filterPage == 0 ? "FILTERSLOPE" : "FILTER2SLOPE";
            const auto enableParamId = filterPage == 0 ? "FILTER1ENABLE" : "FILTER2ENABLE";

            bindParamKnob (30, typeParamId, "Value 1", "Filter mode");
            bindParamKnob (31, slopeParamId, "Value 2", "Filter slope");
            bindParamKnob (32, enableParamId, "Value 3", "Filter enabled");

            knobCards[30]->slider.onValueChange = [this] { syncVirusPanelButtons(); refreshVirusFilterMenuOsd(); };
            knobCards[31]->slider.onValueChange = [this] { syncVirusPanelButtons(); refreshVirusFilterMenuOsd(); };
            knobCards[32]->slider.onValueChange = [this] { syncVirusPanelButtons(); refreshVirusFilterMenuOsd(); };
            return;
        }

        bindParamKnob (30, "FILTERBALANCE", "Value 1", "Filter balance");
        bindParamKnob (31, "FILTERENVAMOUNT", "Value 2", "Filter envelope amount");
        bindParamKnob (32, "CUTOFF2", "Value 3", "Filter 2 cutoff offset");

        knobCards[30]->slider.onValueChange = [this] { refreshVirusFilterMenuOsd(); };
        knobCards[31]->slider.onValueChange = [this] { refreshVirusFilterMenuOsd(); };
        knobCards[32]->slider.onValueChange = [this] { refreshVirusFilterMenuOsd(); };
        return;
    }

    if (virusActiveFxMenu >= 0)
    {
        if (virusActiveFxMenu == 0)
        {
            switch (juce::jlimit (0, 4, virusUpperFxLegendIndex))
            {
                case 0:
                    bindParamKnob (30, "DELAYSEND", "Value 1", "Delay send");
                    bindParamKnob (31, "DELAYTIME", "Value 2", "Delay time");
                    bindParamKnob (32, "DELAYFEEDBACK", "Value 3", "Delay feedback");
                    break;
                case 1:
                    bindParamKnob (30, "REVERBMIX", "Value 1", "Reverb mix");
                    bindParamKnob (31, "REVERBTIME", "Value 2", "Reverb time");
                    bindParamKnob (32, "REVERBDAMP", "Value 3", "Reverb damping");
                    break;
                case 2:
                    bindParamKnob (30, "LOWEQGAIN", "Value 1", "Low EQ gain");
                    bindParamKnob (31, "LOWEQFREQ", "Value 2", "Low EQ frequency");
                    bindParamKnob (32, "LOWEQQ", "Value 3", "Low EQ Q");
                    break;
                case 3:
                    bindParamKnob (30, "MIDEQGAIN", "Value 1", "Mid EQ gain");
                    bindParamKnob (31, "MIDEQFREQ", "Value 2", "Mid EQ frequency");
                    bindParamKnob (32, "MIDEQQ", "Value 3", "Mid EQ Q");
                    break;
                default:
                    bindParamKnob (30, "HIGHEQGAIN", "Value 1", "High EQ gain");
                    bindParamKnob (31, "HIGHEQFREQ", "Value 2", "High EQ frequency");
                    bindParamKnob (32, "HIGHEQQ", "Value 3", "High EQ Q");
                    break;
            }

            knobCards[30]->slider.onValueChange = [this] { refreshVirusFxMenuOsd (false); };
            knobCards[31]->slider.onValueChange = [this] { refreshVirusFxMenuOsd (false); };
            knobCards[32]->slider.onValueChange = [this] { refreshVirusFxMenuOsd (false); };
            return;
        }

        bindParamKnob (30, "FXTYPE", "Value 1", "Lower effect type");
        bindParamKnob (31, "FXMIX", "Value 2", "Lower effect mix");
        bindParamKnob (32, "FXINTENSITY", "Value 3", "Lower effect intensity");

        knobCards[30]->slider.onValueChange = [this] { syncVirusPanelButtons(); refreshVirusFxMenuOsd (true); };
        knobCards[31]->slider.onValueChange = [this] { refreshVirusFxMenuOsd (true); };
        knobCards[32]->slider.onValueChange = [this] { refreshVirusFxMenuOsd (true); };
        return;
    }

    if (virusActiveArpMenu)
    {
        bindParamKnob (30, "ARPENABLE", "Value 1", "Arpeggiator on / off");
        bindParamKnob (31, "ARPMODE", "Value 2", "Arpeggiator mode");
        bindParamKnob (32, "ARPRATE", "Value 3", "Arpeggiator rate");

        knobCards[30]->slider.onValueChange = [this] { syncVirusPanelButtons(); refreshVirusArpMenuOsd(); };
        knobCards[31]->slider.onValueChange = [this] { refreshVirusArpMenuOsd(); };
        knobCards[32]->slider.onValueChange = [this] { refreshVirusArpMenuOsd(); };
        return;
    }

    if (virusActivePresetMenu)
    {
        bindParamKnob (30, "PRESET", "Value 1", "Preset");
        bindParamKnob (31, "PRESET", "Value 2", "Preset");
        bindParamKnob (32, "PRESET", "Value 3", "Preset");

        knobCards[30]->slider.onValueChange = [this] { refreshVirusPresetOsd(); };
        knobCards[31]->slider.onValueChange = [this] { refreshVirusPresetOsd(); };
        knobCards[32]->slider.onValueChange = [this] { refreshVirusPresetOsd(); };
        return;
    }

    bindParamKnob (30, "ARPRATE", "Value 1", "Arp rate");
    bindParamKnob (31, "RHYTHMGATE_RATE", "Value 2", "Rhythm gate rate");
    bindParamKnob (32, "RHYTHMGATE_DEPTH", "Value 3", "Rhythm gate depth");
}

void AdvancedVSTiAudioProcessorEditor::syncVirusPanelButtons()
{
    if (! isTributeVirus())
        return;

    auto rangedParameter = [this] (const juce::String& paramId) -> juce::RangedAudioParameter*
    {
        return dynamic_cast<juce::RangedAudioParameter*> (audioProcessor.apvts.getParameter (paramId));
    };

    if (auto* button = findVirusPanelButton ("arpOn"))
    {
        bool enabled = audioProcessor.isArpHoldEnabled();
        if (! virusShiftMode)
            if (auto* parameter = rangedParameter ("ARPENABLE"))
                enabled = parameter->getValue() >= 0.5f;

        button->setToggleState (enabled, juce::dontSendNotification);
    }

    if (auto* parameter = rangedParameter ("MONOENABLE"))
        if (auto* button = findVirusPanelButton ("mono"))
            button->setToggleState (parameter->getValue() >= 0.5f, juce::dontSendNotification);

    if (auto* parameter = rangedParameter ("FILTER1ENABLE"))
    {
        const bool enabled = parameter->getValue() >= 0.5f;
        if (auto* button = findVirusPanelButton ("filter1ActiveLed"))
            button->setToggleState (enabled, juce::dontSendNotification);
        if (auto* button = findVirusPanelButton ("filter2Focus"))
            button->setToggleState (enabled, juce::dontSendNotification);
    }

    if (auto* parameter = rangedParameter ("FILTER2ENABLE"))
    {
        const bool enabled = parameter->getValue() >= 0.5f;
        if (auto* button = findVirusPanelButton ("filter2ActiveLed"))
            button->setToggleState (enabled, juce::dontSendNotification);
        if (auto* button = findVirusPanelButton ("filterSelect"))
            button->setToggleState (enabled, juce::dontSendNotification);
    }

    if (auto* button = findVirusPanelButton ("filter1Focus"))
        button->setToggleState (virusFilterEditIndex == 0, juce::dontSendNotification);

    auto syncFilterModeColumn = [this, rangedParameter] (const juce::String& paramId, const juce::String& prefix)
    {
        int filterTypeIndex = 0;
        if (auto* parameter = dynamic_cast<juce::AudioParameterChoice*> (rangedParameter (paramId)))
            filterTypeIndex = juce::jlimit (0, parameter->choices.size() - 1, parameter->getIndex());

        if (auto* button = findVirusPanelButton (prefix + "Lp"))
            button->setToggleState (filterTypeIndex == 0, juce::dontSendNotification);
        if (auto* button = findVirusPanelButton (prefix + "Hp"))
            button->setToggleState (filterTypeIndex == 2, juce::dontSendNotification);
        if (auto* button = findVirusPanelButton (prefix + "Bp"))
            button->setToggleState (filterTypeIndex == 1, juce::dontSendNotification);
        if (auto* button = findVirusPanelButton (prefix + "Bs"))
            button->setToggleState (filterTypeIndex == 3, juce::dontSendNotification);
    };

    syncFilterModeColumn ("FILTERTYPE", "filter1Mode");
    syncFilterModeColumn ("FILTER2TYPE", "filter2Mode");

    const std::array<const char*, 6> matrixSlotKeys { "matrixSlot1", "matrixSlot2", "matrixSlot3", "matrixSlot4", "matrixSlot5", "matrixSlot6" };
    for (int index = 0; index < static_cast<int> (matrixSlotKeys.size()); ++index)
        if (auto* button = findVirusPanelButton (matrixSlotKeys[static_cast<size_t> (index)]))
            button->setToggleState (virusMatrixSlotIndex == index, juce::dontSendNotification);

    const std::array<const char*, 3> modKeys { "modLfo1", "modLfo2", "modLfo3" };
    for (int index = 0; index < static_cast<int> (modKeys.size()); ++index)
        if (auto* button = findVirusPanelButton (modKeys[static_cast<size_t> (index)]))
            button->setToggleState (virusModulatorIndex == index, juce::dontSendNotification);

    const auto selectedShapeParamId = [this] () -> juce::String
    {
        switch (virusModulatorIndex)
        {
            case 1: return "LFO2SHAPE";
            case 2: return "LFO3SHAPE";
            default: return "LFO1SHAPE";
        }
    }();

    const auto selectedEnvModeParamId = [this] () -> juce::String
    {
        switch (virusModulatorIndex)
        {
            case 1: return "LFO2ENVMODE";
            case 2: return "LFO3ENVMODE";
            default: return "LFO1ENVMODE";
        }
    }();

    if (auto* button = findVirusPanelButton ("modEnvMode"))
        if (auto* parameter = rangedParameter (selectedEnvModeParamId))
            button->setToggleState (parameter->getValue() >= 0.5f, juce::dontSendNotification);

    if (auto* button = findVirusPanelButton ("modShape"))
        if (auto* parameter = dynamic_cast<juce::AudioParameterChoice*> (rangedParameter (selectedShapeParamId)))
            button->setToggleState (parameter->getIndex() != 0, juce::dontSendNotification);

    const std::array<const char*, 3> oscKeys { "oscLed1", "oscLed2", "oscLed3" };
    for (int index = 0; index < static_cast<int> (oscKeys.size()); ++index)
        if (auto* button = findVirusPanelButton (oscKeys[static_cast<size_t> (index)]))
            button->setToggleState (virusOscillatorIndex == index, juce::dontSendNotification);

    const std::array<const char*, 5> upperFxKeys { "fxDelay", "fxReverb", "fxLowEq", "fxMidEq", "fxHighEq" };
    for (int index = 0; index < static_cast<int> (upperFxKeys.size()); ++index)
        if (auto* button = findVirusPanelButton (upperFxKeys[static_cast<size_t> (index)]))
            button->setToggleState (virusUpperFxLegendIndex == index, juce::dontSendNotification);

    if (auto* parameter = dynamic_cast<juce::AudioParameterChoice*> (rangedParameter ("FXTYPE")))
    {
        const auto fxIndex = juce::jlimit (0, parameter->choices.size() - 1, parameter->getIndex());
        if (fxIndex == 1) virusLowerFxLegendIndex = 0;
        else if (fxIndex == 2) virusLowerFxLegendIndex = 1;
        else if (fxIndex == 3) virusLowerFxLegendIndex = 2;
        else if (fxIndex == 4) virusLowerFxLegendIndex = 3;
        else if (fxIndex >= 5) virusLowerFxLegendIndex = 4;
    }

    const std::array<const char*, 5> lowerFxKeys { "fxDistortion", "fxCharacter", "fxChorus", "fxPhaser", "fxOthers" };
    for (int index = 0; index < static_cast<int> (lowerFxKeys.size()); ++index)
        if (auto* button = findVirusPanelButton (lowerFxKeys[static_cast<size_t> (index)]))
            button->setToggleState (virusLowerFxLegendIndex == index, juce::dontSendNotification);

    if (auto* button = findVirusPanelButton ("displayShift"))
        button->setToggleState (virusShiftMode, juce::dontSendNotification);

    const std::array<const char*, 2> panelModeKeys { "multi", "single" };
    for (int index = 0; index < static_cast<int> (panelModeKeys.size()); ++index)
        if (auto* button = findVirusPanelButton (panelModeKeys[static_cast<size_t> (index)]))
            button->setToggleState (virusPanelModeIndex == index + 1, juce::dontSendNotification);
}

void AdvancedVSTiAudioProcessorEditor::updateVirusOscillatorBindings()
{
    if (! isTributeVirus() || sliderAttachments.size() <= 16 || knobCards.size() <= 16)
        return;

    auto rangedParameter = [this] (const juce::String& paramId) -> juce::RangedAudioParameter*
    {
        return dynamic_cast<juce::RangedAudioParameter*> (audioProcessor.apvts.getParameter (paramId));
    };

    const std::array<std::array<const char*, 4>, 3> paramIds {{
        {{ "OSCTYPE",  "OSC1PW", "OSC1SEMITONE", "OSC1DETUNE" }},
        {{ "OSC2TYPE", "OSC2PW", "OSC2SEMITONE", "OSC2DETUNE" }},
        {{ "OSC3TYPE", "OSC3PW", "OSC3SEMITONE", "OSC3DETUNE" }}
    }};

    const std::array<int, 4> knobIndices { 7, 15, 8, 16 };
    const std::array<std::array<juce::String, 4>, 3> tooltips {{
        {{
            buildControlTooltip ("OSC 1 Wave", "Waveform for oscillator 1"),
            buildControlTooltip ("OSC 1 Wave Select / PW", "Pulse width / shape for oscillator 1"),
            buildControlTooltip ("OSC 1 Semitone", "Tune oscillator 1 in semitones"),
            buildControlTooltip ("OSC 1 Detune", "Fine tune oscillator 1")
        }},
        {{
            buildControlTooltip ("OSC 2 Wave", "Waveform for oscillator 2"),
            buildControlTooltip ("OSC 2 Wave Select / PW", "Pulse width / shape for oscillator 2"),
            buildControlTooltip ("OSC 2 Semitone", "Tune oscillator 2 in semitones"),
            buildControlTooltip ("OSC 2 Detune", "Fine tune oscillator 2")
        }},
        {{
            buildControlTooltip ("OSC 3 Wave", "Waveform for oscillator 3 / sub oscillator"),
            buildControlTooltip ("OSC 3 Wave Select / PW", "Pulse width / shape for oscillator 3"),
            buildControlTooltip ("OSC 3 Semitone", "Tune oscillator 3 in semitones"),
            buildControlTooltip ("OSC 3 Detune", "Fine tune oscillator 3")
        }}
    }};

    const auto oscIndex = juce::jlimit (0, 2, virusOscillatorIndex);
    for (int localIndex = 0; localIndex < static_cast<int> (knobIndices.size()); ++localIndex)
    {
        const auto knobIndex = knobIndices[static_cast<size_t> (localIndex)];
        const auto paramId = juce::String (paramIds[static_cast<size_t> (oscIndex)][static_cast<size_t> (localIndex)]);
        if (auto* parameter = rangedParameter (paramId))
            knobCards[knobIndex]->slider.setValue (parameter->convertFrom0to1 (parameter->getValue()), juce::dontSendNotification);

        sliderAttachments[static_cast<size_t> (knobIndex)] =
            std::make_unique<juce::AudioProcessorValueTreeState::SliderAttachment> (audioProcessor.apvts,
                                                                                     paramId,
                                                                                     knobCards[knobIndex]->slider);
        knobCards[knobIndex]->slider.setTooltip (tooltips[static_cast<size_t> (oscIndex)][static_cast<size_t> (localIndex)]);
    }
}

void AdvancedVSTiAudioProcessorEditor::updateVirusModulatorBindings()
{
    if (! isTributeVirus() || sliderAttachments.size() <= 3 || knobCards.size() <= 3)
        return;

    auto rangedParameter = [this] (const juce::String& paramId) -> juce::RangedAudioParameter*
    {
        return dynamic_cast<juce::RangedAudioParameter*> (audioProcessor.apvts.getParameter (paramId));
    };

    const auto lfoIndex = juce::jlimit (0, 2, virusModulatorIndex);
    const auto rateParamId = virusLfoRateParamId (lfoIndex);

    if (auto* parameter = rangedParameter (rateParamId))
        knobCards[3]->slider.setValue (parameter->convertFrom0to1 (parameter->getValue()), juce::dontSendNotification);

    sliderAttachments[3] =
        std::make_unique<juce::AudioProcessorValueTreeState::SliderAttachment> (audioProcessor.apvts,
                                                                                 rateParamId,
                                                                                 knobCards[3]->slider);
    knobCards[3]->slider.setTooltip (buildControlTooltip ("LFO " + juce::String (lfoIndex + 1) + " Rate",
                                                          "Speed for the selected LFO"));
    knobCards[3]->slider.onValueChange = [this, lfoIndex]
    {
        showVirusOsdForParam (virusLfoRateParamId (lfoIndex),
                              "LFO " + juce::String (lfoIndex + 1) + " Rate",
                              "Selected LFO speed");
        if (virusActiveLfoMenu >= 0)
            refreshVirusLfoMenuOsd();
    };
}

AdvancedVSTiAudioProcessorEditor::LedToggleButton* AdvancedVSTiAudioProcessorEditor::addVirusPanelButton (
    const juce::String& key,
    const juce::String& text,
    const juce::String& tooltip,
    bool latching)
{
    auto* button = virusPanelButtons.add (new LedToggleButton (theme, text, latching));
    button->setTooltip (buildControlTooltip (humanizeVirusControlKey (key), tooltip));
    button->setClickingTogglesState (latching);
    addAndMakeVisible (button);
    virusPanelButtonKeys.push_back (key);
    return button;
}

AdvancedVSTiAudioProcessorEditor::LedToggleButton* AdvancedVSTiAudioProcessorEditor::findVirusPanelButton (const juce::String& key) const
{
    for (int index = 0; index < virusPanelButtons.size() && index < static_cast<int> (virusPanelButtonKeys.size()); ++index)
    {
        if (virusPanelButtonKeys[static_cast<size_t> (index)] == key)
            return virusPanelButtons[index];
    }

    return nullptr;
}

void AdvancedVSTiAudioProcessorEditor::buildVirusPanelButtons()
{
    if (! isTributeVirus() || ! virusPanelButtons.isEmpty())
        return;

    auto addPanelButton = [this] (const juce::String& key, const juce::String& tooltip, bool latching = true)
    {
        return addVirusPanelButton (key, {}, tooltip, latching);
    };

    auto addMomentaryButton = [&addPanelButton] (const juce::String& key, const juce::String& tooltip)
    {
        return addPanelButton (key, tooltip, false);
    };

    addMomentaryButton ("arpEdit", "ARP EDIT: open the arpeggiator page");
    addPanelButton ("arpOn", "ARP ON: enable arp playback   SHIFT+ARP ON: hold");
    addMomentaryButton ("matrixDest", "Decrease the selected modulation depth");
    addMomentaryButton ("matrixSource", "Cycle the modulation source focus");
    addMomentaryButton ("matrixSelect", "Cycle the matrix slot LED");
    addPanelButton ("matrixSlot1", "Select matrix slot 1");
    addPanelButton ("matrixSlot2", "Select matrix slot 2");
    addPanelButton ("matrixSlot3", "Select matrix slot 3");
    addPanelButton ("matrixSlot4", "Select matrix slot 4");
    addPanelButton ("matrixSlot5", "Select matrix slot 5");
    addPanelButton ("matrixSlot6", "Select matrix slot 6");
    addPanelButton ("modLfo1", "LFO 1 focus indicator");
    addPanelButton ("modLfo2", "LFO 2 focus indicator");
    addPanelButton ("modLfo3", "LFO 3 focus indicator");
    addMomentaryButton ("modEdit", "EDIT: cycle the active modulation source focus");
    addMomentaryButton ("modEnvMode", "ENV MODE: toggle note-triggered motion for the selected LFO");
    addMomentaryButton ("modShape", "SHAPE: cycle the waveform for the selected LFO");
    addMomentaryButton ("modSelect", "SELECT: focus LFO 1, then cycle its destination LED");
    addMomentaryButton ("modAssign", "SELECT: focus LFO 2, then cycle its destination LED");
    addMomentaryButton ("modSelectRight", "SELECT: focus LFO 3, then cycle its destination LED");
    addPanelButton ("oscLed1", "Editing oscillator 1");
    addPanelButton ("oscLed2", "Editing oscillator 2");
    addPanelButton ("oscLed3", "Editing oscillator 3");
    addMomentaryButton ("oscSelect", "SELECT: cycle the active oscillator edit focus");
    addMomentaryButton ("oscEdit", "EDIT: step the waveform for the selected oscillator");
    addPanelButton ("osc3On", "Enable oscillator 3 / sub osc");
    addPanelButton ("mono", "MONO: enable mono voice mode   SHIFT+MONO: panic");
    addPanelButton ("syncPanel", "Enable oscillator sync");
    addPanelButton ("fmMode", "Enable FM");
    addMomentaryButton ("fxSelectA", "Decrease the selected upper FX target");
    addMomentaryButton ("fxEditA", "Increase the selected upper FX target");
    addMomentaryButton ("fxSelectB", "Decrease lower FX depth");
    addMomentaryButton ("fxEditB", "Increase lower FX depth");
    addPanelButton ("fxDelay", "Select delay send quick edit");
    addPanelButton ("fxReverb", "Select reverb quick edit");
    addPanelButton ("fxLowEq", "Select low-band quick edit");
    addPanelButton ("fxMidEq", "Select mid-band quick edit");
    addPanelButton ("fxHighEq", "Select high-band quick edit");
    addPanelButton ("fxDistortion", "Select distortion effect");
    addPanelButton ("fxCharacter", "Select character depth quick edit");
    addPanelButton ("fxChorus", "Select chorus effect");
    addPanelButton ("fxPhaser", "Select phaser effect");
    addPanelButton ("fxOthers", "Select flanger / other effect");
    addMomentaryButton ("transposeDown", "Transpose down one octave");
    addMomentaryButton ("transposeUp", "Transpose up one octave");
    addMomentaryButton ("displayExit", "Previous preset");
    addMomentaryButton ("displayTap", "Next preset");
    addMomentaryButton ("displayEdit", "EDIT: open oscillator page   SHIFT+EDIT: multi edit");
    addMomentaryButton ("displayConfig", "CONFIG: configuration page   SHIFT+CONFIG: remote page");
    addMomentaryButton ("displayStore", "STORE: save / step storage page   SHIFT+STORE: random patch");
    addMomentaryButton ("displayUndo", "UNDO: last edit   SHIFT+UNDO: help");
    addPanelButton ("displayShift", "SHIFT: latch secondary red legends");
    addMomentaryButton ("displaySearch", "SEARCH: preset browser   SHIFT+SEARCH: audition current preset");
    addMomentaryButton ("partLeft", "Previous preset");
    addMomentaryButton ("part", "Next preset");
    addPanelButton ("multi", "Select multi panel mode");
    addPanelButton ("single", "Select single panel mode");
    addMomentaryButton ("parameters", "Previous part / multi / single panel mode");
    addMomentaryButton ("parametersNext", "Next part / multi / single panel mode");
    addMomentaryButton ("valueProgram", "Previous preset");
    addMomentaryButton ("valueProgramNext", "Next preset");
    addMomentaryButton ("filterEdit", "Toggle the active filter focus");
    addPanelButton ("filter1Focus", "Select filter 1 for editing and step its type");
    addMomentaryButton ("filterMode", "Cycle filter 2 type");
    addPanelButton ("filter2Focus", "SELECT: toggle filter 1 enabled");
    addPanelButton ("filterSelect", "SELECT: toggle filter 2 enabled");
    addPanelButton ("filter1ActiveLed", "Toggle filter 1 enabled");
    addPanelButton ("filter2ActiveLed", "Toggle filter 2 enabled");
    addPanelButton ("filter1ModeLp", "Filter 1 low-pass active");
    addPanelButton ("filter1ModeHp", "Filter 1 high-pass active");
    addPanelButton ("filter1ModeBp", "Filter 1 band-pass active");
    addPanelButton ("filter1ModeBs", "Filter 1 band-stop / notch active");
    addPanelButton ("filter2ModeLp", "Filter 2 low-pass active");
    addPanelButton ("filter2ModeHp", "Filter 2 high-pass active");
    addPanelButton ("filter2ModeBp", "Filter 2 band-pass active");
    addPanelButton ("filter2ModeBs", "Filter 2 band-stop / notch active");

    auto rangedParameter = [this] (const juce::String& paramId) -> juce::RangedAudioParameter*
    {
        return dynamic_cast<juce::RangedAudioParameter*> (audioProcessor.apvts.getParameter (paramId));
    };

    auto commitActualValue = [this] (juce::RangedAudioParameter& parameter, float actualValue)
    {
        const auto legalValue = parameter.getNormalisableRange().snapToLegalValue (actualValue);
        parameter.beginChangeGesture();
        parameter.setValueNotifyingHost (parameter.convertTo0to1 (legalValue));
        parameter.endChangeGesture();
        if (auto* parameterWithId = dynamic_cast<juce::AudioProcessorParameterWithID*> (&parameter))
            showVirusOsdForParam (parameterWithId->paramID, parameter.getName (36));
        else
            showVirusOsdMessage (parameter.getName (36), parameter.getCurrentValueAsText());
    };

    auto stepChoiceParameter = [rangedParameter, commitActualValue] (const juce::String& paramId, int delta)
    {
        if (auto* parameter = dynamic_cast<juce::AudioParameterChoice*> (rangedParameter (paramId)))
        {
            const int choiceCount = parameter->choices.size();
            if (choiceCount <= 0)
                return;

            const int currentIndex = juce::jlimit (0, choiceCount - 1, parameter->getIndex());
            const int nextIndex = (currentIndex + delta + choiceCount) % choiceCount;
            commitActualValue (*parameter, static_cast<float> (nextIndex));
        }
    };

    auto setChoiceParameter = [rangedParameter, commitActualValue] (const juce::String& paramId, int choiceIndex)
    {
        if (auto* parameter = dynamic_cast<juce::AudioParameterChoice*> (rangedParameter (paramId)))
        {
            const int choiceCount = parameter->choices.size();
            if (choiceCount <= 0)
                return;

            commitActualValue (*parameter, static_cast<float> (juce::jlimit (0, choiceCount - 1, choiceIndex)));
        }
    };

    auto nudgeParameter = [rangedParameter, commitActualValue] (const juce::String& paramId, float delta)
    {
        if (auto* parameter = rangedParameter (paramId))
        {
            const auto range = parameter->getNormalisableRange();
            const auto currentValue = parameter->convertFrom0to1 (parameter->getValue());
            const auto nextValue = range.snapToLegalValue (juce::jlimit (range.start, range.end, currentValue + delta));
            commitActualValue (*parameter, nextValue);
        }
    };

    auto toggleBoolParameter = [rangedParameter, commitActualValue] (const juce::String& paramId)
    {
        if (auto* parameter = dynamic_cast<juce::AudioParameterBool*> (rangedParameter (paramId)))
        {
            const bool nextState = ! parameter->get();
            commitActualValue (*parameter, nextState ? 1.0f : 0.0f);
            return nextState;
        }

        if (auto* parameter = rangedParameter (paramId))
        {
            const bool nextState = parameter->getValue() < 0.5f;
            commitActualValue (*parameter, nextState ? 1.0f : 0.0f);
            return nextState;
        }

        return false;
    };

    auto flashMomentary = [] (LedToggleButton* button)
    {
        if (button == nullptr)
            return;

        button->setToggleState (true, juce::dontSendNotification);
        juce::Component::SafePointer<LedToggleButton> safeButton (button);
        juce::Timer::callAfterDelay (140, [safeButton]
        {
            if (safeButton != nullptr)
                safeButton->setToggleState (false, juce::dontSendNotification);
        });
    };

    auto bindMomentaryAction = [this, flashMomentary] (const juce::String& key, std::function<void()> action)
    {
        if (auto* button = findVirusPanelButton (key))
        {
            button->onClick = [button, flashMomentary, action = std::move (action)]
            {
                if (action != nullptr)
                    action();
                flashMomentary (button);
            };
        }
    };

    auto bindShiftAwareMomentaryAction = [this, flashMomentary] (const juce::String& key,
                                                                 std::function<void()> primaryAction,
                                                                 std::function<void()> shiftedAction)
    {
        if (auto* button = findVirusPanelButton (key))
        {
            button->onClick = [this, button, flashMomentary,
                               primaryAction = std::move (primaryAction),
                               shiftedAction = std::move (shiftedAction)]
            {
                if (virusShiftMode && shiftedAction != nullptr)
                    shiftedAction();
                else if (primaryAction != nullptr)
                    primaryAction();

                flashMomentary (button);
            };
        }
    };

    auto setIndicatorOnly = [this] (const juce::String& key)
    {
        if (auto* button = findVirusPanelButton (key))
        {
            button->setVirusIndicatorOnly (true);
            button->setInterceptsMouseClicks (false, false);
            button->setWantsKeyboardFocus (false);
            button->setTooltip ({});
        }
    };

    auto setTopLedVisible = [this] (const juce::String& key)
    {
        if (auto* button = findVirusPanelButton (key))
            button->setVirusTopLedVisible (true);
    };

    auto setShiftMode = [this] (bool enabled, const juce::String& detail = {})
    {
        virusShiftMode = enabled;
        syncVirusPanelButtons();
        showVirusOsdMessage ("SHIFT",
                             enabled ? "ON" : "OFF",
                             detail.isNotEmpty() ? detail
                                                 : (enabled ? "Secondary red legends active"
                                                            : "Primary functions active"),
                             false,
                             2000.0);
        repaint();
    };

    auto selectMatrixSlot = [this] (int index)
    {
        clearVirusLfoMenu();
        virusMatrixSlotIndex = juce::jlimit (0, 5, index);
        virusActiveMatrixMenu = virusMatrixSlotIndex;
        refreshVirusValueKnobBindings();
        syncVirusPanelButtons();
        refreshVirusMatrixMenuOsd();
    };

    auto selectModulator = [this] (int index)
    {
        virusActiveMatrixMenu = -1;
        virusModulatorIndex = juce::jlimit (0, 2, index);
        updateVirusModulatorBindings();
        syncVirusPanelButtons();
        showVirusOsdMessage ("MODULATOR", "LFO " + juce::String (virusModulatorIndex + 1), "Selected");
        repaint();
    };

    auto focusOrCycleModulator = [this, selectModulator] (int index)
    {
        const auto clampedIndex = juce::jlimit (0, 2, index);
        if (virusModulatorIndex != clampedIndex)
            selectModulator (clampedIndex);

        else
        {
            if (auto* destinationParam = dynamic_cast<juce::AudioParameterChoice*> (audioProcessor.apvts.getParameter (virusLfoDestinationParamId (clampedIndex))))
            {
                const auto nextDestination = (destinationParam->getIndex() + 1) % destinationParam->choices.size();
                setParameterActualValue (audioProcessor.apvts,
                                         virusLfoDestinationParamId (clampedIndex),
                                         static_cast<float> (nextDestination));
            }
            syncVirusPanelButtons();
        }

        virusActiveLfoMenu = clampedIndex;
        refreshVirusValueKnobBindings();
        refreshVirusLfoMenuOsd();
        repaint();
    };

    auto selectOscillator = [this] (int index)
    {
        const bool keepOscMenuOpen = virusActiveOscMenu >= 0;
        clearVirusLfoMenu();
        virusOscillatorIndex = juce::jlimit (0, 2, index);
        updateVirusOscillatorBindings();
        syncVirusPanelButtons();
        if (keepOscMenuOpen)
        {
            virusActiveOscMenu = virusOscillatorIndex;
            refreshVirusValueKnobBindings();
            refreshVirusOscMenuOsd();
        }
        else
        {
            showVirusOsdMessage ("OSCILLATOR", "OSC " + juce::String (virusOscillatorIndex + 1), "Edit focus");
        }
        repaint();
    };

    auto selectUpperFxTarget = [this] (int index)
    {
        const bool keepFxMenuOpen = virusActiveFxMenu == 0;
        clearVirusLfoMenu();
        virusUpperFxLegendIndex = juce::jlimit (0, 4, index);
        syncVirusPanelButtons();
        if (keepFxMenuOpen)
        {
            virusActiveFxMenu = 0;
            refreshVirusValueKnobBindings();
            refreshVirusFxMenuOsd (false);
        }
        else
        {
            showVirusOsdMessage ("UPPER FX",
                                 kVirusUpperFxLabels[static_cast<size_t> (virusUpperFxLegendIndex)],
                                 "SELECT cycles focus   EDIT opens page");
        }
    };

    auto selectLowerFxTarget = [this, setChoiceParameter] (int index)
    {
        const bool keepFxMenuOpen = virusActiveFxMenu == 1;
        clearVirusLfoMenu();
        virusLowerFxLegendIndex = juce::jlimit (0, 4, index);
        switch (virusLowerFxLegendIndex)
        {
            case 0: setChoiceParameter ("FXTYPE", 1); break;
            case 1: setChoiceParameter ("FXTYPE", 2); break;
            case 2: setChoiceParameter ("FXTYPE", 3); break;
            case 3: setChoiceParameter ("FXTYPE", 4); break;
            case 4: setChoiceParameter ("FXTYPE", 5); break;
            default: break;
        }
        syncVirusPanelButtons();
        if (keepFxMenuOpen)
        {
            virusActiveFxMenu = 1;
            refreshVirusValueKnobBindings();
            refreshVirusFxMenuOsd (true);
        }
        else
        {
            showVirusOsdMessage ("LOWER FX",
                                 kVirusLowerFxLabels[static_cast<size_t> (virusLowerFxLegendIndex)],
                                 "SELECT cycles focus   EDIT opens page");
        }
    };

    auto selectPanelMode = [this] (int index)
    {
        clearVirusLfoMenu();
        virusPanelModeIndex = juce::jlimit (0, 2, index);
        syncVirusPanelButtons();
    };

    auto openOscMenu = [this]
    {
        clearVirusLfoMenu();
        virusActiveOscMenu = juce::jlimit (0, 2, virusOscillatorIndex);
        refreshVirusValueKnobBindings();
        refreshVirusOscMenuOsd();
        repaint();
    };

    auto openFilterMenu = [this]
    {
        const auto nextPage = virusActiveFilterMenu >= 0 ? (virusActiveFilterMenu + 1) % 3
                                                         : juce::jlimit (0, 1, virusFilterEditIndex);
        clearVirusLfoMenu();
        virusActiveFilterMenu = nextPage;
        refreshVirusValueKnobBindings();
        syncVirusPanelButtons();
        refreshVirusFilterMenuOsd();
        repaint();
    };

    auto openFxMenu = [this] (bool lowerSection)
    {
        clearVirusLfoMenu();
        virusActiveFxMenu = lowerSection ? 1 : 0;
        refreshVirusValueKnobBindings();
        refreshVirusFxMenuOsd (lowerSection);
        repaint();
    };

    auto openArpMenu = [this]
    {
        clearVirusLfoMenu();
        virusActiveArpMenu = true;
        refreshVirusValueKnobBindings();
        refreshVirusArpMenuOsd();
        repaint();
    };

    auto openSearchPage = [this]
    {
        clearVirusLfoMenu();
        virusActivePresetMenu = true;
        refreshVirusValueKnobBindings();
        refreshVirusPresetOsd();
        repaint();
    };

    auto auditionCurrentPreset = [this]
    {
        audioProcessor.auditionPresetNote();
        showVirusOsdMessage ("AUDITION",
                             audioProcessor.getProgramName (audioProcessor.getCurrentProgram()),
                             "SHIFT+SEARCH   Preview note C4",
                             false,
                             1800.0);
    };

    auto showShiftHelp = [this]
    {
        showVirusOsdMessage ("HELP",
                             "SHIFT+ARP HOLD   SHIFT+MONO PANIC   SHIFT+STORE RANDOM",
                             "SHIFT+EDIT MULTI EDIT   SHIFT+CONFIG REMOTE   SHIFT+SEARCH AUDITION",
                             true,
                             6000.0);
    };

    auto randomPreset = [rangedParameter, commitActualValue, this]
    {
        if (auto* parameter = dynamic_cast<juce::AudioParameterChoice*> (rangedParameter ("PRESET")))
        {
            const int choiceCount = parameter->choices.size();
            if (choiceCount <= 0)
                return;

            int nextIndex = juce::Random::getSystemRandom().nextInt (choiceCount);
            if (choiceCount > 1 && nextIndex == parameter->getIndex())
                nextIndex = (nextIndex + 1) % choiceCount;

            commitActualValue (*parameter, static_cast<float> (nextIndex));
            showVirusOsdMessage ("RANDOM PATCH",
                                 parameter->choices[nextIndex],
                                 "SHIFT+STORE",
                                 false,
                                 2400.0);
        }
    };

    auto openMultiEdit = [this, selectPanelMode]
    {
        clearVirusLfoMenu();
        selectPanelMode (1);
        showVirusOsdMessage ("MULTI EDIT",
                             "Panel focus: MULTI",
                             "SHIFT+EDIT",
                             true,
                             3000.0);
    };

    auto openRemotePage = [this]
    {
        showVirusOsdMessage ("REMOTE",
                             "Not yet modelled",
                             "SHIFT+CONFIG",
                             true,
                             2600.0);
    };

    auto selectedOscillatorWaveParamId = [this] () -> juce::String
    {
        switch (virusOscillatorIndex)
        {
            case 1: return "OSC2TYPE";
            case 2: return "OSC3TYPE";
            default: return "OSCTYPE";
        }
    };

    auto selectedOscillatorSemitoneParamId = [this] () -> juce::String
    {
        switch (virusOscillatorIndex)
        {
            case 1: return "OSC2SEMITONE";
            case 2: return "OSC3SEMITONE";
            default: return "OSC1SEMITONE";
        }
    };

    auto selectedDepthParamId = [this] () -> juce::String
    {
        return virusLfoAmountParamId (virusModulatorIndex);
    };

    auto selectedModulatorShapeParamId = [this] () -> juce::String { return virusLfoShapeParamId (virusModulatorIndex); };

    auto selectedModulatorEnvModeParamId = [this] () -> juce::String { return virusLfoEnvModeParamId (virusModulatorIndex); };

    auto stepSelectedShape = [this, stepChoiceParameter, selectedModulatorShapeParamId]
    {
        stepChoiceParameter (selectedModulatorShapeParamId(), 1);
        virusActiveLfoMenu = juce::jlimit (0, 2, virusModulatorIndex);
        refreshVirusValueKnobBindings();
        syncVirusPanelButtons();
        refreshVirusLfoMenuOsd();
        repaint();
    };

    auto stepPreset = [this, stepChoiceParameter] (int delta)
    {
        clearVirusLfoMenu();
        virusActivePresetMenu = true;
        stepChoiceParameter ("PRESET", delta);
        refreshVirusValueKnobBindings();
        refreshVirusPresetOsd();
    };

    auto stepActiveFilterSlope = [this, stepChoiceParameter]
    {
        stepChoiceParameter (virusFilterEditIndex == 1 ? "FILTER2SLOPE" : "FILTERSLOPE", 1);
    };

    auto toggleActiveFilterFocus = [this]
    {
        const bool keepFilterMenuOpen = virusActiveFilterMenu >= 0;
        clearVirusLfoMenu();
        virusFilterEditIndex = 1 - virusFilterEditIndex;
        syncVirusPanelButtons();
        if (keepFilterMenuOpen)
        {
            virusActiveFilterMenu = virusFilterEditIndex;
            refreshVirusValueKnobBindings();
            refreshVirusFilterMenuOsd();
        }
    };

    auto adjustUpperFxTarget = [this, nudgeParameter] (float direction)
    {
        switch (virusUpperFxLegendIndex)
        {
            case 0: nudgeParameter ("DELAYSEND", direction * 0.08f); break;
            case 1: nudgeParameter ("REVERBMIX", direction * 0.08f); break;
            case 2: nudgeParameter ("CUTOFF2", direction * 250.0f); break;
            case 3: nudgeParameter ("FILTERBALANCE", direction * 0.08f); break;
            default: nudgeParameter ("CUTOFF", direction * 250.0f); break;
        }
    };

    if (auto* button = findVirusPanelButton ("osc3On"))
        buttonAttachments.push_back (std::make_unique<juce::AudioProcessorValueTreeState::ButtonAttachment> (audioProcessor.apvts, "OSC3ENABLE", *button));
    if (auto* button = findVirusPanelButton ("syncPanel"))
        buttonAttachments.push_back (std::make_unique<juce::AudioProcessorValueTreeState::ButtonAttachment> (audioProcessor.apvts, "SYNCENABLE", *button));
    if (auto* button = findVirusPanelButton ("fmMode"))
        buttonAttachments.push_back (std::make_unique<juce::AudioProcessorValueTreeState::ButtonAttachment> (audioProcessor.apvts, "FMENABLE", *button));
    bindMomentaryAction ("arpEdit", [openArpMenu] { openArpMenu(); });
    if (auto* button = findVirusPanelButton ("arpOn"))
    {
        button->setClickingTogglesState (false);
        button->onClick = [this, toggleBoolParameter]
        {
            if (virusShiftMode)
            {
                const bool enabled = audioProcessor.toggleArpHold();
                syncVirusPanelButtons();
                showVirusOsdMessage ("ARP HOLD",
                                     enabled ? "ON" : "OFF",
                                     enabled ? "SHIFT+ARP ON   Latch held notes while arp runs"
                                             : "SHIFT+ARP ON   Release held arp notes",
                                     false,
                                     2200.0);
            }
            else
            {
                toggleBoolParameter ("ARPENABLE");
                syncVirusPanelButtons();
            }

            repaint();
        };
    }

    if (auto* button = findVirusPanelButton ("mono"))
    {
        button->setClickingTogglesState (false);
        button->onClick = [this, toggleBoolParameter]
        {
            if (virusShiftMode)
            {
                audioProcessor.panicAllNotes();
                showVirusOsdMessage ("PANIC",
                                     "ALL NOTES OFF",
                                     "SHIFT+MONO",
                                     false,
                                     1800.0);
            }
            else
            {
                toggleBoolParameter ("MONOENABLE");
                syncVirusPanelButtons();
            }

            repaint();
        };
    }

    bindMomentaryAction ("matrixDest", [this]
    {
        virusMatrixTargetIndex = (virusMatrixTargetIndex + 1) % 3;
        virusActiveMatrixMenu = virusMatrixSlotIndex;
        refreshVirusValueKnobBindings();
        syncVirusPanelButtons();
        refreshVirusMatrixMenuOsd();
        repaint();
    });
    bindMomentaryAction ("matrixSource", [this]
    {
        if (auto* parameter = dynamic_cast<juce::AudioParameterChoice*> (audioProcessor.apvts.getParameter (virusMatrixSourceParamId (virusMatrixSlotIndex))))
        {
            const auto nextSource = (parameter->getIndex() + 1) % parameter->choices.size();
            setParameterActualValue (audioProcessor.apvts, virusMatrixSourceParamId (virusMatrixSlotIndex), static_cast<float> (nextSource));
        }
        virusActiveMatrixMenu = virusMatrixSlotIndex;
        refreshVirusValueKnobBindings();
        syncVirusPanelButtons();
        refreshVirusMatrixMenuOsd();
        repaint();
    });
    bindMomentaryAction ("matrixSelect", [this, selectMatrixSlot] { selectMatrixSlot ((virusMatrixSlotIndex + 1) % 6); });
    bindMomentaryAction ("modEdit", [this, selectModulator] { selectModulator ((virusModulatorIndex + 1) % 3); });
    if (auto* button = findVirusPanelButton ("modEnvMode"))
    {
        button->onClick = [this, rangedParameter, selectedModulatorEnvModeParamId]
        {
            if (auto* parameter = rangedParameter (selectedModulatorEnvModeParamId()))
            {
                const bool nextState = ! (parameter->getValue() >= 0.5f);
                parameter->beginChangeGesture();
                parameter->setValueNotifyingHost (nextState ? 1.0f : 0.0f);
                parameter->endChangeGesture();
            }

            virusActiveLfoMenu = juce::jlimit (0, 2, virusModulatorIndex);
            updateVirusModulatorBindings();
            refreshVirusValueKnobBindings();
            syncVirusPanelButtons();
            refreshVirusLfoMenuOsd();
            repaint();
        };
    }
    bindMomentaryAction ("modShape", [stepSelectedShape] { stepSelectedShape(); });
    if (auto* button = findVirusPanelButton ("modSelect"))
        button->onClick = [focusOrCycleModulator] { focusOrCycleModulator (0); };
    if (auto* button = findVirusPanelButton ("modAssign"))
        button->onClick = [focusOrCycleModulator] { focusOrCycleModulator (1); };
    if (auto* button = findVirusPanelButton ("modSelectRight"))
        button->onClick = [focusOrCycleModulator] { focusOrCycleModulator (2); };
    bindMomentaryAction ("oscSelect", [this, selectOscillator] { selectOscillator ((virusOscillatorIndex + 1) % 3); });
    bindMomentaryAction ("oscEdit", [openOscMenu] { openOscMenu(); });
    bindMomentaryAction ("fxSelectA", [this, selectUpperFxTarget] { selectUpperFxTarget ((virusUpperFxLegendIndex + 1) % 5); });
    bindMomentaryAction ("fxEditA", [openFxMenu] { openFxMenu (false); });
    bindMomentaryAction ("fxSelectB", [this, selectLowerFxTarget] { selectLowerFxTarget ((virusLowerFxLegendIndex + 1) % 5); });
    bindMomentaryAction ("fxEditB", [openFxMenu] { openFxMenu (true); });
    bindMomentaryAction ("transposeDown", [selectedOscillatorSemitoneParamId, nudgeParameter] { nudgeParameter (selectedOscillatorSemitoneParamId(), -12.0f); });
    bindMomentaryAction ("transposeUp", [selectedOscillatorSemitoneParamId, nudgeParameter] { nudgeParameter (selectedOscillatorSemitoneParamId(), 12.0f); });
    bindMomentaryAction ("displayExit", [stepPreset] { stepPreset (-1); });
    bindMomentaryAction ("displayTap", [stepPreset] { stepPreset (1); });
    bindShiftAwareMomentaryAction ("displayEdit",
                                   [openOscMenu] { openOscMenu(); },
                                   [openMultiEdit] { openMultiEdit(); });
    bindShiftAwareMomentaryAction ("displayConfig",
                                   [this]
                                   {
                                       clearVirusLfoMenu();
                                       showVirusOsdMessage ("CONFIG",
                                                            "Global configuration",
                                                            "Primary function",
                                                            true,
                                                            2600.0);
                                   },
                                   [openRemotePage] { openRemotePage(); });
    bindShiftAwareMomentaryAction ("displayStore",
                                   [stepPreset] { stepPreset (-1); },
                                   [randomPreset] { randomPreset(); });
    bindShiftAwareMomentaryAction ("displayUndo",
                                   [this]
                                   {
                                       clearVirusLfoMenu();
                                       showVirusOsdMessage ("UNDO",
                                                            "Previous edit / preset",
                                                            "Primary function",
                                                            false,
                                                            2000.0);
                                   },
                                   [showShiftHelp] { showShiftHelp(); });
    if (auto* button = findVirusPanelButton ("displayShift"))
    {
        button->onClick = [this, button, setShiftMode]
        {
            setShiftMode (button->getToggleState());
        };
    }
    bindShiftAwareMomentaryAction ("displaySearch",
                                   [openSearchPage] { openSearchPage(); },
                                   [auditionCurrentPreset] { auditionCurrentPreset(); });
    bindMomentaryAction ("parameters", [this, selectPanelMode] { selectPanelMode ((virusPanelModeIndex + 2) % 3); });
    bindMomentaryAction ("parametersNext", [this, selectPanelMode] { selectPanelMode ((virusPanelModeIndex + 1) % 3); });
    bindMomentaryAction ("valueProgram", [stepPreset] { stepPreset (-1); });
    bindMomentaryAction ("valueProgramNext", [stepPreset] { stepPreset (1); });
    bindMomentaryAction ("filterEdit", [openFilterMenu] { openFilterMenu(); });
    bindMomentaryAction ("filterMode", [this, stepChoiceParameter]
    {
        const bool keepFilterMenuOpen = virusActiveFilterMenu >= 0;
        clearVirusLfoMenu();
        virusFilterEditIndex = 1;
        stepChoiceParameter ("FILTER2TYPE", 1);
        syncVirusPanelButtons();
        if (keepFilterMenuOpen)
        {
            virusActiveFilterMenu = 1;
            refreshVirusValueKnobBindings();
            refreshVirusFilterMenuOsd();
        }
    });

    const std::array<const char*, 6> matrixSlotKeys { "matrixSlot1", "matrixSlot2", "matrixSlot3", "matrixSlot4", "matrixSlot5", "matrixSlot6" };
    for (const auto* key : matrixSlotKeys)
        setIndicatorOnly (key);

    const std::array<const char*, 3> modKeys { "modLfo1", "modLfo2", "modLfo3" };
    for (const auto* key : modKeys)
        setIndicatorOnly (key);

    const std::array<const char*, 3> oscKeys { "oscLed1", "oscLed2", "oscLed3" };
    for (const auto* key : oscKeys)
        setIndicatorOnly (key);

    const std::array<const char*, 5> upperFxKeys { "fxDelay", "fxReverb", "fxLowEq", "fxMidEq", "fxHighEq" };
    for (const auto* key : upperFxKeys)
        setIndicatorOnly (key);

    const std::array<const char*, 5> lowerFxKeys { "fxDistortion", "fxCharacter", "fxChorus", "fxPhaser", "fxOthers" };
    for (const auto* key : lowerFxKeys)
        setIndicatorOnly (key);

    const std::array<const char*, 8> filterModeKeys {
        "filter1ModeLp", "filter1ModeHp", "filter1ModeBp", "filter1ModeBs",
        "filter2ModeLp", "filter2ModeHp", "filter2ModeBp", "filter2ModeBs"
    };
    for (const auto* key : filterModeKeys)
        setIndicatorOnly (key);

    const std::array<const char*, 2> filterActiveKeys { "filter1ActiveLed", "filter2ActiveLed" };
    for (const auto* key : filterActiveKeys)
        setIndicatorOnly (key);

    const std::array<const char*, 15> topLedButtons {
        "arpEdit", "arpOn",
        "modEdit", "modEnvMode", "modShape",
        "oscSelect", "oscEdit", "osc3On",
        "mono", "syncPanel",
        "displayEdit", "displayConfig", "displayShift",
        "transposeDown", "transposeUp"
    };
    for (const auto* key : topLedButtons)
        setTopLedVisible (key);

    const std::array<const char*, 2> panelModeKeys { "multi", "single" };
    for (int index = 0; index < static_cast<int> (panelModeKeys.size()); ++index)
        if (auto* button = findVirusPanelButton (panelModeKeys[static_cast<size_t> (index)]))
            button->onClick = [selectPanelMode, index] { selectPanelMode (index + 1); };

    bindMomentaryAction ("partLeft", [stepPreset] { stepPreset (-1); });
    bindMomentaryAction ("part", [stepPreset] { stepPreset (1); });

    if (auto* button = findVirusPanelButton ("filter1Focus"))
    {
        button->onClick = [this, stepChoiceParameter]
        {
            const bool keepFilterMenuOpen = virusActiveFilterMenu >= 0;
            clearVirusLfoMenu();
            virusFilterEditIndex = 0;
            stepChoiceParameter ("FILTERTYPE", 1);
            syncVirusPanelButtons();
            if (keepFilterMenuOpen)
            {
                virusActiveFilterMenu = 0;
                refreshVirusValueKnobBindings();
                refreshVirusFilterMenuOsd();
            }
        };
    }

    if (auto* button = findVirusPanelButton ("filter2Focus"))
    {
        button->onClick = [this, button, rangedParameter]
        {
            const bool keepFilterMenuOpen = virusActiveFilterMenu >= 0;
            clearVirusLfoMenu();
            virusFilterEditIndex = 0;
            if (auto* parameter = rangedParameter ("FILTER1ENABLE"))
            {
                parameter->beginChangeGesture();
                parameter->setValueNotifyingHost (button->getToggleState() ? 1.0f : 0.0f);
                parameter->endChangeGesture();
            }
            syncVirusPanelButtons();
            if (keepFilterMenuOpen)
            {
                virusActiveFilterMenu = 0;
                refreshVirusValueKnobBindings();
                refreshVirusFilterMenuOsd();
            }
        };
    }

    if (auto* button = findVirusPanelButton ("filterSelect"))
    {
        button->onClick = [this, button, rangedParameter]
        {
            const bool keepFilterMenuOpen = virusActiveFilterMenu >= 0;
            clearVirusLfoMenu();
            virusFilterEditIndex = 1;
            if (auto* parameter = rangedParameter ("FILTER2ENABLE"))
            {
                parameter->beginChangeGesture();
                parameter->setValueNotifyingHost (button->getToggleState() ? 1.0f : 0.0f);
                parameter->endChangeGesture();
            }
            syncVirusPanelButtons();
            if (keepFilterMenuOpen)
            {
                virusActiveFilterMenu = 1;
                refreshVirusValueKnobBindings();
                refreshVirusFilterMenuOsd();
            }
        };
    }

    updateVirusModulatorBindings();
    refreshVirusValueKnobBindings();
    syncVirusPanelButtons();
}

AdvancedVSTiAudioProcessorEditor::Theme AdvancedVSTiAudioProcessorEditor::buildTheme() const
{
    const auto name = normalizedPluginName (audioProcessor);

    if (name.contains ("vec1"))
    {
        Theme padMachine {
            "AI VEC1 Drum Pads",
            "One pad per VEC1 folder with hardware-style triggering and per-pad sample stepping.",
            juce::Colour::fromRGB (241, 101, 63), juce::Colour::fromRGB (255, 187, 122),
            juce::Colour::fromRGB (14, 16, 18), juce::Colour::fromRGB (28, 31, 35), juce::Colour::fromRGB (78, 86, 95),
            juce::Colour::fromRGB (236, 235, 228), juce::Colour::fromRGB (170, 177, 186)
        };
        padMachine.faceplate = juce::Colour::fromRGB (23, 26, 31);
        padMachine.trim = juce::Colour::fromRGB (85, 93, 101);
        padMachine.legend = juce::Colour::fromRGB (245, 185, 118);
        padMachine.knobBody = juce::Colour::fromRGB (41, 44, 49);
        padMachine.knobCap = juce::Colour::fromRGB (215, 213, 206);
        padMachine.vecPadMachine = true;
        return padMachine;
    }

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
        "Virus Synth",
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

    if (name.contains ("vec1"))
        return specs;

    const auto presetNames = audioProcessor.presetNames();
    if (! presetNames.isEmpty())
        specs.push_back ({ "PRESET", "Preset", isTributeVirus() ? "Performance memory" : "Classic bundled patch", presetNames, {}, false, 0 });

    if (isTributeVirus())
    {
        specs.push_back ({ "OSCTYPE", "Osc 1 Wave", "Primary waveform", { "Sine", "Saw", "Square", "Noise", "Sample", "HyperSaw", "WT Formant", "WT Complex", "WT Metal", "WT Vocal" }, { "SIN", "SAW", "SQR", "NOI", "SMP", "HYP", "WTF", "WTC", "WTM", "WTV" }, true, 5 });
        specs.push_back ({ "OSC2TYPE", "Osc 2 Wave", "Secondary waveform", { "Sine", "Saw", "Square", "Noise", "Sample", "HyperSaw", "WT Formant", "WT Complex", "WT Metal", "WT Vocal" }, { "SIN", "SAW", "SQR", "NOI", "SMP", "HYP", "WTF", "WTC", "WTM", "WTV" }, true, 5 });
        specs.push_back ({ "FILTERTYPE", "Filter 1 Mode", "Main response", { "LP", "BP", "HP", "Notch" }, { "LP", "BP", "HP", "NCH" }, true, 2 });
        specs.push_back ({ "FILTERSLOPE", "Filter 1 Slope", "Roll-off steepness", { "12 dB", "16 dB", "24 dB" }, { "12", "16", "24" }, true, 3 });
        specs.push_back ({ "FILTER2TYPE", "Filter 2 Mode", "Secondary response", { "LP", "BP", "HP", "Notch" }, { "LP", "BP", "HP", "NCH" }, true, 2 });
        specs.push_back ({ "FILTER2SLOPE", "Filter 2 Slope", "Secondary roll-off", { "12 dB", "16 dB", "24 dB" }, { "12", "16", "24" }, true, 3 });
        specs.push_back ({ "FXTYPE", "Effects", "Insert algorithm", { "Off", "Distortion", "Character", "Chorus", "Phaser", "Flanger", "Ring Mod", "Freq Shift" }, { "OFF", "DST", "CHR", "CHO", "PHA", "FLA", "RNG", "FSH" }, true, 3 });
        specs.push_back ({ "LFO1SHAPE", "LFO 1 Shape", "Primary motion", { "Sine", "Triangle", "Saw", "Noise", "Square" }, { "SIN", "TRI", "SAW", "NOI", "SQR" }, true, 3 });
        specs.push_back ({ "LFO2SHAPE", "LFO 2 Shape", "Secondary motion", { "Sine", "Triangle", "Saw", "Noise", "Square" }, { "SIN", "TRI", "SAW", "NOI", "SQR" }, true, 3 });
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

    if (name.contains ("string"))
    {
        specs.push_back ({ "OSCTYPE", "Oscillator", "Core ensemble source", { "Sine", "Saw", "Square", "Noise", "Sample" }, {}, false, 0 });
        specs.push_back ({ "FILTERTYPE", "Filter", "Main timbre curve", { "LP", "BP", "HP", "Notch" }, {}, false, 0 });
        specs.push_back ({ "FILTERSLOPE", "Slope", "Filter roll-off", { "12 dB", "16 dB", "24 dB" }, {}, false, 0 });
        specs.push_back ({ "FXTYPE", "FX Type", "Insert color", { "Off", "Distortion", "Character", "Chorus", "Phaser", "Flanger", "Ring Mod", "Freq Shift" }, {}, false, 0 });
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

    if (name.contains ("vec1"))
        return {};

    if (isTributeVirus())
        return {
            { "MASTERLEVEL", "Master", "Output level" },
            { "UNISON", "Unison", "Voice stack" },
            { "DETUNE", "Spread", "Unison width" },
            { "LFO1RATE", "LFO 1 Rate", "Pitch motion" },
            { "LFO1PITCH", "LFO 1 Amt", "Pitch depth" },
            { "LFO2RATE", "LFO 2 Rate", "Filter motion" },
            { "LFO2FILTER", "LFO 2 Amt", "Filter depth" },

            { "OSCTYPE", "Wave", "Selected oscillator waveform" },
            { "OSC1SEMITONE", "Semitone", "Selected oscillator pitch" },
            { "OSC2MIX", "Osc Mix", "Blend osc 2" },
            { "SUBOSCLEVEL", "Osc 3 / Sub", "Third osc support" },
            { "NOISELEVEL", "Noise", "Noise bed" },
            { "RINGMOD", "Ring Mod", "Metallic edge" },
            { "FMAMOUNT", "FM Amt", "Cross-mod bite" },
            { "SYNC", "Sync", "Hard sync" },
            { "OSC1PW", "Pulse Width", "Selected oscillator shape" },
            { "OSC1DETUNE", "Detune", "Selected oscillator fine tuning" },

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

    if (name.contains ("string"))
        return {
            { "POLYPHONY", "Poly", "Max active notes" },
            { "UNISON", "Unison", "Voices per note" },
            { "DETUNE", "Detune", "Unison spread" },
            { "CUTOFF", "Cutoff", "Brightness and air" },
            { "RESONANCE", "Res", "Filter focus" },
            { "FILTERENVAMOUNT", "Env Amt", "Filter bloom depth" },
            { "FILTATTACK", "Flt Att", "Filter rise" },
            { "FILTDECAY", "Flt Dec", "Filter fall" },
            { "FILTSUSTAIN", "Flt Sus", "Filter hold" },
            { "FILTRELEASE", "Flt Rel", "Filter tail" },
            { "AMPATTACK", "Amp Att", "Level rise" },
            { "AMPDECAY", "Amp Dec", "Level fall" },
            { "AMPSUSTAIN", "Amp Sus", "Level hold" },
            { "AMPRELEASE", "Amp Rel", "Level tail" },
            { "FXMIX", "FX Mix", "Insert blend" },
            { "FXINTENSITY", "FX Int", "Insert depth" },
            { "DELAYSEND", "Delay", "Echo send" },
            { "DELAYTIME", "Dly Time", "Echo space" },
            { "DELAYFEEDBACK", "Feedback", "Echo repeats" },
            { "REVERBMIX", "Reverb", "Room blend" },
        };

    if (name.contains ("pad"))
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
    auto makePadSpec = [] (juce::String levelParamId,
                           juce::String sustainParamId,
                           juce::String releaseParamId,
                           juce::String title,
                           juce::String note,
                           int midiNote)
    {
        DrumPadSpec spec;
        spec.levelParamId = std::move (levelParamId);
        spec.sustainParamId = std::move (sustainParamId);
        spec.releaseParamId = std::move (releaseParamId);
        spec.title = std::move (title);
        spec.note = std::move (note);
        spec.midiNote = midiNote;
        return spec;
    };

    if (audioProcessor.isVec1DrumPadFlavor())
    {
        std::vector<DrumPadSpec> specs;
        specs.reserve (static_cast<size_t> (audioProcessor.externalPadCount()));
        for (int padIndex = 0; padIndex < audioProcessor.externalPadCount(); ++padIndex)
        {
            const auto state = audioProcessor.getExternalPadState (padIndex);
            specs.push_back (makePadSpec (audioProcessor.externalPadLevelParameterId (padIndex),
                                          audioProcessor.externalPadSustainParameterId (padIndex),
                                          audioProcessor.externalPadReleaseParameterId (padIndex),
                                          state.title,
                                          state.note,
                                          state.midiNote));
        }
        return specs;
    }

    std::vector<DrumPadSpec> specs;
    specs.reserve (15);
    specs.push_back (makePadSpec ("DRUMLEVEL_KICK", {}, {}, "Bass Drum", "C1", 36));
    specs.push_back (makePadSpec ("DRUMLEVEL_RIM", {}, {}, "Rim Shot", "C#1", 37));
    specs.push_back (makePadSpec ("DRUMLEVEL_SNARE", {}, {}, "Snare", "D1", 38));
    specs.push_back (makePadSpec ("DRUMLEVEL_CLAP", {}, {}, "Clap", "D#1", 39));
    specs.push_back (makePadSpec ("DRUMLEVEL_CLOSED_HAT", {}, {}, "Closed Hat", "E1", 40));
    specs.push_back (makePadSpec ("DRUMLEVEL_OPEN_HAT", {}, {}, "Open Hat", "F1", 41));
    specs.push_back (makePadSpec ("DRUMLEVEL_LOW_TOM", {}, {}, "Low Tom", "F#1", 42));
    specs.push_back (makePadSpec ("DRUMLEVEL_MID_TOM", {}, {}, "Mid Tom", "G1", 43));
    specs.push_back (makePadSpec ("DRUMLEVEL_HIGH_TOM", {}, {}, "High Tom", "G#1", 44));
    specs.push_back (makePadSpec ("DRUMLEVEL_CRASH", {}, {}, "Crash", "A1", 45));
    specs.push_back (makePadSpec ("DRUMLEVEL_RIDE", {}, {}, "Ride", "A#1", 46));
    specs.push_back (makePadSpec ("DRUMLEVEL_COWBELL", {}, {}, "Cowbell", "B1", 47));
    specs.push_back (makePadSpec ("DRUMLEVEL_CLAVE", {}, {}, "Clave", "C2", 48));
    specs.push_back (makePadSpec ("DRUMLEVEL_MARACA", {}, {}, "Maraca", "C#2", 49));
    specs.push_back (makePadSpec ("DRUMLEVEL_PERC", {}, {}, "Perc", "D2", 50));
    return specs;
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
        if (backgroundImage.isValid())
        {
            g.drawImage (backgroundImage, getLocalBounds().toFloat(), juce::RectanglePlacement::stretchToFit);
            return;
        }

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
        const auto showBackground = virusShowBackground && virusTemplateImage.isValid();
        const auto uiScale = juce::jlimit (0.9f, 1.2f, juce::jmin (getWidth() / static_cast<float> (kVirusTemplateWidth),
                                                                   getHeight() / static_cast<float> (kVirusTemplateHeight)));
        const int surfaceHeight = juce::jmin (getHeight(), juce::roundToInt (kVirusTemplateHeight * uiScale));

        if (showBackground)
        {
            g.drawImageWithin (virusTemplateImage, 0, 0, getWidth(), surfaceHeight, juce::RectanglePlacement::stretchToFit);
            g.setColour (juce::Colours::black.withAlpha (0.03f));
            g.fillRect (juce::Rectangle<int> (0, 0, getWidth(), surfaceHeight));
            if (getHeight() > surfaceHeight)
            {
                juce::ColourGradient keyboardArea (juce::Colour::fromRGB (66, 69, 74), 0.0f, static_cast<float> (surfaceHeight),
                                                   juce::Colour::fromRGB (44, 46, 50), 0.0f, static_cast<float> (getHeight()), false);
                g.setGradientFill (keyboardArea);
                g.fillRect (0, surfaceHeight, getWidth(), getHeight() - surfaceHeight);
                g.setColour (juce::Colours::white.withAlpha (0.14f));
                g.drawLine (0.0f, static_cast<float> (surfaceHeight), static_cast<float> (getWidth()), static_cast<float> (surfaceHeight), 1.0f);
            }
        }
        else
        {
            juce::ColourGradient background (juce::Colour::fromRGB (91, 95, 100), 0.0f, 0.0f,
                                             juce::Colour::fromRGB (55, 58, 63), 0.0f, static_cast<float> (getHeight()), false);
            g.setGradientFill (background);
            g.fillAll();
        }

        if (! showBackground)
        {
            auto scaledRect = [uiScale] (juce::Rectangle<float> rect)
            {
                return juce::Rectangle<float> (rect.getX() * uiScale,
                                               rect.getY() * uiScale,
                                               rect.getWidth() * uiScale,
                                               rect.getHeight() * uiScale);
            };

            for (const auto& group : virusGroupChrome())
            {
                auto frame = scaledRect (group.frame);
                auto tab = scaledRect (group.tab).translated (0.0f, -scaledFloat (3.0f, uiScale));
                const auto frameCorner = scaledFloat (4.0f, uiScale);
                const auto frameOutline = juce::jmax (1.0f, scaledFloat (1.0f, uiScale));

                g.setColour (juce::Colour::fromRGBA (24, 28, 34, 168));
                g.fillRoundedRectangle (frame, frameCorner);
                g.setColour (juce::Colours::white.withAlpha (0.55f));
                g.drawRoundedRectangle (frame, frameCorner, frameOutline);

                g.setColour (juce::Colour::fromRGB (239, 241, 244).withAlpha (0.98f));
                g.setColour (juce::Colours::white.withAlpha (0.8f));
                g.fillRect (tab);
                g.drawRect (tab.toNearestInt(), juce::roundToInt (juce::jmax (1.0f, scaledFloat (1.0f, uiScale))));

                auto textBounds = tab.toNearestInt().reduced (scaledInt (5.0f, uiScale), 0);
                g.setFont (juce::Font (juce::FontOptions { 6.2f * uiScale, juce::Font::bold }));
                g.setColour (juce::Colour::fromRGB (18, 20, 24));
                g.drawFittedText (group.title, textBounds, juce::Justification::centredLeft, 1);
            }
        }

        return;
    }

    if (theme.vecPadMachine)
    {
        auto bounds = getLocalBounds().toFloat();
        juce::ColourGradient background (theme.background.brighter (0.08f), bounds.getTopLeft(),
                                         theme.background.darker (0.22f), bounds.getBottomRight(), false);
        g.setGradientFill (background);
        g.fillAll();

        auto chassis = bounds.reduced (14.0f);
        juce::ColourGradient chassisFill (theme.faceplate.brighter (0.04f), chassis.getTopLeft(),
                                          theme.faceplate.darker (0.16f), chassis.getBottomRight(), false);
        g.setGradientFill (chassisFill);
        g.fillRoundedRectangle (chassis, 24.0f);
        g.setColour (theme.panelEdge.withAlpha (0.9f));
        g.drawRoundedRectangle (chassis, 24.0f, 1.4f);

        auto header = chassis.reduced (18.0f, 16.0f).removeFromTop (80.0f);
        g.setColour (theme.accent.withAlpha (0.78f));
        g.fillRoundedRectangle (header.removeFromBottom (3.0f), 1.5f);
        g.setColour (theme.accentGlow.withAlpha (0.16f));
        g.fillRoundedRectangle (header.removeFromRight (header.getWidth() * 0.45f).translated (0.0f, 6.0f), 16.0f);
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

void AdvancedVSTiAudioProcessorEditor::paintOverChildren (juce::Graphics& g)
{
    if (! isTributeVirus())
        return;

    const bool showBackground = virusShowBackground && virusTemplateImage.isValid();
    const auto uiScale = juce::jlimit (0.9f, 1.2f, juce::jmin (getWidth() / static_cast<float> (kVirusTemplateWidth),
                                                               getHeight() / static_cast<float> (kVirusTemplateHeight)));
    const auto primaryColour = juce::Colour::fromRGB (226, 231, 238);
    const auto secondaryColour = juce::Colour::fromRGB (208, 74, 69);

    auto drawLegendLine = [&g] (juce::Rectangle<int> area, const juce::String& text, juce::Colour colour, const juce::Font& font)
    {
        if (text.isEmpty())
            return;

        g.setFont (font);
        g.setColour (juce::Colours::black.withAlpha (0.72f));
        g.drawFittedText (text, area.translated (0, 1), juce::Justification::centred, 1);
        g.setColour (colour);
        g.drawFittedText (text, area, juce::Justification::centred, 1);
    };

    auto drawLegend = [&] (juce::Rectangle<int> anchor, const VirusLegend& legend)
    {
        if (legend.primary.isEmpty())
            return;

        const int widthPad = scaledInt (static_cast<float> (legend.widthPad), uiScale);
        const int yOffset = scaledInt (static_cast<float> (legend.yOffset), uiScale);
        const int primaryHeight = scaledInt (7.0f, uiScale);
        const int secondaryHeight = scaledInt (6.0f, uiScale);
        const int secondaryGap = scaledInt (1.0f, uiScale);
        auto textBounds = juce::Rectangle<int> (anchor.getX() - widthPad / 2,
                                                anchor.getBottom() + yOffset,
                                                anchor.getWidth() + widthPad,
                                                primaryHeight + secondaryGap + secondaryHeight)
            .getIntersection (getLocalBounds().reduced (1));

        auto primaryBounds = textBounds.removeFromTop (primaryHeight);
        drawLegendLine (primaryBounds, legend.primary, primaryColour,
                        juce::Font (juce::FontOptions { 7.2f * uiScale, juce::Font::bold }));

        if (legend.secondary.isNotEmpty())
        {
            textBounds.removeFromTop (secondaryGap);
            drawLegendLine (textBounds.removeFromTop (secondaryHeight),
                            legend.secondary,
                            legend.accentSecondary ? secondaryColour : primaryColour.withAlpha (0.9f),
                            juce::Font (juce::FontOptions { 5.6f * uiScale, juce::Font::bold }));
        }
    };

    auto drawGroupLegend = [&] (std::initializer_list<const char*> keys, const VirusLegend& legend)
    {
        juce::Rectangle<int> groupBounds;
        for (const auto* key : keys)
        {
            if (auto* button = findVirusPanelButton (key))
            {
                const auto bounds = button->getBounds();
                if (! bounds.isEmpty())
                    groupBounds = groupBounds.isEmpty() ? bounds : groupBounds.getUnion (bounds);
            }
        }

        if (! groupBounds.isEmpty())
            drawLegend (groupBounds, legend);
    };

    auto drawInlineIndicatorLabel = [&] (juce::Rectangle<int> area, const juce::String& text,
                                         juce::Justification justification,
                                         float fontScale, juce::Colour colour)
    {
        if (text.isEmpty() || area.isEmpty())
            return;

        auto clipped = area.getIntersection (getLocalBounds().reduced (1));
        if (clipped.isEmpty())
            return;

        g.setFont (juce::Font (juce::FontOptions { fontScale * uiScale, juce::Font::bold }));
        g.setColour (juce::Colours::black.withAlpha (0.72f));
        g.drawFittedText (text, clipped.translated (0, 1), justification, 1);
        g.setColour (colour);
        g.drawFittedText (text, clipped, justification, 1);
    };

    auto drawLabelAboveButton = [&] (const juce::String& key, const juce::String& text, int width, int yOffset, float fontScale = 6.0f)
    {
        if (auto* button = findVirusPanelButton (key))
        {
            auto bounds = button->getBounds();
            if (bounds.isEmpty())
                return;

            auto labelBounds = juce::Rectangle<int> (bounds.getCentreX() - width / 2,
                                                     bounds.getY() + yOffset,
                                                     width,
                                                     scaledInt (8.0f, uiScale));
            drawInlineIndicatorLabel (labelBounds, text, juce::Justification::centred, fontScale, primaryColour);
        }
    };

    auto drawLabelRightOfButton = [&] (const juce::String& key, const juce::String& text, int xOffset, int width, float fontScale = 6.0f)
    {
        if (auto* button = findVirusPanelButton (key))
        {
            auto bounds = button->getBounds();
            if (bounds.isEmpty())
                return;

            auto labelBounds = juce::Rectangle<int> (bounds.getRight() + xOffset,
                                                     bounds.getY() - scaledInt (1.0f, uiScale),
                                                     width,
                                                     bounds.getHeight() + scaledInt (2.0f, uiScale));
            drawInlineIndicatorLabel (labelBounds, text, juce::Justification::centredLeft, fontScale, primaryColour);
        }
    };

    auto drawLabelLeftOfButton = [&] (const juce::String& key, const juce::String& text, int xOffset, int width, float fontScale = 6.0f)
    {
        if (auto* button = findVirusPanelButton (key))
        {
            auto bounds = button->getBounds();
            if (bounds.isEmpty())
                return;

            auto labelBounds = juce::Rectangle<int> (bounds.getX() - xOffset - width,
                                                     bounds.getY() - scaledInt (1.0f, uiScale),
                                                     width,
                                                     bounds.getHeight() + scaledInt (2.0f, uiScale));
            drawInlineIndicatorLabel (labelBounds, text, juce::Justification::centredRight, fontScale, primaryColour);
        }
    };

    for (int index = 0; index < knobCards.size(); ++index)
    {
        auto legend = virusKnobLegend (index);
        if (showBackground)
        {
            switch (index)
            {
                case 7:
                case 15:
                case 8:
                case 16:
                case 9:
                    legend.yOffset += 3;
                    break;

                case 3:
                    legend.yOffset += 3;
                    break;

                case 17:
                case 18:
                case 19:
                case 21:
                    legend.yOffset += 4;
                    break;

                case 13:
                case 10:
                case 20:
                    legend.yOffset += 5;
                    break;

                case 22:
                case 26:
                    legend.yOffset += 1;
                    break;

                default:
                    break;
            }
        }
        if (! legend.primary.isEmpty() && ! knobCards[index]->getBounds().isEmpty())
        {
            auto anchor = knobCards[index]->getBounds();
            anchor.setHeight (knobCards[index]->slider.getBounds().getBottom());
            drawLegend (anchor, legend);
        }
    }

    const std::array<const char*, 39> buttonKeys {
        "arpEdit", "arpOn",
        "matrixSelect",
        "modEdit", "modEnvMode", "modShape", "modSelect", "modAssign", "modSelectRight",
        "oscSelect", "oscEdit", "osc3On", "mono", "syncPanel",
        "fxSelectA", "fxEditA", "fxSelectB", "fxEditB",
        "displayExit", "displayTap", "displayEdit", "displayConfig", "displayStore", "displayUndo", "displayShift", "displaySearch",
        "partLeft", "part", "multi", "single", "parameters", "parametersNext", "valueProgram", "valueProgramNext",
        "filterEdit", "filter1Focus", "filterMode", "filter2Focus", "filterSelect"
    };

    for (const auto* key : buttonKeys)
    {
        if (auto* button = findVirusPanelButton (key))
        {
            const auto legend = virusButtonLegend (key);
            if (! legend.primary.isEmpty() && ! button->getBounds().isEmpty())
                drawLegend (button->getBounds(), legend);
        }
    }

    drawGroupLegend ({ "matrixDest", "matrixSource" }, { "DESTINATION", {}, false, 42, 1 });
    drawGroupLegend ({ "transposeDown", "transposeUp" }, { "TRANSPOSE", "POWER ON/OFF", false, 24, 1 });
    drawGroupLegend ({ "partLeft", "part" }, { "PART", {}, false, 18, 1 });
    drawGroupLegend ({ "parameters", "parametersNext" }, { "PARAMETERS", "BANK", false, 24, 1 });
    drawGroupLegend ({ "valueProgram", "valueProgramNext" }, { "VALUE", "PROGRAM", false, 18, 1 });
    drawGroupLegend ({ "filter2Focus", "filterSelect" }, { "SELECT", {}, false, 22, 1 });

    drawLabelAboveButton ("modLfo1", "LFO 1", scaledInt (30.0f, uiScale), scaledInt (10.0f, uiScale));
    drawLabelAboveButton ("modLfo2", "LFO 2", scaledInt (30.0f, uiScale), scaledInt (10.0f, uiScale));
    drawLabelAboveButton ("modLfo3", "LFO 3", scaledInt (30.0f, uiScale), scaledInt (10.0f, uiScale));

    drawLabelRightOfButton ("oscLed1", "OSC 1", scaledInt (1.0f, uiScale), scaledInt (28.0f, uiScale));
    drawLabelRightOfButton ("oscLed2", "OSC 2", scaledInt (1.0f, uiScale), scaledInt (28.0f, uiScale));
    drawLabelRightOfButton ("oscLed3", "OSC 3", scaledInt (1.0f, uiScale), scaledInt (28.0f, uiScale));

    drawLabelRightOfButton ("fxDelay", "DELAY", scaledInt (-4.0f, uiScale), scaledInt (20.0f, uiScale), 5.2f);
    drawLabelRightOfButton ("fxReverb", "REVERB", scaledInt (-4.0f, uiScale), scaledInt (26.0f, uiScale), 5.2f);
    drawLabelRightOfButton ("fxLowEq", "LOW-EQ", scaledInt (-4.0f, uiScale), scaledInt (24.0f, uiScale), 5.2f);
    drawLabelRightOfButton ("fxMidEq", "MID-EQ", scaledInt (-4.0f, uiScale), scaledInt (24.0f, uiScale), 5.2f);
    drawLabelRightOfButton ("fxHighEq", "HIGH-EQ", scaledInt (-4.0f, uiScale), scaledInt (28.0f, uiScale), 5.2f);

    drawLabelRightOfButton ("fxDistortion", "DISTORTION", scaledInt (-2.0f, uiScale), scaledInt (38.0f, uiScale), 6.2f);
    drawLabelRightOfButton ("fxCharacter", "CHARACTER", scaledInt (-2.0f, uiScale), scaledInt (38.0f, uiScale), 6.2f);
    drawLabelRightOfButton ("fxChorus", "CHORUS", scaledInt (-2.0f, uiScale), scaledInt (28.0f, uiScale), 6.2f);
    drawLabelRightOfButton ("fxPhaser", "PHASER", scaledInt (-2.0f, uiScale), scaledInt (28.0f, uiScale), 6.2f);
    drawLabelRightOfButton ("fxOthers", "OTHERS", scaledInt (-2.0f, uiScale), scaledInt (28.0f, uiScale), 6.2f);

    if (auto* filter1Led = findVirusPanelButton ("filter1ActiveLed"))
        if (auto* filter1Button = findVirusPanelButton ("filter2Focus"))
        {
            auto ledBounds = filter1Led->getBounds();
            auto buttonBounds = filter1Button->getBounds();
            auto labelBounds = juce::Rectangle<int> (buttonBounds.getCentreX() - scaledInt (13.0f, uiScale),
                                                     ledBounds.getY() - scaledInt (1.0f, uiScale),
                                                     scaledInt (26.0f, uiScale),
                                                     ledBounds.getHeight() + scaledInt (2.0f, uiScale));
            drawInlineIndicatorLabel (labelBounds, "FLT 1", juce::Justification::centred, 5.8f, primaryColour);
        }

    if (auto* filter2Led = findVirusPanelButton ("filter2ActiveLed"))
        if (auto* filter2Button = findVirusPanelButton ("filterSelect"))
        {
            auto ledBounds = filter2Led->getBounds();
            auto buttonBounds = filter2Button->getBounds();
            auto labelBounds = juce::Rectangle<int> (buttonBounds.getCentreX() - scaledInt (13.0f, uiScale),
                                                     ledBounds.getY() - scaledInt (1.0f, uiScale),
                                                     scaledInt (26.0f, uiScale),
                                                     ledBounds.getHeight() + scaledInt (2.0f, uiScale));
            drawInlineIndicatorLabel (labelBounds, "FLT 2", juce::Justification::centred, 5.8f, primaryColour);
        }

    if (auto* leftModeLed = findVirusPanelButton ("filter1ModeBs"))
        if (auto* rightModeLed = findVirusPanelButton ("filter2ModeBs"))
        {
            auto leftBounds = leftModeLed->getBounds();
            auto rightBounds = rightModeLed->getBounds();
            auto labelBounds = juce::Rectangle<int> (leftBounds.getX() - scaledInt (2.0f, uiScale),
                                                     juce::jmax (leftBounds.getBottom(), rightBounds.getBottom()) + scaledInt (6.0f, uiScale),
                                                     (rightBounds.getRight() - leftBounds.getX()) + scaledInt (4.0f, uiScale),
                                                     scaledInt (8.0f, uiScale));
            drawInlineIndicatorLabel (labelBounds, "MODE", juce::Justification::centred, 5.8f, primaryColour);
        }

    for (int slotIndex = 0; slotIndex < 6; ++slotIndex)
        drawLabelRightOfButton ("matrixSlot" + juce::String (slotIndex + 1),
                                "SLOT " + juce::String (slotIndex + 1),
                                scaledInt (1.0f, uiScale),
                                scaledInt (25.0f, uiScale),
                                5.6f);

    if (! showBackground)
    {
        auto drawLedDot = [&] (float centreX, float centreY, bool active)
        {
            const auto radius = scaledFloat (3.2f, uiScale);
            const auto centre = juce::Point<float> (scaledFloat (centreX, uiScale), scaledFloat (centreY, uiScale));
            auto ledBounds = juce::Rectangle<float> (radius * 2.0f, radius * 2.0f).withCentre (centre);
            g.setColour (juce::Colours::black.withAlpha (0.45f));
            g.fillEllipse (ledBounds.translated (0.0f, scaledFloat (0.7f, uiScale)));
            g.setColour ((active ? secondaryColour.brighter (0.25f) : juce::Colour::fromRGB (82, 36, 36)).withAlpha (0.96f));
            g.fillEllipse (ledBounds);
            g.setColour (juce::Colours::white.withAlpha (active ? 0.38f : 0.16f));
            g.drawEllipse (ledBounds.reduced (scaledFloat (0.2f, uiScale)), scaledFloat (0.65f, uiScale));
        };

        auto drawFixedLabel = [&] (float x, float y, float width, const juce::String& text)
        {
            auto labelBounds = juce::Rectangle<int> (scaledInt (x, uiScale),
                                                     scaledInt (y, uiScale),
                                                     scaledInt (width, uiScale),
                                                     scaledInt (8.0f, uiScale));
            drawInlineIndicatorLabel (labelBounds, text, juce::Justification::centredLeft, 5.8f, primaryColour);
        };

        const std::array<juce::String, 5> modLeftLabels { "OSC 1", "OSC 2/3", "PW", "RESO", "FLT GAIN" };
        const std::array<juce::String, 6> modRightLabels { "CUTOFF 1", "CUTOFF 2", "SHAPE", "FM AMT", "PAN", "ASSIGN" };
        int currentLfoDestination = 0;
        if (auto* destinationParam = dynamic_cast<juce::AudioParameterChoice*> (audioProcessor.apvts.getParameter (virusLfoDestinationParamId (juce::jlimit (0, 2, virusModulatorIndex)))))
            currentLfoDestination = juce::jlimit (0, destinationParam->choices.size() - 1, destinationParam->getIndex());

        for (int index = 0; index < static_cast<int> (modLeftLabels.size()); ++index)
        {
            const float y = 142.0f + static_cast<float> (index * 12);
            drawLedDot (222.0f, y + 4.0f, currentLfoDestination == index);
            drawFixedLabel (227.0f, y, 40.0f, modLeftLabels[static_cast<size_t> (index)]);
        }

        for (int index = 0; index < static_cast<int> (modRightLabels.size()); ++index)
        {
            const float y = 140.0f + static_cast<float> (index * 12);
            drawLedDot (300.0f, y + 4.0f, currentLfoDestination == static_cast<int> (modLeftLabels.size()) + index);
            drawFixedLabel (307.0f, y, 42.0f, modRightLabels[static_cast<size_t> (index)]);
        }

        auto selectedShapeIndex = -1;
        const auto selectedShapeParamId = virusModulatorIndex == 0 ? "LFO1SHAPE"
                                        : virusModulatorIndex == 1 ? "LFO2SHAPE"
                                                                   : "LFO3SHAPE";
        if (auto* parameter = dynamic_cast<juce::AudioParameterChoice*> (audioProcessor.apvts.getParameter (selectedShapeParamId)))
            selectedShapeIndex = juce::jlimit (0, static_cast<int> (parameter->choices.size()) - 1, parameter->getIndex());

        const std::array<juce::String, 5> shapeSymbols { "~", "/\\", "/|", "X", "[ ]" };
        for (int index = 0; index < static_cast<int> (shapeSymbols.size()); ++index)
        {
            const float y = 83.0f + static_cast<float> (index * 13);
            const bool active = selectedShapeIndex == index;
            drawLedDot (303.0f, y - 4.0f, active);
            drawFixedLabel (311.0f, y - 6.0f, 22.0f, shapeSymbols[static_cast<size_t> (index)]);
        }
    }

    const std::array<std::pair<const char*, const char*>, 4> filterModeRows {{
        { "filter1ModeLp", "LP" },
        { "filter1ModeHp", "HP" },
        { "filter1ModeBp", "BP" },
        { "filter1ModeBs", "BS" }
    }};

    for (const auto& [leftKey, label] : filterModeRows)
    {
        auto rightKey = juce::String (leftKey).replace ("filter1", "filter2");
        if (auto* leftButton = findVirusPanelButton (leftKey))
        {
            if (auto* rightButton = findVirusPanelButton (rightKey))
            {
                const auto leftBounds = leftButton->getBounds();
                const auto rightBounds = rightButton->getBounds();
                if (! leftBounds.isEmpty() && ! rightBounds.isEmpty())
                {
                    auto labelBounds = juce::Rectangle<int> (leftBounds.getRight() - scaledInt (2.0f, uiScale),
                                                             leftBounds.getY() - scaledInt (1.0f, uiScale),
                                                             juce::jmax (scaledInt (16.0f, uiScale), rightBounds.getX() - leftBounds.getRight() - scaledInt (4.0f, uiScale)),
                                                             leftBounds.getHeight() + scaledInt (2.0f, uiScale));
                    drawInlineIndicatorLabel (labelBounds, label, juce::Justification::centred, 6.2f, primaryColour);
                }
            }
        }
    }

    if (choiceCards.size() > 0
        && choiceCards[0] != nullptr)
    {
        const auto nowMs = juce::Time::getMillisecondCounterHiRes();
        const bool transientOsdActive = virusOsdPinned || virusOsdUntilMs > nowMs;
        juce::String displayTitle = virusOsdTitle;
        juce::String displayValue = virusOsdValue;
        juce::String displayDetail = virusOsdDetail;
        juce::String displayHeaderCenter;
        juce::String displayHeaderRight = "LCD";
        bool usePresetLayout = false;
        int presetIndex = audioProcessor.getCurrentProgram();
        int totalPrograms = audioProcessor.getNumPrograms();

        if (! transientOsdActive || displayValue.isEmpty())
        {
            displayTitle = virusPanelModeIndex == 1 ? "MULTI" : "SINGLE";
            displayValue = audioProcessor.getProgramName (presetIndex);
            if (displayValue.isEmpty())
                displayValue = "Init Preset";
            const auto presetInfo = virusPresetDisplayInfo (displayValue, presetIndex, totalPrograms);
            displayHeaderCenter = "CAT:" + presetInfo.categoryCode;
            displayHeaderRight = presetInfo.bankLabel + " " + presetInfo.slotLabel;
            displayDetail = presetInfo.descriptorA + "|" + presetInfo.descriptorB + "|" + presetInfo.descriptorC;
            usePresetLayout = true;
        }
        else if (displayTitle == "SINGLE" || displayTitle == "MULTI")
        {
            const auto presetInfo = virusPresetDisplayInfo (displayValue, presetIndex, totalPrograms);
            displayHeaderCenter = "CAT:" + presetInfo.categoryCode;
            displayHeaderRight = presetInfo.bankLabel + " " + presetInfo.slotLabel;
            usePresetLayout = true;
        }

        auto baseBounds = choiceCards[0]->getBounds();
        if (! baseBounds.isEmpty() && ! displayValue.isEmpty())
        {
            auto osdBounds = baseBounds.getIntersection (getLocalBounds().reduced (2));

            if (! osdBounds.isEmpty())
            {
                auto bezelBounds = osdBounds.toFloat();
                const auto bezelCorner = scaledFloat (6.0f, uiScale);
                g.setColour (juce::Colour::fromRGB (43, 47, 52).withAlpha (showBackground ? 0.94f : 0.98f));
                g.fillRoundedRectangle (bezelBounds, bezelCorner);
                g.setColour (juce::Colour::fromRGB (12, 14, 18).withAlpha (0.85f));
                g.drawRoundedRectangle (bezelBounds, bezelCorner, scaledFloat (1.2f, uiScale));

                auto lcdBounds = bezelBounds.reduced (scaledFloat (5.0f, uiScale), scaledFloat (4.0f, uiScale));
                juce::ColourGradient lcdFill (juce::Colour::fromRGB (230, 233, 235), lcdBounds.getCentreX(), lcdBounds.getY(),
                                              juce::Colour::fromRGB (198, 204, 207), lcdBounds.getCentreX(), lcdBounds.getBottom(), false);
                g.setGradientFill (lcdFill);
                g.fillRoundedRectangle (lcdBounds, scaledFloat (2.5f, uiScale));
                g.setColour (juce::Colour::fromRGB (88, 94, 100).withAlpha (0.95f));
                g.drawRoundedRectangle (lcdBounds, scaledFloat (2.5f, uiScale), scaledFloat (1.0f, uiScale));

                for (float y = lcdBounds.getY() + scaledFloat (3.0f, uiScale); y < lcdBounds.getBottom(); y += scaledFloat (3.4f, uiScale))
                {
                    g.setColour (juce::Colour::fromRGB (248, 249, 250).withAlpha (0.08f));
                    g.drawLine (lcdBounds.getX() + scaledFloat (2.0f, uiScale), y,
                                lcdBounds.getRight() - scaledFloat (2.0f, uiScale), y,
                                scaledFloat (0.45f, uiScale));
                }

                auto textArea = lcdBounds.toNearestInt().reduced (scaledInt (7.0f, uiScale), scaledInt (4.0f, uiScale));
                auto header = textArea.removeFromTop (scaledInt (9.0f, uiScale));
                auto detailBounds = textArea.removeFromBottom (scaledInt (10.0f, uiScale));
                auto valueBounds = textArea;

                g.setColour (juce::Colour::fromRGB (52, 58, 62));
                g.setFont (juce::Font (juce::FontOptions { 5.2f * uiScale, juce::Font::bold }));
                if (usePresetLayout)
                {
                    auto leftHeader = header.removeFromLeft (scaledInt (32.0f, uiScale));
                    auto rightHeader = header.removeFromRight (scaledInt (34.0f, uiScale));
                    g.drawFittedText (displayTitle, leftHeader, juce::Justification::centredLeft, 1);
                    g.drawFittedText (displayHeaderCenter, header, juce::Justification::centred, 1);
                    g.drawFittedText (displayHeaderRight, rightHeader, juce::Justification::centredRight, 1);
                }
                else
                {
                    g.drawFittedText (displayTitle, header, juce::Justification::centredLeft, 1);
                    g.drawFittedText (displayHeaderRight, header.removeFromRight (scaledInt (26.0f, uiScale)), juce::Justification::centredRight, 1);
                }

                const float valueFontSize = displayValue.length() > 18 ? 9.0f : 12.8f;
                g.setFont (juce::Font (juce::FontOptions { valueFontSize * uiScale, juce::Font::bold }));
                g.setColour (juce::Colour::fromRGB (66, 69, 72));
                g.drawFittedText (displayValue, valueBounds, juce::Justification::centredLeft, 1);

                if (displayDetail.isNotEmpty())
                {
                    g.setFont (juce::Font (juce::FontOptions { 5.5f * uiScale, juce::Font::plain }));
                    g.setColour (juce::Colour::fromRGB (74, 78, 82));
                    if (usePresetLayout && displayDetail.containsChar ('|'))
                    {
                        auto detailParts = juce::StringArray::fromTokens (displayDetail, "|", "");
                        while (detailParts.size() < 3)
                            detailParts.add ({});

                        auto leftThird = detailBounds.removeFromLeft (detailBounds.getWidth() / 3);
                        auto middleThird = detailBounds.removeFromLeft (detailBounds.getWidth() / 2);
                        g.drawFittedText (detailParts[0], leftThird, juce::Justification::centredLeft, 1);
                        g.drawFittedText (detailParts[1], middleThird, juce::Justification::centred, 1);
                        g.drawFittedText (detailParts[2], detailBounds, juce::Justification::centredRight, 1);
                    }
                    else
                    {
                        g.drawFittedText (displayDetail, detailBounds, juce::Justification::centredLeft, 1);
                    }
                }
            }
        }
    }
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
        if (backgroundImage.isValid())
        {
            // Fixed 1120x460 image overlay — absolute pixel positions, no labels
            constexpr int ks = 65;  // knob size in logical pixels
            for (auto* card : knobCards)
                card->setScale (1.0f);
            for (auto* card : choiceCards)
                card->setBounds ({});
            badgeLabel.setBounds ({});
            titleLabel.setBounds ({});
            subtitleLabel.setBounds ({});

            // Knob center rows
            constexpr int r1 = 147;  // top row center Y
            constexpr int r2 = 242;  // bottom row center Y

            auto place = [&] (int idx, int cx, int cy)
            {
                if (juce::isPositiveAndBelow (idx, knobCards.size()))
                    knobCards[idx]->setBounds (cx - ks / 2, cy - ks / 2, ks, ks);
            };

            // Master (0-3): Accent, Volume, Tone, Noise
            place (0,  42, r1);  place (1,  90, r1);
            place (2,  42, r2);  place (3,  90, r2);
            // Bass Drum (4-7): Tune, Level (top), Attack, Decay (bottom)
            place (4, 165, r1);  place (5, 212, r1);
            place (6, 165, r2);  place (7, 212, r2);
            // Snare (8-11): Tune, Level (top), Tone, Snappy (bottom)
            place (8,  283, r1);  place (9,  330, r1);
            place (10, 283, r2);  place (11, 330, r2);
            // Low Tom (12-14): Tune, Level (top), Decay (bottom, centred)
            place (12, 400, r1);  place (13, 447, r1);
            place (14, 423, r2);
            // Mid Tom (15-17): Tune, Level (top), Decay (bottom, centred)
            place (15, 512, r1);  place (16, 559, r1);
            place (17, 535, r2);
            // Hi Tom (18-20): Tune, Level (top), Decay (bottom, centred)
            place (18, 616, r1);  place (19, 654, r1);
            place (20, 635, r2);
            // Rim (21), Clap (22)
            place (21, 717, r1);
            place (22, 788, r1);
            // Hi Hat (23-26): CH Lv, OH Lv (top), CH Dec, OH Dec (bottom)
            place (23, 856, r1);  place (24, 900, r1);
            place (25, 856, r2);  place (26, 900, r2);
            // Cymbal (27-30): Crash Lv, Ride Lv (top), Crash Tun, Ride Tun (bottom)
            place (27,  989, r1);  place (28, 1064, r1);
            place (29,  989, r2);  place (30, 1064, r2);
            // Percussion 31-34 (Cowbell/Clave/Maraca/Perc): hide — no panel area on image
            for (int idx = 31; idx <= 34; ++idx)
                if (juce::isPositiveAndBelow (idx, knobCards.size()))
                    knobCards[idx]->setBounds ({});

            return;
        }

        const auto uiScale = juce::jlimit (0.72f, 1.45f, juce::jmin (getWidth() / 1120.0f, getHeight() / 640.0f));
        const auto s = [uiScale] (float value) { return scaledInt (value, uiScale); };
        badgeLabel.setFont (juce::Font (11.0f * uiScale, juce::Font::bold));
        titleLabel.setFont (juce::Font (28.0f * uiScale, juce::Font::bold));
        subtitleLabel.setFont (juce::Font (13.0f * uiScale, juce::Font::plain));
        for (auto* card : knobCards)
            card->setScale (uiScale);
        for (auto* card : choiceCards)
            card->setScale (uiScale);

        // Match paint() coordinate system exactly
        auto frame = getLocalBounds().reduced (s (12.0f));
        auto hero = frame.removeFromTop (s (82.0f));

        // Paint draws "TR-909" and "RHYTHM COMPOSER" in top 38px of hero;
        // place labels in the remaining hero space below that
        auto labelArea = hero;
        labelArea.removeFromTop (s (38.0f));
        badgeLabel.setBounds ({});
        titleLabel.setBounds (labelArea.removeFromTop (s (24.0f)).reduced (s (10.0f), 0));
        subtitleLabel.setBounds (labelArea.reduced (s (10.0f), 0));

        // Surface layout matching paint() exactly (step strip is purely decorative)
        auto surface = frame.reduced (s (10.0f));
        surface.removeFromTop (s (70.0f));
        auto bottomDecor = surface.removeFromBottom (s (82.0f));
        auto topControls = surface;

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
        for (auto* card : knobCards)
            card->setScale (1.0f);
        for (auto* card : choiceCards)
            card->setScale (1.0f);
        if (virusBackgroundToggle != nullptr)
        {
            virusBackgroundToggle->setScale (0.82f);
            virusBackgroundToggle->setBounds (8, 8, 28, 16);
        }
        if (virusKeyboardToggle != nullptr)
        {
            virusKeyboardToggle->setScale (0.82f);
            virusKeyboardToggle->setBounds (40, 8, 28, 16);
        }

        badgeLabel.setBounds ({});
        titleLabel.setBounds ({});
        subtitleLabel.setBounds ({});

        auto placeChoice = [&] (int index, juce::Rectangle<int> bounds)
        {
            if (juce::isPositiveAndBelow (index, choiceCards.size()))
                choiceCards[index]->setBounds (bounds);
        };

        auto placeVirusPanelButton = [this] (const juce::String& key, juce::Rectangle<int> bounds)
        {
            if (auto* button = findVirusPanelButton (key))
                button->setBounds (bounds);
        };

        auto clearUnused = [&]
        {
            for (int index = 0; index < choiceCards.size(); ++index)
                choiceCards[index]->setBounds ({});
            for (int index = 0; index < knobCards.size(); ++index)
                knobCards[index]->setBounds ({});
            for (auto* button : virusPanelButtons)
                button->setBounds ({});
        };
        clearUnused();

        auto layoutGrid = [&] (juce::Rectangle<int> bounds, std::initializer_list<int> indices, int columns, int horizontalGap, int verticalGap)
        {
            if (indices.size() == 0 || columns <= 0)
                return;

            const int count = static_cast<int> (indices.size());
            const int rows = juce::jmax (1, (count + columns - 1) / columns);
            const int totalWidth = kVirusKnobWidth * columns + horizontalGap * (columns - 1);
            const int totalHeight = kVirusKnobHeight * rows + verticalGap * (rows - 1);
            auto grid = juce::Rectangle<int> (totalWidth, totalHeight).withCentre (bounds.getCentre());

            int localIndex = 0;
            for (int knobIndex : indices)
            {
                if (! juce::isPositiveAndBelow (knobIndex, knobCards.size()))
                    continue;

                const int row = localIndex / columns;
                const int column = localIndex % columns;
                knobCards[knobIndex]->setBounds (grid.getX() + column * (kVirusKnobWidth + horizontalGap),
                                                 grid.getY() + row * (kVirusKnobHeight + verticalGap),
                                                 kVirusKnobWidth,
                                                 kVirusKnobHeight);
                ++localIndex;
            }
        };

        auto placeKnobRow = [this] (std::initializer_list<int> indices, int startX, int y, int gap)
        {
            int localIndex = 0;
            for (int knobIndex : indices)
            {
                if (! juce::isPositiveAndBelow (knobIndex, knobCards.size()))
                    continue;

                knobCards[knobIndex]->setBounds (startX + localIndex * (kVirusKnobWidth + gap),
                                                 y,
                                                 kVirusKnobWidth,
                                                 kVirusKnobHeight);
                ++localIndex;
            }
        };

        auto placeKnobByCentre = [this] (int knobIndex, int centreX, int centreY)
        {
            if (! juce::isPositiveAndBelow (knobIndex, knobCards.size()))
                return;

            knobCards[knobIndex]->setBounds (centreX - (kVirusKnobWidth / 2),
                                             centreY - (kVirusKnobHeight / 2),
                                             kVirusKnobWidth,
                                             kVirusKnobHeight);
        };

        placeKnobByCentre (0, 40, 98);
        placeKnobByCentre (3, 354, 99);

        placeKnobByCentre (7, 428, 98);
        placeKnobByCentre (15, 498, 98);
        placeKnobByCentre (8, 566, 98);
        placeKnobByCentre (16, 639, 98);
        placeKnobByCentre (13, 639, 176);

        placeKnobByCentre (9, 722, 98);
        placeKnobByCentre (10, 722, 177);
        placeKnobByCentre (12, 722, 287);

        placeKnobByCentre (17, 798, 97);
        placeKnobByCentre (18, 865, 98);
        placeKnobByCentre (19, 928, 97);
        placeKnobByCentre (21, 995, 97);
        placeKnobByCentre (20, 798, 177);
        placeKnobByCentre (22, 799, 278);
        placeKnobByCentre (23, 863, 278);
        placeKnobByCentre (24, 929, 278);
        placeKnobByCentre (25, 992, 278);

        if (juce::isPositiveAndBelow (0, choiceCards.size()))
        {
            choiceCards[0]->setBounds ({ 385, 244, 228, 63 });
            choiceCards[0]->setVisible (false);
        }
        layoutGrid ({ 393, 349, 210, kVirusKnobHeight }, { 30, 31, 32 }, 3, 20, 0);
        placeKnobByCentre (35, 122, 288);
        placeKnobByCentre (36, 192, 288);
        placeKnobByCentre (37, 259, 287);
        placeKnobByCentre (33, 121, 375);
        placeKnobByCentre (34, 190, 375);
        placeKnobByCentre (11, 722, 373);
        placeKnobByCentre (26, 799, 371);
        placeKnobByCentre (27, 863, 371);
        placeKnobByCentre (28, 929, 371);
        placeKnobByCentre (29, 992, 371);

        placeVirusPanelButton ("arpEdit", { 25, 153, kVirusSquareButtonSize, 45 });
        placeVirusPanelButton ("arpOn", { 54, 153, kVirusSquareButtonSize, 45 });
        placeVirusPanelButton ("matrixDest", { 101, 87, kVirusSquareButtonSize, kVirusSquareButtonSize });
        placeVirusPanelButton ("matrixSource", { 131, 87, kVirusSquareButtonSize, kVirusSquareButtonSize });
        placeVirusPanelButton ("matrixSelect", { 101, 163, kVirusSquareButtonSize, kVirusSquareButtonSize });
        placeVirusPanelButton ("matrixSlot1", { 133, 140, 14, 10 });
        placeVirusPanelButton ("matrixSlot2", { 133, 151, 14, 10 });
        placeVirusPanelButton ("matrixSlot3", { 133, 162, 14, 10 });
        placeVirusPanelButton ("matrixSlot4", { 133, 173, 14, 10 });
        placeVirusPanelButton ("matrixSlot5", { 133, 184, 14, 10 });
        placeVirusPanelButton ("matrixSlot6", { 133, 195, 14, 10 });
        placeVirusPanelButton ("modLfo1", { 187, 149, 20, 12 });
        placeVirusPanelButton ("modLfo2", { 262, 149, 20, 12 });
        placeVirusPanelButton ("modLfo3", { 347, 150, 20, 12 });
        placeVirusPanelButton ("modEdit", { 181, 75, kVirusSquareButtonSize, 45 });
        placeVirusPanelButton ("modEnvMode", { 210, 75, kVirusSquareButtonSize, 45 });
        placeVirusPanelButton ("modShape", { 260, 75, kVirusSquareButtonSize, 45 });
        placeVirusPanelButton ("modSelect", { 183, 164, kVirusSquareButtonSize, kVirusSquareButtonSize });
        placeVirusPanelButton ("modAssign", { 260, 164, kVirusSquareButtonSize, kVirusSquareButtonSize });
        placeVirusPanelButton ("modSelectRight", { 341, 164, kVirusSquareButtonSize, kVirusSquareButtonSize });
        placeVirusPanelButton ("oscLed1", { 406, 141, 16, 10 });
        placeVirusPanelButton ("oscLed2", { 437, 141, 16, 10 });
        placeVirusPanelButton ("oscLed3", { 468, 141, 16, 10 });
        placeVirusPanelButton ("oscSelect", { 398, 152, kVirusSquareButtonSize, 45 });
        placeVirusPanelButton ("oscEdit", { 427, 152, kVirusSquareButtonSize, 45 });
        placeVirusPanelButton ("osc3On", { 459, 152, kVirusSquareButtonSize, 45 });
        placeVirusPanelButton ("mono", { 540, 152, kVirusSquareButtonSize, 45 });
        placeVirusPanelButton ("syncPanel", { 570, 152, kVirusSquareButtonSize, 45 });
        placeVirusPanelButton ("fxSelectA", { 25, 271, kVirusSquareButtonSize, kVirusSquareButtonSize });
        placeVirusPanelButton ("fxEditA", { 58, 271, kVirusSquareButtonSize, kVirusSquareButtonSize });
        placeVirusPanelButton ("fxSelectB", { 26, 357, kVirusSquareButtonSize, kVirusSquareButtonSize });
        placeVirusPanelButton ("fxEditB", { 59, 357, kVirusSquareButtonSize, kVirusSquareButtonSize });
        placeVirusPanelButton ("fxDelay", { 28, 242, 24, 14 });
        placeVirusPanelButton ("fxReverb", { 80, 242, 24, 14 });
        placeVirusPanelButton ("fxLowEq", { 132, 242, 24, 14 });
        placeVirusPanelButton ("fxMidEq", { 184, 242, 24, 14 });
        placeVirusPanelButton ("fxHighEq", { 236, 242, 24, 14 });
        placeVirusPanelButton ("fxDistortion", { 24, 325, 24, 14 });
        placeVirusPanelButton ("fxCharacter", { 78, 325, 24, 14 });
        placeVirusPanelButton ("fxChorus", { 132, 325, 24, 14 });
        placeVirusPanelButton ("fxPhaser", { 186, 325, 24, 14 });
        placeVirusPanelButton ("fxOthers", { 240, 325, 24, 14 });
        placeVirusPanelButton ("transposeDown", { 232, 354, kVirusSquareButtonSize, 45 });
        placeVirusPanelButton ("transposeUp", { 263, 354, kVirusSquareButtonSize, 45 });
        placeVirusPanelButton ("displayExit", { 312, 225, kVirusSquareButtonSize, kVirusSquareButtonSize });
        placeVirusPanelButton ("displayTap", { 342, 225, kVirusSquareButtonSize, kVirusSquareButtonSize });
        placeVirusPanelButton ("displayEdit", { 312, 266, kVirusSquareButtonSize, 45 });
        placeVirusPanelButton ("displayConfig", { 342, 266, kVirusSquareButtonSize, 45 });
        placeVirusPanelButton ("displayStore", { 312, 322, kVirusSquareButtonSize, kVirusSquareButtonSize });
        placeVirusPanelButton ("displayUndo", { 342, 322, kVirusSquareButtonSize, kVirusSquareButtonSize });
        placeVirusPanelButton ("displayShift", { 312, 367, kVirusSquareButtonSize, kVirusSquareButtonSize });
        placeVirusPanelButton ("displaySearch", { 342, 367, kVirusSquareButtonSize, kVirusSquareButtonSize });
        placeVirusPanelButton ("partLeft", { 617, 226, kVirusSquareButtonSize, kVirusSquareButtonSize });
        placeVirusPanelButton ("part", { 647, 226, kVirusSquareButtonSize, kVirusSquareButtonSize });
        placeVirusPanelButton ("multi", { 617, 275, kVirusSquareButtonSize, kVirusSquareButtonSize });
        placeVirusPanelButton ("single", { 647, 275, kVirusSquareButtonSize, kVirusSquareButtonSize });
        placeVirusPanelButton ("parameters", { 617, 322, kVirusSquareButtonSize, kVirusSquareButtonSize });
        placeVirusPanelButton ("parametersNext", { 647, 322, kVirusSquareButtonSize, kVirusSquareButtonSize });
        placeVirusPanelButton ("valueProgram", { 617, 367, kVirusSquareButtonSize, kVirusSquareButtonSize });
        placeVirusPanelButton ("valueProgramNext", { 647, 367, kVirusSquareButtonSize, kVirusSquareButtonSize });
        placeVirusPanelButton ("filterEdit", { 832, 164, kVirusSquareButtonSize, kVirusSquareButtonSize });
        placeVirusPanelButton ("filter1Focus", { 861, 164, kVirusSquareButtonSize, kVirusSquareButtonSize });
        placeVirusPanelButton ("filterMode", { 931, 164, kVirusSquareButtonSize, kVirusSquareButtonSize });
        placeVirusPanelButton ("filter2Focus", { 963, 164, kVirusSquareButtonSize, kVirusSquareButtonSize });
        placeVirusPanelButton ("filterSelect", { 993, 164, kVirusSquareButtonSize, kVirusSquareButtonSize });
        placeVirusPanelButton ("filter1ActiveLed", { 960, 152, 12, 10 });
        placeVirusPanelButton ("filter2ActiveLed", { 990, 152, 12, 10 });
        placeVirusPanelButton ("filter1ModeLp", { 893, 146, 12, 10 });
        placeVirusPanelButton ("filter1ModeHp", { 893, 160, 12, 10 });
        placeVirusPanelButton ("filter1ModeBp", { 893, 174, 12, 10 });
        placeVirusPanelButton ("filter1ModeBs", { 893, 188, 12, 10 });
        placeVirusPanelButton ("filter2ModeLp", { 918, 146, 12, 10 });
        placeVirusPanelButton ("filter2ModeHp", { 918, 160, 12, 10 });
        placeVirusPanelButton ("filter2ModeBp", { 918, 174, 12, 10 });
        placeVirusPanelButton ("filter2ModeBs", { 918, 188, 12, 10 });

        if (virusKeyboard != nullptr)
        {
            if (virusKeyboardVisible)
                virusKeyboard->setBounds (12, kVirusTemplateHeight + 8, getWidth() - 24, kVirusKeyboardExtraHeight - 18);
            else
                virusKeyboard->setBounds ({});
        }

        for (auto* pad : drumPads)
            pad->setBounds ({});
        return;
    }

    if (virusBackgroundToggle != nullptr)
        virusBackgroundToggle->setBounds ({});
    if (virusKeyboardToggle != nullptr)
        virusKeyboardToggle->setBounds ({});
    if (virusKeyboard != nullptr)
        virusKeyboard->setBounds ({});

    if (usesDrumPadLayout())
    {
        const bool vecPadLayout = audioProcessor.isVec1DrumPadFlavor();
        const auto uiScale = vecPadLayout
                                 ? juce::jlimit (0.72f, 1.15f, juce::jmin (getWidth() / 1280.0f, getHeight() / 900.0f))
                                 : juce::jlimit (0.8f, 1.4f, juce::jmin (getWidth() / 1180.0f, getHeight() / 860.0f));
        badgeLabel.setFont (juce::Font ((vecPadLayout ? 10.0f : 11.0f) * uiScale, juce::Font::bold));
        titleLabel.setFont (juce::Font ((vecPadLayout ? 24.0f : 28.0f) * uiScale, juce::Font::bold));
        subtitleLabel.setFont (juce::Font ((vecPadLayout ? 11.5f : 13.0f) * uiScale, juce::Font::plain));
        for (auto* card : knobCards)
            card->setScale (uiScale);
        for (auto* card : choiceCards)
            card->setScale (uiScale);

        auto area = getLocalBounds().reduced (isTribute909() ? 26 : (vecPadLayout ? 18 : 22));
        auto hero = area.removeFromTop (isTribute909() ? 88 : (vecPadLayout ? 82 : 96));
        badgeLabel.setBounds (hero.removeFromTop (18));
        titleLabel.setBounds (hero.removeFromTop (isTribute909() ? 40 : (vecPadLayout ? 32 : 36)));
        subtitleLabel.setBounds (hero.removeFromTop (vecPadLayout ? 28 : 36));
        area.removeFromTop (vecPadLayout ? 4 : (isTribute909() ? 18 : 10));

        if (! vecPadLayout)
        {
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
        }

        auto padArea = area;
        const int spacing = vecPadLayout ? 10 : (isTribute909() ? 10 : 14);
        const int columns = vecPadLayout ? 5 : (isTribute909() ? 5 : 4);
        const int rows = vecPadLayout ? juce::jmax (1, (drumPads.size() + columns - 1) / columns)
                                      : (isTribute909() ? 3 : 4);
        const int padWidth = juce::jmax (80, (padArea.getWidth() - ((columns - 1) * spacing)) / columns);
        const int padHeight = juce::jmax (116, (padArea.getHeight() - ((rows - 1) * spacing)) / rows);

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
    const int columnCount = area.getWidth() > 900 ? 5 : (area.getWidth() > 720 ? 4 : (area.getWidth() > 560 ? 3 : 2));
    const int rowCount = juce::jmax (1, (knobCards.size() + columnCount - 1) / columnCount);
    const int cardWidth = (area.getWidth() - ((columnCount - 1) * spacing)) / columnCount;
    const int computedCardHeight = rowCount > 0 ? (area.getHeight() - ((rowCount - 1) * spacing)) / rowCount : 220;
    const int cardHeight = juce::jlimit (105, 220, computedCardHeight);
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
