#include "MainWindow.h"
#include "AudioExport.h"
#include <BinaryData.h>

#include <array>
#include <atomic>
#include <chrono>
#include <stdexcept>

namespace aims
{
namespace
{
juce::PropertiesFile::Options nativeWindowSettingsOptions()
{
    juce::PropertiesFile::Options options;
    options.applicationName = "Mutagen";
    options.filenameSuffix = "settings";
    options.osxLibrarySubFolder = "Application Support";
    return options;
}

juce::File nativeLogsDirectory()
{
    return juce::File::getSpecialLocation(juce::File::userApplicationDataDirectory)
        .getChildFile("Mutagen")
        .getChildFile("logs");
}

juce::Image loadMutagenLogoBinaryData(bool preferSmallLogo)
{
    const auto loadFromBinary = [] (const void* data, int dataSize)
    {
        juce::MemoryInputStream stream(data, static_cast<size_t>(dataSize), false);
        return juce::ImageFileFormat::loadFrom(stream);
    };

    if (preferSmallLogo)
    {
        auto image = loadFromBinary(BinaryData::mutagenlogo80_png,
                                    BinaryData::mutagenlogo80_pngSize);
        if (image.isValid())
            return image;
    }

    return loadFromBinary(BinaryData::mutagenlogosource_png,
                          BinaryData::mutagenlogosource_pngSize);
}

int playbackRefreshRateForComponent(const juce::Component& component)
{
    const auto* display = juce::Desktop::getInstance().getDisplays().getDisplayForRect(component.getScreenBounds());
    const auto hz = display != nullptr && display->verticalFrequencyHz.has_value()
        ? *display->verticalFrequencyHz
        : 60.0;
    return juce::jlimit(60, 240, juce::roundToInt(hz));
}

struct SequenceTickOption
{
    int ticks;
    const char* label;
};

struct KeyQuantizeOption
{
    int id;
    int root;
    juce::String scaleId;
    juce::String label;
};

struct VirtualPianoKeySpec
{
    int pitch;
    const char* primary;
    std::initializer_list<const char*> aliases;
};

struct ThemeSpec
{
    const char* name;
    juce::LookAndFeel_V4::ColourScheme scheme;
    juce::Colour mainBackground;
    juce::Colour headerStart;
    juce::Colour headerMid;
    juce::Colour headerEnd;
    juce::Colour surface;
    juce::Colour surfaceAlt;
    juce::Colour outline;
    juce::Colour primaryText;
    juce::Colour secondaryText;
    juce::Colour successText;
    juce::Colour infoText;
    juce::Colour warningText;
    juce::Colour editorPlaceholder;
    juce::Colour lcdBackground;
    juce::Colour lcdFrame;
    juce::Colour lcdGlow;
    juce::Colour lcdLabel;
    juce::Colour lcdGhost;
    juce::Colour lcdValue;
    juce::Colour buttonOff;
    juce::Colour buttonOn;
    juce::Colour buttonText;
};

const std::array<ThemeSpec, 8>& availableThemeSpecs()
{
    using Scheme = juce::LookAndFeel_V4::ColourScheme;

    static const std::array<ThemeSpec, 8> themes { {
        { "Mutagen Dark",
          Scheme(juce::Colour::fromRGB(14, 18, 24), juce::Colour::fromRGB(44, 50, 62), juce::Colour::fromRGB(18, 22, 28),
                 juce::Colour::fromRGB(62, 71, 86), juce::Colour::fromRGB(232, 238, 246), juce::Colour::fromRGB(72, 180, 138),
                 juce::Colour::fromRGB(255, 255, 255), juce::Colour::fromRGB(78, 128, 196), juce::Colour::fromRGB(232, 238, 246)),
          juce::Colour::fromRGB(13, 15, 20), juce::Colour::fromRGB(26, 35, 52), juce::Colour::fromRGB(17, 21, 28),
          juce::Colour::fromRGB(12, 14, 18), juce::Colour::fromRGB(20, 22, 28), juce::Colour::fromRGB(48, 54, 66),
          juce::Colour::fromRGB(56, 64, 79), juce::Colour::fromRGB(235, 239, 244), juce::Colour::fromRGB(190, 199, 210),
          juce::Colour::fromRGB(143, 225, 170), juce::Colour::fromRGB(156, 199, 239), juce::Colour::fromRGB(230, 190, 118),
          juce::Colour::fromRGB(122, 133, 149), juce::Colour::fromRGB(14, 22, 18), juce::Colour::fromRGB(58, 94, 72),
          juce::Colour::fromRGBA(120, 255, 174, 22), juce::Colour::fromRGB(135, 214, 164), juce::Colour::fromRGBA(106, 170, 121, 52),
          juce::Colour::fromRGB(175, 255, 196), juce::Colour::fromRGB(48, 54, 66), juce::Colour::fromRGB(72, 104, 160),
          juce::Colours::white },
        { "Graphite",
          Scheme(juce::Colour::fromRGB(20, 21, 25), juce::Colour::fromRGB(52, 56, 64), juce::Colour::fromRGB(22, 24, 28),
                 juce::Colour::fromRGB(83, 89, 99), juce::Colour::fromRGB(236, 239, 243), juce::Colour::fromRGB(154, 160, 171),
                 juce::Colour::fromRGB(255, 255, 255), juce::Colour::fromRGB(120, 150, 196), juce::Colour::fromRGB(236, 239, 243)),
          juce::Colour::fromRGB(15, 16, 19), juce::Colour::fromRGB(38, 40, 46), juce::Colour::fromRGB(28, 30, 36),
          juce::Colour::fromRGB(16, 17, 21), juce::Colour::fromRGB(22, 24, 29), juce::Colour::fromRGB(54, 58, 66),
          juce::Colour::fromRGB(74, 80, 92), juce::Colour::fromRGB(236, 239, 243), juce::Colour::fromRGB(188, 194, 203),
          juce::Colour::fromRGB(152, 214, 182), juce::Colour::fromRGB(165, 196, 232), juce::Colour::fromRGB(230, 196, 128),
          juce::Colour::fromRGB(128, 134, 146), juce::Colour::fromRGB(24, 26, 30), juce::Colour::fromRGB(78, 84, 92),
          juce::Colour::fromRGBA(192, 212, 230, 20), juce::Colour::fromRGB(180, 188, 196), juce::Colour::fromRGBA(154, 160, 171, 52),
          juce::Colour::fromRGB(235, 239, 243), juce::Colour::fromRGB(54, 58, 66), juce::Colour::fromRGB(94, 112, 154),
          juce::Colour::fromRGB(244, 246, 250) },
        { "Midnight Blue",
          Scheme(juce::Colour::fromRGB(10, 16, 28), juce::Colour::fromRGB(36, 50, 74), juce::Colour::fromRGB(12, 20, 34),
                 juce::Colour::fromRGB(66, 92, 126), juce::Colour::fromRGB(228, 238, 250), juce::Colour::fromRGB(78, 200, 224),
                 juce::Colour::fromRGB(255, 255, 255), juce::Colour::fromRGB(72, 136, 232), juce::Colour::fromRGB(228, 238, 250)),
          juce::Colour::fromRGB(8, 12, 20), juce::Colour::fromRGB(16, 32, 58), juce::Colour::fromRGB(12, 24, 44),
          juce::Colour::fromRGB(8, 12, 22), juce::Colour::fromRGB(12, 22, 38), juce::Colour::fromRGB(40, 56, 82),
          juce::Colour::fromRGB(62, 86, 118), juce::Colour::fromRGB(230, 239, 250), juce::Colour::fromRGB(176, 196, 222),
          juce::Colour::fromRGB(128, 224, 182), juce::Colour::fromRGB(120, 196, 255), juce::Colour::fromRGB(232, 196, 126),
          juce::Colour::fromRGB(122, 142, 170), juce::Colour::fromRGB(12, 24, 30), juce::Colour::fromRGB(42, 88, 118),
          juce::Colour::fromRGBA(126, 224, 255, 18), juce::Colour::fromRGB(146, 224, 234), juce::Colour::fromRGBA(118, 170, 214, 52),
          juce::Colour::fromRGB(188, 250, 255), juce::Colour::fromRGB(40, 56, 82), juce::Colour::fromRGB(58, 118, 204),
          juce::Colour::fromRGB(240, 247, 255) },
        { "Oxide Amber",
          Scheme(juce::Colour::fromRGB(28, 18, 12), juce::Colour::fromRGB(70, 46, 30), juce::Colour::fromRGB(24, 16, 12),
                 juce::Colour::fromRGB(112, 78, 54), juce::Colour::fromRGB(248, 235, 216), juce::Colour::fromRGB(236, 160, 76),
                 juce::Colour::fromRGB(32, 22, 14), juce::Colour::fromRGB(212, 118, 54), juce::Colour::fromRGB(248, 235, 216)),
          juce::Colour::fromRGB(18, 12, 9), juce::Colour::fromRGB(54, 30, 16), juce::Colour::fromRGB(35, 22, 13),
          juce::Colour::fromRGB(16, 10, 8), juce::Colour::fromRGB(26, 18, 13), juce::Colour::fromRGB(72, 48, 31),
          juce::Colour::fromRGB(94, 66, 44), juce::Colour::fromRGB(246, 236, 222), juce::Colour::fromRGB(214, 192, 170),
          juce::Colour::fromRGB(188, 224, 154), juce::Colour::fromRGB(244, 188, 118), juce::Colour::fromRGB(255, 208, 132),
          juce::Colour::fromRGB(162, 138, 118), juce::Colour::fromRGB(22, 18, 12), juce::Colour::fromRGB(112, 74, 32),
          juce::Colour::fromRGBA(255, 196, 110, 18), juce::Colour::fromRGB(255, 206, 132), juce::Colour::fromRGBA(190, 132, 76, 54),
          juce::Colour::fromRGB(255, 224, 176), juce::Colour::fromRGB(72, 48, 31), juce::Colour::fromRGB(132, 86, 42),
          juce::Colour::fromRGB(255, 242, 226) },
        { "Volt Green",
          Scheme(juce::Colour::fromRGB(10, 16, 10), juce::Colour::fromRGB(28, 52, 30), juce::Colour::fromRGB(10, 18, 12),
                 juce::Colour::fromRGB(60, 98, 64), juce::Colour::fromRGB(232, 246, 232), juce::Colour::fromRGB(126, 232, 94),
                 juce::Colour::fromRGB(10, 18, 12), juce::Colour::fromRGB(82, 182, 110), juce::Colour::fromRGB(232, 246, 232)),
          juce::Colour::fromRGB(8, 12, 8), juce::Colour::fromRGB(16, 40, 18), juce::Colour::fromRGB(12, 28, 14),
          juce::Colour::fromRGB(8, 12, 8), juce::Colour::fromRGB(12, 20, 12), juce::Colour::fromRGB(28, 56, 30),
          juce::Colour::fromRGB(48, 84, 52), juce::Colour::fromRGB(232, 246, 232), juce::Colour::fromRGB(176, 204, 176),
          juce::Colour::fromRGB(154, 236, 132), juce::Colour::fromRGB(146, 224, 178), juce::Colour::fromRGB(214, 214, 126),
          juce::Colour::fromRGB(120, 146, 122), juce::Colour::fromRGB(10, 24, 12), juce::Colour::fromRGB(42, 96, 46),
          juce::Colour::fromRGBA(118, 255, 108, 18), juce::Colour::fromRGB(168, 246, 160), juce::Colour::fromRGBA(108, 180, 102, 52),
          juce::Colour::fromRGB(198, 255, 190), juce::Colour::fromRGB(28, 56, 30), juce::Colour::fromRGB(54, 118, 62),
          juce::Colour::fromRGB(242, 250, 242) },
        { "Crimson Night",
          Scheme(juce::Colour::fromRGB(24, 12, 18), juce::Colour::fromRGB(64, 34, 50), juce::Colour::fromRGB(22, 12, 18),
                 juce::Colour::fromRGB(104, 58, 84), juce::Colour::fromRGB(246, 232, 238), juce::Colour::fromRGB(226, 92, 136),
                 juce::Colour::fromRGB(255, 255, 255), juce::Colour::fromRGB(192, 84, 148), juce::Colour::fromRGB(246, 232, 238)),
          juce::Colour::fromRGB(14, 10, 16), juce::Colour::fromRGB(44, 18, 34), juce::Colour::fromRGB(30, 14, 24),
          juce::Colour::fromRGB(12, 8, 14), juce::Colour::fromRGB(22, 14, 22), juce::Colour::fromRGB(62, 36, 52),
          juce::Colour::fromRGB(88, 56, 74), juce::Colour::fromRGB(246, 232, 238), juce::Colour::fromRGB(214, 182, 196),
          juce::Colour::fromRGB(164, 236, 186), juce::Colour::fromRGB(232, 158, 188), juce::Colour::fromRGB(238, 186, 122),
          juce::Colour::fromRGB(158, 132, 144), juce::Colour::fromRGB(20, 12, 18), juce::Colour::fromRGB(96, 44, 68),
          juce::Colour::fromRGBA(255, 134, 156, 18), juce::Colour::fromRGB(255, 164, 178), juce::Colour::fromRGBA(188, 88, 118, 52),
          juce::Colour::fromRGB(255, 192, 204), juce::Colour::fromRGB(62, 36, 52), juce::Colour::fromRGB(118, 52, 86),
          juce::Colour::fromRGB(250, 240, 244) },
        { "Ultraviolet",
          Scheme(juce::Colour::fromRGB(14, 10, 24), juce::Colour::fromRGB(42, 30, 72), juce::Colour::fromRGB(16, 12, 28),
                 juce::Colour::fromRGB(74, 60, 118), juce::Colour::fromRGB(236, 232, 250), juce::Colour::fromRGB(154, 112, 244),
                 juce::Colour::fromRGB(255, 255, 255), juce::Colour::fromRGB(112, 108, 226), juce::Colour::fromRGB(236, 232, 250)),
          juce::Colour::fromRGB(10, 8, 18), juce::Colour::fromRGB(24, 18, 52), juce::Colour::fromRGB(18, 14, 34),
          juce::Colour::fromRGB(10, 8, 18), juce::Colour::fromRGB(18, 14, 28), juce::Colour::fromRGB(44, 34, 74),
          juce::Colour::fromRGB(72, 60, 108), juce::Colour::fromRGB(238, 234, 250), juce::Colour::fromRGB(194, 186, 220),
          juce::Colour::fromRGB(172, 232, 188), juce::Colour::fromRGB(188, 172, 255), juce::Colour::fromRGB(230, 196, 126),
          juce::Colour::fromRGB(138, 132, 162), juce::Colour::fromRGB(16, 12, 28), juce::Colour::fromRGB(64, 52, 118),
          juce::Colour::fromRGBA(182, 148, 255, 18), juce::Colour::fromRGB(200, 188, 255), juce::Colour::fromRGBA(142, 118, 214, 52),
          juce::Colour::fromRGB(220, 212, 255), juce::Colour::fromRGB(44, 34, 74), juce::Colour::fromRGB(80, 68, 150),
          juce::Colour::fromRGB(244, 240, 255) },
        { "Slate Ice",
          Scheme(juce::Colour::fromRGB(18, 24, 28), juce::Colour::fromRGB(48, 60, 68), juce::Colour::fromRGB(18, 26, 30),
                 juce::Colour::fromRGB(86, 100, 110), juce::Colour::fromRGB(236, 242, 245), juce::Colour::fromRGB(118, 198, 214),
                 juce::Colour::fromRGB(18, 24, 28), juce::Colour::fromRGB(112, 164, 198), juce::Colour::fromRGB(236, 242, 245)),
          juce::Colour::fromRGB(12, 16, 20), juce::Colour::fromRGB(24, 34, 42), juce::Colour::fromRGB(18, 25, 31),
          juce::Colour::fromRGB(12, 16, 20), juce::Colour::fromRGB(16, 22, 28), juce::Colour::fromRGB(48, 60, 68),
          juce::Colour::fromRGB(74, 86, 94), juce::Colour::fromRGB(236, 242, 245), juce::Colour::fromRGB(194, 204, 208),
          juce::Colour::fromRGB(168, 228, 194), juce::Colour::fromRGB(166, 214, 232), juce::Colour::fromRGB(226, 198, 132),
          juce::Colour::fromRGB(132, 144, 152), juce::Colour::fromRGB(16, 22, 26), juce::Colour::fromRGB(52, 86, 96),
          juce::Colour::fromRGBA(158, 228, 236, 18), juce::Colour::fromRGB(184, 230, 236), juce::Colour::fromRGBA(132, 170, 178, 52),
          juce::Colour::fromRGB(214, 248, 250), juce::Colour::fromRGB(48, 60, 68), juce::Colour::fromRGB(78, 120, 148),
          juce::Colour::fromRGB(244, 248, 250) }
    } };

    return themes;
}

const ThemeSpec& themeSpecForIndex(int index)
{
    const auto& themes = availableThemeSpecs();
    return themes[static_cast<size_t>(juce::jlimit(0, static_cast<int>(themes.size()) - 1, index))];
}

void applyThemeToComponentTree(juce::Component& component, const ThemeSpec& theme)
{
    if (auto* button = dynamic_cast<juce::TextButton*>(&component))
    {
        button->setColour(juce::TextButton::buttonColourId, theme.buttonOff);
        button->setColour(juce::TextButton::buttonOnColourId, theme.buttonOn);
        button->setColour(juce::TextButton::textColourOffId, theme.buttonText);
        button->setColour(juce::TextButton::textColourOnId, theme.buttonText);
    }
    else if (auto* toggle = dynamic_cast<juce::ToggleButton*>(&component))
    {
        toggle->setColour(juce::ToggleButton::textColourId, theme.primaryText);
        toggle->setColour(juce::ToggleButton::tickColourId, theme.buttonOn);
        toggle->setColour(juce::ToggleButton::tickDisabledColourId, theme.outline);
    }
    else if (auto* editor = dynamic_cast<juce::TextEditor*>(&component))
    {
        editor->setColour(juce::TextEditor::backgroundColourId, theme.surface);
        editor->setColour(juce::TextEditor::textColourId, theme.primaryText);
        editor->setColour(juce::TextEditor::outlineColourId, theme.outline);
        editor->setColour(juce::TextEditor::focusedOutlineColourId, theme.buttonOn);
        editor->setColour(juce::CaretComponent::caretColourId, theme.primaryText);
    }
    else if (auto* combo = dynamic_cast<juce::ComboBox*>(&component))
    {
        combo->setColour(juce::ComboBox::backgroundColourId, theme.surface);
        combo->setColour(juce::ComboBox::textColourId, theme.primaryText);
        combo->setColour(juce::ComboBox::outlineColourId, theme.outline);
        combo->setColour(juce::ComboBox::buttonColourId, theme.surfaceAlt);
        combo->setColour(juce::ComboBox::arrowColourId, theme.secondaryText);
    }
    else if (auto* slider = dynamic_cast<juce::Slider*>(&component))
    {
        slider->setColour(juce::Slider::backgroundColourId, theme.surface);
        slider->setColour(juce::Slider::trackColourId, theme.buttonOn);
        slider->setColour(juce::Slider::thumbColourId, theme.lcdValue);
        slider->setColour(juce::Slider::textBoxBackgroundColourId, theme.surface);
        slider->setColour(juce::Slider::textBoxTextColourId, theme.primaryText);
        slider->setColour(juce::Slider::textBoxOutlineColourId, theme.outline);
    }
    else if (auto* list = dynamic_cast<juce::ListBox*>(&component))
    {
        list->setColour(juce::ListBox::backgroundColourId, theme.surface);
        list->setColour(juce::ListBox::outlineColourId, theme.outline);
        list->setColour(juce::ListBox::textColourId, theme.primaryText);
    }
    else if (auto* header = dynamic_cast<juce::TableHeaderComponent*>(&component))
    {
        header->setColour(juce::TableHeaderComponent::backgroundColourId, theme.surfaceAlt);
        header->setColour(juce::TableHeaderComponent::outlineColourId, theme.outline);
        header->setColour(juce::TableHeaderComponent::textColourId, theme.primaryText);
    }
    else if (auto* label = dynamic_cast<juce::Label*>(&component))
    {
        label->setColour(juce::Label::textColourId, theme.primaryText);
    }

    for (auto* child : component.getChildren())
        applyThemeToComponentTree(*child, theme);
}

const std::array<SequenceTickOption, 10>& sequenceTickOptions()
{
    static const std::array<SequenceTickOption, 10> options { {
        { kTicksPerBeat / 16, "1/64 note" },
        { kTicksPerBeat / 8, "1/32 note" },
        { kTicksPerBeat / 4, "1/16 note" },
        { kTicksPerBeat / 2, "1/8 note" },
        { kTicksPerBeat, "1 beat" },
        { kTicksPerBeat * 2, "1/2 note" },
        { kTicksPerBar, "1 bar" },
        { kTicksPerBar * 2, "2 bars" },
        { kTicksPerBar * 4, "4 bars" },
        { kTicksPerBar * 8, "8 bars" }
    } };

    return options;
}

juce::String sequenceTickLabel(int ticks)
{
    const auto normalisedTicks = juce::jmax(kMinSequenceSnapTicks, ticks);
    for (const auto& option : sequenceTickOptions())
    {
        if (option.ticks == normalisedTicks)
            return option.label;
    }

    if (normalisedTicks % kTicksPerBar == 0)
    {
        const auto bars = normalisedTicks / kTicksPerBar;
        return juce::String(bars) + (bars == 1 ? " bar" : " bars");
    }

    if (normalisedTicks % kTicksPerBeat == 0)
    {
        const auto beats = normalisedTicks / kTicksPerBeat;
        return juce::String(beats) + (beats == 1 ? " beat" : " beats");
    }

    return juce::String(normalisedTicks) + " ticks";
}

const std::vector<KeyQuantizeOption>& keyQuantizeOptions()
{
    static const std::vector<KeyQuantizeOption> options = []
    {
        std::vector<KeyQuantizeOption> values;
        values.push_back({ 1, 0, "chromatic", "All Notes (Chromatic)" });

        int nextId = 2;
        const auto scales = availableKeyQuantizeScales();
        for (int root = 0; root < 12; ++root)
        {
            for (const auto& scaleId : scales)
            {
                if (scaleId.equalsIgnoreCase("chromatic"))
                    continue;

                values.push_back({ nextId++, root, scaleId, keyQuantizeDisplayName(root, scaleId) });
            }
        }

        return values;
    }();

    return options;
}

int keyQuantizeOptionId(const ProjectState& project)
{
    const auto scaleId = normaliseKeyQuantizeScale(project.keyQuantizeScale);
    const auto root = juce::negativeAwareModulo(project.keyQuantizeRoot, 12);
    for (const auto& option : keyQuantizeOptions())
    {
        if (option.root == root && option.scaleId.equalsIgnoreCase(scaleId))
            return option.id;
    }

    return 1;
}

const KeyQuantizeOption* findKeyQuantizeOptionById(int id)
{
    for (const auto& option : keyQuantizeOptions())
    {
        if (option.id == id)
            return &option;
    }

    return nullptr;
}

const std::array<VirtualPianoKeySpec, 24>& virtualPianoKeySpecs()
{
    static const std::array<VirtualPianoKeySpec, 24> specs { {
        { 60, "Z", {} },
        { 61, "S", {} },
        { 62, "X", {} },
        { 63, "D", {} },
        { 64, "C", {} },
        { 65, "V", {} },
        { 66, "G", {} },
        { 67, "B", {} },
        { 68, "H", {} },
        { 69, "N", {} },
        { 70, "J", {} },
        { 71, "M", {} },
        { 72, "Q", { "," } },
        { 73, "2", { "L" } },
        { 74, "W", { "." } },
        { 75, "3", { ";" } },
        { 76, "E", { "/" } },
        { 77, "R", {} },
        { 78, "5", {} },
        { 79, "T", {} },
        { 80, "6", {} },
        { 81, "Y", {} },
        { 82, "7", {} },
        { 83, "U", {} }
    } };

    return specs;
}

juce::String virtualPianoHintText()
{
    return "Click the keys or play from the computer keyboard. "
           "Lower octave: Z S X D C V G B H N J M. "
           "Upper octave: Q 2 W 3 E R 5 T 6 Y 7 U. "
           "Aliases: , L . ; /.";
}

juce::String virtualPianoShortcutLabel(const VirtualPianoKeySpec& spec)
{
    juce::StringArray labels;
    labels.add(spec.primary);
    for (const auto* alias : spec.aliases)
        labels.add(alias);
    return labels.joinIntoString(" / ");
}

juce::String formatLcdSeconds(double seconds)
{
    auto clamped = juce::jmax(0.0, seconds);
    const auto wholeSeconds = static_cast<int>(std::floor(clamped));
    const auto centiseconds = juce::jlimit(0, 99, juce::roundToInt((clamped - static_cast<double>(wholeSeconds)) * 100.0));
    const auto secondsPart = wholeSeconds % 60;
    const auto minutesPart = (wholeSeconds / 60) % 60;
    const auto hoursPart = wholeSeconds / 3600;

    if (hoursPart > 0)
        return juce::String(hoursPart) + ":" + juce::String(minutesPart).paddedLeft('0', 2)
            + ":" + juce::String(secondsPart).paddedLeft('0', 2)
            + "." + juce::String(centiseconds).paddedLeft('0', 2);

    return juce::String(wholeSeconds / 60).paddedLeft('0', 2)
        + ":" + juce::String(secondsPart).paddedLeft('0', 2)
        + "." + juce::String(centiseconds).paddedLeft('0', 2);
}

juce::String lcdGhostForValue(const juce::String& value)
{
    juce::String ghost;
    ghost.preallocateBytes(value.getNumBytesAsUTF8() + 1);
    for (const auto character : value)
    {
        if (juce::CharacterFunctions::isDigit(character))
            ghost << '8';
        else
            ghost << character;
    }
    return ghost;
}

double projectSequenceLengthSeconds(const ProjectState& project)
{
    double endSeconds = juce::jmax(project.rightLocatorSec, project.playheadSec);

    for (const auto& section : project.midiSections)
        endSeconds = juce::jmax(endSeconds, tickToSeconds(project, section.startTick + juce::jmax(kMinSequenceSnapTicks, section.lengthTicks)));

    for (const auto& clip : project.sampleClips)
        endSeconds = juce::jmax(endSeconds, juce::jmax(0.0, clip.startSec + juce::jmax(0.0, clip.durationSec)));

    return endSeconds;
}

juce::String noteNameLabel(int pitch)
{
    static constexpr const char* kNames[] = { "C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B" };
    return juce::String(kNames[pitch % 12]) + juce::String((pitch / 12) - 1);
}

bool isVirtualPianoBlackKey(int pitch)
{
    switch (pitch % 12)
    {
        case 1:
        case 3:
        case 6:
        case 8:
        case 10:
            return true;
        default:
            return false;
    }
}

juce::String normaliseVirtualPianoShortcutKey(const juce::KeyPress& key)
{
    const auto character = key.getTextCharacter();
    if (character == 0)
        return {};

    juce::String text(juce::String::charToString(character).trim());
    if (text.isEmpty())
        return {};

    if (text.length() == 1)
    {
        const auto upper = juce::CharacterFunctions::toUpperCase(text[0]);
        return juce::String::charToString(upper);
    }

    return text.toUpperCase();
}

enum class AppProfileSection
{
    timerCallback,
    refreshFloatingWindows,
    count
};

constexpr auto kAppProfileSectionCount = static_cast<size_t>(AppProfileSection::count);

const char* appProfileSectionName(AppProfileSection section)
{
    switch (section)
    {
        case AppProfileSection::timerCallback: return "timer_callback";
        case AppProfileSection::refreshFloatingWindows: return "refresh_floating_windows";
        case AppProfileSection::count: break;
    }

    return "unknown";
}

int64_t appProfileNowMicroseconds() noexcept
{
    return std::chrono::duration_cast<std::chrono::microseconds>(
        std::chrono::steady_clock::now().time_since_epoch()
    ).count();
}

struct AppProfileCounter
{
    std::atomic<int64_t> count{0};
    std::atomic<int64_t> totalMicros{0};
    std::atomic<int64_t> maxMicros{0};

    void addSample(int64_t durationMicros) noexcept
    {
        count.fetch_add(1, std::memory_order_relaxed);
        totalMicros.fetch_add(durationMicros, std::memory_order_relaxed);

        auto currentMax = maxMicros.load(std::memory_order_relaxed);
        while (durationMicros > currentMax
               && !maxMicros.compare_exchange_weak(currentMax, durationMicros, std::memory_order_relaxed))
        {
        }
    }
};

class AppRuntimeProfiler final
{
public:
    AppRuntimeProfiler()
    {
        const auto logPath = juce::SystemStats::getEnvironmentVariable("AIMS_NATIVE_PROFILE_LOG", {}).trim();
        enabled = logPath.isNotEmpty();
        if (enabled)
            logFile = juce::File(logPath);
    }

    bool isEnabled() const noexcept
    {
        return enabled;
    }

    void addSample(AppProfileSection section, int64_t durationMicros) noexcept
    {
        if (!enabled)
            return;

        counters[static_cast<size_t>(section)].addSample(durationMicros);
    }

    void dump(const juce::String& sectionTitle)
    {
        if (!enabled)
            return;

        bool expected = false;
        if (!dumped.compare_exchange_strong(expected, true, std::memory_order_relaxed))
            return;

        if (logFile == juce::File())
            return;

        logFile.getParentDirectory().createDirectory();
        juce::FileOutputStream output(logFile);
        if (!output.openedOk())
            return;

        output.setPosition(output.getFile().getSize());

        juce::StringArray lines;
        lines.add("[" + sectionTitle + "] time=" + juce::Time::getCurrentTime().toString(true, true));
        for (size_t index = 0; index < counters.size(); ++index)
        {
            const auto count = counters[index].count.load(std::memory_order_relaxed);
            if (count <= 0)
                continue;

            const auto totalMicros = counters[index].totalMicros.load(std::memory_order_relaxed);
            const auto maxMicros = counters[index].maxMicros.load(std::memory_order_relaxed);
            const auto averageMicros = static_cast<double>(totalMicros) / static_cast<double>(count);
            lines.add(juce::String(appProfileSectionName(static_cast<AppProfileSection>(index)))
                      + ": count=" + juce::String(count)
                      + " avg_us=" + juce::String(averageMicros, 2)
                      + " max_us=" + juce::String(maxMicros)
                      + " total_ms=" + juce::String(static_cast<double>(totalMicros) / 1000.0, 2));
        }
        lines.add({});

        output.writeText(lines.joinIntoString("\n"), false, false, nullptr);
    }

private:
    bool enabled = false;
    std::atomic<bool> dumped{false};
    juce::File logFile;
    std::array<AppProfileCounter, kAppProfileSectionCount> counters;
};

AppRuntimeProfiler& appRuntimeProfiler()
{
    static AppRuntimeProfiler profiler;
    return profiler;
}

constexpr juce::uint32 kPlaybackRackEditorSyncIntervalMs = 24;
constexpr juce::uint32 kDeferredEngineParameterFlushIntervalMs = 24;

class ScopedAppProfileSample final
{
public:
    explicit ScopedAppProfileSample(AppProfileSection sectionIn) noexcept
        : section(sectionIn),
          startMicros(appRuntimeProfiler().isEnabled() ? appProfileNowMicroseconds() : 0)
    {
    }

    ~ScopedAppProfileSample()
    {
        if (startMicros <= 0)
            return;

        appRuntimeProfiler().addSample(section, appProfileNowMicroseconds() - startMicros);
    }

private:
    AppProfileSection section;
    int64_t startMicros = 0;
};

}

class HeaderLcdDisplay final : public juce::Component
{
public:
    enum class Mode
    {
        standard,
        yautja,
        virus
    };

    HeaderLcdDisplay()
    {
        yautjaToggle.setButtonText("Y");
        yautjaToggle.setClickingTogglesState(true);
        yautjaToggle.setTooltip("Yautja timecode");
        yautjaToggle.onClick = [this]
        {
            setMode(yautjaToggle.getToggleState() ? Mode::yautja : Mode::standard);
        };
        addAndMakeVisible(yautjaToggle);

        virusToggle.setButtonText("V");
        virusToggle.setClickingTogglesState(true);
        virusToggle.setTooltip("Virus style LCD");
        virusToggle.onClick = [this]
        {
            setMode(virusToggle.getToggleState() ? Mode::virus : Mode::standard);
        };
        addAndMakeVisible(virusToggle);

        setTheme(themeSpecForIndex(0));
    }

    void setTheme(const ThemeSpec& theme)
    {
        lcdBackgroundColour = theme.lcdBackground;
        lcdFrameColour = theme.lcdFrame;
        lcdGlowColour = theme.lcdGlow;
        lcdLabelColour = theme.lcdLabel;
        lcdGhostColour = theme.lcdGhost;
        lcdValueColour = theme.lcdValue;
        yautjaToggle.setColour(juce::TextButton::buttonColourId, theme.surface);
        yautjaToggle.setColour(juce::TextButton::buttonOnColourId, juce::Colour::fromRGB(78, 26, 26));
        yautjaToggle.setColour(juce::TextButton::textColourOffId, theme.lcdLabel);
        yautjaToggle.setColour(juce::TextButton::textColourOnId, juce::Colour::fromRGB(255, 124, 112));
        virusToggle.setColour(juce::TextButton::buttonColourId, theme.surface);
        virusToggle.setColour(juce::TextButton::buttonOnColourId, juce::Colour::fromRGB(68, 90, 116));
        virusToggle.setColour(juce::TextButton::textColourOffId, theme.lcdLabel);
        virusToggle.setColour(juce::TextButton::textColourOnId, juce::Colour::fromRGB(224, 244, 255));
        repaint();
    }

    void setMode(Mode newMode)
    {
        if (mode == newMode)
        {
            yautjaToggle.setToggleState(mode == Mode::yautja, juce::dontSendNotification);
            virusToggle.setToggleState(mode == Mode::virus, juce::dontSendNotification);
            return;
        }

        mode = newMode;
        yautjaToggle.setToggleState(mode == Mode::yautja, juce::dontSendNotification);
        virusToggle.setToggleState(mode == Mode::virus, juce::dontSendNotification);
        repaint();
    }

    void setValues(double playheadSecondsIn,
                   double totalSecondsIn,
                   double leftLocatorSecondsIn,
                   double rightLocatorSecondsIn)
    {
        const auto nextPosition = formatLcdSeconds(playheadSecondsIn);
        const auto nextLength = formatLcdSeconds(totalSecondsIn);
        const auto nextLeft = formatLcdSeconds(leftLocatorSecondsIn);
        const auto nextRight = formatLcdSeconds(rightLocatorSecondsIn);

        if (positionValue == nextPosition
            && totalLengthValue == nextLength
            && leftLocatorValue == nextLeft
            && rightLocatorValue == nextRight)
            return;

        positionValue = nextPosition;
        totalLengthValue = nextLength;
        leftLocatorValue = nextLeft;
        rightLocatorValue = nextRight;
        repaint();
    }

    void resized() override
    {
        auto area = getLocalBounds().reduced(10, 8);
        auto toggleStrip = area.removeFromBottom(16);
        auto centred = toggleStrip.withSizeKeepingCentre(40, 12);
        yautjaToggle.setBounds(centred.removeFromLeft(18));
        centred.removeFromLeft(4);
        virusToggle.setBounds(centred.removeFromLeft(18));
    }

    void paint(juce::Graphics& g) override
    {
        auto bounds = getLocalBounds().toFloat();
        if (bounds.isEmpty())
            return;

        g.setColour(lcdBackgroundColour);
        g.fillRoundedRectangle(bounds, 8.0f);
        g.setColour(lcdFrameColour);
        g.drawRoundedRectangle(bounds.reduced(0.5f), 8.0f, 1.0f);

        auto inner = bounds.reduced(10.0f, 8.0f);
        inner.removeFromBottom(16.0f);
        const auto useYautja = mode == Mode::yautja;
        const auto useVirus = mode == Mode::virus;
        g.setColour(useYautja ? juce::Colour::fromRGBA(255, 96, 96, 18)
                              : (useVirus ? juce::Colour::fromRGBA(168, 212, 255, 24)
                                          : lcdGlowColour));
        g.fillRoundedRectangle(inner, 6.0f);

        if (useVirus)
        {
            g.setColour(juce::Colour::fromRGBA(228, 242, 255, 10));
            for (float y = inner.getY() + 2.0f; y < inner.getBottom(); y += 4.0f)
                g.fillRect(inner.withY(y).withHeight(1.0f));
        }

        auto titleArea = inner.removeFromTop(12.0f);
        drawLabelText(g,
                      titleArea,
                      "TIME",
                      useYautja ? juce::Colour::fromRGB(255, 110, 104)
                                : (useVirus ? juce::Colour::fromRGB(196, 226, 255)
                                            : lcdLabelColour),
                      10.0f,
                      mode);

        inner.removeFromTop(4.0f);
        auto topRow = inner.removeFromTop(24.0f);
        auto bottomRow = inner.removeFromTop(22.0f);

        drawCell(g, topRow.removeFromLeft(topRow.getWidth() * 0.5f), "POS", positionValue, mode);
        drawCell(g, topRow, "LEN", totalLengthValue, mode);
        drawCell(g, bottomRow.removeFromLeft(bottomRow.getWidth() * 0.5f), "L", leftLocatorValue, mode);
        drawCell(g, bottomRow, "R", rightLocatorValue, mode);
    }

private:
    enum GlyphSegment
    {
        segN  = 1 << 0,
        segNE = 1 << 1,
        segE  = 1 << 2,
        segSE = 1 << 3,
        segS  = 1 << 4,
        segSW = 1 << 5,
        segW  = 1 << 6,
        segNW = 1 << 7,
        segC  = 1 << 8
    };

    static unsigned int yautjaMaskForChar(juce::juce_wchar ch)
    {
        switch (juce::CharacterFunctions::toUpperCase(ch))
        {
            case '0': return segN | segNE | segE | segSE | segS | segSW | segW | segNW;
            case '1': return segN | segS;
            case '2': return segN | segNE | segE | segC | segSW | segS;
            case '3': return segN | segNE | segE | segC | segSE | segS;
            case '4': return segNW | segW | segC | segNE | segE;
            case '5': return segN | segNW | segW | segC | segSE | segS;
            case '6': return segN | segNW | segW | segSW | segS | segSE | segC;
            case '7': return segN | segNE | segE;
            case '8': return segN | segNE | segE | segSE | segS | segSW | segW | segNW | segC;
            case '9': return segN | segNE | segE | segSE | segS | segNW | segW | segC;
            case 'A': return segN | segNE | segE | segW | segNW | segC;
            case 'E': return segN | segW | segNW | segS | segSW | segC;
            case 'I': return segN | segS | segC;
            case 'L': return segW | segSW | segS;
            case 'M': return segNW | segN | segNE | segSW | segSE;
            case 'N': return segNW | segNE | segSW | segSE | segC;
            case 'O': return segN | segNE | segE | segSE | segS | segSW | segW | segNW;
            case 'P': return segN | segNE | segE | segW | segNW | segC;
            case 'R': return segN | segNE | segE | segW | segNW | segC | segSE;
            case 'S': return segN | segNW | segW | segC | segSE | segS;
            case 'T': return segN | segC | segS;
            default: return 0;
        }
    }

    static void drawYautjaGlyph(juce::Graphics& g,
                                juce::Rectangle<float> area,
                                juce::juce_wchar ch,
                                juce::Colour colour)
    {
        if (ch == ' ')
            return;

        if (ch == ':')
        {
            const auto radius = juce::jmax(1.2f, area.getWidth() * 0.10f);
            g.setColour(colour);
            g.fillEllipse(area.getCentreX() - radius, area.getY() + area.getHeight() * 0.28f, radius * 2.0f, radius * 2.0f);
            g.fillEllipse(area.getCentreX() - radius, area.getY() + area.getHeight() * 0.64f, radius * 2.0f, radius * 2.0f);
            return;
        }

        if (ch == '.')
        {
            const auto radius = juce::jmax(1.2f, area.getWidth() * 0.10f);
            g.setColour(colour);
            g.fillEllipse(area.getCentreX() - radius, area.getBottom() - (radius * 2.6f), radius * 2.0f, radius * 2.0f);
            return;
        }

        const auto mask = yautjaMaskForChar(ch);
        if (mask == 0)
            return;

        juce::Path path;
        const auto x = area.getX();
        const auto y = area.getY();
        const auto w = area.getWidth();
        const auto h = area.getHeight();
        const auto cx = x + (w * 0.5f);
        const auto cy = y + (h * 0.54f);
        const auto top = y + (h * 0.10f);
        const auto upper = y + (h * 0.30f);
        const auto lower = y + (h * 0.74f);
        const auto bottom = y + (h * 0.92f);
        const auto left = x + (w * 0.10f);
        const auto innerLeft = x + (w * 0.28f);
        const auto innerRight = x + (w * 0.72f);
        const auto right = x + (w * 0.90f);

        const auto addSegment = [&path] (float x1, float y1, float x2, float y2)
        {
            path.startNewSubPath(x1, y1);
            path.lineTo(x2, y2);
        };

        if ((mask & segN) != 0)  addSegment(cx, upper, cx, top);
        if ((mask & segNE) != 0) addSegment(cx + (w * 0.04f), cy - (h * 0.06f), right, top);
        if ((mask & segE) != 0)  addSegment(innerRight, cy, right, cy);
        if ((mask & segSE) != 0) addSegment(cx + (w * 0.04f), cy + (h * 0.04f), innerRight, lower);
        if ((mask & segS) != 0)  addSegment(cx, lower, cx, bottom);
        if ((mask & segSW) != 0) addSegment(cx - (w * 0.04f), cy + (h * 0.04f), left, bottom);
        if ((mask & segW) != 0)  addSegment(left, cy, innerLeft, cy);
        if ((mask & segNW) != 0) addSegment(cx - (w * 0.04f), cy - (h * 0.06f), left, top);
        if ((mask & segC) != 0)  addSegment(innerLeft, cy, innerRight, cy);

        g.setColour(colour);
        g.strokePath(path, juce::PathStrokeType(juce::jmax(1.2f, w * 0.10f),
                                                juce::PathStrokeType::curved,
                                                juce::PathStrokeType::rounded));
    }

    static void drawYautjaText(juce::Graphics& g,
                               juce::Rectangle<float> area,
                               const juce::String& text,
                               juce::Colour colour,
                               float glyphHeight,
                               float xOffset = 0.0f)
    {
        const auto effectiveHeight = juce::jmax(8.0f, glyphHeight);
        auto glyphWidth = effectiveHeight * 0.58f;
        auto glyphGap = effectiveHeight * 0.18f;
        const auto totalWidth = (glyphWidth * static_cast<float>(text.length())) + (glyphGap * juce::jmax(0, text.length() - 1));
        const auto scale = totalWidth > area.getWidth() && totalWidth > 0.0f
            ? (area.getWidth() / totalWidth)
            : 1.0f;
        glyphWidth *= scale;
        glyphGap *= scale;

        auto x = area.getX() + xOffset;
        const auto y = area.getCentreY() - (effectiveHeight * scale * 0.5f);
        for (int index = 0; index < text.length(); ++index)
        {
            drawYautjaGlyph(g,
                            { x, y, glyphWidth, effectiveHeight * scale },
                            text[index],
                            colour);
            x += glyphWidth + glyphGap;
        }
    }

    static void drawLabelText(juce::Graphics& g,
                              juce::Rectangle<float> area,
                              const juce::String& text,
                              juce::Colour colour,
                              float normalFontHeight,
                              Mode displayMode = Mode::standard)
    {
        if (displayMode == Mode::yautja)
        {
            drawYautjaText(g, area, text, colour, area.getHeight() * 0.82f);
            return;
        }

        g.setColour(colour);
        g.setFont(juce::FontOptions(displayMode == Mode::virus ? "Bahnschrift SemiCondensed" : "Consolas",
                                    displayMode == Mode::virus ? normalFontHeight + 0.5f : normalFontHeight,
                                    juce::Font::bold));
        g.drawText(text, area, juce::Justification::centredLeft, false);
    }

    void drawCell(juce::Graphics& g,
                  juce::Rectangle<float> area,
                  const juce::String& label,
                  const juce::String& value,
                  Mode mode) const
    {
        const auto useYautja = mode == Mode::yautja;
        const auto useVirus = mode == Mode::virus;
        auto cell = area.reduced(4.0f, 0.0f);
        auto labelArea = cell.removeFromLeft(useYautja ? 36.0f : (useVirus ? 32.0f : 26.0f));
        auto valueArea = cell;

        const auto labelColour = useYautja ? juce::Colour::fromRGB(255, 120, 112)
                                           : (useVirus ? juce::Colour::fromRGB(198, 226, 255)
                                                       : lcdLabelColour);
        const auto ghostColour = useYautja ? juce::Colour::fromRGBA(172, 72, 68, 44)
                                           : (useVirus ? juce::Colour::fromRGBA(124, 156, 188, 56)
                                                       : lcdGhostColour);
        const auto valueColour = useYautja ? juce::Colour::fromRGB(255, 132, 124)
                                           : (useVirus ? juce::Colour::fromRGB(232, 244, 255)
                                                       : lcdValueColour);

        drawLabelText(g, labelArea, label, labelColour, 10.5f, mode);

        const auto ghostText = lcdGhostForValue(value);
        if (useYautja)
        {
            drawYautjaText(g, valueArea, ghostText, ghostColour, 18.0f, 1.0f);
            drawYautjaText(g, valueArea, value, valueColour, 18.0f, 1.0f);
        }
        else
        {
            g.setFont(juce::FontOptions(useVirus ? "Bahnschrift SemiCondensed" : "Consolas",
                                        useVirus ? 18.5f : 18.0f,
                                        juce::Font::bold));
            g.setColour(ghostColour);
            g.drawText(ghostText, valueArea, juce::Justification::centredLeft, false);
            g.setColour(valueColour);
            g.drawText(value, valueArea, juce::Justification::centredLeft, false);
        }
    }

    juce::String positionValue = "00:00.00";
    juce::String totalLengthValue = "00:00.00";
    juce::String leftLocatorValue = "00:00.00";
    juce::String rightLocatorValue = "00:00.00";
    Mode mode = Mode::standard;
    juce::Colour lcdBackgroundColour = juce::Colour::fromRGB(14, 22, 18);
    juce::Colour lcdFrameColour = juce::Colour::fromRGB(58, 94, 72);
    juce::Colour lcdGlowColour = juce::Colour::fromRGBA(120, 255, 174, 22);
    juce::Colour lcdLabelColour = juce::Colour::fromRGB(135, 214, 164);
    juce::Colour lcdGhostColour = juce::Colour::fromRGBA(106, 170, 121, 52);
    juce::Colour lcdValueColour = juce::Colour::fromRGB(175, 255, 196);
    juce::TextButton yautjaToggle;
    juce::TextButton virusToggle;
};

class FloatingPanelWindow final : public juce::DocumentWindow
{
public:
    explicit FloatingPanelWindow(const juce::String& title, bool alwaysOnTop = false)
        : juce::DocumentWindow(title,
                               juce::Colour::fromRGB(11, 13, 17),
                               juce::DocumentWindow::allButtons)
    {
        setUsingNativeTitleBar(true);
        setResizable(true, true);
        setAlwaysOnTop(alwaysOnTop);
    }

    std::function<void()> onClosePressed;
    std::function<bool(const juce::KeyPress&)> onKeyPressed;

private:
    bool keyPressed(const juce::KeyPress& key) override
    {
        if (onKeyPressed != nullptr && onKeyPressed(key))
            return true;
        return juce::DocumentWindow::keyPressed(key);
    }

    void closeButtonPressed() override
    {
        setVisible(false);
        if (onClosePressed != nullptr)
            onClosePressed();
    }
};

class ActivityLogWindowComponent final : public juce::Component
{
public:
    ActivityLogWindowComponent(std::function<juce::String()> getLogTextIn,
                               std::function<void()> clearLogIn)
        : getLogText(std::move(getLogTextIn)),
          clearLog(std::move(clearLogIn))
    {
        titleLabel.setText("Activity Log", juce::dontSendNotification);
        titleLabel.setColour(juce::Label::textColourId, juce::Colour::fromRGB(230, 235, 242));
        titleLabel.setFont(juce::FontOptions(15.0f, juce::Font::bold));
        addAndMakeVisible(titleLabel);

        clearButton.setButtonText("Clear");
        clearButton.onClick = [this]
        {
            if (clearLog != nullptr)
                clearLog();
        };
        addAndMakeVisible(clearButton);

        logEditor.setMultiLine(true);
        logEditor.setReadOnly(true);
        logEditor.setScrollbarsShown(true);
        logEditor.setCaretVisible(false);
        logEditor.setPopupMenuEnabled(true);
        logEditor.setColour(juce::TextEditor::backgroundColourId, juce::Colour::fromRGB(15, 18, 23));
        logEditor.setColour(juce::TextEditor::textColourId, juce::Colour::fromRGB(220, 226, 235));
        logEditor.setColour(juce::TextEditor::outlineColourId, juce::Colour::fromRGB(62, 71, 86));
        logEditor.setColour(juce::TextEditor::focusedOutlineColourId, juce::Colour::fromRGB(88, 143, 212));
        logEditor.setFont(juce::FontOptions("Consolas", 12.0f, juce::Font::plain));
        addAndMakeVisible(logEditor);
    }

    void refreshFromModel()
    {
        const auto text = getLogText != nullptr ? getLogText() : juce::String();
        if (text == lastRenderedText)
            return;

        lastRenderedText = text;
        logEditor.setText(text, false);
        logEditor.moveCaretToEnd();
    }

    void resized() override
    {
        auto area = getLocalBounds().reduced(10);
        auto header = area.removeFromTop(28);
        titleLabel.setBounds(header.removeFromLeft(180));
        clearButton.setBounds(header.removeFromRight(90));
        area.removeFromTop(8);
        logEditor.setBounds(area);
    }

private:
    std::function<juce::String()> getLogText;
    std::function<void()> clearLog;
    juce::String lastRenderedText;
    juce::Label titleLabel;
    juce::TextButton clearButton;
    juce::TextEditor logEditor;
};

class VstFolderManagerComponent final : public juce::Component
{
public:
    VstFolderManagerComponent(std::function<juce::String()> defaultFolderGetterIn,
                              std::function<juce::StringArray()> userFoldersGetterIn,
                              std::function<void()> addFolderIn,
                              std::function<void(const juce::String&)> removeFolderIn,
                              std::function<void()> refreshCatalogIn)
        : defaultFolderGetter(std::move(defaultFolderGetterIn)),
          userFoldersGetter(std::move(userFoldersGetterIn)),
          addFolder(std::move(addFolderIn)),
          removeFolder(std::move(removeFolderIn)),
          refreshCatalog(std::move(refreshCatalogIn)),
          folderListModel(*this)
    {
        titleLabel.setText("VST Folder Manager", juce::dontSendNotification);
        titleLabel.setColour(juce::Label::textColourId, juce::Colour::fromRGB(230, 235, 242));
        titleLabel.setFont(juce::FontOptions(15.0f, juce::Font::bold));
        addAndMakeVisible(titleLabel);

        defaultFolderLabel.setText("Default VST Folder", juce::dontSendNotification);
        defaultFolderLabel.setColour(juce::Label::textColourId, juce::Colour::fromRGB(190, 199, 210));
        addAndMakeVisible(defaultFolderLabel);

        defaultFolderEditor.setReadOnly(true);
        defaultFolderEditor.setMultiLine(false);
        defaultFolderEditor.setScrollbarsShown(true);
        defaultFolderEditor.setPopupMenuEnabled(true);
        defaultFolderEditor.setColour(juce::TextEditor::backgroundColourId, juce::Colour::fromRGB(15, 18, 23));
        defaultFolderEditor.setColour(juce::TextEditor::textColourId, juce::Colour::fromRGB(226, 232, 240));
        defaultFolderEditor.setColour(juce::TextEditor::outlineColourId, juce::Colour::fromRGB(62, 71, 86));
        defaultFolderEditor.setColour(juce::TextEditor::focusedOutlineColourId, juce::Colour::fromRGB(88, 143, 212));
        addAndMakeVisible(defaultFolderEditor);

        userFoldersLabel.setText("User VST Folders", juce::dontSendNotification);
        userFoldersLabel.setColour(juce::Label::textColourId, juce::Colour::fromRGB(190, 199, 210));
        addAndMakeVisible(userFoldersLabel);

        folderList.setModel(&folderListModel);
        folderList.setRowHeight(30);
        folderList.setColour(juce::ListBox::backgroundColourId, juce::Colour::fromRGB(20, 22, 28));
        folderList.setColour(juce::ListBox::outlineColourId, juce::Colour::fromRGB(62, 71, 86));
        addAndMakeVisible(folderList);

        helperLabel.setText("The bundled folder is scanned automatically. Add project-specific user folders below.",
                            juce::dontSendNotification);
        helperLabel.setColour(juce::Label::textColourId, juce::Colour::fromRGB(150, 159, 170));
        addAndMakeVisible(helperLabel);

        configureButton(addButton, "Add Folder...");
        addButton.onClick = [this]
        {
            if (addFolder != nullptr)
                addFolder();
        };

        configureButton(removeButton, "Remove");
        removeButton.onClick = [this]
        {
            if (removeFolder == nullptr)
                return;

            const auto folder = selectedFolderPath();
            if (folder.isNotEmpty())
                removeFolder(folder);
        };

        configureButton(refreshButton, "Refresh Catalog");
        refreshButton.onClick = [this]
        {
            if (refreshCatalog != nullptr)
                refreshCatalog();
        };

        addAndMakeVisible(addButton);
        addAndMakeVisible(removeButton);
        addAndMakeVisible(refreshButton);

        refreshFromModel();
    }

    void refreshFromModel()
    {
        const auto defaultFolder = defaultFolderGetter != nullptr ? defaultFolderGetter().trim() : juce::String();
        const auto defaultText = defaultFolder.isNotEmpty() ? defaultFolder : juce::String("(Bundled vsti folder not found)");
        if (defaultFolderEditor.getText() != defaultText)
            defaultFolderEditor.setText(defaultText, false);

        const auto selectedFolder = selectedFolderPath();
        const auto latestFolders = userFoldersGetter != nullptr ? userFoldersGetter() : juce::StringArray();
        if (folderPaths.joinIntoString("\n") != latestFolders.joinIntoString("\n"))
        {
            folderPaths = latestFolders;
            folderList.updateContent();
            if (folderPaths.isEmpty())
            {
                folderList.deselectAllRows();
            }
            else
            {
                const auto selectedIndex = folderPaths.indexOf(selectedFolder);
                folderList.selectRow(selectedIndex >= 0 ? selectedIndex : 0);
            }
        }

        removeButton.setEnabled(folderList.getSelectedRow() >= 0);
    }

    void resized() override
    {
        auto area = getLocalBounds().reduced(10);
        auto header = area.removeFromTop(28);
        titleLabel.setBounds(header.removeFromLeft(220));
        area.removeFromTop(6);

        defaultFolderLabel.setBounds(area.removeFromTop(22));
        defaultFolderEditor.setBounds(area.removeFromTop(30));
        area.removeFromTop(8);

        userFoldersLabel.setBounds(area.removeFromTop(22));
        auto buttonRow = area.removeFromBottom(38);
        helperLabel.setBounds(area.removeFromBottom(22));
        area.removeFromBottom(6);
        folderList.setBounds(area);

        addButton.setBounds(buttonRow.removeFromLeft(120));
        buttonRow.removeFromLeft(8);
        removeButton.setBounds(buttonRow.removeFromLeft(100));
        refreshButton.setBounds(buttonRow.removeFromRight(140));
    }

private:
    class FolderListModel final : public juce::ListBoxModel
    {
    public:
        explicit FolderListModel(VstFolderManagerComponent& ownerIn) : owner(ownerIn) {}

        int getNumRows() override
        {
            return owner.folderPaths.size();
        }

        void paintListBoxItem(int rowNumber,
                              juce::Graphics& g,
                              int width,
                              int height,
                              bool rowIsSelected) override
        {
            if (!juce::isPositiveAndBelow(rowNumber, owner.folderPaths.size()))
                return;

            g.fillAll(rowIsSelected ? juce::Colour::fromRGB(46, 88, 138)
                                    : ((rowNumber % 2) == 0 ? juce::Colour::fromRGB(26, 30, 37)
                                                            : juce::Colour::fromRGB(21, 25, 31)));
            g.setColour(rowIsSelected ? juce::Colours::white : juce::Colour::fromRGB(226, 232, 240));
            g.setFont(juce::FontOptions(12.5f));
            g.drawText(owner.folderPaths[rowNumber], 8, 0, width - 12, height, juce::Justification::centredLeft, true);
        }

        void selectedRowsChanged(int) override
        {
            owner.refreshFromModel();
        }

    private:
        VstFolderManagerComponent& owner;
    };

    static void configureButton(juce::TextButton& button, const juce::String& text)
    {
        button.setButtonText(text);
        button.setColour(juce::TextButton::buttonColourId, juce::Colour::fromRGB(46, 52, 64));
        button.setColour(juce::TextButton::buttonOnColourId, juce::Colour::fromRGB(72, 104, 160));
        button.setColour(juce::TextButton::textColourOffId, juce::Colour::fromRGB(235, 239, 244));
    }

    juce::String selectedFolderPath() const
    {
        const auto row = folderList.getSelectedRow();
        if (!juce::isPositiveAndBelow(row, folderPaths.size()))
            return {};
        return folderPaths[row];
    }

    std::function<juce::String()> defaultFolderGetter;
    std::function<juce::StringArray()> userFoldersGetter;
    std::function<void()> addFolder;
    std::function<void(const juce::String&)> removeFolder;
    std::function<void()> refreshCatalog;
    FolderListModel folderListModel;
    juce::StringArray folderPaths;
    juce::Label titleLabel;
    juce::Label defaultFolderLabel;
    juce::TextEditor defaultFolderEditor;
    juce::Label userFoldersLabel;
    juce::Label helperLabel;
    juce::ListBox folderList;
    juce::TextButton addButton;
    juce::TextButton removeButton;
    juce::TextButton refreshButton;
};

class TransportPanelComponent final : public juce::Component
{
public:
    TransportPanelComponent(std::function<void()> jumpToStartIn,
                            std::function<void()> playProjectIn,
                            std::function<void()> playTrackIn,
                            std::function<void()> stopPlaybackIn,
                            std::function<void(int)> setTempoIn,
                            std::function<void(bool)> setLoopEnabledIn,
                            std::function<void(bool)> setMetronomeEnabledIn)
        : jumpToStart(std::move(jumpToStartIn)),
          playProject(std::move(playProjectIn)),
          playTrack(std::move(playTrackIn)),
          stopPlayback(std::move(stopPlaybackIn)),
          setTempo(std::move(setTempoIn)),
          setLoopEnabled(std::move(setLoopEnabledIn)),
          setMetronomeEnabled(std::move(setMetronomeEnabledIn))
    {
        configureButton(homeButton, "Home");
        homeButton.onClick = [this] { if (jumpToStart != nullptr) jumpToStart(); };

        configureButton(playProjectButton, "Play Project");
        playProjectButton.onClick = [this] { if (playProject != nullptr) playProject(); };

        configureButton(playTrackButton, "Play Track");
        playTrackButton.onClick = [this] { if (playTrack != nullptr) playTrack(); };

        configureButton(stopButton, "Stop");
        stopButton.onClick = [this] { if (stopPlayback != nullptr) stopPlayback(); };

        tempoLabel.setText("Tempo", juce::dontSendNotification);
        tempoLabel.setColour(juce::Label::textColourId, juce::Colour::fromRGB(214, 219, 227));
        addAndMakeVisible(tempoLabel);

        tempoSlider.setSliderStyle(juce::Slider::LinearHorizontal);
        tempoSlider.setRange(20.0, 300.0, 1.0);
        tempoSlider.setTextBoxStyle(juce::Slider::TextBoxRight, false, 70, 22);
        tempoSlider.setTextValueSuffix(" BPM");
        tempoSlider.onValueChange = [this]
        {
            if (!syncing && setTempo != nullptr)
                setTempo(juce::roundToInt(tempoSlider.getValue()));
        };
        addAndMakeVisible(tempoSlider);

        loopToggle.setButtonText("Loop");
        loopToggle.onClick = [this]
        {
            if (!syncing && setLoopEnabled != nullptr)
                setLoopEnabled(loopToggle.getToggleState());
        };
        addAndMakeVisible(loopToggle);

        metronomeToggle.setButtonText("Metronome");
        metronomeToggle.onClick = [this]
        {
            if (!syncing && setMetronomeEnabled != nullptr)
                setMetronomeEnabled(metronomeToggle.getToggleState());
        };
        addAndMakeVisible(metronomeToggle);

        statusLabel.setColour(juce::Label::textColourId, juce::Colour::fromRGB(230, 235, 242));
        statusLabel.setJustificationType(juce::Justification::centredLeft);
        addAndMakeVisible(statusLabel);

        playheadLabel.setColour(juce::Label::textColourId, juce::Colour::fromRGB(156, 199, 239));
        playheadLabel.setJustificationType(juce::Justification::centredLeft);
        addAndMakeVisible(playheadLabel);

        cpuUsageLabel.setColour(juce::Label::textColourId, juce::Colour::fromRGB(230, 235, 242));
        cpuUsageLabel.setJustificationType(juce::Justification::centred);
        cpuUsageLabel.setFont(juce::FontOptions(13.0f));
        addAndMakeVisible(cpuUsageLabel);
    }

    void refreshFromState(const ProjectState& project,
                          bool hasTrackSelection,
                          bool rackPlaying,
                          bool projectPlaying,
                          const juce::String& statusText,
                          double cpuUsagePercent,
                          float masterPeakLeftIn,
                          float masterPeakRightIn)
    {
        syncing = true;
        tempoSlider.setValue(project.bpm, juce::dontSendNotification);
        loopToggle.setToggleState(project.loopEnabled, juce::dontSendNotification);
        metronomeToggle.setToggleState(project.metronomeEnabled, juce::dontSendNotification);
        syncing = false;
        masterPeakLeft = juce::jlimit(0.0f, 1.0f, masterPeakLeftIn);
        masterPeakRight = juce::jlimit(0.0f, 1.0f, masterPeakRightIn);
        cpuUsageLabel.setText("CPU\n" + juce::String(juce::roundToInt(cpuUsagePercent)) + "%", juce::dontSendNotification);

        playTrackButton.setEnabled(hasTrackSelection);
        stopButton.setEnabled(rackPlaying || projectPlaying);

        const auto playheadSec = tickToSeconds(project.playheadTick, project.bpm);
        playheadLabel.setText("Playhead: tick "
                                  + juce::String(project.playheadTick)
                                  + "  |  "
                                  + juce::String(playheadSec, 2)
                                  + " s  |  "
                                  + "Locators "
                                  + juce::String(project.leftLocatorTick)
                                  + " - "
                                  + juce::String(project.rightLocatorTick),
                              juce::dontSendNotification);

        juce::String transportState;
        if (projectPlaying)
            transportState = "Project playback active";
        else if (rackPlaying)
            transportState = "Track playback active";
        else
            transportState = "Transport idle";

        if (statusText.trim().isNotEmpty())
            transportState << "  |  " << statusText.trim();
        statusLabel.setText(transportState, juce::dontSendNotification);
    }

    void paint(juce::Graphics& g) override
    {
        g.fillAll(juce::Colour::fromRGB(13, 15, 20));
        g.setColour(juce::Colour::fromRGB(31, 35, 44));
        g.drawRect(getLocalBounds(), 1);

        g.setColour(juce::Colour::fromRGB(36, 41, 52));
        g.fillRoundedRectangle(masterMeterBounds.toFloat(), 6.0f);
        g.setColour(juce::Colour::fromRGB(66, 72, 86));
        g.drawRoundedRectangle(masterMeterBounds.toFloat(), 6.0f, 1.0f);

        auto leftBounds = masterMeterBounds.removeFromLeft(masterMeterBounds.getWidth() / 2).reduced(4, 4);
        auto rightBounds = masterMeterBounds.reduced(4, 4);
        paintMeterBar(g, leftBounds, masterPeakLeft);
        paintMeterBar(g, rightBounds, masterPeakRight);
    }

    void resized() override
    {
        auto area = getLocalBounds().reduced(12);
        auto controls = area.removeFromTop(30);
        homeButton.setBounds(controls.removeFromLeft(72));
        controls.removeFromLeft(6);
        playProjectButton.setBounds(controls.removeFromLeft(116));
        controls.removeFromLeft(6);
        playTrackButton.setBounds(controls.removeFromLeft(96));
        controls.removeFromLeft(6);
        stopButton.setBounds(controls.removeFromLeft(72));
        controls.removeFromLeft(14);
        tempoLabel.setBounds(controls.removeFromLeft(44));
        tempoSlider.setBounds(controls.removeFromLeft(220));
        controls.removeFromLeft(10);
        loopToggle.setBounds(controls.removeFromLeft(72));
        controls.removeFromLeft(6);
        metronomeToggle.setBounds(controls.removeFromLeft(110));
        controls.removeFromLeft(10);
        cpuUsageLabel.setBounds(controls.removeFromLeft(60));
        controls.removeFromLeft(8);
        masterMeterBounds = controls.removeFromLeft(40).reduced(0, 2);

        area.removeFromTop(8);
        playheadLabel.setBounds(area.removeFromTop(24));
        area.removeFromTop(4);
        statusLabel.setBounds(area.removeFromTop(22));
    }

private:
    void configureButton(juce::TextButton& button, const juce::String& text)
    {
        button.setButtonText(text);
        button.setColour(juce::TextButton::buttonColourId, juce::Colour::fromRGB(44, 50, 62));
        button.setColour(juce::TextButton::textColourOffId, juce::Colours::white);
        addAndMakeVisible(button);
    }

    static void paintMeterBar(juce::Graphics& g, juce::Rectangle<int> bounds, float level)
    {
        g.setColour(juce::Colour::fromRGB(20, 23, 30));
        g.fillRoundedRectangle(bounds.toFloat(), 4.0f);

        const auto filledHeight = juce::roundToInt(static_cast<float>(bounds.getHeight()) * juce::jlimit(0.0f, 1.0f, level));
        if (filledHeight <= 0)
            return;

        auto filledBounds = bounds.withTrimmedTop(bounds.getHeight() - filledHeight);
        juce::ColourGradient gradient(juce::Colour::fromRGB(61, 210, 122),
                                      filledBounds.getBottomLeft().toFloat(),
                                      juce::Colour::fromRGB(242, 99, 84),
                                      filledBounds.getTopLeft().toFloat(),
                                      false);
        g.setGradientFill(gradient);
        g.fillRoundedRectangle(filledBounds.toFloat(), 4.0f);
    }

    std::function<void()> jumpToStart;
    std::function<void()> playProject;
    std::function<void()> playTrack;
    std::function<void()> stopPlayback;
    std::function<void(int)> setTempo;
    std::function<void(bool)> setLoopEnabled;
    std::function<void(bool)> setMetronomeEnabled;
    bool syncing = false;

    juce::TextButton homeButton;
    juce::TextButton playProjectButton;
    juce::TextButton playTrackButton;
    juce::TextButton stopButton;
    juce::Label tempoLabel;
    juce::Slider tempoSlider;
    juce::ToggleButton loopToggle;
    juce::ToggleButton metronomeToggle;
    juce::Label cpuUsageLabel;
    juce::Label playheadLabel;
    juce::Label statusLabel;
    juce::Rectangle<int> masterMeterBounds;
    float masterPeakLeft = 0.0f;
    float masterPeakRight = 0.0f;
};

class AudioSettingsPanelComponent final : public juce::Component
{
public:
    AudioSettingsPanelComponent(std::function<juce::Result(const juce::String&,
                                                           const juce::String&,
                                                           int,
                                                           int)> applySettingsIn,
                                std::function<void()> refreshStatusIn)
        : applySettings(std::move(applySettingsIn)),
          refreshStatus(std::move(refreshStatusIn))
    {
        summaryLabel.setColour(juce::Label::textColourId, juce::Colour::fromRGB(230, 235, 242));
        summaryLabel.setJustificationType(juce::Justification::topLeft);
        addAndMakeVisible(summaryLabel);

        configureCombo(driverTypeLabel, driverTypeCombo, "Driver Backend");
        configureCombo(outputDeviceLabel, outputDeviceCombo, "Output Device");
        configureCombo(sampleRateLabel, sampleRateCombo, "Sample Rate");
        configureCombo(bufferSizeLabel, bufferSizeCombo, "Buffer Size");

        driverTypeCombo.onChange = [this]
        {
            if (!syncing)
                applyDriverTypeSelection();
        };

        outputDeviceCombo.onChange = [this]
        {
            if (!syncing)
                applyOutputDeviceSelection();
        };

        sampleRateCombo.onChange = [this]
        {
            if (!syncing)
                applyFormatSelection();
        };

        bufferSizeCombo.onChange = [this]
        {
            if (!syncing)
                applyFormatSelection();
        };

        refreshButton.setButtonText("Refresh");
        refreshButton.onClick = [this]
        {
            if (refreshStatus != nullptr)
                refreshStatus();
        };
        addAndMakeVisible(refreshButton);

        applyButton.setButtonText("Apply");
        applyButton.onClick = [this] { applyCurrentSelection(); };
        addAndMakeVisible(applyButton);

        footerLabel.setColour(juce::Label::textColourId, juce::Colour::fromRGB(156, 199, 239));
        footerLabel.setJustificationType(juce::Justification::topLeft);
        footerLabel.setText("Playback and preview use the shared JUCE host output device.", juce::dontSendNotification);
        addAndMakeVisible(footerLabel);
    }

    void applyHostStatus(const juce::var& status)
    {
        auto* object = status.getDynamicObject();
        if (object == nullptr)
        {
            setStatusMessage("Could not read native audio device status.", true);
            return;
        }

        const auto driverType = object->getProperty("audio_device_type").toString().trim();
        const auto outputDevice = object->getProperty("audio_device_name").toString().trim();
        const auto sampleRate = juce::roundToInt(static_cast<double>(object->getProperty("sample_rate")));
        const auto bufferSize = static_cast<int>(object->getProperty("buffer_size"));

        summaryLabel.setText("Driver: " + (driverType.isNotEmpty() ? driverType : "Default")
                                 + "\nDevice: " + (outputDevice.isNotEmpty() ? outputDevice : "Unavailable")
                                 + "\nFormat: " + juce::String(juce::jmax(1, sampleRate)) + " Hz"
                                 + "  |  " + juce::String(juce::jmax(1, bufferSize)) + " samples",
                             juce::dontSendNotification);

        syncing = true;
        populateStringCombo(driverTypeCombo,
                            object->getProperty("available_audio_device_types"),
                            driverType,
                            true);
        populateStringCombo(outputDeviceCombo,
                            object->getProperty("available_audio_output_devices"),
                            outputDevice,
                            false);
        populateIntCombo(sampleRateCombo,
                         object->getProperty("available_audio_sample_rates"),
                         sampleRate,
                         " Hz");
        populateIntCombo(bufferSizeCombo,
                         object->getProperty("available_audio_buffer_sizes"),
                         bufferSize,
                         " samples");
        syncing = false;

        appliedDriverType = driverType;
        appliedOutputDevice = outputDevice;
        appliedSampleRate = juce::jmax(0, sampleRate);
        appliedBufferSize = juce::jmax(0, bufferSize);

        setStatusMessage("Native audio settings ready.", false);
    }

    void applyAudioDeviceSnapshot(const NativeVstHostSession::AudioDeviceSnapshot& snapshot)
    {
        const auto sampleRate = juce::roundToInt(snapshot.sampleRate);
        const auto bufferSize = snapshot.bufferSize;

        summaryLabel.setText("Driver: " + (snapshot.audioDeviceType.isNotEmpty() ? snapshot.audioDeviceType : "Default")
                                 + "\nDevice: " + (snapshot.audioDeviceName.isNotEmpty() ? snapshot.audioDeviceName : "Unavailable")
                                 + "\nFormat: " + juce::String(juce::jmax(1, sampleRate)) + " Hz"
                                 + "  |  " + juce::String(juce::jmax(1, bufferSize)) + " samples",
                             juce::dontSendNotification);

        syncing = true;
        populateStringCombo(driverTypeCombo, snapshot.availableAudioDeviceTypes, snapshot.audioDeviceType, true);
        populateStringCombo(outputDeviceCombo, snapshot.availableAudioOutputDevices, snapshot.audioDeviceName, false);
        populateIntCombo(sampleRateCombo, snapshot.availableAudioSampleRates, sampleRate, " Hz");
        populateIntCombo(bufferSizeCombo, snapshot.availableAudioBufferSizes, bufferSize, " samples");
        syncing = false;

        appliedDriverType = snapshot.audioDeviceType;
        appliedOutputDevice = snapshot.audioDeviceName;
        appliedSampleRate = juce::jmax(0, sampleRate);
        appliedBufferSize = juce::jmax(0, bufferSize);

        setStatusMessage("Native audio settings ready.", false);
    }

    void setStatusMessage(const juce::String& message, bool isError)
    {
        footerLabel.setColour(juce::Label::textColourId,
                              isError ? juce::Colour::fromRGB(244, 144, 144)
                                      : juce::Colour::fromRGB(156, 199, 239));
        footerLabel.setText(message, juce::dontSendNotification);
    }

    void paint(juce::Graphics& g) override
    {
        g.fillAll(juce::Colour::fromRGB(13, 15, 20));
        g.setColour(juce::Colour::fromRGB(31, 35, 44));
        g.drawRect(getLocalBounds(), 1);
    }

    void resized() override
    {
        auto area = getLocalBounds().reduced(14);
        summaryLabel.setBounds(area.removeFromTop(72));
        area.removeFromTop(10);
        layoutRow(area, driverTypeLabel, driverTypeCombo);
        area.removeFromTop(8);
        layoutRow(area, outputDeviceLabel, outputDeviceCombo);
        area.removeFromTop(8);
        layoutRow(area, sampleRateLabel, sampleRateCombo);
        area.removeFromTop(8);
        layoutRow(area, bufferSizeLabel, bufferSizeCombo);
        area.removeFromTop(12);

        auto buttonRow = area.removeFromTop(30);
        refreshButton.setBounds(buttonRow.removeFromLeft(96));
        buttonRow.removeFromLeft(8);
        applyButton.setBounds(buttonRow.removeFromLeft(96));
        area.removeFromTop(10);
        footerLabel.setBounds(area.removeFromTop(40));
    }

private:
    void configureCombo(juce::Label& label, juce::ComboBox& combo, const juce::String& text)
    {
        label.setText(text, juce::dontSendNotification);
        label.setColour(juce::Label::textColourId, juce::Colour::fromRGB(214, 219, 227));
        addAndMakeVisible(label);
        addAndMakeVisible(combo);
    }

    void layoutRow(juce::Rectangle<int>& area, juce::Label& label, juce::ComboBox& combo)
    {
        auto row = area.removeFromTop(28);
        label.setBounds(row.removeFromLeft(120));
        combo.setBounds(row);
    }

    void populateStringCombo(juce::ComboBox& combo,
                             const juce::var& valuesVar,
                             const juce::String& selectedValue,
                             bool includeAutoItem)
    {
        combo.clear(juce::dontSendNotification);
        int itemId = 1;
        int selectedId = 0;

        if (includeAutoItem)
        {
            combo.addItem("Auto / Default", itemId);
            if (selectedValue.isEmpty())
                selectedId = itemId;
            ++itemId;
        }

        if (auto* values = valuesVar.getArray())
        {
            for (const auto& valueVar : *values)
            {
                const auto text = valueVar.toString().trim();
                if (text.isEmpty())
                    continue;
                combo.addItem(text, itemId);
                if (text.equalsIgnoreCase(selectedValue))
                    selectedId = itemId;
                ++itemId;
            }
        }

        if (selectedId != 0)
            combo.setSelectedId(selectedId, juce::dontSendNotification);
        else if (combo.getNumItems() > 0)
            combo.setSelectedItemIndex(0, juce::dontSendNotification);
    }

    void populateStringCombo(juce::ComboBox& combo,
                             const juce::StringArray& values,
                             const juce::String& selectedValue,
                             bool includeAutoItem)
    {
        combo.clear(juce::dontSendNotification);
        int itemId = 1;
        int selectedId = 0;

        if (includeAutoItem)
        {
            combo.addItem("Auto / Default", itemId);
            if (selectedValue.isEmpty())
                selectedId = itemId;
            ++itemId;
        }

        for (const auto& text : values)
        {
            if (text.trim().isEmpty())
                continue;
            combo.addItem(text, itemId);
            if (text.equalsIgnoreCase(selectedValue))
                selectedId = itemId;
            ++itemId;
        }

        if (selectedId != 0)
            combo.setSelectedId(selectedId, juce::dontSendNotification);
        else if (combo.getNumItems() > 0)
            combo.setSelectedItemIndex(0, juce::dontSendNotification);
    }

    void populateIntCombo(juce::ComboBox& combo,
                          const juce::var& valuesVar,
                          int selectedValue,
                          const juce::String& suffix)
    {
        combo.clear(juce::dontSendNotification);
        int selectedId = 0;

        if (auto* values = valuesVar.getArray())
        {
            for (const auto& valueVar : *values)
            {
                const auto value = static_cast<int>(valueVar);
                if (value <= 0)
                    continue;
                combo.addItem(juce::String(value) + suffix, value);
                if (value == selectedValue)
                    selectedId = value;
            }
        }

        if (selectedId != 0)
            combo.setSelectedId(selectedId, juce::dontSendNotification);
        else if (combo.getNumItems() > 0)
            combo.setSelectedItemIndex(0, juce::dontSendNotification);
    }

    template <typename ValueType>
    void populateIntCombo(juce::ComboBox& combo,
                          const juce::Array<ValueType>& values,
                          int selectedValue,
                          const juce::String& suffix)
    {
        combo.clear(juce::dontSendNotification);
        int selectedId = 0;

        for (int index = 0; index < values.size(); ++index)
        {
            const auto value = static_cast<int>(std::llround(static_cast<double>(values.getReference(index))));
            if (value <= 0)
                continue;
            combo.addItem(juce::String(value) + suffix, value);
            if (value == selectedValue)
                selectedId = value;
        }

        if (selectedId != 0)
            combo.setSelectedId(selectedId, juce::dontSendNotification);
        else if (combo.getNumItems() > 0)
            combo.setSelectedItemIndex(0, juce::dontSendNotification);
    }

    int selectedIntValue(const juce::ComboBox& combo) const
    {
        return combo.getSelectedId();
    }

    juce::String selectedDriverType() const
    {
        const auto selected = driverTypeCombo.getText().trim();
        if (selected == "Auto / Default")
            return {};
        return selected;
    }

    void applyDriverTypeSelection()
    {
        if (applySettings == nullptr)
            return;

        const auto requestedType = selectedDriverType();
        if (requestedType.isEmpty() || requestedType.equalsIgnoreCase(appliedDriverType))
            return;

        const auto result = applySettings(requestedType, {}, 0, 0);
        setStatusMessage(result.wasOk() ? "Switched audio driver backend. Choose a device if needed."
                                        : result.getErrorMessage(),
                         result.failed());
    }

    void applyOutputDeviceSelection()
    {
        if (applySettings == nullptr)
            return;

        const auto requestedOutput = outputDeviceCombo.getText().trim();
        if (requestedOutput.isEmpty() || requestedOutput.equalsIgnoreCase(appliedOutputDevice))
            return;

        const auto result = applySettings({}, requestedOutput, 0, 0);
        setStatusMessage(result.wasOk() ? "Switched audio output device. Choose sample rate or buffer if needed."
                                        : result.getErrorMessage(),
                         result.failed());
    }

    void applyFormatSelection()
    {
        if (applySettings == nullptr)
            return;

        const auto selectedRate = selectedIntValue(sampleRateCombo);
        const auto selectedBuffer = selectedIntValue(bufferSizeCombo);
        if (selectedRate <= 0 && selectedBuffer <= 0)
            return;
        if (selectedRate == appliedSampleRate && selectedBuffer == appliedBufferSize)
            return;

        const auto result = applySettings({}, {}, selectedRate, selectedBuffer);
        setStatusMessage(result.wasOk() ? "Updated native sample rate and buffer size."
                                        : result.getErrorMessage(),
                         result.failed());
    }

    void applyCurrentSelection()
    {
        if (applySettings == nullptr)
            return;

        bool changed = false;

        const auto requestedType = selectedDriverType();
        if (requestedType.isNotEmpty() && !requestedType.equalsIgnoreCase(appliedDriverType))
        {
            const auto result = applySettings(requestedType, {}, 0, 0);
            if (result.failed())
            {
                setStatusMessage(result.getErrorMessage(), true);
                return;
            }
            changed = true;
        }

        const auto requestedOutput = outputDeviceCombo.getText().trim();
        if (requestedOutput.isNotEmpty() && !requestedOutput.equalsIgnoreCase(appliedOutputDevice))
        {
            const auto result = applySettings({}, requestedOutput, 0, 0);
            if (result.failed())
            {
                setStatusMessage(result.getErrorMessage(), true);
                return;
            }
            changed = true;
        }

        const auto selectedRate = selectedIntValue(sampleRateCombo);
        const auto selectedBuffer = selectedIntValue(bufferSizeCombo);
        if ((selectedRate > 0 || selectedBuffer > 0)
            && (selectedRate != appliedSampleRate || selectedBuffer != appliedBufferSize))
        {
            const auto result = applySettings({}, {}, selectedRate, selectedBuffer);
            if (result.failed())
            {
                setStatusMessage(result.getErrorMessage(), true);
                return;
            }
            changed = true;
        }

        setStatusMessage(changed ? "Updated native audio settings."
                                 : "Audio settings already match the current driver.",
                         false);
    }

    std::function<juce::Result(const juce::String&, const juce::String&, int, int)> applySettings;
    std::function<void()> refreshStatus;
    bool syncing = false;
    juce::String appliedDriverType;
    juce::String appliedOutputDevice;
    int appliedSampleRate = 0;
    int appliedBufferSize = 0;

    juce::Label summaryLabel;
    juce::Label driverTypeLabel;
    juce::ComboBox driverTypeCombo;
    juce::Label outputDeviceLabel;
    juce::ComboBox outputDeviceCombo;
    juce::Label sampleRateLabel;
    juce::ComboBox sampleRateCombo;
    juce::Label bufferSizeLabel;
    juce::ComboBox bufferSizeCombo;
    juce::TextButton refreshButton;
    juce::TextButton applyButton;
    juce::Label footerLabel;
};

class AudioWorkspaceWindowComponent final : public juce::Component
{
public:
    AudioWorkspaceWindowComponent(std::function<void()> jumpToStartIn,
                                  std::function<void()> playProjectIn,
                                  std::function<void()> playTrackIn,
                                  std::function<void()> stopPlaybackIn,
                                  std::function<void()> openAudioSettingsIn,
                                  std::function<void(int)> setTempoIn,
                                  std::function<void(bool)> setLoopEnabledIn,
                                  std::function<void(bool)> setMetronomeEnabledIn,
                                  std::function<void()> exportMixIn,
                                  std::function<void()> exportStemsIn,
                                  MixerComponent::ProjectGetter projectGetterIn,
                                  MixerComponent::TrackWriter trackWriterIn,
                                  MixerComponent::MeterGetter meterGetterIn,
                                  std::function<juce::String()> audioSummaryGetterIn)
        : audioSummaryGetter(std::move(audioSummaryGetterIn))
    {
        titleLabel.setText("Audio", juce::dontSendNotification);
        titleLabel.setFont(juce::FontOptions(18.0f, juce::Font::bold));
        titleLabel.setColour(juce::Label::textColourId, juce::Colour::fromRGB(235, 239, 244));
        addAndMakeVisible(titleLabel);

        configureButton(audioSettingsButton, "Audio Settings");
        audioSettingsButton.onClick = [openAudioSettings = std::move(openAudioSettingsIn)]
        {
            if (openAudioSettings != nullptr)
                openAudioSettings();
        };

        configureButton(exportMixButton, "Export Mix");
        exportMixButton.onClick = [exportMix = std::move(exportMixIn)]
        {
            if (exportMix != nullptr)
                exportMix();
        };

        configureButton(exportStemsButton, "Export Stems");
        exportStemsButton.onClick = [exportStems = std::move(exportStemsIn)]
        {
            if (exportStems != nullptr)
                exportStems();
        };

        audioSummaryLabel.setColour(juce::Label::textColourId, juce::Colour::fromRGB(190, 199, 210));
        audioSummaryLabel.setJustificationType(juce::Justification::centredLeft);
        addAndMakeVisible(audioSummaryLabel);

        transportPanel = std::make_unique<TransportPanelComponent>(std::move(jumpToStartIn),
                                                                   std::move(playProjectIn),
                                                                   std::move(playTrackIn),
                                                                   std::move(stopPlaybackIn),
                                                                   std::move(setTempoIn),
                                                                   std::move(setLoopEnabledIn),
                                                                   std::move(setMetronomeEnabledIn));
        addAndMakeVisible(*transportPanel);

        mixer = std::make_unique<MixerComponent>(std::move(projectGetterIn),
                                                 std::move(trackWriterIn),
                                                 std::move(meterGetterIn));
        mixerViewport.setViewedComponent(mixer.get(), false);
        mixerViewport.setScrollBarsShown(true, false);
        addAndMakeVisible(mixerViewport);
    }

    void refreshFromModel(const ProjectState& project,
                          bool hasTrackSelection,
                          bool rackPlaying,
                          bool projectPlaying,
                          const juce::String& statusText,
                          double cpuUsagePercent,
                          float masterPeakLeft,
                          float masterPeakRight,
                          bool deferAudioSummaryRefresh = false)
    {
        if (transportPanel != nullptr)
            transportPanel->refreshFromState(project,
                                             hasTrackSelection,
                                             rackPlaying,
                                             projectPlaying,
                                             statusText,
                                             cpuUsagePercent,
                                             masterPeakLeft,
                                             masterPeakRight);

        if (mixer != nullptr)
        {
            mixer->refreshFromModel();
            mixer->refreshMeters();
        }

        if (!deferAudioSummaryRefresh)
        {
            audioSummaryLabel.setText(audioSummaryGetter != nullptr
                                          ? audioSummaryGetter()
                                          : juce::String("Audio settings unavailable."),
                                      juce::dontSendNotification);
        }
    }

    void focusWorkspace()
    {
        if (transportPanel != nullptr)
            transportPanel->grabKeyboardFocus();
    }

    void paint(juce::Graphics& g) override
    {
        g.fillAll(juce::Colour::fromRGB(13, 15, 20));
        g.setColour(juce::Colour::fromRGB(31, 35, 44));
        g.drawRect(getLocalBounds(), 1);
    }

    void resized() override
    {
        auto area = getLocalBounds().reduced(14);
        auto header = area.removeFromTop(30);
        titleLabel.setBounds(header.removeFromLeft(90));
        audioSettingsButton.setBounds(header.removeFromLeft(120));
        header.removeFromLeft(6);
        exportMixButton.setBounds(header.removeFromLeft(100));
        header.removeFromLeft(6);
        exportStemsButton.setBounds(header.removeFromLeft(110));
        header.removeFromLeft(10);
        audioSummaryLabel.setBounds(header);

        area.removeFromTop(8);
        if (transportPanel != nullptr)
            transportPanel->setBounds(area.removeFromTop(88));
        area.removeFromTop(10);
        mixerViewport.setBounds(area);
    }

private:
    void configureButton(juce::TextButton& button, const juce::String& text)
    {
        button.setButtonText(text);
        button.setColour(juce::TextButton::buttonColourId, juce::Colour::fromRGB(48, 54, 66));
        button.setColour(juce::TextButton::textColourOffId, juce::Colours::white);
        addAndMakeVisible(button);
    }

    std::function<juce::String()> audioSummaryGetter;
    juce::Label titleLabel;
    juce::TextButton audioSettingsButton;
    juce::TextButton exportMixButton;
    juce::TextButton exportStemsButton;
    juce::Label audioSummaryLabel;
    std::unique_ptr<TransportPanelComponent> transportPanel;
    juce::Viewport mixerViewport;
    std::unique_ptr<MixerComponent> mixer;
};

class PanelsWindowComponent final : public juce::Component
{
public:
    PanelsWindowComponent(ArrangementOverviewComponent::ProjectGetter projectGetter,
                          ArrangementOverviewComponent::SelectedSectionGetter selectedSectionGetter,
                          ArrangementOverviewComponent::SectionSelectCallback sectionSelectCallback,
                          AutomationEditorComponent::SelectedTrackIndexGetter selectedTrackIndexGetter,
                          AutomationEditorComponent::TrackWriter trackWriter,
                          ArrangementOverviewComponent::ProjectWriter projectWriter,
                          std::function<EditorToolMode()> toolModeGetterIn,
                          std::function<void(EditorToolMode)> setToolModeIn)
        : toolModeGetter(std::move(toolModeGetterIn)),
          setToolMode(std::move(setToolModeIn)),
          tabs(juce::TabbedButtonBar::TabsAtTop)
    {
        tabs.setTabBarDepth(30);
        addAndMakeVisible(tabs);

        auto arrangementProjectGetter = projectGetter;
        auto arrangementProjectWriter = projectWriter;
        auto arrangementSelectedSectionGetter = selectedSectionGetter;
        auto arrangementSectionSelectCallback = sectionSelectCallback;
        auto* arrangement = new ArrangementOverviewComponent(arrangementProjectGetter,
                                                             arrangementProjectWriter,
                                                             arrangementSelectedSectionGetter,
                                                             arrangementSectionSelectCallback,
                                                             [this] { return toolModeGetter != nullptr ? toolModeGetter() : EditorToolMode::pencil; },
                                                             [this] (EditorToolMode mode)
                                                             {
                                                                 if (setToolMode != nullptr)
                                                                     setToolMode(mode);
                                                             });
        arrangementView = arrangement;
        tabs.addTab("Arrangement",
                    juce::Colour::fromRGB(26, 35, 52),
                    arrangement,
                    true);

        auto automationProjectGetter = projectGetter;
        auto automationSelectedTrackIndexGetter = selectedTrackIndexGetter;
        auto automationTrackWriter = trackWriter;
        auto* automation = new AutomationEditorComponent(automationProjectGetter,
                                                         automationSelectedTrackIndexGetter,
                                                         automationTrackWriter);
        automationView = automation;
        tabs.addTab("Automation",
                    juce::Colour::fromRGB(26, 35, 52),
                    automation,
                    true);

        auto sampleProjectGetter = projectGetter;
        auto sampleProjectWriter = projectWriter;
        auto* samples = new SampleTimelineComponent(sampleProjectGetter,
                                                    sampleProjectWriter);
        sampleTimelineView = samples;
        tabs.addTab("Samples",
                    juce::Colour::fromRGB(26, 35, 52),
                    samples,
                    true);

        auto pianoProjectGetter = projectGetter;
        auto pianoTrackIndexGetter = selectedTrackIndexGetter;
        auto pianoSelectedSectionGetter = selectedSectionGetter;
        auto pianoProjectWriter = projectWriter;
        auto* piano = new PianoRollComponent(pianoProjectGetter,
                                             pianoTrackIndexGetter,
                                             pianoSelectedSectionGetter,
                                             pianoProjectWriter);
        piano->setToolModeChangeCallback(setToolMode);
        pianoRollView = piano;
        tabs.addTab("Piano Roll",
                    juce::Colour::fromRGB(26, 35, 52),
                    piano,
                    true);
    }

    void setNotePreviewCallbacks(PianoRollComponent::NotePreviewCallback noteOnCallback,
                                 PianoRollComponent::NotePreviewCallback noteOffCallback,
                                 PianoRollComponent::PreviewStopCallback stopPreviewCallback = {})
    {
        if (pianoRollView != nullptr)
        {
            pianoRollView->setNotePreviewCallbacks(std::move(noteOnCallback),
                                                   std::move(noteOffCallback),
                                                   std::move(stopPreviewCallback));
        }
    }

    void setArrangementViewScale(float pixelsPerBar, float laneHeightPixels)
    {
        if (arrangementView != nullptr)
        {
            arrangementView->setHorizontalZoom(pixelsPerBar);
            arrangementView->setLaneHeight(laneHeightPixels);
        }
    }

    void setPianoRollViewScale(float pixelsPerBeat, float rowHeightPixels)
    {
        if (pianoRollView != nullptr)
        {
            pianoRollView->setHorizontalZoom(pixelsPerBeat);
            pianoRollView->setNoteRowHeight(rowHeightPixels);
        }
    }

    void refreshFromModel()
    {
        if (arrangementView != nullptr)
            arrangementView->refreshFromModel();
        if (automationView != nullptr)
            automationView->refreshFromModel();
        if (sampleTimelineView != nullptr)
            sampleTimelineView->refreshFromModel();
        if (pianoRollView != nullptr)
        {
            if (toolModeGetter != nullptr)
                pianoRollView->setToolMode(toolModeGetter());
            pianoRollView->refreshFromModel();
        }
    }

    void showArrangementTab()
    {
        tabs.setCurrentTabIndex(0);
        if (arrangementView != nullptr)
            arrangementView->grabKeyboardFocus();
    }

    void showAutomationTab()
    {
        tabs.setCurrentTabIndex(1);
        if (automationView != nullptr)
            automationView->grabKeyboardFocus();
    }

    void showSamplesTab()
    {
        tabs.setCurrentTabIndex(2);
        if (sampleTimelineView != nullptr)
            sampleTimelineView->grabKeyboardFocus();
    }

    void showPianoRollTab()
    {
        tabs.setCurrentTabIndex(3);
        if (pianoRollView != nullptr)
            pianoRollView->grabKeyboardFocus();
    }

    void paint(juce::Graphics& g) override
    {
        g.fillAll(juce::Colour::fromRGB(13, 15, 20));
    }

    void resized() override
    {
        tabs.setBounds(getLocalBounds());
    }

private:
    std::function<EditorToolMode()> toolModeGetter;
    std::function<void(EditorToolMode)> setToolMode;
    juce::TabbedComponent tabs;
    ArrangementOverviewComponent* arrangementView = nullptr;
    AutomationEditorComponent* automationView = nullptr;
    SampleTimelineComponent* sampleTimelineView = nullptr;
    PianoRollComponent* pianoRollView = nullptr;
};

class SampleWorkspaceWindowComponent final : public juce::Component
{
public:
    using ProjectGetter = SampleTimelineComponent::ProjectGetter;
    using ProjectWriter = SampleTimelineComponent::ProjectWriter;

    SampleWorkspaceWindowComponent(ProjectGetter projectGetterIn,
                                   ProjectWriter projectWriterIn,
                                   std::function<int()> selectedAssetGetterIn,
                                   std::function<void(int)> selectedAssetSetterIn,
                                   std::function<void()> importSampleIn,
                                   std::function<void()> placeSampleIn,
                                   std::function<bool()> canPlaceSampleIn)
        : projectGetter(std::move(projectGetterIn)),
          selectedAssetGetter(std::move(selectedAssetGetterIn)),
          importSample(std::move(importSampleIn)),
          placeSample(std::move(placeSampleIn)),
          canPlaceSample(std::move(canPlaceSampleIn)),
          assetListModel(projectGetter, std::move(selectedAssetSetterIn))
    {
        titleLabel.setText("Samples", juce::dontSendNotification);
        titleLabel.setFont(juce::FontOptions(18.0f, juce::Font::bold));
        titleLabel.setColour(juce::Label::textColourId, juce::Colour::fromRGB(235, 239, 244));
        addAndMakeVisible(titleLabel);

        configureButton(importButton, "Import Sample");
        importButton.onClick = [this]
        {
            if (importSample != nullptr)
                importSample();
        };

        configureButton(placeButton, "Place At Playhead");
        placeButton.onClick = [this]
        {
            if (placeSample != nullptr)
                placeSample();
        };

        libraryLabel.setText("Library", juce::dontSendNotification);
        libraryLabel.setColour(juce::Label::textColourId, juce::Colour::fromRGB(190, 199, 210));
        addAndMakeVisible(libraryLabel);

        timelineLabel.setText("Timeline", juce::dontSendNotification);
        timelineLabel.setColour(juce::Label::textColourId, juce::Colour::fromRGB(190, 199, 210));
        addAndMakeVisible(timelineLabel);

        assetList.setModel(&assetListModel);
        assetList.setRowHeight(28);
        assetList.setColour(juce::ListBox::backgroundColourId, juce::Colour::fromRGB(20, 22, 28));
        assetList.setOutlineThickness(1);
        addAndMakeVisible(assetList);

        timeline = std::make_unique<SampleTimelineComponent>(projectGetter,
                                                             std::move(projectWriterIn));
        timelineViewport.setViewedComponent(timeline.get(), false);
        timelineViewport.setScrollBarsShown(true, true);
        addAndMakeVisible(timelineViewport);
    }

    void refreshFromModel()
    {
        const auto selectedRow = selectedAssetGetter != nullptr ? selectedAssetGetter() : -1;
        assetList.updateContent();
        if (selectedRow >= 0)
            assetList.selectRow(selectedRow, false, true);
        else
            assetList.deselectAllRows();

        if (timeline != nullptr)
            timeline->refreshFromModel();

        placeButton.setEnabled(canPlaceSample != nullptr && canPlaceSample());
    }

    void focusLibrary()
    {
        assetList.grabKeyboardFocus();
    }

    void focusTimeline()
    {
        if (timeline != nullptr)
            timeline->grabKeyboardFocus();
    }

    void paint(juce::Graphics& g) override
    {
        g.fillAll(juce::Colour::fromRGB(13, 15, 20));
        g.setColour(juce::Colour::fromRGB(31, 35, 44));
        g.drawRect(getLocalBounds(), 1);
    }

    void resized() override
    {
        auto area = getLocalBounds().reduced(14);
        auto header = area.removeFromTop(30);
        titleLabel.setBounds(header.removeFromLeft(140));
        importButton.setBounds(header.removeFromLeft(120));
        header.removeFromLeft(8);
        placeButton.setBounds(header.removeFromLeft(146));

        area.removeFromTop(10);
        auto content = area;
        auto libraryArea = content.removeFromLeft(juce::jmin(320, juce::roundToInt(static_cast<float>(content.getWidth()) * 0.3f)));
        libraryLabel.setBounds(libraryArea.removeFromTop(24));
        libraryArea.removeFromTop(6);
        assetList.setBounds(libraryArea);

        content.removeFromLeft(10);
        timelineLabel.setBounds(content.removeFromTop(24));
        content.removeFromTop(6);
        timelineViewport.setBounds(content);
    }

private:
    class AssetListModel final : public juce::ListBoxModel
    {
    public:
        AssetListModel(ProjectGetter projectGetterIn,
                       std::function<void(int)> selectedAssetSetterIn)
            : projectGetter(std::move(projectGetterIn)),
              selectedAssetSetter(std::move(selectedAssetSetterIn))
        {
        }

        int getNumRows() override
        {
            return static_cast<int>(projectGetter().sampleAssets.size());
        }

        void paintListBoxItem(int rowNumber,
                              juce::Graphics& g,
                              int width,
                              int height,
                              bool rowIsSelected) override
        {
            juce::ignoreUnused(height);
            const auto& project = projectGetter();
            if (!juce::isPositiveAndBelow(rowNumber, static_cast<int>(project.sampleAssets.size())))
                return;

            const auto& asset = project.sampleAssets[static_cast<size_t>(rowNumber)];
            const auto text = juce::File(asset.path).getFileName()
                + "  (" + juce::String(asset.durationSec, 2) + " s)";

            g.fillAll(rowIsSelected ? juce::Colour::fromRGB(46, 88, 138)
                                    : ((rowNumber % 2) == 0 ? juce::Colour::fromRGB(26, 30, 37)
                                                            : juce::Colour::fromRGB(21, 25, 31)));
            g.setColour(rowIsSelected ? juce::Colours::white : juce::Colour::fromRGB(226, 232, 240));
            g.setFont(juce::FontOptions(12.5f));
            g.drawText(text, 8, 0, width - 12, height, juce::Justification::centredLeft, true);
        }

        void selectedRowsChanged(int lastRowSelected) override
        {
            if (selectedAssetSetter != nullptr)
                selectedAssetSetter(lastRowSelected);
        }

    private:
        ProjectGetter projectGetter;
        std::function<void(int)> selectedAssetSetter;
    };

    void configureButton(juce::TextButton& button, const juce::String& text)
    {
        button.setButtonText(text);
        button.setColour(juce::TextButton::buttonColourId, juce::Colour::fromRGB(48, 54, 66));
        button.setColour(juce::TextButton::textColourOffId, juce::Colours::white);
        addAndMakeVisible(button);
    }

    ProjectGetter projectGetter;
    std::function<int()> selectedAssetGetter;
    std::function<void()> importSample;
    std::function<void()> placeSample;
    std::function<bool()> canPlaceSample;
    AssetListModel assetListModel;

    juce::Label titleLabel;
    juce::TextButton importButton;
    juce::TextButton placeButton;
    juce::Label libraryLabel;
    juce::Label timelineLabel;
    juce::ListBox assetList;
    juce::Viewport timelineViewport;
    std::unique_ptr<SampleTimelineComponent> timeline;
};

class PianoRollWindowComponent final : public juce::Component,
                                       private juce::ScrollBar::Listener
{
public:
    using ProjectGetter = PianoRollComponent::ProjectGetter;
    using TrackIndexGetter = PianoRollComponent::TrackIndexGetter;
    using SelectedSectionIndexGetter = PianoRollComponent::SelectedSectionIndexGetter;
    using ProjectWriter = PianoRollComponent::ProjectWriter;

    PianoRollWindowComponent(ProjectGetter projectGetterIn,
                             TrackIndexGetter trackIndexGetterIn,
                             SelectedSectionIndexGetter selectedSectionIndexGetterIn,
                             ProjectWriter projectWriterIn,
                             std::function<EditorToolMode()> toolModeGetterIn,
                             std::function<void(EditorToolMode)> setToolModeIn,
                             float initialPixelsPerBeat,
                             float initialRowHeightPixels,
                             std::function<void(float)> zoomChangedIn,
                             std::function<void(float)> rowHeightChangedIn,
                             bool showHeaderIn = true)
        : toolModeGetter(std::move(toolModeGetterIn)),
          setToolMode(std::move(setToolModeIn)),
          zoomChanged(std::move(zoomChangedIn)),
          rowHeightChanged(std::move(rowHeightChangedIn)),
          showHeader(showHeaderIn)
    {
        titleLabel.setText("Piano Roll", juce::dontSendNotification);
        titleLabel.setFont(juce::FontOptions(18.0f, juce::Font::bold));
        titleLabel.setColour(juce::Label::textColourId, juce::Colour::fromRGB(235, 239, 244));
        addChildComponent(titleLabel);

        toolHintLabel.setJustificationType(juce::Justification::centredRight);
        toolHintLabel.setColour(juce::Label::textColourId, juce::Colour::fromRGB(190, 199, 210));
        addChildComponent(toolHintLabel);

        zoomLabel.setText("Zoom", juce::dontSendNotification);
        zoomLabel.setColour(juce::Label::textColourId, juce::Colour::fromRGB(214, 219, 227));
        addChildComponent(zoomLabel);

        zoomSlider.setSliderStyle(juce::Slider::LinearHorizontal);
        zoomSlider.setRange(12.0, 96.0, 1.0);
        zoomSlider.setChangeNotificationOnlyOnRelease(false);
        zoomSlider.setTextBoxStyle(juce::Slider::TextBoxRight, false, 52, 22);
        zoomSlider.setTextValueSuffix(" px/beat");
        zoomSlider.setValue(initialPixelsPerBeat, juce::dontSendNotification);
        zoomSlider.onValueChange = [this]
        {
            if (! syncingControls && zoomChanged != nullptr)
                zoomChanged(static_cast<float>(zoomSlider.getValue()));
        };
        addChildComponent(zoomSlider);

        rowHeightLabel.setText("Row", juce::dontSendNotification);
        rowHeightLabel.setColour(juce::Label::textColourId, juce::Colour::fromRGB(214, 219, 227));
        addChildComponent(rowHeightLabel);

        rowHeightSlider.setSliderStyle(juce::Slider::LinearHorizontal);
        rowHeightSlider.setRange(8.0, 32.0, 1.0);
        rowHeightSlider.setChangeNotificationOnlyOnRelease(false);
        rowHeightSlider.setTextBoxStyle(juce::Slider::TextBoxRight, false, 52, 22);
        rowHeightSlider.setTextValueSuffix(" px");
        rowHeightSlider.setValue(initialRowHeightPixels, juce::dontSendNotification);
        rowHeightSlider.onValueChange = [this]
        {
            if (! syncingControls && rowHeightChanged != nullptr)
                rowHeightChanged(static_cast<float>(rowHeightSlider.getValue()));
        };
        addChildComponent(rowHeightSlider);

        noteEditor = std::make_unique<PianoRollComponent>(projectGetterIn,
                                                          trackIndexGetterIn,
                                                          selectedSectionIndexGetterIn,
                                                          projectWriterIn);
        noteEditor->setSurfaceMode(PianoRollComponent::SurfaceMode::notesOnly);
        noteEditor->setToolModeChangeCallback(setToolMode);
        noteViewport.setViewedComponent(noteEditor.get(), false);
        noteViewport.setScrollBarsShown(true, true);
        addAndMakeVisible(noteViewport);

        controllerEditor = std::make_unique<PianoRollComponent>(std::move(projectGetterIn),
                                                                std::move(trackIndexGetterIn),
                                                                std::move(selectedSectionIndexGetterIn),
                                                                std::move(projectWriterIn));
        controllerEditor->setSurfaceMode(PianoRollComponent::SurfaceMode::controllerOnly);
        controllerEditor->setToolModeChangeCallback(setToolMode);
        controllerViewport.setViewedComponent(controllerEditor.get(), false);
        controllerViewport.setScrollBarsShown(false, true);
        addAndMakeVisible(controllerViewport);

        noteViewport.getHorizontalScrollBar().addListener(this);
        controllerViewport.getHorizontalScrollBar().addListener(this);

        titleLabel.setVisible(showHeader);
        toolHintLabel.setVisible(showHeader);
        zoomLabel.setVisible(showHeader);
        zoomSlider.setVisible(showHeader);
        rowHeightLabel.setVisible(showHeader);
        rowHeightSlider.setVisible(showHeader);
    }

    ~PianoRollWindowComponent() override
    {
        noteViewport.getHorizontalScrollBar().removeListener(this);
        controllerViewport.getHorizontalScrollBar().removeListener(this);
    }

    void setNotePreviewCallbacks(PianoRollComponent::NotePreviewCallback noteOnCallback,
                                 PianoRollComponent::NotePreviewCallback noteOffCallback,
                                 PianoRollComponent::PreviewStopCallback stopPreviewCallback = {})
    {
        if (noteEditor != nullptr)
        {
            noteEditor->setNotePreviewCallbacks(noteOnCallback,
                                                noteOffCallback,
                                                stopPreviewCallback);
        }

        if (controllerEditor != nullptr)
        {
            controllerEditor->setNotePreviewCallbacks(std::move(noteOnCallback),
                                                      std::move(noteOffCallback),
                                                      std::move(stopPreviewCallback));
        }
    }

    void setViewScale(float pixelsPerBeat, float rowHeightPixels)
    {
        syncingControls = true;
        zoomSlider.setValue(pixelsPerBeat, juce::dontSendNotification);
        rowHeightSlider.setValue(rowHeightPixels, juce::dontSendNotification);
        syncingControls = false;

        if (noteEditor != nullptr)
        {
            noteEditor->setHorizontalZoom(pixelsPerBeat);
            noteEditor->setNoteRowHeight(rowHeightPixels);
        }

        if (controllerEditor != nullptr)
        {
            controllerEditor->setHorizontalZoom(pixelsPerBeat);
            controllerEditor->setNoteRowHeight(rowHeightPixels);
        }
    }

    void refreshFromModel()
    {
        const auto mode = toolModeGetter != nullptr ? toolModeGetter() : EditorToolMode::pencil;
        if (showHeader)
        {
            toolHintLabel.setText("Tool: " + editorToolModeLabel(mode) + "  |  Right-click for tools",
                                  juce::dontSendNotification);
        }

        if (noteEditor != nullptr)
        {
            noteEditor->setToolMode(mode);
            noteEditor->refreshFromModel();
        }

        if (controllerEditor != nullptr)
        {
            controllerEditor->setToolMode(mode);
            controllerEditor->refreshFromModel();
        }

        syncHorizontalScrollFrom(noteViewport);
    }

    void focusEditor()
    {
        if (noteEditor != nullptr)
            noteEditor->grabKeyboardFocus();
    }

    void paint(juce::Graphics& g) override
    {
        g.fillAll(juce::Colour::fromRGB(13, 15, 20));
        g.setColour(juce::Colour::fromRGB(31, 35, 44));
        g.drawRect(getLocalBounds(), 1);

        if (!controllerViewport.getBounds().isEmpty())
        {
            g.setColour(juce::Colour::fromRGB(36, 42, 52));
            g.drawHorizontalLine(controllerViewport.getY() - 3, 12.0f, static_cast<float>(getWidth() - 12));
        }
    }

    void resized() override
    {
        auto area = getLocalBounds().reduced(14);
        if (showHeader)
        {
            auto header = area.removeFromTop(30);
            titleLabel.setBounds(header.removeFromLeft(120));
            auto controls = header.removeFromRight(540);
            rowHeightSlider.setBounds(controls.removeFromRight(156));
            controls.removeFromRight(8);
            rowHeightLabel.setBounds(controls.removeFromRight(34));
            controls.removeFromRight(10);
            zoomSlider.setBounds(controls.removeFromRight(166));
            controls.removeFromRight(8);
            zoomLabel.setBounds(controls.removeFromRight(42));
            controls.removeFromRight(14);
            toolHintLabel.setBounds(header);
            area.removeFromTop(8);
        }
        else
        {
            zoomLabel.setBounds({});
            zoomSlider.setBounds({});
            rowHeightLabel.setBounds({});
            rowHeightSlider.setBounds({});
        }

        const auto controllerPaneHeight = juce::jlimit(144, 220, area.getHeight() / 3);
        auto controllerArea = area.removeFromBottom(controllerPaneHeight);
        area.removeFromBottom(6);
        noteViewport.setBounds(area);
        controllerViewport.setBounds(controllerArea);
    }

private:
    void scrollBarMoved(juce::ScrollBar* scrollBarThatHasMoved, double newRangeStart) override
    {
        juce::ignoreUnused(newRangeStart);
        if (syncingScroll)
            return;

        if (scrollBarThatHasMoved == &noteViewport.getHorizontalScrollBar())
            syncHorizontalScrollFrom(noteViewport);
        else if (scrollBarThatHasMoved == &controllerViewport.getHorizontalScrollBar())
            syncHorizontalScrollFrom(controllerViewport);
    }

    void syncHorizontalScrollFrom(juce::Viewport& source)
    {
        if (syncingScroll)
            return;

        const auto targetX = source.getViewPositionX();
        syncingScroll = true;
        if (&source != &noteViewport)
            noteViewport.setViewPosition(targetX, noteViewport.getViewPositionY());
        if (&source != &controllerViewport)
            controllerViewport.setViewPosition(targetX, controllerViewport.getViewPositionY());
        syncingScroll = false;
    }

    std::function<EditorToolMode()> toolModeGetter;
    std::function<void(EditorToolMode)> setToolMode;
    std::function<void(float)> zoomChanged;
    std::function<void(float)> rowHeightChanged;
    bool showHeader = true;
    bool syncingScroll = false;
    bool syncingControls = false;

    juce::Label titleLabel;
    juce::Label toolHintLabel;
    juce::Label zoomLabel;
    juce::Slider zoomSlider;
    juce::Label rowHeightLabel;
    juce::Slider rowHeightSlider;
    juce::Viewport noteViewport;
    juce::Viewport controllerViewport;
    std::unique_ptr<PianoRollComponent> noteEditor;
    std::unique_ptr<PianoRollComponent> controllerEditor;
};

class VirtualPianoKeyboardComponent final : public juce::Component,
                                            private juce::Timer
{
public:
    VirtualPianoKeyboardComponent()
    {
        setOpaque(true);
        setWantsKeyboardFocus(true);
        setScalePercent(100);
    }

    void setScalePercent(int percent)
    {
        const auto clampedPercent = juce::jlimit(50, 150, percent);
        if (scalePercent == clampedPercent && !keyLayouts.empty())
            return;

        scalePercent = clampedPercent;
        rebuildKeyLayouts();
        repaint();
    }

    int getScalePercent() const noexcept
    {
        return scalePercent;
    }

    void setNoteTriggeredCallback(std::function<void(int)> callback)
    {
        noteTriggered = std::move(callback);
    }

    void flashPitch(int pitch)
    {
        flashedPitch = pitch;
        startTimer(140);
        repaint();
    }

    void paint(juce::Graphics& g) override
    {
        g.fillAll(juce::Colour::fromRGB(20, 22, 28));

        for (const auto& key : keyLayouts)
        {
            if (key.isBlack)
                continue;

            auto fill = juce::Colour::fromRGB(252, 252, 252);
            auto accent = juce::Colour::fromRGB(208, 214, 224);
            if (flashedPitch == key.pitch)
            {
                fill = juce::Colour::fromRGB(202, 235, 255);
                accent = juce::Colour::fromRGB(112, 186, 255);
            }

            juce::ColourGradient gradient(fill, key.rect.getTopLeft(), accent, key.rect.getBottomLeft(), false);
            g.setGradientFill(gradient);
            g.setColour(fill);
            g.fillRoundedRectangle(key.rect, 4.0f);
            g.setColour(juce::Colour::fromRGB(72, 78, 88));
            g.drawRoundedRectangle(key.rect, 4.0f, 1.0f);

            g.setColour(juce::Colour::fromRGB(44, 48, 56));
            g.setFont(juce::FontOptions(juce::jmax(9.0f, 9.0f * scaleFactor()), juce::Font::bold));
            g.drawText(noteNameLabel(key.pitch),
                       key.rect.withTrimmedTop(key.rect.getHeight() * 0.74f).toNearestInt(),
                       juce::Justification::centred);

            g.setColour(juce::Colour::fromRGB(96, 104, 118));
            g.setFont(juce::FontOptions(juce::jmax(8.0f, 8.0f * scaleFactor())));
            g.drawText(key.shortcutLabel,
                       key.rect.withTrimmedTop(key.rect.getHeight() * 0.86f).toNearestInt(),
                       juce::Justification::centred);
        }

        for (const auto& key : keyLayouts)
        {
            if (!key.isBlack)
                continue;

            auto fill = juce::Colour::fromRGB(72, 82, 96);
            auto accent = juce::Colour::fromRGB(18, 22, 28);
            if (flashedPitch == key.pitch)
            {
                fill = juce::Colour::fromRGB(78, 170, 255);
                accent = juce::Colour::fromRGB(20, 88, 180);
            }

            juce::ColourGradient gradient(fill, key.rect.getTopLeft(), accent, key.rect.getBottomLeft(), false);
            g.setGradientFill(gradient);
            g.fillRoundedRectangle(key.rect, 4.0f);
            g.setColour(juce::Colour::fromRGB(12, 14, 18));
            g.drawRoundedRectangle(key.rect, 4.0f, 1.0f);

            g.setColour(juce::Colour::fromRGB(232, 240, 248));
            g.setFont(juce::FontOptions(juce::jmax(8.0f, 8.0f * scaleFactor()), juce::Font::bold));
            g.drawText(noteNameLabel(key.pitch),
                       key.rect.withTrimmedTop(key.rect.getHeight() * 0.7f).toNearestInt(),
                       juce::Justification::centred);

            g.setColour(juce::Colour::fromRGB(176, 188, 204));
            g.setFont(juce::FontOptions(juce::jmax(7.0f, 7.0f * scaleFactor())));
            g.drawText(key.shortcutLabel,
                       key.rect.withTrimmedTop(key.rect.getHeight() * 0.84f).toNearestInt(),
                       juce::Justification::centred);
        }
    }

    void mouseDown(const juce::MouseEvent& event) override
    {
        if (!event.mods.isLeftButtonDown())
            return;

        grabKeyboardFocus();

        if (const auto* key = hitTestKey(event.position))
        {
            flashPitch(key->pitch);
            if (noteTriggered != nullptr)
                noteTriggered(key->pitch);
        }
    }

private:
    struct KeyLayout
    {
        int pitch = 60;
        bool isBlack = false;
        juce::String shortcutLabel;
        juce::Rectangle<float> rect;
    };

    float scaleFactor() const noexcept
    {
        return static_cast<float>(scalePercent) / 100.0f;
    }

    void timerCallback() override
    {
        stopTimer();
        flashedPitch = -1;
        repaint();
    }

    void rebuildKeyLayouts()
    {
        keyLayouts.clear();
        keyLayouts.reserve(virtualPianoKeySpecs().size());

        const auto whiteKeyWidth = 54.0f * scaleFactor();
        const auto whiteKeyHeight = 184.0f * scaleFactor();
        const auto naturalWidth = (whiteKeyWidth * 14.0f) + 24.0f;
        const auto naturalHeight = whiteKeyHeight + 24.0f;

        setSize(juce::jmax(280, juce::roundToInt(naturalWidth)),
                juce::jmax(96, juce::roundToInt(naturalHeight)));

        const auto area = getLocalBounds().toFloat().reduced(12.0f);
        const auto blackKeyWidth = whiteKeyWidth * 0.62f;
        const auto blackKeyHeight = whiteKeyHeight * 0.62f;

        auto whiteCursor = 0;
        for (const auto& spec : virtualPianoKeySpecs())
        {
            KeyLayout layout;
            layout.pitch = spec.pitch;
            layout.isBlack = isVirtualPianoBlackKey(spec.pitch);
            layout.shortcutLabel = virtualPianoShortcutLabel(spec);

            if (layout.isBlack)
            {
                const auto centerX = area.getX() + (static_cast<float>(whiteCursor) * whiteKeyWidth);
                layout.rect = { centerX - (blackKeyWidth * 0.5f), area.getY(), blackKeyWidth, blackKeyHeight };
            }
            else
            {
                layout.rect = { area.getX() + (static_cast<float>(whiteCursor) * whiteKeyWidth),
                                area.getY(),
                                whiteKeyWidth + 0.5f,
                                whiteKeyHeight };
                ++whiteCursor;
            }

            keyLayouts.push_back(std::move(layout));
        }
    }

    const KeyLayout* hitTestKey(juce::Point<float> position) const
    {
        for (const auto& key : keyLayouts)
        {
            if (key.isBlack && key.rect.contains(position))
                return &key;
        }

        for (const auto& key : keyLayouts)
        {
            if (!key.isBlack && key.rect.contains(position))
                return &key;
        }

        return nullptr;
    }

    std::function<void(int)> noteTriggered;
    std::vector<KeyLayout> keyLayouts;
    int scalePercent = 100;
    int flashedPitch = -1;
};

class VirtualPianoWindowComponent final : public juce::Component
{
public:
    explicit VirtualPianoWindowComponent(std::function<void(int)> triggerPitchIn,
                                         std::function<bool(const juce::KeyPress&)> shortcutHandlerIn)
        : triggerPitch(std::move(triggerPitchIn)),
          shortcutHandler(std::move(shortcutHandlerIn))
    {
        setWantsKeyboardFocus(true);

        titleLabel.setText("Virtual Piano", juce::dontSendNotification);
        titleLabel.setFont(juce::FontOptions(18.0f, juce::Font::bold));
        titleLabel.setColour(juce::Label::textColourId, juce::Colour::fromRGB(235, 239, 244));
        addAndMakeVisible(titleLabel);

        addAndMakeVisible(scaleCombo);

        scaleLabel.setText("Key Scale", juce::dontSendNotification);
        scaleLabel.setColour(juce::Label::textColourId, juce::Colour::fromRGB(210, 218, 228));
        scaleLabel.attachToComponent(&scaleCombo, true);
        addAndMakeVisible(scaleLabel);

        for (const auto percent : { 50, 75, 100, 125, 150 })
            scaleCombo.addItem(juce::String(percent) + "%", percent);
        scaleCombo.setSelectedId(100, juce::dontSendNotification);
        scaleCombo.onChange = [this]
        {
            if (keyboard != nullptr)
                keyboard->setScalePercent(scaleCombo.getSelectedId());
        };

        hintLabel.setText(virtualPianoHintText(), juce::dontSendNotification);
        hintLabel.setColour(juce::Label::textColourId, juce::Colour::fromRGB(199, 208, 220));
        hintLabel.setJustificationType(juce::Justification::centredLeft);
        hintLabel.setFont(juce::FontOptions(12.0f));
        addAndMakeVisible(hintLabel);

        keyboard = std::make_unique<VirtualPianoKeyboardComponent>();
        keyboard->setNoteTriggeredCallback([this] (int pitch)
        {
            if (triggerPitch != nullptr)
                triggerPitch(pitch);
        });

        keyboardViewport.setViewedComponent(keyboard.get(), false);
        keyboardViewport.setScrollBarsShown(true, false);
        addAndMakeVisible(keyboardViewport);
    }

    bool keyPressed(const juce::KeyPress& key) override
    {
        return shortcutHandler != nullptr && shortcutHandler(key);
    }

    void flashPitch(int pitch)
    {
        if (keyboard != nullptr)
            keyboard->flashPitch(pitch);
    }

    void focusKeyboard()
    {
        grabKeyboardFocus();
    }

    void resized() override
    {
        auto area = getLocalBounds().reduced(14);
        auto header = area.removeFromTop(28);
        titleLabel.setBounds(header.removeFromLeft(180));
        auto scaleArea = header.removeFromRight(180);
        scaleCombo.setBounds(scaleArea.removeFromRight(88));
        area.removeFromTop(8);
        hintLabel.setBounds(area.removeFromTop(40));
        area.removeFromTop(8);
        keyboardViewport.setBounds(area);
    }

    void paint(juce::Graphics& g) override
    {
        g.fillAll(juce::Colour::fromRGB(13, 15, 20));
        g.setColour(juce::Colour::fromRGB(31, 35, 44));
        g.drawRect(getLocalBounds(), 1);
    }

private:
    std::function<void(int)> triggerPitch;
    std::function<bool(const juce::KeyPress&)> shortcutHandler;

    juce::Label titleLabel;
    juce::Label scaleLabel;
    juce::ComboBox scaleCombo;
    juce::Label hintLabel;
    juce::Viewport keyboardViewport;
    std::unique_ptr<VirtualPianoKeyboardComponent> keyboard;
};

class TracksWorkspaceWindowComponent final : public juce::Component
{
public:
    using ProjectGetter = std::function<const ProjectState&()>;
    using SelectedTrackGetter = std::function<int()>;
    using SelectedTrackSetter = std::function<void(int)>;
    using SelectedTrackMutator = std::function<void(std::function<void(TrackState&)>, const juce::String&)>;
    using TrackSummaryGetter = std::function<juce::String(int)>;

    TracksWorkspaceWindowComponent(ProjectGetter projectGetterIn,
                                   SelectedTrackGetter selectedTrackGetterIn,
                                   SelectedTrackSetter selectedTrackSetterIn,
                                   SelectedTrackMutator mutateSelectedTrackIn,
                                   TrackSummaryGetter trackSummaryGetterIn,
                                   std::function<void()> addTrackIn,
                                   std::function<void()> duplicateTrackIn,
                                   std::function<void()> removeTrackIn,
                                   std::function<void()> openRackEditorIn,
                                   std::function<void()> saveRackStateIn,
                                   std::function<void()> playTrackIn,
                                   std::function<void()> stopPlaybackIn)
        : projectGetter(std::move(projectGetterIn)),
          selectedTrackGetter(std::move(selectedTrackGetterIn)),
          selectedTrackSetter(std::move(selectedTrackSetterIn)),
          mutateSelectedTrack(std::move(mutateSelectedTrackIn)),
          trackSummaryGetter(std::move(trackSummaryGetterIn)),
          addTrack(std::move(addTrackIn)),
          duplicateTrack(std::move(duplicateTrackIn)),
          removeTrack(std::move(removeTrackIn)),
          openRackEditor(std::move(openRackEditorIn)),
          saveRackState(std::move(saveRackStateIn)),
          playTrack(std::move(playTrackIn)),
          stopPlayback(std::move(stopPlaybackIn)),
          trackListModel(projectGetter, selectedTrackSetter)
    {
        titleLabel.setText("Tracks", juce::dontSendNotification);
        titleLabel.setFont(juce::FontOptions(18.0f, juce::Font::bold));
        titleLabel.setColour(juce::Label::textColourId, juce::Colour::fromRGB(235, 239, 244));
        addAndMakeVisible(titleLabel);

        configureButton(addTrackButton, "Add");
        addTrackButton.onClick = [this]
        {
            if (addTrack != nullptr)
                addTrack();
        };

        configureButton(duplicateTrackButton, "Duplicate");
        duplicateTrackButton.onClick = [this]
        {
            if (duplicateTrack != nullptr)
                duplicateTrack();
        };

        configureButton(removeTrackButton, "Remove");
        removeTrackButton.onClick = [this]
        {
            if (removeTrack != nullptr)
                removeTrack();
        };

        configureButton(openRackEditorButton, "Edit VST");
        openRackEditorButton.onClick = [this]
        {
            if (openRackEditor != nullptr)
                openRackEditor();
        };

        configureButton(saveRackStateButton, "Save Rack");
        saveRackStateButton.onClick = [this]
        {
            if (saveRackState != nullptr)
                saveRackState();
        };

        configureButton(playTrackButton, "Play Track");
        playTrackButton.onClick = [this]
        {
            if (playTrack != nullptr)
                playTrack();
        };

        configureButton(stopButton, "Stop");
        stopButton.onClick = [this]
        {
            if (stopPlayback != nullptr)
                stopPlayback();
        };

        trackList.setModel(&trackListModel);
        trackList.setRowHeight(46);
        trackList.setColour(juce::ListBox::backgroundColourId, juce::Colour::fromRGB(20, 22, 28));
        trackList.setOutlineThickness(1);
        addAndMakeVisible(trackList);

        inspectorLabel.setText("Inspector", juce::dontSendNotification);
        inspectorLabel.setFont(juce::FontOptions(16.0f, juce::Font::bold));
        inspectorLabel.setColour(juce::Label::textColourId, juce::Colour::fromRGB(235, 239, 244));
        addAndMakeVisible(inspectorLabel);

        configureEditor(trackNameEditor, "Track name");
        trackNameEditor.onFocusLost = [this]
        {
            if (syncingControls)
                return;

            const auto value = trackNameEditor.getText().trim();
            applyTrackMutation([value] (TrackState& track)
                               {
                                   track.name = value.isNotEmpty() ? value : "Track";
                               },
                               "Rename Track");
        };
        trackNameEditor.onReturnKey = [this] { trackNameEditor.giveAwayKeyboardFocus(); };

        configureEditor(trackTypeEditor, "Track type");
        trackTypeEditor.onFocusLost = [this]
        {
            if (syncingControls)
                return;

            const auto value = trackTypeEditor.getText().trim();
            applyTrackMutation([value] (TrackState& track)
                               {
                                   track.trackType = value.isNotEmpty() ? value : "instrument";
                               },
                               "Change Track Type");
        };
        trackTypeEditor.onReturnKey = [this] { trackTypeEditor.giveAwayKeyboardFocus(); };

        configureEditor(instrumentModeEditor, "Instrument mode");
        instrumentModeEditor.onFocusLost = [this]
        {
            if (syncingControls)
                return;

            const auto value = instrumentModeEditor.getText().trim();
            applyTrackMutation([value] (TrackState& track)
                               {
                                   track.instrumentMode = value.isNotEmpty() ? value : "General MIDI";
                               },
                               "Change Instrument Mode");
        };
        instrumentModeEditor.onReturnKey = [this] { instrumentModeEditor.giveAwayKeyboardFocus(); };

        configureEditor(instrumentEditor, "Instrument name");
        instrumentEditor.onFocusLost = [this]
        {
            if (syncingControls)
                return;

            const auto value = instrumentEditor.getText().trim();
            applyTrackMutation([value] (TrackState& track)
                               {
                                   track.instrument = value.isNotEmpty() ? value : "Piano";
                               },
                               "Change Instrument");
        };
        instrumentEditor.onReturnKey = [this] { instrumentEditor.giveAwayKeyboardFocus(); };

        configureEditor(rackVstEditor, "Rack VST path or name");
        rackVstEditor.onFocusLost = [this]
        {
            if (syncingControls)
                return;

            const auto value = rackVstEditor.getText().trim();
            applyTrackMutation([value] (TrackState& track)
                               {
                                   track.rackVst = value;
                               },
                               "Change Rack VST");
        };
        rackVstEditor.onReturnKey = [this] { rackVstEditor.giveAwayKeyboardFocus(); };

        configureSlider(midiChannelSlider, 1.0, 16.0, 1.0, " ch");
        midiChannelSlider.onValueChange = [this]
        {
            if (syncingControls)
                return;

            applyTrackMutation([value = juce::roundToInt(midiChannelSlider.getValue())] (TrackState& track)
                               {
                                   track.midiChannel = juce::jlimit(0, 15, value - 1);
                               },
                               "Change MIDI Channel");
        };

        configureSlider(midiProgramSlider, 0.0, 127.0, 1.0, " pgm");
        midiProgramSlider.onValueChange = [this]
        {
            if (syncingControls)
                return;

            applyTrackMutation([value = juce::roundToInt(midiProgramSlider.getValue())] (TrackState& track)
                               {
                                   track.midiProgram = juce::jlimit(0, 127, value);
                               },
                               "Change MIDI Program");
        };

        configureSlider(volumeSlider, 0.0, 1.0, 0.01, "");
        volumeSlider.onValueChange = [this]
        {
            if (syncingControls)
                return;

            applyTrackMutation([value = volumeSlider.getValue()] (TrackState& track)
                               {
                                   track.volume = juce::jlimit(0.0, 1.0, value);
                               },
                               "Change Track Volume");
        };

        configureSlider(panSlider, -1.0, 1.0, 0.01, "");
        panSlider.onValueChange = [this]
        {
            if (syncingControls)
                return;

            applyTrackMutation([value = panSlider.getValue()] (TrackState& track)
                               {
                                   track.pan = juce::jlimit(-1.0, 1.0, value);
                               },
                               "Change Track Pan");
        };

        muteToggle.setButtonText("Mute");
        muteToggle.onClick = [this]
        {
            if (syncingControls)
                return;

            applyTrackMutation([value = muteToggle.getToggleState()] (TrackState& track)
                               {
                                   track.mute = value;
                               },
                               "Toggle Mute");
        };
        addAndMakeVisible(muteToggle);

        soloToggle.setButtonText("Solo");
        soloToggle.onClick = [this]
        {
            if (syncingControls)
                return;

            applyTrackMutation([value = soloToggle.getToggleState()] (TrackState& track)
                               {
                                   track.solo = value;
                               },
                               "Toggle Solo");
        };
        addAndMakeVisible(soloToggle);

        liveArmToggle.setButtonText("Arm");
        liveArmToggle.onClick = [this]
        {
            if (syncingControls)
                return;

            applyTrackMutation([value = liveArmToggle.getToggleState()] (TrackState& track)
                               {
                                   track.liveArmed = value;
                               },
                               "Toggle Record Arm");
        };
        addAndMakeVisible(liveArmToggle);

        summaryEditor.setMultiLine(true);
        summaryEditor.setReadOnly(true);
        summaryEditor.setScrollbarsShown(true);
        summaryEditor.setColour(juce::TextEditor::backgroundColourId, juce::Colour::fromRGB(18, 20, 25));
        summaryEditor.setColour(juce::TextEditor::textColourId, juce::Colour::fromRGB(226, 230, 237));
        summaryEditor.setColour(juce::TextEditor::outlineColourId, juce::Colour::fromRGB(56, 64, 79));
        summaryEditor.setFont(juce::FontOptions(13.0f));
        addAndMakeVisible(summaryEditor);
    }

    void refreshFromModel()
    {
        const auto& project = projectGetter();
        const auto selectedTrack = selectedTrackGetter != nullptr ? selectedTrackGetter() : -1;

        trackList.updateContent();
        if (!project.tracks.empty() && selectedTrack >= 0)
            trackList.selectRow(selectedTrack, false, true);
        else
            trackList.deselectAllRows();

        refreshInspectorControls();

        const auto hasSelection = selectedTrack >= 0 && selectedTrack < static_cast<int>(project.tracks.size());
        duplicateTrackButton.setEnabled(hasSelection);
        removeTrackButton.setEnabled(hasSelection);
        openRackEditorButton.setEnabled(hasSelection);
        saveRackStateButton.setEnabled(hasSelection);
        playTrackButton.setEnabled(hasSelection);
        stopButton.setEnabled(hasSelection);
    }

    void focusTrackList()
    {
        trackList.grabKeyboardFocus();
    }

    void paint(juce::Graphics& g) override
    {
        g.fillAll(juce::Colour::fromRGB(13, 15, 20));
        g.setColour(juce::Colour::fromRGB(31, 35, 44));
        g.drawRect(getLocalBounds(), 1);
    }

    void resized() override
    {
        auto area = getLocalBounds().reduced(14);
        auto header = area.removeFromTop(30);
        titleLabel.setBounds(header.removeFromLeft(120));
        addTrackButton.setBounds(header.removeFromLeft(70));
        header.removeFromLeft(6);
        duplicateTrackButton.setBounds(header.removeFromLeft(90));
        header.removeFromLeft(6);
        removeTrackButton.setBounds(header.removeFromLeft(82));
        header.removeFromLeft(10);
        openRackEditorButton.setBounds(header.removeFromLeft(88));
        header.removeFromLeft(6);
        saveRackStateButton.setBounds(header.removeFromLeft(88));
        header.removeFromLeft(6);
        playTrackButton.setBounds(header.removeFromLeft(86));
        header.removeFromLeft(6);
        stopButton.setBounds(header.removeFromLeft(68));

        area.removeFromTop(10);
        auto leftArea = area.removeFromLeft(juce::jmin(360, juce::roundToInt(static_cast<float>(area.getWidth()) * 0.34f)));
        trackList.setBounds(leftArea);
        area.removeFromLeft(12);

        inspectorLabel.setBounds(area.removeFromTop(24));
        area.removeFromTop(6);
        trackNameEditor.setBounds(area.removeFromTop(24));
        area.removeFromTop(6);

        auto typeRow = area.removeFromTop(24);
        auto typeWidth = juce::roundToInt(static_cast<float>(typeRow.getWidth()) * 0.44f);
        trackTypeEditor.setBounds(typeRow.removeFromLeft(typeWidth));
        typeRow.removeFromLeft(6);
        instrumentModeEditor.setBounds(typeRow);
        area.removeFromTop(6);

        instrumentEditor.setBounds(area.removeFromTop(24));
        area.removeFromTop(6);
        rackVstEditor.setBounds(area.removeFromTop(24));
        area.removeFromTop(6);

        auto midiRow = area.removeFromTop(24);
        auto halfWidth = juce::roundToInt(static_cast<float>(midiRow.getWidth()) * 0.5f) - 3;
        midiChannelSlider.setBounds(midiRow.removeFromLeft(halfWidth));
        midiRow.removeFromLeft(6);
        midiProgramSlider.setBounds(midiRow);
        area.removeFromTop(6);

        auto mixRow = area.removeFromTop(24);
        volumeSlider.setBounds(mixRow.removeFromLeft(halfWidth));
        mixRow.removeFromLeft(6);
        panSlider.setBounds(mixRow);
        area.removeFromTop(8);

        auto toggleRow = area.removeFromTop(24);
        muteToggle.setBounds(toggleRow.removeFromLeft(70));
        toggleRow.removeFromLeft(8);
        soloToggle.setBounds(toggleRow.removeFromLeft(70));
        toggleRow.removeFromLeft(8);
        liveArmToggle.setBounds(toggleRow.removeFromLeft(70));
        area.removeFromTop(8);

        summaryEditor.setBounds(area);
    }

private:
    class TrackListModel final : public juce::ListBoxModel
    {
    public:
        TrackListModel(ProjectGetter projectGetterIn,
                       SelectedTrackSetter selectedTrackSetterIn)
            : projectGetter(std::move(projectGetterIn)),
              selectedTrackSetter(std::move(selectedTrackSetterIn))
        {
        }

        int getNumRows() override
        {
            return static_cast<int>(projectGetter().tracks.size());
        }

        void paintListBoxItem(int rowNumber,
                              juce::Graphics& g,
                              int width,
                              int height,
                              bool rowIsSelected) override
        {
            juce::ignoreUnused(height);
            const auto& project = projectGetter();
            if (!juce::isPositiveAndBelow(rowNumber, static_cast<int>(project.tracks.size())))
                return;

            const auto& track = project.tracks[static_cast<size_t>(rowNumber)];
            juce::StringArray trackFlags;
            if (track.mute)
                trackFlags.add("M");
            if (track.solo)
                trackFlags.add("S");
            if (track.liveArmed)
                trackFlags.add("ARM");

            const auto background = (rowNumber % 2) == 0 ? juce::Colour::fromRGB(26, 30, 37)
                                                         : juce::Colour::fromRGB(21, 25, 31);
            const auto trackColour = trackDisplayColour(track, rowNumber);
            g.fillAll(background);
            g.setColour(trackColour.withAlpha(rowIsSelected ? 0.78f : 0.14f));
            g.fillRect(0, 0, width, height);
            g.setColour(trackColour);
            g.fillRect(0, 0, 5, height);

            g.setColour(rowIsSelected ? trackTextColour(trackColour) : trackColour.brighter(0.18f));
            g.setFont(juce::FontOptions(13.5f, juce::Font::bold));
            g.drawText(track.name, 8, 2, width - 16, 18, juce::Justification::centredLeft, true);

            juce::String detail = track.trackType + " | "
                + (track.instrument.isNotEmpty() ? track.instrument : track.rackVst);
            if (trackFlags.size() > 0)
                detail << " | " << trackFlags.joinIntoString(" ");

            g.setFont(juce::FontOptions(11.8f));
            g.drawText(detail, 8, 22, width - 16, 18, juce::Justification::centredLeft, true);
        }

        void selectedRowsChanged(int lastRowSelected) override
        {
            if (selectedTrackSetter != nullptr)
                selectedTrackSetter(lastRowSelected);
        }

    private:
        ProjectGetter projectGetter;
        SelectedTrackSetter selectedTrackSetter;
    };

    void configureButton(juce::TextButton& button, const juce::String& text)
    {
        button.setButtonText(text);
        button.setColour(juce::TextButton::buttonColourId, juce::Colour::fromRGB(48, 54, 66));
        button.setColour(juce::TextButton::textColourOffId, juce::Colours::white);
        addAndMakeVisible(button);
    }

    void configureEditor(juce::TextEditor& editor, const juce::String& placeholder)
    {
        editor.setMultiLine(false);
        editor.setScrollbarsShown(false);
        editor.setTextToShowWhenEmpty(placeholder, juce::Colour::fromRGB(122, 133, 149));
        editor.setColour(juce::TextEditor::backgroundColourId, juce::Colour::fromRGB(18, 20, 25));
        editor.setColour(juce::TextEditor::textColourId, juce::Colour::fromRGB(235, 239, 244));
        editor.setColour(juce::TextEditor::outlineColourId, juce::Colour::fromRGB(56, 64, 79));
        editor.setColour(juce::CaretComponent::caretColourId, juce::Colour::fromRGB(237, 242, 249));
        editor.setFont(juce::FontOptions(13.0f));
        addAndMakeVisible(editor);
    }

    void configureSlider(juce::Slider& slider,
                         double minimum,
                         double maximum,
                         double step,
                         const juce::String& suffix)
    {
        slider.setSliderStyle(juce::Slider::LinearHorizontal);
        slider.setRange(minimum, maximum, step);
        slider.setChangeNotificationOnlyOnRelease(true);
        slider.setTextBoxStyle(juce::Slider::TextBoxRight, false, 60, 22);
        slider.setTextValueSuffix(suffix);
        addAndMakeVisible(slider);
    }

    void applyTrackMutation(std::function<void(TrackState&)> mutation, const juce::String& actionName)
    {
        if (mutateSelectedTrack != nullptr)
            mutateSelectedTrack(std::move(mutation), actionName);
    }

    void refreshInspectorControls()
    {
        const auto& project = projectGetter();
        const auto selectedTrack = selectedTrackGetter != nullptr ? selectedTrackGetter() : -1;
        const auto* track = juce::isPositiveAndBelow(selectedTrack, static_cast<int>(project.tracks.size()))
            ? &project.tracks[static_cast<size_t>(selectedTrack)]
            : nullptr;

        if (track != nullptr)
        {
            syncingControls = true;
            trackNameEditor.setText(track->name, juce::dontSendNotification);
            trackTypeEditor.setText(track->trackType, juce::dontSendNotification);
            instrumentModeEditor.setText(track->instrumentMode, juce::dontSendNotification);
            instrumentEditor.setText(track->instrument, juce::dontSendNotification);
            rackVstEditor.setText(track->rackVst, juce::dontSendNotification);
            midiChannelSlider.setValue(track->midiChannel + 1, juce::dontSendNotification);
            midiProgramSlider.setValue(track->midiProgram, juce::dontSendNotification);
            volumeSlider.setValue(track->volume, juce::dontSendNotification);
            panSlider.setValue(track->pan, juce::dontSendNotification);
            muteToggle.setToggleState(track->mute, juce::dontSendNotification);
            soloToggle.setToggleState(track->solo, juce::dontSendNotification);
            liveArmToggle.setToggleState(track->liveArmed, juce::dontSendNotification);
            syncingControls = false;

            trackNameEditor.setEnabled(true);
            trackTypeEditor.setEnabled(true);
            instrumentModeEditor.setEnabled(true);
            instrumentEditor.setEnabled(true);
            rackVstEditor.setEnabled(true);
            midiChannelSlider.setEnabled(true);
            midiProgramSlider.setEnabled(true);
            volumeSlider.setEnabled(true);
            panSlider.setEnabled(true);
            muteToggle.setEnabled(true);
            soloToggle.setEnabled(true);
            liveArmToggle.setEnabled(true);
            summaryEditor.setText(trackSummaryGetter != nullptr ? trackSummaryGetter(selectedTrack) : track->name, false);
            return;
        }

        syncingControls = true;
        trackNameEditor.setText({}, juce::dontSendNotification);
        trackTypeEditor.setText({}, juce::dontSendNotification);
        instrumentModeEditor.setText({}, juce::dontSendNotification);
        instrumentEditor.setText({}, juce::dontSendNotification);
        rackVstEditor.setText({}, juce::dontSendNotification);
        midiChannelSlider.setValue(1.0, juce::dontSendNotification);
        midiProgramSlider.setValue(0.0, juce::dontSendNotification);
        volumeSlider.setValue(0.8, juce::dontSendNotification);
        panSlider.setValue(0.0, juce::dontSendNotification);
        muteToggle.setToggleState(false, juce::dontSendNotification);
        soloToggle.setToggleState(false, juce::dontSendNotification);
        liveArmToggle.setToggleState(false, juce::dontSendNotification);
        syncingControls = false;

        trackNameEditor.setEnabled(false);
        trackTypeEditor.setEnabled(false);
        instrumentModeEditor.setEnabled(false);
        instrumentEditor.setEnabled(false);
        rackVstEditor.setEnabled(false);
        midiChannelSlider.setEnabled(false);
        midiProgramSlider.setEnabled(false);
        volumeSlider.setEnabled(false);
        panSlider.setEnabled(false);
        muteToggle.setEnabled(false);
        soloToggle.setEnabled(false);
        liveArmToggle.setEnabled(false);
        summaryEditor.setText("No track selected.", false);
    }

    ProjectGetter projectGetter;
    SelectedTrackGetter selectedTrackGetter;
    SelectedTrackSetter selectedTrackSetter;
    SelectedTrackMutator mutateSelectedTrack;
    TrackSummaryGetter trackSummaryGetter;
    std::function<void()> addTrack;
    std::function<void()> duplicateTrack;
    std::function<void()> removeTrack;
    std::function<void()> openRackEditor;
    std::function<void()> saveRackState;
    std::function<void()> playTrack;
    std::function<void()> stopPlayback;
    bool syncingControls = false;
    TrackListModel trackListModel;

    juce::Label titleLabel;
    juce::TextButton addTrackButton;
    juce::TextButton duplicateTrackButton;
    juce::TextButton removeTrackButton;
    juce::TextButton openRackEditorButton;
    juce::TextButton saveRackStateButton;
    juce::TextButton playTrackButton;
    juce::TextButton stopButton;
    juce::ListBox trackList;
    juce::Label inspectorLabel;
    juce::TextEditor trackNameEditor;
    juce::TextEditor trackTypeEditor;
    juce::TextEditor instrumentModeEditor;
    juce::TextEditor instrumentEditor;
    juce::TextEditor rackVstEditor;
    juce::Slider midiChannelSlider;
    juce::Slider midiProgramSlider;
    juce::Slider volumeSlider;
    juce::Slider panSlider;
    juce::ToggleButton muteToggle;
    juce::ToggleButton soloToggle;
    juce::ToggleButton liveArmToggle;
    juce::TextEditor summaryEditor;
};

class RackBrowserWindowComponent final : public juce::Component
{
public:
    using ProjectGetter = std::function<const ProjectState&()>;
    using SelectedTrackGetter = std::function<int()>;

    RackBrowserWindowComponent(ProjectGetter projectGetterIn,
                               SelectedTrackGetter selectedTrackGetterIn,
                               std::function<void(const juce::String&)> assignRackIn,
                               std::function<void()> clearRackIn,
                               std::function<void()> autoAssignRackIn,
                               std::function<void()> importRackPluginIn,
                               std::function<void()> refreshCatalogIn,
                               std::function<void()> openRackEditorIn,
                               std::function<void()> saveRackStateIn,
                               std::function<void()> playTrackIn,
                               std::function<void()> stopPlaybackIn)
        : projectGetter(std::move(projectGetterIn)),
          selectedTrackGetter(std::move(selectedTrackGetterIn)),
          assignRack(std::move(assignRackIn)),
          clearRack(std::move(clearRackIn)),
          autoAssignRack(std::move(autoAssignRackIn)),
          importRackPlugin(std::move(importRackPluginIn)),
          refreshCatalog(std::move(refreshCatalogIn)),
          openRackEditor(std::move(openRackEditorIn)),
          saveRackState(std::move(saveRackStateIn)),
          playTrack(std::move(playTrackIn)),
          stopPlayback(std::move(stopPlaybackIn)),
          rackListModel(*this)
    {
        titleLabel.setText("Rack Browser", juce::dontSendNotification);
        titleLabel.setFont(juce::FontOptions(18.0f, juce::Font::bold));
        titleLabel.setColour(juce::Label::textColourId, juce::Colour::fromRGB(235, 239, 244));
        addAndMakeVisible(titleLabel);

        selectedTrackLabel.setColour(juce::Label::textColourId, juce::Colour::fromRGB(190, 199, 210));
        addAndMakeVisible(selectedTrackLabel);

        configureButton(refreshButton, "Refresh Folders");
        refreshButton.onClick = [this]
        {
            if (refreshCatalog != nullptr)
                refreshCatalog();
        };

        configureButton(importButton, "Import Plugin");
        importButton.onClick = [this]
        {
            if (importRackPlugin != nullptr)
                importRackPlugin();
        };

        configureButton(assignButton, "Assign To Track");
        assignButton.onClick = [this]
        {
            if (assignRack == nullptr)
                return;
            if (const auto* entry = currentEntry())
                assignRack(entry->path.isNotEmpty() ? entry->path : entry->name);
        };

        configureButton(autoAssignButton, "Auto Assign");
        autoAssignButton.onClick = [this]
        {
            if (autoAssignRack != nullptr)
                autoAssignRack();
        };

        configureButton(clearButton, "Clear Track Rack");
        clearButton.onClick = [this]
        {
            if (clearRack != nullptr)
                clearRack();
        };

        configureButton(openEditorButton, "Open Editor");
        openEditorButton.onClick = [this]
        {
            if (openRackEditor != nullptr)
                openRackEditor();
        };

        configureButton(saveStateButton, "Save State");
        saveStateButton.onClick = [this]
        {
            if (saveRackState != nullptr)
                saveRackState();
        };

        configureButton(playTrackButton, "Play Track");
        playTrackButton.onClick = [this]
        {
            if (playTrack != nullptr)
                playTrack();
        };

        configureButton(stopButton, "Stop");
        stopButton.onClick = [this]
        {
            if (stopPlayback != nullptr)
                stopPlayback();
        };

        rackList.setModel(&rackListModel);
        rackList.setRowHeight(48);
        rackList.setColour(juce::ListBox::backgroundColourId, juce::Colour::fromRGB(20, 22, 28));
        rackList.setOutlineThickness(1);
        addAndMakeVisible(rackList);

        detailsEditor.setMultiLine(true);
        detailsEditor.setReadOnly(true);
        detailsEditor.setScrollbarsShown(true);
        detailsEditor.setColour(juce::TextEditor::backgroundColourId, juce::Colour::fromRGB(18, 20, 25));
        detailsEditor.setColour(juce::TextEditor::textColourId, juce::Colour::fromRGB(226, 230, 237));
        detailsEditor.setColour(juce::TextEditor::outlineColourId, juce::Colour::fromRGB(56, 64, 79));
        detailsEditor.setFont(juce::FontOptions(13.0f));
        addAndMakeVisible(detailsEditor);
    }

    void refreshFromModel()
    {
        const auto& project = projectGetter();
        if (!juce::isPositiveAndBelow(selectedRackRow, static_cast<int>(project.vstRack.size())))
            selectedRackRow = -1;

        const auto selectedTrackIndex = selectedTrackGetter != nullptr ? selectedTrackGetter() : -1;
        if (selectedRackRow < 0)
        {
            if (const auto assignedIndex = assignedRackIndexForSelectedTrack(); assignedIndex >= 0)
                selectedRackRow = assignedIndex;
        }

        rackList.updateContent();
        if (selectedRackRow >= 0)
            rackList.selectRow(selectedRackRow, false, true);
        else
            rackList.deselectAllRows();

        if (juce::isPositiveAndBelow(selectedTrackIndex, static_cast<int>(project.tracks.size())))
        {
            const auto& track = project.tracks[static_cast<size_t>(selectedTrackIndex)];
            selectedTrackLabel.setText("Selected track: " + track.name
                                           + "  |  Mode: " + track.instrumentMode
                                           + "  |  Rack: " + (displayRackName(project, track).isNotEmpty() ? displayRackName(project, track) : "(none)"),
                                       juce::dontSendNotification);
        }
        else
        {
            selectedTrackLabel.setText("No track selected.", juce::dontSendNotification);
        }

        detailsEditor.setText(describeSelection(), false);

        const auto hasTrackSelection = juce::isPositiveAndBelow(selectedTrackIndex, static_cast<int>(project.tracks.size()));
        const auto hasRackSelection = currentEntry() != nullptr;
        assignButton.setEnabled(hasTrackSelection && hasRackSelection);
        clearButton.setEnabled(hasTrackSelection);
        autoAssignButton.setEnabled(hasTrackSelection);
        openEditorButton.setEnabled(hasTrackSelection);
        saveStateButton.setEnabled(hasTrackSelection);
        playTrackButton.setEnabled(hasTrackSelection);
        stopButton.setEnabled(hasTrackSelection);
    }

    void focusRackList()
    {
        rackList.grabKeyboardFocus();
    }

    void selectAssignedRack()
    {
        if (const auto assignedIndex = assignedRackIndexForSelectedTrack(); assignedIndex >= 0)
            selectRackByReference(projectGetter().vstRack[static_cast<size_t>(assignedIndex)].path);
    }

    void selectRackByReference(const juce::String& reference)
    {
        const auto index = findRackInstrumentIndexByReference(projectGetter(), reference);
        if (index < 0)
            return;

        selectedRackRow = index;
        rackList.selectRow(index, true, true);
        detailsEditor.setText(describeSelection(), false);
    }

    void paint(juce::Graphics& g) override
    {
        g.fillAll(juce::Colour::fromRGB(13, 15, 20));
        g.setColour(juce::Colour::fromRGB(31, 35, 44));
        g.drawRect(getLocalBounds(), 1);
    }

    void resized() override
    {
        auto area = getLocalBounds().reduced(14);
        auto header = area.removeFromTop(28);
        titleLabel.setBounds(header.removeFromLeft(160));
        selectedTrackLabel.setBounds(header);

        area.removeFromTop(8);
        auto row1 = area.removeFromTop(30);
        refreshButton.setBounds(row1.removeFromLeft(126));
        row1.removeFromLeft(6);
        importButton.setBounds(row1.removeFromLeft(116));
        row1.removeFromLeft(6);
        assignButton.setBounds(row1.removeFromLeft(126));
        row1.removeFromLeft(6);
        autoAssignButton.setBounds(row1.removeFromLeft(100));
        row1.removeFromLeft(6);
        clearButton.setBounds(row1.removeFromLeft(128));

        area.removeFromTop(8);
        auto row2 = area.removeFromTop(30);
        openEditorButton.setBounds(row2.removeFromLeft(96));
        row2.removeFromLeft(6);
        saveStateButton.setBounds(row2.removeFromLeft(90));
        row2.removeFromLeft(6);
        playTrackButton.setBounds(row2.removeFromLeft(90));
        row2.removeFromLeft(6);
        stopButton.setBounds(row2.removeFromLeft(70));

        area.removeFromTop(10);
        auto listArea = area.removeFromLeft(juce::jmin(420, juce::roundToInt(static_cast<float>(area.getWidth()) * 0.4f)));
        rackList.setBounds(listArea);
        area.removeFromLeft(12);
        detailsEditor.setBounds(area);
    }

private:
    class RackListModel final : public juce::ListBoxModel
    {
    public:
        explicit RackListModel(RackBrowserWindowComponent& ownerIn)
            : owner(ownerIn)
        {
        }

        int getNumRows() override
        {
            return static_cast<int>(owner.projectGetter().vstRack.size());
        }

        void paintListBoxItem(int rowNumber,
                              juce::Graphics& g,
                              int width,
                              int height,
                              bool rowIsSelected) override
        {
            juce::ignoreUnused(height);
            const auto& project = owner.projectGetter();
            if (!juce::isPositiveAndBelow(rowNumber, static_cast<int>(project.vstRack.size())))
                return;

            const auto& entry = project.vstRack[static_cast<size_t>(rowNumber)];
            const auto name = entry.name.isNotEmpty() ? entry.name
                : (entry.pluginName.isNotEmpty() ? entry.pluginName : juce::File(entry.path).getFileNameWithoutExtension());
            const auto detail = entry.path.isNotEmpty() ? entry.path : "(no plugin path)";
            const auto exists = entry.path.isNotEmpty() && juce::File(entry.path).exists();

            g.fillAll(rowIsSelected ? juce::Colour::fromRGB(46, 88, 138)
                                    : ((rowNumber % 2) == 0 ? juce::Colour::fromRGB(26, 30, 37)
                                                            : juce::Colour::fromRGB(21, 25, 31)));

            g.setColour(rowIsSelected ? juce::Colours::white : juce::Colour::fromRGB(235, 239, 244));
            g.setFont(juce::FontOptions(13.5f, juce::Font::bold));
            g.drawText(name, 8, 2, width - 90, 18, juce::Justification::centredLeft, true);

            g.setFont(juce::FontOptions(11.5f));
            g.drawText(detail, 8, 22, width - 90, 18, juce::Justification::centredLeft, true);

            const auto badgeBounds = juce::Rectangle<int>(width - 78, 12, 68, 20);
            g.setColour((entry.hostSupported && exists) ? juce::Colour::fromRGB(66, 139, 92)
                                                        : juce::Colour::fromRGB(137, 76, 76));
            g.fillRoundedRectangle(badgeBounds.toFloat(), 5.0f);
            g.setColour(juce::Colours::white);
            g.setFont(juce::FontOptions(10.5f, juce::Font::bold));
            g.drawFittedText((entry.hostSupported && exists) ? "READY" : "CHECK",
                             badgeBounds,
                             juce::Justification::centred,
                             1);
        }

        void selectedRowsChanged(int lastRowSelected) override
        {
            owner.selectedRackRow = lastRowSelected;
            owner.detailsEditor.setText(owner.describeSelection(), false);
        }

    private:
        RackBrowserWindowComponent& owner;
    };

    const VstInstrument* currentEntry() const
    {
        const auto& project = projectGetter();
        if (!juce::isPositiveAndBelow(selectedRackRow, static_cast<int>(project.vstRack.size())))
            return nullptr;
        return &project.vstRack[static_cast<size_t>(selectedRackRow)];
    }

    int assignedRackIndexForSelectedTrack() const
    {
        const auto& project = projectGetter();
        const auto selectedTrackIndex = selectedTrackGetter != nullptr ? selectedTrackGetter() : -1;
        if (!juce::isPositiveAndBelow(selectedTrackIndex, static_cast<int>(project.tracks.size())))
            return -1;
        return findRackInstrumentIndexByReference(project, project.tracks[static_cast<size_t>(selectedTrackIndex)].rackVst);
    }

    juce::String describeSelection() const
    {
        juce::StringArray lines;
        const auto& project = projectGetter();
        const auto selectedTrackIndex = selectedTrackGetter != nullptr ? selectedTrackGetter() : -1;
        if (juce::isPositiveAndBelow(selectedTrackIndex, static_cast<int>(project.tracks.size())))
        {
            const auto& track = project.tracks[static_cast<size_t>(selectedTrackIndex)];
            lines.add("Track");
            lines.add("Name: " + track.name);
            lines.add("Mode: " + track.instrumentMode);
            lines.add("Rack: " + (displayRackName(project, track).isNotEmpty() ? displayRackName(project, track) : "(none)"));
            lines.add("Resolved path: " + (resolveRackPluginPath(project, track).isNotEmpty() ? resolveRackPluginPath(project, track) : "(unresolved)"));
            lines.add("State path: " + (track.vstiStatePath.isNotEmpty() ? track.vstiStatePath : "(none)"));
        }
        else
        {
            lines.add("Track");
            lines.add("No track selected.");
        }

        lines.add({});
        lines.add("Rack Entry");

        if (const auto* entry = currentEntry())
        {
            const auto exists = entry->path.isNotEmpty() && juce::File(entry->path).exists();
            lines.add("Name: " + (entry->name.isNotEmpty() ? entry->name : "(unnamed)"));
            lines.add("Plugin name: " + (entry->pluginName.isNotEmpty() ? entry->pluginName : "(unknown)"));
            lines.add("Path: " + (entry->path.isNotEmpty() ? entry->path : "(none)"));
            lines.add("Category: " + (entry->category.isNotEmpty() ? entry->category : "(none)"));
            lines.add("Instrument: " + juce::String(entry->isInstrument ? "yes" : "no"));
            lines.add("Host supported: " + juce::String(entry->hostSupported ? "yes" : "no"));
            lines.add("Path exists: " + juce::String(exists ? "yes" : "no"));
            if (entry->hostError.isNotEmpty())
                lines.add("Host error: " + entry->hostError);
        }
        else
        {
            lines.add("No rack entry selected.");
        }

        return lines.joinIntoString("\n");
    }

    void configureButton(juce::TextButton& button, const juce::String& text)
    {
        button.setButtonText(text);
        button.setColour(juce::TextButton::buttonColourId, juce::Colour::fromRGB(48, 54, 66));
        button.setColour(juce::TextButton::textColourOffId, juce::Colours::white);
        addAndMakeVisible(button);
    }

    ProjectGetter projectGetter;
    SelectedTrackGetter selectedTrackGetter;
    std::function<void(const juce::String&)> assignRack;
    std::function<void()> clearRack;
    std::function<void()> autoAssignRack;
    std::function<void()> importRackPlugin;
    std::function<void()> refreshCatalog;
    std::function<void()> openRackEditor;
    std::function<void()> saveRackState;
    std::function<void()> playTrack;
    std::function<void()> stopPlayback;
    int selectedRackRow = -1;
    RackListModel rackListModel;

    juce::Label titleLabel;
    juce::Label selectedTrackLabel;
    juce::TextButton refreshButton;
    juce::TextButton importButton;
    juce::TextButton assignButton;
    juce::TextButton autoAssignButton;
    juce::TextButton clearButton;
    juce::TextButton openEditorButton;
    juce::TextButton saveStateButton;
    juce::TextButton playTrackButton;
    juce::TextButton stopButton;
    juce::ListBox rackList;
    juce::TextEditor detailsEditor;
};

class RenderManagerWindowComponent final : public juce::Component
{
public:
    using ProjectGetter = std::function<const ProjectState&()>;
    using SelectedTrackGetter = std::function<int()>;
    using SelectedTrackSetter = std::function<void(int)>;

    RenderManagerWindowComponent(ProjectGetter projectGetterIn,
                                 SelectedTrackGetter selectedTrackGetterIn,
                                 SelectedTrackSetter selectedTrackSetterIn,
                                 std::function<void()> exportTrackIn,
                                 std::function<void()> exportStemsIn,
                                 std::function<void()> relinkRenderIn,
                                 std::function<void()> clearRenderIn,
                                 std::function<void()> importRenderIn,
                                 std::function<void()> placeRenderIn)
        : projectGetter(std::move(projectGetterIn)),
          selectedTrackGetter(std::move(selectedTrackGetterIn)),
          selectedTrackSetter(std::move(selectedTrackSetterIn)),
          exportTrack(std::move(exportTrackIn)),
          exportStems(std::move(exportStemsIn)),
          relinkRender(std::move(relinkRenderIn)),
          clearRender(std::move(clearRenderIn)),
          importRender(std::move(importRenderIn)),
          placeRender(std::move(placeRenderIn)),
          trackListModel(*this)
    {
        titleLabel.setText("Render Manager", juce::dontSendNotification);
        titleLabel.setFont(juce::FontOptions(18.0f, juce::Font::bold));
        titleLabel.setColour(juce::Label::textColourId, juce::Colour::fromRGB(235, 239, 244));
        addAndMakeVisible(titleLabel);

        configureButton(exportTrackButton, "Export Track");
        exportTrackButton.onClick = [this]
        {
            if (exportTrack != nullptr)
                exportTrack();
        };

        configureButton(exportStemsButton, "Export Stems");
        exportStemsButton.onClick = [this]
        {
            if (exportStems != nullptr)
                exportStems();
        };

        configureButton(relinkButton, "Relink File");
        relinkButton.onClick = [this]
        {
            if (relinkRender != nullptr)
                relinkRender();
        };

        configureButton(clearButton, "Clear Render");
        clearButton.onClick = [this]
        {
            if (clearRender != nullptr)
                clearRender();
        };

        configureButton(importButton, "Add To Library");
        importButton.onClick = [this]
        {
            if (importRender != nullptr)
                importRender();
        };

        configureButton(placeButton, "Place At Playhead");
        placeButton.onClick = [this]
        {
            if (placeRender != nullptr)
                placeRender();
        };

        trackList.setModel(&trackListModel);
        trackList.setRowHeight(50);
        trackList.setColour(juce::ListBox::backgroundColourId, juce::Colour::fromRGB(20, 22, 28));
        trackList.setOutlineThickness(1);
        addAndMakeVisible(trackList);

        detailsEditor.setMultiLine(true);
        detailsEditor.setReadOnly(true);
        detailsEditor.setScrollbarsShown(true);
        detailsEditor.setColour(juce::TextEditor::backgroundColourId, juce::Colour::fromRGB(18, 20, 25));
        detailsEditor.setColour(juce::TextEditor::textColourId, juce::Colour::fromRGB(226, 230, 237));
        detailsEditor.setColour(juce::TextEditor::outlineColourId, juce::Colour::fromRGB(56, 64, 79));
        detailsEditor.setFont(juce::FontOptions(13.0f));
        addAndMakeVisible(detailsEditor);
    }

    void refreshFromModel()
    {
        const auto selectedTrack = selectedTrackGetter != nullptr ? selectedTrackGetter() : -1;
        trackList.updateContent();
        if (selectedTrack >= 0)
            trackList.selectRow(selectedTrack, false, true);
        else
            trackList.deselectAllRows();

        detailsEditor.setText(describeSelection(), false);

        const auto hasSelection = selectedTrack >= 0
            && selectedTrack < static_cast<int>(projectGetter().tracks.size());
        exportTrackButton.setEnabled(hasSelection);
        relinkButton.setEnabled(hasSelection);
        clearButton.setEnabled(hasSelection);
        importButton.setEnabled(hasSelection && selectedTrackHasRenderFile());
        placeButton.setEnabled(hasSelection && selectedTrackHasRenderFile());
        exportStemsButton.setEnabled(!projectGetter().tracks.empty());
    }

    void focusTrackList()
    {
        trackList.grabKeyboardFocus();
    }

    void paint(juce::Graphics& g) override
    {
        g.fillAll(juce::Colour::fromRGB(13, 15, 20));
        g.setColour(juce::Colour::fromRGB(31, 35, 44));
        g.drawRect(getLocalBounds(), 1);
    }

    void resized() override
    {
        auto area = getLocalBounds().reduced(14);
        auto header = area.removeFromTop(30);
        titleLabel.setBounds(header.removeFromLeft(160));
        exportTrackButton.setBounds(header.removeFromLeft(104));
        header.removeFromLeft(6);
        exportStemsButton.setBounds(header.removeFromLeft(102));
        header.removeFromLeft(6);
        relinkButton.setBounds(header.removeFromLeft(94));
        header.removeFromLeft(6);
        clearButton.setBounds(header.removeFromLeft(100));
        header.removeFromLeft(6);
        importButton.setBounds(header.removeFromLeft(102));
        header.removeFromLeft(6);
        placeButton.setBounds(header.removeFromLeft(126));

        area.removeFromTop(10);
        auto listArea = area.removeFromLeft(juce::jmin(360, juce::roundToInt(static_cast<float>(area.getWidth()) * 0.33f)));
        trackList.setBounds(listArea);
        area.removeFromLeft(12);
        detailsEditor.setBounds(area);
    }

private:
    class TrackListModel final : public juce::ListBoxModel
    {
    public:
        explicit TrackListModel(RenderManagerWindowComponent& ownerIn)
            : owner(ownerIn)
        {
        }

        int getNumRows() override
        {
            return static_cast<int>(owner.projectGetter().tracks.size());
        }

        void paintListBoxItem(int rowNumber,
                              juce::Graphics& g,
                              int width,
                              int height,
                              bool rowIsSelected) override
        {
            juce::ignoreUnused(height);
            const auto& project = owner.projectGetter();
            if (!juce::isPositiveAndBelow(rowNumber, static_cast<int>(project.tracks.size())))
                return;

            const auto& track = project.tracks[static_cast<size_t>(rowNumber)];
            const auto renderFile = juce::File(track.renderedAudioPath);
            const auto hasRender = track.renderedAudioPath.isNotEmpty();
            const auto renderReady = hasRender && renderFile.existsAsFile();

            const auto background = (rowNumber % 2) == 0 ? juce::Colour::fromRGB(26, 30, 37)
                                                         : juce::Colour::fromRGB(21, 25, 31);
            const auto trackColour = trackDisplayColour(track, rowNumber);
            g.fillAll(background);
            g.setColour(trackColour.withAlpha(rowIsSelected ? 0.78f : 0.14f));
            g.fillRect(0, 0, width, height);
            g.setColour(trackColour);
            g.fillRect(0, 0, 5, height);

            g.setColour(rowIsSelected ? trackTextColour(trackColour) : trackColour.brighter(0.18f));
            g.setFont(juce::FontOptions(13.5f, juce::Font::bold));
            g.drawText(track.name.trim().isNotEmpty() ? track.name : ("Track " + juce::String(rowNumber + 1)),
                       8, 2, width - 90, 18, juce::Justification::centredLeft, true);

            juce::String detail = track.trackType + " | ";
            if (renderReady)
                detail << renderFile.getFileName();
            else if (hasRender)
                detail << "missing render";
            else
                detail << "no render";

            g.setFont(juce::FontOptions(11.6f));
            g.drawText(detail, 8, 24, width - 90, 18, juce::Justification::centredLeft, true);

            const auto badgeBounds = juce::Rectangle<int>(width - 78, 14, 68, 20);
            g.setColour(renderReady ? juce::Colour::fromRGB(66, 139, 92)
                                    : (hasRender ? juce::Colour::fromRGB(151, 109, 62)
                                                 : juce::Colour::fromRGB(86, 92, 104)));
            g.fillRoundedRectangle(badgeBounds.toFloat(), 5.0f);
            g.setColour(juce::Colours::white);
            g.setFont(juce::FontOptions(10.2f, juce::Font::bold));
            g.drawFittedText(renderReady ? "READY" : (hasRender ? "MISSING" : "EMPTY"),
                             badgeBounds,
                             juce::Justification::centred,
                             1);
        }

        void selectedRowsChanged(int lastRowSelected) override
        {
            if (owner.selectedTrackSetter != nullptr)
                owner.selectedTrackSetter(lastRowSelected);
            owner.detailsEditor.setText(owner.describeSelection(), false);
        }

    private:
        RenderManagerWindowComponent& owner;
    };

    bool selectedTrackHasRenderFile() const
    {
        const auto& project = projectGetter();
        const auto selectedTrack = selectedTrackGetter != nullptr ? selectedTrackGetter() : -1;
        if (!juce::isPositiveAndBelow(selectedTrack, static_cast<int>(project.tracks.size())))
            return false;
        const auto path = project.tracks[static_cast<size_t>(selectedTrack)].renderedAudioPath.trim();
        return path.isNotEmpty() && juce::File(path).existsAsFile();
    }

    juce::String describeSelection() const
    {
        const auto& project = projectGetter();
        const auto selectedTrack = selectedTrackGetter != nullptr ? selectedTrackGetter() : -1;
        if (!juce::isPositiveAndBelow(selectedTrack, static_cast<int>(project.tracks.size())))
            return "No track selected.";

        const auto& track = project.tracks[static_cast<size_t>(selectedTrack)];
        const auto renderPath = track.renderedAudioPath.trim();
        const auto renderFile = juce::File(renderPath);

        int clipReferenceCount = 0;
        for (const auto& clip : project.sampleClips)
        {
            if (clip.path.equalsIgnoreCase(renderPath))
                ++clipReferenceCount;
        }

        juce::StringArray lines;
        lines.add("Track");
        lines.add("Name: " + (track.name.trim().isNotEmpty() ? track.name : ("Track " + juce::String(selectedTrack + 1))));
        lines.add("Type: " + track.trackType);
        lines.add("Instrument: " + track.instrument);
        lines.add("Rack: " + (displayRackName(project, track).isNotEmpty() ? displayRackName(project, track) : "(none)"));
        lines.add("Notes: " + juce::String(static_cast<int>(track.notes.size())));
        lines.add("Locator render path: " + (renderPath.isNotEmpty() ? renderPath : "(none)"));
        lines.add("Render exists: " + juce::String(renderPath.isNotEmpty() && renderFile.existsAsFile() ? "yes" : "no"));
        lines.add("Clip references: " + juce::String(clipReferenceCount));
        if (renderPath.isNotEmpty() && renderFile.existsAsFile())
        {
            lines.add("Render file: " + renderFile.getFileName());
            lines.add("Render folder: " + renderFile.getParentDirectory().getFullPathName());
        }
        return lines.joinIntoString("\n");
    }

    void configureButton(juce::TextButton& button, const juce::String& text)
    {
        button.setButtonText(text);
        button.setColour(juce::TextButton::buttonColourId, juce::Colour::fromRGB(48, 54, 66));
        button.setColour(juce::TextButton::textColourOffId, juce::Colours::white);
        addAndMakeVisible(button);
    }

    ProjectGetter projectGetter;
    SelectedTrackGetter selectedTrackGetter;
    SelectedTrackSetter selectedTrackSetter;
    std::function<void()> exportTrack;
    std::function<void()> exportStems;
    std::function<void()> relinkRender;
    std::function<void()> clearRender;
    std::function<void()> importRender;
    std::function<void()> placeRender;
    TrackListModel trackListModel;

    juce::Label titleLabel;
    juce::TextButton exportTrackButton;
    juce::TextButton exportStemsButton;
    juce::TextButton relinkButton;
    juce::TextButton clearButton;
    juce::TextButton importButton;
    juce::TextButton placeButton;
    juce::ListBox trackList;
    juce::TextEditor detailsEditor;
};

namespace
{
constexpr int kColumnMute = 1;
constexpr int kColumnSolo = 2;
constexpr int kColumnVstView = 3;
constexpr int kColumnName = 4;
constexpr int kColumnType = 5;
constexpr int kColumnMode = 6;
constexpr int kColumnRack = 7;
constexpr int kColumnNotes = 8;
constexpr int kColumnChannel = 9;
constexpr int kColumnVolume = 10;
constexpr int kColumnPan = 11;
constexpr int kColumnFlags = 12;
constexpr int kDefaultWindowWidth = 1480;
constexpr int kDefaultWindowHeight = 920;
const char* kProjectFileWildcard = "*.aims;*.json";
const char* kMidiFileWildcard = "*.mid;*.midi";

void paintTrackLevelBar(juce::Graphics& g,
                        juce::Rectangle<int> bounds,
                        float level,
                        float volumeMarker,
                        juce::Colour accentColour)
{
    bounds = bounds.reduced(2);
    if (bounds.isEmpty())
        return;

    g.setColour(juce::Colour::fromRGB(20, 23, 30));
    g.fillRect(bounds);
    g.setColour(accentColour.withAlpha(0.42f));
    g.drawRect(bounds, 1);

    const auto filledHeight = juce::roundToInt(static_cast<float>(bounds.getHeight()) * juce::jlimit(0.0f, 1.0f, level));
    if (filledHeight > 0)
    {
        auto filledBounds = bounds.withTrimmedTop(bounds.getHeight() - filledHeight).reduced(1, 1);
        const auto meterColour = level >= 0.85f
            ? juce::Colour::fromRGB(242, 99, 84)
            : (level >= 0.60f ? juce::Colour::fromRGB(234, 192, 78)
                              : juce::Colour::fromRGB(61, 210, 122));
        g.setColour(meterColour);
        g.fillRect(filledBounds);
    }

    const auto clampedMarker = juce::jlimit(0.0f, 1.0f, volumeMarker);
    const auto markerY = juce::roundToInt(static_cast<float>(bounds.getBottom()) - (static_cast<float>(bounds.getHeight()) * clampedMarker));
    g.setColour(accentColour.brighter(0.3f));
    g.drawLine(static_cast<float>(bounds.getX() + 2),
               static_cast<float>(markerY),
               static_cast<float>(bounds.getRight() - 2),
               static_cast<float>(markerY),
               1.6f);
}

void discoverBundledVstEntriesRecursive(const juce::File& directory, std::vector<VstInstrument>& entries)
{
    juce::Array<juce::File> children;
    directory.findChildFiles(children, juce::File::findFilesAndDirectories, false);

    for (const auto& child : children)
    {
        if (child.isDirectory())
        {
            const auto name = child.getFileName();
            if (name.endsWithIgnoreCase(".disabled"))
                continue;

            if (name.endsWithIgnoreCase(".vst3"))
            {
                VstInstrument instrument;
                instrument.name = child.getFileNameWithoutExtension();
                instrument.path = child.getFullPathName();
                instrument.pluginName = instrument.name;
                instrument.isInstrument = true;
                instrument.isEffect = false;
                instrument.category = "Instrument";
                entries.push_back(std::move(instrument));
                continue;
            }

            discoverBundledVstEntriesRecursive(child, entries);
            continue;
        }

        if (!child.hasFileExtension(".dll;.so;.vst3"))
            continue;

        VstInstrument instrument;
        instrument.name = child.getFileNameWithoutExtension();
        instrument.path = child.getFullPathName();
        instrument.pluginName = instrument.name;
        instrument.isInstrument = true;
        instrument.isEffect = false;
        instrument.category = "Instrument";
        entries.push_back(std::move(instrument));
    }
}

juce::File findBundledVstDirectory()
{
    juce::Array<juce::File> baseDirectories;
    baseDirectories.add(juce::File::getCurrentWorkingDirectory());
    baseDirectories.add(juce::File::getSpecialLocation(juce::File::currentApplicationFile).getParentDirectory());

    for (const auto& baseDirectory : baseDirectories)
    {
        auto probe = baseDirectory;
        for (int depth = 0; depth < 8 && probe != juce::File(); ++depth)
        {
            const auto candidate = probe.getChildFile("vsti");
            if (candidate.isDirectory())
                return candidate;

            probe = probe.getParentDirectory();
        }
    }

    return {};
}

std::vector<VstInstrument> discoverBundledVstCatalog()
{
    const auto bundledDirectory = findBundledVstDirectory();
    if (bundledDirectory == juce::File())
        return {};

    std::vector<VstInstrument> entries;
    discoverBundledVstEntriesRecursive(bundledDirectory, entries);

    std::sort(entries.begin(),
              entries.end(),
              [] (const VstInstrument& lhs, const VstInstrument& rhs)
              {
                  return lhs.name.compareIgnoreCase(rhs.name) < 0;
              });

    entries.erase(std::unique(entries.begin(),
                              entries.end(),
                              [] (const VstInstrument& lhs, const VstInstrument& rhs)
                              {
                                  return lhs.path.equalsIgnoreCase(rhs.path);
                              }),
                  entries.end());
    return entries;
}

std::vector<VstInstrument> discoverVstCatalogInDirectory(const juce::File& directory)
{
    if (directory == juce::File() || !directory.isDirectory())
        return {};

    std::vector<VstInstrument> entries;
    discoverBundledVstEntriesRecursive(directory, entries);

    std::sort(entries.begin(),
              entries.end(),
              [] (const VstInstrument& lhs, const VstInstrument& rhs)
              {
                  return lhs.path.compareIgnoreCase(rhs.path) < 0;
              });

    entries.erase(std::unique(entries.begin(),
                              entries.end(),
                              [] (const VstInstrument& lhs, const VstInstrument& rhs)
                              {
                                  return lhs.path.equalsIgnoreCase(rhs.path);
                              }),
                  entries.end());
    return entries;
}

juce::String makeTrackFlags(const TrackState& track)
{
    juce::StringArray flags;
    if (track.liveArmed)
        flags.add("ARM");
    return flags.joinIntoString(" ");
}

juce::String describeTrack(const ProjectState& project, const TrackState& track)
{
    juce::StringArray lines;
    lines.add("Name: " + track.name);
    lines.add("Type: " + track.trackType);
    lines.add("Instrument: " + track.instrument);
    lines.add("Mode: " + track.instrumentMode);
    lines.add("Rack VST: " + (displayRackName(project, track).isNotEmpty() ? displayRackName(project, track) : "(none)"));
    const auto resolvedRackPath = resolveRackPluginPath(project, track);
    lines.add("Rack path: " + (resolvedRackPath.isNotEmpty() ? resolvedRackPath : "(unresolved)"));
    lines.add("MIDI channel: " + juce::String(track.midiChannel + 1));
    lines.add("Program: " + juce::String(track.midiProgram));
    lines.add("Notes: " + juce::String(static_cast<int>(track.notes.size())));
    lines.add("Automation lanes: " + juce::String(static_cast<int>(track.automationLanes.size())));
    lines.add("Volume: " + juce::String(track.volume, 2));
    lines.add("Pan: " + juce::String(track.pan, 2));
    lines.add("Render path: " + (track.renderedAudioPath.isNotEmpty() ? track.renderedAudioPath : "(none)"));

    if (!track.notes.empty())
    {
        lines.add({});
        lines.add("First notes:");
        const auto previewCount = juce::jmin<int>(10, static_cast<int>(track.notes.size()));
        for (int index = 0; index < previewCount; ++index)
        {
            const auto& note = track.notes[static_cast<size_t>(index)];
            lines.add("  Tick " + juce::String(note.startTick)
                + " len " + juce::String(note.durationTick)
                + " pitch " + juce::String(note.pitch)
                + " vel " + juce::String(note.velocity)
                + (note.selected ? " [selected]" : ""));
        }
    }

    return lines.joinIntoString("\n");
}

juce::NamedValueSet parameterValuesFromHostStatus(const juce::var& status)
{
    juce::NamedValueSet values;
    auto* object = status.getDynamicObject();
    if (object == nullptr)
        return values;

    auto* parameterArray = object->getProperty("plugin_parameters").getArray();
    if (parameterArray == nullptr)
        return values;

    for (int index = 0; index < parameterArray->size(); ++index)
    {
        const auto& parameterVar = parameterArray->getReference(index);
        auto* parameterObject = parameterVar.getDynamicObject();
        if (parameterObject == nullptr)
            continue;

        auto name = parameterObject->getProperty("name").toString().trim();
        if (name.isEmpty() || name == "-" || name == "--" || name == "---")
            name = "Param " + juce::String(index + 1);

        auto normalizedValue = static_cast<double>(parameterObject->hasProperty("normalized_value")
            ? parameterObject->getProperty("normalized_value")
            : parameterObject->getProperty("value"));
        if (!std::isfinite(normalizedValue))
            normalizedValue = 0.0;

        values.set(juce::Identifier(name), juce::jlimit(0.0, 100.0, normalizedValue * 100.0));
    }

    return values;
}

juce::String parameterSetSignature(const juce::NamedValueSet& values)
{
    juce::StringArray entries;
    for (int index = 0; index < values.size(); ++index)
    {
        const auto name = values.getName(index).toString().trim();
        if (name.isEmpty())
            continue;

        const auto value = static_cast<double>(values.getValueAt(index));
        entries.add(name + "=" + juce::String(std::isfinite(value) ? value : 0.0, 4));
    }

    entries.sort(true);
    return entries.joinIntoString("|");
}

juce::String noteTransportSignature(const std::vector<MidiNote>& notes)
{
    juce::StringArray entries;
    entries.ensureStorageAllocated(static_cast<int>(notes.size()));
    for (const auto& note : notes)
    {
        entries.add(juce::String(note.startTick)
            + ":" + juce::String(note.durationTick)
            + ":" + juce::String(note.pitch)
            + ":" + juce::String(note.velocity));
    }
    return entries.joinIntoString("|");
}

bool projectTrackIsAudible(const ProjectState& project, int trackIndex)
{
    bool anySolo = false;
    for (const auto& track : project.tracks)
    {
        if (track.solo)
        {
            anySolo = true;
            break;
        }
    }

    if (!juce::isPositiveAndBelow(trackIndex, static_cast<int>(project.tracks.size())))
        return false;

    const auto& track = project.tracks[static_cast<size_t>(trackIndex)];
    if (track.mute)
        return false;
    if (anySolo && !track.solo)
        return false;
    return true;
}

void normaliseTrack(TrackState& track)
{
    std::sort(track.notes.begin(),
              track.notes.end(),
              [] (const MidiNote& lhs, const MidiNote& rhs)
              {
                  if (lhs.startTick != rhs.startTick)
                      return lhs.startTick < rhs.startTick;
                  if (lhs.pitch != rhs.pitch)
                      return lhs.pitch > rhs.pitch;
                  return lhs.durationTick < rhs.durationTick;
              });

    track.midiProgram = juce::jlimit(0, 127, track.midiProgram);
    track.midiChannel = juce::jlimit(0, 15, track.midiChannel);
    sanitiseAutomationLanes(track);
}

void normaliseProject(ProjectState& project)
{
    if (project.tracks.empty())
        project.tracks.push_back(TrackState{});

    for (size_t index = 0; index < project.tracks.size(); ++index)
    {
        auto& track = project.tracks[index];
        if (track.colorHex.trim().isEmpty())
            track.colorHex = defaultTrackColour(static_cast<int>(index)).toDisplayString(false);
        normaliseTrack(track);
    }

    project.recalculateTimeFields();
}

class TrackEditAction final : public juce::UndoableAction
{
public:
    TrackEditAction(std::function<void(const TrackState&)> applyStateIn,
                    TrackState beforeStateIn,
                    TrackState afterStateIn)
        : applyState(std::move(applyStateIn)),
          beforeState(std::move(beforeStateIn)),
          afterState(std::move(afterStateIn))
    {
    }

    bool perform() override
    {
        applyState(afterState);
        return true;
    }

    bool undo() override
    {
        applyState(beforeState);
        return true;
    }

    int getSizeInUnits() override
    {
        return 32;
    }

private:
    std::function<void(const TrackState&)> applyState;
    TrackState beforeState;
    TrackState afterState;
};

class ProjectEditAction final : public juce::UndoableAction
{
public:
    ProjectEditAction(std::function<void(const ProjectState&)> applyStateIn,
                      ProjectState beforeStateIn,
                      ProjectState afterStateIn)
        : applyState(std::move(applyStateIn)),
          beforeState(std::move(beforeStateIn)),
          afterState(std::move(afterStateIn))
    {
    }

    bool perform() override
    {
        applyState(afterState);
        return true;
    }

    bool undo() override
    {
        applyState(beforeState);
        return true;
    }

    int getSizeInUnits() override
    {
        return 64;
    }

private:
    std::function<void(const ProjectState&)> applyState;
    ProjectState beforeState;
    ProjectState afterState;
};
} // namespace

StudioShellComponent::TrackTableModel::TrackTableModel(StudioShellComponent& ownerIn)
    : owner(ownerIn)
{
}

int StudioShellComponent::TrackTableModel::getNumRows()
{
    return static_cast<int>(owner.documentState.project.tracks.size());
}

void StudioShellComponent::TrackTableModel::paintRowBackground(juce::Graphics& g,
                                                               int rowNumber,
                                                               int width,
                                                               int height,
                                                               bool rowIsSelected)
{
    juce::ignoreUnused(height);

    const auto background = (rowNumber % 2) == 0 ? juce::Colour::fromRGB(26, 29, 36)
                                                 : juce::Colour::fromRGB(21, 24, 30);
    g.fillAll(background);

    if (!juce::isPositiveAndBelow(rowNumber, static_cast<int>(owner.documentState.project.tracks.size())))
        return;

    const auto trackColour = trackDisplayColour(owner.documentState.project.tracks[static_cast<size_t>(rowNumber)], rowNumber);
    g.setColour(trackColour.withAlpha(rowIsSelected ? 0.78f : 0.16f));
    g.fillRect(0, 0, width, height);
    g.setColour(trackColour);
    g.fillRect(0, 0, 5, height);
}

void StudioShellComponent::TrackTableModel::paintCell(juce::Graphics& g,
                                                      int rowNumber,
                                                      int columnId,
                                                      int width,
                                                      int height,
                                                      bool rowIsSelected)
{
    juce::ignoreUnused(height);
    if (rowNumber < 0 || rowNumber >= static_cast<int>(owner.documentState.project.tracks.size()))
        return;

    const auto& track = owner.documentState.project.tracks[static_cast<size_t>(rowNumber)];
    const auto trackColour = trackDisplayColour(track, rowNumber);

    if (columnId == kColumnMute || columnId == kColumnSolo || columnId == kColumnVstView)
    {
        const bool canToggleVstView = track.trackType != "sample"
            && track.instrumentMode.containsIgnoreCase("VST");
        const auto active = columnId == kColumnMute ? track.mute
                           : (columnId == kColumnSolo ? track.solo
                                                      : (canToggleVstView && owner.isTrackBeingLiveEdited(rowNumber)));
        const auto label = columnId == kColumnMute ? "M"
                         : (columnId == kColumnSolo ? "S" : "V");
        const auto activeColour = columnId == kColumnMute
            ? juce::Colour::fromRGB(232, 96, 96)
            : (columnId == kColumnSolo
                   ? juce::Colour::fromRGB(244, 211, 94)
                   : juce::Colour::fromRGB(108, 212, 255));
        auto chipBounds = juce::Rectangle<int>(6, 6, juce::jmax(20, width - 12), juce::jmax(22, height - 12));

        const auto inactiveChipColour = columnId == kColumnVstView && !canToggleVstView
            ? juce::Colour::fromRGB(30, 34, 42)
            : juce::Colour::fromRGB(43, 49, 61);
        g.setColour(active ? activeColour : inactiveChipColour);
        g.fillRoundedRectangle(chipBounds.toFloat(), 6.0f);
        g.setColour(active ? activeColour.brighter(0.08f)
                           : (rowIsSelected
                                  ? trackColour.withAlpha(0.85f)
                                  : (columnId == kColumnVstView && !canToggleVstView
                                         ? juce::Colour::fromRGB(52, 57, 70)
                                         : juce::Colour::fromRGB(78, 86, 102))));
        g.drawRoundedRectangle(chipBounds.toFloat(), 6.0f, 1.2f);
        g.setColour(active
                        ? (columnId == kColumnSolo ? juce::Colour::fromRGB(28, 28, 28) : juce::Colours::white)
                        : (columnId == kColumnVstView && !canToggleVstView
                               ? juce::Colour::fromRGB(112, 120, 134)
                               : juce::Colour::fromRGB(222, 228, 236)));
        g.setFont(juce::FontOptions(13.5f, juce::Font::bold));
        g.drawText(label, chipBounds, juce::Justification::centred, false);

        g.setColour(juce::Colour::fromRGB(45, 50, 62));
        g.fillRect(width - 1, 0, 1, height);
        return;
    }

    if (columnId == kColumnVolume)
    {
        const auto meterLevel = juce::isPositiveAndBelow(rowNumber, static_cast<int>(owner.trackMeterLevels.size()))
            ? owner.trackMeterLevels[static_cast<size_t>(rowNumber)]
            : 0.0f;
        auto meterBounds = juce::Rectangle<int>(0, 0, width, height).reduced(10, 5);
        meterBounds = meterBounds.withSizeKeepingCentre(18, juce::jmax(18, meterBounds.getHeight()));
        paintTrackLevelBar(g, meterBounds, meterLevel, static_cast<float>(track.volume), trackColour);

        g.setColour(juce::Colour::fromRGB(45, 50, 62));
        g.fillRect(width - 1, 0, 1, height);
        return;
    }

    juce::String text;

    switch (columnId)
    {
        case kColumnName: text = track.name; break;
        case kColumnType: text = track.trackType; break;
        case kColumnMode: text = track.instrumentMode; break;
        case kColumnRack: text = displayRackName(owner.documentState.project, track); break;
        case kColumnNotes: text = juce::String(static_cast<int>(track.notes.size())); break;
        case kColumnChannel: text = juce::String(track.midiChannel + 1); break;
        case kColumnPan: text = juce::String(track.pan, 2); break;
        case kColumnFlags: text = makeTrackFlags(track); break;
        default: break;
    }

    auto textColour = rowIsSelected ? trackTextColour(trackColour)
                                    : juce::Colour::fromRGB(227, 233, 241);
    if (!rowIsSelected && columnId == kColumnName)
        textColour = trackColour.brighter(0.18f);

    g.setColour(textColour);
    g.setFont(columnId == kColumnName ? 14.0f : 13.0f);
    g.drawText(text, 8, 0, width - 12, height, juce::Justification::centredLeft, true);

    g.setColour(juce::Colour::fromRGB(45, 50, 62));
    g.fillRect(width - 1, 0, 1, height);
}

void StudioShellComponent::TrackTableModel::selectedRowsChanged(int)
{
    owner.ensureSelectedMidiSectionForTrack(owner.getSelectedTrackIndex());
    owner.refreshInspector();
    owner.updateEditorState();
    owner.scheduleSelectedTrackRackPreviewWarmup();
}

void StudioShellComponent::TrackTableModel::cellClicked(int rowNumber, int columnId, const juce::MouseEvent& event)
{
    if (!juce::isPositiveAndBelow(rowNumber, static_cast<int>(owner.documentState.project.tracks.size())))
        return;

    owner.setSelectedTrackIndex(rowNumber);
    if (!event.mods.isRightButtonDown()
        && (columnId == kColumnMute || columnId == kColumnSolo || columnId == kColumnVstView || columnId == kColumnVolume))
    {
        auto updatedTrack = owner.documentState.project.tracks[static_cast<size_t>(rowNumber)];
        if (columnId == kColumnMute)
        {
            updatedTrack.mute = !updatedTrack.mute;
            owner.applyTrackStateEdit(rowNumber, updatedTrack, "Toggle Mute");
        }
        else
        {
            if (columnId == kColumnSolo)
            {
                updatedTrack.solo = !updatedTrack.solo;
                owner.applyTrackStateEdit(rowNumber, updatedTrack, "Toggle Solo");
            }
            else
            {
                if (columnId == kColumnVolume)
                {
                    if (owner.isMixerWindowVisible())
                        owner.setMixerWindowVisible(false);
                    else
                        owner.focusMixerPanel();
                }
                else
                {
                    const bool canToggleVstView = updatedTrack.trackType != "sample"
                        && updatedTrack.instrumentMode.containsIgnoreCase("VST");
                    if (!canToggleVstView)
                        return;

                    if (auto* session = owner.findRackEditorSession(rowNumber);
                        session != nullptr && session->editorOpen)
                    {
                        const auto closeResult = owner.nativeVstHost.closeAudioEngineTrackEditor(rowNumber);
                        if (closeResult.failed())
                        {
                            juce::AlertWindow::showMessageBoxAsync(juce::AlertWindow::WarningIcon,
                                                                   "Editor Close Failed",
                                                                   closeResult.getErrorMessage());
                        }
                        else
                        {
                            owner.closeRackEditorSession(rowNumber);
                            owner.refreshPollingTimerState();
                            owner.updateEditorState();
                            owner.statusLabel.setText("Hid VST editor for " + updatedTrack.name + ".", juce::dontSendNotification);
                        }
                    }
                    else
                    {
                        owner.openSelectedTrackRackEditor();
                    }

                    owner.trackTable.repaint();
                }
            }
        }
        return;
    }

    if (event.mods.isRightButtonDown())
        owner.showTrackContextMenu(rowNumber, event.getScreenPosition().roundToInt());
}

StudioShellComponent::SampleAssetListModel::SampleAssetListModel(StudioShellComponent& ownerIn)
    : owner(ownerIn)
{
}

int StudioShellComponent::SampleAssetListModel::getNumRows()
{
    return static_cast<int>(owner.documentState.project.sampleAssets.size());
}

void StudioShellComponent::SampleAssetListModel::paintListBoxItem(int rowNumber,
                                                                  juce::Graphics& g,
                                                                  int width,
                                                                  int height,
                                                                  bool rowIsSelected)
{
    juce::ignoreUnused(height);
    if (!juce::isPositiveAndBelow(rowNumber, static_cast<int>(owner.documentState.project.sampleAssets.size())))
        return;

    const auto& asset = owner.documentState.project.sampleAssets[static_cast<size_t>(rowNumber)];
    const auto text = juce::File(asset.path).getFileName()
        + "  (" + juce::String(asset.durationSec, 2) + " s)";

    g.fillAll(rowIsSelected ? juce::Colour::fromRGB(46, 88, 138)
                            : ((rowNumber % 2) == 0 ? juce::Colour::fromRGB(26, 30, 37)
                                                    : juce::Colour::fromRGB(21, 25, 31)));
    g.setColour(rowIsSelected ? juce::Colours::white : juce::Colour::fromRGB(226, 232, 240));
    g.setFont(juce::FontOptions(12.5f));
    g.drawText(text, 8, 0, width - 12, height, juce::Justification::centredLeft, true);
}

void StudioShellComponent::SampleAssetListModel::selectedRowsChanged(int)
{
    owner.updateEditorState();
}

StudioShellComponent::StudioShellComponent()
    : tableModel(*this),
      trackTable("Tracks", &tableModel),
      sampleAssetListModel(*this)
{
    documentState = makeDefaultProjectFile();
    syncBundledRackCatalogInProject();
    activityLogFile = nativeLogsDirectory().getChildFile("native-activity-"
        + juce::Time::getCurrentTime().formatted("%Y%m%d-%H%M%S")
        + ".log");
    windowStateSettings = std::make_unique<juce::PropertiesFile>(nativeWindowSettingsOptions());
    restorePersistedThemeSelection();
    restorePersistedWindowVisibility();
    aiClient.setActivityLogCallback([safeThis = juce::Component::SafePointer<StudioShellComponent>(this)] (const juce::String& title,
                                                                                                         const juce::String& body)
    {
        juce::MessageManager::callAsync([safeThis, title, body]
        {
            if (safeThis == nullptr)
                return;

            safeThis->appendActivityLog(title, body);
        });
    });
    nativeVstHost.setEditorStateCallback([safeThis = juce::Component::SafePointer<StudioShellComponent>(this)] (bool isOpen)
    {
        auto applyState = [safeThis, isOpen]
        {
            if (safeThis == nullptr)
                return;

            safeThis->loadedRackEditorOpen = isOpen;
            safeThis->refreshPollingTimerState();
        };

        if (auto* messageManager = juce::MessageManager::getInstanceWithoutCreating();
            messageManager != nullptr && messageManager->isThisTheMessageThread())
        {
            applyState();
        }
        else
        {
            juce::MessageManager::callAsync([applyState]
            {
                applyState();
            });
        }
    });

    headerLogoImage = loadMutagenLogoBinaryData(false);
    if (headerLogoImage.isValid())
        headerLogo.setImage(headerLogoImage);
    headerLogo.setImagePlacement(juce::RectanglePlacement::centred);
    addAndMakeVisible(headerLogo);
    headerLabel.setVisible(false);

    fileLabel.setColour(juce::Label::textColourId, juce::Colour::fromRGB(165, 176, 191));
    addAndMakeVisible(fileLabel);

    headerTimecodeDisplay = std::make_unique<HeaderLcdDisplay>();
    addAndMakeVisible(*headerTimecodeDisplay);

    statsLabel.setColour(juce::Label::textColourId, juce::Colour::fromRGB(210, 216, 224));
    addAndMakeVisible(statsLabel);

    statusLabel.setColour(juce::Label::textColourId, juce::Colour::fromRGB(143, 225, 170));
    addAndMakeVisible(statusLabel);

    aiStatusSummaryLabel.setColour(juce::Label::textColourId, juce::Colour::fromRGB(156, 199, 239));
    addAndMakeVisible(aiStatusSummaryLabel);

    inspectorLabel.setText("Selected Track", juce::dontSendNotification);
    inspectorLabel.setFont(juce::FontOptions(16.0f, juce::Font::bold));
    inspectorLabel.setColour(juce::Label::textColourId, juce::Colour::fromRGB(235, 239, 244));
    addAndMakeVisible(inspectorLabel);

    configureInspectorTextEditor(trackNameEditor, "Track name");
    trackNameEditor.onFocusLost = [this]
    {
        if (syncingInspectorControls)
            return;
        const auto name = trackNameEditor.getText().trim();
        applySelectedTrackMutation([name] (TrackState& track)
                                   {
                                       track.name = name.isNotEmpty() ? name : "Track";
                                   },
                                   "Rename Track");
    };
    trackNameEditor.onReturnKey = [this] { trackNameEditor.giveAwayKeyboardFocus(); };

    configureInspectorTextEditor(trackTypeEditor, "Track type");
    trackTypeEditor.onFocusLost = [this]
    {
        if (syncingInspectorControls)
            return;
        const auto value = trackTypeEditor.getText().trim();
        applySelectedTrackMutation([value] (TrackState& track)
                                   {
                                       track.trackType = value.isNotEmpty() ? value : "instrument";
                                   },
                                   "Change Track Type");
    };
    trackTypeEditor.onReturnKey = [this] { trackTypeEditor.giveAwayKeyboardFocus(); };

    configureInspectorTextEditor(instrumentModeEditor, "Instrument mode");
    instrumentModeEditor.onFocusLost = [this]
    {
        if (syncingInspectorControls)
            return;
        const auto value = instrumentModeEditor.getText().trim();
        applySelectedTrackMutation([value] (TrackState& track)
                                   {
                                       track.instrumentMode = value.isNotEmpty() ? value : "General MIDI";
                                   },
                                   "Change Instrument Mode");
    };
    instrumentModeEditor.onReturnKey = [this] { instrumentModeEditor.giveAwayKeyboardFocus(); };

    configureInspectorTextEditor(instrumentEditor, "Instrument name");
    instrumentEditor.onFocusLost = [this]
    {
        if (syncingInspectorControls)
            return;
        const auto value = instrumentEditor.getText().trim();
        applySelectedTrackMutation([value] (TrackState& track)
                                   {
                                       track.instrument = value.isNotEmpty() ? value : "Piano";
                                   },
                                   "Change Instrument");
    };
    instrumentEditor.onReturnKey = [this] { instrumentEditor.giveAwayKeyboardFocus(); };

    configureInspectorTextEditor(rackVstEditor, "Rack VST path or name");
    rackVstEditor.onFocusLost = [this]
    {
        if (syncingInspectorControls)
            return;
        const auto value = rackVstEditor.getText().trim();
        applySelectedTrackMutation([value] (TrackState& track)
                                   {
                                       track.rackVst = value;
                                   },
                                   "Change Rack VST");
    };
    rackVstEditor.onReturnKey = [this] { rackVstEditor.giveAwayKeyboardFocus(); };

    configureInspectorSlider(midiChannelSlider, 1.0, 16.0, 1.0, " ch");
    midiChannelSlider.onValueChange = [this]
    {
        if (syncingInspectorControls)
            return;
        applySelectedTrackMutation([value = juce::roundToInt(midiChannelSlider.getValue())] (TrackState& track)
                                   {
                                       track.midiChannel = juce::jlimit(0, 15, value - 1);
                                   },
                                   "Change MIDI Channel");
    };

    configureInspectorSlider(midiProgramSlider, 0.0, 127.0, 1.0, " pgm");
    midiProgramSlider.onValueChange = [this]
    {
        if (syncingInspectorControls)
            return;
        applySelectedTrackMutation([value = juce::roundToInt(midiProgramSlider.getValue())] (TrackState& track)
                                   {
                                       track.midiProgram = juce::jlimit(0, 127, value);
                                   },
                                   "Change MIDI Program");
    };

    configureInspectorSlider(volumeSlider, 0.0, 1.0, 0.01, "");
    volumeSlider.onValueChange = [this]
    {
        if (syncingInspectorControls)
            return;
        applySelectedTrackMutation([value = volumeSlider.getValue()] (TrackState& track)
                                   {
                                       track.volume = juce::jlimit(0.0, 1.0, value);
                                   },
                                   "Change Track Volume");
    };

    configureInspectorSlider(panSlider, -1.0, 1.0, 0.01, "");
    panSlider.onValueChange = [this]
    {
        if (syncingInspectorControls)
            return;
        applySelectedTrackMutation([value = panSlider.getValue()] (TrackState& track)
                                   {
                                       track.pan = juce::jlimit(-1.0, 1.0, value);
                                   },
                                   "Change Track Pan");
    };

    muteToggle.setButtonText("Mute");
    muteToggle.onClick = [this]
    {
        if (syncingInspectorControls)
            return;
        applySelectedTrackMutation([value = muteToggle.getToggleState()] (TrackState& track)
                                   {
                                       track.mute = value;
                                   },
                                   "Toggle Mute");
    };
    addAndMakeVisible(muteToggle);

    soloToggle.setButtonText("Solo");
    soloToggle.onClick = [this]
    {
        if (syncingInspectorControls)
            return;
        applySelectedTrackMutation([value = soloToggle.getToggleState()] (TrackState& track)
                                   {
                                       track.solo = value;
                                   },
                                   "Toggle Solo");
    };
    addAndMakeVisible(soloToggle);

    liveArmToggle.setButtonText("Arm");
    liveArmToggle.onClick = [this]
    {
        if (syncingInspectorControls)
            return;
        applySelectedTrackMutation([value = liveArmToggle.getToggleState()] (TrackState& track)
                                   {
                                       track.liveArmed = value;
                                   },
                                   "Toggle Record Arm");
    };
    addAndMakeVisible(liveArmToggle);

    inspectorEditor.setMultiLine(true);
    inspectorEditor.setReadOnly(true);
    inspectorEditor.setScrollbarsShown(true);
    inspectorEditor.setColour(juce::TextEditor::backgroundColourId, juce::Colour::fromRGB(18, 20, 25));
    inspectorEditor.setColour(juce::TextEditor::textColourId, juce::Colour::fromRGB(226, 230, 237));
    inspectorEditor.setColour(juce::TextEditor::outlineColourId, juce::Colour::fromRGB(56, 64, 79));
    inspectorEditor.setFont(juce::FontOptions(13.0f));
    addAndMakeVisible(inspectorEditor);

    mixerLabel.setText("Mixer", juce::dontSendNotification);
    mixerLabel.setFont(juce::FontOptions(16.0f, juce::Font::bold));
    mixerLabel.setColour(juce::Label::textColourId, juce::Colour::fromRGB(235, 239, 244));
    addAndMakeVisible(mixerLabel);

    mixerComponent = std::make_unique<MixerComponent>(
        [this] () -> const ProjectState& { return documentState.project; },
        [this] (int trackIndex, const TrackState& track, bool undoable, const juce::String& actionName)
        {
            if (trackIndex < 0)
                return;

            if (undoable)
                applyTrackStateEdit(trackIndex, track, actionName);
            else
                replaceTrackStateNoUndo(trackIndex, track);
        },
        [this] (int trackIndex) -> float
        {
            if (!juce::isPositiveAndBelow(trackIndex, static_cast<int>(trackMeterLevels.size())))
                return 0.0f;
            return trackMeterLevels[static_cast<size_t>(trackIndex)];
        });
    mixerViewport.setViewedComponent(mixerComponent.get(), false);
    mixerViewport.setScrollBarsShown(true, false);
    addAndMakeVisible(mixerViewport);

    samplesLabel.setText("Samples", juce::dontSendNotification);
    samplesLabel.setFont(juce::FontOptions(16.0f, juce::Font::bold));
    samplesLabel.setColour(juce::Label::textColourId, juce::Colour::fromRGB(235, 239, 244));
    addAndMakeVisible(samplesLabel);

    sampleLibraryLabel.setText("Library", juce::dontSendNotification);
    sampleLibraryLabel.setColour(juce::Label::textColourId, juce::Colour::fromRGB(190, 199, 210));
    addAndMakeVisible(sampleLibraryLabel);

    sampleTimelineLabel.setText("Timeline", juce::dontSendNotification);
    sampleTimelineLabel.setColour(juce::Label::textColourId, juce::Colour::fromRGB(190, 199, 210));
    addAndMakeVisible(sampleTimelineLabel);

    sampleAssetList.setModel(&sampleAssetListModel);
    sampleAssetList.setRowHeight(28);
    sampleAssetList.setColour(juce::ListBox::backgroundColourId, juce::Colour::fromRGB(20, 22, 28));
    sampleAssetList.setOutlineThickness(1);
    addAndMakeVisible(sampleAssetList);

    sampleTimeline = std::make_unique<SampleTimelineComponent>(
        [this] () -> const ProjectState& { return documentState.project; },
        [this] (const ProjectState& project, bool undoable, const juce::String& actionName)
        {
            if (undoable)
                applyProjectStateEdit(project, actionName);
            else
                setProjectStateInternal(project);
        });
    sampleTimelineViewport.setViewedComponent(sampleTimeline.get(), false);
    sampleTimelineViewport.setScrollBarsShown(true, true);
    addAndMakeVisible(sampleTimelineViewport);

    automationLabel.setText("Automation", juce::dontSendNotification);
    automationLabel.setFont(juce::FontOptions(16.0f, juce::Font::bold));
    automationLabel.setColour(juce::Label::textColourId, juce::Colour::fromRGB(235, 239, 244));
    addAndMakeVisible(automationLabel);

    automationEditor = std::make_unique<AutomationEditorComponent>(
        [this] () -> const ProjectState& { return documentState.project; },
        [this] () -> int { return getSelectedTrackIndex(); },
        [this] (int trackIndex, const TrackState& track, bool undoable, const juce::String& actionName)
        {
            if (trackIndex < 0)
                return;

            if (undoable)
                applyTrackStateEdit(trackIndex, track, actionName);
            else
                replaceTrackStateNoUndo(trackIndex, track);
        });
    addAndMakeVisible(*automationEditor);

    pianoRollLabel.setText("Piano Roll  |  Right-click for tools", juce::dontSendNotification);
    pianoRollLabel.setFont(juce::FontOptions(16.0f, juce::Font::bold));
    pianoRollLabel.setColour(juce::Label::textColourId, juce::Colour::fromRGB(235, 239, 244));
    addAndMakeVisible(pianoRollLabel);
    pianoRollLabel.setVisible(false);

    arrangementLabel.setText("Sequencer  |  Right-click for tools", juce::dontSendNotification);
    arrangementLabel.setFont(juce::FontOptions(16.0f, juce::Font::bold));
    arrangementLabel.setColour(juce::Label::textColourId, juce::Colour::fromRGB(235, 239, 244));
    addAndMakeVisible(arrangementLabel);

    arrangementOverview = std::make_unique<ArrangementOverviewComponent>(
        [this] () -> const ProjectState& { return documentState.project; },
        [this] (const ProjectState& project, bool undoable, const juce::String& actionName)
        {
            if (undoable)
                applyProjectStateEdit(project, actionName);
            else
                setProjectStateInternal(project);
        },
        [this] () { return getSelectedMidiSectionIndex(); },
        [this] (int sectionIndex, bool focusEditor)
        {
            setSelectedMidiSectionIndex(sectionIndex, true);
            if (focusEditor)
                focusMidiSectionInPianoRoll(sectionIndex);
        },
        [this] () { return editorToolMode; },
        [this] (EditorToolMode mode)
        {
            setEditorToolMode(mode);
        });
    arrangementViewport.setViewedComponent(arrangementOverview.get(), false);
    arrangementViewport.setScrollBarsShown(true, true);
    addAndMakeVisible(arrangementViewport);

    pianoRoll = std::make_unique<PianoRollComponent>([this] () -> const ProjectState& { return documentState.project; },
                                                     [this] () { return getSelectedTrackIndex(); },
                                                     [this] () { return getSelectedMidiSectionIndex(); },
                                                     [this] (const ProjectState& project, bool undoable, const juce::String& actionName)
                                                     {
                                                         if (undoable)
                                                             applyProjectStateEdit(project, actionName);
                                                         else
                                                             setProjectStateInternal(project);
                                                     });
    pianoRoll->setNotePreviewCallbacks([this] (int pitch, int velocity)
                                       {
                                           previewSelectedTrackMidiNoteOn(pitch, velocity);
                                       },
                                       [this] (int pitch, int velocity)
                                       {
                                           previewSelectedTrackMidiNoteOff(pitch, velocity);
                                       },
                                       [this]
                                       {
                                           stopSelectedTrackMidiPreview();
                                       });
    pianoRoll->setToolModeChangeCallback([this] (EditorToolMode mode)
                                         {
                                             setEditorToolMode(mode);
                                         });
    pianoRollViewport.setViewedComponent(pianoRoll.get(), false);
    pianoRollViewport.setScrollBarsShown(true, true);
    addAndMakeVisible(pianoRollViewport);
    pianoRollViewport.setVisible(false);

    createToolbarButton(newButton, "New");
    createToolbarButton(openButton, "Open");
    createToolbarButton(saveButton, "Save");
    createToolbarButton(saveAsButton, "Save As");
    createToolbarButton(importMidiButton, "Import MIDI");
    createToolbarButton(exportWavButton, "Export WAV");
    createToolbarButton(exportMidiButton, "Export MIDI");
    createToolbarButton(importSampleButton, "Import Sample");
    createToolbarButton(placeSampleButton, "Place At Playhead");
    createToolbarButton(aiSettingsButton, "AI Settings");
    createToolbarButton(aiComposeButton, "AI Compose");
    createToolbarButton(playProjectButton, "Play Project");
    createToolbarButton(playTrackButton, "Play Track");
    createToolbarButton(stopTrackButton, "Stop");
    createToolbarButton(undoButton, "Undo");
    createToolbarButton(redoButton, "Redo");

    newButton.onClick = [this] { createNewProject(); };
    openButton.onClick = [this] { promptOpenProject(); };
    saveButton.onClick = [this] { saveProject(); };
    saveAsButton.onClick = [this] { saveProjectAs(); };
    importMidiButton.onClick = [this] { promptImportMidi(); };
    exportWavButton.onClick = [this] { promptExportWav(); };
    exportMidiButton.onClick = [this] { promptExportMidi(); };
    importSampleButton.onClick = [this] { promptImportSample(); };
    placeSampleButton.onClick = [this] { placeSelectedSampleAtPlayhead(); };
    aiSettingsButton.onClick = [this] { showAiSettingsDialog(); };
    aiComposeButton.onClick = [this] { composeWithAi(); };
    playProjectButton.onClick = [this] { playFullProjectThroughNativeEngine(); };
    playTrackButton.onClick = [this] { playSelectedTrackThroughRack(); };
    stopTrackButton.onClick = [this] { stopRackPreview(); };
    undoButton.onClick = [this] { undo(); };
    redoButton.onClick = [this] { redo(); };

    tempoLabel.setText("Tempo", juce::dontSendNotification);
    tempoLabel.setColour(juce::Label::textColourId, juce::Colour::fromRGB(214, 219, 227));
    addAndMakeVisible(tempoLabel);

    tempoSlider.setRange(20.0, 300.0, 1.0);
    tempoSlider.setTextBoxStyle(juce::Slider::TextBoxRight, false, 72, 22);
    tempoSlider.setValue(documentState.project.bpm, juce::dontSendNotification);
    tempoSlider.onValueChange = [this] { handleTempoChanged(); };
    addAndMakeVisible(tempoSlider);

    patternBarsLabel.setText("Pattern Size", juce::dontSendNotification);
    patternBarsLabel.setColour(juce::Label::textColourId, juce::Colour::fromRGB(214, 219, 227));
    addAndMakeVisible(patternBarsLabel);

    for (const auto& option : sequenceTickOptions())
        patternBarsBox.addItem(option.label, option.ticks);
    patternBarsBox.setTextWhenNothingSelected("Pattern size");
    patternBarsBox.onChange = [this] { handlePatternBarsChanged(); };
    addAndMakeVisible(patternBarsBox);

    keyQuantizeLabel.setText("Key Quantize", juce::dontSendNotification);
    keyQuantizeLabel.setColour(juce::Label::textColourId, juce::Colour::fromRGB(214, 219, 227));
    addAndMakeVisible(keyQuantizeLabel);

    for (const auto& option : keyQuantizeOptions())
        keyQuantizeBox.addItem(option.label, option.id);
    keyQuantizeBox.setTextWhenNothingSelected("All Notes");
    keyQuantizeBox.onChange = [this] { handleKeyQuantizeChanged(); };
    addAndMakeVisible(keyQuantizeBox);

    arrangementSnapLabel.setText("Seq Snap", juce::dontSendNotification);
    arrangementSnapLabel.setColour(juce::Label::textColourId, juce::Colour::fromRGB(214, 219, 227));
    addAndMakeVisible(arrangementSnapLabel);

    for (const auto& option : sequenceTickOptions())
        arrangementSnapBox.addItem(option.label, option.ticks);
    arrangementSnapBox.setTextWhenNothingSelected("Sequencer snap");
    arrangementSnapBox.onChange = [this] { handleArrangementSnapChanged(); };
    addAndMakeVisible(arrangementSnapBox);

    auto configureViewSlider = [this] (juce::Slider& slider,
                                       double minimum,
                                       double maximum,
                                       double step,
                                       const juce::String& suffix,
                                       double value,
                                       std::function<void()> onChange)
    {
        slider.setSliderStyle(juce::Slider::LinearHorizontal);
        slider.setRange(minimum, maximum, step);
        slider.setChangeNotificationOnlyOnRelease(false);
        slider.setTextBoxStyle(juce::Slider::TextBoxRight, false, 52, 22);
        slider.setTextValueSuffix(suffix);
        slider.setValue(value, juce::dontSendNotification);
        slider.onValueChange = std::move(onChange);
        addAndMakeVisible(slider);
    };

    arrangementZoomLabel.setText("Seq Zoom", juce::dontSendNotification);
    arrangementZoomLabel.setColour(juce::Label::textColourId, juce::Colour::fromRGB(214, 219, 227));
    addAndMakeVisible(arrangementZoomLabel);
    configureViewSlider(arrangementZoomSlider,
                        48.0,
                        320.0,
                        4.0,
                        " px/bar",
                        arrangementZoomPixelsPerBar,
                        [this] { handleArrangementZoomChanged(); });

    arrangementLaneHeightLabel.setText("Seq Lane", juce::dontSendNotification);
    arrangementLaneHeightLabel.setColour(juce::Label::textColourId, juce::Colour::fromRGB(214, 219, 227));
    addAndMakeVisible(arrangementLaneHeightLabel);
    configureViewSlider(arrangementLaneHeightSlider,
                        22.0,
                        80.0,
                        1.0,
                        " px",
                        arrangementLaneHeightPixels,
                        [this] { handleArrangementLaneHeightChanged(); });

    pianoRollZoomLabel.setVisible(false);
    pianoRollZoomSlider.setVisible(false);
    pianoRollRowHeightLabel.setVisible(false);
    pianoRollRowHeightSlider.setVisible(false);

    loopToggle.setButtonText("Loop");
    loopToggle.setToggleState(documentState.project.loopEnabled, juce::dontSendNotification);
    loopToggle.onClick = [this] { setTransportLoopEnabled(loopToggle.getToggleState()); };
    addAndMakeVisible(loopToggle);

    metronomeToggle.setButtonText("Metronome");
    metronomeToggle.setToggleState(documentState.project.metronomeEnabled, juce::dontSendNotification);
    metronomeToggle.onClick = [this] { setTransportMetronomeEnabled(metronomeToggle.getToggleState()); };
    addAndMakeVisible(metronomeToggle);

    trackTable.getHeader().addColumn("M", kColumnMute, 42, 36, 56);
    trackTable.getHeader().addColumn("S", kColumnSolo, 42, 36, 56);
    trackTable.getHeader().addColumn("V", kColumnVstView, 42, 36, 56);
    trackTable.getHeader().addColumn("Track", kColumnName, 180, 80, 320);
    trackTable.getHeader().addColumn("Vol", kColumnVolume, 52, 42, 72);
    trackTable.getHeader().addColumn("Type", kColumnType, 90, 60, 140);
    trackTable.getHeader().addColumn("Mode", kColumnMode, 140, 100, 220);
    trackTable.getHeader().addColumn("Rack / Instrument", kColumnRack, 220, 140, 340);
    trackTable.getHeader().addColumn("Notes", kColumnNotes, 70, 50, 100);
    trackTable.getHeader().addColumn("Ch", kColumnChannel, 55, 40, 70);
    trackTable.getHeader().addColumn("Pan", kColumnPan, 70, 50, 90);
    trackTable.getHeader().addColumn("Arm", kColumnFlags, 72, 56, 120);
    trackTable.setColour(juce::ListBox::backgroundColourId, juce::Colour::fromRGB(20, 22, 28));
    trackTable.setOutlineThickness(1);
    addAndMakeVisible(trackTable);

    setupFloatingWindows();
    applyEditorViewScaleState();
    setWantsKeyboardFocus(true);
    refreshPollingTimerState();
    refreshUi();
    applyTheme();
    updateAiStatusSummary();
    trackTable.selectRow(0);
    pianoRoll->grabKeyboardFocus();
    appendActivityLog("Log File", "Activity session log\n" + activityLogFile.getFullPathName());
    appendActivityLog("App", "Mutagen shell started.");
}

StudioShellComponent::~StudioShellComponent()
{
    closeAllRackEditorSessions();
    nativeVstHost.setEditorStateCallback({});
    appRuntimeProfiler().dump("native_app_profile");
}

void StudioShellComponent::restorePersistedThemeSelection()
{
    if (windowStateSettings == nullptr)
        return;

    currentThemeIndex = juce::jlimit(0,
                                     static_cast<int>(availableThemeSpecs().size()) - 1,
                                     windowStateSettings->getIntValue("ui_theme", currentThemeIndex));

    if (auto* lookAndFeel = dynamic_cast<juce::LookAndFeel_V4*>(&juce::LookAndFeel::getDefaultLookAndFeel()))
        lookAndFeel->setColourScheme(themeSpecForIndex(currentThemeIndex).scheme);
}

void StudioShellComponent::persistThemeSelection() const
{
    if (windowStateSettings == nullptr)
        return;

    windowStateSettings->setValue("ui_theme", currentThemeIndex);
    windowStateSettings->saveIfNeeded();
}

int StudioShellComponent::getCurrentThemeIndex() const noexcept
{
    return currentThemeIndex;
}

int StudioShellComponent::getThemeCount() const noexcept
{
    return static_cast<int>(availableThemeSpecs().size());
}

juce::String StudioShellComponent::getThemeName(int index) const
{
    return themeSpecForIndex(index).name;
}

void StudioShellComponent::setThemeIndex(int index)
{
    const auto clamped = juce::jlimit(0, getThemeCount() - 1, index);
    if (currentThemeIndex == clamped)
        return;

    currentThemeIndex = clamped;
    applyTheme();
    appendActivityLog("Theme", "Switched to " + getThemeName(currentThemeIndex) + ".");
}

void StudioShellComponent::applyTheme()
{
    const auto& theme = themeSpecForIndex(currentThemeIndex);

    if (auto* lookAndFeel = dynamic_cast<juce::LookAndFeel_V4*>(&juce::LookAndFeel::getDefaultLookAndFeel()))
        lookAndFeel->setColourScheme(theme.scheme);

    applyThemeToComponentTree(*this, theme);

    if (headerTimecodeDisplay != nullptr)
        headerTimecodeDisplay->setTheme(theme);

    fileLabel.setColour(juce::Label::textColourId, theme.secondaryText);
    statsLabel.setColour(juce::Label::textColourId, theme.primaryText.withAlpha(0.92f));
    statusLabel.setColour(juce::Label::textColourId, theme.successText);
    inspectorLabel.setColour(juce::Label::textColourId, theme.primaryText);
    mixerLabel.setColour(juce::Label::textColourId, theme.primaryText);
    samplesLabel.setColour(juce::Label::textColourId, theme.primaryText);
    sampleLibraryLabel.setColour(juce::Label::textColourId, theme.secondaryText);
    sampleTimelineLabel.setColour(juce::Label::textColourId, theme.secondaryText);
    automationLabel.setColour(juce::Label::textColourId, theme.primaryText);
    pianoRollLabel.setColour(juce::Label::textColourId, theme.primaryText);
    arrangementLabel.setColour(juce::Label::textColourId, theme.primaryText);
    tempoLabel.setColour(juce::Label::textColourId, theme.primaryText);
    patternBarsLabel.setColour(juce::Label::textColourId, theme.primaryText);
    keyQuantizeLabel.setColour(juce::Label::textColourId, theme.primaryText);
    arrangementSnapLabel.setColour(juce::Label::textColourId, theme.primaryText);
    arrangementZoomLabel.setColour(juce::Label::textColourId, theme.primaryText);
    arrangementLaneHeightLabel.setColour(juce::Label::textColourId, theme.primaryText);

    trackTable.setColour(juce::ListBox::backgroundColourId, theme.surface);
    trackTable.setColour(juce::ListBox::outlineColourId, theme.outline);
    trackTable.getHeader().setColour(juce::TableHeaderComponent::backgroundColourId, theme.surfaceAlt);
    trackTable.getHeader().setColour(juce::TableHeaderComponent::outlineColourId, theme.outline);
    trackTable.getHeader().setColour(juce::TableHeaderComponent::textColourId, theme.primaryText);
    sampleAssetList.setColour(juce::ListBox::backgroundColourId, theme.surface);
    sampleAssetList.setColour(juce::ListBox::outlineColourId, theme.outline);
    inspectorEditor.setTextToShowWhenEmpty("Track details", theme.editorPlaceholder);

    const auto styleFloatingWindow = [&theme] (FloatingPanelWindow* window)
    {
        if (window == nullptr)
            return;

        window->setColour(juce::ResizableWindow::backgroundColourId, theme.mainBackground);
        if (auto* content = window->getContentComponent())
            applyThemeToComponentTree(*content, theme);
        window->repaint();
    };

    styleFloatingWindow(transportWindow.get());
    styleFloatingWindow(mixerWindow.get());
    styleFloatingWindow(audioWindow.get());
    styleFloatingWindow(panelsWindow.get());
    styleFloatingWindow(tracksWindow.get());
    styleFloatingWindow(rackBrowserWindow.get());
    styleFloatingWindow(renderManagerWindow.get());
    styleFloatingWindow(arrangementWindow.get());
    styleFloatingWindow(automationWindow.get());
    styleFloatingWindow(samplesWindow.get());
    styleFloatingWindow(pianoRollWindow.get());
    styleFloatingWindow(virtualPianoWindow.get());
    styleFloatingWindow(activityLogWindow.get());
    styleFloatingWindow(audioSettingsWindow.get());
    styleFloatingWindow(vstFolderManagerWindow.get());

    updateAiStatusSummary();
    repaint();
    if (auto* topLevel = getTopLevelComponent())
        topLevel->repaint();

    persistThemeSelection();
}

void StudioShellComponent::createToolbarButton(juce::TextButton& button, const juce::String& text)
{
    const auto& theme = themeSpecForIndex(currentThemeIndex);
    button.setButtonText(text);
    button.setColour(juce::TextButton::buttonColourId, theme.buttonOff);
    button.setColour(juce::TextButton::buttonOnColourId, theme.buttonOn);
    button.setColour(juce::TextButton::textColourOffId, theme.buttonText);
    button.setColour(juce::TextButton::textColourOnId, theme.buttonText);
    addAndMakeVisible(button);
}

void StudioShellComponent::configureInspectorTextEditor(juce::TextEditor& editor, const juce::String& placeholder)
{
    const auto& theme = themeSpecForIndex(currentThemeIndex);
    editor.setMultiLine(false);
    editor.setScrollbarsShown(false);
    editor.setTextToShowWhenEmpty(placeholder, theme.editorPlaceholder);
    editor.setColour(juce::TextEditor::backgroundColourId, theme.surface);
    editor.setColour(juce::TextEditor::textColourId, theme.primaryText);
    editor.setColour(juce::TextEditor::outlineColourId, theme.outline);
    editor.setColour(juce::CaretComponent::caretColourId, theme.primaryText);
    editor.setFont(juce::FontOptions(13.0f));
    addAndMakeVisible(editor);
}

void StudioShellComponent::configureInspectorSlider(juce::Slider& slider,
                                                    double minimum,
                                                    double maximum,
                                                    double step,
                                                    const juce::String& suffix)
{
    slider.setSliderStyle(juce::Slider::LinearHorizontal);
    slider.setRange(minimum, maximum, step);
    slider.setChangeNotificationOnlyOnRelease(true);
    slider.setTextBoxStyle(juce::Slider::TextBoxRight, false, 60, 22);
    slider.setTextValueSuffix(suffix);
    addAndMakeVisible(slider);
}

void StudioShellComponent::paint(juce::Graphics& g)
{
    const auto& theme = themeSpecForIndex(currentThemeIndex);
    g.fillAll(theme.mainBackground);

    juce::ColourGradient background(theme.headerStart,
                                    0.0f,
                                    0.0f,
                                    theme.headerEnd,
                                    0.0f,
                                    460.0f,
                                    false);
    background.addColour(0.45, theme.headerMid);
    g.setGradientFill(background);
    g.fillRect(getLocalBounds());
}

void StudioShellComponent::resized()
{
    auto bounds = getLocalBounds();
    auto headerArea = bounds.removeFromTop(124);
    headerLogo.setBounds(4, 4, 120, 120);

    auto headerContent = headerArea.reduced(18, 4);
    headerContent.removeFromLeft(114);
    headerContent.removeFromLeft(8);

    auto lcdArea = headerContent.removeFromRight(juce::jlimit(280, 360, headerContent.getWidth() / 3));
    headerContent.removeFromRight(12);
    if (headerTimecodeDisplay != nullptr)
        headerTimecodeDisplay->setBounds(lcdArea.reduced(0, 4));

    auto fileArea = headerContent.removeFromTop(24);
    fileLabel.setBounds(fileArea);

    headerContent.removeFromTop(6);
    auto toolbar1 = headerContent.removeFromTop(34);
    playProjectButton.setBounds(toolbar1.removeFromLeft(116).reduced(0, 2));
    toolbar1.removeFromLeft(6);
    playTrackButton.setBounds(toolbar1.removeFromLeft(98).reduced(0, 2));
    toolbar1.removeFromLeft(6);
    stopTrackButton.setBounds(toolbar1.removeFromLeft(76).reduced(0, 2));

    headerContent.removeFromTop(4);
    auto toolbar2 = headerContent.removeFromTop(34);
    aiComposeButton.setBounds(toolbar2.removeFromLeft(122).reduced(0, 2));
    toolbar2.removeFromLeft(18);
    tempoLabel.setBounds(toolbar2.removeFromLeft(54));
    tempoSlider.setBounds(toolbar2.removeFromLeft(180));
    toolbar2.removeFromLeft(10);
    patternBarsLabel.setBounds(toolbar2.removeFromLeft(86));
    patternBarsBox.setBounds(toolbar2.removeFromLeft(118).reduced(0, 2));
    toolbar2.removeFromLeft(10);
    keyQuantizeLabel.setBounds(toolbar2.removeFromLeft(92));
    keyQuantizeBox.setBounds(toolbar2.removeFromLeft(192).reduced(0, 2));
    toolbar2.removeFromLeft(10);
    arrangementSnapLabel.setBounds(toolbar2.removeFromLeft(68));
    arrangementSnapBox.setBounds(toolbar2.removeFromLeft(118).reduced(0, 2));
    toolbar2.removeFromLeft(10);
    loopToggle.setBounds(toolbar2.removeFromLeft(74));
    toolbar2.removeFromLeft(8);
    metronomeToggle.setBounds(toolbar2.removeFromLeft(110));
    toolbar2.removeFromLeft(18);
    aiStatusSummaryLabel.setBounds(toolbar2);

    newButton.setBounds({});
    openButton.setBounds({});
    saveButton.setBounds({});
    saveAsButton.setBounds({});
    importMidiButton.setBounds({});
    exportWavButton.setBounds({});
    exportMidiButton.setBounds({});
    aiSettingsButton.setBounds({});
    undoButton.setBounds({});
    redoButton.setBounds({});

    auto area = bounds.reduced(18, 0);
    area.removeFromTop(4);
    area.removeFromTop(4);
    auto toolbar3 = area.removeFromTop(28);
    arrangementZoomLabel.setBounds(toolbar3.removeFromLeft(66));
    arrangementZoomSlider.setBounds(toolbar3.removeFromLeft(164));
    toolbar3.removeFromLeft(10);
    arrangementLaneHeightLabel.setBounds(toolbar3.removeFromLeft(64));
    arrangementLaneHeightSlider.setBounds(toolbar3.removeFromLeft(140));
    pianoRollZoomLabel.setBounds({});
    pianoRollZoomSlider.setBounds({});
    pianoRollRowHeightLabel.setBounds({});
    pianoRollRowHeightSlider.setBounds({});

    area.removeFromTop(4);
    statsLabel.setBounds(area.removeFromTop(24));
    area.removeFromTop(4);
    statusLabel.setBounds(area.removeFromTop(24));
    area.removeFromTop(12);

    auto workspaceArea = area;
    auto leftColumn = workspaceArea.removeFromLeft(juce::jlimit(360,
                                                                520,
                                                                juce::roundToInt(static_cast<float>(workspaceArea.getWidth()) * 0.36f)));
    workspaceArea.removeFromLeft(14);
    auto sequenceArea = workspaceArea;

    const auto leftHeight = leftColumn.getHeight();
    trackTable.setBounds(leftColumn.removeFromTop(juce::jlimit(240,
                                                               juce::jmax(260, leftHeight - 280),
                                                               juce::roundToInt(static_cast<float>(leftHeight) * 0.56f))));
    leftColumn.removeFromTop(10);
    inspectorLabel.setBounds(leftColumn.removeFromTop(24));
    leftColumn.removeFromTop(6);

    trackNameEditor.setBounds(leftColumn.removeFromTop(24));
    leftColumn.removeFromTop(6);

    auto typeRow = leftColumn.removeFromTop(24);
    auto typeWidth = juce::roundToInt(static_cast<float>(typeRow.getWidth()) * 0.44f);
    trackTypeEditor.setBounds(typeRow.removeFromLeft(typeWidth));
    typeRow.removeFromLeft(6);
    instrumentModeEditor.setBounds(typeRow);
    leftColumn.removeFromTop(6);

    instrumentEditor.setBounds(leftColumn.removeFromTop(24));
    leftColumn.removeFromTop(6);
    rackVstEditor.setBounds(leftColumn.removeFromTop(24));
    leftColumn.removeFromTop(6);

    auto midiRow = leftColumn.removeFromTop(24);
    auto halfWidth = juce::roundToInt(static_cast<float>(midiRow.getWidth()) * 0.5f) - 3;
    midiChannelSlider.setBounds(midiRow.removeFromLeft(halfWidth));
    midiRow.removeFromLeft(6);
    midiProgramSlider.setBounds(midiRow);
    leftColumn.removeFromTop(6);

    auto mixRow = leftColumn.removeFromTop(24);
    volumeSlider.setBounds(mixRow.removeFromLeft(halfWidth));
    mixRow.removeFromLeft(6);
    panSlider.setBounds(mixRow);
    leftColumn.removeFromTop(6);

    auto toggleRow = leftColumn.removeFromTop(24);
    muteToggle.setBounds(toggleRow.removeFromLeft(64));
    toggleRow.removeFromLeft(8);
    soloToggle.setBounds(toggleRow.removeFromLeft(64));
    toggleRow.removeFromLeft(8);
    liveArmToggle.setBounds(toggleRow.removeFromLeft(72));
    leftColumn.removeFromTop(8);
    inspectorEditor.setBounds(leftColumn);

    arrangementLabel.setBounds(sequenceArea.removeFromTop(24));
    sequenceArea.removeFromTop(6);
    arrangementViewport.setBounds(sequenceArea);

    pianoRollLabel.setBounds({});
    pianoRollViewport.setBounds({});
    mixerLabel.setBounds({});
    mixerViewport.setBounds({});
    samplesLabel.setBounds({});
    sampleLibraryLabel.setBounds({});
    sampleTimelineLabel.setBounds({});
    sampleAssetList.setBounds({});
    sampleTimelineViewport.setBounds({});
    importSampleButton.setBounds({});
    placeSampleButton.setBounds({});
    automationLabel.setBounds({});
    if (automationEditor != nullptr)
        automationEditor->setBounds({});
}

bool StudioShellComponent::keyPressed(const juce::KeyPress& key)
{
    if (key.getKeyCode() == juce::KeyPress::F2Key)
    {
        setTransportWindowVisible(!isTransportWindowVisible());
        return true;
    }

    if (key.getKeyCode() == juce::KeyPress::spaceKey && !key.getModifiers().isAnyModifierKeyDown())
    {
        if (rackPreviewRunning || projectPreviewRunning)
            stopRackPreview();
        else
            playFullProjectThroughNativeEngine();
        return true;
    }

    if (tryHandleVirtualPianoShortcut(key))
        return true;

    if (key.getModifiers().isCommandDown() && (key.getTextCharacter() == 'a' || key.getTextCharacter() == 'A'))
    {
        if (pianoRoll != nullptr && pianoRoll->selectAllNotes())
            return true;
    }

    if (key.getModifiers().isCommandDown() && (key.getTextCharacter() == 'c' || key.getTextCharacter() == 'C'))
    {
        if (pianoRoll != nullptr && pianoRoll->copySelected())
            return true;
    }

    if (key.getModifiers().isCommandDown() && (key.getTextCharacter() == 'x' || key.getTextCharacter() == 'X'))
    {
        if (pianoRoll != nullptr && pianoRoll->cutSelected())
            return true;
    }

    if (key.getModifiers().isCommandDown() && (key.getTextCharacter() == 'v' || key.getTextCharacter() == 'V'))
    {
        if (pianoRoll != nullptr && pianoRoll->pasteClipboard())
            return true;
    }

    if (key.getModifiers().isCommandDown() && (key.getTextCharacter() == 'n' || key.getTextCharacter() == 'N'))
    {
        createNewProject();
        return true;
    }

    if (key.getModifiers().isCommandDown() && key.getModifiers().isShiftDown()
        && (key.getTextCharacter() == 'o' || key.getTextCharacter() == 'O'))
    {
        promptImportMidi();
        return true;
    }

    if (key.getModifiers().isCommandDown() && (key.getTextCharacter() == 'o' || key.getTextCharacter() == 'O'))
    {
        promptOpenProject();
        return true;
    }

    if (key.getModifiers().isCommandDown() && key.getModifiers().isShiftDown()
        && (key.getTextCharacter() == 's' || key.getTextCharacter() == 'S'))
    {
        saveProjectAs();
        return true;
    }

    if (key.getModifiers().isCommandDown() && (key.getTextCharacter() == 's' || key.getTextCharacter() == 'S'))
    {
        saveProject();
        return true;
    }

    if (key.getModifiers().isCommandDown() && (key.getTextCharacter() == 'g' || key.getTextCharacter() == 'G'))
    {
        composeWithAi();
        return true;
    }

    if (key.getModifiers().isCommandDown() && key.getKeyCode() == ',')
    {
        showAiSettingsDialog();
        return true;
    }

    if (key.getModifiers().isCommandDown() && (key.getTextCharacter() == 'e' || key.getTextCharacter() == 'E'))
    {
        if (key.getModifiers().isShiftDown())
            promptExportMidi();
        else
            promptExportWav();
        return true;
    }

    if (key.getModifiers().isCommandDown()
        && (key.getTextCharacter() == 'z' || key.getTextCharacter() == 'Z')
        && !key.getModifiers().isShiftDown())
    {
        undo();
        return true;
    }

    if ((key.getModifiers().isCommandDown() && key.getModifiers().isShiftDown()
         && (key.getTextCharacter() == 'z' || key.getTextCharacter() == 'Z'))
        || (key.getModifiers().isCommandDown() && (key.getTextCharacter() == 'y' || key.getTextCharacter() == 'Y')))
    {
        redo();
        return true;
    }

    if (key.getModifiers().isCommandDown() && (key.getTextCharacter() == 'q' || key.getTextCharacter() == 'Q'))
    {
        quantizeSelectedNotes();
        return true;
    }

    return false;
}

void StudioShellComponent::timerCallback()
{
    const ScopedAppProfileSample profileSample(AppProfileSection::timerCallback);
    const auto nowMs = juce::Time::getMillisecondCounter();
    pollAiComposeFuture();

    const bool hasSharedRackHost = nativeVstHost.isReady();
    const bool hasOpenRackEditors = hasOpenRackEditorSessions();

    if (!hasSharedRackHost && !hasOpenRackEditors)
    {
        refreshFloatingWindows();
        return;
    }

    if (!rackPreviewRunning && !projectPreviewRunning && !hasOpenRackEditors)
    {
        if (hasSharedRackHost)
        {
            NativeVstHostSession::TransportSnapshot transportSnapshot;
            if (nativeVstHost.queryTransportSnapshot(transportSnapshot, false).wasOk())
            {
                transportCpuUsagePercent = juce::jmax(0.0, transportSnapshot.cpuUsage * 100.0);
                transportMasterPeakLeft = transportSnapshot.masterPeakLeft;
                transportMasterPeakRight = transportSnapshot.masterPeakRight;
            }
        }

        syncOpenRackEditorSessions(false);
        refreshFloatingWindows();
        return;
    }

    const bool shouldRefreshTrackMetersThisTick = rackPreviewRunning || ((playbackUiTickCounter % 3) == 0);
    NativeVstHostSession::TransportSnapshot transportSnapshot;
    const auto snapshotResult = hasSharedRackHost
        ? nativeVstHost.queryTransportSnapshot(transportSnapshot, shouldRefreshTrackMetersThisTick)
        : juce::Result::fail("Native VST host session is not ready.");
    if (snapshotResult.failed())
    {
        syncOpenRackEditorSessions(true);
        refreshFloatingWindows(false);
        return;
    }

    ++playbackUiTickCounter;
    const bool shouldRefreshEditorsThisTick = (playbackUiTickCounter % 4) == 0;
    const bool shouldFlushDeferredParameterSyncThisTick = (projectPreviewRunning || rackPreviewRunning)
        && (nowMs - lastDeferredParameterFlushMs) >= kDeferredEngineParameterFlushIntervalMs;

    if (shouldFlushDeferredParameterSyncThisTick
        && juce::isPositiveAndBelow(pendingLiveRackParameterEngineSyncTrack, static_cast<int>(documentState.project.tracks.size())))
    {
        const auto deferredTrackIndex = pendingLiveRackParameterEngineSyncTrack;
        const auto& deferredTrack = documentState.project.tracks[static_cast<size_t>(deferredTrackIndex)];
        const auto parameterSyncResult = nativeVstHost.setAudioEngineTrackParameters(deferredTrackIndex, deferredTrack);
        lastDeferredParameterFlushMs = nowMs;
        if (parameterSyncResult.wasOk())
        {
            pendingLiveRackParameterEngineSyncTrack = -1;
            audioEngineStateValid = true;
            audioEngineStateDirty = false;
        }
        else
        {
            pendingLiveRackParameterEngineSyncTrack = -1;
            statusLabel.setText("Live modulation sync failed: " + parameterSyncResult.getErrorMessage(),
                                juce::dontSendNotification);
        }
    }

    const auto sampleRate = juce::jmax(1.0, transportSnapshot.sampleRate);
    const auto inprocessRunning = transportSnapshot.inprocessTransportRunning;
    const auto audioEngineRunning = transportSnapshot.audioEngineRunning || transportSnapshot.audioEngineTailing;
    transportCpuUsagePercent = juce::jmax(0.0, transportSnapshot.cpuUsage * 100.0);
    transportMasterPeakLeft = transportSnapshot.masterPeakLeft;
    transportMasterPeakRight = transportSnapshot.masterPeakRight;
    syncOpenRackEditorSessions(true);

    const auto positionFrame = (projectPreviewRunning || rackPreviewRunning)
        ? transportSnapshot.audioEnginePositionFrame
        : transportSnapshot.inprocessTransportPositionFrame;
    const auto stillRunning = (projectPreviewRunning || rackPreviewRunning) ? audioEngineRunning : inprocessRunning;

    bool trackMeterLevelsChanged = false;
    if (projectPreviewRunning || rackPreviewRunning)
    {
        if (shouldRefreshTrackMetersThisTick)
        {
            std::fill(trackMeterLevels.begin(), trackMeterLevels.end(), 0.0f);
            for (int trackIndex = 0; trackIndex < transportSnapshot.audioEngineTrackPeakLevels.size(); ++trackIndex)
            {
                if (!juce::isPositiveAndBelow(trackIndex, static_cast<int>(trackMeterLevels.size())))
                    continue;
                trackMeterLevels[static_cast<size_t>(trackIndex)] = transportSnapshot.audioEngineTrackPeakLevels.getReference(trackIndex);
            }
            trackMeterLevelsChanged = true;
        }
    }

    documentState.project.playheadTick = juce::jmax(0, secondsToTick(static_cast<double>(positionFrame) / sampleRate,
                                                                     documentState.project.bpm));
    documentState.project.recalculateTimeFields();
    if (headerTimecodeDisplay != nullptr)
    {
        headerTimecodeDisplay->setValues(documentState.project.playheadSec,
                                         projectSequenceLengthSeconds(documentState.project),
                                         documentState.project.leftLocatorSec,
                                         documentState.project.rightLocatorSec);
    }
    if (pianoRoll != nullptr && !pianoRollViewport.getBounds().isEmpty())
        pianoRoll->repaint();
    if (arrangementOverview != nullptr && !arrangementViewport.getBounds().isEmpty())
        arrangementOverview->repaint();
    if (sampleTimeline != nullptr && !sampleTimelineViewport.getBounds().isEmpty())
        sampleTimeline->repaint();
    if (automationEditor != nullptr && !automationEditor->getBounds().isEmpty())
        automationEditor->refreshViewState();
    if (mixerComponent != nullptr)
        mixerComponent->refreshMeters();
    if (trackMeterLevelsChanged)
        repaintTrackVolumeMeters();

    if (!stillRunning)
    {
        rackPreviewRunning = false;
        projectPreviewRunning = false;
        playbackUiTickCounter = 0;
        std::fill(trackMeterLevels.begin(), trackMeterLevels.end(), 0.0f);
        transportMasterPeakLeft = 0.0f;
        transportMasterPeakRight = 0.0f;
        if (mixerComponent != nullptr)
            mixerComponent->refreshMeters();
        repaintTrackVolumeMeters();
        updateEditorState();
        statusLabel.setText("Native preview finished.", juce::dontSendNotification);
    }

    refreshFloatingWindows(shouldRefreshEditorsThisTick);
}

juce::Result StudioShellComponent::openProjectFile(const juce::File& file)
{
    if (rackPreviewRunning && nativeVstHost.isReady())
        nativeVstHost.stopAudioEngine();
    if (projectPreviewRunning && nativeVstHost.isReady())
        nativeVstHost.stopAudioEngine();
    closeAllRackEditorSessions();
    resetRackHostTracking();
    rackPreviewRunning = false;
    projectPreviewRunning = false;
    playbackUiTickCounter = 0;
    pendingLiveRackParameterEngineSyncTrack = -1;
    audioEngineStateValid = false;
    audioEngineStateDirty = true;

    ProjectFileData loaded;
    const auto result = loadProjectFile(file, loaded);
    if (result.failed())
        return result;

    documentState = std::move(loaded);
    syncBundledRackCatalogInProject();
    currentProjectFile = file;
    clearDirty();
    undoManager.clearUndoHistory();
    refreshUi();
    trackTable.selectRow(0);
    statusLabel.setText("Loaded project: " + file.getFileName(), juce::dontSendNotification);
    appendActivityLog("Project", "Loaded project\n" + file.getFullPathName());
    return juce::Result::ok();
}

void StudioShellComponent::refreshUi()
{
    normaliseProject(documentState.project);
    trackMeterLevels.assign(documentState.project.tracks.size(), 0.0f);

    tempoSlider.setValue(documentState.project.bpm, juce::dontSendNotification);
    loopToggle.setToggleState(documentState.project.loopEnabled, juce::dontSendNotification);
    metronomeToggle.setToggleState(documentState.project.metronomeEnabled, juce::dontSendNotification);
    refreshProjectSummaryLabels();

    const auto trackCount = static_cast<int>(documentState.project.tracks.size());
    const auto selectedRow = getSelectedTrackIndex();
    const auto selectedSampleAssetRow = sampleAssetList.getSelectedRow();
    trackTable.updateContent();
    sampleAssetList.updateContent();
    if (trackCount > 0)
        trackTable.selectRow(juce::jlimit(0, trackCount - 1, selectedRow >= 0 ? selectedRow : 0), juce::dontSendNotification);
    if (!documentState.project.sampleAssets.empty())
        sampleAssetList.selectRow(juce::jlimit(0,
                                               static_cast<int>(documentState.project.sampleAssets.size()) - 1,
                                               selectedSampleAssetRow >= 0 ? selectedSampleAssetRow : 0),
                                  juce::dontSendNotification);
    else
        sampleAssetList.deselectAllRows();

    ensureSelectedMidiSectionForTrack(getSelectedTrackIndex());
    auto patternBarsSelection = defaultPatternLengthTicks(documentState.project);
    if (juce::isPositiveAndBelow(selectedMidiSectionIndex, static_cast<int>(documentState.project.midiSections.size())))
    {
        const auto& section = documentState.project.midiSections[static_cast<size_t>(selectedMidiSectionIndex)];
        if (const auto* pattern = findMidiPattern(documentState.project, section.patternId))
            patternBarsSelection = patternLengthTicks(*pattern);
    }
    patternBarsBox.setSelectedId(patternBarsSelection, juce::dontSendNotification);
    if (patternBarsBox.getSelectedId() != patternBarsSelection)
        patternBarsBox.setText(sequenceTickLabel(patternBarsSelection), juce::dontSendNotification);
    keyQuantizeBox.setSelectedId(keyQuantizeOptionId(documentState.project), juce::dontSendNotification);
    if (keyQuantizeBox.getSelectedId() == 0)
        keyQuantizeBox.setText(keyQuantizeDisplayName(documentState.project.keyQuantizeRoot, documentState.project.keyQuantizeScale),
                               juce::dontSendNotification);
    arrangementSnapBox.setSelectedId(arrangementSnapTickLength(documentState.project), juce::dontSendNotification);
    arrangementZoomSlider.setValue(arrangementZoomPixelsPerBar, juce::dontSendNotification);
    arrangementLaneHeightSlider.setValue(arrangementLaneHeightPixels, juce::dontSendNotification);
    pianoRollZoomSlider.setValue(pianoRollZoomPixelsPerBeat, juce::dontSendNotification);
    pianoRollRowHeightSlider.setValue(pianoRollRowHeightPixels, juce::dontSendNotification);
    applyEditorViewScaleState();
    refreshInspector();
    updateEditorState();
    updateAiStatusSummary();
}

void StudioShellComponent::refreshProjectSummaryLabels()
{
    const auto trackCount = static_cast<int>(documentState.project.tracks.size());
    const auto noteCount = documentState.project.getTotalNoteCount();
    const auto locatorLengthSec = juce::jmax(0.0, documentState.project.rightLocatorSec - documentState.project.leftLocatorSec);
    const auto totalLengthSec = projectSequenceLengthSeconds(documentState.project);

    if (headerTimecodeDisplay != nullptr)
    {
        headerTimecodeDisplay->setValues(documentState.project.playheadSec,
                                         totalLengthSec,
                                         documentState.project.leftLocatorSec,
                                         documentState.project.rightLocatorSec);
    }

    statsLabel.setText("Tracks: " + juce::String(trackCount)
            + "   Notes: " + juce::String(noteCount)
            + "   Patterns: " + juce::String(static_cast<int>(documentState.project.midiPatterns.size()))
            + "   Clips: " + juce::String(static_cast<int>(documentState.project.midiSections.size()))
            + "   BPM: " + juce::String(documentState.project.bpm)
            + "   Key: " + keyQuantizeDisplayName(documentState.project.keyQuantizeRoot, documentState.project.keyQuantizeScale)
            + "   Seq len: " + juce::String(totalLengthSec, 2) + " s"
            + "   Loop span: " + juce::String(locatorLengthSec, 2) + " s"
            + "   Samples: " + juce::String(static_cast<int>(documentState.project.sampleClips.size()))
            + "   Rack items: " + juce::String(static_cast<int>(documentState.project.vstRack.size()))
            + "   Undo: " + juce::String(undoManager.canUndo() ? "yes" : "no")
            + "   Redo: " + juce::String(undoManager.canRedo() ? "yes" : "no"),
        juce::dontSendNotification);

    auto displayPath = currentProjectFile.existsAsFile()
        ? currentProjectFile.getFullPathName()
        : juce::String("Unsaved native project");
    if (dirty)
        displayPath << " *";
    fileLabel.setText(displayPath, juce::dontSendNotification);
}

void StudioShellComponent::applyEditorViewScaleState()
{
    if (arrangementOverview != nullptr)
    {
        arrangementOverview->setHorizontalZoom(arrangementZoomPixelsPerBar);
        arrangementOverview->setLaneHeight(arrangementLaneHeightPixels);
    }

    if (floatingArrangementOverview != nullptr)
    {
        floatingArrangementOverview->setHorizontalZoom(arrangementZoomPixelsPerBar);
        floatingArrangementOverview->setLaneHeight(arrangementLaneHeightPixels);
    }

    if (pianoRoll != nullptr)
    {
        pianoRoll->setHorizontalZoom(pianoRollZoomPixelsPerBeat);
        pianoRoll->setNoteRowHeight(pianoRollRowHeightPixels);
    }

    if (floatingPianoRollWorkspace != nullptr)
        floatingPianoRollWorkspace->setViewScale(pianoRollZoomPixelsPerBeat, pianoRollRowHeightPixels);

    if (panelsWindowContent != nullptr)
    {
        panelsWindowContent->setArrangementViewScale(arrangementZoomPixelsPerBar, arrangementLaneHeightPixels);
        panelsWindowContent->setPianoRollViewScale(pianoRollZoomPixelsPerBeat, pianoRollRowHeightPixels);
    }
}

void StudioShellComponent::repaintTrackVolumeMeters()
{
    if (!trackTable.isShowing())
        return;

    const auto rowCount = static_cast<int>(documentState.project.tracks.size());
    for (int row = 0; row < rowCount; ++row)
    {
        const auto cellBounds = trackTable.getCellPosition(kColumnVolume, row, true);
        if (!cellBounds.isEmpty())
            trackTable.repaint(cellBounds);
    }
}

void StudioShellComponent::refreshInspector()
{
    if (const auto* track = getSelectedTrack())
    {
        syncingInspectorControls = true;
        trackNameEditor.setText(track->name, juce::dontSendNotification);
        trackTypeEditor.setText(track->trackType, juce::dontSendNotification);
        instrumentModeEditor.setText(track->instrumentMode, juce::dontSendNotification);
        instrumentEditor.setText(track->instrument, juce::dontSendNotification);
        rackVstEditor.setText(track->rackVst, juce::dontSendNotification);
        midiChannelSlider.setValue(track->midiChannel + 1, juce::dontSendNotification);
        midiProgramSlider.setValue(track->midiProgram, juce::dontSendNotification);
        volumeSlider.setValue(track->volume, juce::dontSendNotification);
        panSlider.setValue(track->pan, juce::dontSendNotification);
        muteToggle.setToggleState(track->mute, juce::dontSendNotification);
        soloToggle.setToggleState(track->solo, juce::dontSendNotification);
        liveArmToggle.setToggleState(track->liveArmed, juce::dontSendNotification);
        syncingInspectorControls = false;

        trackNameEditor.setEnabled(true);
        trackTypeEditor.setEnabled(true);
        instrumentModeEditor.setEnabled(true);
        instrumentEditor.setEnabled(true);
        rackVstEditor.setEnabled(true);
        midiChannelSlider.setEnabled(true);
        midiProgramSlider.setEnabled(true);
        volumeSlider.setEnabled(true);
        panSlider.setEnabled(true);
        muteToggle.setEnabled(true);
        soloToggle.setEnabled(true);
        liveArmToggle.setEnabled(true);
        inspectorEditor.setText(describeTrack(documentState.project, *track), false);
    }
    else
    {
        syncingInspectorControls = true;
        trackNameEditor.setText({}, juce::dontSendNotification);
        trackTypeEditor.setText({}, juce::dontSendNotification);
        instrumentModeEditor.setText({}, juce::dontSendNotification);
        instrumentEditor.setText({}, juce::dontSendNotification);
        rackVstEditor.setText({}, juce::dontSendNotification);
        midiChannelSlider.setValue(1.0, juce::dontSendNotification);
        midiProgramSlider.setValue(0.0, juce::dontSendNotification);
        volumeSlider.setValue(0.8, juce::dontSendNotification);
        panSlider.setValue(0.0, juce::dontSendNotification);
        muteToggle.setToggleState(false, juce::dontSendNotification);
        soloToggle.setToggleState(false, juce::dontSendNotification);
        liveArmToggle.setToggleState(false, juce::dontSendNotification);
        syncingInspectorControls = false;

        trackNameEditor.setEnabled(false);
        trackTypeEditor.setEnabled(false);
        instrumentModeEditor.setEnabled(false);
        instrumentEditor.setEnabled(false);
        rackVstEditor.setEnabled(false);
        midiChannelSlider.setEnabled(false);
        midiProgramSlider.setEnabled(false);
        volumeSlider.setEnabled(false);
        panSlider.setEnabled(false);
        muteToggle.setEnabled(false);
        soloToggle.setEnabled(false);
        liveArmToggle.setEnabled(false);
        inspectorEditor.setText("No track selected.", false);
    }
}

void StudioShellComponent::updateEditorState()
{
    auto patternBarsSelection = defaultPatternLengthTicks(documentState.project);
    if (juce::isPositiveAndBelow(selectedMidiSectionIndex, static_cast<int>(documentState.project.midiSections.size())))
    {
        const auto& section = documentState.project.midiSections[static_cast<size_t>(selectedMidiSectionIndex)];
        if (const auto* pattern = findMidiPattern(documentState.project, section.patternId))
            patternBarsSelection = patternLengthTicks(*pattern);
    }
    patternBarsBox.setSelectedId(patternBarsSelection, juce::dontSendNotification);
    if (patternBarsBox.getSelectedId() != patternBarsSelection)
        patternBarsBox.setText(sequenceTickLabel(patternBarsSelection), juce::dontSendNotification);
    keyQuantizeBox.setSelectedId(keyQuantizeOptionId(documentState.project), juce::dontSendNotification);
    if (keyQuantizeBox.getSelectedId() == 0)
        keyQuantizeBox.setText(keyQuantizeDisplayName(documentState.project.keyQuantizeRoot, documentState.project.keyQuantizeScale),
                               juce::dontSendNotification);
    arrangementSnapBox.setSelectedId(arrangementSnapTickLength(documentState.project), juce::dontSendNotification);

    if (mixerComponent != nullptr)
        mixerComponent->refreshFromModel();

    if (arrangementOverview != nullptr)
        arrangementOverview->refreshFromModel();

    if (automationEditor != nullptr)
        automationEditor->refreshFromModel();

    if (sampleTimeline != nullptr)
        sampleTimeline->refreshFromModel();

    if (pianoRoll != nullptr)
    {
        pianoRoll->setToolMode(editorToolMode);
        pianoRoll->refreshFromModel();
    }

    const auto hasSelection = getSelectedTrack() != nullptr;
    placeSampleButton.setEnabled(sampleAssetList.getSelectedRow() >= 0 && findPreferredSampleTrackIndex() >= 0);
    playProjectButton.setEnabled(!documentState.project.tracks.empty());
    playTrackButton.setEnabled(hasSelection);
    stopTrackButton.setEnabled((rackPreviewRunning || projectPreviewRunning) && nativeVstHost.isReady());
    exportWavButton.setEnabled(!documentState.project.tracks.empty() || !documentState.project.sampleClips.empty());
    aiSettingsButton.setEnabled(!aiComposeBusy);
    aiComposeButton.setEnabled(!aiComposeBusy);
    undoButton.setEnabled(undoManager.canUndo());
    redoButton.setEnabled(undoManager.canRedo());
    refreshFloatingWindows();
}

void StudioShellComponent::markDirty()
{
    dirty = true;
}

void StudioShellComponent::clearDirty()
{
    dirty = false;
}

void StudioShellComponent::restorePersistedWindowVisibility()
{
    if (windowStateSettings == nullptr)
        return;

    transportWindowVisible = windowStateSettings->getBoolValue("window_transport_visible", transportWindowVisible);
    mixerWindowVisible = windowStateSettings->getBoolValue("window_mixer_visible", mixerWindowVisible);
    audioWindowVisible = windowStateSettings->getBoolValue("window_audio_visible", audioWindowVisible);
    panelsWindowVisible = windowStateSettings->getBoolValue("window_panels_visible", panelsWindowVisible);
    tracksWindowVisible = windowStateSettings->getBoolValue("window_tracks_visible", tracksWindowVisible);
    rackBrowserWindowVisible = windowStateSettings->getBoolValue("window_rack_browser_visible", rackBrowserWindowVisible);
    renderManagerWindowVisible = windowStateSettings->getBoolValue("window_render_manager_visible", renderManagerWindowVisible);
    arrangementWindowVisible = windowStateSettings->getBoolValue("window_arrangement_visible", arrangementWindowVisible);
    automationWindowVisible = windowStateSettings->getBoolValue("window_automation_visible", automationWindowVisible);
    samplesWindowVisible = windowStateSettings->getBoolValue("window_samples_visible", samplesWindowVisible);
    pianoRollWindowVisible = windowStateSettings->getBoolValue("window_piano_roll_visible", pianoRollWindowVisible);
    virtualPianoWindowVisible = windowStateSettings->getBoolValue("window_virtual_piano_visible", virtualPianoWindowVisible);
    activityLogWindowVisible = windowStateSettings->getBoolValue("window_activity_log_visible", activityLogWindowVisible);
}

void StudioShellComponent::persistWindowVisibilityState() const
{
    if (windowStateSettings == nullptr)
        return;

    windowStateSettings->setValue("window_transport_visible", transportWindowVisible);
    windowStateSettings->setValue("window_mixer_visible", mixerWindowVisible);
    windowStateSettings->setValue("window_audio_visible", audioWindowVisible);
    windowStateSettings->setValue("window_panels_visible", panelsWindowVisible);
    windowStateSettings->setValue("window_tracks_visible", tracksWindowVisible);
    windowStateSettings->setValue("window_rack_browser_visible", rackBrowserWindowVisible);
    windowStateSettings->setValue("window_render_manager_visible", renderManagerWindowVisible);
    windowStateSettings->setValue("window_arrangement_visible", arrangementWindowVisible);
    windowStateSettings->setValue("window_automation_visible", automationWindowVisible);
    windowStateSettings->setValue("window_samples_visible", samplesWindowVisible);
    windowStateSettings->setValue("window_piano_roll_visible", pianoRollWindowVisible);
    windowStateSettings->setValue("window_virtual_piano_visible", virtualPianoWindowVisible);
    windowStateSettings->setValue("window_activity_log_visible", activityLogWindowVisible);
    windowStateSettings->saveIfNeeded();
}

void StudioShellComponent::appendActivityLog(const juce::String& title, const juce::String& body)
{
    const auto timestamp = juce::Time::getCurrentTime().formatted("[%H:%M:%S]");
    const auto header = title.trim().isNotEmpty() ? timestamp + " " + title.trim() : timestamp;
    const auto entry = body.trim().isNotEmpty() ? (header + "\n" + body.trim()) : header;

    {
        const juce::ScopedLock lock(activityLogLock);
        activityLogEntries.add(entry);
        constexpr int kMaxActivityLogEntries = 400;
        while (activityLogEntries.size() > kMaxActivityLogEntries)
            activityLogEntries.remove(0);
    }

    appendActivityLogToFile(entry);

    auto refreshUiNow = [safeThis = juce::Component::SafePointer<StudioShellComponent>(this)]
    {
        if (safeThis == nullptr)
            return;

        if (safeThis->activityLogWindowContent != nullptr)
            safeThis->activityLogWindowContent->refreshFromModel();
    };

    if (auto* messageManager = juce::MessageManager::getInstanceWithoutCreating();
        messageManager != nullptr && messageManager->isThisTheMessageThread())
    {
        refreshUiNow();
    }
    else
    {
        juce::MessageManager::callAsync([refreshUiNow]
        {
            refreshUiNow();
        });
    }
}

void StudioShellComponent::appendActivityLogToFile(const juce::String& entry) const
{
    if (activityLogFile == juce::File())
        return;

    const auto parent = activityLogFile.getParentDirectory();
    if (parent != juce::File())
        parent.createDirectory();

    if (!activityLogFile.existsAsFile())
        activityLogFile.replaceWithText("Mutagen Activity Log\n\n", false, false, "\n");

    activityLogFile.appendText(entry + "\n\n", false, false, "\n");
}

juce::String StudioShellComponent::activityLogTextSnapshot() const
{
    const juce::ScopedLock lock(activityLogLock);
    return activityLogEntries.joinIntoString("\n\n");
}

void StudioShellComponent::clearActivityLog()
{
    {
        const juce::ScopedLock lock(activityLogLock);
        activityLogEntries.clear();
    }

    if (activityLogWindowContent != nullptr)
        activityLogWindowContent->refreshFromModel();
}

StudioShellComponent::RackEditorSession* StudioShellComponent::findRackEditorSession(int trackIndex)
{
    for (auto& session : rackEditorSessions)
    {
        if (session != nullptr && session->trackIndex == trackIndex)
            return session.get();
    }

    return nullptr;
}

const StudioShellComponent::RackEditorSession* StudioShellComponent::findRackEditorSession(int trackIndex) const
{
    for (const auto& session : rackEditorSessions)
    {
        if (session != nullptr && session->trackIndex == trackIndex)
            return session.get();
    }

    return nullptr;
}

bool StudioShellComponent::hasOpenRackEditorSessions() const
{
    for (const auto& session : rackEditorSessions)
    {
        if (session != nullptr && session->editorOpen)
            return true;
    }

    return false;
}

bool StudioShellComponent::isTrackBeingLiveEdited(int trackIndex) const
{
    if (trackIndex < 0)
        return false;

    if (const auto* session = findRackEditorSession(trackIndex))
        return session->editorOpen;

    return false;
}

juce::Result StudioShellComponent::ensureRackEditorSessionReadyForTrack(int trackIndex,
                                                                        const TrackState& track,
                                                                        RackEditorSession*& outSession)
{
    outSession = nullptr;

    const auto pluginPath = resolveRackPluginPath(documentState.project, track);
    if (pluginPath.isEmpty())
        return juce::Result::fail("The selected track does not resolve to a native rack plugin. Assign a bundled rack instrument or point Rack VST to a plugin path.");

    const auto pluginFile = juce::File(pluginPath);
    if (!pluginFile.exists())
        return juce::Result::fail("The resolved Rack VST path does not exist:\n" + pluginPath);

    auto result = ensureNativeAudioEnginePrepared(projectPreviewRunning);
    if (result.failed())
        return result;

    if (auto* existingSession = findRackEditorSession(trackIndex))
    {
        bool editorOpen = false;
        existingSession->editorOpen = nativeVstHost.queryAudioEngineTrackEditorOpen(trackIndex, editorOpen).wasOk()
            && editorOpen;
        outSession = existingSession;
        return juce::Result::ok();
    }

    auto session = std::make_unique<RackEditorSession>();
    session->trackIndex = trackIndex;
    bool editorOpen = false;
    session->editorOpen = nativeVstHost.queryAudioEngineTrackEditorOpen(trackIndex, editorOpen).wasOk()
        && editorOpen;

    outSession = session.get();
    rackEditorSessions.push_back(std::move(session));
    refreshPollingTimerState();
    return juce::Result::ok();
}

void StudioShellComponent::syncOpenRackEditorSessions(bool duringPlayback)
{
    const auto nowMs = juce::Time::getMillisecondCounter();
    if (duringPlayback && (nowMs - lastRackEditorSessionSyncMs) < kPlaybackRackEditorSyncIntervalMs)
        return;

    if (duringPlayback)
        lastRackEditorSessionSyncMs = nowMs;

    juce::Array<int> sharedSessionsToClose;
    for (auto& session : rackEditorSessions)
    {
        if (session == nullptr)
            continue;

        if (!session->editorOpen)
            continue;

        bool editorOpen = false;
        if (nativeVstHost.queryAudioEngineTrackEditorOpen(session->trackIndex, editorOpen).failed())
            continue;

        if (editorOpen)
            continue;

        NativeVstHostSession::RackParameterSnapshot rackSnapshot;
        if (nativeVstHost.queryAudioEngineTrackParameterSnapshot(session->trackIndex, rackSnapshot).wasOk())
        {
            session->lastStateGeneration = rackSnapshot.stateGeneration;
            syncTrackRackParametersFromValues(session->trackIndex,
                                              rackSnapshot.parameterValues,
                                              true);
        }

        session->editorOpen = false;
        sharedSessionsToClose.addIfNotAlreadyThere(session->trackIndex);
    }

    for (const auto trackIndex : sharedSessionsToClose)
        closeRackEditorSession(trackIndex);

    if (!sharedSessionsToClose.isEmpty())
    {
        refreshPollingTimerState();
        updateEditorState();
        trackTable.repaint();
    }
}

void StudioShellComponent::closeRackEditorSession(int trackIndex)
{
    for (auto it = rackEditorSessions.begin(); it != rackEditorSessions.end(); ++it)
    {
        auto& session = *it;
        if (session == nullptr || session->trackIndex != trackIndex)
            continue;

        nativeVstHost.closeAudioEngineTrackEditor(trackIndex);

        rackEditorSessions.erase(it);
        break;
    }
}

void StudioShellComponent::closeAllRackEditorSessions()
{
    for (auto& session : rackEditorSessions)
    {
        if (session == nullptr)
            continue;

        nativeVstHost.closeAudioEngineTrackEditor(session->trackIndex);
    }

    rackEditorSessions.clear();
}

juce::Result StudioShellComponent::ensureNativeAudioEnginePrepared(bool preserveTransport)
{
    auto result = nativeVstHost.ensureCreated();
    if (result.failed())
        return result;

    if (!audioEngineStateValid || audioEngineStateDirty)
    {
        result = nativeVstHost.setAudioEngineState(documentState.project, preserveTransport);
        if (result.failed())
            return result;

        pendingLiveRackParameterEngineSyncTrack = -1;
        audioEngineStateValid = true;
        audioEngineStateDirty = false;
    }

    return juce::Result::ok();
}

void StudioShellComponent::showTrackContextMenu(int rowNumber, juce::Point<int> screenPosition)
{
    if (!juce::isPositiveAndBelow(rowNumber, static_cast<int>(documentState.project.tracks.size())))
        return;

    setSelectedTrackIndex(rowNumber);
    const auto& track = documentState.project.tracks[static_cast<size_t>(rowNumber)];

    juce::PopupMenu menu;
    constexpr int menuEditVst = 1;
    constexpr int menuDuplicateTrack = 2;
    constexpr int menuRemoveTrack = 3;

    menu.addItem(menuEditVst,
                 "Edit VST",
                 track.trackType != "sample" && track.instrumentMode.containsIgnoreCase("VST"));
    menu.addSeparator();
    menu.addItem(menuDuplicateTrack, "Duplicate Track");
    menu.addItem(menuRemoveTrack, "Remove Track");

    menu.showMenuAsync(juce::PopupMenu::Options().withTargetScreenArea({ screenPosition.x, screenPosition.y, 1, 1 }),
                       [safeThis = juce::Component::SafePointer<StudioShellComponent>(this), rowNumber](int result)
                       {
                           if (safeThis == nullptr)
                               return;

                           safeThis->setSelectedTrackIndex(rowNumber);
                           switch (result)
                           {
                               case menuEditVst: safeThis->openSelectedTrackRackEditor(); break;
                               case menuDuplicateTrack: safeThis->duplicateSelectedTrack(); break;
                               case menuRemoveTrack: safeThis->removeSelectedTrack(); break;
                               default: break;
                           }
                       });
}

void StudioShellComponent::setupFloatingWindows()
{
    const auto displayArea = juce::Desktop::getInstance().getDisplays().getPrimaryDisplay() != nullptr
        ? juce::Desktop::getInstance().getDisplays().getPrimaryDisplay()->userArea
        : juce::Rectangle<int>(80, 80, 1600, 900);
    const auto attachVirtualPianoShortcutHandler = [this] (FloatingPanelWindow* window)
    {
        if (window == nullptr)
            return;

        window->onKeyPressed = [this] (const juce::KeyPress& key)
        {
            return tryHandleVirtualPianoShortcut(key);
        };
    };

    transportWindow = std::make_unique<FloatingPanelWindow>("Transport", true);
    attachVirtualPianoShortcutHandler(transportWindow.get());
    transportWindow->onClosePressed = [this]
    {
        transportWindowVisible = false;
        persistWindowVisibilityState();
    };
    auto* newTransportPanel = new TransportPanelComponent([this] { jumpPlayheadToStart(); },
                                                          [this] { playFullProjectThroughNativeEngine(); },
                                                          [this] { playSelectedTrackThroughRack(); },
                                                          [this] { stopRackPreview(); },
                                                          [this] (int bpm) { setTransportTempo(bpm); },
                                                          [this] (bool enabled) { setTransportLoopEnabled(enabled); },
                                                          [this] (bool enabled) { setTransportMetronomeEnabled(enabled); });
    transportPanel = newTransportPanel;
    transportWindow->setContentOwned(newTransportPanel, true);
    transportWindow->setBounds(displayArea.getX() + juce::jmax(24, (displayArea.getWidth() - 980) / 2),
                               displayArea.getY() + 18,
                               juce::jmin(980, displayArea.getWidth() - 48),
                               128);
    transportWindow->setVisible(transportWindowVisible);

    audioWindow = std::make_unique<FloatingPanelWindow>("Audio");
    attachVirtualPianoShortcutHandler(audioWindow.get());
    audioWindow->onClosePressed = [this]
    {
        audioWindowVisible = false;
        persistWindowVisibilityState();
    };
    auto* newFloatingAudioWorkspace = new AudioWorkspaceWindowComponent([this] { jumpPlayheadToStart(); },
                                                                        [this] { playFullProjectThroughNativeEngine(); },
                                                                        [this] { playSelectedTrackThroughRack(); },
                                                                        [this] { stopRackPreview(); },
                                                                        [this] { showAudioSettingsWindow(); },
                                                                        [this] (int bpm) { setTransportTempo(bpm); },
                                                                        [this] (bool enabled) { setTransportLoopEnabled(enabled); },
                                                                        [this] (bool enabled) { setTransportMetronomeEnabled(enabled); },
                                                                        [this] { promptExportWav(); },
                                                                        [this] { promptExportProjectStems(); },
                                                                        [this] () -> const ProjectState& { return documentState.project; },
                                                                        [this] (int trackIndex,
                                                                                const TrackState& track,
                                                                                bool undoable,
                                                                                const juce::String& actionName)
                                                                        {
                                                                            if (undoable)
                                                                                applyTrackStateEdit(trackIndex, track, actionName);
                                                                            else
                                                                                replaceTrackStateNoUndo(trackIndex, track);
                                                                        },
                                                                        [this] (int trackIndex) -> float
                                                                        {
                                                                            return juce::isPositiveAndBelow(trackIndex, static_cast<int>(trackMeterLevels.size()))
                                                                                ? trackMeterLevels[static_cast<size_t>(trackIndex)]
                                                                                : 0.0f;
                                                                        },
                                                                        [this] () -> juce::String
                                                                        {
                                                                            const auto createResult = nativeVstHost.ensureCreated();
                                                                            if (createResult.failed())
                                                                                return "Audio host unavailable.";

                                                                            NativeVstHostSession::AudioDeviceSnapshot snapshot;
                                                                            const auto snapshotResult = nativeVstHost.queryAudioDeviceSnapshot(snapshot);
                                                                            if (snapshotResult.failed())
                                                                                return "Audio device status unavailable.";

                                                                            const auto sampleRate = juce::roundToInt(snapshot.sampleRate);
                                                                            const auto bufferSize = snapshot.bufferSize;

                                                                            return "Driver: "
                                                                                + (snapshot.audioDeviceType.isNotEmpty() ? snapshot.audioDeviceType : "Default")
                                                                                + "  |  Device: "
                                                                                + (snapshot.audioDeviceName.isNotEmpty() ? snapshot.audioDeviceName : "Unavailable")
                                                                                + "  |  "
                                                                                + juce::String(juce::jmax(1, sampleRate))
                                                                                + " Hz  |  "
                                                                                + juce::String(juce::jmax(1, bufferSize))
                                                                                + " samples";
                                                                        });
    floatingAudioWorkspace = newFloatingAudioWorkspace;
    audioWindow->setContentOwned(newFloatingAudioWorkspace, true);
    audioWindow->setBounds(displayArea.getRight() - juce::jmin(1160, displayArea.getWidth() - 80) - 36,
                           displayArea.getY() + 154,
                           juce::jmin(1160, displayArea.getWidth() - 80),
                           juce::jmin(520, displayArea.getHeight() - 230));
    audioWindow->setVisible(audioWindowVisible);

    mixerWindow = std::make_unique<FloatingPanelWindow>("Mixer");
    attachVirtualPianoShortcutHandler(mixerWindow.get());
    mixerWindow->onClosePressed = [this]
    {
        mixerWindowVisible = false;
        persistWindowVisibilityState();
    };
    auto* newFloatingMixer = new MixerComponent([this] () -> const ProjectState& { return documentState.project; },
                                                [this] (int trackIndex,
                                                        const TrackState& track,
                                                        bool undoable,
                                                        const juce::String& actionName)
                                                {
                                                    if (undoable)
                                                        applyTrackStateEdit(trackIndex, track, actionName);
                                                    else
                                                        replaceTrackStateNoUndo(trackIndex, track);
                                                },
                                                [this] (int trackIndex) -> float
                                                {
                                                    return juce::isPositiveAndBelow(trackIndex, static_cast<int>(trackMeterLevels.size()))
                                                        ? trackMeterLevels[static_cast<size_t>(trackIndex)]
                                                        : 0.0f;
                                                });
    floatingMixerComponent = newFloatingMixer;
    mixerWindow->setContentOwned(newFloatingMixer, true);
    mixerWindow->setBounds(displayArea.getX() + 30,
                           displayArea.getY() + 150,
                           juce::jmin(1180, displayArea.getWidth() - 60),
                           juce::jmin(420, displayArea.getHeight() - 220));
    mixerWindow->setVisible(mixerWindowVisible);

    tracksWindow = std::make_unique<FloatingPanelWindow>("Tracks");
    attachVirtualPianoShortcutHandler(tracksWindow.get());
    tracksWindow->onClosePressed = [this]
    {
        tracksWindowVisible = false;
        persistWindowVisibilityState();
    };
    auto* newFloatingTracksWorkspace = new TracksWorkspaceWindowComponent([this] () -> const ProjectState& { return documentState.project; },
                                                                          [this] { return getSelectedTrackIndex(); },
                                                                          [this] (int row) { setSelectedTrackIndex(row); },
                                                                          [this] (std::function<void(TrackState&)> mutation,
                                                                                  const juce::String& actionName)
                                                                          {
                                                                              applySelectedTrackMutation(std::move(mutation), actionName);
                                                                          },
                                                                          [this] (int trackIndex) -> juce::String
                                                                          {
                                                                              if (!juce::isPositiveAndBelow(trackIndex, static_cast<int>(documentState.project.tracks.size())))
                                                                                  return "No track selected.";
                                                                              return describeTrack(documentState.project,
                                                                                                   documentState.project.tracks[static_cast<size_t>(trackIndex)]);
                                                                          },
                                                                          [this] { addTrack(); },
                                                                          [this] { duplicateSelectedTrack(); },
                                                                          [this] { removeSelectedTrack(); },
                                                                          [this] { openSelectedTrackRackEditor(); },
                                                                          [this] { saveSelectedTrackRackState(); },
                                                                          [this] { playSelectedTrackThroughRack(); },
                                                                          [this] { stopRackPreview(); });
    floatingTracksWorkspace = newFloatingTracksWorkspace;
    tracksWindow->setContentOwned(newFloatingTracksWorkspace, true);
    tracksWindow->setBounds(displayArea.getX() + 36,
                            displayArea.getY() + 176,
                            juce::jmin(1220, displayArea.getWidth() - 72),
                            juce::jmin(720, displayArea.getHeight() - 230));
    tracksWindow->setVisible(tracksWindowVisible);

    rackBrowserWindow = std::make_unique<FloatingPanelWindow>("Rack Browser");
    attachVirtualPianoShortcutHandler(rackBrowserWindow.get());
    rackBrowserWindow->onClosePressed = [this]
    {
        rackBrowserWindowVisible = false;
        persistWindowVisibilityState();
    };
    auto* newFloatingRackBrowser = new RackBrowserWindowComponent([this] () -> const ProjectState& { return documentState.project; },
                                                                  [this] { return getSelectedTrackIndex(); },
                                                                  [this] (const juce::String& reference) { assignSelectedTrackRackByReference(reference); },
                                                                  [this] { clearSelectedTrackRackAssignment(); },
                                                                  [this] { materialiseSelectedTrackRackAssignment(); },
                                                                  [this] { promptImportRackPlugin(); },
                                                                  [this] { refreshRackCatalog(); },
                                                                  [this] { openSelectedTrackRackEditor(); },
                                                                  [this] { saveSelectedTrackRackState(); },
                                                                  [this] { playSelectedTrackThroughRack(); },
                                                                  [this] { stopRackPreview(); });
    floatingRackBrowser = newFloatingRackBrowser;
    rackBrowserWindow->setContentOwned(newFloatingRackBrowser, true);
    rackBrowserWindow->setBounds(displayArea.getX() + 82,
                                 displayArea.getY() + 162,
                                 juce::jmin(1180, displayArea.getWidth() - 100),
                                 juce::jmin(700, displayArea.getHeight() - 210));
    rackBrowserWindow->setVisible(rackBrowserWindowVisible);

    renderManagerWindow = std::make_unique<FloatingPanelWindow>("Render Manager");
    attachVirtualPianoShortcutHandler(renderManagerWindow.get());
    renderManagerWindow->onClosePressed = [this]
    {
        renderManagerWindowVisible = false;
        persistWindowVisibilityState();
    };
    auto* newFloatingRenderManager = new RenderManagerWindowComponent([this] () -> const ProjectState& { return documentState.project; },
                                                                      [this] { return getSelectedTrackIndex(); },
                                                                      [this] (int row) { setSelectedTrackIndex(row); },
                                                                      [this] { promptExportSelectedTrackWav(); },
                                                                      [this] { promptExportProjectStems(); },
                                                                      [this] { promptRelinkSelectedTrackRenderedAudio(); },
                                                                      [this] { clearSelectedTrackRenderedAudioPath(); },
                                                                      [this] { importSelectedTrackRenderToSampleLibrary(); },
                                                                      [this] { placeSelectedTrackRenderAtPlayhead(); });
    floatingRenderManager = newFloatingRenderManager;
    renderManagerWindow->setContentOwned(newFloatingRenderManager, true);
    renderManagerWindow->setBounds(displayArea.getX() + 110,
                                   displayArea.getY() + 188,
                                   juce::jmin(1120, displayArea.getWidth() - 120),
                                   juce::jmin(620, displayArea.getHeight() - 220));
    renderManagerWindow->setVisible(renderManagerWindowVisible);

    arrangementWindow = std::make_unique<FloatingPanelWindow>("Arrangement");
    attachVirtualPianoShortcutHandler(arrangementWindow.get());
    arrangementWindow->onClosePressed = [this]
    {
        arrangementWindowVisible = false;
        persistWindowVisibilityState();
    };
    auto* newFloatingArrangement = new ArrangementOverviewComponent([this] () -> const ProjectState& { return documentState.project; },
                                                                    [this] (const ProjectState& project,
                                                                            bool undoable,
                                                                            const juce::String& actionName)
                                                                    {
                                                                        if (undoable)
                                                                            applyProjectStateEdit(project, actionName);
                                                                        else
                                                                            setProjectStateInternal(project);
                                                                    },
                                                                    [this] () { return getSelectedMidiSectionIndex(); },
                                                                    [this] (int sectionIndex, bool focusEditor)
                                                                    {
                                                                        setSelectedMidiSectionIndex(sectionIndex, true);
                                                                        if (focusEditor)
                                                                            focusMidiSectionInPianoRoll(sectionIndex);
                                                                    },
                                                                    [this] () { return editorToolMode; },
                                                                    [this] (EditorToolMode mode)
                                                                    {
                                                                        setEditorToolMode(mode);
                                                                    });
    floatingArrangementOverview = newFloatingArrangement;
    arrangementWindow->setContentOwned(newFloatingArrangement, true);
    arrangementWindow->setBounds(displayArea.getX() + 42,
                                 displayArea.getY() + 180,
                                 juce::jmin(1120, displayArea.getWidth() - 84),
                                 juce::jmin(360, displayArea.getHeight() - 260));
    arrangementWindow->setVisible(arrangementWindowVisible);

    automationWindow = std::make_unique<FloatingPanelWindow>("Automation");
    attachVirtualPianoShortcutHandler(automationWindow.get());
    automationWindow->onClosePressed = [this]
    {
        automationWindowVisible = false;
        persistWindowVisibilityState();
    };
    auto* newFloatingAutomation = new AutomationEditorComponent([this] () -> const ProjectState& { return documentState.project; },
                                                                [this] () -> int { return getSelectedTrackIndex(); },
                                                                [this] (int trackIndex,
                                                                        const TrackState& track,
                                                                        bool undoable,
                                                                        const juce::String& actionName)
                                                                {
                                                                    if (trackIndex < 0)
                                                                        return;

                                                                    if (undoable)
                                                                        applyTrackStateEdit(trackIndex, track, actionName);
                                                                    else
                                                                        replaceTrackStateNoUndo(trackIndex, track);
                                                                });
    floatingAutomationEditor = newFloatingAutomation;
    automationWindow->setContentOwned(newFloatingAutomation, true);
    automationWindow->setBounds(displayArea.getX() + 88,
                                displayArea.getY() + 206,
                                juce::jmin(980, displayArea.getWidth() - 176),
                                juce::jmin(560, displayArea.getHeight() - 260));
    automationWindow->setVisible(automationWindowVisible);

    samplesWindow = std::make_unique<FloatingPanelWindow>("Samples");
    attachVirtualPianoShortcutHandler(samplesWindow.get());
    samplesWindow->onClosePressed = [this]
    {
        samplesWindowVisible = false;
        persistWindowVisibilityState();
    };
    auto* newFloatingSampleWorkspace = new SampleWorkspaceWindowComponent([this] () -> const ProjectState& { return documentState.project; },
                                                                          [this] (const ProjectState& project,
                                                                                  bool undoable,
                                                                                  const juce::String& actionName)
                                                                          {
                                                                              if (undoable)
                                                                                  applyProjectStateEdit(project, actionName);
                                                                              else
                                                                                  setProjectStateInternal(project);
                                                                          },
                                                                          [this] { return getSelectedSampleAssetIndex(); },
                                                                          [this] (int row) { setSelectedSampleAssetIndex(row); },
                                                                          [this] { promptImportSample(); },
                                                                          [this] { placeSelectedSampleAtPlayhead(); },
                                                                          [this]
                                                                          {
                                                                              return getSelectedSampleAssetIndex() >= 0
                                                                                  && findPreferredSampleTrackIndex() >= 0;
                                                                          });
    floatingSampleWorkspace = newFloatingSampleWorkspace;
    samplesWindow->setContentOwned(newFloatingSampleWorkspace, true);
    samplesWindow->setBounds(displayArea.getX() + 120,
                             displayArea.getY() + 236,
                             juce::jmin(1180, displayArea.getWidth() - 180),
                             juce::jmin(680, displayArea.getHeight() - 260));
    samplesWindow->setVisible(samplesWindowVisible);

    pianoRollWindow = std::make_unique<FloatingPanelWindow>("Piano Roll");
    attachVirtualPianoShortcutHandler(pianoRollWindow.get());
    pianoRollWindow->onClosePressed = [this]
    {
        pianoRollWindowVisible = false;
        persistWindowVisibilityState();
    };
    auto* newFloatingPianoRoll = new PianoRollWindowComponent([this] () -> const ProjectState& { return documentState.project; },
                                                              [this] () { return getSelectedTrackIndex(); },
                                                              [this] () { return getSelectedMidiSectionIndex(); },
                                                              [this] (const ProjectState& project,
                                                                      bool undoable,
                                                                      const juce::String& actionName)
                                                              {
                                                                  if (undoable)
                                                                      applyProjectStateEdit(project, actionName);
                                                                  else
                                                                      setProjectStateInternal(project);
                                                               },
                                                               [this] { return editorToolMode; },
                                                               [this] (EditorToolMode mode) { setEditorToolMode(mode); },
                                                               pianoRollZoomPixelsPerBeat,
                                                               pianoRollRowHeightPixels,
                                                               [this] (float zoom)
                                                               {
                                                                   pianoRollZoomPixelsPerBeat = zoom;
                                                                   applyEditorViewScaleState();
                                                               },
                                                               [this] (float rowHeight)
                                                               {
                                                                   pianoRollRowHeightPixels = rowHeight;
                                                                   applyEditorViewScaleState();
                                                               });
    newFloatingPianoRoll->setNotePreviewCallbacks([this] (int pitch, int velocity)
                                                  {
                                                      previewSelectedTrackMidiNoteOn(pitch, velocity);
                                                  },
                                                  [this] (int pitch, int velocity)
                                                  {
                                                      previewSelectedTrackMidiNoteOff(pitch, velocity);
                                                  },
                                                  [this]
                                                  {
                                                      stopSelectedTrackMidiPreview();
                                                  });
    floatingPianoRollWorkspace = newFloatingPianoRoll;
    pianoRollWindow->setContentOwned(newFloatingPianoRoll, true);
    pianoRollWindow->setBounds(displayArea.getX() + 146,
                               displayArea.getY() + 262,
                               juce::jmin(1140, displayArea.getWidth() - 200),
                               juce::jmin(720, displayArea.getHeight() - 260));
    pianoRollWindow->setVisible(pianoRollWindowVisible);

    virtualPianoWindow = std::make_unique<FloatingPanelWindow>("Virtual Piano");
    attachVirtualPianoShortcutHandler(virtualPianoWindow.get());
    virtualPianoWindow->onClosePressed = [this]
    {
        virtualPianoWindowVisible = false;
        persistWindowVisibilityState();
    };
    virtualPianoWindow->onKeyPressed = [this] (const juce::KeyPress& key)
    {
        return tryHandleVirtualPianoShortcut(key);
    };
    auto* newVirtualPianoWindowContent = new VirtualPianoWindowComponent([this] (int pitch)
                                                                         {
                                                                             insertVirtualKeyboardNote(pitch, false);
                                                                         },
                                                                         [this] (const juce::KeyPress& key)
                                                                         {
                                                                             return tryHandleVirtualPianoShortcut(key);
                                                                         });
    virtualPianoWindowContent = newVirtualPianoWindowContent;
    virtualPianoWindow->setContentOwned(newVirtualPianoWindowContent, true);
    virtualPianoWindow->setBounds(displayArea.getX() + 188,
                                  displayArea.getY() + 286,
                                  juce::jmin(920, displayArea.getWidth() - 240),
                                  290);
    virtualPianoWindow->setVisible(virtualPianoWindowVisible);

    activityLogWindow = std::make_unique<FloatingPanelWindow>("Activity Log");
    attachVirtualPianoShortcutHandler(activityLogWindow.get());
    activityLogWindow->onClosePressed = [this]
    {
        activityLogWindowVisible = false;
        persistWindowVisibilityState();
    };
    auto* newActivityLogWindowContent = new ActivityLogWindowComponent([this]
                                                                       {
                                                                           return activityLogTextSnapshot();
                                                                       },
                                                                       [this]
                                                                       {
                                                                           clearActivityLog();
                                                                       });
    activityLogWindowContent = newActivityLogWindowContent;
    activityLogWindow->setContentOwned(newActivityLogWindowContent, true);
    activityLogWindow->setBounds(displayArea.getX() + 168,
                                 displayArea.getY() + 108,
                                 juce::jmin(920, displayArea.getWidth() - 220),
                                 juce::jmin(680, displayArea.getHeight() - 180));
    activityLogWindow->setVisible(activityLogWindowVisible);

    panelsWindow = std::make_unique<FloatingPanelWindow>("Panels");
    attachVirtualPianoShortcutHandler(panelsWindow.get());
    panelsWindow->onClosePressed = [this]
    {
        panelsWindowVisible = false;
        persistWindowVisibilityState();
    };
    auto* newPanelsWindowContent = new PanelsWindowComponent([this] () -> const ProjectState& { return documentState.project; },
                                                             [this] () { return getSelectedMidiSectionIndex(); },
                                                             [this] (int sectionIndex, bool focusEditor)
                                                             {
                                                                 setSelectedMidiSectionIndex(sectionIndex, true);
                                                                 if (focusEditor)
                                                                     focusMidiSectionInPianoRoll(sectionIndex);
                                                             },
                                                             [this] { return getSelectedTrackIndex(); },
                                                             [this] (int trackIndex,
                                                                     const TrackState& track,
                                                                     bool undoable,
                                                                     const juce::String& actionName)
                                                             {
                                                                 if (undoable)
                                                                     applyTrackStateEdit(trackIndex, track, actionName);
                                                                 else
                                                                     replaceTrackStateNoUndo(trackIndex, track);
                                                             },
                                                             [this] (const ProjectState& project,
                                                                     bool undoable,
                                                                     const juce::String& actionName)
                                                             {
                                                                 if (undoable)
                                                                     applyProjectStateEdit(project, actionName);
                                                                 else
                                                                     setProjectStateInternal(project);
                                                              },
                                                              [this] { return editorToolMode; },
                                                              [this] (EditorToolMode mode) { setEditorToolMode(mode); });
    newPanelsWindowContent->setNotePreviewCallbacks([this] (int pitch, int velocity)
                                                    {
                                                        previewSelectedTrackMidiNoteOn(pitch, velocity);
                                                    },
                                                    [this] (int pitch, int velocity)
                                                    {
                                                        previewSelectedTrackMidiNoteOff(pitch, velocity);
                                                    },
                                                    [this]
                                                    {
                                                        stopSelectedTrackMidiPreview();
                                                    });
    panelsWindowContent = newPanelsWindowContent;
    panelsWindow->setContentOwned(newPanelsWindowContent, true);
    panelsWindow->setBounds(displayArea.getX() + juce::jmax(40, displayArea.getWidth() / 8),
                            displayArea.getY() + juce::jmax(84, displayArea.getHeight() / 8),
                            juce::jmin(1120, displayArea.getWidth() - 80),
                            juce::jmin(760, displayArea.getHeight() - 120));
    panelsWindow->setVisible(panelsWindowVisible);

    audioSettingsWindow = std::make_unique<FloatingPanelWindow>("Audio Settings");
    attachVirtualPianoShortcutHandler(audioSettingsWindow.get());
    auto* newAudioSettingsPanel = new AudioSettingsPanelComponent([this] (const juce::String& audioDeviceType,
                                                                          const juce::String& audioDeviceName,
                                                                          int sampleRate,
                                                                          int bufferSize)
                                                                 {
                                                                     return applyAudioDeviceSettings(audioDeviceType,
                                                                                                     audioDeviceName,
                                                                                                     sampleRate,
                                                                                                     bufferSize);
                                                                 },
                                                                 [this] { refreshAudioSettingsFromHost(true); });
    audioSettingsPanel = newAudioSettingsPanel;
    audioSettingsWindow->setContentOwned(newAudioSettingsPanel, true);
    audioSettingsWindow->setBounds(displayArea.getCentreX() - 260,
                                   displayArea.getCentreY() - 170,
                                   520,
                                   300);
    audioSettingsWindow->setVisible(false);

    vstFolderManagerWindow = std::make_unique<FloatingPanelWindow>("VST Folder Manager");
    attachVirtualPianoShortcutHandler(vstFolderManagerWindow.get());
    auto* newVstFolderManagerPanel = new VstFolderManagerComponent([this]
                                                                   {
                                                                       return defaultVstFolderPath();
                                                                   },
                                                                   [this]
                                                                   {
                                                                       return userManagedVstFolderPaths();
                                                                   },
                                                                   [this]
                                                                   {
                                                                       promptAddUserVstFolder();
                                                                   },
                                                                   [this] (const juce::String& folderPath)
                                                                   {
                                                                       removeUserVstFolder(folderPath);
                                                                   },
                                                                   [this]
                                                                   {
                                                                       refreshRackCatalog();
                                                                   });
    vstFolderManagerPanel = newVstFolderManagerPanel;
    vstFolderManagerWindow->setContentOwned(newVstFolderManagerPanel, true);
    vstFolderManagerWindow->setBounds(displayArea.getCentreX() - 320,
                                      displayArea.getCentreY() - 190,
                                      640,
                                      380);
    vstFolderManagerWindow->setVisible(false);

    refreshFloatingWindows();
}

void StudioShellComponent::refreshFloatingWindows(bool includeEditorRefresh)
{
    const ScopedAppProfileSample profileSample(AppProfileSection::refreshFloatingWindows);
    const bool deferHostUiQueries = (loadedRackEditorOpen || hasOpenRackEditorSessions())
        && !rackPreviewRunning
        && !projectPreviewRunning;

    if (transportPanel != nullptr && transportWindow != nullptr && transportWindow->isVisible())
    {
        transportPanel->refreshFromState(documentState.project,
                                         getSelectedTrack() != nullptr,
                                         rackPreviewRunning,
                                         projectPreviewRunning,
                                         statusLabel.getText(),
                                         transportCpuUsagePercent,
                                         transportMasterPeakLeft,
                                         transportMasterPeakRight);
    }

    if (floatingAudioWorkspace != nullptr && audioWindow != nullptr && audioWindow->isVisible())
    {
        floatingAudioWorkspace->refreshFromModel(documentState.project,
                                                 getSelectedTrack() != nullptr,
                                                 rackPreviewRunning,
                                                 projectPreviewRunning,
                                                 statusLabel.getText(),
                                                 transportCpuUsagePercent,
                                                 transportMasterPeakLeft,
                                                 transportMasterPeakRight,
                                                 deferHostUiQueries);
    }

    if (floatingMixerComponent != nullptr && mixerWindow != nullptr && mixerWindow->isVisible())
    {
        if (includeEditorRefresh)
            floatingMixerComponent->refreshFromModel();
        floatingMixerComponent->refreshMeters();
    }

    if (includeEditorRefresh && floatingTracksWorkspace != nullptr && tracksWindow != nullptr && tracksWindow->isVisible())
        floatingTracksWorkspace->refreshFromModel();

    if (includeEditorRefresh && floatingRackBrowser != nullptr && rackBrowserWindow != nullptr && rackBrowserWindow->isVisible())
        floatingRackBrowser->refreshFromModel();

    if (activityLogWindowContent != nullptr && activityLogWindow != nullptr && activityLogWindow->isVisible())
        activityLogWindowContent->refreshFromModel();

    if (includeEditorRefresh && floatingRenderManager != nullptr && renderManagerWindow != nullptr && renderManagerWindow->isVisible())
        floatingRenderManager->refreshFromModel();

    if (includeEditorRefresh && floatingArrangementOverview != nullptr && arrangementWindow != nullptr && arrangementWindow->isVisible())
        floatingArrangementOverview->refreshFromModel();

    if (includeEditorRefresh && floatingAutomationEditor != nullptr && automationWindow != nullptr && automationWindow->isVisible())
        floatingAutomationEditor->refreshFromModel();

    if (includeEditorRefresh && floatingSampleWorkspace != nullptr && samplesWindow != nullptr && samplesWindow->isVisible())
        floatingSampleWorkspace->refreshFromModel();

    if (includeEditorRefresh && floatingPianoRollWorkspace != nullptr && pianoRollWindow != nullptr && pianoRollWindow->isVisible())
        floatingPianoRollWorkspace->refreshFromModel();

    if (includeEditorRefresh && panelsWindowContent != nullptr && panelsWindow != nullptr && panelsWindow->isVisible())
        panelsWindowContent->refreshFromModel();

    if (includeEditorRefresh && audioSettingsWindow != nullptr && audioSettingsWindow->isVisible() && !deferHostUiQueries)
        refreshAudioSettingsFromHost(false);

    if (includeEditorRefresh && vstFolderManagerPanel != nullptr && vstFolderManagerWindow != nullptr && vstFolderManagerWindow->isVisible())
        vstFolderManagerPanel->refreshFromModel();
}

void StudioShellComponent::jumpPlayheadToStart()
{
    if (documentState.project.playheadTick == 0)
        return;

    auto updatedProject = documentState.project;
    updatedProject.playheadTick = 0;
    updatedProject.recalculateTimeFields();
    applyProjectStateEdit(updatedProject, "Go To Start");
    if (projectPreviewRunning && nativeVstHost.isReady())
        nativeVstHost.updateAudioEngineTransport(documentState.project);
    statusLabel.setText("Moved playhead to start.", juce::dontSendNotification);
}

void StudioShellComponent::setTransportTempo(int bpm)
{
    const auto newTempo = juce::jlimit(20, 300, bpm);
    if (documentState.project.bpm == newTempo)
        return;

    auto updatedProject = documentState.project;
    updatedProject.bpm = newTempo;
    updatedProject.recalculateTimeFields();
    applyProjectStateEdit(updatedProject, "Change Tempo");
    if (projectPreviewRunning && nativeVstHost.isReady())
        nativeVstHost.updateAudioEngineTransport(documentState.project);
    statusLabel.setText("Updated tempo to " + juce::String(newTempo) + " BPM.", juce::dontSendNotification);
}

void StudioShellComponent::setTransportLoopEnabled(bool enabled)
{
    if (documentState.project.loopEnabled == enabled)
        return;

    auto updatedProject = documentState.project;
    updatedProject.loopEnabled = enabled;
    applyProjectStateEdit(updatedProject, "Toggle Loop");
    if (projectPreviewRunning && nativeVstHost.isReady())
        nativeVstHost.updateAudioEngineTransport(documentState.project);
    statusLabel.setText("Updated loop setting.", juce::dontSendNotification);
}

void StudioShellComponent::setTransportMetronomeEnabled(bool enabled)
{
    if (documentState.project.metronomeEnabled == enabled)
        return;

    auto updatedProject = documentState.project;
    updatedProject.metronomeEnabled = enabled;
    applyProjectStateEdit(updatedProject, "Toggle Metronome");
    if (projectPreviewRunning && nativeVstHost.isReady())
        nativeVstHost.updateAudioEngineTransport(documentState.project);
    statusLabel.setText("Updated metronome setting.", juce::dontSendNotification);
}

void StudioShellComponent::selectAllNotesFromMenu()
{
    if (pianoRoll != nullptr && pianoRoll->selectAllNotes())
        statusLabel.setText("Selected all notes.", juce::dontSendNotification);
}

void StudioShellComponent::copyNotesFromMenu()
{
    if (pianoRoll != nullptr && pianoRoll->copySelected())
        statusLabel.setText("Copied selected notes.", juce::dontSendNotification);
}

void StudioShellComponent::cutNotesFromMenu()
{
    if (pianoRoll != nullptr && pianoRoll->cutSelected())
        statusLabel.setText("Cut selected notes.", juce::dontSendNotification);
}

void StudioShellComponent::deleteNotesFromMenu()
{
    if (pianoRoll != nullptr && pianoRoll->deleteSelected())
        statusLabel.setText("Deleted selected notes.", juce::dontSendNotification);
}

void StudioShellComponent::duplicateNotesFromMenu()
{
    if (pianoRoll != nullptr && pianoRoll->duplicateSelectedByGrid())
        statusLabel.setText("Duplicated selected notes.", juce::dontSendNotification);
}

void StudioShellComponent::pasteNotesFromMenu()
{
    if (pianoRoll != nullptr && pianoRoll->pasteClipboard())
        statusLabel.setText("Pasted notes.", juce::dontSendNotification);
}

void StudioShellComponent::focusArrangementPanel()
{
    if (arrangementWindow != nullptr)
    {
        setArrangementWindowVisible(true);
        if (floatingArrangementOverview != nullptr)
            floatingArrangementOverview->grabKeyboardFocus();
        return;
    }

    if (panelsWindowContent != nullptr)
    {
        setPanelsWindowVisible(true);
        panelsWindowContent->showArrangementTab();
        return;
    }

    if (arrangementOverview != nullptr)
        arrangementOverview->grabKeyboardFocus();
}

void StudioShellComponent::focusAutomationPanel()
{
    if (automationWindow != nullptr)
    {
        setAutomationWindowVisible(true);
        if (floatingAutomationEditor != nullptr)
            floatingAutomationEditor->grabKeyboardFocus();
        return;
    }

    if (panelsWindowContent != nullptr)
    {
        setPanelsWindowVisible(true);
        panelsWindowContent->showAutomationTab();
        return;
    }

    if (automationEditor != nullptr)
        automationEditor->grabKeyboardFocus();
}

void StudioShellComponent::focusSamplesPanel()
{
    if (samplesWindow != nullptr)
    {
        setSamplesWindowVisible(true);
        if (floatingSampleWorkspace != nullptr)
            floatingSampleWorkspace->focusLibrary();
        return;
    }

    if (panelsWindowContent != nullptr)
    {
        setPanelsWindowVisible(true);
        panelsWindowContent->showSamplesTab();
        return;
    }

    sampleAssetList.grabKeyboardFocus();
}

void StudioShellComponent::focusTracksPanel()
{
    if (tracksWindow != nullptr)
    {
        setTracksWindowVisible(true);
        if (floatingTracksWorkspace != nullptr)
            floatingTracksWorkspace->focusTrackList();
        return;
    }

    trackTable.grabKeyboardFocus();
}

void StudioShellComponent::focusRackBrowserPanel()
{
    if (rackBrowserWindow != nullptr)
    {
        setRackBrowserWindowVisible(true);
        if (floatingRackBrowser != nullptr)
        {
            floatingRackBrowser->selectAssignedRack();
            floatingRackBrowser->focusRackList();
        }
        return;
    }
}

void StudioShellComponent::focusRenderManagerPanel()
{
    if (renderManagerWindow != nullptr)
    {
        setRenderManagerWindowVisible(true);
        if (floatingRenderManager != nullptr)
            floatingRenderManager->focusTrackList();
        return;
    }
}

void StudioShellComponent::focusAudioPanel()
{
    if (audioWindow != nullptr)
    {
        setAudioWindowVisible(true);
        if (floatingAudioWorkspace != nullptr)
            floatingAudioWorkspace->focusWorkspace();
        return;
    }

    showAudioSettingsWindow();
}

void StudioShellComponent::focusMixerPanel()
{
    if (mixerWindow != nullptr)
    {
        setMixerWindowVisible(true);
        if (floatingMixerComponent != nullptr)
            floatingMixerComponent->grabKeyboardFocus();
        return;
    }

    if (mixerComponent != nullptr)
        mixerComponent->grabKeyboardFocus();
}

void StudioShellComponent::focusPianoRollPanel()
{
    if (pianoRollWindow != nullptr)
    {
        setPianoRollWindowVisible(true);
        if (floatingPianoRollWorkspace != nullptr)
            floatingPianoRollWorkspace->focusEditor();
        return;
    }

    if (panelsWindowContent != nullptr)
    {
        setPanelsWindowVisible(true);
        panelsWindowContent->showPianoRollTab();
        return;
    }

    if (pianoRoll != nullptr)
        pianoRoll->grabKeyboardFocus();
}

void StudioShellComponent::focusVirtualPianoPanel()
{
    if (virtualPianoWindow == nullptr)
        return;

    setVirtualPianoWindowVisible(true);
    if (virtualPianoWindowContent != nullptr)
        virtualPianoWindowContent->focusKeyboard();
}

void StudioShellComponent::showAudioSettingsWindow()
{
    if (audioSettingsWindow == nullptr)
        return;

    audioSettingsWindow->setVisible(true);
    audioSettingsWindow->toFront(true);
    refreshAudioSettingsFromHost(true);
}

void StudioShellComponent::showVstFolderManagerWindow()
{
    if (vstFolderManagerWindow == nullptr)
        return;

    if (vstFolderManagerPanel != nullptr)
        vstFolderManagerPanel->refreshFromModel();

    vstFolderManagerWindow->setVisible(true);
    vstFolderManagerWindow->toFront(false);
}

void StudioShellComponent::refreshAudioSettingsFromHost(bool forceCreate)
{
    if (audioSettingsPanel == nullptr)
        return;
    if (!forceCreate && (audioSettingsWindow == nullptr || !audioSettingsWindow->isVisible()))
        return;

    const auto createResult = nativeVstHost.ensureCreated();
    if (createResult.failed())
    {
        audioSettingsPanel->setStatusMessage(createResult.getErrorMessage(), true);
        return;
    }

    NativeVstHostSession::AudioDeviceSnapshot snapshot;
    const auto snapshotResult = nativeVstHost.queryAudioDeviceSnapshot(snapshot);
    if (snapshotResult.failed())
    {
        audioSettingsPanel->setStatusMessage(snapshotResult.getErrorMessage(), true);
        return;
    }

    audioSettingsPanel->applyAudioDeviceSnapshot(snapshot);
}

juce::Result StudioShellComponent::applyAudioDeviceSettings(const juce::String& audioDeviceType,
                                                            const juce::String& audioDeviceName,
                                                            int sampleRate,
                                                            int bufferSize)
{
    const auto createResult = nativeVstHost.ensureCreated();
    if (createResult.failed())
        return createResult;

    if (rackPreviewRunning || projectPreviewRunning)
        stopRackPreview();

    const auto result = nativeVstHost.configureAudioDevice(audioDeviceType,
                                                           audioDeviceName,
                                                           static_cast<double>(sampleRate),
                                                           bufferSize);
    if (result.failed())
    {
        refreshAudioSettingsFromHost(true);
        return result;
    }

    refreshAudioSettingsFromHost(true);

    audioEngineStateValid = false;
    audioEngineStateDirty = true;
    statusLabel.setText("Updated native audio device settings.", juce::dontSendNotification);
    refreshFloatingWindows();
    return juce::Result::ok();
}

void StudioShellComponent::setTransportWindowVisible(bool shouldBeVisible)
{
    transportWindowVisible = shouldBeVisible;
    persistWindowVisibilityState();
    if (transportWindow == nullptr)
        return;

    transportWindow->setVisible(shouldBeVisible);
    if (shouldBeVisible)
    {
        refreshFloatingWindows();
        transportWindow->toFront(true);
    }
}

bool StudioShellComponent::isTransportWindowVisible() const noexcept
{
    return transportWindowVisible && transportWindow != nullptr && transportWindow->isVisible();
}

void StudioShellComponent::setMixerWindowVisible(bool shouldBeVisible)
{
    mixerWindowVisible = shouldBeVisible;
    persistWindowVisibilityState();
    if (mixerWindow == nullptr)
        return;

    mixerWindow->setVisible(shouldBeVisible);
    if (shouldBeVisible)
    {
        refreshFloatingWindows();
        mixerWindow->toFront(true);
    }
}

bool StudioShellComponent::isMixerWindowVisible() const noexcept
{
    return mixerWindowVisible && mixerWindow != nullptr && mixerWindow->isVisible();
}

void StudioShellComponent::setAudioWindowVisible(bool shouldBeVisible)
{
    audioWindowVisible = shouldBeVisible;
    persistWindowVisibilityState();
    if (audioWindow == nullptr)
        return;

    audioWindow->setVisible(shouldBeVisible);
    if (shouldBeVisible)
    {
        refreshFloatingWindows();
        audioWindow->toFront(true);
    }
}

bool StudioShellComponent::isAudioWindowVisible() const noexcept
{
    return audioWindowVisible && audioWindow != nullptr && audioWindow->isVisible();
}

void StudioShellComponent::setPanelsWindowVisible(bool shouldBeVisible)
{
    panelsWindowVisible = shouldBeVisible;
    persistWindowVisibilityState();
    if (panelsWindow == nullptr)
        return;

    panelsWindow->setVisible(shouldBeVisible);
    if (shouldBeVisible)
    {
        refreshFloatingWindows();
        panelsWindow->toFront(true);
    }
}

bool StudioShellComponent::isPanelsWindowVisible() const noexcept
{
    return panelsWindowVisible && panelsWindow != nullptr && panelsWindow->isVisible();
}

void StudioShellComponent::setTracksWindowVisible(bool shouldBeVisible)
{
    tracksWindowVisible = shouldBeVisible;
    persistWindowVisibilityState();
    if (tracksWindow == nullptr)
        return;

    tracksWindow->setVisible(shouldBeVisible);
    if (shouldBeVisible)
    {
        refreshFloatingWindows();
        tracksWindow->toFront(true);
    }
}

bool StudioShellComponent::isTracksWindowVisible() const noexcept
{
    return tracksWindowVisible && tracksWindow != nullptr && tracksWindow->isVisible();
}

void StudioShellComponent::setRackBrowserWindowVisible(bool shouldBeVisible)
{
    rackBrowserWindowVisible = shouldBeVisible;
    persistWindowVisibilityState();
    if (rackBrowserWindow == nullptr)
        return;

    rackBrowserWindow->setVisible(shouldBeVisible);
    if (shouldBeVisible)
    {
        refreshFloatingWindows();
        rackBrowserWindow->toFront(true);
    }
}

bool StudioShellComponent::isRackBrowserWindowVisible() const noexcept
{
    return rackBrowserWindowVisible && rackBrowserWindow != nullptr && rackBrowserWindow->isVisible();
}

void StudioShellComponent::setRenderManagerWindowVisible(bool shouldBeVisible)
{
    renderManagerWindowVisible = shouldBeVisible;
    persistWindowVisibilityState();
    if (renderManagerWindow == nullptr)
        return;

    renderManagerWindow->setVisible(shouldBeVisible);
    if (shouldBeVisible)
    {
        refreshFloatingWindows();
        renderManagerWindow->toFront(true);
    }
}

bool StudioShellComponent::isRenderManagerWindowVisible() const noexcept
{
    return renderManagerWindowVisible && renderManagerWindow != nullptr && renderManagerWindow->isVisible();
}

void StudioShellComponent::setArrangementWindowVisible(bool shouldBeVisible)
{
    arrangementWindowVisible = shouldBeVisible;
    persistWindowVisibilityState();
    if (arrangementWindow == nullptr)
        return;

    arrangementWindow->setVisible(shouldBeVisible);
    if (shouldBeVisible)
    {
        refreshFloatingWindows();
        arrangementWindow->toFront(true);
    }
}

bool StudioShellComponent::isArrangementWindowVisible() const noexcept
{
    return arrangementWindowVisible && arrangementWindow != nullptr && arrangementWindow->isVisible();
}

void StudioShellComponent::setAutomationWindowVisible(bool shouldBeVisible)
{
    automationWindowVisible = shouldBeVisible;
    persistWindowVisibilityState();
    if (automationWindow == nullptr)
        return;

    automationWindow->setVisible(shouldBeVisible);
    if (shouldBeVisible)
    {
        refreshFloatingWindows();
        automationWindow->toFront(true);
    }
}

bool StudioShellComponent::isAutomationWindowVisible() const noexcept
{
    return automationWindowVisible && automationWindow != nullptr && automationWindow->isVisible();
}

void StudioShellComponent::setSamplesWindowVisible(bool shouldBeVisible)
{
    samplesWindowVisible = shouldBeVisible;
    persistWindowVisibilityState();
    if (samplesWindow == nullptr)
        return;

    samplesWindow->setVisible(shouldBeVisible);
    if (shouldBeVisible)
    {
        refreshFloatingWindows();
        samplesWindow->toFront(true);
    }
}

bool StudioShellComponent::isSamplesWindowVisible() const noexcept
{
    return samplesWindowVisible && samplesWindow != nullptr && samplesWindow->isVisible();
}

void StudioShellComponent::setPianoRollWindowVisible(bool shouldBeVisible)
{
    pianoRollWindowVisible = shouldBeVisible;
    persistWindowVisibilityState();
    if (pianoRollWindow == nullptr)
        return;

    pianoRollWindow->setVisible(shouldBeVisible);
    if (shouldBeVisible)
    {
        refreshFloatingWindows();
        pianoRollWindow->toFront(true);
    }
}

bool StudioShellComponent::isPianoRollWindowVisible() const noexcept
{
    return pianoRollWindowVisible && pianoRollWindow != nullptr && pianoRollWindow->isVisible();
}

void StudioShellComponent::setVirtualPianoWindowVisible(bool shouldBeVisible)
{
    virtualPianoWindowVisible = shouldBeVisible;
    persistWindowVisibilityState();
    if (virtualPianoWindow == nullptr)
        return;

    virtualPianoWindow->setVisible(shouldBeVisible);
    if (shouldBeVisible)
    {
        virtualPianoWindow->toFront(true);
        if (virtualPianoWindowContent != nullptr)
            virtualPianoWindowContent->focusKeyboard();
    }
}

bool StudioShellComponent::isVirtualPianoWindowVisible() const noexcept
{
    return virtualPianoWindowVisible && virtualPianoWindow != nullptr && virtualPianoWindow->isVisible();
}

void StudioShellComponent::setActivityLogWindowVisible(bool shouldBeVisible)
{
    activityLogWindowVisible = shouldBeVisible;
    persistWindowVisibilityState();
    if (activityLogWindow == nullptr)
        return;

    activityLogWindow->setVisible(shouldBeVisible);
    if (shouldBeVisible)
    {
        refreshFloatingWindows();
        activityLogWindow->toFront(true);
    }
}

bool StudioShellComponent::isActivityLogWindowVisible() const noexcept
{
    return activityLogWindowVisible && activityLogWindow != nullptr && activityLogWindow->isVisible();
}

void StudioShellComponent::createNewProject()
{
    if (rackPreviewRunning && nativeVstHost.isReady())
        nativeVstHost.stopAudioEngine();
    if (projectPreviewRunning && nativeVstHost.isReady())
        nativeVstHost.stopAudioEngine();
    closeAllRackEditorSessions();
    resetRackHostTracking();
    rackPreviewRunning = false;
    projectPreviewRunning = false;
    playbackUiTickCounter = 0;
    pendingLiveRackParameterEngineSyncTrack = -1;
    audioEngineStateValid = false;
    audioEngineStateDirty = true;

    documentState = makeDefaultProjectFile();
    syncBundledRackCatalogInProject();
    currentProjectFile = {};
    clearDirty();
    undoManager.clearUndoHistory();
    refreshUi();
    trackTable.selectRow(0);
    statusLabel.setText("Created new native project.", juce::dontSendNotification);
    appendActivityLog("Project", "Created new native project.");
}

void StudioShellComponent::promptOpenProject()
{
    activeFileChooser = std::make_unique<juce::FileChooser>("Open Mutagen project",
                                                            currentProjectFile.existsAsFile() ? currentProjectFile : juce::File(),
                                                            kProjectFileWildcard);
    activeFileChooser->launchAsync(juce::FileBrowserComponent::openMode | juce::FileBrowserComponent::canSelectFiles,
                                   [safeThis = juce::Component::SafePointer<StudioShellComponent>(this)](const juce::FileChooser& chooser)
                                   {
                                       const auto selected = chooser.getResult();
                                       if (safeThis == nullptr)
                                           return;

                                       safeThis->activeFileChooser.reset();
                                       if (selected == juce::File())
                                           return;

                                       const auto result = safeThis->openProjectFile(selected);
                                       if (result.failed())
                                       {
                                           juce::AlertWindow::showMessageBoxAsync(juce::AlertWindow::WarningIcon,
                                                                                  "Open Project Failed",
                                                                                  result.getErrorMessage());
                                       }
                                   });
}

void StudioShellComponent::saveProject()
{
    if (currentProjectFile == juce::File())
    {
        saveProjectAs();
        return;
    }

    for (const auto& session : rackEditorSessions)
    {
        if (session == nullptr || !session->editorOpen)
            continue;

        NativeVstHostSession::RackParameterSnapshot rackSnapshot;
        if (nativeVstHost.queryAudioEngineTrackParameterSnapshot(session->trackIndex, rackSnapshot).wasOk())
        {
            syncTrackRackParametersFromValues(session->trackIndex,
                                              rackSnapshot.parameterValues,
                                              true);
        }
    }

    const auto result = saveProjectFile(currentProjectFile, documentState);
    if (result.failed())
    {
        juce::AlertWindow::showMessageBoxAsync(juce::AlertWindow::WarningIcon,
                                               "Save Project Failed",
                                               result.getErrorMessage());
        return;
    }

    clearDirty();
    refreshUi();
    statusLabel.setText("Saved project: " + currentProjectFile.getFileName(), juce::dontSendNotification);
    appendActivityLog("Project", "Saved project\n" + currentProjectFile.getFullPathName());
}

void StudioShellComponent::saveProjectAs()
{
    activeFileChooser = std::make_unique<juce::FileChooser>("Save Mutagen project",
                                                            suggestProjectFile(),
                                                            kProjectFileWildcard);
    activeFileChooser->launchAsync(juce::FileBrowserComponent::saveMode
                                       | juce::FileBrowserComponent::canSelectFiles
                                       | juce::FileBrowserComponent::warnAboutOverwriting,
                                   [safeThis = juce::Component::SafePointer<StudioShellComponent>(this)](const juce::FileChooser& chooser)
                                   {
                                       const auto selected = chooser.getResult();
                                       if (safeThis == nullptr)
                                           return;

                                       safeThis->activeFileChooser.reset();
                                       if (selected == juce::File())
                                           return;

                                       safeThis->currentProjectFile = safeThis->ensureProjectSuffix(selected);
                                       safeThis->saveProject();
                                   });
}

void StudioShellComponent::addTrack()
{
    auto updatedProject = documentState.project;
    TrackState track;
    track.name = "Track " + juce::String(static_cast<int>(updatedProject.tracks.size()) + 1);
    track.midiChannel = static_cast<int>(updatedProject.tracks.size()) % 16;
    track.colorHex = defaultTrackColour(static_cast<int>(updatedProject.tracks.size())).toDisplayString(false);
    updatedProject.tracks.push_back(std::move(track));
    applyProjectStateEdit(updatedProject, "Add Track");
    trackTable.selectRow(static_cast<int>(documentState.project.tracks.size()) - 1);
    statusLabel.setText("Added native track.", juce::dontSendNotification);
}

void StudioShellComponent::duplicateSelectedTrack()
{
    const auto selected = getSelectedTrackIndex();
    if (selected < 0)
        return;

    auto updatedProject = documentState.project;
    auto copy = updatedProject.tracks[static_cast<size_t>(selected)];
    copy.name = copy.name + " Copy";
    if (copy.colorHex.trim().isEmpty())
        copy.colorHex = defaultTrackColour(selected + 1).toDisplayString(false);
    updatedProject.tracks.insert(updatedProject.tracks.begin() + selected + 1, std::move(copy));

    for (auto& section : updatedProject.midiSections)
    {
        if (section.trackIndex > selected)
            ++section.trackIndex;
    }

    std::vector<MidiSection> duplicatedSections;
    for (const auto& section : documentState.project.midiSections)
    {
        if (section.trackIndex != selected)
            continue;

        auto duplicateSection = section;
        duplicateSection.trackIndex = selected + 1;
        duplicatedSections.push_back(std::move(duplicateSection));
    }

    updatedProject.midiSections.insert(updatedProject.midiSections.end(),
                                       duplicatedSections.begin(),
                                       duplicatedSections.end());
    applyProjectStateEdit(updatedProject, "Duplicate Track");
    trackTable.selectRow(selected + 1);
    statusLabel.setText("Duplicated track.", juce::dontSendNotification);
}

void StudioShellComponent::removeSelectedTrack()
{
    const auto selected = getSelectedTrackIndex();
    if (selected < 0)
        return;

    auto updatedProject = documentState.project;
    updatedProject.tracks.erase(updatedProject.tracks.begin() + selected);
    if (updatedProject.tracks.empty())
        updatedProject.tracks.push_back(TrackState{});

    updatedProject.midiSections.erase(std::remove_if(updatedProject.midiSections.begin(),
                                                     updatedProject.midiSections.end(),
                                                     [selected] (const MidiSection& section)
                                                     {
                                                         return section.trackIndex == selected;
                                                     }),
                                      updatedProject.midiSections.end());
    for (auto& section : updatedProject.midiSections)
    {
        if (section.trackIndex > selected)
            --section.trackIndex;
    }

    applyProjectStateEdit(updatedProject, "Remove Track");
    trackTable.selectRow(juce::jlimit(0, static_cast<int>(documentState.project.tracks.size()) - 1, selected));
    statusLabel.setText("Removed track.", juce::dontSendNotification);
}

void StudioShellComponent::handleTempoChanged()
{
    setTransportTempo(juce::roundToInt(tempoSlider.getValue()));
}

void StudioShellComponent::handlePatternBarsChanged()
{
    const auto tickLength = normaliseSequenceTickLength(patternBarsBox.getSelectedId());
    auto updatedProject = documentState.project;
    updatedProject.defaultPatternTicks = tickLength;
    if (juce::isPositiveAndBelow(selectedMidiSectionIndex, static_cast<int>(updatedProject.midiSections.size())))
    {
        const auto patternId = updatedProject.midiSections[static_cast<size_t>(selectedMidiSectionIndex)].patternId;
        if (auto* pattern = findMidiPattern(updatedProject, patternId))
        {
            if (pattern->lengthTicks != tickLength)
                pattern->lengthTicks = tickLength;
        }
    }

    if (fingerprint(updatedProject) == fingerprint(documentState.project))
        return;

    updatedProject.recalculateTimeFields();
    applyProjectStateEdit(updatedProject, "Change Pattern Size");
}

void StudioShellComponent::handleKeyQuantizeChanged()
{
    const auto* option = findKeyQuantizeOptionById(keyQuantizeBox.getSelectedId());
    if (option == nullptr)
        return;

    auto updatedProject = documentState.project;
    updatedProject.keyQuantizeRoot = option->root;
    updatedProject.keyQuantizeScale = option->scaleId;

    if (fingerprint(updatedProject) == fingerprint(documentState.project))
        return;

    updatedProject.recalculateTimeFields();
    applyProjectStateEdit(updatedProject, "Change Key Quantize");
}

void StudioShellComponent::handleArrangementSnapChanged()
{
    const auto tickLength = normaliseSequenceTickLength(arrangementSnapBox.getSelectedId());
    auto updatedProject = documentState.project;
    updatedProject.arrangementSnapTicks = tickLength;

    if (fingerprint(updatedProject) == fingerprint(documentState.project))
        return;

    updatedProject.recalculateTimeFields();
    applyProjectStateEdit(updatedProject, "Change Sequencer Snap");
}

void StudioShellComponent::handleArrangementZoomChanged()
{
    arrangementZoomPixelsPerBar = static_cast<float>(arrangementZoomSlider.getValue());
    applyEditorViewScaleState();
}

void StudioShellComponent::handleArrangementLaneHeightChanged()
{
    arrangementLaneHeightPixels = static_cast<float>(arrangementLaneHeightSlider.getValue());
    applyEditorViewScaleState();
}

void StudioShellComponent::handlePianoRollZoomChanged()
{
    pianoRollZoomPixelsPerBeat = static_cast<float>(pianoRollZoomSlider.getValue());
    applyEditorViewScaleState();
}

void StudioShellComponent::handlePianoRollRowHeightChanged()
{
    pianoRollRowHeightPixels = static_cast<float>(pianoRollRowHeightSlider.getValue());
    applyEditorViewScaleState();
}

void StudioShellComponent::promptImportMidi()
{
    activeFileChooser = std::make_unique<juce::FileChooser>("Import MIDI",
                                                            juce::File(),
                                                            kMidiFileWildcard);
    activeFileChooser->launchAsync(juce::FileBrowserComponent::openMode | juce::FileBrowserComponent::canSelectFiles,
                                   [safeThis = juce::Component::SafePointer<StudioShellComponent>(this)](const juce::FileChooser& chooser)
                                   {
                                       const auto selected = chooser.getResult();
                                       if (safeThis == nullptr)
                                           return;

                                       safeThis->activeFileChooser.reset();
                                       if (selected == juce::File())
                                           return;

                                       auto importedProject = safeThis->documentState.project;
                                       const auto result = importMidiFileToProject(selected, importedProject);
                                       if (result.failed())
                                       {
                                           juce::AlertWindow::showMessageBoxAsync(juce::AlertWindow::WarningIcon,
                                                                                  "Import MIDI Failed",
                                                                                  result.getErrorMessage());
                                           return;
                                       }

                                       safeThis->applyProjectStateEdit(importedProject, "Import MIDI");
                                       safeThis->trackTable.selectRow(0);
                                       safeThis->statusLabel.setText("Imported MIDI: " + selected.getFileName(), juce::dontSendNotification);
                                   });
}

void StudioShellComponent::promptExportMidi()
{
    activeFileChooser = std::make_unique<juce::FileChooser>("Export MIDI",
                                                            juce::File::getSpecialLocation(juce::File::userDocumentsDirectory).getChildFile("ai-music-studio-native.mid"),
                                                            kMidiFileWildcard);
    activeFileChooser->launchAsync(juce::FileBrowserComponent::saveMode
                                       | juce::FileBrowserComponent::canSelectFiles
                                       | juce::FileBrowserComponent::warnAboutOverwriting,
                                   [safeThis = juce::Component::SafePointer<StudioShellComponent>(this)](const juce::FileChooser& chooser)
                                   {
                                       const auto selected = chooser.getResult();
                                       if (safeThis == nullptr)
                                           return;

                                       safeThis->activeFileChooser.reset();
                                       if (selected == juce::File())
                                           return;

                                       auto target = selected;
                                       if (!target.hasFileExtension(".mid") && !target.hasFileExtension(".midi"))
                                           target = target.withFileExtension(".mid");

                                       const auto result = exportProjectToMidiFile(target, safeThis->documentState.project);
                                       if (result.failed())
                                       {
                                           juce::AlertWindow::showMessageBoxAsync(juce::AlertWindow::WarningIcon,
                                                                                  "Export MIDI Failed",
                                                                                  result.getErrorMessage());
                                           return;
                                       }

                                       safeThis->statusLabel.setText("Exported MIDI: " + target.getFileName(), juce::dontSendNotification);
                                   });
}

void StudioShellComponent::promptExportWav()
{
    if (documentState.project.rightLocatorSec <= documentState.project.leftLocatorSec)
    {
        juce::AlertWindow::showMessageBoxAsync(juce::AlertWindow::WarningIcon,
                                               "Invalid Locators",
                                               "Right locator must be greater than left locator for WAV export.");
        return;
    }

    if (rackPreviewRunning || projectPreviewRunning)
        stopRackPreview();

    activeFileChooser = std::make_unique<juce::FileChooser>("Export Sequence as WAV",
                                                            juce::File::getSpecialLocation(juce::File::userDocumentsDirectory)
                                                                .getChildFile("ai-music-studio-native-sequence.wav"),
                                                            "*.wav");
    activeFileChooser->launchAsync(juce::FileBrowserComponent::saveMode
                                       | juce::FileBrowserComponent::canSelectFiles
                                       | juce::FileBrowserComponent::warnAboutOverwriting,
                                   [safeThis = juce::Component::SafePointer<StudioShellComponent>(this)](const juce::FileChooser& chooser)
                                   {
                                       const auto selected = chooser.getResult();
                                       if (safeThis == nullptr)
                                           return;

                                       safeThis->activeFileChooser.reset();
                                       if (selected == juce::File())
                                           return;

                                       auto target = selected;
                                       if (!target.hasFileExtension(".wav"))
                                           target = target.withFileExtension(".wav");

                                       AudioExportSummary summary;
                                       const auto result = exportProjectRangeToWavFile(target,
                                                                                       safeThis->documentState.project,
                                                                                       safeThis->nativeVstHost,
                                                                                       summary);
                                       if (result.failed())
                                       {
                                           juce::AlertWindow::showMessageBoxAsync(juce::AlertWindow::WarningIcon,
                                                                                  "Export WAV Failed",
                                                                                  result.getErrorMessage());
                                           return;
                                       }

                                       auto statusMessage = "Exported WAV: " + target.getFileName()
                                           + "  (" + juce::String(summary.renderedInstrumentTrackCount) + " instrument track"
                                           + (summary.renderedInstrumentTrackCount == 1 ? "" : "s")
                                           + ", " + juce::String(summary.mixedAudioTrackCount) + " audio track"
                                           + (summary.mixedAudioTrackCount == 1 ? "" : "s")
                                           + ")";
                                       if (summary.warnings.size() > 0)
                                           statusMessage << "  Warnings: " << juce::String(summary.warnings.size());
                                       safeThis->statusLabel.setText(statusMessage, juce::dontSendNotification);

                                       if (summary.warnings.size() > 0)
                                       {
                                           juce::StringArray previewWarnings;
                                           const auto previewCount = juce::jmin(6, summary.warnings.size());
                                           for (int index = 0; index < previewCount; ++index)
                                               previewWarnings.add(summary.warnings[index]);
                                           if (summary.warnings.size() > previewCount)
                                               previewWarnings.add("...");

                                           juce::AlertWindow::showMessageBoxAsync(juce::AlertWindow::InfoIcon,
                                                                                  "Export WAV Completed With Warnings",
                                                                                  previewWarnings.joinIntoString("\n"));
                                       }
                                   });
}

void StudioShellComponent::promptExportSelectedTrackWav()
{
    const auto selected = getSelectedTrackIndex();
    if (!juce::isPositiveAndBelow(selected, static_cast<int>(documentState.project.tracks.size())))
    {
        juce::AlertWindow::showMessageBoxAsync(juce::AlertWindow::InfoIcon,
                                               "No Track Selected",
                                               "Select a track first before exporting a native track WAV.");
        return;
    }

    if (documentState.project.rightLocatorSec <= documentState.project.leftLocatorSec)
    {
        juce::AlertWindow::showMessageBoxAsync(juce::AlertWindow::WarningIcon,
                                               "Invalid Locators",
                                               "Right locator must be greater than left locator for WAV export.");
        return;
    }

    if (rackPreviewRunning || projectPreviewRunning)
        stopRackPreview();

    const auto& track = documentState.project.tracks[static_cast<size_t>(selected)];
    const auto suggestedTrackName = track.name.trim().isNotEmpty() ? track.name.trim() : ("Track " + juce::String(selected + 1));
    const auto defaultStemName = juce::File::createLegalFileName(suggestedTrackName.replaceCharacter(' ', '_'));
    const auto defaultTarget = juce::File::getSpecialLocation(juce::File::userDocumentsDirectory)
        .getChildFile((defaultStemName.isNotEmpty() ? defaultStemName : "track") + ".wav");

    activeFileChooser = std::make_unique<juce::FileChooser>("Export Selected Track as WAV",
                                                            defaultTarget,
                                                            "*.wav");
    activeFileChooser->launchAsync(juce::FileBrowserComponent::saveMode
                                       | juce::FileBrowserComponent::canSelectFiles
                                       | juce::FileBrowserComponent::warnAboutOverwriting,
                                   [safeThis = juce::Component::SafePointer<StudioShellComponent>(this),
                                    trackIndex = selected](const juce::FileChooser& chooser)
                                   {
                                       const auto selectedFile = chooser.getResult();
                                       if (safeThis == nullptr)
                                           return;

                                       safeThis->activeFileChooser.reset();
                                       if (selectedFile == juce::File())
                                           return;

                                       if (!juce::isPositiveAndBelow(trackIndex, static_cast<int>(safeThis->documentState.project.tracks.size())))
                                           return;

                                       auto target = selectedFile;
                                       if (!target.hasFileExtension(".wav"))
                                           target = target.withFileExtension(".wav");

                                       AudioExportSummary summary;
                                       const auto result = exportTrackRangeToWavFile(target,
                                                                                     safeThis->documentState.project,
                                                                                     trackIndex,
                                                                                     safeThis->nativeVstHost,
                                                                                     summary);
                                       if (result.failed())
                                       {
                                           juce::AlertWindow::showMessageBoxAsync(juce::AlertWindow::WarningIcon,
                                                                                  "Track Export Failed",
                                                                                  result.getErrorMessage());
                                           return;
                                       }

                                       auto updatedTrack = safeThis->documentState.project.tracks[static_cast<size_t>(trackIndex)];
                                       updatedTrack.renderedAudioPath = target.getFullPathName();
                                       safeThis->applyTrackStateEdit(trackIndex, updatedTrack, "Export Track WAV");

                                       auto statusMessage = "Exported track WAV: " + target.getFileName();
                                       if (summary.warnings.size() > 0)
                                           statusMessage << "  Warnings: " << juce::String(summary.warnings.size());
                                       safeThis->statusLabel.setText(statusMessage, juce::dontSendNotification);

                                       if (summary.warnings.size() > 0)
                                       {
                                           juce::StringArray previewWarnings;
                                           const auto previewCount = juce::jmin(6, summary.warnings.size());
                                           for (int index = 0; index < previewCount; ++index)
                                               previewWarnings.add(summary.warnings[index]);
                                           if (summary.warnings.size() > previewCount)
                                               previewWarnings.add("...");

                                           juce::AlertWindow::showMessageBoxAsync(juce::AlertWindow::InfoIcon,
                                                                                  "Track Export Completed With Warnings",
                                                                                  previewWarnings.joinIntoString("\n"));
                                       }
                                   });
}

void StudioShellComponent::promptExportProjectStems()
{
    if (documentState.project.rightLocatorSec <= documentState.project.leftLocatorSec)
    {
        juce::AlertWindow::showMessageBoxAsync(juce::AlertWindow::WarningIcon,
                                               "Invalid Locators",
                                               "Right locator must be greater than left locator for stem export.");
        return;
    }

    if (rackPreviewRunning || projectPreviewRunning)
        stopRackPreview();

    activeFileChooser = std::make_unique<juce::FileChooser>("Export Native Stems",
                                                            juce::File::getSpecialLocation(juce::File::userDocumentsDirectory)
                                                                .getChildFile("ai-music-studio-native-stems"));
    activeFileChooser->launchAsync(juce::FileBrowserComponent::openMode
                                       | juce::FileBrowserComponent::canSelectDirectories,
                                   [safeThis = juce::Component::SafePointer<StudioShellComponent>(this)](const juce::FileChooser& chooser)
                                   {
                                       const auto selectedFolder = chooser.getResult();
                                       if (safeThis == nullptr)
                                           return;

                                       safeThis->activeFileChooser.reset();
                                       if (selectedFolder == juce::File())
                                           return;

                                       AudioExportBatchSummary summary;
                                       const auto result = exportProjectStemsToFolder(selectedFolder,
                                                                                      safeThis->documentState.project,
                                                                                      safeThis->nativeVstHost,
                                                                                      summary);
                                       if (result.failed())
                                       {
                                           juce::AlertWindow::showMessageBoxAsync(juce::AlertWindow::WarningIcon,
                                                                                  "Stem Export Failed",
                                                                                  result.getErrorMessage());
                                           return;
                                       }

                                       auto updatedProject = safeThis->documentState.project;
                                       for (const auto& exportedFile : summary.exportedFiles)
                                       {
                                           if (!juce::isPositiveAndBelow(exportedFile.trackIndex, static_cast<int>(updatedProject.tracks.size())))
                                               continue;
                                           updatedProject.tracks[static_cast<size_t>(exportedFile.trackIndex)].renderedAudioPath = exportedFile.filePath;
                                       }
                                       safeThis->applyProjectStateEdit(updatedProject, "Export Stems");

                                       auto statusMessage = "Exported " + juce::String(summary.exportedTrackCount)
                                           + " stem" + (summary.exportedTrackCount == 1 ? "" : "s")
                                           + " to " + selectedFolder.getFullPathName();
                                       if (summary.warnings.size() > 0)
                                           statusMessage << "  Warnings: " << juce::String(summary.warnings.size());
                                       safeThis->statusLabel.setText(statusMessage, juce::dontSendNotification);

                                       if (summary.warnings.size() > 0)
                                       {
                                           juce::StringArray previewWarnings;
                                           const auto previewCount = juce::jmin(8, summary.warnings.size());
                                           for (int index = 0; index < previewCount; ++index)
                                               previewWarnings.add(summary.warnings[index]);
                                           if (summary.warnings.size() > previewCount)
                                               previewWarnings.add("...");

                                           juce::AlertWindow::showMessageBoxAsync(juce::AlertWindow::InfoIcon,
                                                                                  "Stem Export Completed With Warnings",
                                                                                  previewWarnings.joinIntoString("\n"));
                                       }
                                   });
}

juce::Result StudioShellComponent::ensureSampleAssetForFile(const juce::File& file, int& outAssetIndex)
{
    outAssetIndex = -1;
    const auto targetPath = file.getFullPathName();
    for (int index = 0; index < static_cast<int>(documentState.project.sampleAssets.size()); ++index)
    {
        if (documentState.project.sampleAssets[static_cast<size_t>(index)].path.equalsIgnoreCase(targetPath))
        {
            outAssetIndex = index;
            setSelectedSampleAssetIndex(index);
            return juce::Result::ok();
        }
    }

    SampleAsset asset;
    const auto loadResult = loadSampleAssetFromFile(file, asset);
    if (loadResult.failed())
        return loadResult;

    auto updatedProject = documentState.project;
    updatedProject.sampleAssets.push_back(std::move(asset));
    updatedProject.samplePaths.addIfNotAlreadyThere(targetPath);
    const auto insertedIndex = static_cast<int>(updatedProject.sampleAssets.size()) - 1;
    applyProjectStateEdit(updatedProject, "Import Sample");
    outAssetIndex = insertedIndex;
    setSelectedSampleAssetIndex(insertedIndex);
    return juce::Result::ok();
}

juce::Result StudioShellComponent::placeSampleAssetAtPlayhead(int assetIndex,
                                                              const juce::String& actionName,
                                                              juce::String& outTrackName)
{
    if (!juce::isPositiveAndBelow(assetIndex, static_cast<int>(documentState.project.sampleAssets.size())))
        return juce::Result::fail("Select a sample from the native sample library first.");

    const auto trackIndex = findPreferredSampleTrackIndex();
    if (trackIndex < 0)
        return juce::Result::fail("Set a track type to sample before placing audio clips.");

    const auto& asset = documentState.project.sampleAssets[static_cast<size_t>(assetIndex)];
    auto updatedProject = documentState.project;
    SampleClip clip;
    clip.path = asset.path;
    clip.trackIndex = trackIndex;
    clip.startSec = documentState.project.playheadSec;
    clip.durationSec = asset.durationSec;
    clip.sampleRate = asset.sampleRate;
    clip.waveformPreview = asset.waveformPreview;
    updatedProject.sampleClips.push_back(std::move(clip));
    outTrackName = updatedProject.tracks[static_cast<size_t>(trackIndex)].name;
    applyProjectStateEdit(updatedProject, actionName);
    return juce::Result::ok();
}

void StudioShellComponent::promptImportSample()
{
    activeFileChooser = std::make_unique<juce::FileChooser>("Import Sample",
                                                            juce::File(),
                                                            "*.wav;*.aiff;*.aif;*.flac;*.ogg;*.mp3");
    activeFileChooser->launchAsync(juce::FileBrowserComponent::openMode | juce::FileBrowserComponent::canSelectFiles,
                                   [safeThis = juce::Component::SafePointer<StudioShellComponent>(this)](const juce::FileChooser& chooser)
                                   {
                                       const auto selected = chooser.getResult();
                                       if (safeThis == nullptr)
                                           return;

                                       safeThis->activeFileChooser.reset();
                                       if (selected == juce::File())
                                           return;

                                       int assetIndex = -1;
                                       const auto result = safeThis->ensureSampleAssetForFile(selected, assetIndex);
                                       if (result.failed())
                                       {
                                           juce::AlertWindow::showMessageBoxAsync(juce::AlertWindow::WarningIcon,
                                                                                  "Import Sample Failed",
                                                                                  result.getErrorMessage());
                                           return;
                                       }

                                        safeThis->statusLabel.setText("Imported sample: " + selected.getFileName(), juce::dontSendNotification);
                                    });
}

juce::String StudioShellComponent::defaultVstFolderPath() const
{
    const auto bundledDirectory = findBundledVstDirectory();
    return bundledDirectory != juce::File() ? bundledDirectory.getFullPathName() : juce::String();
}

juce::StringArray StudioShellComponent::userManagedVstFolderPaths() const
{
    juce::StringArray folders;
    const auto defaultFolder = defaultVstFolderPath().trim();

    for (const auto& rawFolder : documentState.project.vstiFolderPaths)
    {
        auto normalisedFolder = rawFolder.trim();
        if (normalisedFolder.isEmpty())
            continue;

        normalisedFolder = juce::File(normalisedFolder).getFullPathName();
        if (defaultFolder.isNotEmpty() && normalisedFolder.equalsIgnoreCase(defaultFolder))
            continue;

        bool alreadyAdded = false;
        for (const auto& existing : folders)
        {
            if (existing.equalsIgnoreCase(normalisedFolder))
            {
                alreadyAdded = true;
                break;
            }
        }

        if (!alreadyAdded)
            folders.add(normalisedFolder);
    }

    return folders;
}

void StudioShellComponent::promptAddUserVstFolder()
{
    activeFileChooser = std::make_unique<juce::FileChooser>("Select User VST Folder",
                                                            juce::File(defaultVstFolderPath()),
                                                            "*");
    activeFileChooser->launchAsync(juce::FileBrowserComponent::openMode | juce::FileBrowserComponent::canSelectDirectories,
                                   [safeThis = juce::Component::SafePointer<StudioShellComponent>(this)](const juce::FileChooser& chooser)
                                   {
                                       const auto selected = chooser.getResult();
                                       if (safeThis == nullptr)
                                           return;

                                       safeThis->activeFileChooser.reset();
                                       if (selected == juce::File())
                                           return;

                                       if (!selected.isDirectory())
                                       {
                                           juce::AlertWindow::showMessageBoxAsync(juce::AlertWindow::WarningIcon,
                                                                                  "Add VST Folder Failed",
                                                                                  "Choose a VST folder directory.");
                                           return;
                                       }

                                       const auto selectedPath = selected.getFullPathName();
                                       const auto defaultFolder = safeThis->defaultVstFolderPath().trim();
                                       if (defaultFolder.isNotEmpty() && selectedPath.equalsIgnoreCase(defaultFolder))
                                       {
                                           safeThis->statusLabel.setText("The bundled default VST folder is already scanned automatically.",
                                                                         juce::dontSendNotification);
                                           return;
                                       }

                                       auto updatedProject = safeThis->documentState.project;
                                       bool alreadyTracked = false;
                                       for (const auto& existing : updatedProject.vstiFolderPaths)
                                       {
                                           if (juce::File(existing.trim()).getFullPathName().equalsIgnoreCase(selectedPath))
                                           {
                                               alreadyTracked = true;
                                               break;
                                           }
                                       }

                                       if (alreadyTracked)
                                       {
                                           safeThis->statusLabel.setText("User VST folder already added: " + selectedPath,
                                                                         juce::dontSendNotification);
                                           return;
                                       }

                                       updatedProject.vstiFolderPaths.add(selectedPath);
                                       safeThis->applyProjectStateEdit(updatedProject, "Add VST Folder");
                                       safeThis->refreshRackCatalog();
                                       safeThis->statusLabel.setText("Added user VST folder: " + selectedPath,
                                                                     juce::dontSendNotification);
                                   });
}

void StudioShellComponent::removeUserVstFolder(const juce::String& folderPath)
{
    const auto trimmedPath = folderPath.trim();
    if (trimmedPath.isEmpty())
        return;

    auto updatedProject = documentState.project;
    bool removed = false;
    for (int index = updatedProject.vstiFolderPaths.size(); --index >= 0;)
    {
        if (juce::File(updatedProject.vstiFolderPaths[index].trim()).getFullPathName().equalsIgnoreCase(trimmedPath))
        {
            updatedProject.vstiFolderPaths.remove(index);
            removed = true;
        }
    }

    if (!removed)
        return;

    applyProjectStateEdit(updatedProject, "Remove VST Folder");
    statusLabel.setText("Removed user VST folder: " + trimmedPath, juce::dontSendNotification);
}

void StudioShellComponent::promptImportRackPlugin()
{
    activeFileChooser = std::make_unique<juce::FileChooser>("Import Rack Plugin",
                                                            juce::File(),
                                                            "*.vst3;*.dll;*.so");
    activeFileChooser->launchAsync(juce::FileBrowserComponent::openMode
                                       | juce::FileBrowserComponent::canSelectFiles
                                       | juce::FileBrowserComponent::canSelectDirectories,
                                   [safeThis = juce::Component::SafePointer<StudioShellComponent>(this)](const juce::FileChooser& chooser)
                                   {
                                       const auto selected = chooser.getResult();
                                       if (safeThis == nullptr)
                                           return;

                                       safeThis->activeFileChooser.reset();
                                       if (selected == juce::File())
                                           return;

                                       const auto selectedPath = selected.getFullPathName();
                                       const auto hasSupportedExtension = selected.hasFileExtension(".vst3;.dll;.so");
                                       if (!hasSupportedExtension)
                                       {
                                           juce::AlertWindow::showMessageBoxAsync(juce::AlertWindow::WarningIcon,
                                                                                  "Import Rack Plugin Failed",
                                                                                  "Choose a `.vst3`, `.dll`, or `.so` plugin path.");
                                           return;
                                       }

                                       auto updatedProject = safeThis->documentState.project;
                                       updatedProject.vstiPaths.addIfNotAlreadyThere(selectedPath);
                                       updatedProject.vstiFolderPaths.addIfNotAlreadyThere(selected.getParentDirectory().getFullPathName());

                                       VstInstrument entry;
                                       entry.name = selected.getFileNameWithoutExtension();
                                       entry.path = selectedPath;
                                       entry.pluginName = entry.name;
                                       entry.isInstrument = true;
                                       entry.isEffect = false;
                                       entry.category = "Instrument";
                                       entry.hostSupported = selected.exists();
                                       entry.hostError = selected.exists() ? juce::String() : juce::String("Plugin path does not exist.");

                                       bool changed = false;
                                       if (const auto existingIndex = findRackInstrumentIndexByReference(updatedProject, selectedPath); existingIndex >= 0)
                                       {
                                           auto& existing = updatedProject.vstRack[static_cast<size_t>(existingIndex)];
                                           if (existing.path != entry.path
                                               || existing.name != entry.name
                                               || existing.pluginName != entry.pluginName
                                               || existing.hostSupported != entry.hostSupported)
                                           {
                                               existing.path = entry.path;
                                               existing.name = entry.name;
                                               existing.pluginName = entry.pluginName;
                                               existing.isInstrument = true;
                                               existing.isEffect = false;
                                               existing.category = entry.category;
                                               existing.hostSupported = entry.hostSupported;
                                               existing.hostError = entry.hostError;
                                               changed = true;
                                           }
                                       }
                                       else
                                       {
                                           updatedProject.vstRack.push_back(entry);
                                           changed = true;
                                       }

                                       if (!changed)
                                       {
                                           safeThis->statusLabel.setText("Rack plugin already available: " + selected.getFileName(),
                                                                         juce::dontSendNotification);
                                           if (safeThis->floatingRackBrowser != nullptr)
                                               safeThis->floatingRackBrowser->selectRackByReference(selectedPath);
                                           return;
                                       }

                                       safeThis->applyProjectStateEdit(updatedProject, "Import Rack Plugin");
                                       if (safeThis->floatingRackBrowser != nullptr)
                                           safeThis->floatingRackBrowser->selectRackByReference(selectedPath);
                                       safeThis->statusLabel.setText("Imported rack plugin: " + selected.getFileName(), juce::dontSendNotification);
                                   });
}

void StudioShellComponent::promptRelinkSelectedTrackRenderedAudio()
{
    const auto selected = getSelectedTrackIndex();
    if (!juce::isPositiveAndBelow(selected, static_cast<int>(documentState.project.tracks.size())))
        return;

    activeFileChooser = std::make_unique<juce::FileChooser>("Relink Rendered Audio",
                                                            juce::File(),
                                                            "*.wav;*.aiff;*.aif;*.flac;*.ogg;*.mp3");
    activeFileChooser->launchAsync(juce::FileBrowserComponent::openMode | juce::FileBrowserComponent::canSelectFiles,
                                   [safeThis = juce::Component::SafePointer<StudioShellComponent>(this),
                                    trackIndex = selected](const juce::FileChooser& chooser)
                                   {
                                       const auto selectedFile = chooser.getResult();
                                       if (safeThis == nullptr)
                                           return;

                                       safeThis->activeFileChooser.reset();
                                       if (selectedFile == juce::File())
                                           return;

                                       if (!juce::isPositiveAndBelow(trackIndex, static_cast<int>(safeThis->documentState.project.tracks.size())))
                                           return;

                                       auto updatedTrack = safeThis->documentState.project.tracks[static_cast<size_t>(trackIndex)];
                                       updatedTrack.renderedAudioPath = selectedFile.getFullPathName();
                                       safeThis->applyTrackStateEdit(trackIndex, updatedTrack, "Relink Rendered Audio");
                                       safeThis->statusLabel.setText("Relinked rendered audio for " + updatedTrack.name + ".",
                                                                     juce::dontSendNotification);
                                   });
}

void StudioShellComponent::placeSelectedSampleAtPlayhead()
{
    const auto assetRow = sampleAssetList.getSelectedRow();
    juce::String trackName;
    const auto result = placeSampleAssetAtPlayhead(assetRow, "Place Sample Clip", trackName);
    if (result.failed())
    {
        juce::AlertWindow::showMessageBoxAsync(juce::AlertWindow::InfoIcon,
                                               "Place Sample Failed",
                                               result.getErrorMessage());
        return;
    }

    statusLabel.setText("Placed sample clip on " + trackName + ".", juce::dontSendNotification);
}

void StudioShellComponent::importSelectedTrackRenderToSampleLibrary()
{
    const auto* track = getSelectedTrack();
    if (track == nullptr)
        return;

    const auto renderFile = juce::File(track->renderedAudioPath.trim());
    if (!renderFile.existsAsFile())
    {
        juce::AlertWindow::showMessageBoxAsync(juce::AlertWindow::InfoIcon,
                                               "Missing Render File",
                                               "Export or relink a rendered audio file for the selected track first.");
        return;
    }

    int assetIndex = -1;
    const auto result = ensureSampleAssetForFile(renderFile, assetIndex);
    if (result.failed())
    {
        juce::AlertWindow::showMessageBoxAsync(juce::AlertWindow::WarningIcon,
                                               "Import Render Failed",
                                               result.getErrorMessage());
        return;
    }

    statusLabel.setText("Added rendered audio for " + track->name + " to the native sample library.",
                        juce::dontSendNotification);
}

void StudioShellComponent::placeSelectedTrackRenderAtPlayhead()
{
    const auto* track = getSelectedTrack();
    if (track == nullptr)
        return;

    const auto renderFile = juce::File(track->renderedAudioPath.trim());
    if (!renderFile.existsAsFile())
    {
        juce::AlertWindow::showMessageBoxAsync(juce::AlertWindow::InfoIcon,
                                               "Missing Render File",
                                               "Export or relink a rendered audio file for the selected track first.");
        return;
    }

    int assetIndex = -1;
    auto result = ensureSampleAssetForFile(renderFile, assetIndex);
    if (result.failed())
    {
        juce::AlertWindow::showMessageBoxAsync(juce::AlertWindow::WarningIcon,
                                               "Place Render Failed",
                                               result.getErrorMessage());
        return;
    }

    juce::String trackName;
    result = placeSampleAssetAtPlayhead(assetIndex, "Place Render Clip", trackName);
    if (result.failed())
    {
        juce::AlertWindow::showMessageBoxAsync(juce::AlertWindow::InfoIcon,
                                               "Place Render Failed",
                                               result.getErrorMessage());
        return;
    }

    statusLabel.setText("Placed rendered audio on " + trackName + ".", juce::dontSendNotification);
}

void StudioShellComponent::showAiSettingsDialog()
{
    auto* dialog = new juce::AlertWindow("Native AI Settings",
                                         "Configure the native C++ AI provider. Saved Python preferences and auth are reused when available.",
                                         juce::AlertWindow::NoIcon);
    dialog->addComboBox("provider", juce::StringArray{ "OpenAI", "Ollama" }, "Provider");
    if (auto* providerBox = dialog->getComboBoxComponent("provider"))
        providerBox->setSelectedItemIndex(aiClient.getProvider() == AIClient::Provider::ollama ? 1 : 0);

    dialog->addTextEditor("remoteModel", aiClient.getRemoteModel(), "OpenAI model");
    dialog->addTextEditor("apiKey", {}, "OpenAI API key (leave blank to keep saved key)");
    if (auto* apiKeyEditor = dialog->getTextEditor("apiKey"))
        apiKeyEditor->setPasswordCharacter(0x2022);
    dialog->addTextEditor("ollamaBaseUrl", aiClient.getOllamaBaseUrl(), "Ollama endpoint");
    dialog->addTextEditor("ollamaModel", aiClient.getOllamaModel(), "Ollama model");
    dialog->addTextEditor("timeoutSeconds", juce::String(aiClient.getRequestTimeoutSeconds()), "Request timeout (seconds)");

    dialog->addButton("Save", 1, juce::KeyPress(juce::KeyPress::returnKey));
    dialog->addButton("Clear Auth", 2);
    dialog->addButton("Cancel", 0, juce::KeyPress(juce::KeyPress::escapeKey));
    auto safeThis = juce::Component::SafePointer<StudioShellComponent>(this);
    auto safeDialog = juce::Component::SafePointer<juce::AlertWindow>(dialog);
    dialog->enterModalState(true,
                            juce::ModalCallbackFunction::create([safeThis, safeDialog] (int result)
                            {
                                if (safeThis == nullptr || safeDialog == nullptr || result == 0)
                                    return;

                                if (result == 2)
                                {
                                    safeThis->aiClient.clearAuth();
                                    safeThis->updateAiStatusSummary();
                                    safeThis->statusLabel.setText("Cleared native AI credentials.", juce::dontSendNotification);
                                    safeThis->appendActivityLog("AI Settings", "Cleared saved AI credentials.");
                                    return;
                                }

                                const auto providerText = safeDialog->getComboBoxComponent("provider") != nullptr
                                    ? safeDialog->getComboBoxComponent("provider")->getText().trim()
                                    : juce::String("OpenAI");
                                safeThis->aiClient.setProvider(providerText.equalsIgnoreCase("Ollama") ? AIClient::Provider::ollama
                                                                                                       : AIClient::Provider::openAI);
                                safeThis->aiClient.setRemoteModel(safeDialog->getTextEditorContents("remoteModel"));
                                safeThis->aiClient.setOllamaConnection(safeDialog->getTextEditorContents("ollamaBaseUrl"),
                                                                       safeDialog->getTextEditorContents("ollamaModel"));
                                safeThis->aiClient.setRequestTimeoutSeconds(safeDialog->getTextEditorContents("timeoutSeconds").getIntValue());
                                safeThis->aiClient.saveSettings();

                                const auto apiKey = safeDialog->getTextEditorContents("apiKey").trim();
                                if (apiKey.isNotEmpty())
                                    safeThis->aiClient.setApiKey(apiKey);

                                safeThis->updateAiStatusSummary();
                                safeThis->statusLabel.setText("Updated native AI settings.", juce::dontSendNotification);
                                safeThis->appendActivityLog("AI Settings",
                                                            "Updated AI settings\nProvider: "
                                                                + providerText
                                                                + "\nStatus: "
                                                                + safeThis->aiClient.authStatus());
                            }),
                            true);
}

void StudioShellComponent::composeWithAi()
{
    if (aiComposeBusy)
    {
        statusLabel.setText("AI composition is already running in the background.", juce::dontSendNotification);
        return;
    }

    if (!aiClient.isEnabled())
    {
        juce::AlertWindow::showMessageBoxAsync(juce::AlertWindow::InfoIcon,
                                               "AI Not Configured",
                                               "Open AI Settings first and connect either OpenAI or a local Ollama model.");
        return;
    }

    auto* dialog = new juce::AlertWindow("AI Compose",
                                         "Describe the arrangement you want the native C++ composer to generate.",
                                         juce::AlertWindow::NoIcon);
    dialog->addTextEditor("prompt", aiComposeDefaultPrompt, "Prompt");
    if (auto* promptEditor = dialog->getTextEditor("prompt"))
    {
        promptEditor->setMultiLine(true, true);
        promptEditor->setReturnKeyStartsNewLine(true);
        promptEditor->setSize(promptEditor->getWidth(), 180);
    }
    dialog->addTextEditor("bars", juce::String(aiComposeDefaultBars), "Bars");
    dialog->addButton("Compose", 1, juce::KeyPress(juce::KeyPress::returnKey));
    dialog->addButton("Cancel", 0, juce::KeyPress(juce::KeyPress::escapeKey));

    auto safeThis = juce::Component::SafePointer<StudioShellComponent>(this);
    auto safeDialog = juce::Component::SafePointer<juce::AlertWindow>(dialog);
    dialog->enterModalState(true,
                            juce::ModalCallbackFunction::create([safeThis, safeDialog] (int result)
                            {
                                if (safeThis == nullptr || safeDialog == nullptr || result != 1)
                                    return;

                                const auto prompt = safeDialog->getTextEditorContents("prompt").trim();
                                const auto bars = juce::jlimit(1, 128, safeDialog->getTextEditorContents("bars").getIntValue());
                                if (prompt.isEmpty())
                                    return;

                                safeThis->aiComposeDefaultPrompt = prompt;
                                safeThis->aiComposeDefaultBars = bars;
                                safeThis->aiComposeRequestedBars = bars;
                                safeThis->setAiComposeBusy(true, "AI processing...");
                                safeThis->statusLabel.setText("AI processing composition via " + safeThis->aiClient.authStatus() + ".",
                                                              juce::dontSendNotification);
                                safeThis->appendActivityLog("AI Compose",
                                                            "Compose requested\nBars: "
                                                                + juce::String(bars)
                                                                + "\nBPM: "
                                                                + juce::String(safeThis->documentState.project.bpm)
                                                                + "\nProvider: "
                                                                + safeThis->aiClient.authStatus()
                                                                + "\n\nPrompt\n"
                                                                + prompt);

                                auto clientCopy = safeThis->aiClient;
                                safeThis->aiComposeFuture = std::async(std::launch::async,
                                                                       [client = std::move(clientCopy),
                                                                        prompt,
                                                                        bars,
                                                                        bpm = safeThis->documentState.project.bpm] () mutable
                                                                       {
                                                                           AIComposer composer(std::move(client));
                                                                           return composer.compose(prompt, bars, bpm);
                                                                       });
                            }),
                            true);
}

void StudioShellComponent::setAiComposeBusy(bool busy, const juce::String& detail)
{
    aiComposeBusy = busy;
    aiComposeBusyDetail = busy ? detail : juce::String();
    refreshPollingTimerState();
    updateEditorState();
    updateAiStatusSummary();
}

void StudioShellComponent::refreshPollingTimerState()
{
    const bool hasOpenRackEditors = hasOpenRackEditorSessions();
    const bool shouldPauseForRackEditor = loadedRackEditorOpen
        && !hasOpenRackEditors
        && !rackPreviewRunning
        && !projectPreviewRunning
        && !aiComposeBusy;

    int targetHz = 0;
    if (shouldPauseForRackEditor)
    {
        targetHz = 0;
    }
    else if (projectPreviewRunning || rackPreviewRunning)
    {
        targetHz = playbackRefreshRateForComponent(*this);
    }
    else if (aiComposeBusy)
    {
        targetHz = 6;
    }
    else if (loadedRackEditorOpen || hasOpenRackEditors)
    {
        targetHz = 4;
    }

    if (targetHz <= 0)
    {
        if (pollingTimerHz != 0 || isTimerRunning())
        {
            stopTimer();
            pollingTimerHz = 0;
        }
        return;
    }

    if (pollingTimerHz != targetHz || !isTimerRunning())
    {
        startTimerHz(targetHz);
        pollingTimerHz = targetHz;
    }
}

void StudioShellComponent::pollAiComposeFuture()
{
    if (!aiComposeBusy || !aiComposeFuture.valid())
        return;

    if (aiComposeFuture.wait_for(std::chrono::seconds(0)) != std::future_status::ready)
        return;

    try
    {
        auto result = aiComposeFuture.get();
        applyAiComposeResult(result, aiComposeRequestedBars);
        statusLabel.setText("AI generated " + juce::String(static_cast<int>(result.tracks.size())) + " track(s).",
                            juce::dontSendNotification);
        appendActivityLog("AI Compose",
                          "Compose completed successfully\nGenerated tracks: "
                              + juce::String(static_cast<int>(result.tracks.size())));
    }
    catch (const std::exception& exc)
    {
        juce::AlertWindow::showMessageBoxAsync(juce::AlertWindow::WarningIcon,
                                               "AI Composition Failed",
                                               juce::String::fromUTF8(exc.what()));
        statusLabel.setText("AI composition failed.", juce::dontSendNotification);
        appendActivityLog("AI Compose Error", juce::String::fromUTF8(exc.what()));
    }

    setAiComposeBusy(false);
}

void StudioShellComponent::updateAiStatusSummary()
{
    const auto& theme = themeSpecForIndex(currentThemeIndex);
    auto text = aiClient.authStatus();
    if (aiComposeBusy)
        text = (aiComposeBusyDetail.isNotEmpty() ? aiComposeBusyDetail : juce::String("AI processing...")) + "   " + text;

    aiStatusSummaryLabel.setColour(juce::Label::textColourId,
                                   aiComposeBusy ? theme.infoText
                                                 : (aiClient.isEnabled() ? theme.successText
                                                                         : theme.warningText));
    aiStatusSummaryLabel.setText(text, juce::dontSendNotification);
}

void StudioShellComponent::applyAiComposeResult(const AIComposeResult& result, int requestedBars)
{
    if (result.tracks.empty())
        throw std::runtime_error("AI returned no tracks.");

    syncBundledRackCatalogInProject();

    auto updatedProject = documentState.project;
    updatedProject.tracks.clear();
    updatedProject.midiPatterns.clear();
    updatedProject.midiSections.clear();
    updatedProject.sampleClips.clear();
    updatedProject.leftLocatorTick = 0;
    updatedProject.playheadTick = 0;

    auto maxEndTick = juce::jmax(kTicksPerBar, juce::jmax(1, requestedBars) * kTicksPerBar);
    for (const auto& sourceTrack : result.tracks)
    {
        TrackState track;
        track.name = sourceTrack.name.trim().isNotEmpty()
            ? sourceTrack.name.trim()
            : "AI Track " + juce::String(static_cast<int>(updatedProject.tracks.size()) + 1);
        track.trackType = "instrument";
        track.instrument = sourceTrack.instrument.trim().isNotEmpty() ? sourceTrack.instrument.trim() : track.name;
        track.midiProgram = defaultMidiProgramForInstrumentName(track.instrument);
        track.synthProfile = inferSynthProfile(track.instrument, track.midiProgram);

        const auto instrumentName = track.instrument.toLowerCase();
        if (track.synthProfile == "noise_kit" || instrumentName.contains("drum") || instrumentName.contains("kit"))
        {
            track.midiChannel = 9;
            if (track.midiProgram == 0)
                track.midiProgram = 112;
        }
        else
        {
            track.midiChannel = static_cast<int>(updatedProject.tracks.size()) % 16;
            if (track.midiChannel == 9)
                track.midiChannel = 10;
        }

        MidiPattern pattern;
        pattern.id = juce::Uuid().toString();
        pattern.name = track.name;
        auto trackMaxEndTick = 0;
        pattern.notes.reserve(sourceTrack.notes.size());
        for (const auto& sourceNote : sourceTrack.notes)
        {
            MidiNote note;
            note.startTick = juce::jmax(0, static_cast<int>(std::llround(sourceNote.startBeat * kTicksPerBeat)));
            note.durationTick = juce::jmax(1, static_cast<int>(std::llround(sourceNote.durationBeat * kTicksPerBeat)));
            note.pitch = juce::jlimit(12, 84, sourceNote.pitch);
            note.velocity = juce::jlimit(1, 127, sourceNote.velocity);
            trackMaxEndTick = juce::jmax(trackMaxEndTick, note.startTick + note.durationTick);
            pattern.notes.push_back(note);
        }

        if (pattern.notes.empty())
            continue;

        pattern.lengthTicks = juce::jmax(kTicksPerBar, trackMaxEndTick);

        materialiseNativeInstrumentTrack(updatedProject, track);

        updatedProject.tracks.push_back(track);
        updatedProject.midiPatterns.push_back(pattern);

        MidiSection section;
        section.trackIndex = static_cast<int>(updatedProject.tracks.size()) - 1;
        section.startTick = 0;
        section.lengthTicks = pattern.lengthTicks;
        section.name = pattern.name;
        section.patternId = pattern.id;
        updatedProject.midiSections.push_back(std::move(section));

        maxEndTick = juce::jmax(maxEndTick, trackMaxEndTick);
    }

    if (updatedProject.tracks.empty())
        throw std::runtime_error("AI returned invalid track data.");

    updatedProject.rightLocatorTick = juce::jmax(kTicksPerBar, maxEndTick);
    updatedProject.recalculateTimeFields();
    applyProjectStateEdit(updatedProject, "AI Compose");
    trackTable.selectRow(0);
    if (pianoRoll != nullptr)
        pianoRoll->grabKeyboardFocus();
}

void StudioShellComponent::syncBundledRackCatalogInProject()
{
    auto discoveredEntries = discoverBundledVstCatalog();
    for (const auto& folderPath : userManagedVstFolderPaths())
    {
        const auto extraEntries = discoverVstCatalogInDirectory(juce::File(folderPath));
        discoveredEntries.insert(discoveredEntries.end(), extraEntries.begin(), extraEntries.end());
    }

    std::sort(discoveredEntries.begin(),
              discoveredEntries.end(),
              [] (const VstInstrument& lhs, const VstInstrument& rhs)
              {
                  return lhs.path.compareIgnoreCase(rhs.path) < 0;
              });

    discoveredEntries.erase(std::unique(discoveredEntries.begin(),
                                        discoveredEntries.end(),
                                        [] (const VstInstrument& lhs, const VstInstrument& rhs)
                                        {
                                            return lhs.path.equalsIgnoreCase(rhs.path);
                                        }),
                            discoveredEntries.end());

    if (discoveredEntries.empty())
        return;

    bool changed = false;
    for (const auto& entry : discoveredEntries)
    {
        documentState.project.vstiPaths.addIfNotAlreadyThere(entry.path);

        const auto existingIndex = findRackInstrumentIndexByReference(documentState.project, entry.path);
        if (existingIndex >= 0)
        {
            auto& existing = documentState.project.vstRack[static_cast<size_t>(existingIndex)];
            if (existing.path.isEmpty())
            {
                existing.path = entry.path;
                changed = true;
            }
            if (existing.name.isEmpty())
            {
                existing.name = entry.name;
                changed = true;
            }
            if (existing.pluginName.isEmpty())
            {
                existing.pluginName = entry.pluginName;
                changed = true;
            }
            existing.isInstrument = existing.isInstrument || entry.isInstrument;
            existing.hostSupported = true;
            continue;
        }

        const auto nameIndex = findRackInstrumentIndexByReference(documentState.project, entry.name);
        if (nameIndex >= 0)
        {
            auto& existing = documentState.project.vstRack[static_cast<size_t>(nameIndex)];
            if (!existing.path.equalsIgnoreCase(entry.path))
            {
                existing.path = entry.path;
                changed = true;
            }
            if (existing.pluginName.isEmpty())
            {
                existing.pluginName = entry.pluginName;
                changed = true;
            }
            existing.isInstrument = true;
            existing.hostSupported = true;
            continue;
        }

        documentState.project.vstRack.push_back(entry);
        changed = true;
    }

    if (changed)
        normaliseProject(documentState.project);
}

void StudioShellComponent::refreshRackCatalog()
{
    const auto before = fingerprint(documentState.project);
    syncBundledRackCatalogInProject();
    const auto after = fingerprint(documentState.project);

    if (before != after)
    {
        markDirty();
        refreshUi();
        statusLabel.setText("Refreshed native VST catalog.", juce::dontSendNotification);
        return;
    }

    refreshFloatingWindows();
    statusLabel.setText("Native VST catalog is already up to date.", juce::dontSendNotification);
}

void StudioShellComponent::assignSelectedTrackRackByReference(const juce::String& reference)
{
    const auto trimmedReference = reference.trim();
    if (trimmedReference.isEmpty())
        return;

    const auto rackIndex = findRackInstrumentIndexByReference(documentState.project, trimmedReference);
    const auto entryLabel = rackIndex >= 0
        ? [&project = documentState.project, rackIndex]
        {
            const auto& entry = project.vstRack[static_cast<size_t>(rackIndex)];
            if (entry.name.isNotEmpty())
                return entry.name;
            if (entry.pluginName.isNotEmpty())
                return entry.pluginName;
            if (entry.path.isNotEmpty())
                return juce::File(entry.path).getFileNameWithoutExtension();
            return juce::String();
        }()
        : trimmedReference;

    applySelectedTrackMutation([entryLabel] (TrackState& track)
                               {
                                   track.instrumentMode = "VSTI Rack";
                                   track.rackVst = entryLabel;
                                   if (track.instrument.trim().isEmpty())
                                       track.instrument = entryLabel;
                                   track.vstiStatePath.clear();
                                   track.vstiStateBase64.clear();
                                   track.vstiParameters.clear();
                                   track.synthProfile = "vst_instrument";
                               },
                               "Assign Rack Plugin");
    statusLabel.setText("Assigned native rack: " + entryLabel + ".", juce::dontSendNotification);
}

void StudioShellComponent::clearSelectedTrackRackAssignment()
{
    const auto selected = getSelectedTrackIndex();
    if (!juce::isPositiveAndBelow(selected, static_cast<int>(documentState.project.tracks.size())))
        return;

    auto updatedTrack = documentState.project.tracks[static_cast<size_t>(selected)];
    updatedTrack.rackVst.clear();
    if (updatedTrack.instrumentMode.trim().equalsIgnoreCase("VSTI Rack"))
        updatedTrack.instrumentMode = "General MIDI";
    updatedTrack.vstiStatePath.clear();
    updatedTrack.vstiStateBase64.clear();
    updatedTrack.vstiParameters.clear();
    updatedTrack.synthProfile = inferSynthProfile(updatedTrack.instrument, updatedTrack.midiProgram);
    applyTrackStateEdit(selected, updatedTrack, "Clear Rack Assignment");
    statusLabel.setText("Cleared native rack assignment for " + updatedTrack.name + ".", juce::dontSendNotification);
}

void StudioShellComponent::materialiseSelectedTrackRackAssignment()
{
    const auto selected = getSelectedTrackIndex();
    if (!juce::isPositiveAndBelow(selected, static_cast<int>(documentState.project.tracks.size())))
        return;

    auto updatedTrack = documentState.project.tracks[static_cast<size_t>(selected)];
    if (!materialiseNativeInstrumentTrack(documentState.project, updatedTrack))
    {
        juce::AlertWindow::showMessageBoxAsync(juce::AlertWindow::InfoIcon,
                                               "No Native Rack Available",
                                               "Could not find a suitable native rack instrument for the selected track.");
        return;
    }

    applyTrackStateEdit(selected, updatedTrack, "Auto Assign Rack");
    statusLabel.setText("Auto-assigned native rack for " + updatedTrack.name + ".", juce::dontSendNotification);
}

void StudioShellComponent::clearSelectedTrackRenderedAudioPath()
{
    const auto selected = getSelectedTrackIndex();
    if (!juce::isPositiveAndBelow(selected, static_cast<int>(documentState.project.tracks.size())))
        return;

    auto updatedTrack = documentState.project.tracks[static_cast<size_t>(selected)];
    if (updatedTrack.renderedAudioPath.trim().isEmpty())
        return;

    updatedTrack.renderedAudioPath.clear();
    applyTrackStateEdit(selected, updatedTrack, "Clear Render Path");
    statusLabel.setText("Cleared rendered audio path for " + updatedTrack.name + ".", juce::dontSendNotification);
}

void StudioShellComponent::scheduleSelectedTrackRackPreviewWarmup(int delayMs)
{
    if (rackPreviewWarmupPending || projectPreviewRunning || rackPreviewRunning)
        return;

    const auto selected = getSelectedTrackIndex();
    const auto* track = getSelectedTrack();
    if (selected < 0 || track == nullptr)
        return;

    if (resolveRackPluginPath(documentState.project, *track).isEmpty())
        return;

    rackPreviewWarmupPending = true;
    juce::Timer::callAfterDelay(juce::jmax(0, delayMs),
                                [safeThis = juce::Component::SafePointer<StudioShellComponent>(this), selected]
                                {
                                    if (safeThis == nullptr)
                                        return;

                                    safeThis->rackPreviewWarmupPending = false;

                                    if (safeThis->projectPreviewRunning || safeThis->rackPreviewRunning)
                                        return;

                                    const auto currentIndex = safeThis->getSelectedTrackIndex();
                                    const auto* currentTrack = safeThis->getSelectedTrack();
                                    if (currentTrack == nullptr || currentIndex != selected)
                                        return;

                                    const auto result = safeThis->ensureNativeAudioEnginePrepared(false);
                                    if (result.failed())
                                    {
                                        safeThis->statusLabel.setText("Preview warmup failed: " + result.getErrorMessage(),
                                                                      juce::dontSendNotification);
                                    }
                                });
}

bool StudioShellComponent::virtualPianoShortcutsEnabled() const
{
    for (auto* focused = juce::Component::getCurrentlyFocusedComponent();
         focused != nullptr;
         focused = focused->getParentComponent())
    {
        if (dynamic_cast<juce::TextEditor*>(focused) != nullptr
            || dynamic_cast<juce::ComboBox*>(focused) != nullptr)
        {
            return false;
        }
    }

    return true;
}

bool StudioShellComponent::tryHandleVirtualPianoShortcut(const juce::KeyPress& key)
{
    const auto modifiers = key.getModifiers();
    if (modifiers.isAltDown() || modifiers.isCtrlDown() || modifiers.isCommandDown())
        return false;

    if (!virtualPianoShortcutsEnabled())
        return false;

    const auto shortcut = normaliseVirtualPianoShortcutKey(key);
    if (shortcut.isEmpty())
        return false;

    for (const auto& spec : virtualPianoKeySpecs())
    {
        if (shortcut.equalsIgnoreCase(spec.primary))
        {
            insertVirtualKeyboardNote(spec.pitch, true);
            return true;
        }

        for (const auto* alias : spec.aliases)
        {
            if (shortcut.equalsIgnoreCase(alias))
            {
                insertVirtualKeyboardNote(spec.pitch, true);
                return true;
            }
        }
    }

    return false;
}

void StudioShellComponent::insertVirtualKeyboardNote(int pitch, bool fromShortcut)
{
    juce::ignoreUnused(fromShortcut);

    if (documentState.project.tracks.empty())
        addTrack();

    if (getSelectedTrackIndex() < 0 && !documentState.project.tracks.empty())
        setSelectedTrackIndex(0);

    const auto trackIndex = getSelectedTrackIndex();
    if (!juce::isPositiveAndBelow(trackIndex, static_cast<int>(documentState.project.tracks.size())))
        return;

    auto sectionIndex = getSelectedMidiSectionIndex();
    if (!juce::isPositiveAndBelow(sectionIndex, static_cast<int>(documentState.project.midiSections.size()))
        || documentState.project.midiSections[static_cast<size_t>(sectionIndex)].trackIndex != trackIndex)
    {
        ensureSelectedMidiSectionForTrack(trackIndex);
        sectionIndex = getSelectedMidiSectionIndex();
    }

    auto createdSection = false;
    if (!juce::isPositiveAndBelow(sectionIndex, static_cast<int>(documentState.project.midiSections.size())))
    {
        auto updatedProject = documentState.project;
        const auto patternLength = defaultPatternLengthTicks(updatedProject);
        const auto snapTicks = arrangementSnapTickLength(updatedProject);

        MidiPattern pattern;
        pattern.id = juce::Uuid().toString();
        pattern.name = updatedProject.tracks[static_cast<size_t>(trackIndex)].name.trim().isNotEmpty()
            ? updatedProject.tracks[static_cast<size_t>(trackIndex)].name.trim() + " Pattern"
            : "Pattern " + juce::String(static_cast<int>(updatedProject.midiPatterns.size()) + 1);
        pattern.lengthTicks = patternLength;

        MidiSection section;
        section.trackIndex = trackIndex;
        section.startTick = snapTicks > 0
            ? (updatedProject.playheadTick / snapTicks) * snapTicks
            : updatedProject.playheadTick;
        section.lengthTicks = patternLength;
        section.name = pattern.name;
        section.patternId = pattern.id;

        updatedProject.midiPatterns.push_back(std::move(pattern));
        updatedProject.midiSections.push_back(std::move(section));
        updatedProject.recalculateTimeFields();
        sectionIndex = static_cast<int>(updatedProject.midiSections.size()) - 1;
        applyProjectStateEdit(updatedProject, "Create Pattern");
        setSelectedTrackIndex(trackIndex);
        setSelectedMidiSectionIndex(sectionIndex, true);
        createdSection = true;
    }

    auto updatedProject = documentState.project;
    auto* pattern = findMidiPattern(updatedProject,
                                    updatedProject.midiSections[static_cast<size_t>(sectionIndex)].patternId);
    if (pattern == nullptr)
        return;

    const auto clampedPitch = juce::jlimit(12, 84, pitch);
    const auto durationTick = kTicksPerBeat / 2;
    auto cursorTick = 0;
    for (const auto& note : pattern->notes)
        cursorTick = juce::jmax(cursorTick, note.startTick + note.durationTick);

    MidiNote note;
    note.startTick = cursorTick;
    note.durationTick = durationTick;
    note.pitch = clampedPitch;
    note.velocity = 100;
    pattern->notes.push_back(note);
    pattern->lengthTicks = juce::jmax(pattern->lengthTicks, note.startTick + note.durationTick);
    updatedProject.recalculateTimeFields();

    applyProjectStateEdit(updatedProject, "Insert Live Note");
    setSelectedTrackIndex(trackIndex);
    setSelectedMidiSectionIndex(sectionIndex, true);

    if (fromShortcut && virtualPianoWindowContent != nullptr)
        virtualPianoWindowContent->flashPitch(clampedPitch);

    previewSelectedTrackMidiNoteOn(clampedPitch, 100);
    juce::Timer::callAfterDelay(juce::jlimit(90,
                                             600,
                                             juce::roundToInt(tickToSeconds(durationTick, documentState.project.bpm) * 1000.0)),
                                [safeThis = juce::Component::SafePointer<StudioShellComponent>(this)]
                                {
                                    if (safeThis != nullptr)
                                        safeThis->stopSelectedTrackMidiPreview();
                                });

    if (createdSection)
        statusLabel.setText("Created a new pattern and inserted " + noteNameLabel(clampedPitch) + ".", juce::dontSendNotification);
}

juce::Result StudioShellComponent::previewSelectedTrackMidiNoteOn(int pitch, int velocity)
{
    const auto selected = getSelectedTrackIndex();
    const auto* track = getSelectedTrack();
    if (selected < 0 || track == nullptr)
        return juce::Result::fail("No track selected.");

    auto result = ensureNativeAudioEnginePrepared(projectPreviewRunning || rackPreviewRunning);
    if (result.failed())
    {
        scheduleSelectedTrackRackPreviewWarmup(0);
        return result;
    }

    return nativeVstHost.noteOnAudioEngineTrack(selected,
                                                juce::jlimit(0, 127, pitch),
                                                juce::jlimit(1, 16, track->midiChannel + 1),
                                                static_cast<float>(juce::jlimit(1, 127, velocity)) / 127.0f);
}

void StudioShellComponent::previewSelectedTrackMidiNoteOff(int pitch, int velocity)
{
    const auto selected = getSelectedTrackIndex();
    const auto* track = getSelectedTrack();
    if (selected < 0 || track == nullptr || !nativeVstHost.isReady())
        return;

    nativeVstHost.noteOffAudioEngineTrack(selected,
                                          juce::jlimit(0, 127, pitch),
                                          juce::jlimit(1, 16, track->midiChannel + 1),
                                          static_cast<float>(juce::jlimit(0, 127, velocity)) / 127.0f);
}

void StudioShellComponent::stopSelectedTrackMidiPreview()
{
    const auto selected = getSelectedTrackIndex();
    if (selected < 0 || !nativeVstHost.isReady())
        return;

    nativeVstHost.allNotesOffAudioEngineTrack(selected);
}

void StudioShellComponent::openSelectedTrackRackEditor()
{
    const auto* track = getSelectedTrack();
    if (track == nullptr)
        return;

    const auto trackIndex = getSelectedTrackIndex();
    RackEditorSession* session = nullptr;
    const auto result = ensureRackEditorSessionReadyForTrack(trackIndex, *track, session);
    if (result.failed())
    {
        juce::AlertWindow::showMessageBoxAsync(juce::AlertWindow::WarningIcon,
                                               "Native Host Error",
                                               result.getErrorMessage());
        return;
    }

    jassert(session != nullptr);
    const auto openResult = session != nullptr
        ? nativeVstHost.openAudioEngineTrackEditor(trackIndex)
        : juce::Result::fail("The native editor session could not be prepared.");
    if (openResult.failed())
    {
        juce::AlertWindow::showMessageBoxAsync(juce::AlertWindow::WarningIcon,
                                               "Editor Open Failed",
                                               openResult.getErrorMessage());
        return;
    }

    if (session != nullptr)
        session->editorOpen = true;
    refreshPollingTimerState();
    updateEditorState();
    trackTable.repaint();
    statusLabel.setText("Opened native rack editor for " + track->name + ".", juce::dontSendNotification);
    appendActivityLog("VST Editor", "Opened editor for track: " + track->name);
}

void StudioShellComponent::saveSelectedTrackRackState()
{
    const auto selected = getSelectedTrackIndex();
    if (selected < 0)
        return;

    const auto* track = getSelectedTrack();
    if (track == nullptr)
        return;

    auto* rackEditorSession = findRackEditorSession(selected);
    auto result = ensureNativeAudioEnginePrepared(false);
    if (result.failed())
    {
        juce::AlertWindow::showMessageBoxAsync(juce::AlertWindow::WarningIcon,
                                               "Native Host Error",
                                               result.getErrorMessage());
        return;
    }

    NativeVstHostSession::RackParameterSnapshot rackSnapshot;
    if (nativeVstHost.queryAudioEngineTrackParameterSnapshot(selected, rackSnapshot).wasOk())
    {
        if (rackEditorSession != nullptr)
            rackEditorSession->lastStateGeneration = rackSnapshot.stateGeneration;
        syncTrackRackParametersFromValues(selected, rackSnapshot.parameterValues, true);
    }

    const auto stateFile = suggestTrackStateFile(selected, *track);
    if (!stateFile.getParentDirectory().createDirectory())
    {
        juce::AlertWindow::showMessageBoxAsync(juce::AlertWindow::WarningIcon,
                                               "State Save Failed",
                                               "Could not create the native state folder:\n" + stateFile.getParentDirectory().getFullPathName());
        return;
    }

    result = nativeVstHost.saveAudioEngineTrackPluginState(selected, stateFile.getFullPathName());
    if (result.failed())
    {
        juce::AlertWindow::showMessageBoxAsync(juce::AlertWindow::WarningIcon,
                                               "State Save Failed",
                                               result.getErrorMessage());
        return;
    }

    auto updatedTrack = *track;
    updatedTrack.vstiStatePath = stateFile.getFullPathName();
    applyTrackStateEdit(selected, updatedTrack, "Save Rack State");
    statusLabel.setText("Saved native rack state for " + track->name + ".", juce::dontSendNotification);
}

void StudioShellComponent::playSelectedTrackThroughRack()
{
    const auto* track = getSelectedTrack();
    if (track == nullptr)
        return;

    if (track->notes.empty())
    {
        juce::AlertWindow::showMessageBoxAsync(juce::AlertWindow::WarningIcon,
                                               "No Notes To Play",
                                               "The selected track does not have any MIDI notes yet.");
        return;
    }

    const auto selected = getSelectedTrackIndex();
    if (projectPreviewRunning)
        stopRackPreview();

    auto previewProject = documentState.project;
    for (int trackIndex = 0; trackIndex < static_cast<int>(previewProject.tracks.size()); ++trackIndex)
    {
        if (trackIndex == selected)
            continue;

        previewProject.tracks[static_cast<size_t>(trackIndex)].mute = true;
        previewProject.tracks[static_cast<size_t>(trackIndex)].solo = false;
    }
    previewProject.tracks[static_cast<size_t>(selected)].mute = false;
    previewProject.tracks[static_cast<size_t>(selected)].solo = true;
    previewProject.recalculateTimeFields();

    auto result = nativeVstHost.ensureCreated();
    if (result.failed())
    {
        juce::AlertWindow::showMessageBoxAsync(juce::AlertWindow::WarningIcon,
                                               "Project Playback Setup Failed",
                                               result.getErrorMessage());
        return;
    }

    result = nativeVstHost.setAudioEngineState(previewProject);
    if (result.failed())
    {
        juce::AlertWindow::showMessageBoxAsync(juce::AlertWindow::WarningIcon,
                                               "Project Playback Setup Failed",
                                               result.getErrorMessage());
        return;
    }

    result = nativeVstHost.startAudioEngine(documentState.project.playheadTick);
    if (result.failed())
    {
        juce::AlertWindow::showMessageBoxAsync(juce::AlertWindow::WarningIcon,
                                               "Playback Failed",
                                               result.getErrorMessage());
        return;
    }

    rackPreviewRunning = true;
    projectPreviewRunning = false;
    pendingLiveRackParameterEngineSyncTrack = -1;
    audioEngineStateValid = false;
    audioEngineStateDirty = true;
    refreshPollingTimerState();
    updateEditorState();
    statusLabel.setText("Playing selected rack track from tick " + juce::String(documentState.project.playheadTick) + ".", juce::dontSendNotification);
    appendActivityLog("Playback",
                      "Started selected-track preview\nTrack: "
                          + track->name
                          + "\nTick: "
                          + juce::String(documentState.project.playheadTick));
}

void StudioShellComponent::playFullProjectThroughNativeEngine()
{
    if (documentState.project.tracks.empty() && !documentState.project.metronomeEnabled)
        return;

    if (rackPreviewRunning)
        stopRackPreview();

    auto result = nativeVstHost.ensureCreated();
    if (result.failed())
    {
        juce::AlertWindow::showMessageBoxAsync(juce::AlertWindow::WarningIcon,
                                               "Native Host Error",
                                               result.getErrorMessage());
        return;
    }

    if (!audioEngineStateValid || audioEngineStateDirty)
    {
        result = nativeVstHost.setAudioEngineState(documentState.project);
        if (result.failed())
        {
            juce::AlertWindow::showMessageBoxAsync(juce::AlertWindow::WarningIcon,
                                                   "Project Playback Setup Failed",
                                                   result.getErrorMessage());
            return;
        }

        audioEngineStateValid = true;
        audioEngineStateDirty = false;
    }

    result = nativeVstHost.startAudioEngine(documentState.project.playheadTick);
    if (result.failed())
    {
        juce::AlertWindow::showMessageBoxAsync(juce::AlertWindow::WarningIcon,
                                               "Project Playback Failed",
                                               result.getErrorMessage());
        return;
    }

    rackPreviewRunning = false;
    projectPreviewRunning = true;
    playbackUiTickCounter = 0;
    pendingLiveRackParameterEngineSyncTrack = -1;
    refreshPollingTimerState();
    updateEditorState();
    statusLabel.setText("Playing native project preview from tick " + juce::String(documentState.project.playheadTick) + ".", juce::dontSendNotification);
    appendActivityLog("Playback", "Started project playback\nTick: " + juce::String(documentState.project.playheadTick));
}

void StudioShellComponent::stopRackPreview()
{
    juce::Result result = juce::Result::ok();
    if (projectPreviewRunning || rackPreviewRunning)
        result = nativeVstHost.stopAudioEngine();

    if (result.failed())
    {
        juce::AlertWindow::showMessageBoxAsync(juce::AlertWindow::WarningIcon,
                                               "Stop Failed",
                                               result.getErrorMessage());
        return;
    }

    rackPreviewRunning = false;
    projectPreviewRunning = false;
    playbackUiTickCounter = 0;
    pendingLiveRackParameterEngineSyncTrack = -1;
    audioEngineStateValid = false;
    audioEngineStateDirty = true;
    refreshPollingTimerState();
    std::fill(trackMeterLevels.begin(), trackMeterLevels.end(), 0.0f);
    if (mixerComponent != nullptr)
        mixerComponent->refreshMeters();
    repaintTrackVolumeMeters();
    updateEditorState();
    statusLabel.setText("Stopped native preview playback.", juce::dontSendNotification);
    appendActivityLog("Playback", "Stopped playback.");
}

void StudioShellComponent::undo()
{
    if (!undoManager.canUndo())
        return;

    undoManager.undo();
    markDirty();
    refreshUi();
    statusLabel.setText("Undo.", juce::dontSendNotification);
}

void StudioShellComponent::redo()
{
    if (!undoManager.canRedo())
        return;

    undoManager.redo();
    markDirty();
    refreshUi();
    statusLabel.setText("Redo.", juce::dontSendNotification);
}

void StudioShellComponent::quantizeSelectedNotes()
{
    if (pianoRoll != nullptr && pianoRoll->quantizeSelected())
        statusLabel.setText("Quantized selected notes.", juce::dontSendNotification);
}

void StudioShellComponent::setEditorToolMode(EditorToolMode mode)
{
    if (editorToolMode == mode)
        return;

    editorToolMode = mode;
    if (arrangementOverview != nullptr)
        arrangementOverview->refreshFromModel();
    if (floatingArrangementOverview != nullptr)
        floatingArrangementOverview->refreshFromModel();
    if (pianoRoll != nullptr)
        pianoRoll->setToolMode(mode);
    if (floatingPianoRollWorkspace != nullptr)
        floatingPianoRollWorkspace->refreshFromModel();
    if (panelsWindowContent != nullptr)
        panelsWindowContent->refreshFromModel();
    statusLabel.setText("Editor tool: " + editorToolModeLabel(mode) + ".", juce::dontSendNotification);
    if (pianoRoll != nullptr)
        pianoRoll->grabKeyboardFocus();
}

void StudioShellComponent::resetRackHostTracking()
{
    loadedRackEditorOpen = false;
    rackPreviewWarmupPending = false;
    pendingLiveRackParameterEngineSyncTrack = -1;
    refreshPollingTimerState();
}

bool StudioShellComponent::syncTrackRackParametersFromValues(int trackIndex,
                                                             const juce::NamedValueSet& hostParameters,
                                                             bool skipParameterEnginePush)
{
    if (!juce::isPositiveAndBelow(trackIndex, static_cast<int>(documentState.project.tracks.size())))
        return false;

    if (hostParameters.size() <= 0)
        return false;

    auto updatedTrack = documentState.project.tracks[static_cast<size_t>(trackIndex)];
    if (parameterSetSignature(updatedTrack.vstiParameters) == parameterSetSignature(hostParameters))
        return false;

    updatedTrack.vstiParameters = hostParameters;
    applyTrackStateSyncNoUndo(trackIndex,
                              updatedTrack,
                              isTrackBeingLiveEdited(trackIndex) && projectPreviewRunning && !skipParameterEnginePush,
                              skipParameterEnginePush);
    return true;
}

void StudioShellComponent::replaceTrackStateNoUndo(int trackIndex, const TrackState& updatedTrack)
{
    setTrackStateInternal(trackIndex, updatedTrack);
}

void StudioShellComponent::applyTrackStateSyncNoUndo(int trackIndex,
                                                     const TrackState& updatedTrack,
                                                     bool deferParameterEnginePush,
                                                     bool skipParameterEnginePush)
{
    if (trackIndex < 0 || trackIndex >= static_cast<int>(documentState.project.tracks.size()))
        return;

    auto before = documentState.project.tracks[static_cast<size_t>(trackIndex)];
    auto after = updatedTrack;
    normaliseTrack(before);
    normaliseTrack(after);

    if (fingerprint(before) == fingerprint(after))
        return;

    markDirty();
    setTrackStateInternal(trackIndex, after, deferParameterEnginePush, skipParameterEnginePush);
}

void StudioShellComponent::applyTrackStateEdit(int trackIndex, const TrackState& updatedTrack, const juce::String& actionName)
{
    if (trackIndex < 0 || trackIndex >= static_cast<int>(documentState.project.tracks.size()))
        return;

    auto before = documentState.project.tracks[static_cast<size_t>(trackIndex)];
    auto after = updatedTrack;
    normaliseTrack(before);
    normaliseTrack(after);

    if (fingerprint(before) == fingerprint(after))
        return;

    markDirty();
    undoManager.perform(new TrackEditAction([this, trackIndex] (const TrackState& track)
                                            {
                                                setTrackStateInternal(trackIndex, track);
                                            },
                                            before,
                                            after),
                        actionName);
}

void StudioShellComponent::applySelectedTrackMutation(std::function<void(TrackState&)> mutation,
                                                      const juce::String& actionName)
{
    const auto selected = getSelectedTrackIndex();
    if (selected < 0)
        return;

    auto updatedTrack = documentState.project.tracks[static_cast<size_t>(selected)];
    mutation(updatedTrack);
    applyTrackStateEdit(selected, updatedTrack, actionName);
}

void StudioShellComponent::applyProjectStateEdit(const ProjectState& updatedProject, const juce::String& actionName)
{
    auto before = documentState.project;
    auto after = updatedProject;
    normaliseProject(before);
    normaliseProject(after);

    if (fingerprint(before) == fingerprint(after))
        return;

    markDirty();
    undoManager.perform(new ProjectEditAction([this] (const ProjectState& project)
                                              {
                                                  setProjectStateInternal(project);
                                              },
                                              before,
                                              after),
                        actionName);
}

void StudioShellComponent::setTrackStateInternal(int trackIndex,
                                                 const TrackState& updatedTrack,
                                                 bool deferParameterEnginePush,
                                                 bool skipParameterEnginePush)
{
    if (trackIndex < 0 || trackIndex >= static_cast<int>(documentState.project.tracks.size()))
        return;

    const auto previousProject = documentState.project;
    const auto previousTrack = documentState.project.tracks[static_cast<size_t>(trackIndex)];
    documentState.project.tracks[static_cast<size_t>(trackIndex)] = updatedTrack;
    normaliseProject(documentState.project);
    audioEngineStateDirty = true;

    const auto& currentTrack = documentState.project.tracks[static_cast<size_t>(trackIndex)];
    const bool rackBindingChanged = !previousTrack.rackVst.equalsIgnoreCase(currentTrack.rackVst)
        || !previousTrack.instrumentMode.equalsIgnoreCase(currentTrack.instrumentMode)
        || !previousTrack.trackType.equalsIgnoreCase(currentTrack.trackType)
        || !previousTrack.vstiStatePath.equalsIgnoreCase(currentTrack.vstiStatePath);
    const bool noteContentChanged = noteTransportSignature(previousTrack.notes) != noteTransportSignature(currentTrack.notes);
    const bool parameterContentChanged = parameterSetSignature(previousTrack.vstiParameters) != parameterSetSignature(currentTrack.vstiParameters);
    const bool mixStateChanged = previousTrack.volume != currentTrack.volume
        || previousTrack.pan != currentTrack.pan
        || previousTrack.vstiOutputGainDb != currentTrack.vstiOutputGainDb
        || previousTrack.mute != currentTrack.mute
        || previousTrack.solo != currentTrack.solo
        || projectTrackIsAudible(previousProject, trackIndex) != projectTrackIsAudible(documentState.project, trackIndex);
    const auto previousRackPath = resolveRackPluginPath(previousProject, previousTrack);
    const auto currentRackPath = resolveRackPluginPath(documentState.project, currentTrack);
    const bool instrumentActivationChanged = (!previousRackPath.isEmpty() || !currentRackPath.isEmpty())
        && (previousTrack.notes.empty() != currentTrack.notes.empty());
    const bool requiresFullEngineState = rackBindingChanged
        || !previousTrack.renderedAudioPath.equalsIgnoreCase(currentTrack.renderedAudioPath)
        || instrumentActivationChanged;
    const bool parameterOnlyContentChanged = parameterContentChanged
        && !rackBindingChanged
        && !noteContentChanged
        && !mixStateChanged
        && !instrumentActivationChanged
        && previousTrack.renderedAudioPath.equalsIgnoreCase(currentTrack.renderedAudioPath);
    const bool lightweightTrackUiRefresh = !parameterOnlyContentChanged
        && !rackBindingChanged
        && !mixStateChanged
        && !instrumentActivationChanged
        && previousTrack.renderedAudioPath.equalsIgnoreCase(currentTrack.renderedAudioPath);

    if (rackBindingChanged)
        closeRackEditorSession(trackIndex);

    if (parameterOnlyContentChanged)
    {
        if (!isTrackBeingLiveEdited(trackIndex))
        {
            refreshProjectSummaryLabels();
            refreshFloatingWindows(false);
        }
    }
    else if (lightweightTrackUiRefresh)
    {
        refreshProjectSummaryLabels();
        trackTable.repaint();
        if (trackIndex == getSelectedTrackIndex())
            refreshInspector();
        updateEditorState();
    }
    else
    {
        refreshUi();
    }

    if ((projectPreviewRunning || rackPreviewRunning) && nativeVstHost.isReady())
    {
        auto result = juce::Result::ok();
        bool usedIncrementalUpdate = false;
        bool deferredParameterSync = false;

        if (requiresFullEngineState)
        {
            result = nativeVstHost.setAudioEngineState(documentState.project, true);
        }
        else
        {
            if (mixStateChanged)
            {
                result = nativeVstHost.setAudioEngineTrackMixState(documentState.project, trackIndex, currentTrack);
                usedIncrementalUpdate = true;
            }

            if (result.wasOk() && noteContentChanged)
            {
                result = nativeVstHost.setAudioEngineTrackNotes(trackIndex, documentState.project, currentTrack);
                usedIncrementalUpdate = true;
            }

            if (result.wasOk() && parameterContentChanged)
            {
                if (skipParameterEnginePush)
                {
                    usedIncrementalUpdate = true;
                }
                else if (deferParameterEnginePush)
                {
                    pendingLiveRackParameterEngineSyncTrack = trackIndex;
                    deferredParameterSync = true;
                }
                else
                {
                    result = nativeVstHost.setAudioEngineTrackParameters(trackIndex, currentTrack);
                    usedIncrementalUpdate = true;
                }
            }

            if (result.failed() || (!usedIncrementalUpdate && !deferredParameterSync))
                result = result.wasOk() && !usedIncrementalUpdate
                    ? juce::Result::ok()
                    : nativeVstHost.setAudioEngineState(documentState.project, true);
        }

        if (result.failed())
        {
            statusLabel.setText("Native preview update failed: " + result.getErrorMessage(), juce::dontSendNotification);
        }
        else if (requiresFullEngineState || usedIncrementalUpdate)
        {
            if (requiresFullEngineState || (usedIncrementalUpdate && parameterContentChanged))
                pendingLiveRackParameterEngineSyncTrack = -1;
            audioEngineStateValid = true;
            audioEngineStateDirty = false;
        }
    }
}

void StudioShellComponent::setProjectStateInternal(const ProjectState& updatedProject)
{
    if (updatedProject.tracks.size() != documentState.project.tracks.size())
    {
        resetRackHostTracking();
        closeAllRackEditorSessions();
    }

    documentState.project = updatedProject;
    normaliseProject(documentState.project);
    audioEngineStateDirty = true;
    refreshUi();

    if ((projectPreviewRunning || rackPreviewRunning) && nativeVstHost.isReady())
    {
        const auto result = nativeVstHost.setAudioEngineState(documentState.project, true);
        if (result.failed())
            statusLabel.setText("Native preview update failed: " + result.getErrorMessage(), juce::dontSendNotification);
        else
        {
            pendingLiveRackParameterEngineSyncTrack = -1;
            audioEngineStateValid = true;
            audioEngineStateDirty = false;
        }
    }
}

juce::File StudioShellComponent::suggestProjectFile() const
{
    if (currentProjectFile != juce::File())
        return currentProjectFile;
    return juce::File::getSpecialLocation(juce::File::userDocumentsDirectory)
        .getChildFile("ai-music-studio-native.aims");
}

juce::File StudioShellComponent::suggestTrackStateFile(int trackIndex, const TrackState& track) const
{
    auto stateRoot = currentProjectFile.existsAsFile()
        ? currentProjectFile.getParentDirectory().getChildFile(currentProjectFile.getFileNameWithoutExtension() + "_native_states")
        : juce::File::getSpecialLocation(juce::File::userDocumentsDirectory).getChildFile("Mutagen Native States");

    auto safeTrackName = juce::File::createLegalFileName(track.name.trim());
    if (safeTrackName.isEmpty())
        safeTrackName = "track";

    return stateRoot.getChildFile(juce::String(trackIndex + 1) + "_" + safeTrackName).withFileExtension(".vststate");
}

juce::File StudioShellComponent::ensureProjectSuffix(const juce::File& file) const
{
    if (file.hasFileExtension(".aims") || file.hasFileExtension(".json"))
        return file;
    return file.withFileExtension(".aims");
}

int StudioShellComponent::findPreferredSampleTrackIndex() const
{
    const auto selected = getSelectedTrackIndex();
    if (juce::isPositiveAndBelow(selected, static_cast<int>(documentState.project.tracks.size()))
        && documentState.project.tracks[static_cast<size_t>(selected)].trackType.equalsIgnoreCase("sample"))
    {
        return selected;
    }

    for (int trackIndex = 0; trackIndex < static_cast<int>(documentState.project.tracks.size()); ++trackIndex)
    {
        if (documentState.project.tracks[static_cast<size_t>(trackIndex)].trackType.equalsIgnoreCase("sample"))
            return trackIndex;
    }

    return -1;
}

int StudioShellComponent::getSelectedTrackIndex() const
{
    return trackTable.getSelectedRow();
}

void StudioShellComponent::setSelectedTrackIndex(int row)
{
    if (row < 0 || row >= static_cast<int>(documentState.project.tracks.size()))
    {
        if (trackTable.getSelectedRow() >= 0)
            trackTable.deselectAllRows();
        return;
    }

    if (trackTable.getSelectedRow() == row)
        return;

    trackTable.selectRow(row);
}

int StudioShellComponent::getSelectedMidiSectionIndex() const noexcept
{
    return selectedMidiSectionIndex;
}

void StudioShellComponent::setSelectedMidiSectionIndex(int sectionIndex, bool bringTrackIntoFocus)
{
    const auto validSection = juce::isPositiveAndBelow(sectionIndex, static_cast<int>(documentState.project.midiSections.size()))
        ? sectionIndex
        : -1;
    const auto previousSection = selectedMidiSectionIndex;
    selectedMidiSectionIndex = validSection;

    if (selectedMidiSectionIndex >= 0 && bringTrackIntoFocus)
    {
        const auto trackIndex = documentState.project.midiSections[static_cast<size_t>(selectedMidiSectionIndex)].trackIndex;
        if (trackIndex != getSelectedTrackIndex())
            setSelectedTrackIndex(trackIndex);
    }

    if (previousSection != selectedMidiSectionIndex)
        updateEditorState();
}

void StudioShellComponent::ensureSelectedMidiSectionForTrack(int trackIndex)
{
    if (!juce::isPositiveAndBelow(trackIndex, static_cast<int>(documentState.project.tracks.size())))
    {
        selectedMidiSectionIndex = -1;
        return;
    }

    if (juce::isPositiveAndBelow(selectedMidiSectionIndex, static_cast<int>(documentState.project.midiSections.size())))
    {
        const auto& section = documentState.project.midiSections[static_cast<size_t>(selectedMidiSectionIndex)];
        if (section.trackIndex == trackIndex)
            return;
    }

    selectedMidiSectionIndex = -1;
    for (int sectionIndex = 0; sectionIndex < static_cast<int>(documentState.project.midiSections.size()); ++sectionIndex)
    {
        if (documentState.project.midiSections[static_cast<size_t>(sectionIndex)].trackIndex == trackIndex)
        {
            selectedMidiSectionIndex = sectionIndex;
            break;
        }
    }
}

void StudioShellComponent::focusMidiSectionInPianoRoll(int sectionIndex)
{
    setSelectedMidiSectionIndex(sectionIndex, true);
    focusPianoRollPanel();
}

int StudioShellComponent::getSelectedSampleAssetIndex() const
{
    return sampleAssetList.getSelectedRow();
}

void StudioShellComponent::setSelectedSampleAssetIndex(int row)
{
    if (row < 0 || row >= static_cast<int>(documentState.project.sampleAssets.size()))
    {
        if (sampleAssetList.getSelectedRow() >= 0)
            sampleAssetList.deselectAllRows();
        return;
    }

    if (sampleAssetList.getSelectedRow() == row)
        return;

    sampleAssetList.selectRow(row);
}

TrackState* StudioShellComponent::getSelectedTrack()
{
    const auto selected = getSelectedTrackIndex();
    if (selected < 0 || selected >= static_cast<int>(documentState.project.tracks.size()))
        return nullptr;
    return &documentState.project.tracks[static_cast<size_t>(selected)];
}

const TrackState* StudioShellComponent::getSelectedTrack() const
{
    const auto selected = getSelectedTrackIndex();
    if (selected < 0 || selected >= static_cast<int>(documentState.project.tracks.size()))
        return nullptr;
    return &documentState.project.tracks[static_cast<size_t>(selected)];
}

MainWindow::MainWindow(const juce::File& startupProject)
    : juce::DocumentWindow("Mutagen",
                           juce::Colour::fromRGB(11, 13, 17),
                           juce::DocumentWindow::allButtons)
{
    setUsingNativeTitleBar(true);
    setResizable(true, true);
    setResizeLimits(1080, 720, 2400, 1800);
    setMenuBar(this);

    shell = new StudioShellComponent();
    setContentOwned(shell, true);
    centreWithSize(kDefaultWindowWidth, kDefaultWindowHeight);
    setVisible(true);

    if (startupProject.existsAsFile())
        shell->openProjectFile(startupProject);
}

MainWindow::~MainWindow()
{
    setMenuBar(nullptr);
}

juce::StringArray MainWindow::getMenuBarNames()
{
    return { "File", "Edit", "Settings", "Windows" };
}

juce::PopupMenu MainWindow::getMenuForIndex(int topLevelMenuIndex, const juce::String& menuName)
{
    juce::ignoreUnused(menuName);

    juce::PopupMenu menu;
    if (shell == nullptr)
        return menu;

    switch (topLevelMenuIndex)
    {
        case 0:
            menu.addItem(menuFileNew, "New Project");
            menu.addItem(menuFileOpen, "Open Project...");
            menu.addItem(menuFileSave, "Save Project");
            menu.addItem(menuFileSaveAs, "Save Project As...");
            menu.addSeparator();
            menu.addItem(menuFileImportMidi, "Import MIDI...");
            menu.addItem(menuFileImportSample, "Import Sample...");
            menu.addSeparator();
            menu.addItem(menuFileExportWav, "Export Sequence as WAV...");
            menu.addItem(menuFileExportTrackWav, "Export Selected Track as WAV...");
            menu.addItem(menuFileExportStems, "Export Stems...");
            menu.addItem(menuFileExportMidi, "Export MIDI...");
            break;

        case 1:
            menu.addItem(menuEditUndo, "Undo", shell->undoManager.canUndo(), false);
            menu.addItem(menuEditRedo, "Redo", shell->undoManager.canRedo(), false);
            menu.addSeparator();
            menu.addItem(menuEditQuantize, "Quantize Selected Notes");
            menu.addSeparator();
            menu.addItem(menuEditSelectAll, "Select All Notes");
            menu.addItem(menuEditCopy, "Copy Notes");
            menu.addItem(menuEditCut, "Cut Notes");
            menu.addItem(menuEditDelete, "Delete Notes");
            menu.addItem(menuEditDuplicate, "Duplicate Notes");
            menu.addItem(menuEditPaste, "Paste Notes");
            break;

        case 2:
            menu.addItem(menuSettingsAudio, "Audio Settings...");
            menu.addItem(menuSettingsVstFolders, "VST Folder Manager...");
            menu.addItem(menuSettingsAi, "AI Settings...");
            {
                juce::PopupMenu themesMenu;
                for (int themeIndex = 0; themeIndex < shell->getThemeCount(); ++themeIndex)
                {
                    themesMenu.addItem(menuThemeBase + themeIndex,
                                       shell->getThemeName(themeIndex),
                                       true,
                                       themeIndex == shell->getCurrentThemeIndex());
                }
                menu.addSeparator();
                menu.addSubMenu("Themes", themesMenu);
            }
            break;

        case 3:
            menu.addItem(menuWindowsPanels, "Show Panels Window", true, shell->isPanelsWindowVisible());
            menu.addItem(menuWindowsTransport, "Show Transport Window", true, shell->isTransportWindowVisible());
            menu.addItem(menuWindowsMixer, "Show Mixer Window", true, shell->isMixerWindowVisible());
            menu.addItem(menuWindowsAudio, "Show Audio Window", true, shell->isAudioWindowVisible());
            menu.addItem(menuWindowsTracks, "Show Tracks Window", true, shell->isTracksWindowVisible());
            menu.addItem(menuWindowsRackBrowser, "Show Rack Browser Window", true, shell->isRackBrowserWindowVisible());
            menu.addItem(menuWindowsRenderManager, "Show Render Manager Window", true, shell->isRenderManagerWindowVisible());
            menu.addSeparator();
            menu.addItem(menuWindowsArrangement, "Show Arrangement Window", true, shell->isArrangementWindowVisible());
            menu.addItem(menuWindowsAutomation, "Show Automation Window", true, shell->isAutomationWindowVisible());
            menu.addItem(menuWindowsSamples, "Show Samples Window", true, shell->isSamplesWindowVisible());
            menu.addItem(menuWindowsPianoRoll, "Show Piano Roll Window", true, shell->isPianoRollWindowVisible());
            menu.addItem(menuWindowsVirtualPiano, "Show Virtual Piano Window", true, shell->isVirtualPianoWindowVisible());
            menu.addItem(menuWindowsActivityLog, "Show Activity Log Window", true, shell->isActivityLogWindowVisible());
            break;

        default:
            break;
    }

    return menu;
}

void MainWindow::menuItemSelected(int menuItemID, int topLevelMenuIndex)
{
    juce::ignoreUnused(topLevelMenuIndex);
    if (shell == nullptr)
        return;

    if (menuItemID >= menuThemeBase && menuItemID < menuThemeBase + shell->getThemeCount())
    {
        shell->setThemeIndex(menuItemID - menuThemeBase);
        return;
    }

    switch (menuItemID)
    {
        case menuFileNew: shell->createNewProject(); break;
        case menuFileOpen: shell->promptOpenProject(); break;
        case menuFileSave: shell->saveProject(); break;
        case menuFileSaveAs: shell->saveProjectAs(); break;
        case menuFileImportMidi: shell->promptImportMidi(); break;
        case menuFileImportSample: shell->promptImportSample(); break;
        case menuFileExportWav: shell->promptExportWav(); break;
        case menuFileExportTrackWav: shell->promptExportSelectedTrackWav(); break;
        case menuFileExportStems: shell->promptExportProjectStems(); break;
        case menuFileExportMidi: shell->promptExportMidi(); break;
        case menuEditUndo: shell->undo(); break;
        case menuEditRedo: shell->redo(); break;
        case menuEditQuantize: shell->quantizeSelectedNotes(); break;
        case menuEditSelectAll: shell->selectAllNotesFromMenu(); break;
        case menuEditCopy: shell->copyNotesFromMenu(); break;
        case menuEditCut: shell->cutNotesFromMenu(); break;
        case menuEditDelete: shell->deleteNotesFromMenu(); break;
        case menuEditDuplicate: shell->duplicateNotesFromMenu(); break;
        case menuEditPaste: shell->pasteNotesFromMenu(); break;
        case menuSettingsAudio: shell->showAudioSettingsWindow(); break;
        case menuSettingsVstFolders: shell->showVstFolderManagerWindow(); break;
        case menuSettingsAi: shell->showAiSettingsDialog(); break;
        case menuWindowsPanels: shell->setPanelsWindowVisible(!shell->isPanelsWindowVisible()); break;
        case menuWindowsTransport: shell->setTransportWindowVisible(!shell->isTransportWindowVisible()); break;
        case menuWindowsMixer: shell->setMixerWindowVisible(!shell->isMixerWindowVisible()); break;
        case menuWindowsAudio:
            if (shell->isAudioWindowVisible())
                shell->setAudioWindowVisible(false);
            else
                shell->focusAudioPanel();
            break;
        case menuWindowsTracks:
            if (shell->isTracksWindowVisible())
                shell->setTracksWindowVisible(false);
            else
                shell->focusTracksPanel();
            break;
        case menuWindowsRackBrowser:
            if (shell->isRackBrowserWindowVisible())
                shell->setRackBrowserWindowVisible(false);
            else
                shell->focusRackBrowserPanel();
            break;
        case menuWindowsRenderManager:
            if (shell->isRenderManagerWindowVisible())
                shell->setRenderManagerWindowVisible(false);
            else
                shell->focusRenderManagerPanel();
            break;
        case menuWindowsArrangement:
            if (shell->isArrangementWindowVisible())
                shell->setArrangementWindowVisible(false);
            else
                shell->focusArrangementPanel();
            break;
        case menuWindowsAutomation:
            if (shell->isAutomationWindowVisible())
                shell->setAutomationWindowVisible(false);
            else
                shell->focusAutomationPanel();
            break;
        case menuWindowsSamples:
            if (shell->isSamplesWindowVisible())
                shell->setSamplesWindowVisible(false);
            else
                shell->focusSamplesPanel();
            break;
        case menuWindowsPianoRoll:
            if (shell->isPianoRollWindowVisible())
                shell->setPianoRollWindowVisible(false);
            else
                shell->focusPianoRollPanel();
            break;
        case menuWindowsVirtualPiano:
            if (shell->isVirtualPianoWindowVisible())
                shell->setVirtualPianoWindowVisible(false);
            else
                shell->focusVirtualPianoPanel();
            break;
        case menuWindowsActivityLog:
            shell->setActivityLogWindowVisible(!shell->isActivityLogWindowVisible());
            break;
        default: break;
    }
}

void MainWindow::closeButtonPressed()
{
    juce::JUCEApplication::getInstance()->systemRequestedQuit();
}

} // namespace aims
