#include "MainWindow.h"
#include "UiStyle.h"

#include <BinaryData.h>

namespace
{
juce::String chooseMutagenUiFont()
{
    const auto available = juce::Font::findAllTypefaceNames();
    constexpr std::array<const char*, 6> candidates
    {
        "Bahnschrift SemiCondensed",
        "Bahnschrift Condensed",
        "Bahnschrift",
        "Agency FB",
        "Franklin Gothic Medium",
        "Segoe UI"
    };

    for (const auto* candidate : candidates)
        if (available.contains(candidate))
            return candidate;

    return {};
}

juce::Image loadMutagenSplashLogo()
{
    juce::MemoryInputStream stream(BinaryData::mutagenlogosource_png,
                                   static_cast<size_t>(BinaryData::mutagenlogosource_pngSize),
                                   false);
    return juce::ImageFileFormat::loadFrom(stream);
}

class StartupSplashComponent final : public juce::Component
{
public:
    explicit StartupSplashComponent(juce::String versionIn)
        : version(std::move(versionIn)),
          logo(loadMutagenSplashLogo())
    {
        setOpaque(true);
        setSize(520, 520);
    }

    void setStatus(const juce::String& newStatus)
    {
        status = newStatus;
        repaint();
    }

    void paint(juce::Graphics& g) override
    {
        juce::ColourGradient background(juce::Colour::fromRGB(8, 11, 14),
                                        0.0f,
                                        0.0f,
                                        juce::Colour::fromRGB(5, 21, 17),
                                        0.0f,
                                        static_cast<float>(getHeight()),
                                        false);
        background.addColour(0.45, juce::Colour::fromRGB(18, 52, 38));
        background.addColour(0.8, juce::Colour::fromRGB(8, 18, 28));
        g.setGradientFill(background);
        g.fillAll();

        auto area = getLocalBounds().reduced(24);
        auto logoArea = area.removeFromTop(360);
        if (logo.isValid())
        {
            g.setImageResamplingQuality(juce::Graphics::highResamplingQuality);
            g.drawImageWithin(logo,
                              logoArea.getX(),
                              logoArea.getY(),
                              logoArea.getWidth(),
                              logoArea.getHeight(),
                              juce::RectanglePlacement::centred | juce::RectanglePlacement::onlyReduceInSize,
                              false);
        }

        auto titleArea = area.removeFromTop(56);
        g.setColour(juce::Colour::fromRGB(194, 250, 202));
        g.setFont(juce::FontOptions(aims::ui::scaleValue(34.0f), juce::Font::bold));
        g.drawFittedText("Mutagen", titleArea, juce::Justification::centred, 1);

        auto versionArea = area.removeFromTop(22);
        g.setColour(juce::Colour::fromRGB(150, 164, 182));
        g.setFont(juce::FontOptions(aims::ui::scaleValue(15.0f)));
        g.drawFittedText("Version " + version, versionArea, juce::Justification::centred, 1);

        area.removeFromTop(10);
        auto statusArea = area.removeFromTop(28);
        g.setColour(juce::Colour::fromRGB(118, 222, 170));
        g.setFont(juce::FontOptions(aims::ui::scaleValue(16.0f), juce::Font::bold));
        g.drawFittedText(status, statusArea, juce::Justification::centred, 1);
    }

private:
    juce::String version;
    juce::String status = "Starting Mutagen...";
    juce::Image logo;
};

class StartupSplashWindow final : public juce::DocumentWindow
{
public:
    explicit StartupSplashWindow(const juce::String& version)
        : juce::DocumentWindow("Mutagen",
                               juce::Colours::black,
                               0)
    {
        setUsingNativeTitleBar(false);
        setResizable(false, false);
        setAlwaysOnTop(true);
        setIcon(loadMutagenSplashLogo());
        setContentOwned(new StartupSplashComponent(version), true);
        centreWithSize(520, 520);
        setVisible(true);
        toFront(false);
        repaint();
        if (auto* peer = getPeer())
            peer->performAnyPendingRepaintsNow();
    }

    void closeButtonPressed() override
    {
    }

    void setStatus(const juce::String& status)
    {
        if (auto* splash = dynamic_cast<StartupSplashComponent*>(getContentComponent()))
            splash->setStatus(status);
    }
};

class StudioLookAndFeel final : public juce::LookAndFeel_V4
{
public:
    using juce::LookAndFeel_V4::LookAndFeel_V4;

    juce::Font getTextButtonFont(juce::TextButton&, int buttonHeight) override
    {
        return juce::FontOptions(juce::jlimit(aims::ui::scaleValue(aims::ui::kTinyTextSize),
                                              aims::ui::scaleValue(aims::ui::kStrongTextSize),
                                              static_cast<float>(buttonHeight) * 0.38f),
                                 juce::Font::bold);
    }

    juce::Font getComboBoxFont(juce::ComboBox&) override
    {
        return aims::ui::font();
    }

    juce::Font getPopupMenuFont() override
    {
        return aims::ui::font();
    }

    juce::Font getLabelFont(juce::Label&) override
    {
        return aims::ui::font();
    }

    juce::Font getMenuBarFont(juce::MenuBarComponent&, int, const juce::String&) override
    {
        return aims::ui::strongFont();
    }

    int getDefaultMenuBarHeight() override
    {
        return juce::roundToInt(juce::jmax(24.0f, aims::ui::scaleValue(24.0f)));
    }

    juce::Font getAlertWindowTitleFont() override
    {
        return aims::ui::titleFont();
    }

    juce::Font getAlertWindowMessageFont() override
    {
        return aims::ui::font();
    }

    juce::Font getAlertWindowFont() override
    {
        return aims::ui::font();
    }

    juce::Label* createSliderTextBox(juce::Slider& slider) override
    {
        auto* label = juce::LookAndFeel_V4::createSliderTextBox(slider);
        label->setFont(aims::ui::tinyFont());
        label->setJustificationType(juce::Justification::centred);
        label->setMinimumHorizontalScale(1.0f);
        return label;
    }

    juce::Label* createComboBoxTextBox(juce::ComboBox& box) override
    {
        auto* label = juce::LookAndFeel_V4::createComboBoxTextBox(box);
        label->setFont(aims::ui::font());
        label->setJustificationType(juce::Justification::centredLeft);
        label->setMinimumHorizontalScale(1.0f);
        return label;
    }
};
} // namespace

class AIMusicStudioNativeApplication final : public juce::JUCEApplication
{
public:
    const juce::String getApplicationName() override      { return "Mutagen"; }
    const juce::String getApplicationVersion() override   { return "0.3.3"; }
    bool moreThanOneInstanceAllowed() override            { return true; }

    void initialise(const juce::String&) override
    {
        lookAndFeel.setDefaultSansSerifTypefaceName(chooseMutagenUiFont());
        juce::LookAndFeel::setDefaultLookAndFeel(&lookAndFeel);

        const auto startupProject = parseStartupProject();
        startupSequenceToken = std::make_shared<int>(1);
        splashWindow = std::make_unique<StartupSplashWindow>(getApplicationVersion());
        pumpSplash("Loading native shell...");

        auto continueStartup = [this,
                                startupProject,
                                token = std::weak_ptr<int>(startupSequenceToken)] (const juce::String& status,
                                                                                   int delayMs,
                                                                                   auto&& next) mutable
        {
            juce::Timer::callAfterDelay(delayMs,
                                        [this,
                                         startupProject,
                                         status,
                                         token,
                                         next = std::forward<decltype(next)>(next)]() mutable
                                        {
                                            if (token.expired())
                                                return;

                                            pumpSplash(status);
                                            next();
                                        });
        };

        continueStartup(startupProject.existsAsFile() ? "Opening project..." : "Preparing interface...",
                        40,
                        [this,
                         startupProject,
                         token = std::weak_ptr<int>(startupSequenceToken)]() mutable
                        {
                            if (token.expired())
                                return;

                            mainWindow = std::make_unique<aims::MainWindow>(startupProject);
                            pumpSplash("Ready");

                            juce::Timer::callAfterDelay(120,
                                                        [this, token]() mutable
                                                        {
                                                            if (token.expired())
                                                                return;

                                                            splashWindow.reset();
                                                        });
                        });
    }

    void shutdown() override
    {
        startupSequenceToken.reset();
        splashWindow.reset();
        mainWindow.reset();
        juce::LookAndFeel::setDefaultLookAndFeel(nullptr);
    }

    void systemRequestedQuit() override
    {
        quit();
    }

private:
    juce::File parseStartupProject() const
    {
        const auto args = getCommandLineParameterArray();
        for (int index = 0; index < args.size(); ++index)
        {
            if (args[index] == "--project" && index + 1 < args.size())
                return juce::File(args[index + 1].unquoted());
        }
        return {};
    }

    void pumpSplash(const juce::String& status)
    {
        if (splashWindow != nullptr)
        {
            splashWindow->setStatus(status);
            splashWindow->setVisible(true);
            splashWindow->toFront(false);
            splashWindow->repaint();
            if (auto* peer = splashWindow->getPeer())
                peer->performAnyPendingRepaintsNow();
        }
    }

    StudioLookAndFeel lookAndFeel { juce::LookAndFeel_V4::getDarkColourScheme() };
    std::shared_ptr<int> startupSequenceToken;
    std::unique_ptr<StartupSplashWindow> splashWindow;
    std::unique_ptr<aims::MainWindow> mainWindow;
};

START_JUCE_APPLICATION(AIMusicStudioNativeApplication)
