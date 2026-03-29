#include "MainWindow.h"

#include <BinaryData.h>

namespace
{
juce::String chooseMutagenUiFont()
{
    const auto available = juce::Font::findAllTypefaceNames();
    constexpr std::array<const char*, 4> candidates
    {
        "Bahnschrift",
        "Bahnschrift SemiCondensed",
        "Agency FB",
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
        g.setFont(juce::FontOptions(34.0f, juce::Font::bold));
        g.drawFittedText("Mutagen", titleArea, juce::Justification::centred, 1);

        auto versionArea = area.removeFromTop(22);
        g.setColour(juce::Colour::fromRGB(150, 164, 182));
        g.setFont(juce::FontOptions(15.0f));
        g.drawFittedText("Version " + version, versionArea, juce::Justification::centred, 1);

        area.removeFromTop(10);
        auto statusArea = area.removeFromTop(28);
        g.setColour(juce::Colour::fromRGB(118, 222, 170));
        g.setFont(juce::FontOptions(16.0f, juce::Font::bold));
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
        setContentOwned(new StartupSplashComponent(version), true);
        centreWithSize(520, 520);
        setVisible(true);
        toFront(false);
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
} // namespace

class AIMusicStudioNativeApplication final : public juce::JUCEApplication
{
public:
    const juce::String getApplicationName() override      { return "Mutagen"; }
    const juce::String getApplicationVersion() override   { return "0.1.0"; }
    bool moreThanOneInstanceAllowed() override            { return true; }

    void initialise(const juce::String&) override
    {
        lookAndFeel.setDefaultSansSerifTypefaceName(chooseMutagenUiFont());
        juce::LookAndFeel::setDefaultLookAndFeel(&lookAndFeel);

        const auto startupProject = parseStartupProject();
        splashWindow = std::make_unique<StartupSplashWindow>(getApplicationVersion());
        pumpSplash("Loading native shell...", 40);
        pumpSplash(startupProject.existsAsFile() ? "Opening project..." : "Preparing interface...", 40);

        mainWindow = std::make_unique<aims::MainWindow>(startupProject);

        pumpSplash("Ready", 120);
        splashWindow.reset();
    }

    void shutdown() override
    {
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

    void pumpSplash(const juce::String& status, int milliseconds)
    {
        if (splashWindow != nullptr)
        {
            splashWindow->setStatus(status);
            splashWindow->repaint();
            if (auto* peer = splashWindow->getPeer())
                peer->performAnyPendingRepaintsNow();
        }

        if (milliseconds > 0)
            juce::Thread::sleep(milliseconds);
    }

    juce::LookAndFeel_V4 lookAndFeel { juce::LookAndFeel_V4::getDarkColourScheme() };
    std::unique_ptr<StartupSplashWindow> splashWindow;
    std::unique_ptr<aims::MainWindow> mainWindow;
};

START_JUCE_APPLICATION(AIMusicStudioNativeApplication)
