#include "MixerComponent.h"

#include <cmath>

namespace aims
{
namespace
{
const juce::Colour kMixerBackground = juce::Colour::fromRGB(15, 18, 24);
const juce::Colour kStripBackground = juce::Colour::fromRGB(24, 29, 37);
const juce::Colour kStripBorder = juce::Colour::fromRGB(54, 62, 76);
const juce::Colour kMeterBackground = juce::Colour::fromRGB(12, 15, 20);
const juce::Colour kMeterLow = juce::Colour::fromRGB(92, 201, 124);
const juce::Colour kMeterMid = juce::Colour::fromRGB(255, 208, 92);
const juce::Colour kMeterHigh = juce::Colour::fromRGB(255, 96, 96);
}

MixerComponent::MixerComponent(ProjectGetter projectGetterIn,
                               TrackWriter trackWriterIn,
                               MeterGetter meterGetterIn)
    : projectGetter(std::move(projectGetterIn)),
      trackWriter(std::move(trackWriterIn)),
      meterGetter(std::move(meterGetterIn))
{
}

void MixerComponent::refreshFromModel()
{
    const auto& project = projectGetter();
    ensureStripCount(static_cast<int>(project.tracks.size()));

    for (int index = 0; index < strips.size(); ++index)
    {
        if (!juce::isPositiveAndBelow(index, static_cast<int>(project.tracks.size())))
            continue;
        strips[index]->applyTrack(project.tracks[static_cast<size_t>(index)]);
        strips[index]->refreshMeter();
    }

    resized();
    repaint();
}

void MixerComponent::refreshMeters()
{
    for (auto* strip : strips)
    {
        if (strip != nullptr)
            strip->refreshMeter();
    }
}

void MixerComponent::paint(juce::Graphics& g)
{
    g.fillAll(kMixerBackground);
}

void MixerComponent::resized()
{
    auto area = getLocalBounds().reduced(8, 4);
    const int stripWidth = 104;
    const int gap = 8;

    for (auto* strip : strips)
    {
        if (strip == nullptr)
            continue;
        strip->setBounds(area.removeFromLeft(stripWidth));
        area.removeFromLeft(gap);
    }
}

void MixerComponent::MeterComponent::setLevel(float newLevel)
{
    const auto clamped = juce::jlimit(0.0f, 1.0f, newLevel);
    if (std::abs(clamped - level) < 0.001f)
        return;
    level = clamped;
    repaint();
}

void MixerComponent::MeterComponent::paint(juce::Graphics& g)
{
    g.setColour(kMeterBackground);
    g.fillRoundedRectangle(getLocalBounds().toFloat(), 4.0f);

    const auto bounds = getLocalBounds().toFloat().reduced(3.0f);
    const auto filledHeight = bounds.getHeight() * level;
    auto fill = bounds.withTrimmedTop(bounds.getHeight() - filledHeight);

    juce::ColourGradient gradient(kMeterLow, fill.getCentreX(), fill.getBottom(),
                                  kMeterHigh, fill.getCentreX(), fill.getY(), false);
    gradient.addColour(0.55, kMeterMid);
    g.setGradientFill(gradient);
    g.fillRoundedRectangle(fill, 3.0f);
}

MixerComponent::ChannelStrip::ChannelStrip(int trackIndexIn,
                                           TrackWriter trackWriterIn,
                                           MeterGetter meterGetterIn)
    : trackIndex(trackIndexIn),
      trackWriter(std::move(trackWriterIn)),
      meterGetter(std::move(meterGetterIn))
{
    nameLabel.setJustificationType(juce::Justification::centred);
    nameLabel.setColour(juce::Label::textColourId, juce::Colours::white);
    nameLabel.setFont(juce::FontOptions(13.0f, juce::Font::bold));
    addAndMakeVisible(nameLabel);

    addAndMakeVisible(meter);

    volumeSlider.setSliderStyle(juce::Slider::LinearVertical);
    volumeSlider.setRange(0.0, 100.0, 1.0);
    volumeSlider.setTextBoxStyle(juce::Slider::NoTextBox, false, 0, 0);
    volumeSlider.onValueChange = [this]
    {
        if (syncing)
            return;
        currentTrack.volume = volumeSlider.getValue() / 100.0;
        volumeLabel.setText(juce::String(juce::roundToInt(volumeSlider.getValue())) + "%", juce::dontSendNotification);
    };
    volumeSlider.onDragEnd = [this] { commitTrackEdit("Change Mixer Volume"); };
    addAndMakeVisible(volumeSlider);

    volumeLabel.setJustificationType(juce::Justification::centred);
    volumeLabel.setColour(juce::Label::textColourId, juce::Colour::fromRGB(214, 220, 228));
    volumeLabel.setFont(juce::FontOptions(11.0f));
    addAndMakeVisible(volumeLabel);

    panSlider.setSliderStyle(juce::Slider::RotaryHorizontalVerticalDrag);
    panSlider.setTextBoxStyle(juce::Slider::NoTextBox, false, 0, 0);
    panSlider.setRange(-100.0, 100.0, 1.0);
    panSlider.onValueChange = [this]
    {
        if (syncing)
            return;
        currentTrack.pan = panSlider.getValue() / 100.0;
    };
    panSlider.onDragEnd = [this] { commitTrackEdit("Change Mixer Pan"); };
    addAndMakeVisible(panSlider);

    muteButton.setButtonText("M");
    muteButton.onClick = [this]
    {
        if (syncing)
            return;
        currentTrack.mute = muteButton.getToggleState();
        commitTrackEdit("Toggle Mixer Mute");
    };
    addAndMakeVisible(muteButton);

    soloButton.setButtonText("S");
    soloButton.onClick = [this]
    {
        if (syncing)
            return;
        currentTrack.solo = soloButton.getToggleState();
        commitTrackEdit("Toggle Mixer Solo");
    };
    addAndMakeVisible(soloButton);
}

void MixerComponent::ChannelStrip::applyTrack(const TrackState& track)
{
    syncing = true;
    currentTrack = track;
    nameLabel.setText(track.name, juce::dontSendNotification);
    volumeSlider.setValue(track.volume * 100.0, juce::dontSendNotification);
    volumeLabel.setText(juce::String(juce::roundToInt(track.volume * 100.0)) + "%", juce::dontSendNotification);
    panSlider.setValue(track.pan * 100.0, juce::dontSendNotification);
    muteButton.setToggleState(track.mute, juce::dontSendNotification);
    soloButton.setToggleState(track.solo, juce::dontSendNotification);
    syncing = false;
}

void MixerComponent::ChannelStrip::refreshMeter()
{
    meter.setLevel(meterGetter != nullptr ? meterGetter(trackIndex) : 0.0f);
}

void MixerComponent::ChannelStrip::paint(juce::Graphics& g)
{
    g.setColour(kStripBackground);
    g.fillRoundedRectangle(getLocalBounds().toFloat(), 8.0f);
    g.setColour(kStripBorder);
    g.drawRoundedRectangle(getLocalBounds().toFloat().reduced(0.5f), 8.0f, 1.0f);
}

void MixerComponent::ChannelStrip::resized()
{
    auto area = getLocalBounds().reduced(8);
    nameLabel.setBounds(area.removeFromTop(24));
    area.removeFromTop(6);

    auto meterArea = area.removeFromTop(160);
    meter.setBounds(meterArea.removeFromLeft(20));
    meterArea.removeFromLeft(8);
    volumeSlider.setBounds(meterArea);
    area.removeFromTop(4);
    volumeLabel.setBounds(area.removeFromTop(18));
    area.removeFromTop(6);

    panSlider.setBounds(area.removeFromTop(48));
    area.removeFromTop(6);

    auto buttonRow = area.removeFromTop(24);
    muteButton.setBounds(buttonRow.removeFromLeft(34));
    buttonRow.removeFromLeft(6);
    soloButton.setBounds(buttonRow.removeFromLeft(34));
}

void MixerComponent::ChannelStrip::commitTrackEdit(const juce::String& actionName)
{
    if (trackWriter != nullptr)
        trackWriter(trackIndex, currentTrack, true, actionName);
}

void MixerComponent::ensureStripCount(int trackCount)
{
    while (strips.size() > trackCount)
    {
        removeChildComponent(strips.getLast());
        strips.removeLast();
    }

    while (strips.size() < trackCount)
    {
        auto* strip = strips.add(new ChannelStrip(strips.size(), trackWriter, meterGetter));
        addAndMakeVisible(strip);
    }

    const auto width = juce::jmax(720, (trackCount * 112) + 16);
    setSize(width, juce::jmax(260, getHeight()));
}

} // namespace aims
