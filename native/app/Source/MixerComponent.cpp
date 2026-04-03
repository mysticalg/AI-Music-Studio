#include "MixerComponent.h"
#include "UiStyle.h"

#include <cmath>

namespace aims
{
namespace
{
const juce::Colour kMixerBackground = juce::Colour::fromRGB(15, 18, 24);
const juce::Colour kStripBaseBackground = juce::Colour::fromRGB(24, 29, 37);
const juce::Colour kStripBorder = juce::Colour::fromRGB(62, 71, 86);
const juce::Colour kMeterBackground = juce::Colour::fromRGB(12, 15, 20);
const juce::Colour kMeterLow = juce::Colour::fromRGB(92, 201, 124);
const juce::Colour kMeterMid = juce::Colour::fromRGB(255, 208, 92);
const juce::Colour kMeterHigh = juce::Colour::fromRGB(255, 96, 96);
const juce::Colour kMasterAccent = juce::Colour::fromRGB(111, 226, 203);
constexpr int kStripWidth = 126;
constexpr int kStripGap = 8;

const VstInstrument* findRackEntryByReference(const ProjectState& project,
                                              const juce::String& reference,
                                              bool wantEffect)
{
    const auto trimmed = reference.trim();
    if (trimmed.isEmpty())
        return nullptr;

    const auto normalisedPath = juce::File(trimmed).getFullPathName();
    for (const auto& entry : project.vstRack)
    {
        if (!entry.hostSupported)
            continue;
        if (wantEffect && !entry.isEffect)
            continue;
        if (!wantEffect && !entry.isInstrument)
            continue;

        if (entry.name.equalsIgnoreCase(trimmed)
            || entry.pluginName.equalsIgnoreCase(trimmed)
            || entry.path.equalsIgnoreCase(trimmed)
            || (!normalisedPath.isEmpty() && juce::File(entry.path).getFullPathName().equalsIgnoreCase(normalisedPath)))
        {
            return &entry;
        }
    }

    return nullptr;
}

juce::String referenceForEntry(const VstInstrument& entry)
{
    return entry.path.isNotEmpty() ? entry.path : entry.name;
}

juce::String displayNameForEntry(const VstInstrument& entry)
{
    if (entry.name.trim().isNotEmpty())
        return entry.name;
    if (entry.pluginName.trim().isNotEmpty())
        return entry.pluginName;
    if (entry.path.trim().isNotEmpty())
        return juce::File(entry.path).getFileNameWithoutExtension();
    return "Plugin";
}

juce::String describeFxChain(const ProjectState& project,
                             const juce::StringArray& fxChain,
                             bool bypassed)
{
    if (fxChain.isEmpty())
        return bypassed ? "FX bypassed" : "No FX";

    juce::StringArray labels;
    for (const auto& reference : fxChain)
    {
        if (const auto* entry = findRackEntryByReference(project, reference, true))
            labels.add(displayNameForEntry(*entry));
        else
            labels.add(reference.containsAnyOf("\\/")
                           ? juce::File(reference).getFileNameWithoutExtension()
                           : reference);
    }

    juce::String summary;
    if (labels.size() <= 2)
    {
        summary = labels.joinIntoString(" + ");
    }
    else
    {
        summary = labels[0] + " + " + labels[1] + " +" + juce::String(labels.size() - 2);
    }

    return bypassed ? "Bypassed: " + summary : summary;
}

void configureSmallTextButton(juce::TextButton& button)
{
    button.setColour(juce::TextButton::buttonColourId, juce::Colour::fromRGB(45, 53, 67));
    button.setColour(juce::TextButton::buttonOnColourId, juce::Colour::fromRGB(72, 94, 125));
    button.setColour(juce::TextButton::textColourOffId, juce::Colour::fromRGB(230, 235, 242));
    button.setColour(juce::TextButton::textColourOnId, juce::Colours::white);
}

void ensureFxBypassStateSize(std::vector<bool>& bypassStates, int effectCount)
{
    if (effectCount < 0)
        effectCount = 0;
    if (static_cast<int>(bypassStates.size()) < effectCount)
        bypassStates.resize(static_cast<size_t>(effectCount), false);
    else if (static_cast<int>(bypassStates.size()) > effectCount)
        bypassStates.resize(static_cast<size_t>(effectCount));
}

bool fxSlotBypassed(const std::vector<bool>& bypassStates, int index)
{
    return index >= 0
        && index < static_cast<int>(bypassStates.size())
        && bypassStates[static_cast<size_t>(index)];
}

juce::String gainDbLabel(double gain)
{
    if (gain <= 0.00001)
        return "-inf dB";

    const auto gainDb = 20.0 * std::log10(gain);
    return juce::String(gainDb, 1) + " dB";
}

juce::String peakDbfsLabel(float level)
{
    if (level <= 0.00001f)
        return "-inf dBFS";

    const auto peakDb = 20.0 * std::log10(static_cast<double>(level));
    return juce::String(peakDb, 1) + " dBFS";
}
}

MixerComponent::MixerComponent(ProjectGetter projectGetterIn,
                               TrackWriter trackWriterIn,
                               ProjectWriter projectWriterIn,
                               MeterGetter meterGetterIn,
                               MasterMeterGetter masterMeterGetterIn,
                               TrackEffectEditorOpener trackEffectEditorOpenerIn,
                               MasterEffectEditorOpener masterEffectEditorOpenerIn)
    : projectGetter(std::move(projectGetterIn)),
      trackWriter(std::move(trackWriterIn)),
      projectWriter(std::move(projectWriterIn)),
      meterGetter(std::move(meterGetterIn)),
      masterMeterGetter(std::move(masterMeterGetterIn)),
      trackEffectEditorOpener(std::move(trackEffectEditorOpenerIn)),
      masterEffectEditorOpener(std::move(masterEffectEditorOpenerIn))
{
    masterStrip = std::make_unique<MasterStrip>(projectGetter,
                                                projectWriter,
                                                masterMeterGetter,
                                                masterEffectEditorOpener);
    addAndMakeVisible(masterStrip.get());
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

    if (masterStrip != nullptr)
    {
        masterStrip->applyProject(project);
        masterStrip->refreshMeter();
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

    if (masterStrip != nullptr)
        masterStrip->refreshMeter();
}

void MixerComponent::paint(juce::Graphics& g)
{
    g.fillAll(kMixerBackground);
}

void MixerComponent::resized()
{
    auto area = getLocalBounds().reduced(8, 4);

    for (auto* strip : strips)
    {
        if (strip == nullptr)
            continue;

        strip->setBounds(area.removeFromLeft(kStripWidth));
        area.removeFromLeft(kStripGap);
    }

    if (masterStrip != nullptr)
        masterStrip->setBounds(area.removeFromLeft(kStripWidth));
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

void MixerComponent::MeterScaleComponent::paint(juce::Graphics& g)
{
    static constexpr std::array<float, 5> kDbMarks { 0.0f, -6.0f, -12.0f, -18.0f, -24.0f };

    auto bounds = getLocalBounds().toFloat();
    if (bounds.getHeight() <= 0.0f)
        return;

    g.setColour(juce::Colour::fromRGB(205, 212, 222).withAlpha(0.72f));
    g.setFont(ui::tinyFont());

    for (const auto db : kDbMarks)
    {
        const auto amplitude = static_cast<float>(std::pow(10.0, static_cast<double>(db) / 20.0));
        const auto y = bounds.getBottom() - (bounds.getHeight() * amplitude);
        g.drawLine(bounds.getRight() - 5.0f, y, bounds.getRight(), y, 1.0f);

        auto labelBounds = bounds.toNearestInt().withY(juce::roundToInt(y) - 5).withHeight(10);
        g.drawText(juce::String(db, 0),
                   labelBounds,
                   juce::Justification::centredRight,
                   false);
    }
}

MixerComponent::ChannelStrip::ChannelStrip(int trackIndexIn,
                                           ProjectGetter projectGetterIn,
                                           TrackWriter trackWriterIn,
                                           MeterGetter meterGetterIn,
                                           TrackEffectEditorOpener trackEffectEditorOpenerIn)
    : trackIndex(trackIndexIn),
      projectGetter(std::move(projectGetterIn)),
      trackWriter(std::move(trackWriterIn)),
      meterGetter(std::move(meterGetterIn)),
      trackEffectEditorOpener(std::move(trackEffectEditorOpenerIn))
{
    nameLabel.setJustificationType(juce::Justification::centred);
    nameLabel.setFont(ui::sectionFont());
    addAndMakeVisible(nameLabel);

    fxSummaryLabel.setJustificationType(juce::Justification::centred);
    fxSummaryLabel.setFont(ui::font());
    addAndMakeVisible(fxSummaryLabel);

    fxButton.setButtonText("FX");
    configureSmallTextButton(fxButton);
    fxButton.onClick = [this] { showFxMenu(); };
    addAndMakeVisible(fxButton);

    fxAddButton.setButtonText("+");
    fxAddButton.setTooltip("Add FX");
    configureSmallTextButton(fxAddButton);
    fxAddButton.onClick = [this] { showAddFxMenu(&fxAddButton); };
    addAndMakeVisible(fxAddButton);

    fxBypassButton.setButtonText("Byp");
    fxBypassButton.setClickingTogglesState(true);
    configureSmallTextButton(fxBypassButton);
    fxBypassButton.onClick = [this]
    {
        if (syncing)
            return;

        currentTrack.vstFxBypassed = fxBypassButton.getToggleState();
        commitTrackEdit("Toggle Track FX Bypass");
    };
    addAndMakeVisible(fxBypassButton);

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
    volumeSlider.onDragStart = [this] { volumeDragging = true; };
    volumeSlider.onDragEnd = [this]
    {
        volumeDragging = false;
        currentTrack.volume = volumeSlider.getValue() / 100.0;
        commitTrackEdit("Change Mixer Volume");
    };
    addAndMakeVisible(volumeSlider);

    volumeLabel.setJustificationType(juce::Justification::centred);
    volumeLabel.setColour(juce::Label::textColourId, juce::Colour::fromRGB(214, 220, 228));
    volumeLabel.setFont(ui::font());
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
    panSlider.onDragStart = [this] { panDragging = true; };
    panSlider.onDragEnd = [this]
    {
        panDragging = false;
        currentTrack.pan = panSlider.getValue() / 100.0;
        commitTrackEdit("Change Mixer Pan");
    };
    addAndMakeVisible(panSlider);

    muteButton.setButtonText("M");
    muteButton.setClickingTogglesState(true);
    configureSmallTextButton(muteButton);
    muteButton.setTooltip("Mute");
    muteButton.onClick = [this]
    {
        if (syncing)
            return;

        currentTrack.mute = muteButton.getToggleState();
        commitTrackEdit("Toggle Mixer Mute");
    };
    addAndMakeVisible(muteButton);

    soloButton.setButtonText("S");
    soloButton.setClickingTogglesState(true);
    configureSmallTextButton(soloButton);
    soloButton.setTooltip("Solo");
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
    const auto liveVolume = volumeSlider.getValue() / 100.0;
    const auto livePan = panSlider.getValue() / 100.0;
    currentTrack = track;

    const auto accent = trackDisplayColour(track, trackIndex);
    const auto textColour = trackTextColour(accent);
    nameLabel.setText(track.name, juce::dontSendNotification);
    nameLabel.setColour(juce::Label::textColourId, textColour);
    fxSummaryLabel.setText(track.vstFxChain.isEmpty() ? "No FX" : "FX chain", juce::dontSendNotification);
    fxSummaryLabel.setColour(juce::Label::textColourId, textColour.withAlpha(0.80f));
    syncFxRows();
    updateButtonColours(accent);

    if (volumeDragging)
    {
        currentTrack.volume = liveVolume;
        volumeLabel.setText(juce::String(juce::roundToInt(volumeSlider.getValue())) + "%", juce::dontSendNotification);
    }
    else
    {
        volumeSlider.setValue(track.volume * 100.0, juce::dontSendNotification);
        volumeLabel.setText(juce::String(juce::roundToInt(track.volume * 100.0)) + "%", juce::dontSendNotification);
    }

    if (panDragging)
    {
        currentTrack.pan = livePan;
    }
    else
    {
        panSlider.setValue(track.pan * 100.0, juce::dontSendNotification);
    }

    fxBypassButton.setToggleState(track.vstFxBypassed, juce::dontSendNotification);
    muteButton.setToggleState(track.mute, juce::dontSendNotification);
    soloButton.setToggleState(track.solo, juce::dontSendNotification);
    syncing = false;
}

void MixerComponent::ChannelStrip::refreshMeter()
{
    meter.setLevel(meterGetter != nullptr ? meterGetter(trackIndex) : 0.0f);
}

void MixerComponent::ChannelStrip::syncFxRows()
{
    const auto& project = projectGetter();
    const auto effectCount = currentTrack.vstFxChain.size();

    while (static_cast<int>(fxRows.size()) < effectCount)
    {
        auto row = std::make_unique<FxRowWidgets>();
        row->nameLabel = std::make_unique<juce::Label>();
        row->nameLabel->setJustificationType(juce::Justification::centredLeft);
        row->nameLabel->setFont(ui::strongFont());
        addAndMakeVisible(*row->nameLabel);

        row->viewButton = std::make_unique<juce::TextButton>("V");
        row->viewButton->setTooltip("Open FX Editor");
        configureSmallTextButton(*row->viewButton);
        row->viewButton->onClick = [this, raw = row.get()]
        {
            if (raw != nullptr)
                openEffectEditor(raw->effectIndex);
        };
        addAndMakeVisible(*row->viewButton);

        row->removeButton = std::make_unique<juce::TextButton>("-");
        row->removeButton->setTooltip("Remove FX");
        configureSmallTextButton(*row->removeButton);
        row->removeButton->onClick = [this, raw = row.get()]
        {
            if (raw != nullptr)
                removeEffectAt(raw->effectIndex);
        };
        addAndMakeVisible(*row->removeButton);

        fxRows.push_back(std::move(row));
    }

    while (static_cast<int>(fxRows.size()) > effectCount)
    {
        auto& row = fxRows.back();
        removeChildComponent(row->nameLabel.get());
        removeChildComponent(row->viewButton.get());
        removeChildComponent(row->removeButton.get());
        fxRows.pop_back();
    }

    for (int index = 0; index < effectCount; ++index)
    {
        auto& row = fxRows[static_cast<size_t>(index)];
        row->effectIndex = index;

        const auto reference = currentTrack.vstFxChain[index];
        juce::String label = reference;
        if (const auto* entry = findRackEntryByReference(project, reference, true))
            label = displayNameForEntry(*entry);
        else if (reference.containsAnyOf("\\/"))
            label = juce::File(reference).getFileNameWithoutExtension();

        row->nameLabel->setText(label, juce::dontSendNotification);
        row->nameLabel->setTooltip(reference);
        row->nameLabel->setVisible(true);
        row->viewButton->setVisible(true);
        row->removeButton->setVisible(true);
    }

    fxSummaryLabel.setVisible(effectCount == 0);
}

void MixerComponent::ChannelStrip::showAddFxMenu(juce::Component* target)
{
    const auto& project = projectGetter();
    juce::PopupMenu menu;
    int nextId = 100;
    std::vector<std::pair<int, juce::String>> effectAssignments;

    for (const auto& entry : project.vstRack)
    {
        if (!entry.hostSupported || !entry.isEffect)
            continue;

        const auto reference = referenceForEntry(entry).trim();
        if (reference.isEmpty())
            continue;

        menu.addItem(nextId, displayNameForEntry(entry), true, false);
        effectAssignments.emplace_back(nextId, reference);
        ++nextId;
    }

    if (effectAssignments.empty())
        menu.addItem(nextId, "No Rack FX Found", false);

    menu.showMenuAsync(juce::PopupMenu::Options().withTargetComponent(target != nullptr ? target : &fxAddButton),
                       [safeThis = juce::Component::SafePointer<ChannelStrip>(this),
                        effectAssignments = std::move(effectAssignments)](int result)
                       {
                           if (safeThis == nullptr || result == 0)
                               return;

                           for (const auto& [itemId, reference] : effectAssignments)
                           {
                               if (result != itemId)
                                   continue;

                               safeThis->currentTrack.vstFxChain.add(reference);
                               safeThis->commitTrackEdit("Add Track FX");
                               return;
                           }
                       });
}

void MixerComponent::ChannelStrip::openEffectEditor(int effectIndex)
{
    if (!juce::isPositiveAndBelow(effectIndex, currentTrack.vstFxChain.size()))
        return;

    if (trackEffectEditorOpener != nullptr)
        trackEffectEditorOpener(trackIndex, effectIndex);
}

void MixerComponent::ChannelStrip::removeEffectAt(int effectIndex)
{
    if (!juce::isPositiveAndBelow(effectIndex, currentTrack.vstFxChain.size()))
        return;

    currentTrack.vstFxChain.remove(effectIndex);
    commitTrackEdit("Remove Track FX");
}

void MixerComponent::ChannelStrip::paint(juce::Graphics& g)
{
    const auto accent = trackDisplayColour(currentTrack, trackIndex);
    auto fill = kStripBaseBackground.interpolatedWith(accent, 0.18f);
    fill = fill.withSaturation(juce::jmin(1.0f, fill.getSaturation() + 0.06f));

    g.setColour(fill);
    g.fillRoundedRectangle(getLocalBounds().toFloat(), 10.0f);

    g.setColour(accent.withAlpha(0.46f));
    g.drawRoundedRectangle(getLocalBounds().toFloat().reduced(0.5f), 10.0f, 1.2f);

    auto topGlow = getLocalBounds().toFloat();
    g.setColour(accent.withAlpha(0.12f));
    g.fillRoundedRectangle(topGlow.removeFromTop(30.0f), 10.0f);

    auto accentBar = getLocalBounds().toFloat().reduced(3.0f);
    g.setColour(accent.withAlpha(0.62f));
    g.fillRoundedRectangle(accentBar.removeFromLeft(3.0f), 3.0f);
}

void MixerComponent::ChannelStrip::resized()
{
    auto area = getLocalBounds().reduced(7);
    nameLabel.setBounds(area.removeFromTop(22));
    area.removeFromTop(4);

    auto bottomArea = area.removeFromBottom(258);

    auto buttonRow = bottomArea.removeFromBottom(22);
    const int controlWidth = juce::jmin(28, (buttonRow.getWidth() - 4) / 2);
    auto controls = buttonRow.withSizeKeepingCentre((controlWidth * 2) + 4, buttonRow.getHeight());
    muteButton.setBounds(controls.removeFromLeft(controlWidth));
    controls.removeFromLeft(4);
    soloButton.setBounds(controls.removeFromLeft(controlWidth));

    bottomArea.removeFromBottom(6);
    panSlider.setBounds(bottomArea.removeFromBottom(40));

    bottomArea.removeFromBottom(4);
    volumeLabel.setBounds(bottomArea.removeFromBottom(16));

    bottomArea.removeFromBottom(2);
    auto meterArea = bottomArea.removeFromBottom(154);
    auto levelLane = meterArea.withSizeKeepingCentre(34, meterArea.getHeight());
    meter.setBounds(levelLane.removeFromLeft(8));
    levelLane.removeFromLeft(4);
    volumeSlider.setBounds(levelLane);

    bottomArea.removeFromBottom(6);
    auto fxRow = bottomArea.removeFromBottom(20);
    const int fxButtonWidth = 24;
    auto fxButtons = fxRow.withSizeKeepingCentre((fxButtonWidth * 3) + 12, fxRow.getHeight());
    fxButton.setBounds(fxButtons.removeFromLeft(fxButtonWidth + 2));
    fxButtons.removeFromLeft(4);
    fxAddButton.setBounds(fxButtons.removeFromLeft(fxButtonWidth));
    fxButtons.removeFromLeft(4);
    fxBypassButton.setBounds(fxButtons.removeFromLeft(fxButtonWidth + 10));

    auto fxArea = area;
    fxSummaryLabel.setBounds(fxArea);

    const int rowHeight = 18;
    int y = fxArea.getY();
    for (auto& row : fxRows)
    {
        const bool fits = y + rowHeight <= fxArea.getBottom();
        row->nameLabel->setVisible(fits);
        row->viewButton->setVisible(fits);
        row->removeButton->setVisible(fits);
        if (!fits)
            continue;

        auto rowBounds = juce::Rectangle<int>(fxArea.getX(), y, fxArea.getWidth(), rowHeight);
        row->nameLabel->setBounds(rowBounds.removeFromLeft(juce::jmax(8, rowBounds.getWidth() - 40)));
        rowBounds.removeFromLeft(2);
        row->viewButton->setBounds(rowBounds.removeFromLeft(18));
        rowBounds.removeFromLeft(2);
        row->removeButton->setBounds(rowBounds.removeFromLeft(18));
        y += rowHeight + 2;
    }
}

void MixerComponent::ChannelStrip::showFxMenu()
{
    const auto& project = projectGetter();

    juce::PopupMenu menu;
    constexpr int menuClearFx = 1;
    int nextId = 100;
    std::vector<std::pair<int, juce::String>> effectAssignments;

    menu.addItem(menuClearFx, "Clear FX Chain", currentTrack.vstFxChain.size() > 0);
    menu.addSeparator();

    for (const auto& entry : project.vstRack)
    {
        if (!entry.hostSupported || !entry.isEffect)
            continue;

        const auto reference = referenceForEntry(entry).trim();
        if (reference.isEmpty())
            continue;

        const bool assigned = currentTrack.vstFxChain.contains(reference);
        menu.addItem(nextId,
                     displayNameForEntry(entry),
                     true,
                     assigned);
        effectAssignments.emplace_back(nextId, reference);
        ++nextId;
    }

    if (effectAssignments.empty())
        menu.addItem(nextId, "No Rack FX Found", false);

    menu.showMenuAsync(juce::PopupMenu::Options().withTargetComponent(&fxButton),
                       [safeThis = juce::Component::SafePointer<ChannelStrip>(this),
                        effectAssignments = std::move(effectAssignments)](int result)
                       {
                           if (safeThis == nullptr || result == 0)
                               return;

                           if (result == menuClearFx)
                           {
                               if (safeThis->currentTrack.vstFxChain.size() <= 0)
                                   return;

                               safeThis->currentTrack.vstFxChain.clear();
                               safeThis->commitTrackEdit("Clear Track FX Chain");
                               return;
                           }

                           for (const auto& [itemId, reference] : effectAssignments)
                           {
                               if (result != itemId)
                                   continue;

                               if (safeThis->currentTrack.vstFxChain.contains(reference))
                                   safeThis->currentTrack.vstFxChain.removeString(reference);
                               else
                                   safeThis->currentTrack.vstFxChain.add(reference);

                               safeThis->commitTrackEdit("Update Track FX Chain");
                               return;
                           }
                       });
}

void MixerComponent::ChannelStrip::updateButtonColours(const juce::Colour& accentColour)
{
    const auto offColour = kStripBaseBackground.interpolatedWith(accentColour, 0.24f);
    const auto onColour = kStripBaseBackground.interpolatedWith(accentColour, 0.44f);

    fxButton.setColour(juce::TextButton::buttonColourId, offColour);
    fxButton.setColour(juce::TextButton::buttonOnColourId, onColour);
    fxButton.setColour(juce::TextButton::textColourOffId, trackTextColour(offColour));
    fxButton.setColour(juce::TextButton::textColourOnId, trackTextColour(onColour));
    fxAddButton.setColour(juce::TextButton::buttonColourId, offColour);
    fxAddButton.setColour(juce::TextButton::buttonOnColourId, onColour);
    fxAddButton.setColour(juce::TextButton::textColourOffId, trackTextColour(offColour));
    fxAddButton.setColour(juce::TextButton::textColourOnId, trackTextColour(onColour));
    fxBypassButton.setColour(juce::TextButton::buttonColourId, offColour);
    fxBypassButton.setColour(juce::TextButton::buttonOnColourId, accentColour.withMultipliedBrightness(0.72f));
    fxBypassButton.setColour(juce::TextButton::textColourOffId, trackTextColour(offColour));
    fxBypassButton.setColour(juce::TextButton::textColourOnId, juce::Colours::white);
    muteButton.setColour(juce::TextButton::buttonColourId, offColour);
    muteButton.setColour(juce::TextButton::buttonOnColourId, juce::Colour::fromRGB(204, 74, 74));
    soloButton.setColour(juce::TextButton::buttonColourId, offColour);
    soloButton.setColour(juce::TextButton::buttonOnColourId, juce::Colour::fromRGB(225, 181, 76));
    muteButton.setColour(juce::TextButton::textColourOffId, trackTextColour(offColour));
    muteButton.setColour(juce::TextButton::textColourOnId, juce::Colours::white);
    soloButton.setColour(juce::TextButton::textColourOffId, trackTextColour(offColour));
    soloButton.setColour(juce::TextButton::textColourOnId, juce::Colour::fromRGB(28, 24, 16));

    for (auto& row : fxRows)
    {
        row->nameLabel->setColour(juce::Label::textColourId, trackTextColour(accentColour).withAlpha(0.90f));
        row->viewButton->setColour(juce::TextButton::buttonColourId, offColour);
        row->viewButton->setColour(juce::TextButton::buttonOnColourId, onColour);
        row->viewButton->setColour(juce::TextButton::textColourOffId, trackTextColour(offColour));
        row->viewButton->setColour(juce::TextButton::textColourOnId, trackTextColour(onColour));
        row->removeButton->setColour(juce::TextButton::buttonColourId, offColour);
        row->removeButton->setColour(juce::TextButton::buttonOnColourId, juce::Colour::fromRGB(204, 74, 74));
        row->removeButton->setColour(juce::TextButton::textColourOffId, trackTextColour(offColour));
        row->removeButton->setColour(juce::TextButton::textColourOnId, juce::Colours::white);
    }
}

void MixerComponent::ChannelStrip::commitTrackEdit(const juce::String& actionName)
{
    if (trackWriter != nullptr)
        trackWriter(trackIndex, currentTrack, true, actionName);
}

MixerComponent::MasterStrip::MasterStrip(ProjectGetter projectGetterIn,
                                         ProjectWriter projectWriterIn,
                                         MasterMeterGetter masterMeterGetterIn,
                                         MasterEffectEditorOpener masterEffectEditorOpenerIn)
    : projectGetter(std::move(projectGetterIn)),
      projectWriter(std::move(projectWriterIn)),
      masterMeterGetter(std::move(masterMeterGetterIn)),
      masterEffectEditorOpener(std::move(masterEffectEditorOpenerIn))
{
    nameLabel.setText("Master Out", juce::dontSendNotification);
    nameLabel.setJustificationType(juce::Justification::centred);
    nameLabel.setColour(juce::Label::textColourId, juce::Colours::white);
    nameLabel.setFont(ui::sectionFont());
    addAndMakeVisible(nameLabel);

    fxSummaryLabel.setJustificationType(juce::Justification::centred);
    fxSummaryLabel.setColour(juce::Label::textColourId, juce::Colour::fromRGB(227, 234, 240));
    fxSummaryLabel.setFont(ui::font());
    addAndMakeVisible(fxSummaryLabel);

    fxButton.setButtonText("FX");
    configureSmallTextButton(fxButton);
    fxButton.onClick = [this] { showFxMenu(); };
    addAndMakeVisible(fxButton);

    fxAddButton.setButtonText("+");
    fxAddButton.setTooltip("Add FX");
    configureSmallTextButton(fxAddButton);
    fxAddButton.onClick = [this] { showAddFxMenu(&fxAddButton); };
    addAndMakeVisible(fxAddButton);

    fxBypassButton.setButtonText("Byp");
    fxBypassButton.setClickingTogglesState(true);
    configureSmallTextButton(fxBypassButton);
    fxBypassButton.onClick = [this]
    {
        if (syncing)
            return;

        currentProject.masterFxBypassed = fxBypassButton.getToggleState();
        commitProjectEdit("Toggle Master FX Bypass");
    };
    addAndMakeVisible(fxBypassButton);

    addAndMakeVisible(leftMeter);
    addAndMakeVisible(rightMeter);

    volumeSlider.setSliderStyle(juce::Slider::LinearVertical);
    volumeSlider.setRange(0.0, 100.0, 1.0);
    volumeSlider.setTextBoxStyle(juce::Slider::NoTextBox, false, 0, 0);
    volumeSlider.onValueChange = [this]
    {
        if (syncing)
            return;

        currentProject.masterVolume = volumeSlider.getValue() / 100.0;
        volumeLabel.setText(juce::String(juce::roundToInt(volumeSlider.getValue())) + "%", juce::dontSendNotification);
    };
    volumeSlider.onDragStart = [this] { volumeDragging = true; };
    volumeSlider.onDragEnd = [this]
    {
        volumeDragging = false;
        currentProject.masterVolume = volumeSlider.getValue() / 100.0;
        commitProjectEdit("Change Master Volume");
    };
    addAndMakeVisible(volumeSlider);

    volumeLabel.setJustificationType(juce::Justification::centred);
    volumeLabel.setColour(juce::Label::textColourId, juce::Colour::fromRGB(227, 234, 240));
    volumeLabel.setFont(ui::font());
    addAndMakeVisible(volumeLabel);
}

void MixerComponent::MasterStrip::applyProject(const ProjectState& project)
{
    syncing = true;
    const auto liveVolume = volumeSlider.getValue() / 100.0;
    currentProject = project;
    fxSummaryLabel.setText(project.masterFxChain.isEmpty() ? "No FX" : "FX chain", juce::dontSendNotification);
    syncFxRows();

    if (volumeDragging)
    {
        currentProject.masterVolume = liveVolume;
        volumeLabel.setText(juce::String(juce::roundToInt(volumeSlider.getValue())) + "%", juce::dontSendNotification);
    }
    else
    {
        volumeSlider.setValue(project.masterVolume * 100.0, juce::dontSendNotification);
        volumeLabel.setText(juce::String(juce::roundToInt(project.masterVolume * 100.0)) + "%", juce::dontSendNotification);
    }

    const auto offColour = kStripBaseBackground.interpolatedWith(kMasterAccent, 0.22f);
    const auto onColour = kStripBaseBackground.interpolatedWith(kMasterAccent, 0.42f);
    fxButton.setColour(juce::TextButton::buttonColourId, offColour);
    fxButton.setColour(juce::TextButton::buttonOnColourId, onColour);
    fxButton.setColour(juce::TextButton::textColourOffId, trackTextColour(offColour));
    fxButton.setColour(juce::TextButton::textColourOnId, trackTextColour(onColour));
    fxAddButton.setColour(juce::TextButton::buttonColourId, offColour);
    fxAddButton.setColour(juce::TextButton::buttonOnColourId, onColour);
    fxAddButton.setColour(juce::TextButton::textColourOffId, trackTextColour(offColour));
    fxAddButton.setColour(juce::TextButton::textColourOnId, trackTextColour(onColour));
    fxBypassButton.setColour(juce::TextButton::buttonColourId, offColour);
    fxBypassButton.setColour(juce::TextButton::buttonOnColourId, kMasterAccent.withMultipliedBrightness(0.8f));
    fxBypassButton.setColour(juce::TextButton::textColourOffId, trackTextColour(offColour));
    fxBypassButton.setColour(juce::TextButton::textColourOnId, juce::Colours::white);
    fxBypassButton.setToggleState(project.masterFxBypassed, juce::dontSendNotification);

    for (auto& row : fxRows)
    {
        row->nameLabel->setColour(juce::Label::textColourId, juce::Colour::fromRGB(227, 234, 240));
        row->viewButton->setColour(juce::TextButton::buttonColourId, offColour);
        row->viewButton->setColour(juce::TextButton::buttonOnColourId, onColour);
        row->viewButton->setColour(juce::TextButton::textColourOffId, trackTextColour(offColour));
        row->viewButton->setColour(juce::TextButton::textColourOnId, trackTextColour(onColour));
        row->removeButton->setColour(juce::TextButton::buttonColourId, offColour);
        row->removeButton->setColour(juce::TextButton::buttonOnColourId, juce::Colour::fromRGB(204, 74, 74));
        row->removeButton->setColour(juce::TextButton::textColourOffId, trackTextColour(offColour));
        row->removeButton->setColour(juce::TextButton::textColourOnId, juce::Colours::white);
    }
    syncing = false;
}

void MixerComponent::MasterStrip::refreshMeter()
{
    const auto levels = masterMeterGetter != nullptr ? masterMeterGetter() : std::make_pair(0.0f, 0.0f);
    leftMeter.setLevel(levels.first);
    rightMeter.setLevel(levels.second);
}

void MixerComponent::MasterStrip::syncFxRows()
{
    const auto& project = projectGetter();
    const auto effectCount = currentProject.masterFxChain.size();

    while (static_cast<int>(fxRows.size()) < effectCount)
    {
        auto row = std::make_unique<FxRowWidgets>();
        row->nameLabel = std::make_unique<juce::Label>();
        row->nameLabel->setJustificationType(juce::Justification::centredLeft);
        row->nameLabel->setFont(ui::strongFont());
        addAndMakeVisible(*row->nameLabel);

        row->viewButton = std::make_unique<juce::TextButton>("V");
        row->viewButton->setTooltip("Open FX Editor");
        configureSmallTextButton(*row->viewButton);
        row->viewButton->onClick = [this, raw = row.get()]
        {
            if (raw != nullptr)
                openEffectEditor(raw->effectIndex);
        };
        addAndMakeVisible(*row->viewButton);

        row->removeButton = std::make_unique<juce::TextButton>("-");
        row->removeButton->setTooltip("Remove FX");
        configureSmallTextButton(*row->removeButton);
        row->removeButton->onClick = [this, raw = row.get()]
        {
            if (raw != nullptr)
                removeEffectAt(raw->effectIndex);
        };
        addAndMakeVisible(*row->removeButton);

        fxRows.push_back(std::move(row));
    }

    while (static_cast<int>(fxRows.size()) > effectCount)
    {
        auto& row = fxRows.back();
        removeChildComponent(row->nameLabel.get());
        removeChildComponent(row->viewButton.get());
        removeChildComponent(row->removeButton.get());
        fxRows.pop_back();
    }

    for (int index = 0; index < effectCount; ++index)
    {
        auto& row = fxRows[static_cast<size_t>(index)];
        row->effectIndex = index;

        const auto reference = currentProject.masterFxChain[index];
        juce::String label = reference;
        if (const auto* entry = findRackEntryByReference(project, reference, true))
            label = displayNameForEntry(*entry);
        else if (reference.containsAnyOf("\\/"))
            label = juce::File(reference).getFileNameWithoutExtension();

        row->nameLabel->setText(label, juce::dontSendNotification);
        row->nameLabel->setTooltip(reference);
        row->nameLabel->setVisible(true);
        row->viewButton->setVisible(true);
        row->removeButton->setVisible(true);
    }

    fxSummaryLabel.setVisible(effectCount == 0);
}

void MixerComponent::MasterStrip::showAddFxMenu(juce::Component* target)
{
    const auto& project = projectGetter();
    juce::PopupMenu menu;
    int nextId = 100;
    std::vector<std::pair<int, juce::String>> effectAssignments;

    for (const auto& entry : project.vstRack)
    {
        if (!entry.hostSupported || !entry.isEffect)
            continue;

        const auto reference = referenceForEntry(entry).trim();
        if (reference.isEmpty())
            continue;

        menu.addItem(nextId, displayNameForEntry(entry), true, false);
        effectAssignments.emplace_back(nextId, reference);
        ++nextId;
    }

    if (effectAssignments.empty())
        menu.addItem(nextId, "No Rack FX Found", false);

    menu.showMenuAsync(juce::PopupMenu::Options().withTargetComponent(target != nullptr ? target : &fxAddButton),
                       [safeThis = juce::Component::SafePointer<MasterStrip>(this),
                        effectAssignments = std::move(effectAssignments)](int result)
                       {
                           if (safeThis == nullptr || result == 0)
                               return;

                           for (const auto& [itemId, reference] : effectAssignments)
                           {
                               if (result != itemId)
                                   continue;

                               safeThis->currentProject.masterFxChain.add(reference);
                               safeThis->commitProjectEdit("Add Master FX");
                               return;
                           }
                       });
}

void MixerComponent::MasterStrip::openEffectEditor(int effectIndex)
{
    if (!juce::isPositiveAndBelow(effectIndex, currentProject.masterFxChain.size()))
        return;

    if (masterEffectEditorOpener != nullptr)
        masterEffectEditorOpener(effectIndex);
}

void MixerComponent::MasterStrip::removeEffectAt(int effectIndex)
{
    if (!juce::isPositiveAndBelow(effectIndex, currentProject.masterFxChain.size()))
        return;

    currentProject.masterFxChain.remove(effectIndex);
    commitProjectEdit("Remove Master FX");
}

void MixerComponent::MasterStrip::paint(juce::Graphics& g)
{
    const auto fill = kStripBaseBackground.interpolatedWith(kMasterAccent, 0.12f);
    g.setColour(fill);
    g.fillRoundedRectangle(getLocalBounds().toFloat(), 10.0f);

    g.setColour(kMasterAccent.withAlpha(0.46f));
    g.drawRoundedRectangle(getLocalBounds().toFloat().reduced(0.5f), 10.0f, 1.2f);

    auto topGlow = getLocalBounds().toFloat();
    g.setColour(kMasterAccent.withAlpha(0.10f));
    g.fillRoundedRectangle(topGlow.removeFromTop(30.0f), 10.0f);
}

void MixerComponent::MasterStrip::resized()
{
    auto area = getLocalBounds().reduced(7);
    nameLabel.setBounds(area.removeFromTop(22));
    area.removeFromTop(4);

    auto bottomArea = area.removeFromBottom(204);
    bottomArea.removeFromBottom(4);
    volumeLabel.setBounds(bottomArea.removeFromBottom(16));
    bottomArea.removeFromBottom(2);
    auto meterArea = bottomArea.removeFromBottom(154);
    auto levelLane = meterArea.withSizeKeepingCentre(42, meterArea.getHeight());
    leftMeter.setBounds(levelLane.removeFromLeft(7));
    levelLane.removeFromLeft(2);
    rightMeter.setBounds(levelLane.removeFromLeft(7));
    levelLane.removeFromLeft(4);
    volumeSlider.setBounds(levelLane);

    bottomArea.removeFromBottom(6);
    auto fxRow = bottomArea.removeFromBottom(20);
    const int fxButtonWidth = 24;
    auto fxButtons = fxRow.withSizeKeepingCentre((fxButtonWidth * 3) + 12, fxRow.getHeight());
    fxButton.setBounds(fxButtons.removeFromLeft(fxButtonWidth + 2));
    fxButtons.removeFromLeft(4);
    fxAddButton.setBounds(fxButtons.removeFromLeft(fxButtonWidth));
    fxButtons.removeFromLeft(4);
    fxBypassButton.setBounds(fxButtons.removeFromLeft(fxButtonWidth + 10));

    auto fxArea = area;
    fxSummaryLabel.setBounds(fxArea);

    const int rowHeight = 18;
    int y = fxArea.getY();
    for (auto& row : fxRows)
    {
        const bool fits = y + rowHeight <= fxArea.getBottom();
        row->nameLabel->setVisible(fits);
        row->viewButton->setVisible(fits);
        row->removeButton->setVisible(fits);
        if (!fits)
            continue;

        auto rowBounds = juce::Rectangle<int>(fxArea.getX(), y, fxArea.getWidth(), rowHeight);
        row->nameLabel->setBounds(rowBounds.removeFromLeft(juce::jmax(8, rowBounds.getWidth() - 40)));
        rowBounds.removeFromLeft(2);
        row->viewButton->setBounds(rowBounds.removeFromLeft(18));
        rowBounds.removeFromLeft(2);
        row->removeButton->setBounds(rowBounds.removeFromLeft(18));
        y += rowHeight + 2;
    }
}

void MixerComponent::MasterStrip::showFxMenu()
{
    const auto& project = projectGetter();

    juce::PopupMenu menu;
    constexpr int menuClearFx = 1;
    int nextId = 100;
    std::vector<std::pair<int, juce::String>> effectAssignments;

    menu.addItem(menuClearFx, "Clear Master FX", currentProject.masterFxChain.size() > 0);
    menu.addSeparator();

    for (const auto& entry : project.vstRack)
    {
        if (!entry.hostSupported || !entry.isEffect)
            continue;

        const auto reference = referenceForEntry(entry).trim();
        if (reference.isEmpty())
            continue;

        const bool assigned = currentProject.masterFxChain.contains(reference);
        menu.addItem(nextId,
                     displayNameForEntry(entry),
                     true,
                     assigned);
        effectAssignments.emplace_back(nextId, reference);
        ++nextId;
    }

    if (effectAssignments.empty())
        menu.addItem(nextId, "No Rack FX Found", false);

    menu.showMenuAsync(juce::PopupMenu::Options().withTargetComponent(&fxButton),
                       [safeThis = juce::Component::SafePointer<MasterStrip>(this),
                        effectAssignments = std::move(effectAssignments)](int result)
                       {
                           if (safeThis == nullptr || result == 0)
                               return;

                           if (result == menuClearFx)
                           {
                               if (safeThis->currentProject.masterFxChain.size() <= 0)
                                   return;

                               safeThis->currentProject.masterFxChain.clear();
                               safeThis->commitProjectEdit("Clear Master FX");
                               return;
                           }

                           for (const auto& [itemId, reference] : effectAssignments)
                           {
                               if (result != itemId)
                                   continue;

                               if (safeThis->currentProject.masterFxChain.contains(reference))
                                   safeThis->currentProject.masterFxChain.removeString(reference);
                               else
                                   safeThis->currentProject.masterFxChain.add(reference);

                               safeThis->commitProjectEdit("Update Master FX Chain");
                               return;
                           }
                       });
}

void MixerComponent::MasterStrip::commitProjectEdit(const juce::String& actionName)
{
    if (projectWriter != nullptr)
        projectWriter(currentProject, true, actionName);
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
        auto* strip = strips.add(new ChannelStrip(strips.size(),
                                                  projectGetter,
                                                  trackWriter,
                                                  meterGetter,
                                                  trackEffectEditorOpener));
        addAndMakeVisible(strip);
    }

    const int stripWidth = 118;
    const int gap = 8;
    const auto width = juce::jmax(760, ((trackCount + 1) * stripWidth) + (juce::jmax(0, trackCount) * gap) + 18);
    setSize(width, juce::jmax(360, getHeight()));
}

} // namespace aims
