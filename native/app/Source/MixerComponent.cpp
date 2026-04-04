#include "MixerComponent.h"
#include "UiStyle.h"

#include <array>
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

juce::String meterTooltipText(double gain, float level)
{
    return "Fader: " + gainDbLabel(gain) + " | Peak: " + peakDbfsLabel(level);
}

juce::String stereoMeterTooltipText(double gain, float leftLevel, float rightLevel)
{
    return "Fader: " + gainDbLabel(gain)
        + " | L: " + peakDbfsLabel(leftLevel)
        + " | R: " + peakDbfsLabel(rightLevel);
}

void removeFxSlotAt(juce::StringArray& chain, std::vector<bool>& bypassStates, int index)
{
    if (!juce::isPositiveAndBelow(index, chain.size()))
        return;

    chain.remove(index);
    if (juce::isPositiveAndBelow(index, static_cast<int>(bypassStates.size())))
        bypassStates.erase(bypassStates.begin() + index);

    ensureFxBypassStateSize(bypassStates, chain.size());
}

void appendFxSlot(juce::StringArray& chain, std::vector<bool>& bypassStates, const juce::String& reference)
{
    chain.add(reference);
    ensureFxBypassStateSize(bypassStates, chain.size());
    if (juce::isPositiveAndBelow(chain.size() - 1, static_cast<int>(bypassStates.size())))
        bypassStates[static_cast<size_t>(chain.size() - 1)] = false;
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

    fxUpButton.setButtonText("^");
    fxUpButton.setTooltip("Show higher FX slots");
    configureSmallTextButton(fxUpButton);
    fxUpButton.onClick = [this]
    {
        --fxScrollOffset;
        clampFxScrollOffset();
        resized();
        repaint();
    };
    addAndMakeVisible(fxUpButton);

    fxDownButton.setButtonText("v");
    fxDownButton.setTooltip("Show lower FX slots");
    configureSmallTextButton(fxDownButton);
    fxDownButton.onClick = [this]
    {
        ++fxScrollOffset;
        clampFxScrollOffset();
        resized();
        repaint();
    };
    addAndMakeVisible(fxDownButton);

    addAndMakeVisible(meterScale);
    addAndMakeVisible(meter);

    volumeSlider.setSliderStyle(juce::Slider::LinearVertical);
    volumeSlider.setRange(0.0, 100.0, 1.0);
    volumeSlider.setTextBoxStyle(juce::Slider::NoTextBox, false, 0, 0);
    volumeSlider.onValueChange = [this]
    {
        if (syncing)
            return;

        currentTrack.volume = volumeSlider.getValue() / 100.0;
        volumeLabel.setText(gainDbLabel(currentTrack.volume), juce::dontSendNotification);
        updateMeterTooltip();
    };
    volumeSlider.onDragStart = [this] { volumeDragging = true; };
    volumeSlider.onDragEnd = [this]
    {
        volumeDragging = false;
        currentTrack.volume = volumeSlider.getValue() / 100.0;
        volumeLabel.setText(gainDbLabel(currentTrack.volume), juce::dontSendNotification);
        updateMeterTooltip();
        commitTrackEdit("Change Mixer Volume");
    };
    addAndMakeVisible(volumeSlider);

    volumeLabel.setJustificationType(juce::Justification::centred);
    volumeLabel.setColour(juce::Label::textColourId, juce::Colour::fromRGB(214, 220, 228));
    volumeLabel.setFont(ui::font());
    addAndMakeVisible(volumeLabel);

    meterLabel.setJustificationType(juce::Justification::centred);
    meterLabel.setColour(juce::Label::textColourId, juce::Colour::fromRGB(214, 220, 228).withAlpha(0.78f));
    meterLabel.setFont(ui::tinyFont(true));
    addAndMakeVisible(meterLabel);

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
    ensureFxBypassStateSize(currentTrack.vstFxSlotBypassed, currentTrack.vstFxChain.size());

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
        volumeLabel.setText(gainDbLabel(currentTrack.volume), juce::dontSendNotification);
    }
    else
    {
        volumeSlider.setValue(track.volume * 100.0, juce::dontSendNotification);
        volumeLabel.setText(gainDbLabel(track.volume), juce::dontSendNotification);
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
    clampFxScrollOffset();
    updateMeterTooltip();
    syncing = false;
}

void MixerComponent::ChannelStrip::refreshMeter()
{
    currentMeterLevel = meterGetter != nullptr ? meterGetter(trackIndex) : 0.0f;
    meter.setLevel(currentMeterLevel);
    meterLabel.setText(peakDbfsLabel(currentMeterLevel), juce::dontSendNotification);
    updateMeterTooltip();
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
        row->nameLabel->setMouseCursor(juce::MouseCursor::DraggingHandCursor);
        row->nameLabel->addMouseListener(this, false);
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

        row->bypassButton = std::make_unique<juce::TextButton>("B");
        row->bypassButton->setTooltip("Bypass FX");
        row->bypassButton->setClickingTogglesState(true);
        configureSmallTextButton(*row->bypassButton);
        row->bypassButton->onClick = [this, raw = row.get()]
        {
            if (syncing || raw == nullptr || !juce::isPositiveAndBelow(raw->effectIndex, currentTrack.vstFxChain.size()))
                return;

            ensureFxBypassStateSize(currentTrack.vstFxSlotBypassed, currentTrack.vstFxChain.size());
            if (juce::isPositiveAndBelow(raw->effectIndex, static_cast<int>(currentTrack.vstFxSlotBypassed.size())))
                currentTrack.vstFxSlotBypassed[static_cast<size_t>(raw->effectIndex)] = raw->bypassButton->getToggleState();

            commitTrackEdit("Toggle Track FX Slot Bypass");
        };
        addAndMakeVisible(*row->bypassButton);

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
        removeChildComponent(row->bypassButton.get());
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
        row->nameLabel->setTooltip(label + "  |  Drag to reorder");
        row->nameLabel->setVisible(true);
        row->viewButton->setVisible(true);
        row->bypassButton->setToggleState(fxSlotBypassed(currentTrack.vstFxSlotBypassed, index), juce::dontSendNotification);
        row->bypassButton->setVisible(true);
        row->removeButton->setVisible(true);
    }

    fxSummaryLabel.setVisible(effectCount == 0);
    clampFxScrollOffset();
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

                               appendFxSlot(safeThis->currentTrack.vstFxChain,
                                            safeThis->currentTrack.vstFxSlotBypassed,
                                            reference);
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

    removeFxSlotAt(currentTrack.vstFxChain, currentTrack.vstFxSlotBypassed, effectIndex);
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

    if (fxDragActive && fxDropInsertIndex >= 0)
    {
        const auto lineY = fxInsertionLineY(fxDropInsertIndex);
        if (lineY > 0)
        {
            const auto left = static_cast<float>(getLocalBounds().getX() + 12);
            const auto right = static_cast<float>(getLocalBounds().getRight() - 12);
            g.setColour(accent.withAlpha(0.95f));
            g.fillRoundedRectangle(juce::Rectangle<float>(left, static_cast<float>(lineY - 1), right - left, 3.0f), 1.5f);
        }
    }
}

void MixerComponent::ChannelStrip::resized()
{
    auto area = getLocalBounds().reduced(7);
    nameLabel.setBounds(area.removeFromTop(18));
    area.removeFromTop(4);

    const int minimumFxAreaHeight = 44;
    const int preferredBottomHeight = 214;
    const auto maxBottomHeight = juce::jmax(0, area.getHeight() - minimumFxAreaHeight);
    auto bottomArea = area.removeFromBottom(juce::jmin(preferredBottomHeight, maxBottomHeight));

    fxListBounds = area;
    fxSummaryLabel.setBounds(fxListBounds);

    auto buttonRow = bottomArea.removeFromBottom(24);
    const int controlWidth = juce::jmin(30, (buttonRow.getWidth() - 4) / 2);
    auto controls = buttonRow.withSizeKeepingCentre((controlWidth * 2) + 4, buttonRow.getHeight());
    muteButton.setBounds(controls.removeFromLeft(controlWidth));
    controls.removeFromLeft(4);
    soloButton.setBounds(controls.removeFromLeft(controlWidth));

    bottomArea.removeFromBottom(4);
    panSlider.setBounds(bottomArea.removeFromBottom(34));

    bottomArea.removeFromBottom(2);
    volumeLabel.setBounds(bottomArea.removeFromBottom(14));

    bottomArea.removeFromBottom(2);
    meterLabel.setBounds(bottomArea.removeFromBottom(12));

    bottomArea.removeFromBottom(2);
    auto meterArea = bottomArea.removeFromBottom(118);
    auto levelLane = meterArea.withSizeKeepingCentre(54, meterArea.getHeight());
    meterScale.setBounds(levelLane.removeFromLeft(18));
    meter.setBounds(levelLane.removeFromLeft(11));
    levelLane.removeFromLeft(4);
    volumeSlider.setBounds(levelLane);

    bottomArea.removeFromBottom(4);
    auto fxRow = bottomArea.removeFromBottom(20);
    auto fxButtons = fxRow.withSizeKeepingCentre(juce::jmin(fxRow.getWidth(), 116), fxRow.getHeight());
    fxButton.setBounds(fxButtons.removeFromLeft(22));
    fxButtons.removeFromLeft(4);
    fxAddButton.setBounds(fxButtons.removeFromLeft(18));
    fxButtons.removeFromLeft(4);
    fxUpButton.setBounds(fxButtons.removeFromLeft(18));
    fxButtons.removeFromLeft(4);
    fxDownButton.setBounds(fxButtons.removeFromLeft(18));
    fxButtons.removeFromLeft(4);
    fxBypassButton.setBounds(fxButtons.removeFromLeft(24));

    const int rowHeight = 18;
    const int rowGap = 2;
    const int rowPitch = rowHeight + rowGap;
    fxVisibleRowCount = juce::jmax(0, (fxListBounds.getHeight() + rowGap) / rowPitch);
    clampFxScrollOffset();
    const auto effectCount = static_cast<int>(fxRows.size());
    const auto startIndex = juce::jlimit(0, juce::jmax(0, effectCount), fxScrollOffset);
    const auto endIndex = juce::jmin(effectCount, startIndex + fxVisibleRowCount);

    fxUpButton.setEnabled(startIndex > 0);
    fxDownButton.setEnabled(endIndex < effectCount);

    int y = fxListBounds.getY();
    for (int index = 0; index < effectCount; ++index)
    {
        auto& row = fxRows[static_cast<size_t>(index)];
        const bool visible = index >= startIndex && index < endIndex && y + rowHeight <= fxListBounds.getBottom();
        row->rowBounds = {};
        row->nameLabel->setVisible(visible);
        row->viewButton->setVisible(visible);
        row->bypassButton->setVisible(visible);
        row->removeButton->setVisible(visible);
        if (!visible)
            continue;

        auto rowBounds = juce::Rectangle<int>(fxListBounds.getX(), y, fxListBounds.getWidth(), rowHeight);
        row->rowBounds = rowBounds;
        auto rowButtons = rowBounds.removeFromRight(58);
        row->nameLabel->setBounds(rowBounds);
        row->viewButton->setBounds(rowButtons.removeFromLeft(18));
        rowButtons.removeFromLeft(2);
        row->bypassButton->setBounds(rowButtons.removeFromLeft(18));
        rowButtons.removeFromLeft(2);
        row->removeButton->setBounds(rowButtons.removeFromLeft(18));
        y += rowPitch;
    }
}

void MixerComponent::ChannelStrip::mouseWheelMove(const juce::MouseEvent& event, const juce::MouseWheelDetails& wheel)
{
    if (!fxListBounds.contains(event.getEventRelativeTo(this).getPosition()) || fxRows.size() <= static_cast<size_t>(fxVisibleRowCount))
    {
        juce::Component::mouseWheelMove(event, wheel);
        return;
    }

    if (std::abs(wheel.deltaY) < 0.001f)
        return;

    fxScrollOffset += wheel.deltaY < 0.0f ? 1 : -1;
    clampFxScrollOffset();
    resized();
    repaint();
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
                               safeThis->currentTrack.vstFxSlotBypassed.clear();
                               safeThis->commitTrackEdit("Clear Track FX Chain");
                               return;
                           }

                           for (const auto& [itemId, reference] : effectAssignments)
                           {
                               if (result != itemId)
                                   continue;

                               int existingIndex = -1;
                               for (int index = 0; index < safeThis->currentTrack.vstFxChain.size(); ++index)
                               {
                                   if (safeThis->currentTrack.vstFxChain[index].equalsIgnoreCase(reference))
                                   {
                                       existingIndex = index;
                                       break;
                                   }
                               }

                               if (existingIndex >= 0)
                                   removeFxSlotAt(safeThis->currentTrack.vstFxChain, safeThis->currentTrack.vstFxSlotBypassed, existingIndex);
                               else
                                   appendFxSlot(safeThis->currentTrack.vstFxChain, safeThis->currentTrack.vstFxSlotBypassed, reference);

                               safeThis->commitTrackEdit("Update Track FX Chain");
                               return;
                           }
                       });
}

void MixerComponent::ChannelStrip::updateButtonColours(const juce::Colour& accentColour)
{
    const auto offColour = kStripBaseBackground.interpolatedWith(accentColour, 0.24f);
    const auto onColour = kStripBaseBackground.interpolatedWith(accentColour, 0.44f);

    volumeLabel.setColour(juce::Label::textColourId, trackTextColour(accentColour).withAlpha(0.90f));
    meterLabel.setColour(juce::Label::textColourId, trackTextColour(accentColour).withAlpha(0.72f));
    fxButton.setColour(juce::TextButton::buttonColourId, offColour);
    fxButton.setColour(juce::TextButton::buttonOnColourId, onColour);
    fxButton.setColour(juce::TextButton::textColourOffId, trackTextColour(offColour));
    fxButton.setColour(juce::TextButton::textColourOnId, trackTextColour(onColour));
    fxAddButton.setColour(juce::TextButton::buttonColourId, offColour);
    fxAddButton.setColour(juce::TextButton::buttonOnColourId, onColour);
    fxAddButton.setColour(juce::TextButton::textColourOffId, trackTextColour(offColour));
    fxAddButton.setColour(juce::TextButton::textColourOnId, trackTextColour(onColour));
    fxUpButton.setColour(juce::TextButton::buttonColourId, offColour);
    fxUpButton.setColour(juce::TextButton::buttonOnColourId, onColour);
    fxUpButton.setColour(juce::TextButton::textColourOffId, trackTextColour(offColour));
    fxUpButton.setColour(juce::TextButton::textColourOnId, trackTextColour(onColour));
    fxDownButton.setColour(juce::TextButton::buttonColourId, offColour);
    fxDownButton.setColour(juce::TextButton::buttonOnColourId, onColour);
    fxDownButton.setColour(juce::TextButton::textColourOffId, trackTextColour(offColour));
    fxDownButton.setColour(juce::TextButton::textColourOnId, trackTextColour(onColour));
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
        row->bypassButton->setColour(juce::TextButton::buttonColourId, offColour);
        row->bypassButton->setColour(juce::TextButton::buttonOnColourId, accentColour.withMultipliedBrightness(0.72f));
        row->bypassButton->setColour(juce::TextButton::textColourOffId, trackTextColour(offColour));
        row->bypassButton->setColour(juce::TextButton::textColourOnId, juce::Colours::white);
        row->removeButton->setColour(juce::TextButton::buttonColourId, offColour);
        row->removeButton->setColour(juce::TextButton::buttonOnColourId, juce::Colour::fromRGB(204, 74, 74));
        row->removeButton->setColour(juce::TextButton::textColourOffId, trackTextColour(offColour));
        row->removeButton->setColour(juce::TextButton::textColourOnId, juce::Colours::white);
    }
}

void MixerComponent::ChannelStrip::updateMeterTooltip()
{
    const auto tooltip = meterTooltipText(currentTrack.volume, currentMeterLevel);
    meter.setTooltip(tooltip);
    meterScale.setTooltip("Meter scale in dBFS");
    meterLabel.setTooltip("Peak level: " + peakDbfsLabel(currentMeterLevel));
    volumeSlider.setTooltip("Fader: " + gainDbLabel(currentTrack.volume));
}

void MixerComponent::ChannelStrip::clampFxScrollOffset()
{
    const auto maxOffset = juce::jmax(0, static_cast<int>(fxRows.size()) - juce::jmax(1, fxVisibleRowCount));
    fxScrollOffset = juce::jlimit(0, maxOffset, fxScrollOffset);
}

int MixerComponent::ChannelStrip::findFxRowIndexForComponent(const juce::Component* component) const
{
    if (component == nullptr)
        return -1;

    for (const auto& row : fxRows)
    {
        if (row != nullptr && row->nameLabel.get() == component)
            return row->effectIndex;
    }

    return -1;
}

int MixerComponent::ChannelStrip::findFxInsertIndexForY(int y) const
{
    int insertIndex = 0;
    bool foundVisibleRow = false;

    for (const auto& row : fxRows)
    {
        if (row == nullptr || row->rowBounds.isEmpty())
            continue;

        foundVisibleRow = true;
        if (y < row->rowBounds.getCentreY())
            return row->effectIndex;

        insertIndex = row->effectIndex + 1;
    }

    return foundVisibleRow ? insertIndex : currentTrack.vstFxChain.size();
}

int MixerComponent::ChannelStrip::fxInsertionLineY(int insertIndex) const
{
    const auto clampedIndex = juce::jlimit(0, currentTrack.vstFxChain.size(), insertIndex);

    for (const auto& row : fxRows)
    {
        if (row == nullptr || row->rowBounds.isEmpty())
            continue;

        if (row->effectIndex == clampedIndex)
            return row->rowBounds.getY();
    }

    for (auto rowIt = fxRows.rbegin(); rowIt != fxRows.rend(); ++rowIt)
    {
        const auto& row = *rowIt;
        if (row == nullptr || row->rowBounds.isEmpty())
            continue;

        return row->rowBounds.getBottom();
    }

    return 0;
}

void MixerComponent::ChannelStrip::finishFxDrag(bool commitChanges)
{
    const auto sourceIndex = fxDragSourceIndex;
    const auto insertIndex = fxDropInsertIndex;
    const auto wasActive = fxDragActive;

    fxDragActive = false;
    fxDragSourceIndex = -1;
    fxDropInsertIndex = -1;
    repaint();

    if (!commitChanges || !wasActive)
        return;

    moveEffectToInsertIndex(sourceIndex, insertIndex);
}

void MixerComponent::ChannelStrip::moveEffectToInsertIndex(int sourceIndex, int insertIndex)
{
    const auto effectCount = currentTrack.vstFxChain.size();
    if (!juce::isPositiveAndBelow(sourceIndex, effectCount))
        return;

    auto adjustedInsertIndex = juce::jlimit(0, effectCount, insertIndex);
    if (adjustedInsertIndex == sourceIndex || adjustedInsertIndex == sourceIndex + 1)
        return;

    const auto movedReference = currentTrack.vstFxChain[sourceIndex];
    const auto movedBypassed = fxSlotBypassed(currentTrack.vstFxSlotBypassed, sourceIndex);

    currentTrack.vstFxChain.remove(sourceIndex);
    if (juce::isPositiveAndBelow(sourceIndex, static_cast<int>(currentTrack.vstFxSlotBypassed.size())))
        currentTrack.vstFxSlotBypassed.erase(currentTrack.vstFxSlotBypassed.begin() + sourceIndex);

    if (adjustedInsertIndex > sourceIndex)
        --adjustedInsertIndex;

    currentTrack.vstFxChain.insert(adjustedInsertIndex, movedReference);
    currentTrack.vstFxSlotBypassed.insert(currentTrack.vstFxSlotBypassed.begin() + adjustedInsertIndex, movedBypassed);
    ensureFxBypassStateSize(currentTrack.vstFxSlotBypassed, currentTrack.vstFxChain.size());
    commitTrackEdit("Reorder Track FX");
}

void MixerComponent::ChannelStrip::mouseDown(const juce::MouseEvent& event)
{
    const auto* component = event.originalComponent != nullptr ? event.originalComponent : event.eventComponent;
    const auto rowIndex = findFxRowIndexForComponent(component);
    if (rowIndex < 0)
        return;

    fxDragActive = true;
    fxDragSourceIndex = rowIndex;
    fxDropInsertIndex = rowIndex;
    repaint();
}

void MixerComponent::ChannelStrip::mouseDrag(const juce::MouseEvent& event)
{
    if (!fxDragActive)
        return;

    fxDropInsertIndex = findFxInsertIndexForY(juce::roundToInt(event.getEventRelativeTo(this).position.y));
    repaint();
}

void MixerComponent::ChannelStrip::mouseUp(const juce::MouseEvent&)
{
    finishFxDrag(true);
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

    fxUpButton.setButtonText("^");
    fxUpButton.setTooltip("Show higher FX slots");
    configureSmallTextButton(fxUpButton);
    fxUpButton.onClick = [this]
    {
        --fxScrollOffset;
        clampFxScrollOffset();
        resized();
        repaint();
    };
    addAndMakeVisible(fxUpButton);

    fxDownButton.setButtonText("v");
    fxDownButton.setTooltip("Show lower FX slots");
    configureSmallTextButton(fxDownButton);
    fxDownButton.onClick = [this]
    {
        ++fxScrollOffset;
        clampFxScrollOffset();
        resized();
        repaint();
    };
    addAndMakeVisible(fxDownButton);

    addAndMakeVisible(meterScale);
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
        volumeLabel.setText(gainDbLabel(currentProject.masterVolume), juce::dontSendNotification);
        updateMeterTooltip();
    };
    volumeSlider.onDragStart = [this] { volumeDragging = true; };
    volumeSlider.onDragEnd = [this]
    {
        volumeDragging = false;
        currentProject.masterVolume = volumeSlider.getValue() / 100.0;
        volumeLabel.setText(gainDbLabel(currentProject.masterVolume), juce::dontSendNotification);
        updateMeterTooltip();
        commitProjectEdit("Change Master Volume");
    };
    addAndMakeVisible(volumeSlider);

    volumeLabel.setJustificationType(juce::Justification::centred);
    volumeLabel.setColour(juce::Label::textColourId, juce::Colour::fromRGB(227, 234, 240));
    volumeLabel.setFont(ui::font());
    addAndMakeVisible(volumeLabel);

    meterLabel.setJustificationType(juce::Justification::centred);
    meterLabel.setColour(juce::Label::textColourId, juce::Colour::fromRGB(227, 234, 240).withAlpha(0.76f));
    meterLabel.setFont(ui::tinyFont(true));
    addAndMakeVisible(meterLabel);
}

void MixerComponent::MasterStrip::applyProject(const ProjectState& project)
{
    syncing = true;
    const auto liveVolume = volumeSlider.getValue() / 100.0;
    currentProject = project;
    ensureFxBypassStateSize(currentProject.masterFxSlotBypassed, currentProject.masterFxChain.size());
    fxSummaryLabel.setText(project.masterFxChain.isEmpty() ? "No FX" : "FX chain", juce::dontSendNotification);
    syncFxRows();

    if (volumeDragging)
    {
        currentProject.masterVolume = liveVolume;
        volumeLabel.setText(gainDbLabel(currentProject.masterVolume), juce::dontSendNotification);
    }
    else
    {
        volumeSlider.setValue(project.masterVolume * 100.0, juce::dontSendNotification);
        volumeLabel.setText(gainDbLabel(project.masterVolume), juce::dontSendNotification);
    }

    const auto offColour = kStripBaseBackground.interpolatedWith(kMasterAccent, 0.22f);
    const auto onColour = kStripBaseBackground.interpolatedWith(kMasterAccent, 0.42f);
    volumeLabel.setColour(juce::Label::textColourId, juce::Colour::fromRGB(227, 234, 240));
    meterLabel.setColour(juce::Label::textColourId, juce::Colour::fromRGB(227, 234, 240).withAlpha(0.76f));
    fxButton.setColour(juce::TextButton::buttonColourId, offColour);
    fxButton.setColour(juce::TextButton::buttonOnColourId, onColour);
    fxButton.setColour(juce::TextButton::textColourOffId, trackTextColour(offColour));
    fxButton.setColour(juce::TextButton::textColourOnId, trackTextColour(onColour));
    fxAddButton.setColour(juce::TextButton::buttonColourId, offColour);
    fxAddButton.setColour(juce::TextButton::buttonOnColourId, onColour);
    fxAddButton.setColour(juce::TextButton::textColourOffId, trackTextColour(offColour));
    fxAddButton.setColour(juce::TextButton::textColourOnId, trackTextColour(onColour));
    fxUpButton.setColour(juce::TextButton::buttonColourId, offColour);
    fxUpButton.setColour(juce::TextButton::buttonOnColourId, onColour);
    fxUpButton.setColour(juce::TextButton::textColourOffId, trackTextColour(offColour));
    fxUpButton.setColour(juce::TextButton::textColourOnId, trackTextColour(onColour));
    fxDownButton.setColour(juce::TextButton::buttonColourId, offColour);
    fxDownButton.setColour(juce::TextButton::buttonOnColourId, onColour);
    fxDownButton.setColour(juce::TextButton::textColourOffId, trackTextColour(offColour));
    fxDownButton.setColour(juce::TextButton::textColourOnId, trackTextColour(onColour));
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
        row->bypassButton->setColour(juce::TextButton::buttonColourId, offColour);
        row->bypassButton->setColour(juce::TextButton::buttonOnColourId, kMasterAccent.withMultipliedBrightness(0.8f));
        row->bypassButton->setColour(juce::TextButton::textColourOffId, trackTextColour(offColour));
        row->bypassButton->setColour(juce::TextButton::textColourOnId, juce::Colours::white);
        row->removeButton->setColour(juce::TextButton::buttonColourId, offColour);
        row->removeButton->setColour(juce::TextButton::buttonOnColourId, juce::Colour::fromRGB(204, 74, 74));
        row->removeButton->setColour(juce::TextButton::textColourOffId, trackTextColour(offColour));
        row->removeButton->setColour(juce::TextButton::textColourOnId, juce::Colours::white);
    }
    clampFxScrollOffset();
    updateMeterTooltip();
    syncing = false;
}

void MixerComponent::MasterStrip::refreshMeter()
{
    currentMeterLevels = masterMeterGetter != nullptr ? masterMeterGetter() : std::make_pair(0.0f, 0.0f);
    leftMeter.setLevel(currentMeterLevels.first);
    rightMeter.setLevel(currentMeterLevels.second);
    meterLabel.setText(peakDbfsLabel(juce::jmax(currentMeterLevels.first, currentMeterLevels.second)), juce::dontSendNotification);
    updateMeterTooltip();
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
        row->nameLabel->setMouseCursor(juce::MouseCursor::DraggingHandCursor);
        row->nameLabel->addMouseListener(this, false);
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

        row->bypassButton = std::make_unique<juce::TextButton>("B");
        row->bypassButton->setTooltip("Bypass FX");
        row->bypassButton->setClickingTogglesState(true);
        configureSmallTextButton(*row->bypassButton);
        row->bypassButton->onClick = [this, raw = row.get()]
        {
            if (syncing || raw == nullptr || !juce::isPositiveAndBelow(raw->effectIndex, currentProject.masterFxChain.size()))
                return;

            ensureFxBypassStateSize(currentProject.masterFxSlotBypassed, currentProject.masterFxChain.size());
            if (juce::isPositiveAndBelow(raw->effectIndex, static_cast<int>(currentProject.masterFxSlotBypassed.size())))
                currentProject.masterFxSlotBypassed[static_cast<size_t>(raw->effectIndex)] = raw->bypassButton->getToggleState();

            commitProjectEdit("Toggle Master FX Slot Bypass");
        };
        addAndMakeVisible(*row->bypassButton);

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
        removeChildComponent(row->bypassButton.get());
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
        row->nameLabel->setTooltip(label + "  |  Drag to reorder");
        row->nameLabel->setVisible(true);
        row->viewButton->setVisible(true);
        row->bypassButton->setToggleState(fxSlotBypassed(currentProject.masterFxSlotBypassed, index), juce::dontSendNotification);
        row->bypassButton->setVisible(true);
        row->removeButton->setVisible(true);
    }

    fxSummaryLabel.setVisible(effectCount == 0);
    clampFxScrollOffset();
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

                               appendFxSlot(safeThis->currentProject.masterFxChain,
                                            safeThis->currentProject.masterFxSlotBypassed,
                                            reference);
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

    removeFxSlotAt(currentProject.masterFxChain, currentProject.masterFxSlotBypassed, effectIndex);
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

    if (fxDragActive && fxDropInsertIndex >= 0)
    {
        const auto lineY = fxInsertionLineY(fxDropInsertIndex);
        if (lineY > 0)
        {
            const auto left = static_cast<float>(getLocalBounds().getX() + 12);
            const auto right = static_cast<float>(getLocalBounds().getRight() - 12);
            g.setColour(kMasterAccent.withAlpha(0.95f));
            g.fillRoundedRectangle(juce::Rectangle<float>(left, static_cast<float>(lineY - 1), right - left, 3.0f), 1.5f);
        }
    }
}

void MixerComponent::MasterStrip::resized()
{
    auto area = getLocalBounds().reduced(7);
    nameLabel.setBounds(area.removeFromTop(18));
    area.removeFromTop(4);

    const int minimumFxAreaHeight = 44;
    const int preferredBottomHeight = 182;
    const auto maxBottomHeight = juce::jmax(0, area.getHeight() - minimumFxAreaHeight);
    auto bottomArea = area.removeFromBottom(juce::jmin(preferredBottomHeight, maxBottomHeight));

    fxListBounds = area;
    fxSummaryLabel.setBounds(fxListBounds);

    bottomArea.removeFromBottom(2);
    volumeLabel.setBounds(bottomArea.removeFromBottom(14));
    bottomArea.removeFromBottom(2);
    meterLabel.setBounds(bottomArea.removeFromBottom(12));
    bottomArea.removeFromBottom(2);
    auto meterArea = bottomArea.removeFromBottom(118);
    auto levelLane = meterArea.withSizeKeepingCentre(64, meterArea.getHeight());
    meterScale.setBounds(levelLane.removeFromLeft(18));
    leftMeter.setBounds(levelLane.removeFromLeft(7));
    levelLane.removeFromLeft(2);
    rightMeter.setBounds(levelLane.removeFromLeft(7));
    levelLane.removeFromLeft(4);
    volumeSlider.setBounds(levelLane);

    bottomArea.removeFromBottom(4);
    auto fxRow = bottomArea.removeFromBottom(20);
    auto fxButtons = fxRow.withSizeKeepingCentre(juce::jmin(fxRow.getWidth(), 116), fxRow.getHeight());
    fxButton.setBounds(fxButtons.removeFromLeft(22));
    fxButtons.removeFromLeft(4);
    fxAddButton.setBounds(fxButtons.removeFromLeft(18));
    fxButtons.removeFromLeft(4);
    fxUpButton.setBounds(fxButtons.removeFromLeft(18));
    fxButtons.removeFromLeft(4);
    fxDownButton.setBounds(fxButtons.removeFromLeft(18));
    fxButtons.removeFromLeft(4);
    fxBypassButton.setBounds(fxButtons.removeFromLeft(24));

    const int rowHeight = 18;
    const int rowGap = 2;
    const int rowPitch = rowHeight + rowGap;
    fxVisibleRowCount = juce::jmax(0, (fxListBounds.getHeight() + rowGap) / rowPitch);
    clampFxScrollOffset();
    const auto effectCount = static_cast<int>(fxRows.size());
    const auto startIndex = juce::jlimit(0, juce::jmax(0, effectCount), fxScrollOffset);
    const auto endIndex = juce::jmin(effectCount, startIndex + fxVisibleRowCount);

    fxUpButton.setEnabled(startIndex > 0);
    fxDownButton.setEnabled(endIndex < effectCount);

    int y = fxListBounds.getY();
    for (int index = 0; index < effectCount; ++index)
    {
        auto& row = fxRows[static_cast<size_t>(index)];
        const bool visible = index >= startIndex && index < endIndex && y + rowHeight <= fxListBounds.getBottom();
        row->rowBounds = {};
        row->nameLabel->setVisible(visible);
        row->viewButton->setVisible(visible);
        row->bypassButton->setVisible(visible);
        row->removeButton->setVisible(visible);
        if (!visible)
            continue;

        auto rowBounds = juce::Rectangle<int>(fxListBounds.getX(), y, fxListBounds.getWidth(), rowHeight);
        row->rowBounds = rowBounds;
        auto rowButtons = rowBounds.removeFromRight(58);
        row->nameLabel->setBounds(rowBounds);
        row->viewButton->setBounds(rowButtons.removeFromLeft(18));
        rowButtons.removeFromLeft(2);
        row->bypassButton->setBounds(rowButtons.removeFromLeft(18));
        rowButtons.removeFromLeft(2);
        row->removeButton->setBounds(rowButtons.removeFromLeft(18));
        y += rowPitch;
    }
}

void MixerComponent::MasterStrip::mouseWheelMove(const juce::MouseEvent& event, const juce::MouseWheelDetails& wheel)
{
    if (!fxListBounds.contains(event.getEventRelativeTo(this).getPosition()) || fxRows.size() <= static_cast<size_t>(fxVisibleRowCount))
    {
        juce::Component::mouseWheelMove(event, wheel);
        return;
    }

    if (std::abs(wheel.deltaY) < 0.001f)
        return;

    fxScrollOffset += wheel.deltaY < 0.0f ? 1 : -1;
    clampFxScrollOffset();
    resized();
    repaint();
}

void MixerComponent::MasterStrip::updateMeterTooltip()
{
    const auto tooltip = stereoMeterTooltipText(currentProject.masterVolume,
                                                currentMeterLevels.first,
                                                currentMeterLevels.second);
    leftMeter.setTooltip(tooltip);
    rightMeter.setTooltip(tooltip);
    meterScale.setTooltip("Meter scale in dBFS");
    meterLabel.setTooltip("Peak level: " + peakDbfsLabel(juce::jmax(currentMeterLevels.first, currentMeterLevels.second)));
    volumeSlider.setTooltip("Fader: " + gainDbLabel(currentProject.masterVolume));
}

void MixerComponent::MasterStrip::clampFxScrollOffset()
{
    const auto maxOffset = juce::jmax(0, static_cast<int>(fxRows.size()) - juce::jmax(1, fxVisibleRowCount));
    fxScrollOffset = juce::jlimit(0, maxOffset, fxScrollOffset);
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
                               safeThis->currentProject.masterFxSlotBypassed.clear();
                               safeThis->commitProjectEdit("Clear Master FX");
                               return;
                           }

                           for (const auto& [itemId, reference] : effectAssignments)
                           {
                               if (result != itemId)
                                   continue;

                               int existingIndex = -1;
                               for (int index = 0; index < safeThis->currentProject.masterFxChain.size(); ++index)
                               {
                                   if (safeThis->currentProject.masterFxChain[index].equalsIgnoreCase(reference))
                                   {
                                       existingIndex = index;
                                       break;
                                   }
                               }

                               if (existingIndex >= 0)
                                   removeFxSlotAt(safeThis->currentProject.masterFxChain, safeThis->currentProject.masterFxSlotBypassed, existingIndex);
                               else
                                   appendFxSlot(safeThis->currentProject.masterFxChain, safeThis->currentProject.masterFxSlotBypassed, reference);

                               safeThis->commitProjectEdit("Update Master FX Chain");
                               return;
                           }
                       });
}

int MixerComponent::MasterStrip::findFxRowIndexForComponent(const juce::Component* component) const
{
    if (component == nullptr)
        return -1;

    for (const auto& row : fxRows)
    {
        if (row != nullptr && row->nameLabel.get() == component)
            return row->effectIndex;
    }

    return -1;
}

int MixerComponent::MasterStrip::findFxInsertIndexForY(int y) const
{
    int insertIndex = 0;
    bool foundVisibleRow = false;

    for (const auto& row : fxRows)
    {
        if (row == nullptr || row->rowBounds.isEmpty())
            continue;

        foundVisibleRow = true;
        if (y < row->rowBounds.getCentreY())
            return row->effectIndex;

        insertIndex = row->effectIndex + 1;
    }

    return foundVisibleRow ? insertIndex : currentProject.masterFxChain.size();
}

int MixerComponent::MasterStrip::fxInsertionLineY(int insertIndex) const
{
    const auto clampedIndex = juce::jlimit(0, currentProject.masterFxChain.size(), insertIndex);

    for (const auto& row : fxRows)
    {
        if (row == nullptr || row->rowBounds.isEmpty())
            continue;

        if (row->effectIndex == clampedIndex)
            return row->rowBounds.getY();
    }

    for (auto rowIt = fxRows.rbegin(); rowIt != fxRows.rend(); ++rowIt)
    {
        const auto& row = *rowIt;
        if (row == nullptr || row->rowBounds.isEmpty())
            continue;

        return row->rowBounds.getBottom();
    }

    return 0;
}

void MixerComponent::MasterStrip::finishFxDrag(bool commitChanges)
{
    const auto sourceIndex = fxDragSourceIndex;
    const auto insertIndex = fxDropInsertIndex;
    const auto wasActive = fxDragActive;

    fxDragActive = false;
    fxDragSourceIndex = -1;
    fxDropInsertIndex = -1;
    repaint();

    if (!commitChanges || !wasActive)
        return;

    moveEffectToInsertIndex(sourceIndex, insertIndex);
}

void MixerComponent::MasterStrip::moveEffectToInsertIndex(int sourceIndex, int insertIndex)
{
    const auto effectCount = currentProject.masterFxChain.size();
    if (!juce::isPositiveAndBelow(sourceIndex, effectCount))
        return;

    auto adjustedInsertIndex = juce::jlimit(0, effectCount, insertIndex);
    if (adjustedInsertIndex == sourceIndex || adjustedInsertIndex == sourceIndex + 1)
        return;

    const auto movedReference = currentProject.masterFxChain[sourceIndex];
    const auto movedBypassed = fxSlotBypassed(currentProject.masterFxSlotBypassed, sourceIndex);

    currentProject.masterFxChain.remove(sourceIndex);
    if (juce::isPositiveAndBelow(sourceIndex, static_cast<int>(currentProject.masterFxSlotBypassed.size())))
        currentProject.masterFxSlotBypassed.erase(currentProject.masterFxSlotBypassed.begin() + sourceIndex);

    if (adjustedInsertIndex > sourceIndex)
        --adjustedInsertIndex;

    currentProject.masterFxChain.insert(adjustedInsertIndex, movedReference);
    currentProject.masterFxSlotBypassed.insert(currentProject.masterFxSlotBypassed.begin() + adjustedInsertIndex, movedBypassed);
    ensureFxBypassStateSize(currentProject.masterFxSlotBypassed, currentProject.masterFxChain.size());
    commitProjectEdit("Reorder Master FX");
}

void MixerComponent::MasterStrip::mouseDown(const juce::MouseEvent& event)
{
    const auto* component = event.originalComponent != nullptr ? event.originalComponent : event.eventComponent;
    const auto rowIndex = findFxRowIndexForComponent(component);
    if (rowIndex < 0)
        return;

    fxDragActive = true;
    fxDragSourceIndex = rowIndex;
    fxDropInsertIndex = rowIndex;
    repaint();
}

void MixerComponent::MasterStrip::mouseDrag(const juce::MouseEvent& event)
{
    if (!fxDragActive)
        return;

    fxDropInsertIndex = findFxInsertIndexForY(juce::roundToInt(event.getEventRelativeTo(this).position.y));
    repaint();
}

void MixerComponent::MasterStrip::mouseUp(const juce::MouseEvent&)
{
    finishFxDrag(true);
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

    const auto width = juce::jmax(760, ((trackCount + 1) * kStripWidth) + (juce::jmax(0, trackCount) * kStripGap) + 18);
    setSize(width, juce::jmax(360, getHeight()));
}

} // namespace aims
