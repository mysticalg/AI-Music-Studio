#include "PianoRollComponent.h"

#include <array>
#include <algorithm>
#include <cmath>
#include <limits>
#include <set>

namespace aims
{
namespace
{
constexpr int kPitchMin = 12;
constexpr int kPitchMax = 84;
const char* kClipboardType = "aims.native.piano_roll_notes";
const juce::Colour kBackgroundColour = juce::Colour::fromRGB(15, 17, 22);
const juce::Colour kLaneDark = juce::Colour::fromRGB(24, 28, 34);
const juce::Colour kLaneLight = juce::Colour::fromRGB(31, 35, 42);
const juce::Colour kBlackKeyLane = juce::Colour::fromRGB(20, 23, 28);
const juce::Colour kGridMajor = juce::Colour::fromRGB(76, 87, 105);
const juce::Colour kGridMinor = juce::Colour::fromRGB(47, 54, 66);
const juce::Colour kGridSnap = juce::Colour::fromRGBA(110, 120, 136, 70);
const juce::Colour kPlayheadColour = juce::Colour::fromRGB(255, 102, 102);
const juce::Colour kLeftLocatorColour = juce::Colour::fromRGB(120, 212, 255);
const juce::Colour kRightLocatorColour = juce::Colour::fromRGB(255, 209, 102);
const juce::Colour kSelectionOutline = juce::Colour::fromRGB(244, 248, 255);
const juce::Colour kMarqueeFill = juce::Colour::fromRGBA(128, 176, 255, 46);
const juce::Colour kMarqueeOutline = juce::Colour::fromRGB(144, 196, 255);
const juce::Colour kControllerHeader = juce::Colour::fromRGB(19, 22, 28);
const juce::Colour kControllerLaneBackground = juce::Colour::fromRGB(17, 20, 26);
const juce::Colour kControllerLaneGrid = juce::Colour::fromRGB(42, 48, 58);
const juce::Colour kControllerLaneValue = juce::Colour::fromRGB(110, 190, 255);
const juce::Colour kControllerLaneValueFill = juce::Colour::fromRGBA(110, 190, 255, 42);
const juce::Colour kControllerLanePoint = juce::Colour::fromRGB(232, 242, 255);
constexpr float kTransportHandleWidth = 16.0f;
constexpr float kTransportHandleHeight = 10.0f;

bool isBlackPitch(int pitch)
{
    switch (pitch % 12)
    {
        case 1: case 3: case 6: case 8: case 10:
            return true;
        default:
            return false;
    }
}

juce::String pitchLabel(int pitch)
{
    static const char* names[] = { "C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B" };
    return juce::String(names[pitch % 12]) + juce::String((pitch / 12) - 1);
}

juce::String serialiseClipboardNotes(const std::vector<MidiNote>& notes)
{
    if (notes.empty())
        return {};

    int earliestTick = notes.front().startTick;
    for (const auto& note : notes)
        earliestTick = juce::jmin(earliestTick, note.startTick);

    juce::Array<juce::var> noteArray;
    for (const auto& note : notes)
    {
        auto* noteObject = new juce::DynamicObject();
        noteObject->setProperty("startTick", juce::jmax(0, note.startTick - earliestTick));
        noteObject->setProperty("durationTick", juce::jmax(1, note.durationTick));
        noteObject->setProperty("pitch", juce::jlimit(0, 127, note.pitch));
        noteObject->setProperty("velocity", juce::jlimit(1, 127, note.velocity));
        noteArray.add(juce::var(noteObject));
    }

    auto* root = new juce::DynamicObject();
    root->setProperty("type", kClipboardType);
    root->setProperty("notes", juce::var(noteArray));
    return juce::JSON::toString(juce::var(root), true);
}

bool parseClipboardNotes(const juce::String& text, std::vector<MidiNote>& outNotes)
{
    outNotes.clear();
    const auto clipboard = text.trim();
    if (clipboard.isEmpty())
        return false;

    const auto json = juce::JSON::parse(clipboard);
    auto* object = json.getDynamicObject();
    if (object == nullptr)
        return false;

    if (!object->getProperty("type").toString().equalsIgnoreCase(kClipboardType))
        return false;

    const auto notesVar = object->getProperty("notes");
    auto* noteArray = notesVar.getArray();
    if (noteArray == nullptr || noteArray->isEmpty())
        return false;

    outNotes.reserve(static_cast<size_t>(noteArray->size()));
    for (const auto& item : *noteArray)
    {
        auto* noteObject = item.getDynamicObject();
        if (noteObject == nullptr)
            continue;

        MidiNote note;
        note.startTick = juce::jmax(0, static_cast<int>(noteObject->getProperty("startTick")));
        note.durationTick = juce::jmax(1, static_cast<int>(noteObject->getProperty("durationTick")));
        note.pitch = juce::jlimit(0, 127, static_cast<int>(noteObject->getProperty("pitch")));
        note.velocity = juce::jlimit(1, 127, static_cast<int>(noteObject->getProperty("velocity")));
        note.selected = true;
        outNotes.push_back(note);
    }

    return !outNotes.empty();
}

void sortMidiNotes(std::vector<MidiNote>& notes)
{
    std::sort(notes.begin(),
              notes.end(),
              [] (const MidiNote& lhs, const MidiNote& rhs)
              {
                  if (lhs.startTick != rhs.startTick)
                      return lhs.startTick < rhs.startTick;
                  if (lhs.pitch != rhs.pitch)
                      return lhs.pitch > rhs.pitch;
                  return lhs.durationTick < rhs.durationTick;
              });
}

bool noteRangesTouchOrOverlap(const MidiNote& lhs, const MidiNote& rhs)
{
    const auto lhsEndTick = lhs.startTick + lhs.durationTick;
    const auto rhsEndTick = rhs.startTick + rhs.durationTick;
    return lhsEndTick >= rhs.startTick && rhsEndTick >= lhs.startTick;
}

juce::String pencilDrawModeLabel(PianoRollComponent::PencilDrawMode mode)
{
    switch (mode)
    {
        case PianoRollComponent::PencilDrawMode::step: return "Single";
        case PianoRollComponent::PencilDrawMode::brush: return "Brush";
        case PianoRollComponent::PencilDrawMode::line: return "Line";
        case PianoRollComponent::PencilDrawMode::box: return "Box";
        case PianoRollComponent::PencilDrawMode::sine: return "Sine";
        case PianoRollComponent::PencilDrawMode::square: return "Square";
        case PianoRollComponent::PencilDrawMode::saw: return "Saw";
        case PianoRollComponent::PencilDrawMode::triangle: return "Triangle";
        case PianoRollComponent::PencilDrawMode::circle: return "Circle";
    }

    return "Single";
}

juce::String velocityControllerTarget()
{
    return "velocity";
}
} // namespace

PianoRollComponent::PianoRollComponent(ProjectGetter projectGetterIn,
                                       TrackIndexGetter trackIndexGetterIn,
                                       SelectedSectionIndexGetter selectedSectionIndexGetterIn,
                                       ProjectWriter projectWriterIn)
    : projectGetter(std::move(projectGetterIn)),
      trackIndexGetter(std::move(trackIndexGetterIn)),
      selectedSectionIndexGetter(std::move(selectedSectionIndexGetterIn)),
      projectWriter(std::move(projectWriterIn))
{
    setWantsKeyboardFocus(true);
    controllerTargetLabel.setText("Lane", juce::dontSendNotification);
    controllerTargetLabel.setColour(juce::Label::textColourId, juce::Colour::fromRGB(210, 218, 228));
    addAndMakeVisible(controllerTargetLabel);

    controllerTargetBox.setColour(juce::ComboBox::backgroundColourId, juce::Colour::fromRGB(35, 40, 48));
    controllerTargetBox.setColour(juce::ComboBox::outlineColourId, juce::Colour::fromRGB(74, 83, 99));
    controllerTargetBox.setColour(juce::ComboBox::textColourId, juce::Colour::fromRGB(233, 238, 246));
    controllerTargetBox.onChange = [this]
    {
        const auto index = controllerTargetBox.getSelectedItemIndex();
        if (juce::isPositiveAndBelow(index, static_cast<int>(controllerTargetOptions.size())))
        {
            selectedControllerTarget = controllerTargetOptions[static_cast<size_t>(index)];
            repaint();
        }
    };
    addAndMakeVisible(controllerTargetBox);

    refreshControllerTargetChoices();
    updateContentSize();
}

PianoRollComponent::~PianoRollComponent()
{
    stopPreviewNote();
}

void PianoRollComponent::refreshFromModel()
{
    refreshControllerTargetChoices();
    const auto hasPattern = currentPattern() != nullptr;
    const auto showControllerSelectors = hasPattern && showsControllerEditor();
    controllerTargetLabel.setVisible(showControllerSelectors);
    controllerTargetBox.setVisible(showControllerSelectors);
    if (!previewActive)
        updateContentSize();
    repaint();
}

void PianoRollComponent::setToolMode(EditorToolMode mode)
{
    toolMode = mode;
}

void PianoRollComponent::setSurfaceMode(SurfaceMode mode)
{
    if (surfaceMode == mode)
        return;

    surfaceMode = mode;
    refreshFromModel();
}

void PianoRollComponent::setHorizontalZoom(float pixelsPerBeat)
{
    const auto clamped = juce::jlimit(12.0f, 96.0f, pixelsPerBeat);
    if (std::abs(cellWidth - clamped) < 0.01f)
        return;

    cellWidth = clamped;
    if (!previewActive)
        updateContentSize();
    repaint();
}

float PianoRollComponent::getHorizontalZoom() const noexcept
{
    return cellWidth;
}

void PianoRollComponent::setNoteRowHeight(float height)
{
    const auto clamped = juce::jlimit(8.0f, 32.0f, height);
    if (std::abs(cellHeight - clamped) < 0.01f)
        return;

    cellHeight = clamped;
    if (!previewActive)
        updateContentSize();
    repaint();
}

float PianoRollComponent::getNoteRowHeight() const noexcept
{
    return cellHeight;
}

void PianoRollComponent::setNotePreviewCallbacks(NotePreviewCallback noteOnCallbackIn,
                                                 NotePreviewCallback noteOffCallbackIn,
                                                 PreviewStopCallback stopPreviewCallbackIn)
{
    notePreviewOn = std::move(noteOnCallbackIn);
    notePreviewOff = std::move(noteOffCallbackIn);
    stopPreviewCallback = std::move(stopPreviewCallbackIn);
}

void PianoRollComponent::setToolModeChangeCallback(ToolModeChangeCallback toolModeChangeCallbackIn)
{
    toolModeChangeCallback = std::move(toolModeChangeCallbackIn);
}

bool PianoRollComponent::copySelected() const
{
    const auto* pattern = currentPattern();
    if (pattern == nullptr || !hasSelectedNotes(*pattern))
        return false;

    std::vector<MidiNote> notes;
    for (const auto& note : pattern->notes)
    {
        if (note.selected)
            notes.push_back(note);
    }

    sortMidiNotes(notes);

    const auto clipboard = serialiseClipboardNotes(notes);
    if (clipboard.isEmpty())
        return false;

    juce::SystemClipboard::copyTextToClipboard(clipboard);
    return true;
}

bool PianoRollComponent::cutSelected()
{
    return copySelected() && deleteSelected();
}

bool PianoRollComponent::deleteSelected()
{
    const auto sectionIndex = currentSectionIndex();
    auto updatedProject = project();
    auto* pattern = patternForSection(updatedProject, sectionIndex);
    if (pattern == nullptr || !hasSelectedNotes(*pattern))
        return false;

    pattern->notes.erase(std::remove_if(pattern->notes.begin(),
                                        pattern->notes.end(),
                                        [] (const MidiNote& note) { return note.selected; }),
                         pattern->notes.end());
    projectWriter(updatedProject, true, "Delete Notes");
    return true;
}

bool PianoRollComponent::pasteClipboard()
{
    const auto sectionIndex = currentSectionIndex();
    auto updatedProject = project();
    auto* pattern = patternForSection(updatedProject, sectionIndex);
    if (pattern == nullptr)
        return false;

    std::vector<MidiNote> clipboardNotes;
    if (!parseClipboardNotes(juce::SystemClipboard::getTextFromClipboard(), clipboardNotes))
        return false;

    int destinationTick = 0;
    if (hasSelectedNotes(*pattern))
    {
        bool sawSelected = false;
        for (const auto& note : pattern->notes)
        {
            if (!note.selected)
                continue;
            destinationTick = sawSelected ? juce::jmin(destinationTick, note.startTick) : note.startTick;
            sawSelected = true;
        }
        destinationTick += gridTick();
    }
    else if (!pattern->notes.empty())
    {
        for (const auto& note : pattern->notes)
            destinationTick = juce::jmax(destinationTick, note.startTick + note.durationTick);

        const auto step = gridTick();
        destinationTick = ((destinationTick + step - 1) / step) * step;
    }

    clearSelection(*pattern);
    for (auto note : clipboardNotes)
    {
        note.startTick = juce::jmax(0, destinationTick + note.startTick);
        note.selected = true;
        pattern->notes.push_back(note);
    }

    sortMidiNotes(pattern->notes);

    projectWriter(updatedProject, true, "Paste Notes");
    return true;
}

bool PianoRollComponent::quantizeSelected()
{
    const auto sectionIndex = currentSectionIndex();
    auto updatedProject = project();
    auto* pattern = patternForSection(updatedProject, sectionIndex);
    if (pattern == nullptr || !hasSelectedNotes(*pattern))
        return false;

    const auto step = gridTick();
    for (auto& note : pattern->notes)
    {
        if (!note.selected)
            continue;

        note.startTick = juce::jmax(0, static_cast<int>(std::round(static_cast<double>(note.startTick) / static_cast<double>(step))) * step);
        note.durationTick = juce::jmax(step, static_cast<int>(std::round(static_cast<double>(note.durationTick) / static_cast<double>(step))) * step);
    }

    projectWriter(updatedProject, true, "Quantize Notes");
    return true;
}

bool PianoRollComponent::duplicateSelectedByGrid()
{
    const auto* sourcePattern = currentPattern();
    if (sourcePattern == nullptr || !hasSelectedNotes(*sourcePattern))
        return false;

    const auto sectionIndex = currentSectionIndex();
    auto updatedProject = project();
    auto* pattern = patternForSection(updatedProject, sectionIndex);
    if (pattern == nullptr)
        return false;

    const auto step = gridTick();
    for (auto& note : pattern->notes)
        note.selected = false;

    std::vector<MidiNote> duplicates;
    for (const auto& note : sourcePattern->notes)
    {
        if (!note.selected)
            continue;

        auto duplicate = note;
        duplicate.startTick = juce::jmax(0, note.startTick + step);
        duplicate.selected = true;
        duplicates.push_back(std::move(duplicate));
    }

    pattern->notes.insert(pattern->notes.end(), duplicates.begin(), duplicates.end());
    sortMidiNotes(pattern->notes);
    projectWriter(updatedProject, true, "Duplicate Notes");
    return true;
}

bool PianoRollComponent::selectAllNotes()
{
    const auto sectionIndex = currentSectionIndex();
    auto updatedProject = project();
    auto* pattern = patternForSection(updatedProject, sectionIndex);
    if (pattern == nullptr || pattern->notes.empty())
        return false;

    bool changed = false;
    for (auto& note : pattern->notes)
    {
        changed = changed || !note.selected;
        note.selected = true;
    }

    if (!changed)
        return false;

    projectWriter(updatedProject, false, {});
    return true;
}

void PianoRollComponent::paint(juce::Graphics& g)
{
    g.fillAll(kBackgroundColour);

    const auto contentBounds = getLocalBounds().toFloat();
    const auto pianoWidth = pitchLaneWidth();
    const auto headerHeight = rulerHeight();
    const auto pitches = visiblePitches();
    const auto pitchCount = static_cast<int>(pitches.size());

    g.setColour(juce::Colour::fromRGB(13, 15, 19));
    g.fillRect(0.0f, 0.0f, pianoWidth, contentBounds.getHeight());

    if (showsNoteEditor())
    {
        for (int row = 0; row < pitchCount; ++row)
        {
            const auto pitch = rowToPitch(row);
            const auto y = headerHeight + (static_cast<float>(row) * cellHeight);
            const auto laneColour = isBlackPitch(pitch) ? kBlackKeyLane : ((row % 2) == 0 ? kLaneDark : kLaneLight);

            g.setColour(laneColour);
            g.fillRect(pianoWidth, y, contentBounds.getWidth() - pianoWidth, cellHeight);

            g.setColour(isBlackPitch(pitch) ? juce::Colour::fromRGB(28, 32, 38) : juce::Colour::fromRGB(42, 48, 58));
            g.fillRect(0.0f, y, pianoWidth, cellHeight);

            g.setColour(juce::Colour::fromRGB(170, 180, 196));
            g.setFont(11.0f);
            g.drawText(pitchLabel(pitch),
                       4,
                       juce::roundToInt(y),
                       juce::roundToInt(pianoWidth) - 8,
                       juce::roundToInt(cellHeight),
                       juce::Justification::centredLeft);

            g.setColour(juce::Colour::fromRGB(36, 42, 52));
            g.drawHorizontalLine(juce::roundToInt(y), pianoWidth, contentBounds.getWidth());
        }

        g.setColour(juce::Colour::fromRGB(18, 21, 28));
        g.fillRect(pianoWidth, 0.0f, contentBounds.getWidth() - pianoWidth, headerHeight);
    }
    g.setColour(juce::Colour::fromRGB(58, 66, 80));
    g.drawRect(contentBounds.toNearestInt(), 1);

    const auto totalTicks = contentTickLength();
    const auto totalBeats = juce::jmax(minimumBeats, static_cast<int>(std::ceil(static_cast<double>(totalTicks) / static_cast<double>(kTicksPerBeat))));
    const auto step = gridTick();

    if (showsNoteEditor())
    {
        for (int beat = 0; beat <= totalBeats; ++beat)
        {
            const auto tick = beat * kTicksPerBeat;
            const auto x = tickToX(tick);
            g.setColour((beat % 4) == 0 ? kGridMajor : kGridMinor);
            g.drawVerticalLine(juce::roundToInt(x), 0.0f, contentBounds.getHeight());

            if ((beat % 4) == 0)
            {
                g.setColour(juce::Colour::fromRGB(210, 216, 224));
                g.drawText(juce::String((beat / 4) + 1),
                           juce::roundToInt(x) + 3,
                           0,
                           30,
                           juce::roundToInt(headerHeight),
                           juce::Justification::centredLeft);
            }
        }

        if (step < kTicksPerBeat)
        {
            for (int tick = step; tick < totalTicks; tick += step)
            {
                if ((tick % kTicksPerBeat) == 0)
                    continue;
                const auto x = tickToX(tick);
                g.setColour(kGridSnap);
                g.drawVerticalLine(juce::roundToInt(x), headerHeight, contentBounds.getHeight());
            }
        }

        if (const auto* section = currentSection())
        {
            const auto& projectState = project();
            for (const auto& locator : { std::pair(projectState.leftLocatorTick, kLeftLocatorColour),
                                         std::pair(projectState.rightLocatorTick, kRightLocatorColour),
                                         std::pair(projectState.playheadTick, kPlayheadColour) })
            {
                const auto relativeTick = locator.first - section->startTick;
                if (relativeTick < 0 || relativeTick > totalTicks)
                    continue;

                const auto x = tickToX(relativeTick);
                g.setColour(locator.second);
                g.drawVerticalLine(juce::roundToInt(x), 0.0f, contentBounds.getHeight());
                const auto handle = transportHandleBounds(locator.first);
                g.fillRoundedRectangle(handle, 4.0f);
                juce::Path pointer;
                pointer.addTriangle(x,
                                    handle.getBottom() + 4.0f,
                                    x - 4.5f,
                                    handle.getBottom() - 1.0f,
                                    x + 4.5f,
                                    handle.getBottom() - 1.0f);
                g.fillPath(pointer);
            }
        }
    }

    if (currentPattern() == nullptr)
    {
        g.setColour(juce::Colour::fromRGB(192, 199, 208));
        g.setFont(juce::FontOptions(18.0f, juce::Font::bold));
        g.drawFittedText("Select or create a pattern clip in the sequencer.",
                         getLocalBounds().reduced(24),
                         juce::Justification::centred,
                         2);
        return;
    }

    const auto& track = currentTrack() != nullptr ? *currentTrack() : TrackState{};
    const auto trackIndex = currentTrackIndex();
    const auto trackColour = trackDisplayColour(track, trackIndex);
    if (showsNoteEditor())
    {
        for (const auto& note : drawingPattern().notes)
        {
            auto rect = noteRect(note);
            if (rect.isEmpty())
                continue;
            const auto fill = note.selected ? trackColour.brighter(0.25f) : trackColour.withAlpha(0.88f);
            const auto outline = note.selected ? kSelectionOutline : trackColour.darker(0.7f);

            g.setColour(fill);
            g.fillRoundedRectangle(rect.reduced(1.0f, 1.0f), 3.0f);
            g.setColour(outline);
            g.drawRoundedRectangle(rect.reduced(1.0f, 1.0f), 3.0f, note.selected ? 2.0f : 1.0f);
        }
    }

    if (showsControllerEditor())
    {
        const auto controllerHeader = controllerHeaderBounds();
        const auto controllerLane = controllerLaneBounds();
        g.setColour(kControllerHeader);
        g.fillRect(controllerHeader);
        g.setColour(kControllerLaneBackground);
        g.fillRect(controllerLane);
        g.setColour(juce::Colour::fromRGB(40, 46, 56));
        g.drawHorizontalLine(juce::roundToInt(controllerHeader.getY()), 0.0f, static_cast<float>(getWidth()));
        g.drawHorizontalLine(juce::roundToInt(controllerHeader.getBottom()), 0.0f, static_cast<float>(getWidth()));

        const auto controllerBounds = controllerTargetBounds();
        g.setColour(juce::Colour::fromRGB(170, 180, 196));
        g.setFont(11.0f);
        g.drawText(juce::String(controllerBounds.second, controllerTargetEditsVelocity() || controllerTargetEditsPatternLane() ? 0 : 2),
                   4,
                   juce::roundToInt(controllerLane.getY() + 2.0f),
                   juce::roundToInt(pitchLaneWidth()) - 8,
                   18,
                   juce::Justification::centredLeft);
        g.drawText(juce::String(controllerBounds.first, controllerTargetEditsVelocity() || controllerTargetEditsPatternLane() ? 0 : 2),
                   4,
                   juce::roundToInt(controllerLane.getBottom() - 20.0f),
                   juce::roundToInt(pitchLaneWidth()) - 8,
                   18,
                   juce::Justification::centredLeft);

        for (int guideIndex = 0; guideIndex <= 4; ++guideIndex)
        {
            const auto y = juce::jmap(static_cast<float>(guideIndex), 0.0f, 4.0f, controllerLane.getY(), controllerLane.getBottom());
            g.setColour(guideIndex == 0 || guideIndex == 4 ? kGridMajor : kControllerLaneGrid);
            g.drawHorizontalLine(juce::roundToInt(y), pitchLaneWidth(), static_cast<float>(getWidth()));
        }

        for (int beat = 0; beat <= totalBeats; ++beat)
        {
            const auto tick = beat * kTicksPerBeat;
            const auto x = tickToX(tick);
            g.setColour((beat % 4) == 0 ? kGridMajor : kGridMinor);
            g.drawVerticalLine(juce::roundToInt(x), controllerHeader.getY(), controllerLane.getBottom());
        }

        if (step < kTicksPerBeat)
        {
            for (int tick = step; tick < totalTicks; tick += step)
            {
                if ((tick % kTicksPerBeat) == 0)
                    continue;
                const auto x = tickToX(tick);
                g.setColour(kGridSnap);
                g.drawVerticalLine(juce::roundToInt(x), controllerLane.getY(), controllerLane.getBottom());
            }
        }

        if (controllerTargetEditsVelocity())
        {
            for (const auto& note : drawingPattern().notes)
            {
                auto bar = controllerBarRect(note).getIntersection(controllerLane);
                if (bar.isEmpty())
                    continue;

                g.setColour(note.selected ? trackColour.brighter(0.2f) : trackColour.withAlpha(0.75f));
                g.fillRoundedRectangle(bar.reduced(0.5f, 0.0f), 2.0f);
                g.setColour(note.selected ? kSelectionOutline : trackColour.darker(0.7f));
                g.drawRoundedRectangle(bar.reduced(0.5f, 0.0f), 2.0f, note.selected ? 1.5f : 1.0f);
            }
        }
        else
        {
            const auto points = displayedControllerPoints();
            if (!points.empty())
            {
                juce::Path fillPath;
                juce::Path strokePath;

                const auto firstX = tickToX(points.front().tick);
                const auto firstY = yForControllerValue(points.front().value);
                fillPath.startNewSubPath(firstX, controllerLane.getBottom());
                fillPath.lineTo(firstX, firstY);
                strokePath.startNewSubPath(firstX, firstY);

                for (size_t pointIndex = 1; pointIndex < points.size(); ++pointIndex)
                {
                    const auto x = tickToX(points[pointIndex].tick);
                    const auto y = yForControllerValue(points[pointIndex].value);
                    fillPath.lineTo(x, y);
                    strokePath.lineTo(x, y);
                }

                const auto lastX = tickToX(points.back().tick);
                fillPath.lineTo(lastX, controllerLane.getBottom());
                fillPath.closeSubPath();

                g.setColour(kControllerLaneValueFill);
                g.fillPath(fillPath);
                g.setColour(kControllerLaneValue);
                g.strokePath(strokePath, juce::PathStrokeType(2.0f));

                for (const auto& point : points)
                {
                    const auto x = tickToX(point.tick);
                    const auto y = yForControllerValue(point.value);
                    g.setColour(kControllerLanePoint);
                    g.fillEllipse(x - 3.0f, y - 3.0f, 6.0f, 6.0f);
                    g.setColour(kControllerLaneValue.darker(0.3f));
                    g.drawEllipse(x - 3.0f, y - 3.0f, 6.0f, 6.0f, 1.0f);
                }
            }
        }
    }

    if (previewActive && interaction == Interaction::marqueeSelect && !marqueeRect.isEmpty())
    {
        auto drawRect = marqueeRect.getIntersection(controllerLaneInteraction ? controllerLaneBounds() : noteGridBounds()).reduced(0.5f);
        if (!drawRect.isEmpty())
        {
            g.setColour(kMarqueeFill);
            g.fillRect(drawRect);
            g.setColour(kMarqueeOutline);
            g.drawRect(drawRect, 1.5f);
        }
    }
}

void PianoRollComponent::resized()
{
    updateContentSize();

    if (showsControllerEditor())
    {
        auto headerBounds = controllerHeaderBounds().toNearestInt().reduced(8, 4);
        controllerTargetLabel.setBounds(headerBounds.removeFromLeft(36));
        controllerTargetBox.setBounds(headerBounds.removeFromLeft(260));
    }
    else
    {
        controllerTargetLabel.setBounds({});
        controllerTargetBox.setBounds({});
    }
}

void PianoRollComponent::mouseMove(const juce::MouseEvent& event)
{
    updateCursorForPosition(event.position);
}

void PianoRollComponent::mouseDown(const juce::MouseEvent& event)
{
    if (event.mods.isPopupMenu())
    {
        showContextMenu(event.getScreenPosition());
        return;
    }

    if (currentPattern() == nullptr)
        return;

    if (showsNoteEditor()
        && event.position.x > pitchLaneWidth() && event.position.y <= rulerHeight())
    {
        const auto markerInteraction = hitTestTransportMarker(event.position);
        if (markerInteraction != Interaction::none && event.mods.isLeftButtonDown())
        {
            grabKeyboardFocus();
            pitchLanePreviewActive = false;
            beforeProject = project();
            previewProject = beforeProject;
            previewActive = true;
            previewDirty = false;
            activeNoteIndex = -1;
            interaction = markerInteraction;
            controllerLaneInteraction = false;
            marqueeRect = {};
            drawPathPoints.clear();
            selectedSnapshots.clear();
            repaint();
            return;
        }

        auto updatedProject = project();
        const auto relativeTick = xToGridStartTick(event.position.x);
        const auto baseTick = currentSection() != nullptr ? currentSection()->startTick : 0;
        const auto absoluteTick = juce::jmax(0, baseTick + relativeTick);

        if (event.mods.isShiftDown())
        {
            updatedProject.leftLocatorTick = absoluteTick;
            updatedProject.rightLocatorTick = juce::jmax(updatedProject.leftLocatorTick + gridTick(), updatedProject.rightLocatorTick);
            updatedProject.recalculateTimeFields();
            projectWriter(updatedProject, true, "Set Left Locator");
        }
        else if (event.mods.isAltDown() || event.mods.isRightButtonDown())
        {
            updatedProject.rightLocatorTick = juce::jmax(updatedProject.leftLocatorTick + gridTick(), absoluteTick);
            updatedProject.recalculateTimeFields();
            projectWriter(updatedProject, true, "Set Right Locator");
        }
        else
        {
            updatedProject.playheadTick = absoluteTick;
            updatedProject.recalculateTimeFields();
            projectWriter(updatedProject, false, "Move Playhead");
        }

        refreshFromModel();
        return;
    }

    if (event.position.x <= pitchLaneWidth()
        && event.position.y > rulerHeight()
        && event.position.y < noteGridBounds().getBottom()
        && event.mods.isLeftButtonDown())
    {
        pitchLanePreviewActive = true;
        startPreviewNote(pitchForY(event.position.y));
        return;
    }

    grabKeyboardFocus();
    pitchLanePreviewActive = false;

    beforeProject = project();
    previewProject = beforeProject;
    previewActive = true;
    previewDirty = false;
    activeNoteIndex = -1;
    interaction = Interaction::none;
    controllerLaneInteraction = false;
    marqueeRect = {};
    drawPathPoints.clear();
    selectedSnapshots.clear();

    auto* previewPattern = patternForSection(previewProject, currentSectionIndex());
    if (previewPattern == nullptr)
    {
        previewActive = false;
        return;
    }

    juce::String edge;
    const auto noteIndex = hitTestNote(event.position, edge);
    anchorGridTick = xToGridStartTick(event.position.x);
    anchorPitchRow = pitchRowForY(event.position.y);
    anchorControllerValue = controllerValueForY(event.position.y);

    if (isControllerLanePosition(event.position)
        && event.mods.isLeftButtonDown()
        && event.position.x > pitchLaneWidth())
    {
        controllerLaneInteraction = true;

        if (toolMode == EditorToolMode::selection && controllerTargetEditsVelocity())
        {
            marqueeStart = controllerLaneBounds().getConstrainedPoint(event.position);
            marqueeRect = juce::Rectangle<float>(marqueeStart.x, marqueeStart.y, 0.0f, 0.0f);
            interaction = Interaction::marqueeSelect;
        }
        else if (toolMode == EditorToolMode::pencil)
        {
            interaction = Interaction::drawShape;
            shapeAmplitudeRows = 6;
            shapeFrequencyCycles = 1;
            drawPathPoints.push_back(event.position);
            rebuildControllerPreview(event.position, false);
            previewDirty = true;
        }
        else
        {
            previewActive = false;
            previewDirty = false;
            interaction = Interaction::none;
        }

        repaint();
        return;
    }

    if (toolMode == EditorToolMode::eraser
        && event.mods.isLeftButtonDown()
        && noteGridBounds().contains(event.position))
    {
        clearSelection(*previewPattern);
        interaction = Interaction::erase;
        const auto constrainedPoint = noteGridBounds().getConstrainedPoint(event.position);
        drawPathPoints.push_back(constrainedPoint);
        previewDirty = eraseNotesAtPosition(*previewPattern, constrainedPoint);
        repaint();
        return;
    }

    if (noteIndex >= 0)
    {
        if (toolMode == EditorToolMode::glue && event.mods.isLeftButtonDown())
        {
            previewActive = false;
            previewDirty = false;
            interaction = Interaction::none;
            if (glueSelectedOrClickedNotes(noteIndex))
                refreshFromModel();
            else
                repaint();
            return;
        }

        if (!previewPattern->notes[static_cast<size_t>(noteIndex)].selected)
            selectSingleNote(*previewPattern, noteIndex);

        activeNoteIndex = noteIndex;
        anchorNoteStartTick = previewPattern->notes[static_cast<size_t>(noteIndex)].startTick;
        anchorNoteEndTick = anchorNoteStartTick + previewPattern->notes[static_cast<size_t>(noteIndex)].durationTick;

        for (int index = 0; index < static_cast<int>(previewPattern->notes.size()); ++index)
        {
            const auto& note = previewPattern->notes[static_cast<size_t>(index)];
            if (!note.selected)
                continue;
            selectedSnapshots.push_back({ index, note.startTick, note.pitch, note.durationTick });
        }

        if (edge == "left")
            interaction = Interaction::resizeLeft;
        else if (edge == "right")
            interaction = Interaction::resizeRight;
        else
            interaction = Interaction::drag;
    }
    else if (toolMode == EditorToolMode::pencil
             && event.mods.isLeftButtonDown()
             && event.position.x > pitchLaneWidth()
             && event.position.y > rulerHeight())
    {
        clearSelection(*previewPattern);

        if (pencilDrawMode == PencilDrawMode::step)
        {
            std::vector<int> insertedIndices;
            const auto noteStartTick = xToGridStartTick(event.position.x);
            const auto notePitch = pitchForY(event.position.y);
            insertNotesForCurrentMode(*previewPattern,
                                      noteStartTick,
                                      notePitch,
                                      noteLengthTick(),
                                      100,
                                      true,
                                      &insertedIndices);
            activeNoteIndex = insertedIndices.empty() ? -1 : insertedIndices.front();
            anchorNoteStartTick = noteStartTick;
            anchorNoteEndTick = noteStartTick + noteLengthTick();
            for (const auto index : insertedIndices)
            {
                const auto& note = previewPattern->notes[static_cast<size_t>(index)];
                selectedSnapshots.push_back({ index, note.startTick, note.pitch, note.durationTick });
            }
            interaction = Interaction::create;
            previewDirty = true;
            startPreviewNote(notePitch, 100);
        }
        else
        {
            interaction = Interaction::drawShape;
            shapeAmplitudeRows = 6;
            shapeFrequencyCycles = 1;
            drawPathPoints.push_back(event.position);
            rebuildShapePreview(event.position, false);
            previewDirty = true;
            startPreviewNote(pitchForY(event.position.y), 100);
        }
    }
    else if (toolMode == EditorToolMode::selection
             && event.mods.isLeftButtonDown()
             && event.position.x > pitchLaneWidth()
             && event.position.y > rulerHeight())
    {
        clearSelection(*previewPattern);
        marqueeStart = noteGridBounds().getConstrainedPoint(event.position);
        marqueeRect = juce::Rectangle<float>(marqueeStart.x, marqueeStart.y, 0.0f, 0.0f);
        interaction = Interaction::marqueeSelect;
    }
    else
    {
        clearSelection(*previewPattern);
    }

    repaint();
}

void PianoRollComponent::showContextMenu(juce::Point<int> screenPosition)
{
    juce::PopupMenu menu;

    const auto* pattern = currentPattern();
    const bool hasPattern = pattern != nullptr;
    const bool hasSelection = hasPattern && hasSelectedNotes(*pattern);
    std::vector<MidiNote> clipboardNotes;
    const bool canPaste = parseClipboardNotes(juce::SystemClipboard::getTextFromClipboard(), clipboardNotes);

    enum MenuItemIds
    {
        menuToolPencil = 1,
        menuToolSelect,
        menuToolGlue,
        menuToolEraser,
        menuDrawSingle,
        menuDrawBrush,
        menuDrawLine,
        menuDrawBox,
        menuDrawSine,
        menuDrawSquare,
        menuDrawSaw,
        menuDrawTriangle,
        menuDrawCircle,
        menuInsertSingle,
        menuInsertMajorTriad,
        menuInsertMinorTriad,
        menuInsertMajorFifth,
        menuInsertMinorFifth,
        menuInsertMajorSeventh,
        menuInsertMinorSeventh,
        menuInsertDominantSeventh,
        menuInsertSuspended2,
        menuInsertSuspended4,
        menuInsertResolvedMajor,
        menuInsertResolvedMinor,
        menuEditCut,
        menuEditCopy,
        menuEditPaste,
        menuEditDelete,
        menuEditDuplicate,
        menuEditQuantize,
        menuEditSelectAll
    };

    menu.addSectionHeader("Tools");
    menu.addItem(menuToolPencil, "Pencil", true, toolMode == EditorToolMode::pencil);
    menu.addItem(menuToolSelect, "Select", true, toolMode == EditorToolMode::selection);
    menu.addItem(menuToolGlue, "Glue", true, toolMode == EditorToolMode::glue);
    menu.addItem(menuToolEraser, "Eraser", true, toolMode == EditorToolMode::eraser);
    juce::PopupMenu drawMenu;
    drawMenu.addItem(menuDrawSingle, pencilDrawModeLabel(PencilDrawMode::step), true, pencilDrawMode == PencilDrawMode::step);
    drawMenu.addItem(menuDrawBrush, pencilDrawModeLabel(PencilDrawMode::brush), true, pencilDrawMode == PencilDrawMode::brush);
    drawMenu.addItem(menuDrawLine, pencilDrawModeLabel(PencilDrawMode::line), true, pencilDrawMode == PencilDrawMode::line);
    drawMenu.addItem(menuDrawBox, pencilDrawModeLabel(PencilDrawMode::box), true, pencilDrawMode == PencilDrawMode::box);
    drawMenu.addItem(menuDrawSine, pencilDrawModeLabel(PencilDrawMode::sine), true, pencilDrawMode == PencilDrawMode::sine);
    drawMenu.addItem(menuDrawSquare, pencilDrawModeLabel(PencilDrawMode::square), true, pencilDrawMode == PencilDrawMode::square);
    drawMenu.addItem(menuDrawSaw, pencilDrawModeLabel(PencilDrawMode::saw), true, pencilDrawMode == PencilDrawMode::saw);
    drawMenu.addItem(menuDrawTriangle, pencilDrawModeLabel(PencilDrawMode::triangle), true, pencilDrawMode == PencilDrawMode::triangle);
    drawMenu.addItem(menuDrawCircle, pencilDrawModeLabel(PencilDrawMode::circle), true, pencilDrawMode == PencilDrawMode::circle);
    menu.addSubMenu("Pencil Shape", drawMenu, true);

    juce::PopupMenu insertMenu;
    insertMenu.addItem(menuInsertSingle, pianoRollInsertModeLabel(PianoRollInsertMode::singleNote), true, insertMode == PianoRollInsertMode::singleNote);
    insertMenu.addItem(menuInsertMajorTriad, pianoRollInsertModeLabel(PianoRollInsertMode::majorTriad), true, insertMode == PianoRollInsertMode::majorTriad);
    insertMenu.addItem(menuInsertMinorTriad, pianoRollInsertModeLabel(PianoRollInsertMode::minorTriad), true, insertMode == PianoRollInsertMode::minorTriad);
    insertMenu.addItem(menuInsertMajorFifth, pianoRollInsertModeLabel(PianoRollInsertMode::majorFifth), true, insertMode == PianoRollInsertMode::majorFifth);
    insertMenu.addItem(menuInsertMinorFifth, pianoRollInsertModeLabel(PianoRollInsertMode::minorFifth), true, insertMode == PianoRollInsertMode::minorFifth);
    insertMenu.addItem(menuInsertMajorSeventh, pianoRollInsertModeLabel(PianoRollInsertMode::majorSeventh), true, insertMode == PianoRollInsertMode::majorSeventh);
    insertMenu.addItem(menuInsertMinorSeventh, pianoRollInsertModeLabel(PianoRollInsertMode::minorSeventh), true, insertMode == PianoRollInsertMode::minorSeventh);
    insertMenu.addItem(menuInsertDominantSeventh, pianoRollInsertModeLabel(PianoRollInsertMode::dominantSeventh), true, insertMode == PianoRollInsertMode::dominantSeventh);
    insertMenu.addItem(menuInsertSuspended2, pianoRollInsertModeLabel(PianoRollInsertMode::suspended2), true, insertMode == PianoRollInsertMode::suspended2);
    insertMenu.addItem(menuInsertSuspended4, pianoRollInsertModeLabel(PianoRollInsertMode::suspended4), true, insertMode == PianoRollInsertMode::suspended4);
    insertMenu.addItem(menuInsertResolvedMajor, pianoRollInsertModeLabel(PianoRollInsertMode::resolvedMajor), true, insertMode == PianoRollInsertMode::resolvedMajor);
    insertMenu.addItem(menuInsertResolvedMinor, pianoRollInsertModeLabel(PianoRollInsertMode::resolvedMinor), true, insertMode == PianoRollInsertMode::resolvedMinor);
    menu.addSubMenu("Chord / Insert", insertMenu, true);
    menu.addSeparator();
    menu.addSectionHeader("Notes");
    menu.addItem(menuEditCut, "Cut", hasSelection);
    menu.addItem(menuEditCopy, "Copy", hasSelection);
    menu.addItem(menuEditPaste, "Paste", canPaste && hasPattern);
    menu.addItem(menuEditDelete, "Delete", hasSelection);
    menu.addItem(menuEditDuplicate, "Duplicate", hasSelection);
    menu.addItem(menuEditQuantize, "Quantize", hasSelection);
    menu.addItem(menuEditSelectAll, "Select All", hasPattern && !pattern->notes.empty());

    menu.showMenuAsync(juce::PopupMenu::Options().withTargetScreenArea(juce::Rectangle<int>(screenPosition.x, screenPosition.y, 1, 1)),
                       [safeThis = juce::Component::SafePointer<PianoRollComponent>(this)] (int result)
                       {
                           if (safeThis == nullptr || result == 0)
                               return;

                           switch (result)
                           {
                                case menuToolPencil:
                                    if (safeThis->toolModeChangeCallback != nullptr)
                                        safeThis->toolModeChangeCallback(EditorToolMode::pencil);
                                    else
                                        safeThis->setToolMode(EditorToolMode::pencil);
                                    safeThis->refreshFromModel();
                                    break;

                                case menuToolSelect:
                                    if (safeThis->toolModeChangeCallback != nullptr)
                                        safeThis->toolModeChangeCallback(EditorToolMode::selection);
                                    else
                                        safeThis->setToolMode(EditorToolMode::selection);
                                    safeThis->refreshFromModel();
                                    break;

                                case menuToolGlue:
                                    if (safeThis->toolModeChangeCallback != nullptr)
                                        safeThis->toolModeChangeCallback(EditorToolMode::glue);
                                    else
                                        safeThis->setToolMode(EditorToolMode::glue);
                                    safeThis->refreshFromModel();
                                    break;

                                case menuToolEraser:
                                    if (safeThis->toolModeChangeCallback != nullptr)
                                        safeThis->toolModeChangeCallback(EditorToolMode::eraser);
                                    else
                                        safeThis->setToolMode(EditorToolMode::eraser);
                                    safeThis->refreshFromModel();
                                    break;

                                case menuDrawSingle: safeThis->pencilDrawMode = PencilDrawMode::step; break;
                                case menuDrawBrush: safeThis->pencilDrawMode = PencilDrawMode::brush; break;
                                case menuDrawLine: safeThis->pencilDrawMode = PencilDrawMode::line; break;
                                case menuDrawBox: safeThis->pencilDrawMode = PencilDrawMode::box; break;
                                case menuDrawSine: safeThis->pencilDrawMode = PencilDrawMode::sine; break;
                                case menuDrawSquare: safeThis->pencilDrawMode = PencilDrawMode::square; break;
                                case menuDrawSaw: safeThis->pencilDrawMode = PencilDrawMode::saw; break;
                                case menuDrawTriangle: safeThis->pencilDrawMode = PencilDrawMode::triangle; break;
                                case menuDrawCircle: safeThis->pencilDrawMode = PencilDrawMode::circle; break;

                                case menuInsertSingle: safeThis->insertMode = PianoRollInsertMode::singleNote; break;
                                case menuInsertMajorTriad: safeThis->insertMode = PianoRollInsertMode::majorTriad; break;
                                case menuInsertMinorTriad: safeThis->insertMode = PianoRollInsertMode::minorTriad; break;
                                case menuInsertMajorFifth: safeThis->insertMode = PianoRollInsertMode::majorFifth; break;
                                case menuInsertMinorFifth: safeThis->insertMode = PianoRollInsertMode::minorFifth; break;
                                case menuInsertMajorSeventh: safeThis->insertMode = PianoRollInsertMode::majorSeventh; break;
                                case menuInsertMinorSeventh: safeThis->insertMode = PianoRollInsertMode::minorSeventh; break;
                                case menuInsertDominantSeventh: safeThis->insertMode = PianoRollInsertMode::dominantSeventh; break;
                                case menuInsertSuspended2: safeThis->insertMode = PianoRollInsertMode::suspended2; break;
                                case menuInsertSuspended4: safeThis->insertMode = PianoRollInsertMode::suspended4; break;
                                case menuInsertResolvedMajor: safeThis->insertMode = PianoRollInsertMode::resolvedMajor; break;
                                case menuInsertResolvedMinor: safeThis->insertMode = PianoRollInsertMode::resolvedMinor; break;

                               case menuEditCut:
                                   safeThis->cutSelected();
                                   break;

                               case menuEditCopy:
                                   safeThis->copySelected();
                                   break;

                               case menuEditPaste:
                                   safeThis->pasteClipboard();
                                   break;

                               case menuEditDelete:
                                   safeThis->deleteSelected();
                                   break;

                               case menuEditDuplicate:
                                   safeThis->duplicateSelectedByGrid();
                                   break;

                               case menuEditQuantize:
                                   safeThis->quantizeSelected();
                                   break;

                               case menuEditSelectAll:
                                   safeThis->selectAllNotes();
                                   break;

                               default:
                                   break;
                           }
                       });
}

void PianoRollComponent::mouseDrag(const juce::MouseEvent& event)
{
    if (pitchLanePreviewActive)
    {
        updatePreviewNotePitch(pitchForY(event.position.y));
        return;
    }

    if (!previewActive)
        return;

    if (interaction == Interaction::moveLeftLocator
        || interaction == Interaction::moveRightLocator
        || interaction == Interaction::movePlayhead)
    {
        previewProject = beforeProject;
        const auto relativeTick = xToGridStartTick(event.position.x);
        const auto baseTick = currentSection() != nullptr ? currentSection()->startTick : 0;
        const auto absoluteTick = juce::jmax(0, baseTick + relativeTick);
        const auto minimumSpan = gridTick();

        if (interaction == Interaction::moveLeftLocator)
            previewProject.leftLocatorTick = juce::jmin(absoluteTick, juce::jmax(0, previewProject.rightLocatorTick - minimumSpan));
        else if (interaction == Interaction::moveRightLocator)
            previewProject.rightLocatorTick = juce::jmax(previewProject.leftLocatorTick + minimumSpan, absoluteTick);
        else
            previewProject.playheadTick = absoluteTick;

        previewProject.recalculateTimeFields();
        previewDirty = true;
        repaint();
        return;
    }

    auto* previewPattern = patternForSection(previewProject, currentSectionIndex());
    if (previewPattern == nullptr)
        return;

    if (interaction == Interaction::marqueeSelect)
    {
        const auto current = (controllerLaneInteraction ? controllerLaneBounds() : noteGridBounds()).getConstrainedPoint(event.position);
        const auto left = juce::jmin(marqueeStart.x, current.x);
        const auto top = juce::jmin(marqueeStart.y, current.y);
        const auto right = juce::jmax(marqueeStart.x, current.x);
        const auto bottom = juce::jmax(marqueeStart.y, current.y);
        marqueeRect = { left, top, right - left, bottom - top };

        if (controllerLaneInteraction && controllerTargetEditsVelocity())
        {
            for (auto& note : previewPattern->notes)
                note.selected = controllerBarRect(note).intersects(marqueeRect);
        }
        else
        {
            for (auto& note : previewPattern->notes)
                note.selected = noteRect(note).intersects(marqueeRect);
        }

        repaint();
        return;
    }

    if (controllerLaneInteraction)
    {
        if (interaction == Interaction::drawShape)
        {
            rebuildControllerPreview(event.position, event.mods.isShiftDown());
            previewDirty = true;
            repaint();
        }
        return;
    }

    if (interaction == Interaction::drag)
    {
        const auto deltaTick = xToGridStartTick(event.position.x) - anchorGridTick;
        const auto deltaRows = pitchRowForY(event.position.y) - anchorPitchRow;

        for (const auto& snapshot : selectedSnapshots)
        {
            auto& note = previewPattern->notes[static_cast<size_t>(snapshot.index)];
            note.startTick = juce::jmax(0, snapshot.startTick + deltaTick);
            note.pitch = transposePitchByVisibleRows(snapshot.pitch, deltaRows);
            note.durationTick = juce::jmax(1, snapshot.durationTick);
        }

        previewDirty = true;
        repaint();
        return;
    }

    if (interaction == Interaction::erase)
    {
        const auto currentPoint = noteGridBounds().getConstrainedPoint(event.position);
        const auto startPoint = drawPathPoints.empty() ? currentPoint : drawPathPoints.back();
        previewDirty = eraseNotesAlongPath(*previewPattern, startPoint, currentPoint) || previewDirty;
        drawPathPoints.clear();
        drawPathPoints.push_back(currentPoint);
        repaint();
        return;
    }

    if (interaction == Interaction::drawShape)
    {
        rebuildShapePreview(event.position, event.mods.isShiftDown());
        previewDirty = true;
        repaint();
        return;
    }

    if (interaction == Interaction::resizeLeft && activeNoteIndex >= 0)
    {
        auto& note = previewPattern->notes[static_cast<size_t>(activeNoteIndex)];
        const auto newStart = juce::jmin(anchorNoteEndTick - gridTick(), xToGridStartTick(event.position.x));
        note.startTick = juce::jmax(0, newStart);
        note.durationTick = juce::jmax(gridTick(), anchorNoteEndTick - note.startTick);
        previewDirty = true;
        repaint();
        return;
    }

    if (interaction == Interaction::create && !selectedSnapshots.empty())
    {
        const auto newEnd = juce::jmax(anchorNoteStartTick + gridTick(), xToGridEndTick(event.position.x));
        for (const auto& snapshot : selectedSnapshots)
        {
            auto& note = previewPattern->notes[static_cast<size_t>(snapshot.index)];
            note.startTick = snapshot.startTick;
            note.durationTick = juce::jmax(gridTick(), newEnd - note.startTick);
        }
        previewDirty = true;
        repaint();
        return;
    }

    if (interaction == Interaction::resizeRight && activeNoteIndex >= 0)
    {
        auto& note = previewPattern->notes[static_cast<size_t>(activeNoteIndex)];
        const auto newEnd = juce::jmax(note.startTick + gridTick(), xToGridEndTick(event.position.x));
        note.durationTick = juce::jmax(gridTick(), newEnd - note.startTick);
        previewDirty = true;
        repaint();
    }
}

void PianoRollComponent::mouseExit(const juce::MouseEvent&)
{
    pitchLanePreviewActive = false;
    stopPreviewNote();
}

void PianoRollComponent::mouseUp(const juce::MouseEvent&)
{
    if (pitchLanePreviewActive)
    {
        pitchLanePreviewActive = false;
        stopPreviewNote();
        return;
    }

    stopPreviewNote();

    if (!previewActive)
        return;

    const auto changed = fingerprint(previewProject) != fingerprint(beforeProject);
    if (previewDirty && changed)
    {
        juce::String actionName = "Edit Notes";
        switch (interaction)
        {
            case Interaction::drag: actionName = "Move Notes"; break;
            case Interaction::resizeLeft:
            case Interaction::resizeRight: actionName = "Resize Notes"; break;
            case Interaction::create: actionName = "Add Note"; break;
            case Interaction::erase: actionName = "Erase Notes"; break;
            case Interaction::moveLeftLocator: actionName = "Move Left Locator"; break;
            case Interaction::moveRightLocator: actionName = "Move Right Locator"; break;
            case Interaction::movePlayhead: actionName = "Move Playhead"; break;
            case Interaction::drawShape:
                if (controllerLaneInteraction)
                {
                    if (controllerTargetEditsVelocity())
                        actionName = pencilDrawMode == PencilDrawMode::brush ? "Brush Velocity" : "Draw Velocity";
                    else
                        actionName = pencilDrawMode == PencilDrawMode::brush ? "Brush Controller" : "Draw Controller";
                }
                else
                {
                    actionName = pencilDrawMode == PencilDrawMode::brush ? "Brush Notes" : "Draw Notes";
                }
                break;
            case Interaction::none: break;
            case Interaction::marqueeSelect: break;
        }
        if (interaction == Interaction::movePlayhead)
            applyPreviewNoUndo();
        else
            commitPreviewUndoable(actionName);
    }
    else if (changed)
    {
        applyPreviewNoUndo();
    }

    previewActive = false;
    previewDirty = false;
    interaction = Interaction::none;
    controllerLaneInteraction = false;
    activeNoteIndex = -1;
    marqueeRect = {};
    selectedSnapshots.clear();
    updateCursorForPosition({ -1.0f, -1.0f });
    refreshFromModel();
}

void PianoRollComponent::startPreviewNote(int pitch, int velocity)
{
    const auto clampedPitch = juce::jlimit(kPitchMin, kPitchMax, pitch);
    const auto clampedVelocity = juce::jlimit(1, 127, velocity);

    if (activePreviewPitch == clampedPitch && activePreviewVelocity == clampedVelocity)
        return;

    stopPreviewNote();

    activePreviewPitch = clampedPitch;
    activePreviewVelocity = clampedVelocity;

    if (notePreviewOn != nullptr)
        notePreviewOn(activePreviewPitch, activePreviewVelocity);
}

void PianoRollComponent::updatePreviewNotePitch(int pitch, int velocity)
{
    const auto clampedPitch = juce::jlimit(kPitchMin, kPitchMax, pitch);
    const auto clampedVelocity = juce::jlimit(1, 127, velocity);

    if (activePreviewPitch == clampedPitch)
        return;

    startPreviewNote(clampedPitch, clampedVelocity);
}

void PianoRollComponent::stopPreviewNote()
{
    if (activePreviewPitch < 0)
        return;

    const auto pitch = activePreviewPitch;
    const auto velocity = activePreviewVelocity;
    activePreviewPitch = -1;
    activePreviewVelocity = 0;

    if (notePreviewOff != nullptr)
        notePreviewOff(pitch, velocity);
    else if (stopPreviewCallback != nullptr)
        stopPreviewCallback();
}

bool PianoRollComponent::keyPressed(const juce::KeyPress& key)
{
    if (key.getModifiers().isCommandDown()
        && (key.getTextCharacter() == 'c' || key.getTextCharacter() == 'C'))
        return copySelected();

    if (key.getModifiers().isCommandDown()
        && (key.getTextCharacter() == 'x' || key.getTextCharacter() == 'X'))
        return cutSelected();

    if (key.getModifiers().isCommandDown()
        && (key.getTextCharacter() == 'v' || key.getTextCharacter() == 'V'))
        return pasteClipboard();

    if (key.getModifiers().isCommandDown()
        && (key.getTextCharacter() == 'a' || key.getTextCharacter() == 'A'))
        return selectAllNotes();

    if (key == juce::KeyPress::deleteKey || key == juce::KeyPress::backspaceKey)
        return deleteSelected();

    if (key.getModifiers().isCommandDown()
        && (key.getTextCharacter() == 'q' || key.getTextCharacter() == 'Q'))
        return quantizeSelected();

    if (key.getModifiers().isCommandDown()
        && (key.getTextCharacter() == 'd' || key.getTextCharacter() == 'D'))
        return duplicateSelectedByGrid();

    const auto sectionIndex = currentSectionIndex();
    auto updatedProject = project();
    auto* pattern = patternForSection(updatedProject, sectionIndex);
    if (pattern == nullptr || !hasSelectedNotes(*pattern))
        return false;

    const auto tickStep = gridTick();
    int deltaTick = 0;
    int deltaRows = 0;

    if (key.getKeyCode() == juce::KeyPress::leftKey)
        deltaTick = -tickStep;
    else if (key.getKeyCode() == juce::KeyPress::rightKey)
        deltaTick = tickStep;
    else if (key.getKeyCode() == juce::KeyPress::upKey)
        deltaRows = -1;
    else if (key.getKeyCode() == juce::KeyPress::downKey)
        deltaRows = 1;
    else
        return false;

    for (auto& note : pattern->notes)
    {
        if (!note.selected)
            continue;
        note.startTick = juce::jmax(0, note.startTick + deltaTick);
        note.pitch = transposePitchByVisibleRows(note.pitch, deltaRows);
    }

    projectWriter(updatedProject, true, "Nudge Notes");
    return true;
}

const ProjectState& PianoRollComponent::project() const
{
    return projectGetter();
}

const ProjectState& PianoRollComponent::displayedProject() const
{
    return previewActive ? previewProject : project();
}

int PianoRollComponent::currentTrackIndex() const
{
    if (const auto* section = currentSection())
        return juce::jmax(0, section->trackIndex);
    return trackIndexGetter != nullptr ? trackIndexGetter() : -1;
}

int PianoRollComponent::currentSectionIndex() const
{
    return selectedSectionIndexGetter != nullptr ? selectedSectionIndexGetter() : -1;
}

const MidiSection* PianoRollComponent::currentSection() const
{
    const auto sectionIndex = currentSectionIndex();
    if (!juce::isPositiveAndBelow(sectionIndex, static_cast<int>(project().midiSections.size())))
        return nullptr;
    return &project().midiSections[static_cast<size_t>(sectionIndex)];
}

const MidiPattern* PianoRollComponent::currentPattern() const
{
    const auto* section = currentSection();
    if (section == nullptr)
        return nullptr;
    return findMidiPattern(project(), section->patternId);
}

const TrackState* PianoRollComponent::currentTrack() const
{
    const auto trackIndex = currentTrackIndex();
    if (!juce::isPositiveAndBelow(trackIndex, static_cast<int>(project().tracks.size())))
        return nullptr;
    return &project().tracks[static_cast<size_t>(trackIndex)];
}

const MidiPattern& PianoRollComponent::drawingPattern() const
{
    if (previewActive)
    {
        if (auto* pattern = patternForSection(previewProject, currentSectionIndex()))
            return *pattern;
    }
    else if (const auto* pattern = patternForSection(project(), currentSectionIndex()))
    {
        return *pattern;
    }

    static const MidiPattern emptyPattern;
    return emptyPattern;
}

MidiPattern* PianoRollComponent::patternForSection(ProjectState& state, int sectionIndex) const
{
    if (!juce::isPositiveAndBelow(sectionIndex, static_cast<int>(state.midiSections.size())))
        return nullptr;
    return findMidiPattern(state, state.midiSections[static_cast<size_t>(sectionIndex)].patternId);
}

const MidiPattern* PianoRollComponent::patternForSection(const ProjectState& state, int sectionIndex) const
{
    if (!juce::isPositiveAndBelow(sectionIndex, static_cast<int>(state.midiSections.size())))
        return nullptr;
    return findMidiPattern(state, state.midiSections[static_cast<size_t>(sectionIndex)].patternId);
}

void PianoRollComponent::updateContentSize()
{
    const auto pitchCount = static_cast<int>(visiblePitches().size());
    const auto width = juce::roundToInt(tickToX(contentTickLength() + kTicksPerBar));
    auto height = 0.0f;
    if (showsNoteEditor())
        height += rulerHeight() + (static_cast<float>(pitchCount) * cellHeight);
    if (showsControllerEditor())
        height += controllerHeaderHeight() + controllerLaneHeight();
    const auto minimumHeight = surfaceMode == SurfaceMode::controllerOnly
        ? juce::roundToInt(controllerHeaderHeight() + controllerLaneHeight())
        : 240;
    setSize(juce::jmax(width, 800), juce::jmax(juce::roundToInt(height), minimumHeight));
}

bool PianoRollComponent::showsNoteEditor() const noexcept
{
    return surfaceMode != SurfaceMode::controllerOnly;
}

bool PianoRollComponent::showsControllerEditor() const noexcept
{
    return surfaceMode != SurfaceMode::notesOnly;
}

float PianoRollComponent::noteRowsHeight() const
{
    return static_cast<float>(visiblePitches().size()) * cellHeight;
}

int PianoRollComponent::contentTickLength() const
{
    int lastTick = juce::jmax(kTicksPerBar, minimumBeats * kTicksPerBeat);
    if (const auto* pattern = currentPattern())
        lastTick = juce::jmax(lastTick, patternLengthTicks(*pattern));
    else
        lastTick = juce::jmax(lastTick, defaultPatternLengthTicks(project()));

    if (const auto* section = currentSection())
    {
        lastTick = juce::jmax(lastTick, juce::jmax(0, project().rightLocatorTick - section->startTick));
        lastTick = juce::jmax(lastTick, juce::jmax(0, project().playheadTick - section->startTick));
    }

    return lastTick;
}

float PianoRollComponent::pitchLaneWidth() const
{
    return 56.0f;
}

float PianoRollComponent::rulerHeight() const
{
    return 22.0f;
}

float PianoRollComponent::controllerHeaderHeight() const
{
    return 28.0f;
}

float PianoRollComponent::controllerLaneHeight() const
{
    return 132.0f;
}

float PianoRollComponent::tickToX(int tick) const
{
    return pitchLaneWidth() + ((static_cast<float>(juce::jmax(0, tick)) / static_cast<float>(kTicksPerBeat)) * cellWidth);
}

float PianoRollComponent::durationToWidth(int durationTick) const
{
    return juce::jmax(1.0f, (static_cast<float>(juce::jmax(1, durationTick)) / static_cast<float>(kTicksPerBeat)) * cellWidth);
}

std::vector<int> PianoRollComponent::visiblePitches() const
{
    return visiblePitchesForProjectScale(displayedProject(), kPitchMin, kPitchMax);
}

int PianoRollComponent::pitchToRow(int pitch) const
{
    const auto pitches = visiblePitches();
    const auto iterator = std::find(pitches.begin(), pitches.end(), juce::jlimit(kPitchMin, kPitchMax, pitch));
    if (iterator == pitches.end())
        return -1;

    return static_cast<int>(std::distance(pitches.begin(), iterator));
}

int PianoRollComponent::nearestVisibleRowForPitch(int pitch) const
{
    const auto pitches = visiblePitches();
    if (pitches.empty())
        return 0;

    auto bestRow = 0;
    auto bestDistance = std::numeric_limits<int>::max();
    const auto clampedPitch = juce::jlimit(kPitchMin, kPitchMax, pitch);
    for (int row = 0; row < static_cast<int>(pitches.size()); ++row)
    {
        const auto distance = std::abs(pitches[static_cast<size_t>(row)] - clampedPitch);
        if (distance < bestDistance)
        {
            bestDistance = distance;
            bestRow = row;
        }
    }

    return bestRow;
}

int PianoRollComponent::rowToPitch(int row) const
{
    const auto pitches = visiblePitches();
    if (pitches.empty())
        return juce::jlimit(kPitchMin, kPitchMax, kPitchMax - row);

    return pitches[static_cast<size_t>(juce::jlimit(0, static_cast<int>(pitches.size()) - 1, row))];
}

int PianoRollComponent::pitchRowForY(float y) const
{
    const auto pitches = visiblePitches();
    const auto pitchCount = juce::jmax(1, static_cast<int>(pitches.size()));
    const auto row = static_cast<int>(std::floor((juce::jmax(rulerHeight(), y) - rulerHeight()) / cellHeight));
    return juce::jlimit(0, pitchCount - 1, row);
}

int PianoRollComponent::pitchForY(float y) const
{
    return rowToPitch(pitchRowForY(y));
}

int PianoRollComponent::transposePitchByVisibleRows(int pitch, int deltaRows) const
{
    const auto pitches = visiblePitches();
    if (pitches.empty())
        return juce::jlimit(kPitchMin, kPitchMax, pitch - deltaRows);

    const auto originalRow = nearestVisibleRowForPitch(pitch);
    const auto targetRow = juce::jlimit(0, static_cast<int>(pitches.size()) - 1, originalRow + deltaRows);
    return pitches[static_cast<size_t>(targetRow)];
}

int PianoRollComponent::gridTick() const
{
    const auto quantizeDiv = juce::jmax(1, project().quantizeDiv);
    auto beats = 4.0 / static_cast<double>(quantizeDiv);
    if (project().quantizeTriplet)
        beats *= (2.0 / 3.0);
    return juce::jmax(1, static_cast<int>(std::round(beats * static_cast<double>(kTicksPerBeat))));
}

int PianoRollComponent::noteLengthTick() const
{
    return gridTick();
}

float PianoRollComponent::gridWidthPixels() const
{
    return durationToWidth(gridTick());
}

juce::Rectangle<float> PianoRollComponent::transportHandleBounds(int absoluteTick) const
{
    const auto* section = currentSection();
    const auto relativeTick = absoluteTick - (section != nullptr ? section->startTick : 0);
    const auto x = tickToX(juce::jmax(0, relativeTick));
    return { x - (kTransportHandleWidth * 0.5f),
             2.0f,
             kTransportHandleWidth,
             kTransportHandleHeight };
}

PianoRollComponent::Interaction PianoRollComponent::hitTestTransportMarker(juce::Point<float> position) const
{
    if (!showsNoteEditor())
        return Interaction::none;

    if (position.y < 0.0f || position.y > (rulerHeight() + 6.0f) || position.x <= pitchLaneWidth())
        return Interaction::none;

    const auto* section = currentSection();
    if (section == nullptr)
        return Interaction::none;

    const auto totalTicks = contentTickLength();
    const auto markerMatches = [this, position, section, totalTicks] (int absoluteTick)
    {
        const auto relativeTick = absoluteTick - section->startTick;
        if (relativeTick < 0 || relativeTick > totalTicks)
            return false;
        const auto handle = transportHandleBounds(absoluteTick).expanded(4.0f, 4.0f);
        const auto x = tickToX(relativeTick);
        return handle.contains(position) || (position.y <= rulerHeight() && std::abs(position.x - x) <= 5.0f);
    };

    const auto& state = displayedProject();
    if (markerMatches(state.leftLocatorTick))
        return Interaction::moveLeftLocator;
    if (markerMatches(state.rightLocatorTick))
        return Interaction::moveRightLocator;
    if (markerMatches(state.playheadTick))
        return Interaction::movePlayhead;
    return Interaction::none;
}

int PianoRollComponent::xToGridStartTick(float x) const
{
    const auto relative = juce::jmax(0.0f, x - pitchLaneWidth());
    const auto cell = static_cast<int>(std::floor(relative / juce::jmax(1.0f, gridWidthPixels())));
    return juce::jmax(0, cell * gridTick());
}

int PianoRollComponent::xToGridEndTick(float x) const
{
    return xToGridStartTick(x) + gridTick();
}

juce::Rectangle<float> PianoRollComponent::noteRect(const MidiNote& note) const
{
    const auto row = pitchToRow(note.pitch);
    if (row < 0)
        return {};

    return { tickToX(note.startTick),
             rulerHeight() + (static_cast<float>(row) * cellHeight),
             durationToWidth(note.durationTick),
             cellHeight };
}

juce::Rectangle<float> PianoRollComponent::controllerBarRect(const MidiNote& note) const
{
    const auto laneBounds = controllerLaneBounds();
    if (laneBounds.isEmpty())
        return {};

    const auto barWidth = juce::jmax(3.0f, durationToWidth(note.durationTick));
    const auto y = yForControllerValue(note.velocity);
    return { tickToX(note.startTick),
             y,
             barWidth,
             juce::jmax(1.0f, laneBounds.getBottom() - y) };
}

juce::Rectangle<float> PianoRollComponent::noteGridBounds() const
{
    if (!showsNoteEditor())
        return {};

    const auto noteHeight = noteRowsHeight();
    return { pitchLaneWidth(),
             rulerHeight(),
             static_cast<float>(getWidth()) - pitchLaneWidth(),
             noteHeight };
}

juce::Rectangle<float> PianoRollComponent::controllerHeaderBounds() const
{
    if (!showsControllerEditor())
        return {};

    const auto notesBounds = noteGridBounds();
    return { pitchLaneWidth(),
             showsNoteEditor() ? notesBounds.getBottom() : 0.0f,
             static_cast<float>(getWidth()) - pitchLaneWidth(),
             controllerHeaderHeight() };
}

juce::Rectangle<float> PianoRollComponent::controllerLaneBounds() const
{
    if (!showsControllerEditor())
        return {};

    const auto headerBounds = controllerHeaderBounds();
    return { pitchLaneWidth(),
             headerBounds.getBottom(),
             static_cast<float>(getWidth()) - pitchLaneWidth(),
             controllerLaneHeight() };
}

bool PianoRollComponent::isControllerLanePosition(juce::Point<float> position) const
{
    return controllerLaneBounds().contains(position);
}

void PianoRollComponent::refreshControllerTargetChoices()
{
    juce::StringArray targets;
    targets.add(velocityControllerTarget());
    targets.addArray(defaultMidiControllerTargets());

    if (const auto* track = currentTrack())
        targets.addArray(availableAutomationTargets(*track));

    targets.removeDuplicates(true);
    if (!targets.contains(selectedControllerTarget, true))
        selectedControllerTarget = velocityControllerTarget();

    const auto selectedTargetIndex = [&targets, this] () -> int
    {
        for (int index = 0; index < targets.size(); ++index)
        {
            if (targets[index].equalsIgnoreCase(selectedControllerTarget))
                return index;
        }
        return 0;
    }();

    if (controllerTargetOptions.size() == static_cast<size_t>(targets.size()))
    {
        bool sameTargets = true;
        for (int index = 0; index < targets.size(); ++index)
        {
            if (!controllerTargetOptions[static_cast<size_t>(index)].equalsIgnoreCase(targets[index]))
            {
                sameTargets = false;
                break;
            }
        }

        if (sameTargets)
        {
            controllerTargetBox.setSelectedItemIndex(selectedTargetIndex, juce::dontSendNotification);
            return;
        }
    }

    controllerTargetOptions.clear();
    controllerTargetOptions.reserve(static_cast<size_t>(targets.size()));
    for (const auto& target : targets)
        controllerTargetOptions.push_back(target);
    controllerTargetBox.clear(juce::dontSendNotification);
    for (int index = 0; index < targets.size(); ++index)
        controllerTargetBox.addItem(controllerTargetDisplayText(targets[index]), index + 1);
    controllerTargetBox.setSelectedItemIndex(selectedTargetIndex, juce::dontSendNotification);
}

juce::String PianoRollComponent::currentControllerTarget() const
{
    return selectedControllerTarget.trim().isNotEmpty() ? selectedControllerTarget : velocityControllerTarget();
}

bool PianoRollComponent::controllerTargetEditsVelocity() const
{
    return currentControllerTarget().equalsIgnoreCase(velocityControllerTarget());
}

bool PianoRollComponent::controllerTargetEditsPatternLane() const
{
    return controllerTargetIsMidiCc(currentControllerTarget());
}

int PianoRollComponent::controllerStoredTickFromLocalTick(int localTick) const
{
    if (controllerTargetEditsPatternLane() || controllerTargetEditsVelocity())
        return juce::jmax(0, localTick);

    const auto baseTick = currentSection() != nullptr ? currentSection()->startTick : 0;
    return juce::jmax(0, baseTick + localTick);
}

int PianoRollComponent::controllerLocalTickFromStoredTick(int storedTick) const
{
    if (controllerTargetEditsPatternLane() || controllerTargetEditsVelocity())
        return juce::jmax(0, storedTick);

    const auto baseTick = currentSection() != nullptr ? currentSection()->startTick : 0;
    return juce::jmax(0, storedTick - baseTick);
}

std::pair<double, double> PianoRollComponent::controllerTargetBounds() const
{
    if (controllerTargetEditsVelocity() || controllerTargetEditsPatternLane())
        return { 0.0, 127.0 };

    const auto& state = displayedProject();
    const auto trackIndex = currentTrackIndex();
    if (juce::isPositiveAndBelow(trackIndex, static_cast<int>(state.tracks.size())))
        return automationTargetBounds(state.tracks[static_cast<size_t>(trackIndex)], currentControllerTarget());

    return { 0.0, 1.0 };
}

double PianoRollComponent::controllerDefaultValue() const
{
    if (controllerTargetEditsVelocity())
        return 100.0;
    if (controllerTargetEditsPatternLane())
        return 0.0;
    const auto& state = displayedProject();
    const auto trackIndex = currentTrackIndex();
    if (juce::isPositiveAndBelow(trackIndex, static_cast<int>(state.tracks.size())))
        return automationTargetDefaultValue(state.tracks[static_cast<size_t>(trackIndex)], currentControllerTarget());
    return 0.0;
}

double PianoRollComponent::controllerValueForY(float y) const
{
    const auto laneBounds = controllerLaneBounds();
    if (laneBounds.isEmpty())
        return controllerDefaultValue();

    const auto bounds = controllerTargetBounds();
    const auto clampedY = juce::jlimit(laneBounds.getY(), laneBounds.getBottom(), y);
    const auto progress = 1.0 - juce::jlimit(0.0, 1.0, static_cast<double>((clampedY - laneBounds.getY()) / juce::jmax(1.0f, laneBounds.getHeight())));
    return juce::jmap(progress, bounds.first, bounds.second);
}

float PianoRollComponent::yForControllerValue(double value) const
{
    const auto laneBounds = controllerLaneBounds();
    const auto bounds = controllerTargetBounds();
    const auto span = juce::jmax(0.0001, bounds.second - bounds.first);
    const auto progress = juce::jlimit(0.0, 1.0, (value - bounds.first) / span);
    return laneBounds.getBottom() - static_cast<float>(progress) * laneBounds.getHeight();
}

const AutomationLane* PianoRollComponent::currentControllerLane() const
{
    const auto target = currentControllerTarget();
    if (controllerTargetEditsVelocity())
        return nullptr;

    if (controllerTargetEditsPatternLane())
    {
        const auto* pattern = patternForSection(displayedProject(), currentSectionIndex());
        if (pattern == nullptr)
            return nullptr;

        for (const auto& lane : pattern->controllerLanes)
        {
            if (lane.target.equalsIgnoreCase(target))
                return &lane;
        }
        return nullptr;
    }

    const auto& state = displayedProject();
    const auto trackIndex = currentTrackIndex();
    if (!juce::isPositiveAndBelow(trackIndex, static_cast<int>(state.tracks.size())))
        return nullptr;

    const auto& track = state.tracks[static_cast<size_t>(trackIndex)];
    for (const auto& lane : track.automationLanes)
    {
        if (lane.target.equalsIgnoreCase(target))
            return &lane;
    }

    return nullptr;
}

AutomationLane* PianoRollComponent::editableControllerLane(ProjectState& state, bool createIfMissing)
{
    const auto target = currentControllerTarget();
    if (controllerTargetEditsVelocity())
        return nullptr;

    auto* lanes = [&]() -> std::vector<AutomationLane>*
    {
        if (controllerTargetEditsPatternLane())
        {
            auto* pattern = patternForSection(state, currentSectionIndex());
            return pattern != nullptr ? &pattern->controllerLanes : nullptr;
        }

        const auto trackIndex = currentTrackIndex();
        if (!juce::isPositiveAndBelow(trackIndex, static_cast<int>(state.tracks.size())))
            return nullptr;
        return &state.tracks[static_cast<size_t>(trackIndex)].automationLanes;
    }();

    if (lanes == nullptr)
        return nullptr;

    for (auto& lane : *lanes)
    {
        if (lane.target.equalsIgnoreCase(target))
            return &lane;
    }

    if (!createIfMissing)
        return nullptr;

    AutomationLane lane;
    lane.target = target;
    lane.enabled = true;
    lanes->push_back(std::move(lane));
    return &lanes->back();
}

std::vector<AutomationPoint> PianoRollComponent::displayedControllerPoints() const
{
    std::vector<AutomationPoint> points;
    const auto* lane = currentControllerLane();
    if (lane == nullptr)
        return points;

    const auto localLength = contentTickLength();
    for (const auto& point : lane->points)
    {
        AutomationPoint local = point;
        local.tick = controllerLocalTickFromStoredTick(point.tick);
        if (local.tick < 0 || local.tick > localLength)
            continue;
        points.push_back(local);
    }

    std::sort(points.begin(),
              points.end(),
              [] (const AutomationPoint& lhs, const AutomationPoint& rhs)
              {
                  return lhs.tick < rhs.tick;
              });
    return points;
}

juce::String PianoRollComponent::controllerTargetDisplayText(const juce::String& target) const
{
    if (target.equalsIgnoreCase(velocityControllerTarget()))
        return "Velocity";
    if (controllerTargetIsMidiCc(target))
        return midiControllerTargetLabel(target);
    const auto& state = displayedProject();
    const auto trackIndex = currentTrackIndex();
    if (juce::isPositiveAndBelow(trackIndex, static_cast<int>(state.tracks.size())))
        return automationTargetLabel(state.tracks[static_cast<size_t>(trackIndex)], target);
    return target;
}

void PianoRollComponent::clearSelection(MidiPattern& pattern) const
{
    for (auto& note : pattern.notes)
        note.selected = false;
}

void PianoRollComponent::selectSingleNote(MidiPattern& pattern, int noteIndex) const
{
    clearSelection(pattern);
    if (juce::isPositiveAndBelow(noteIndex, static_cast<int>(pattern.notes.size())))
        pattern.notes[static_cast<size_t>(noteIndex)].selected = true;
}

bool PianoRollComponent::hasSelectedNotes(const MidiPattern& pattern) const
{
    return std::any_of(pattern.notes.begin(), pattern.notes.end(), [] (const MidiNote& note) { return note.selected; });
}

bool PianoRollComponent::eraseNotesAtPosition(MidiPattern& pattern, juce::Point<float> position) const
{
    const auto bounds = noteGridBounds();
    if (bounds.isEmpty() || pattern.notes.empty())
        return false;

    const auto constrained = bounds.getConstrainedPoint(position);
    const auto brushWidth = juce::jmax(8.0f, gridWidthPixels() * 0.9f);
    const auto brushHeight = juce::jmax(8.0f, cellHeight * 0.9f);
    const juce::Rectangle<float> brushBounds(constrained.x - (brushWidth * 0.5f),
                                             constrained.y - (brushHeight * 0.5f),
                                             brushWidth,
                                             brushHeight);

    const auto originalSize = pattern.notes.size();
    pattern.notes.erase(std::remove_if(pattern.notes.begin(),
                                       pattern.notes.end(),
                                       [this, brushBounds] (const MidiNote& note)
                                       {
                                           return noteRect(note).intersects(brushBounds);
                                       }),
                        pattern.notes.end());
    return pattern.notes.size() != originalSize;
}

bool PianoRollComponent::eraseNotesAlongPath(MidiPattern& pattern,
                                             juce::Point<float> startPosition,
                                             juce::Point<float> endPosition) const
{
    const auto bounds = noteGridBounds();
    if (bounds.isEmpty())
        return false;

    const auto start = bounds.getConstrainedPoint(startPosition);
    const auto end = bounds.getConstrainedPoint(endPosition);
    const auto tickDistance = std::abs(xToGridStartTick(end.x) - xToGridStartTick(start.x));
    const auto rowDistance = std::abs(pitchRowForY(end.y) - pitchRowForY(start.y));
    const auto sampleCount = juce::jmax(1, juce::jmax(tickDistance / juce::jmax(1, gridTick()), rowDistance));

    bool changed = false;
    for (int sample = 0; sample <= sampleCount; ++sample)
    {
        const auto progress = static_cast<float>(sample) / static_cast<float>(sampleCount);
        const juce::Point<float> samplePoint(juce::jmap(progress, start.x, end.x),
                                             juce::jmap(progress, start.y, end.y));
        changed = eraseNotesAtPosition(pattern, samplePoint) || changed;
    }

    return changed;
}

bool PianoRollComponent::glueSelectedOrClickedNotes(int noteIndex)
{
    const auto sectionIndex = currentSectionIndex();
    auto updatedProject = project();
    auto* pattern = patternForSection(updatedProject, sectionIndex);
    if (pattern == nullptr || !juce::isPositiveAndBelow(noteIndex, static_cast<int>(pattern->notes.size())))
        return false;

    std::vector<int> candidateIndices;
    const auto clickedNote = pattern->notes[static_cast<size_t>(noteIndex)];
    const bool clickedWasSelected = clickedNote.selected && hasSelectedNotes(*pattern);

    if (clickedWasSelected)
    {
        for (int index = 0; index < static_cast<int>(pattern->notes.size()); ++index)
        {
            if (pattern->notes[static_cast<size_t>(index)].selected)
                candidateIndices.push_back(index);
        }
    }
    else
    {
        std::vector<bool> included(pattern->notes.size(), false);
        candidateIndices.push_back(noteIndex);
        included[static_cast<size_t>(noteIndex)] = true;

        bool expanded = true;
        while (expanded)
        {
            expanded = false;
            for (int index = 0; index < static_cast<int>(pattern->notes.size()); ++index)
            {
                if (included[static_cast<size_t>(index)])
                    continue;

                const auto& candidate = pattern->notes[static_cast<size_t>(index)];
                if (candidate.pitch != clickedNote.pitch)
                    continue;

                for (const auto connectedIndex : candidateIndices)
                {
                    if (noteRangesTouchOrOverlap(candidate, pattern->notes[static_cast<size_t>(connectedIndex)]))
                    {
                        included[static_cast<size_t>(index)] = true;
                        candidateIndices.push_back(index);
                        expanded = true;
                        break;
                    }
                }
            }
        }
    }

    if (candidateIndices.size() < 2)
        return false;

    std::array<std::vector<int>, 128> groupsByPitch;
    for (const auto index : candidateIndices)
    {
        const auto pitch = juce::jlimit(0, 127, pattern->notes[static_cast<size_t>(index)].pitch);
        groupsByPitch[static_cast<size_t>(pitch)].push_back(index);
    }

    std::vector<bool> shouldRemove(pattern->notes.size(), false);
    std::vector<MidiNote> mergedNotes;
    int mergeCount = 0;

    for (size_t pitchIndex = 0; pitchIndex < groupsByPitch.size(); ++pitchIndex)
    {
        const auto& group = groupsByPitch[pitchIndex];
        if (group.size() < 2)
            continue;

        MidiNote merged;
        merged.pitch = static_cast<int>(pitchIndex);
        merged.velocity = 1;
        merged.selected = true;
        int minStartTick = std::numeric_limits<int>::max();
        int maxEndTick = 0;

        for (const auto index : group)
        {
            const auto& note = pattern->notes[static_cast<size_t>(index)];
            minStartTick = juce::jmin(minStartTick, note.startTick);
            maxEndTick = juce::jmax(maxEndTick, note.startTick + note.durationTick);
            merged.velocity = juce::jmax(merged.velocity, note.velocity);
            shouldRemove[static_cast<size_t>(index)] = true;
        }

        merged.startTick = juce::jmax(0, minStartTick);
        merged.durationTick = juce::jmax(1, maxEndTick - merged.startTick);
        mergedNotes.push_back(merged);
        ++mergeCount;
    }

    if (mergeCount == 0)
        return false;

    std::vector<MidiNote> replacementNotes;
    replacementNotes.reserve(pattern->notes.size() - static_cast<size_t>(mergeCount) + mergedNotes.size());
    for (size_t index = 0; index < pattern->notes.size(); ++index)
    {
        if (!shouldRemove[index])
            replacementNotes.push_back(pattern->notes[index]);
    }

    replacementNotes.insert(replacementNotes.end(), mergedNotes.begin(), mergedNotes.end());
    sortMidiNotes(replacementNotes);
    pattern->notes = std::move(replacementNotes);
    projectWriter(updatedProject, true, "Glue Notes");
    return true;
}

std::vector<int> PianoRollComponent::insertIntervals() const
{
    switch (insertMode)
    {
        case PianoRollInsertMode::singleNote: return { 0 };
        case PianoRollInsertMode::majorTriad: return { 0, 4, 7 };
        case PianoRollInsertMode::minorTriad: return { 0, 3, 7 };
        case PianoRollInsertMode::majorFifth: return { 0, 7 };
        case PianoRollInsertMode::minorFifth: return { 0, 6 };
        case PianoRollInsertMode::majorSeventh: return { 0, 4, 7, 11 };
        case PianoRollInsertMode::minorSeventh: return { 0, 3, 7, 10 };
        case PianoRollInsertMode::dominantSeventh: return { 0, 4, 7, 10 };
        case PianoRollInsertMode::suspended2: return { 0, 2, 7 };
        case PianoRollInsertMode::suspended4: return { 0, 5, 7 };
        case PianoRollInsertMode::resolvedMajor: return { 0, 4, 7, 12 };
        case PianoRollInsertMode::resolvedMinor: return { 0, 3, 7, 12 };
    }

    return { 0 };
}

void PianoRollComponent::insertNotesForCurrentMode(MidiPattern& pattern,
                                                   int startTick,
                                                   int basePitch,
                                                   int durationTick,
                                                   int velocity,
                                                   bool select,
                                                   std::vector<int>* insertedIndices) const
{
    const auto intervals = insertIntervals();
    for (const auto interval : intervals)
    {
        MidiNote note;
        note.startTick = juce::jmax(0, startTick);
        note.durationTick = juce::jmax(gridTick(), durationTick);
        note.pitch = juce::jlimit(kPitchMin, kPitchMax, basePitch + interval);
        note.velocity = juce::jlimit(1, 127, velocity);
        note.selected = select;
        pattern.notes.push_back(note);
        if (insertedIndices != nullptr)
            insertedIndices->push_back(static_cast<int>(pattern.notes.size()) - 1);
    }
}

int PianoRollComponent::rowForProgress(float progress, int targetRow) const
{
    progress = juce::jlimit(0.0f, 1.0f, progress);

    switch (pencilDrawMode)
    {
        case PencilDrawMode::step:
        case PencilDrawMode::brush:
        case PencilDrawMode::line:
        case PencilDrawMode::box:
            return juce::roundToInt(juce::jmap(progress,
                                               static_cast<float>(anchorPitchRow),
                                               static_cast<float>(targetRow)));

        case PencilDrawMode::sine:
        {
            const auto angle = progress * juce::MathConstants<float>::twoPi * static_cast<float>(shapeFrequencyCycles)
                - juce::MathConstants<float>::halfPi;
            return juce::roundToInt(static_cast<float>(anchorPitchRow) + std::sin(angle) * static_cast<float>(shapeAmplitudeRows));
        }

        case PencilDrawMode::square:
        {
            const auto phase = std::fmod(progress * static_cast<float>(shapeFrequencyCycles), 1.0f);
            const auto offset = phase < 0.5f ? -shapeAmplitudeRows : shapeAmplitudeRows;
            return anchorPitchRow + offset;
        }

        case PencilDrawMode::saw:
        {
            const auto phase = std::fmod(progress * static_cast<float>(shapeFrequencyCycles), 1.0f);
            return juce::roundToInt(static_cast<float>(anchorPitchRow - shapeAmplitudeRows)
                                    + (phase * static_cast<float>(shapeAmplitudeRows * 2)));
        }

        case PencilDrawMode::triangle:
        {
            const auto phase = std::fmod(progress * static_cast<float>(shapeFrequencyCycles), 1.0f);
            const auto triangle = phase < 0.5f ? (phase * 2.0f) : (2.0f - (phase * 2.0f));
            return juce::roundToInt(static_cast<float>(anchorPitchRow - shapeAmplitudeRows)
                                    + (triangle * static_cast<float>(shapeAmplitudeRows * 2)));
        }

        case PencilDrawMode::circle:
        {
            const auto angle = progress * juce::MathConstants<float>::twoPi * static_cast<float>(shapeFrequencyCycles);
            const auto circleProgress = 0.5f - (0.5f * std::cos(angle));
            const auto vertical = std::sin(angle);
            juce::ignoreUnused(circleProgress);
            return juce::roundToInt(static_cast<float>(anchorPitchRow) + vertical * static_cast<float>(shapeAmplitudeRows));
        }
    }

    return targetRow;
}

double PianoRollComponent::valueForProgress(float progress, double targetValue) const
{
    progress = juce::jlimit(0.0f, 1.0f, progress);
    const auto bounds = controllerTargetBounds();
    const auto lowValue = juce::jmin(anchorControllerValue, targetValue);
    const auto highValue = juce::jmax(anchorControllerValue, targetValue);

    auto clampValueToBounds = [&bounds] (double value)
    {
        return juce::jlimit(bounds.first, bounds.second, value);
    };

    switch (pencilDrawMode)
    {
        case PencilDrawMode::step:
        case PencilDrawMode::box:
            return clampValueToBounds(targetValue);

        case PencilDrawMode::brush:
        case PencilDrawMode::line:
            return clampValueToBounds(juce::jmap(static_cast<double>(progress), anchorControllerValue, targetValue));

        case PencilDrawMode::sine:
        {
            const auto angle = progress * juce::MathConstants<float>::twoPi * static_cast<float>(shapeFrequencyCycles)
                - juce::MathConstants<float>::halfPi;
            return clampValueToBounds(anchorControllerValue + std::sin(angle) * (targetValue - anchorControllerValue));
        }

        case PencilDrawMode::square:
        {
            const auto phase = std::fmod(progress * static_cast<float>(shapeFrequencyCycles), 1.0f);
            return clampValueToBounds(phase < 0.5f ? lowValue : highValue);
        }

        case PencilDrawMode::saw:
        {
            const auto phase = std::fmod(progress * static_cast<float>(shapeFrequencyCycles), 1.0f);
            return clampValueToBounds(lowValue + (phase * (highValue - lowValue)));
        }

        case PencilDrawMode::triangle:
        {
            const auto phase = std::fmod(progress * static_cast<float>(shapeFrequencyCycles), 1.0f);
            const auto triangle = phase < 0.5f ? (phase * 2.0f) : (2.0f - (phase * 2.0f));
            return clampValueToBounds(lowValue + (triangle * (highValue - lowValue)));
        }

        case PencilDrawMode::circle:
        {
            const auto angle = progress * juce::MathConstants<float>::twoPi * static_cast<float>(shapeFrequencyCycles);
            const auto vertical = std::sin(angle);
            return clampValueToBounds(anchorControllerValue + vertical * (targetValue - anchorControllerValue));
        }
    }

    return clampValueToBounds(targetValue);
}

void PianoRollComponent::rebuildShapePreview(juce::Point<float> currentPosition, bool adjustFrequency)
{
    previewProject = beforeProject;
    auto* pattern = patternForSection(previewProject, currentSectionIndex());
    if (pattern == nullptr)
        return;

    clearSelection(*pattern);

    const auto currentTick = xToGridStartTick(currentPosition.x);
    const auto currentRow = pitchRowForY(currentPosition.y);
    const auto minTick = juce::jmin(anchorGridTick, currentTick);
    const auto maxTick = juce::jmax(anchorGridTick, currentTick);
    const auto leftToRight = currentTick >= anchorGridTick;
    const auto lineStartRow = leftToRight ? anchorPitchRow : currentRow;
    const auto lineEndRow = leftToRight ? currentRow : anchorPitchRow;

    if (adjustFrequency
        && pencilDrawMode != PencilDrawMode::line
        && pencilDrawMode != PencilDrawMode::box
        && pencilDrawMode != PencilDrawMode::brush)
        shapeFrequencyCycles = juce::jlimit(1, 16, std::abs(currentRow - anchorPitchRow) + 1);
    else
        shapeAmplitudeRows = juce::jlimit(1, 36, std::abs(currentRow - anchorPitchRow));

    const auto step = gridTick();
    const auto rootVelocity = 100;
    std::set<juce::int64> insertedKeys;
    std::vector<std::pair<int, int>> placements;

    const auto appendPlacement = [&placements, &insertedKeys] (int tick, int pitch)
    {
        const auto key = (static_cast<juce::int64>(tick) << 8) | static_cast<juce::int64>(juce::jlimit(0, 127, pitch));
        if (insertedKeys.insert(key).second)
            placements.emplace_back(juce::jmax(0, tick), juce::jlimit(0, 127, pitch));
    };

    if (pencilDrawMode == PencilDrawMode::brush)
    {
        if (drawPathPoints.empty())
            drawPathPoints.push_back({ tickToX(anchorGridTick), rulerHeight() + (static_cast<float>(anchorPitchRow) * cellHeight) });
        drawPathPoints.push_back(currentPosition);

        for (size_t pointIndex = 1; pointIndex < drawPathPoints.size(); ++pointIndex)
        {
            const auto startPoint = drawPathPoints[pointIndex - 1];
            const auto endPoint = drawPathPoints[pointIndex];
            const auto sampleCount = juce::jmax(1,
                                                juce::jmax(std::abs(xToGridStartTick(endPoint.x) - xToGridStartTick(startPoint.x)) / juce::jmax(1, step),
                                                           std::abs(pitchRowForY(endPoint.y) - pitchRowForY(startPoint.y))));

            for (int sample = 0; sample <= sampleCount; ++sample)
            {
                const auto progress = static_cast<float>(sample) / static_cast<float>(sampleCount);
                const auto x = juce::jmap(progress, startPoint.x, endPoint.x);
                const auto y = juce::jmap(progress, startPoint.y, endPoint.y);
                appendPlacement(xToGridStartTick(x), pitchForY(y));
            }
        }
    }
    else if (pencilDrawMode == PencilDrawMode::box)
    {
        const auto rowMin = juce::jmin(anchorPitchRow, currentRow);
        const auto rowMax = juce::jmax(anchorPitchRow, currentRow);
        for (int tick = minTick; tick <= maxTick; tick += step)
        {
            for (int row = rowMin; row <= rowMax; ++row)
                appendPlacement(tick, rowToPitch(row));
        }
    }
    else
    {
        const auto stepCount = juce::jmax(1, ((maxTick - minTick) / juce::jmax(1, step)) + 1);
        for (int stepIndex = 0; stepIndex < stepCount; ++stepIndex)
        {
            const auto progress = stepCount <= 1 ? 0.0f : static_cast<float>(stepIndex) / static_cast<float>(stepCount - 1);
            const auto tick = minTick + (stepIndex * step);
            const auto row = pencilDrawMode == PencilDrawMode::line
                ? juce::roundToInt(juce::jmap(progress,
                                              static_cast<float>(lineStartRow),
                                              static_cast<float>(lineEndRow)))
                : rowForProgress(progress, currentRow);
            appendPlacement(tick, rowToPitch(row));
        }
    }

    std::vector<int> insertedIndices;
    for (const auto& placement : placements)
        insertNotesForCurrentMode(*pattern, placement.first, placement.second, noteLengthTick(), rootVelocity, true, &insertedIndices);

    sortMidiNotes(pattern->notes);
    selectedSnapshots.clear();
    for (int index = 0; index < static_cast<int>(pattern->notes.size()); ++index)
    {
        const auto& note = pattern->notes[static_cast<size_t>(index)];
        if (!note.selected)
            continue;
        selectedSnapshots.push_back({ index, note.startTick, note.pitch, note.durationTick });
    }
}

void PianoRollComponent::rebuildControllerPreview(juce::Point<float> currentPosition, bool adjustFrequency)
{
    previewProject = beforeProject;
    auto* pattern = patternForSection(previewProject, currentSectionIndex());
    if (pattern == nullptr)
        return;

    const auto currentTick = xToGridStartTick(currentPosition.x);
    const auto targetValue = controllerValueForY(currentPosition.y);
    const auto minTick = juce::jmin(anchorGridTick, currentTick);
    const auto maxTick = juce::jmax(anchorGridTick, currentTick);
    const auto step = gridTick();

    if (adjustFrequency
        && pencilDrawMode != PencilDrawMode::line
        && pencilDrawMode != PencilDrawMode::box
        && pencilDrawMode != PencilDrawMode::brush)
    {
        const auto bounds = controllerTargetBounds();
        const auto span = juce::jmax(1.0, bounds.second - bounds.first);
        shapeFrequencyCycles = juce::jlimit(1,
                                            16,
                                            1 + juce::roundToInt((std::abs(targetValue - anchorControllerValue) / span) * 12.0));
    }

    std::vector<AutomationPoint> sampledPoints;
    std::set<int> usedTicks;
    const auto addPoint = [&sampledPoints, &usedTicks] (int tick, double value)
    {
        const auto clampedTick = juce::jmax(0, tick);
        if (!usedTicks.insert(clampedTick).second)
        {
            for (auto& point : sampledPoints)
            {
                if (point.tick == clampedTick)
                {
                    point.value = value;
                    return;
                }
            }
        }

        AutomationPoint point;
        point.tick = clampedTick;
        point.value = value;
        sampledPoints.push_back(point);
    };

    if (pencilDrawMode == PencilDrawMode::brush)
    {
        if (drawPathPoints.empty())
            drawPathPoints.push_back({ tickToX(anchorGridTick), yForControllerValue(anchorControllerValue) });
        drawPathPoints.push_back(currentPosition);

        for (size_t pointIndex = 1; pointIndex < drawPathPoints.size(); ++pointIndex)
        {
            const auto startPoint = drawPathPoints[pointIndex - 1];
            const auto endPoint = drawPathPoints[pointIndex];
            const auto sampleCount = juce::jmax(1,
                                                std::abs(xToGridStartTick(endPoint.x) - xToGridStartTick(startPoint.x)) / juce::jmax(1, step));
            for (int sample = 0; sample <= sampleCount; ++sample)
            {
                const auto progress = static_cast<float>(sample) / static_cast<float>(sampleCount);
                const auto x = juce::jmap(progress, startPoint.x, endPoint.x);
                const auto y = juce::jmap(progress, startPoint.y, endPoint.y);
                addPoint(xToGridStartTick(x), controllerValueForY(y));
            }
        }
    }
    else if (pencilDrawMode == PencilDrawMode::box || pencilDrawMode == PencilDrawMode::step)
    {
        addPoint(minTick, targetValue);
        addPoint(maxTick, targetValue);
    }
    else
    {
        const auto stepCount = juce::jmax(1, ((maxTick - minTick) / juce::jmax(1, step)) + 1);
        for (int stepIndex = 0; stepIndex < stepCount; ++stepIndex)
        {
            const auto progress = stepCount <= 1 ? 0.0f : static_cast<float>(stepIndex) / static_cast<float>(stepCount - 1);
            const auto tick = minTick + (stepIndex * step);
            addPoint(tick, valueForProgress(progress, targetValue));
        }
    }

    std::sort(sampledPoints.begin(),
              sampledPoints.end(),
              [] (const AutomationPoint& lhs, const AutomationPoint& rhs)
              {
                  return lhs.tick < rhs.tick;
              });

    if (controllerTargetEditsVelocity())
    {
        for (auto& note : pattern->notes)
        {
            if (note.startTick < minTick || note.startTick > maxTick)
                continue;

            double nearestValue = controllerDefaultValue();
            auto bestDistance = std::numeric_limits<int>::max();
            for (const auto& point : sampledPoints)
            {
                const auto distance = std::abs(point.tick - note.startTick);
                if (distance < bestDistance)
                {
                    bestDistance = distance;
                    nearestValue = point.value;
                }
            }

            note.velocity = juce::jlimit(1, 127, juce::roundToInt(nearestValue));
        }
        sortMidiNotes(pattern->notes);
        return;
    }

    auto* lane = editableControllerLane(previewProject, true);
    if (lane == nullptr)
        return;

    const auto storedMinTick = controllerStoredTickFromLocalTick(minTick);
    const auto storedMaxTick = controllerStoredTickFromLocalTick(maxTick);
    lane->points.erase(std::remove_if(lane->points.begin(),
                                      lane->points.end(),
                                      [storedMinTick, storedMaxTick] (const AutomationPoint& point)
                                      {
                                          return point.tick >= storedMinTick && point.tick <= storedMaxTick;
                                      }),
                       lane->points.end());

    for (auto point : sampledPoints)
    {
        point.tick = controllerStoredTickFromLocalTick(point.tick);
        lane->points.push_back(point);
    }

    if (controllerTargetEditsPatternLane())
    {
        sanitisePatternControllerLanes(*pattern);
    }
    else
    {
        const auto trackIndex = currentTrackIndex();
        if (juce::isPositiveAndBelow(trackIndex, static_cast<int>(previewProject.tracks.size())))
            sanitiseAutomationLanes(previewProject.tracks[static_cast<size_t>(trackIndex)]);
    }
}

int PianoRollComponent::hitTestNote(juce::Point<float> position, juce::String& edgeOut) const
{
    const auto resizeMargin = 5.0f;
    for (int index = static_cast<int>(drawingPattern().notes.size()) - 1; index >= 0; --index)
    {
        const auto rect = noteRect(drawingPattern().notes[static_cast<size_t>(index)]);
        if (!rect.contains(position))
            continue;

        if (position.x <= rect.getX() + resizeMargin)
            edgeOut = "left";
        else if (position.x >= rect.getRight() - resizeMargin)
            edgeOut = "right";
        else
            edgeOut = "body";
        return index;
    }

    edgeOut.clear();
    return -1;
}

void PianoRollComponent::updateCursorForPosition(juce::Point<float> position)
{
    if (hitTestTransportMarker(position) != Interaction::none)
    {
        setMouseCursor(juce::MouseCursor::PointingHandCursor);
        return;
    }

    if (isControllerLanePosition(position))
    {
        setMouseCursor(toolMode == EditorToolMode::pencil ? juce::MouseCursor::CrosshairCursor
                                                          : juce::MouseCursor::NormalCursor);
        return;
    }

    if (toolMode == EditorToolMode::eraser)
    {
        setMouseCursor(juce::MouseCursor::CrosshairCursor);
        return;
    }

    juce::String edge;
    const auto hit = hitTestNote(position, edge);
    if (hit < 0)
    {
        setMouseCursor(toolMode == EditorToolMode::pencil ? juce::MouseCursor::CrosshairCursor
                                                          : juce::MouseCursor::NormalCursor);
        return;
    }

    if (toolMode == EditorToolMode::glue)
    {
        setMouseCursor(juce::MouseCursor::PointingHandCursor);
        return;
    }

    if (edge == "left" || edge == "right")
        setMouseCursor(juce::MouseCursor::LeftRightResizeCursor);
    else
        setMouseCursor(juce::MouseCursor::DraggingHandCursor);
}

void PianoRollComponent::applyPreviewNoUndo()
{
    projectWriter(previewProject, false, {});
}

void PianoRollComponent::commitPreviewUndoable(const juce::String& actionName)
{
    projectWriter(previewProject, true, actionName);
}

} // namespace aims
