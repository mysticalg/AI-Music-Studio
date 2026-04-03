#include "ArrangementOverviewComponent.h"
#include "UiStyle.h"

#include <algorithm>
#include <cmath>
#include <limits>

namespace aims
{
namespace
{
const juce::Colour kBackground = juce::Colour::fromRGB(14, 16, 21);
const juce::Colour kLaneEven = juce::Colour::fromRGB(26, 30, 37);
const juce::Colour kLaneOdd = juce::Colour::fromRGB(21, 25, 31);
const juce::Colour kHeader = juce::Colour::fromRGB(18, 21, 27);
const juce::Colour kGridMajor = juce::Colour::fromRGB(63, 72, 88);
const juce::Colour kGridMinor = juce::Colour::fromRGBA(90, 100, 118, 86);
const juce::Colour kClip = juce::Colour::fromRGB(126, 213, 140);
const juce::Colour kPlayhead = juce::Colour::fromRGB(255, 102, 102);
const juce::Colour kLeftLocator = juce::Colour::fromRGB(120, 212, 255);
const juce::Colour kRightLocator = juce::Colour::fromRGB(255, 209, 102);
const char* kClipboardType = "aims.native.pattern_clip";
constexpr float kTransportHandleWidth = 16.0f;
constexpr float kTransportHandleHeight = 10.0f;

juce::String serialiseClipClipboard(const MidiSection& section, const MidiPattern& pattern)
{
    auto root = juce::var(new juce::DynamicObject());
    auto patternObject = juce::var(new juce::DynamicObject());
    juce::Array<juce::var> notes;
    juce::Array<juce::var> controllerLanes;

    for (const auto& note : pattern.notes)
    {
        auto noteObject = juce::var(new juce::DynamicObject());
        if (auto* object = noteObject.getDynamicObject())
        {
            object->setProperty("start_tick", note.startTick);
            object->setProperty("duration_tick", note.durationTick);
            object->setProperty("pitch", note.pitch);
            object->setProperty("velocity", note.velocity);
        }
        notes.add(noteObject);
    }

    for (const auto& lane : pattern.controllerLanes)
    {
        auto laneObject = juce::var(new juce::DynamicObject());
        juce::Array<juce::var> points;
        for (const auto& point : lane.points)
        {
            auto pointObject = juce::var(new juce::DynamicObject());
            if (auto* object = pointObject.getDynamicObject())
            {
                object->setProperty("tick", point.tick);
                object->setProperty("value", point.value);
            }
            points.add(pointObject);
        }

        if (auto* object = laneObject.getDynamicObject())
        {
            object->setProperty("target", lane.target);
            object->setProperty("enabled", lane.enabled);
            object->setProperty("points", juce::var(points));
        }
        controllerLanes.add(laneObject);
    }

    if (auto* patternDyn = patternObject.getDynamicObject())
    {
        patternDyn->setProperty("name", pattern.name);
        patternDyn->setProperty("length_ticks", pattern.lengthTicks);
        patternDyn->setProperty("length_bars", juce::jmax(1, (pattern.lengthTicks + kTicksPerBar - 1) / kTicksPerBar));
        patternDyn->setProperty("notes", juce::var(notes));
        patternDyn->setProperty("controller_lanes", juce::var(controllerLanes));
    }

    if (auto* rootDyn = root.getDynamicObject())
    {
        rootDyn->setProperty("type", kClipboardType);
        rootDyn->setProperty("section_name", section.name);
        rootDyn->setProperty("pattern", patternObject);
    }

    return juce::JSON::toString(root, true);
}

bool parseClipClipboard(const juce::String& text, MidiSection& outSection, MidiPattern& outPattern)
{
    outSection = MidiSection{};
    outPattern = MidiPattern{};

    const auto clipboard = text.trim();
    if (clipboard.isEmpty())
        return false;

    const auto parsed = juce::JSON::parse(clipboard);
    auto* root = parsed.getDynamicObject();
    if (root == nullptr || !root->hasProperty("type") || root->getProperty("type").toString() != kClipboardType)
        return false;

    auto* patternObject = root->getProperty("pattern").getDynamicObject();
    if (patternObject == nullptr)
        return false;

    outSection.name = root->getProperty("section_name").toString().trim();
    outPattern.id = juce::Uuid().toString();
    outPattern.name = patternObject->getProperty("name").toString().trim();
    const auto legacyBars = juce::jmax(1, static_cast<int>(patternObject->getProperty("length_bars")));
    const auto lengthTicksVar = patternObject->hasProperty("length_ticks")
        ? patternObject->getProperty("length_ticks")
        : juce::var(legacyBars * kTicksPerBar);
    outPattern.lengthTicks = juce::jmax(kMinSequenceSnapTicks,
                                        static_cast<int>(lengthTicksVar));

    if (auto* notesArray = patternObject->getProperty("notes").getArray())
    {
        outPattern.notes.reserve(static_cast<size_t>(notesArray->size()));
        for (const auto& item : *notesArray)
        {
            auto* noteObject = item.getDynamicObject();
            if (noteObject == nullptr)
                continue;

            MidiNote note;
            note.startTick = juce::jmax(0, static_cast<int>(noteObject->getProperty("start_tick")));
            note.durationTick = juce::jmax(1, static_cast<int>(noteObject->getProperty("duration_tick")));
            note.pitch = juce::jlimit(0, 127, static_cast<int>(noteObject->getProperty("pitch")));
            note.velocity = juce::jlimit(1, 127, static_cast<int>(noteObject->getProperty("velocity")));
            outPattern.notes.push_back(note);
        }
    }

    if (auto* laneArray = patternObject->getProperty("controller_lanes").getArray())
    {
        outPattern.controllerLanes.reserve(static_cast<size_t>(laneArray->size()));
        for (const auto& laneVar : *laneArray)
        {
            auto* laneObject = laneVar.getDynamicObject();
            if (laneObject == nullptr)
                continue;

            AutomationLane lane;
            lane.target = laneObject->getProperty("target").toString().trim();
            lane.enabled = !laneObject->hasProperty("enabled") || static_cast<bool>(laneObject->getProperty("enabled"));
            if (auto* pointsArray = laneObject->getProperty("points").getArray())
            {
                lane.points.reserve(static_cast<size_t>(pointsArray->size()));
                for (const auto& pointVar : *pointsArray)
                {
                    auto* pointObject = pointVar.getDynamicObject();
                    if (pointObject == nullptr)
                        continue;

                    AutomationPoint point;
                    point.tick = juce::jmax(0, static_cast<int>(pointObject->getProperty("tick")));
                    point.value = static_cast<double>(pointObject->getProperty("value"));
                    lane.points.push_back(point);
                }
            }
            outPattern.controllerLanes.push_back(std::move(lane));
        }
    }

    sanitisePatternControllerLanes(outPattern);

    outSection.patternId = outPattern.id;
    outSection.lengthTicks = patternLengthTicks(outPattern);
    outSection.name = outSection.name.isNotEmpty() ? outSection.name : outPattern.name;
    return true;
}

void trimPatternRight(MidiPattern& pattern, int newLengthTicks)
{
    pattern.lengthTicks = juce::jmax(kMinSequenceSnapTicks, newLengthTicks);
    pattern.notes.erase(std::remove_if(pattern.notes.begin(),
                                       pattern.notes.end(),
                                       [newLengthTicks] (const MidiNote& note)
                                       {
                                           return note.startTick < 0
                                               || (note.startTick + note.durationTick) > newLengthTicks;
                                       }),
                        pattern.notes.end());
    for (auto& lane : pattern.controllerLanes)
    {
        lane.points.erase(std::remove_if(lane.points.begin(),
                                         lane.points.end(),
                                         [newLengthTicks] (const AutomationPoint& point)
                                         {
                                             return point.tick < 0 || point.tick > newLengthTicks;
                                         }),
                          lane.points.end());
    }
}

void shiftPatternForLeftResize(MidiPattern& pattern, int shiftTicks)
{
    if (shiftTicks == 0)
        return;

    const auto oldLengthTicks = patternLengthTicks(pattern);
    if (shiftTicks > 0)
    {
        for (auto& note : pattern.notes)
            note.startTick += shiftTicks;
        pattern.lengthTicks = juce::jmax(kMinSequenceSnapTicks, oldLengthTicks + shiftTicks);
        return;
    }

    const auto trimTicks = -shiftTicks;
    pattern.notes.erase(std::remove_if(pattern.notes.begin(),
                                       pattern.notes.end(),
                                       [trimTicks] (const MidiNote& note)
                                       {
                                           return note.startTick < trimTicks;
                                       }),
                        pattern.notes.end());
    for (auto& note : pattern.notes)
        note.startTick = juce::jmax(0, note.startTick - trimTicks);

    for (auto& lane : pattern.controllerLanes)
    {
        lane.points.erase(std::remove_if(lane.points.begin(),
                                         lane.points.end(),
                                         [trimTicks] (const AutomationPoint& point)
                                         {
                                             return point.tick < trimTicks;
                                         }),
                          lane.points.end());
        for (auto& point : lane.points)
            point.tick = juce::jmax(0, point.tick - trimTicks);
    }

    const auto newLengthTicks = juce::jmax(kMinSequenceSnapTicks, oldLengthTicks - trimTicks);
    pattern.lengthTicks = newLengthTicks;
}

bool tickRangesTouchOrOverlap(int startTick, int endTick, int otherStartTick, int otherEndTick)
{
    return endTick >= otherStartTick && otherEndTick >= startTick;
}

void sortPatternNotes(std::vector<MidiNote>& notes)
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
}

ArrangementOverviewComponent::ArrangementOverviewComponent(ProjectGetter projectGetterIn,
                                                           ProjectWriter projectWriterIn,
                                                           SelectedSectionGetter selectedSectionGetterIn,
                                                           SectionSelectCallback sectionSelectCallbackIn,
                                                           ToolGetter toolGetterIn,
                                                           ToolModeChangeCallback toolModeChangeCallbackIn)
    : projectGetter(std::move(projectGetterIn)),
      projectWriter(std::move(projectWriterIn)),
      selectedSectionGetter(std::move(selectedSectionGetterIn)),
      sectionSelectCallback(std::move(sectionSelectCallbackIn)),
      toolGetter(std::move(toolGetterIn)),
      toolModeChangeCallback(std::move(toolModeChangeCallbackIn))
{
    setWantsKeyboardFocus(true);
    updateContentSize();
}

void ArrangementOverviewComponent::refreshFromModel()
{
    selectedSectionIndices.erase(std::remove_if(selectedSectionIndices.begin(),
                                                selectedSectionIndices.end(),
                                                [this] (int index)
                                                {
                                                    return !juce::isPositiveAndBelow(index,
                                                                                    static_cast<int>(project().midiSections.size()));
                                                }),
                                 selectedSectionIndices.end());
    if (!previewActive)
        updateContentSize();
    repaint();
}

void ArrangementOverviewComponent::setHorizontalZoom(float pixelsPerBarIn)
{
    const auto clamped = juce::jlimit(48.0f, 320.0f, pixelsPerBarIn);
    if (std::abs(pixelsPerBar - clamped) < 0.01f)
        return;

    pixelsPerBar = clamped;
    if (!previewActive)
        updateContentSize();
    repaint();
}

float ArrangementOverviewComponent::getHorizontalZoom() const noexcept
{
    return pixelsPerBar;
}

void ArrangementOverviewComponent::setLaneHeight(float laneHeightIn)
{
    const auto clamped = juce::jlimit(22.0f, 80.0f, laneHeightIn);
    if (std::abs(laneHeightPixels - clamped) < 0.01f)
        return;

    laneHeightPixels = clamped;
    if (!previewActive)
        updateContentSize();
    repaint();
}

float ArrangementOverviewComponent::getLaneHeight() const noexcept
{
    return laneHeightPixels;
}

void ArrangementOverviewComponent::paint(juce::Graphics& g)
{
    g.fillAll(kBackground);

    const auto& state = displayedProject();
    const auto laneCount = juce::jmax(1, static_cast<int>(state.tracks.size()));
    const auto headerWidth = laneHeaderWidth();
    const auto topHeight = rulerHeight();
    const auto selectedSectionIndex = selectedSectionGetter != nullptr ? selectedSectionGetter() : -1;
    const auto projectBarTicks = ticksPerBar(state);
    const auto projectBeatTicks = ticksPerTimeSignatureBeat(state);

    g.setColour(kHeader);
    g.fillRect(0.0f, 0.0f, static_cast<float>(getWidth()), topHeight);
    g.fillRect(0.0f, 0.0f, headerWidth, static_cast<float>(getHeight()));

    for (int lane = 0; lane < laneCount; ++lane)
    {
        const auto y = topHeight + (static_cast<float>(lane) * laneHeight());
        g.setColour((lane % 2) == 0 ? kLaneEven : kLaneOdd);
        g.fillRect(headerWidth, y, static_cast<float>(getWidth()) - headerWidth, laneHeight());

        const auto label = juce::isPositiveAndBelow(lane, static_cast<int>(state.tracks.size()))
            ? state.tracks[static_cast<size_t>(lane)].name
            : "Track " + juce::String(lane + 1);
        g.setColour(juce::Colour::fromRGB(182, 190, 201));
        g.setFont(ui::font());
        g.drawText(label,
                   6,
                   juce::roundToInt(y),
                   juce::roundToInt(headerWidth) - 8,
                   juce::roundToInt(laneHeight()),
                   juce::Justification::centredLeft,
                   true);

        g.setColour(juce::Colour::fromRGB(46, 52, 64));
        g.drawHorizontalLine(juce::roundToInt(y), 0.0f, static_cast<float>(getWidth()));
    }

    const auto totalBars = displayedBarCount();
    for (int bar = 0; bar <= totalBars; ++bar)
    {
        const auto barTick = bar * projectBarTicks;
        const auto x = tickToX(barTick);
        g.setColour(kGridMajor);
        g.drawVerticalLine(juce::roundToInt(x), 0.0f, static_cast<float>(getHeight()));
        if (bar < totalBars)
        {
            for (int beat = 1; beat < state.timeSigNumerator; ++beat)
            {
                const auto beatX = tickToX(barTick + (beat * projectBeatTicks));
                g.setColour(kGridMinor);
                g.drawVerticalLine(juce::roundToInt(beatX), topHeight, static_cast<float>(getHeight()));
            }
        }

        if (bar < totalBars)
        {
            g.setColour(juce::Colour::fromRGB(219, 224, 232));
            g.setFont(ui::font());
            g.drawText(juce::String(bar + 1),
                       juce::roundToInt(x) + 3,
                       0,
                       28,
                       juce::roundToInt(topHeight),
                       juce::Justification::centredLeft);
        }
    }

    for (const auto& clip : state.sampleClips)
    {
        auto rect = clipRect(clip).reduced(1.5f, 4.0f);
        g.setColour(kClip.withAlpha(0.55f));
        g.fillRoundedRectangle(rect, 4.0f);
        g.setColour(kClip.darker(0.35f));
        g.drawRoundedRectangle(rect, 4.0f, 1.0f);
    }

    for (int sectionIndex = 0; sectionIndex < static_cast<int>(state.midiSections.size()); ++sectionIndex)
    {
        const auto& section = state.midiSections[static_cast<size_t>(sectionIndex)];
        if (!juce::isPositiveAndBelow(section.trackIndex, static_cast<int>(state.tracks.size())))
            continue;

        auto rect = sectionRect(section, state).reduced(1.5f, 4.0f);
        const auto baseColour = trackDisplayColour(state.tracks[static_cast<size_t>(section.trackIndex)], section.trackIndex);
        auto fill = baseColour.withAlpha(0.86f);
        if (previewActive && sectionIndex == draggedSectionIndex)
            fill = fill.brighter(0.18f);
        g.setColour(fill);
        g.fillRoundedRectangle(rect, 5.0f);

        const auto isSelected = sectionIndex == selectedSectionIndex
            || std::find(selectedSectionIndices.begin(), selectedSectionIndices.end(), sectionIndex) != selectedSectionIndices.end();
        g.setColour(isSelected ? juce::Colours::white : baseColour.darker(0.55f));
        g.drawRoundedRectangle(rect, 5.0f, isSelected ? 2.0f : 1.2f);

        auto clipTitle = section.name.trim();
        if (const auto* pattern = findMidiPattern(state, section.patternId))
            clipTitle = pattern->name.trim().isNotEmpty() ? pattern->name : clipTitle;
        if (clipTitle.isEmpty())
            clipTitle = "Pattern";

        g.setColour(trackTextColour(baseColour));
        g.setFont(ui::font());
        g.drawText(clipTitle,
                   rect.toNearestInt().reduced(6, 0),
                   juce::Justification::centredLeft,
                   true);
    }

    for (const auto& locator : { std::pair(state.leftLocatorTick, kLeftLocator),
                                 std::pair(state.rightLocatorTick, kRightLocator),
                                 std::pair(state.playheadTick, kPlayhead) })
    {
        const auto x = tickToX(locator.first);
        g.setColour(locator.second);
        g.drawVerticalLine(juce::roundToInt(x), 0.0f, static_cast<float>(getHeight()));
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

    if (dragMode == DragMode::marqueeSelect && !marqueeRect.isEmpty())
    {
        auto drawRect = marqueeRect.getIntersection(getLocalBounds().toFloat()).reduced(0.5f);
        if (!drawRect.isEmpty())
        {
            g.setColour(juce::Colour::fromRGBA(128, 176, 255, 42));
            g.fillRect(drawRect);
            g.setColour(juce::Colour::fromRGB(144, 196, 255));
            g.drawRect(drawRect, 1.5f);
        }
    }
}

void ArrangementOverviewComponent::resized()
{
    updateContentSize();
}

void ArrangementOverviewComponent::mouseMove(const juce::MouseEvent& event)
{
    updateCursorForPosition(event.position);
}

void ArrangementOverviewComponent::mouseDown(const juce::MouseEvent& event)
{
    const auto toolMode = toolGetter != nullptr ? toolGetter() : EditorToolMode::pencil;

    if (event.mods.isLeftButtonDown())
        grabKeyboardFocus();

    if (event.position.x > laneHeaderWidth() && event.position.y <= rulerHeight())
    {
        const auto markerDragMode = hitTestTransportMarker(event.position);
        if (markerDragMode != DragMode::none && event.mods.isLeftButtonDown())
        {
            beforeProject = project();
            previewProject = beforeProject;
            previewActive = true;
            previewDirty = false;
            dragMode = markerDragMode;
            draggedSectionIndex = -1;
            repaint();
            return;
        }

        auto updatedProject = project();
        const auto snappedTick = xToSnapStartTick(event.position.x);

        if (event.mods.isShiftDown())
            updatedProject.leftLocatorTick = juce::jmax(0, snappedTick);
        else if (event.mods.isAltDown() || event.mods.isRightButtonDown())
            updatedProject.rightLocatorTick = juce::jmax(updatedProject.leftLocatorTick + kTicksPerBeat, snappedTick);
        else
            updatedProject.playheadTick = juce::jmax(0, snappedTick);

        updatedProject.recalculateTimeFields();
        projectWriter(updatedProject, false, "Move Arrangement Transport");
        refreshFromModel();
        return;
    }

    juce::String hitEdge;
    const auto hitSectionIndex = hitTestMidiSection(event.position, hitEdge);
    contextMenuTrackIndex = yToTrackIndex(event.position.y);
    contextMenuTick = xToSnapStartTick(event.position.x);
    if (event.mods.isPopupMenu())
    {
        if (hitSectionIndex >= 0 && sectionSelectCallback != nullptr)
            sectionSelectCallback(hitSectionIndex, false);
        showSectionContextMenu(hitSectionIndex, event.getScreenPosition(), contextMenuTrackIndex, contextMenuTick);
        return;
    }

    previewActive = false;
    previewDirty = false;
    dragMode = DragMode::none;
    draggedSectionIndex = -1;
    dragOffsetTick = 0;
    createStartTick = 0;
    resizeAnchorStartTick = 0;
    resizeAnchorLengthTicks = ticksPerBar(project());
    dragCreatesCopy = false;
    marqueeStart = {};
    marqueeRect = {};

    if (hitSectionIndex >= 0)
    {
        selectedSectionIndices.clear();
        selectedSectionIndices.push_back(hitSectionIndex);
        if (sectionSelectCallback != nullptr)
            sectionSelectCallback(hitSectionIndex, event.getNumberOfClicks() > 1);

        if (event.getNumberOfClicks() > 1)
            return;

        if (toolMode == EditorToolMode::eraser && event.mods.isLeftButtonDown())
        {
            beforeProject = project();
            previewProject = beforeProject;
            previewActive = true;
            previewDirty = false;
            dragMode = DragMode::eraseSections;
            draggedSectionIndex = -1;
            erasedSectionIndices.clear();
            if (markSectionForErase(hitSectionIndex))
            {
                previewDirty = true;
                if (sectionSelectCallback != nullptr)
                    sectionSelectCallback(-1, false);
            }
            repaint();
            return;
        }

        if (toolMode == EditorToolMode::glue && event.mods.isLeftButtonDown())
        {
            if (glueSectionsAtIndex(hitSectionIndex))
                refreshFromModel();
            return;
        }

        const auto& state = project();
        beforeProject = state;
        previewProject = state;
        previewActive = true;
        previewDirty = false;
        dragMode = DragMode::moveSection;
        draggedSectionIndex = hitSectionIndex;
        const auto& sourceSection = state.midiSections[static_cast<size_t>(hitSectionIndex)];
        const bool duplicateDragRequested = event.mods.isLeftButtonDown()
            && hitEdge == "body"
            && (event.mods.isAltDown() || event.mods.isCtrlDown() || event.mods.isCommandDown());

        if (duplicateDragRequested)
        {
            if (const auto* sourcePattern = findMidiPattern(previewProject, sourceSection.patternId))
            {
                auto duplicatePattern = *sourcePattern;
                duplicatePattern.id = juce::Uuid().toString();

                auto duplicateSection = sourceSection;
                duplicateSection.patternId = duplicatePattern.id;

                previewProject.midiPatterns.push_back(std::move(duplicatePattern));
                previewProject.midiSections.push_back(std::move(duplicateSection));
                draggedSectionIndex = static_cast<int>(previewProject.midiSections.size()) - 1;
                dragCreatesCopy = true;
            }
        }

        const auto& draggedSection = previewProject.midiSections[static_cast<size_t>(draggedSectionIndex)];
        dragOffsetTick = juce::jmax(0, xToSnapStartTick(event.position.x) - draggedSection.startTick);
        resizeAnchorStartTick = draggedSection.startTick;
        resizeAnchorLengthTicks = clipLengthTicks(draggedSection, previewProject);
        if (hitEdge == "left")
            dragMode = DragMode::resizeLeft;
        else if (hitEdge == "right")
            dragMode = DragMode::resizeRight;
        repaint();
        return;
    }

    if (toolMode == EditorToolMode::selection && event.mods.isLeftButtonDown())
    {
        beforeProject = project();
        previewProject = beforeProject;
        previewActive = true;
        previewDirty = false;
        dragMode = DragMode::marqueeSelect;
        draggedSectionIndex = -1;
        selectedSectionIndices.clear();
        marqueeStart = event.position;
        marqueeRect = { marqueeStart, marqueeStart };
        if (sectionSelectCallback != nullptr)
            sectionSelectCallback(-1, false);
        repaint();
        return;
    }

    if (toolMode != EditorToolMode::pencil)
    {
        selectedSectionIndices.clear();
        if (sectionSelectCallback != nullptr)
            sectionSelectCallback(-1, false);
        updateCursorForPosition(event.position);
        return;
    }

    const auto trackIndex = yToTrackIndex(event.position.y);
    if (!juce::isPositiveAndBelow(trackIndex, static_cast<int>(project().tracks.size())))
        return;

    beforeProject = project();
    previewProject = project();
    previewActive = true;
    previewDirty = true;
    dragMode = DragMode::createSection;
    createStartTick = xToSnapStartTick(event.position.x);

    MidiPattern pattern;
    pattern.id = juce::Uuid().toString();
    pattern.lengthTicks = defaultPatternLengthTicks(previewProject);
    const auto trackName = previewProject.tracks[static_cast<size_t>(trackIndex)].name.trim();
    pattern.name = trackName.isNotEmpty() ? (trackName + " Pattern") : "Pattern";
    previewProject.midiPatterns.push_back(pattern);

    MidiSection section;
    section.trackIndex = trackIndex;
    section.startTick = createStartTick;
    section.lengthTicks = pattern.lengthTicks;
    section.name = pattern.name;
    section.patternId = pattern.id;
    previewProject.midiSections.push_back(section);
    draggedSectionIndex = static_cast<int>(previewProject.midiSections.size()) - 1;
    repaint();
}

void ArrangementOverviewComponent::mouseDrag(const juce::MouseEvent& event)
{
    if (!previewActive)
        return;

    if (dragMode == DragMode::moveLeftLocator
        || dragMode == DragMode::moveRightLocator
        || dragMode == DragMode::movePlayhead)
    {
        previewProject = beforeProject;
        const auto snappedTick = juce::jmax(0, xToSnapStartTick(event.position.x));
        const auto minimumSpan = arrangementSnapTickLength(previewProject);

        if (dragMode == DragMode::moveLeftLocator)
            previewProject.leftLocatorTick = juce::jmin(snappedTick, juce::jmax(0, previewProject.rightLocatorTick - minimumSpan));
        else if (dragMode == DragMode::moveRightLocator)
            previewProject.rightLocatorTick = juce::jmax(previewProject.leftLocatorTick + minimumSpan, snappedTick);
        else
            previewProject.playheadTick = snappedTick;

        previewProject.recalculateTimeFields();
        previewDirty = true;
        repaint();
        return;
    }

    if (dragMode == DragMode::eraseSections)
    {
        juce::String hitEdge;
        const auto hitSectionIndex = hitTestMidiSection(beforeProject, event.position, hitEdge);
        if (hitSectionIndex >= 0 && markSectionForErase(hitSectionIndex))
        {
            previewDirty = true;
            repaint();
        }
        return;
    }

    if (dragMode == DragMode::marqueeSelect)
    {
        const auto current = event.position;
        marqueeRect = juce::Rectangle<float>::leftTopRightBottom(juce::jmin(marqueeStart.x, current.x),
                                                                 juce::jmin(marqueeStart.y, current.y),
                                                                 juce::jmax(marqueeStart.x, current.x),
                                                                 juce::jmax(marqueeStart.y, current.y));
        selectedSectionIndices.clear();
        for (int sectionIndex = 0; sectionIndex < static_cast<int>(beforeProject.midiSections.size()); ++sectionIndex)
        {
            if (sectionRect(beforeProject.midiSections[static_cast<size_t>(sectionIndex)], beforeProject).intersects(marqueeRect))
                selectedSectionIndices.push_back(sectionIndex);
        }
        repaint();
        return;
    }

    if (!juce::isPositiveAndBelow(draggedSectionIndex, static_cast<int>(previewProject.midiSections.size())))
        return;

    if (dragMode != DragMode::createSection && !dragCreatesCopy)
        previewProject = beforeProject;

    auto& section = previewProject.midiSections[static_cast<size_t>(draggedSectionIndex)];

    if (dragMode == DragMode::moveSection)
    {
        section.startTick = juce::jmax(0, xToSnapStartTick(event.position.x) - dragOffsetTick);
        section.trackIndex = juce::jlimit(0,
                                          juce::jmax(0, static_cast<int>(previewProject.tracks.size()) - 1),
                                          yToTrackIndex(event.position.y));
        previewDirty = true;
        repaint();
        return;
    }

    if (dragMode == DragMode::resizeRight)
    {
        const auto newEndTick = juce::jmax(section.startTick + arrangementSnapTickLength(previewProject),
                                           xToSnapEndTick(event.position.x));
        const auto newLengthTicks = juce::jmax(kMinSequenceSnapTicks, newEndTick - section.startTick);
        if (auto* pattern = findMidiPattern(previewProject, section.patternId))
            trimPatternRight(*pattern, newLengthTicks);
        section.lengthTicks = newLengthTicks;
        previewDirty = true;
        repaint();
        return;
    }

    if (dragMode == DragMode::resizeLeft)
    {
        const auto oldEndTick = resizeAnchorStartTick + resizeAnchorLengthTicks;
        const auto newStartTick = juce::jlimit(0,
                                               oldEndTick - arrangementSnapTickLength(previewProject),
                                               xToSnapStartTick(event.position.x));
        section.startTick = newStartTick;
        if (auto* pattern = findMidiPattern(previewProject, section.patternId))
        {
            shiftPatternForLeftResize(*pattern, resizeAnchorStartTick - newStartTick);
            section.lengthTicks = patternLengthTicks(*pattern);
        }
        previewDirty = true;
        repaint();
        return;
    }

    if (dragMode == DragMode::createSection)
    {
        const auto dragEndTick = juce::jmax(createStartTick + arrangementSnapTickLength(previewProject),
                                            xToSnapEndTick(event.position.x));
        const auto lengthTicks = juce::jmax(kMinSequenceSnapTicks, dragEndTick - createStartTick);
        section.lengthTicks = lengthTicks;
        section.trackIndex = juce::jlimit(0,
                                          juce::jmax(0, static_cast<int>(previewProject.tracks.size()) - 1),
                                          yToTrackIndex(event.position.y));
        if (auto* pattern = findMidiPattern(previewProject, section.patternId))
            pattern->lengthTicks = lengthTicks;
        repaint();
    }
}

void ArrangementOverviewComponent::mouseUp(const juce::MouseEvent&)
{
    if (!previewActive)
        return;

    if (dragMode == DragMode::marqueeSelect)
    {
        previewActive = false;
        previewDirty = false;
        dragMode = DragMode::none;
        draggedSectionIndex = -1;
        dragOffsetTick = 0;
        createStartTick = 0;
        resizeAnchorStartTick = 0;
        resizeAnchorLengthTicks = ticksPerBar(project());
        dragCreatesCopy = false;
        marqueeStart = {};
        marqueeRect = {};
        erasedSectionIndices.clear();

        if (sectionSelectCallback != nullptr)
        {
            if (selectedSectionIndices.size() == 1)
                sectionSelectCallback(selectedSectionIndices.front(), false);
            else
                sectionSelectCallback(-1, false);
        }

        repaint();
        return;
    }

    if (previewDirty)
    {
        juce::String actionName = dragCreatesCopy ? "Duplicate Pattern Clip" : "Move Pattern Clip";
        bool undoable = true;
        if (dragMode == DragMode::createSection)
            actionName = "Create Pattern Clip";
        else if (dragMode == DragMode::resizeLeft || dragMode == DragMode::resizeRight)
            actionName = "Resize Pattern Clip";
        else if (dragMode == DragMode::moveLeftLocator)
            actionName = "Move Left Locator";
        else if (dragMode == DragMode::moveRightLocator)
            actionName = "Move Right Locator";
        else if (dragMode == DragMode::movePlayhead)
        {
            actionName = "Move Playhead";
            undoable = false;
        }
        else if (dragMode == DragMode::eraseSections)
        {
            actionName = "Erase Pattern Clips";
        }
        projectWriter(previewProject, undoable, actionName);
        if (dragMode == DragMode::eraseSections && sectionSelectCallback != nullptr)
            sectionSelectCallback(-1, false);
        else if (draggedSectionIndex >= 0 && sectionSelectCallback != nullptr)
            sectionSelectCallback(draggedSectionIndex, false);
    }

    previewActive = false;
    previewDirty = false;
    dragMode = DragMode::none;
    draggedSectionIndex = -1;
    dragOffsetTick = 0;
    createStartTick = 0;
    resizeAnchorStartTick = 0;
    resizeAnchorLengthTicks = ticksPerBar(project());
    dragCreatesCopy = false;
    marqueeStart = {};
    marqueeRect = {};
    erasedSectionIndices.clear();
    updateCursorForPosition({ -1.0f, -1.0f });
    refreshFromModel();
}

void ArrangementOverviewComponent::mouseExit(const juce::MouseEvent&)
{
    updateCursorForPosition({ -1.0f, -1.0f });
}

bool ArrangementOverviewComponent::keyPressed(const juce::KeyPress& key)
{
    if (key != juce::KeyPress::deleteKey && key != juce::KeyPress::backspaceKey)
        return false;

    std::vector<int> indicesToDelete = selectedSectionIndices;
    if (indicesToDelete.empty())
    {
        const auto selectedSectionIndex = selectedSectionGetter != nullptr ? selectedSectionGetter() : -1;
        if (juce::isPositiveAndBelow(selectedSectionIndex, static_cast<int>(project().midiSections.size())))
            indicesToDelete.push_back(selectedSectionIndex);
    }

    if (indicesToDelete.empty())
        return false;

    std::sort(indicesToDelete.begin(), indicesToDelete.end());
    indicesToDelete.erase(std::unique(indicesToDelete.begin(), indicesToDelete.end()), indicesToDelete.end());

    auto updatedProject = project();
    for (auto iterator = indicesToDelete.rbegin(); iterator != indicesToDelete.rend(); ++iterator)
    {
        if (juce::isPositiveAndBelow(*iterator, static_cast<int>(updatedProject.midiSections.size())))
            updatedProject.midiSections.erase(updatedProject.midiSections.begin() + *iterator);
    }

    selectedSectionIndices.clear();
    projectWriter(updatedProject, true, "Delete Pattern Clips");
    if (sectionSelectCallback != nullptr)
        sectionSelectCallback(-1, false);
    return true;
}

const ProjectState& ArrangementOverviewComponent::project() const
{
    return projectGetter();
}

const ProjectState& ArrangementOverviewComponent::displayedProject() const
{
    return previewActive ? previewProject : project();
}

void ArrangementOverviewComponent::updateContentSize()
{
    const auto laneCount = juce::jmax(1, static_cast<int>(project().tracks.size()));
    const auto width = juce::roundToInt(tickToX(displayedBarCount() * ticksPerBar(displayedProject())));
    const auto height = juce::roundToInt(rulerHeight() + (static_cast<float>(laneCount) * laneHeight()));
    setSize(juce::jmax(width, 900), juce::jmax(height, 180));
}

float ArrangementOverviewComponent::laneHeaderWidth() const
{
    return 132.0f;
}

float ArrangementOverviewComponent::rulerHeight() const
{
    return 24.0f;
}

float ArrangementOverviewComponent::laneHeight() const
{
    return laneHeightPixels;
}

juce::Rectangle<float> ArrangementOverviewComponent::transportHandleBounds(int tick) const
{
    const auto x = tickToX(tick);
    return { x - (kTransportHandleWidth * 0.5f),
             2.0f,
             kTransportHandleWidth,
             kTransportHandleHeight };
}

ArrangementOverviewComponent::DragMode ArrangementOverviewComponent::hitTestTransportMarker(juce::Point<float> position) const
{
    if (position.y < 0.0f || position.y > (rulerHeight() + 6.0f) || position.x <= laneHeaderWidth())
        return DragMode::none;

    const auto& state = displayedProject();
    const auto matchesMarker = [this, position] (int tick)
    {
        const auto handle = transportHandleBounds(tick).expanded(4.0f, 4.0f);
        const auto x = tickToX(tick);
        return handle.contains(position) || (position.y <= rulerHeight() && std::abs(position.x - x) <= 5.0f);
    };

    if (matchesMarker(state.leftLocatorTick))
        return DragMode::moveLeftLocator;
    if (matchesMarker(state.rightLocatorTick))
        return DragMode::moveRightLocator;
    if (matchesMarker(state.playheadTick))
        return DragMode::movePlayhead;
    return DragMode::none;
}

float ArrangementOverviewComponent::tickToX(int tick) const
{
    return laneHeaderWidth()
        + (static_cast<float>(juce::jmax(0, tick)) / static_cast<float>(ticksPerBar(displayedProject()))) * pixelsPerBar;
}

int ArrangementOverviewComponent::xToSnapStartTick(float x) const
{
    const auto relative = juce::jmax(0.0f, x - laneHeaderWidth());
    const auto pixelsPerTick = pixelsPerBar / static_cast<float>(ticksPerBar(project()));
    const auto snapTicks = arrangementSnapTickLength(project());
    const auto rawTick = static_cast<int>(std::floor(relative / juce::jmax(0.001f, pixelsPerTick)));
    return juce::jmax(0, (rawTick / snapTicks) * snapTicks);
}

int ArrangementOverviewComponent::xToSnapEndTick(float x) const
{
    return xToSnapStartTick(x) + arrangementSnapTickLength(project());
}

int ArrangementOverviewComponent::yToTrackIndex(float y) const
{
    const auto relative = juce::jmax(0.0f, y - rulerHeight());
    return static_cast<int>(relative / laneHeight());
}

int ArrangementOverviewComponent::displayedBarCount() const
{
    const auto& state = displayedProject();
    const auto projectBarTicks = ticksPerBar(state);
    int lastTick = juce::jmax(projectBarTicks * minimumBars, state.rightLocatorTick + projectBarTicks);

    for (const auto& section : state.midiSections)
        lastTick = juce::jmax(lastTick, section.startTick + clipLengthTicks(section, state));

    for (const auto& clip : state.sampleClips)
    {
        const auto startTick = secondsToTick(state, clip.startSec);
        const auto endTick = secondsToTick(state, clip.startSec + juce::jmax(0.0, clip.durationSec));
        lastTick = juce::jmax(lastTick, endTick);
        lastTick = juce::jmax(lastTick, startTick + projectBarTicks);
    }

    return juce::jmax(minimumBars, (lastTick + projectBarTicks - 1) / projectBarTicks);
}

int ArrangementOverviewComponent::clipLengthTicks(const MidiSection& section, const ProjectState& state) const
{
    if (const auto* pattern = findMidiPattern(state, section.patternId))
        return patternLengthTicks(*pattern);
    return juce::jmax(kMinSequenceSnapTicks, section.lengthTicks);
}

juce::Rectangle<float> ArrangementOverviewComponent::sectionRect(const MidiSection& section, const ProjectState& state) const
{
    const auto lengthTicks = clipLengthTicks(section, state);
    return { tickToX(section.startTick),
             rulerHeight() + (static_cast<float>(juce::jmax(0, section.trackIndex)) * laneHeight()),
             juce::jmax(14.0f, tickToX(section.startTick + lengthTicks) - tickToX(section.startTick)),
             laneHeight() };
}

juce::Rectangle<float> ArrangementOverviewComponent::clipRect(const SampleClip& clip) const
{
    const auto& state = displayedProject();
    const auto startTick = secondsToTick(state, clip.startSec);
    const auto endTick = secondsToTick(state, clip.startSec + juce::jmax(0.0, clip.durationSec));
    return { tickToX(startTick),
             rulerHeight() + (static_cast<float>(juce::jmax(0, clip.trackIndex)) * laneHeight()),
             juce::jmax(10.0f, tickToX(juce::jmax(startTick + 1, endTick)) - tickToX(startTick)),
             laneHeight() };
}

int ArrangementOverviewComponent::hitTestMidiSection(const ProjectState& state, juce::Point<float> position, juce::String& edgeOut) const
{
    edgeOut.clear();
    for (int sectionIndex = static_cast<int>(state.midiSections.size()) - 1; sectionIndex >= 0; --sectionIndex)
    {
        auto rect = sectionRect(state.midiSections[static_cast<size_t>(sectionIndex)], state).reduced(1.5f, 4.0f);
        if (rect.contains(position))
        {
            constexpr float edgeWidth = 8.0f;
            if (rect.getWidth() > edgeWidth * 2.0f)
            {
                if (position.x <= rect.getX() + edgeWidth)
                    edgeOut = "left";
                else if (position.x >= rect.getRight() - edgeWidth)
                    edgeOut = "right";
                else
                    edgeOut = "body";
            }
            else
            {
                edgeOut = "body";
            }
            return sectionIndex;
        }
    }

    return -1;
}

int ArrangementOverviewComponent::hitTestMidiSection(juce::Point<float> position, juce::String& edgeOut) const
{
    return hitTestMidiSection(displayedProject(), position, edgeOut);
}

void ArrangementOverviewComponent::updateCursorForPosition(juce::Point<float> position)
{
    if (!getLocalBounds().toFloat().contains(position))
    {
        setMouseCursor(juce::MouseCursor::NormalCursor);
        return;
    }

    if (hitTestTransportMarker(position) != DragMode::none)
    {
        setMouseCursor(juce::MouseCursor::PointingHandCursor);
        return;
    }

    juce::String hitEdge;
    const auto hitSectionIndex = hitTestMidiSection(position, hitEdge);
    if (hitSectionIndex >= 0)
    {
        const auto toolMode = toolGetter != nullptr ? toolGetter() : EditorToolMode::pencil;
        if (toolMode == EditorToolMode::eraser)
        {
            setMouseCursor(juce::MouseCursor::CrosshairCursor);
            return;
        }

        if (toolMode == EditorToolMode::glue)
        {
            setMouseCursor(juce::MouseCursor::PointingHandCursor);
            return;
        }

        if (hitEdge == "left" || hitEdge == "right")
            setMouseCursor(juce::MouseCursor::LeftRightResizeCursor);
        else
            setMouseCursor(juce::MouseCursor::DraggingHandCursor);
        return;
    }

    const auto toolMode = toolGetter != nullptr ? toolGetter() : EditorToolMode::pencil;
    setMouseCursor((toolMode == EditorToolMode::pencil || toolMode == EditorToolMode::eraser)
                       ? juce::MouseCursor::CrosshairCursor
                       : juce::MouseCursor::NormalCursor);
}

bool ArrangementOverviewComponent::copySectionToClipboard(int sectionIndex) const
{
    const auto& state = project();
    if (!juce::isPositiveAndBelow(sectionIndex, static_cast<int>(state.midiSections.size())))
        return false;

    const auto& section = state.midiSections[static_cast<size_t>(sectionIndex)];
    const auto* pattern = findMidiPattern(state, section.patternId);
    if (pattern == nullptr)
        return false;

    juce::SystemClipboard::copyTextToClipboard(serialiseClipClipboard(section, *pattern));
    return true;
}

bool ArrangementOverviewComponent::pasteClipboardAt(ProjectState& updatedProject,
                                                    int targetTrackIndex,
                                                    int targetStartTick,
                                                    int& outSectionIndex) const
{
    outSectionIndex = -1;

    if (!juce::isPositiveAndBelow(targetTrackIndex, static_cast<int>(updatedProject.tracks.size())))
        return false;

    MidiSection clipboardSection;
    MidiPattern clipboardPattern;
    if (!parseClipClipboard(juce::SystemClipboard::getTextFromClipboard(), clipboardSection, clipboardPattern))
        return false;

    clipboardPattern.id = juce::Uuid().toString();
    clipboardSection.patternId = clipboardPattern.id;
    clipboardSection.trackIndex = targetTrackIndex;
    clipboardSection.startTick = juce::jmax(0, targetStartTick);
    clipboardSection.lengthTicks = patternLengthTicks(clipboardPattern);
    clipboardSection.name = clipboardSection.name.trim().isNotEmpty() ? clipboardSection.name.trim()
                                                                      : clipboardPattern.name.trim();

    updatedProject.midiPatterns.push_back(std::move(clipboardPattern));
    updatedProject.midiSections.push_back(std::move(clipboardSection));
    outSectionIndex = static_cast<int>(updatedProject.midiSections.size()) - 1;
    return true;
}

bool ArrangementOverviewComponent::glueSectionsAtIndex(int sectionIndex)
{
    const auto& sourceProject = project();
    if (!juce::isPositiveAndBelow(sectionIndex, static_cast<int>(sourceProject.midiSections.size())))
        return false;

    const auto& clickedSection = sourceProject.midiSections[static_cast<size_t>(sectionIndex)];
    std::vector<int> cluster { sectionIndex };
    std::vector<bool> included(sourceProject.midiSections.size(), false);
    included[static_cast<size_t>(sectionIndex)] = true;

    bool expanded = true;
    while (expanded)
    {
        expanded = false;
        int clusterStartTick = std::numeric_limits<int>::max();
        int clusterEndTick = 0;

        for (const auto index : cluster)
        {
            const auto& section = sourceProject.midiSections[static_cast<size_t>(index)];
            clusterStartTick = juce::jmin(clusterStartTick, section.startTick);
            clusterEndTick = juce::jmax(clusterEndTick,
                                        section.startTick + clipLengthTicks(section, sourceProject));
        }

        for (int index = 0; index < static_cast<int>(sourceProject.midiSections.size()); ++index)
        {
            if (included[static_cast<size_t>(index)])
                continue;

            const auto& section = sourceProject.midiSections[static_cast<size_t>(index)];
            if (section.trackIndex != clickedSection.trackIndex)
                continue;

            const auto sectionEndTick = section.startTick + clipLengthTicks(section, sourceProject);
            if (!tickRangesTouchOrOverlap(clusterStartTick, clusterEndTick, section.startTick, sectionEndTick))
                continue;

            included[static_cast<size_t>(index)] = true;
            cluster.push_back(index);
            expanded = true;
        }
    }

    if (cluster.size() < 2)
        return false;

    std::sort(cluster.begin(),
              cluster.end(),
              [&sourceProject] (int lhsIndex, int rhsIndex)
              {
                  const auto& lhs = sourceProject.midiSections[static_cast<size_t>(lhsIndex)];
                  const auto& rhs = sourceProject.midiSections[static_cast<size_t>(rhsIndex)];
                  if (lhs.startTick != rhs.startTick)
                      return lhs.startTick < rhs.startTick;
                  return lhsIndex < rhsIndex;
              });

    const auto* clickedPattern = findMidiPattern(sourceProject, clickedSection.patternId);
    if (clickedPattern == nullptr)
        return false;

    const auto newStartTick = sourceProject.midiSections[static_cast<size_t>(cluster.front())].startTick;
    int newEndTick = newStartTick + clipLengthTicks(sourceProject.midiSections[static_cast<size_t>(cluster.front())],
                                                    sourceProject);

    MidiPattern gluedPattern;
    gluedPattern.id = juce::Uuid().toString();
    gluedPattern.name = clickedPattern->name.trim().isNotEmpty() ? clickedPattern->name.trim()
                                                                 : clickedSection.name.trim();

    for (const auto index : cluster)
    {
        const auto& section = sourceProject.midiSections[static_cast<size_t>(index)];
        newEndTick = juce::jmax(newEndTick, section.startTick + clipLengthTicks(section, sourceProject));

        if (const auto* pattern = findMidiPattern(sourceProject, section.patternId))
        {
            for (auto note : pattern->notes)
            {
                note.startTick = juce::jmax(0, note.startTick + (section.startTick - newStartTick));
                note.selected = false;
                gluedPattern.notes.push_back(std::move(note));
            }
        }
    }

    gluedPattern.lengthTicks = juce::jmax(kMinSequenceSnapTicks, newEndTick - newStartTick);
    sortPatternNotes(gluedPattern.notes);

    MidiSection gluedSection;
    gluedSection.trackIndex = clickedSection.trackIndex;
    gluedSection.startTick = newStartTick;
    gluedSection.lengthTicks = gluedPattern.lengthTicks;
    gluedSection.name = clickedSection.name.trim().isNotEmpty() ? clickedSection.name.trim()
                                                                : gluedPattern.name;
    gluedSection.patternId = gluedPattern.id;

    auto updatedProject = sourceProject;
    const auto insertionIndex = *std::min_element(cluster.begin(), cluster.end());
    for (auto iter = cluster.rbegin(); iter != cluster.rend(); ++iter)
        updatedProject.midiSections.erase(updatedProject.midiSections.begin() + *iter);

    updatedProject.midiPatterns.push_back(gluedPattern);

    const auto boundedInsertIndex = juce::jlimit(0,
                                                 static_cast<int>(updatedProject.midiSections.size()),
                                                 insertionIndex);
    updatedProject.midiSections.insert(updatedProject.midiSections.begin() + boundedInsertIndex,
                                       gluedSection);
    projectWriter(updatedProject, true, "Glue Pattern Clips");
    if (sectionSelectCallback != nullptr)
        sectionSelectCallback(boundedInsertIndex, false);
    return true;
}

void ArrangementOverviewComponent::showSectionContextMenu(int sectionIndex,
                                                          juce::Point<int> screenPosition,
                                                          int targetTrackIndex,
                                                          int targetStartTick)
{
    const bool hasSection = juce::isPositiveAndBelow(sectionIndex, static_cast<int>(project().midiSections.size()));
    MidiSection clipboardSection;
    MidiPattern clipboardPattern;
    const bool canPaste = parseClipClipboard(juce::SystemClipboard::getTextFromClipboard(),
                                             clipboardSection,
                                             clipboardPattern)
        && juce::isPositiveAndBelow(targetTrackIndex, static_cast<int>(project().tracks.size()));

    enum MenuItemIds
    {
        menuToolPencil = 1,
        menuToolSelect,
        menuToolGlue,
        menuToolEraser,
        menuOpen,
        menuCut,
        menuCopy,
        menuPaste,
        menuDelete,
        menuDuplicate
    };

    juce::PopupMenu menu;
    const auto toolMode = toolGetter != nullptr ? toolGetter() : EditorToolMode::pencil;

    menu.addSectionHeader("Tools");
    menu.addItem(menuToolPencil, "Pencil", true, toolMode == EditorToolMode::pencil);
    menu.addItem(menuToolSelect, "Select", true, toolMode == EditorToolMode::selection);
    menu.addItem(menuToolGlue, "Glue", true, toolMode == EditorToolMode::glue);
    menu.addItem(menuToolEraser, "Eraser", true, toolMode == EditorToolMode::eraser);
    menu.addSeparator();
    menu.addSectionHeader("Clips");
    menu.addItem(menuOpen, "Open Pattern", hasSection);
    menu.addItem(menuCut, "Cut", hasSection);
    menu.addItem(menuCopy, "Copy", hasSection);
    menu.addItem(menuPaste, "Paste", canPaste);
    menu.addItem(menuDelete, "Delete", hasSection);
    menu.addItem(menuDuplicate, "Duplicate", hasSection);

    menu.showMenuAsync(juce::PopupMenu::Options().withTargetScreenArea(juce::Rectangle<int>(screenPosition.x, screenPosition.y, 1, 1)),
                       [safeThis = juce::Component::SafePointer<ArrangementOverviewComponent>(this),
                        sectionIndex,
                        targetTrackIndex,
                        targetStartTick] (int result)
                       {
                           if (safeThis == nullptr || result == 0)
                               return;

                           switch (result)
                           {
                                case menuToolPencil:
                                    if (safeThis->toolModeChangeCallback != nullptr)
                                        safeThis->toolModeChangeCallback(EditorToolMode::pencil);
                                    safeThis->refreshFromModel();
                                    return;

                                case menuToolSelect:
                                    if (safeThis->toolModeChangeCallback != nullptr)
                                        safeThis->toolModeChangeCallback(EditorToolMode::selection);
                                    safeThis->refreshFromModel();
                                    return;

                                case menuToolGlue:
                                    if (safeThis->toolModeChangeCallback != nullptr)
                                        safeThis->toolModeChangeCallback(EditorToolMode::glue);
                                    safeThis->refreshFromModel();
                                    return;

                                case menuToolEraser:
                                    if (safeThis->toolModeChangeCallback != nullptr)
                                        safeThis->toolModeChangeCallback(EditorToolMode::eraser);
                                    safeThis->refreshFromModel();
                                    return;

                               default:
                                   break;
                           }

                           const bool sectionAvailable = juce::isPositiveAndBelow(sectionIndex,
                                                                                 static_cast<int>(safeThis->project().midiSections.size()));

                           if (result == menuPaste)
                           {
                               auto updatedProject = safeThis->project();
                               int pastedSectionIndex = -1;
                               if (safeThis->pasteClipboardAt(updatedProject, targetTrackIndex, targetStartTick, pastedSectionIndex))
                               {
                                   safeThis->projectWriter(updatedProject, true, "Paste Pattern Clip");
                                   if (safeThis->sectionSelectCallback != nullptr)
                                       safeThis->sectionSelectCallback(pastedSectionIndex, false);
                               }
                               return;
                           }

                           if (!sectionAvailable)
                               return;

                           if (result == menuOpen)
                           {
                               if (safeThis->sectionSelectCallback != nullptr)
                                   safeThis->sectionSelectCallback(sectionIndex, true);
                               return;
                           }

                           if (result == menuCopy)
                           {
                               safeThis->copySectionToClipboard(sectionIndex);
                               return;
                           }

                           auto updatedProject = safeThis->project();
                           const auto section = updatedProject.midiSections[static_cast<size_t>(sectionIndex)];

                           if (result == menuCut)
                           {
                               if (!safeThis->copySectionToClipboard(sectionIndex))
                                   return;

                               updatedProject.midiSections.erase(updatedProject.midiSections.begin() + sectionIndex);
                               safeThis->projectWriter(updatedProject, true, "Cut Pattern Clip");
                               if (safeThis->sectionSelectCallback != nullptr)
                               {
                                   const auto nextIndex = juce::jmin(sectionIndex,
                                                                     static_cast<int>(updatedProject.midiSections.size()) - 1);
                                   safeThis->sectionSelectCallback(nextIndex, false);
                               }
                               return;
                           }

                           if (result == menuDuplicate)
                           {
                               auto duplicate = section;
                               duplicate.startTick = juce::jmax(0,
                                                                section.startTick
                                                                    + safeThis->clipLengthTicks(section, updatedProject));
                               updatedProject.midiSections.insert(updatedProject.midiSections.begin() + sectionIndex + 1,
                                                                  std::move(duplicate));
                               safeThis->projectWriter(updatedProject, true, "Duplicate Pattern Clip");
                               if (safeThis->sectionSelectCallback != nullptr)
                                   safeThis->sectionSelectCallback(sectionIndex + 1, false);
                               return;
                           }

                           if (result == menuDelete)
                           {
                               updatedProject.midiSections.erase(updatedProject.midiSections.begin() + sectionIndex);
                               safeThis->projectWriter(updatedProject, true, "Delete Pattern Clip");
                               if (safeThis->sectionSelectCallback != nullptr)
                               {
                                   const auto nextIndex = juce::jmin(sectionIndex,
                                                                     static_cast<int>(updatedProject.midiSections.size()) - 1);
                                   safeThis->sectionSelectCallback(nextIndex, false);
                               }
                               return;
                           }
                       });
}

bool ArrangementOverviewComponent::markSectionForErase(int sectionIndex)
{
    if (!juce::isPositiveAndBelow(sectionIndex, static_cast<int>(beforeProject.midiSections.size())))
        return false;

    if (std::find(erasedSectionIndices.begin(), erasedSectionIndices.end(), sectionIndex) != erasedSectionIndices.end())
        return false;

    erasedSectionIndices.push_back(sectionIndex);
    rebuildErasePreview();
    return true;
}

void ArrangementOverviewComponent::rebuildErasePreview()
{
    previewProject = beforeProject;

    std::sort(erasedSectionIndices.begin(), erasedSectionIndices.end());
    erasedSectionIndices.erase(std::unique(erasedSectionIndices.begin(), erasedSectionIndices.end()), erasedSectionIndices.end());

    for (auto iter = erasedSectionIndices.rbegin(); iter != erasedSectionIndices.rend(); ++iter)
    {
        if (juce::isPositiveAndBelow(*iter, static_cast<int>(previewProject.midiSections.size())))
            previewProject.midiSections.erase(previewProject.midiSections.begin() + *iter);
    }
}

} // namespace aims
