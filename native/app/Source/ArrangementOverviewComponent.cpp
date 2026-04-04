#include "ArrangementOverviewComponent.h"
#include "UiStyle.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <optional>

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

bool isSupportedAudioSampleFile(const juce::String& path)
{
    const auto extension = juce::File(path).getFileExtension().toLowerCase();
    return extension == ".wav"
        || extension == ".aiff"
        || extension == ".aif"
        || extension == ".flac"
        || extension == ".ogg"
        || extension == ".mp3";
}

bool matchesCommandShortcut(const juce::KeyPress& key, char letter)
{
    if (!key.getModifiers().isCommandDown())
        return false;

    const auto normalisedLetter = juce::CharacterFunctions::toLowerCase(static_cast<wchar_t>(letter));
    const auto normalisedKeyCode = juce::CharacterFunctions::toLowerCase(static_cast<wchar_t>(key.getKeyCode()));
    const auto normalisedTextCharacter = juce::CharacterFunctions::toLowerCase(static_cast<wchar_t>(key.getTextCharacter()));
    return normalisedKeyCode == normalisedLetter || normalisedTextCharacter == normalisedLetter;
}

bool isZoomWheelGesture(const juce::MouseEvent& event, const juce::MouseWheelDetails& wheel)
{
    return (event.mods.isCtrlDown() || event.mods.isCommandDown()) && std::abs(wheel.deltaY) > 0.0001f;
}

bool isLaneHeightWheelGesture(const juce::MouseEvent& event, const juce::MouseWheelDetails& wheel)
{
    return event.mods.isAltDown()
        && !event.mods.isCtrlDown()
        && !event.mods.isCommandDown()
        && std::abs(wheel.deltaY) > 0.0001f;
}

float zoomScaleFromWheelDelta(float deltaY)
{
    return std::pow(1.12f, juce::jlimit(-6.0f, 6.0f, deltaY * 4.0f));
}

float laneHeightDeltaFromWheel(float deltaY)
{
    return juce::jlimit(-8.0f, 8.0f, deltaY * 18.0f);
}

juce::Image createToolMenuIcon(EditorToolMode toolMode)
{
    constexpr int iconSize = 18;
    auto image = juce::Image(juce::Image::ARGB, iconSize, iconSize, true);
    juce::Graphics g(image);

    const auto strokeColour = juce::Colour::fromRGB(229, 233, 240);
    const auto accentColour = [toolMode, strokeColour]()
    {
        switch (toolMode)
        {
            case EditorToolMode::pencil:
                return juce::Colour::fromRGB(126, 213, 140);
            case EditorToolMode::selection:
                return juce::Colour::fromRGB(120, 212, 255);
            case EditorToolMode::scissors:
                return juce::Colour::fromRGB(255, 171, 145);
            case EditorToolMode::glue:
                return juce::Colour::fromRGB(255, 209, 102);
            case EditorToolMode::eraser:
                return juce::Colour::fromRGB(255, 128, 128);
        }

        return strokeColour;
    }();

    g.setColour(accentColour.withAlpha(0.15f));
    g.fillRoundedRectangle(1.0f, 1.0f, static_cast<float>(iconSize - 2), static_cast<float>(iconSize - 2), 4.0f);

    switch (toolMode)
    {
        case EditorToolMode::pencil:
        {
            juce::Path shaft;
            shaft.startNewSubPath(4.5f, 13.5f);
            shaft.lineTo(10.8f, 7.2f);
            g.setColour(strokeColour);
            g.strokePath(shaft, juce::PathStrokeType(2.0f, juce::PathStrokeType::curved, juce::PathStrokeType::rounded));

            juce::Path body;
            body.addRoundedRectangle(5.2f, 11.0f, 6.8f, 2.4f, 1.0f);
            auto rotation = juce::AffineTransform::rotation(juce::degreesToRadians(-45.0f), 5.2f, 12.2f);
            body.applyTransform(rotation);
            g.fillPath(body);

            juce::Path tip;
            tip.addTriangle(11.8f, 6.2f, 14.3f, 3.7f, 13.0f, 7.7f);
            g.setColour(accentColour.brighter(0.15f));
            g.fillPath(tip);
            break;
        }

        case EditorToolMode::selection:
        {
            g.setColour(accentColour);
            const float left = 4.0f;
            const float top = 4.0f;
            const float right = 12.0f;
            const float bottom = 12.0f;
            const float segment = 2.0f;
            const float strokeWidth = 1.5f;
            g.drawLine(left, top, left + segment, top, strokeWidth);
            g.drawLine(left + 3.0f, top, right - 3.0f, top, strokeWidth);
            g.drawLine(right - segment, top, right, top, strokeWidth);
            g.drawLine(left, bottom, left + segment, bottom, strokeWidth);
            g.drawLine(left + 3.0f, bottom, right - 3.0f, bottom, strokeWidth);
            g.drawLine(right - segment, bottom, right, bottom, strokeWidth);
            g.drawLine(left, top, left, top + segment, strokeWidth);
            g.drawLine(left, top + 3.0f, left, bottom - 3.0f, strokeWidth);
            g.drawLine(left, bottom - segment, left, bottom, strokeWidth);
            g.drawLine(right, top, right, top + segment, strokeWidth);
            g.drawLine(right, top + 3.0f, right, bottom - 3.0f, strokeWidth);
            g.drawLine(right, bottom - segment, right, bottom, strokeWidth);

            juce::Path cursor;
            cursor.startNewSubPath(9.8f, 9.2f);
            cursor.lineTo(14.2f, 14.8f);
            cursor.lineTo(11.8f, 14.4f);
            cursor.lineTo(10.9f, 16.2f);
            cursor.lineTo(9.7f, 15.7f);
            cursor.lineTo(10.6f, 13.8f);
            cursor.lineTo(8.9f, 13.5f);
            cursor.closeSubPath();
            g.setColour(strokeColour);
            g.fillPath(cursor);
            break;
        }

        case EditorToolMode::scissors:
        {
            juce::Path upperBlade;
            upperBlade.startNewSubPath(6.3f, 7.2f);
            upperBlade.lineTo(13.8f, 3.8f);
            g.setColour(strokeColour);
            g.strokePath(upperBlade, juce::PathStrokeType(1.9f, juce::PathStrokeType::curved, juce::PathStrokeType::rounded));

            juce::Path lowerBlade;
            lowerBlade.startNewSubPath(6.4f, 10.6f);
            lowerBlade.lineTo(13.9f, 14.2f);
            g.strokePath(lowerBlade, juce::PathStrokeType(1.9f, juce::PathStrokeType::curved, juce::PathStrokeType::rounded));

            g.setColour(accentColour);
            g.drawEllipse(3.2f, 5.8f, 4.1f, 4.1f, 1.6f);
            g.drawEllipse(3.2f, 9.7f, 4.1f, 4.1f, 1.6f);

            juce::Path hinge;
            hinge.addEllipse(7.0f, 8.2f, 2.2f, 2.2f);
            g.fillPath(hinge);
            break;
        }

        case EditorToolMode::glue:
        {
            g.setColour(strokeColour);
            juce::Path leftLoop;
            leftLoop.addRoundedRectangle(3.8f, 6.0f, 5.8f, 4.6f, 2.2f);
            leftLoop.applyTransform(juce::AffineTransform::rotation(juce::degreesToRadians(-28.0f), 6.7f, 8.3f));
            g.strokePath(leftLoop, juce::PathStrokeType(1.7f, juce::PathStrokeType::curved, juce::PathStrokeType::rounded));

            juce::Path rightLoop;
            rightLoop.addRoundedRectangle(8.6f, 7.3f, 5.8f, 4.6f, 2.2f);
            rightLoop.applyTransform(juce::AffineTransform::rotation(juce::degreesToRadians(-28.0f), 11.5f, 9.6f));
            g.strokePath(rightLoop, juce::PathStrokeType(1.7f, juce::PathStrokeType::curved, juce::PathStrokeType::rounded));

            juce::Path join;
            join.startNewSubPath(7.2f, 10.4f);
            join.lineTo(10.9f, 7.9f);
            g.setColour(accentColour);
            g.strokePath(join, juce::PathStrokeType(2.0f, juce::PathStrokeType::curved, juce::PathStrokeType::rounded));
            break;
        }

        case EditorToolMode::eraser:
        {
            juce::Path eraser;
            eraser.startNewSubPath(5.0f, 12.4f);
            eraser.lineTo(9.3f, 7.2f);
            eraser.lineTo(13.8f, 11.5f);
            eraser.lineTo(9.4f, 15.7f);
            eraser.closeSubPath();
            g.setColour(strokeColour);
            g.fillPath(eraser);

            juce::Path accent;
            accent.startNewSubPath(7.6f, 9.2f);
            accent.lineTo(11.3f, 13.0f);
            g.setColour(accentColour);
            g.strokePath(accent, juce::PathStrokeType(1.7f, juce::PathStrokeType::curved, juce::PathStrokeType::rounded));

            juce::Path dust;
            dust.startNewSubPath(10.0f, 14.6f);
            dust.lineTo(13.9f, 14.6f);
            g.strokePath(dust, juce::PathStrokeType(1.3f, juce::PathStrokeType::curved, juce::PathStrokeType::rounded));
            break;
        }
    }

    return image;
}

juce::MouseCursor createGlueToolCursor()
{
    constexpr int cursorSize = 24;
    auto image = juce::Image(juce::Image::ARGB, cursorSize, cursorSize, true);
    juce::Graphics g(image);

    const auto bottleColour = juce::Colour::fromRGB(244, 245, 247);
    const auto accentColour = juce::Colour::fromRGB(255, 209, 102);
    const auto outlineColour = juce::Colour::fromRGBA(10, 12, 16, 180);

    juce::Path body;
    body.startNewSubPath(9.0f, 3.0f);
    body.lineTo(14.5f, 3.0f);
    body.lineTo(14.5f, 7.0f);
    body.lineTo(17.4f, 10.2f);
    body.lineTo(15.8f, 12.2f);
    body.lineTo(17.0f, 17.8f);
    body.lineTo(10.0f, 20.8f);
    body.lineTo(6.6f, 13.4f);
    body.lineTo(9.0f, 11.9f);
    body.lineTo(9.0f, 3.0f);
    body.closeSubPath();

    g.setColour(outlineColour);
    g.fillPath(body, juce::AffineTransform::translation(1.0f, 1.0f));
    g.setColour(bottleColour);
    g.fillPath(body);
    g.setColour(outlineColour.withAlpha(0.55f));
    g.strokePath(body, juce::PathStrokeType(1.1f, juce::PathStrokeType::curved, juce::PathStrokeType::rounded));

    juce::Path label;
    label.addRoundedRectangle(8.6f, 10.1f, 5.0f, 5.0f, 1.5f);
    g.setColour(accentColour.withAlpha(0.95f));
    g.fillPath(label);

    juce::Path nozzle;
    nozzle.addTriangle(12.5f, 18.2f, 16.6f, 20.6f, 11.2f, 21.4f);
    g.setColour(accentColour.brighter(0.12f));
    g.fillPath(nozzle);

    juce::Path glueDrop;
    glueDrop.addEllipse(17.2f, 19.6f, 2.8f, 2.8f);
    g.setColour(accentColour.withAlpha(0.8f));
    g.fillPath(glueDrop);

    return juce::MouseCursor(image, 16, 20);
}

juce::MouseCursor createScissorsToolCursor()
{
    constexpr int cursorSize = 24;
    auto image = juce::Image(juce::Image::ARGB, cursorSize, cursorSize, true);
    juce::Graphics g(image);

    const auto handleColour = juce::Colour::fromRGB(255, 171, 145);
    const auto bladeColour = juce::Colour::fromRGB(244, 245, 247);
    const auto outlineColour = juce::Colour::fromRGBA(10, 12, 16, 180);

    juce::Path upperBlade;
    upperBlade.startNewSubPath(8.0f, 10.2f);
    upperBlade.lineTo(18.6f, 4.3f);
    g.setColour(outlineColour.withAlpha(0.45f));
    g.strokePath(upperBlade, juce::PathStrokeType(3.5f, juce::PathStrokeType::curved, juce::PathStrokeType::rounded));
    g.setColour(bladeColour);
    g.strokePath(upperBlade, juce::PathStrokeType(2.2f, juce::PathStrokeType::curved, juce::PathStrokeType::rounded));

    juce::Path lowerBlade;
    lowerBlade.startNewSubPath(8.0f, 11.2f);
    lowerBlade.lineTo(18.7f, 17.4f);
    g.setColour(outlineColour.withAlpha(0.45f));
    g.strokePath(lowerBlade, juce::PathStrokeType(3.5f, juce::PathStrokeType::curved, juce::PathStrokeType::rounded));
    g.setColour(bladeColour);
    g.strokePath(lowerBlade, juce::PathStrokeType(2.2f, juce::PathStrokeType::curved, juce::PathStrokeType::rounded));

    g.setColour(handleColour);
    g.fillEllipse(2.8f, 6.4f, 5.8f, 5.8f);
    g.fillEllipse(2.8f, 12.0f, 5.8f, 5.8f);
    g.setColour(outlineColour.withAlpha(0.6f));
    g.drawEllipse(2.8f, 6.4f, 5.8f, 5.8f, 1.1f);
    g.drawEllipse(2.8f, 12.0f, 5.8f, 5.8f, 1.1f);

    g.setColour(outlineColour.withAlpha(0.75f));
    g.fillEllipse(7.0f, 9.2f, 2.6f, 2.6f);

    return juce::MouseCursor(image, 18, 11);
}

const juce::MouseCursor& glueToolCursor()
{
    static const auto cursor = createGlueToolCursor();
    return cursor;
}

const juce::MouseCursor& scissorsToolCursor()
{
    static const auto cursor = createScissorsToolCursor();
    return cursor;
}

std::optional<double> interpolatedControllerValueAtTick(const AutomationLane& lane, int tick)
{
    const AutomationPoint* previous = nullptr;
    const AutomationPoint* next = nullptr;

    for (const auto& point : lane.points)
    {
        if (point.tick == tick)
            return point.value;

        if (point.tick < tick)
        {
            if (previous == nullptr || previous->tick < point.tick)
                previous = &point;
        }
        else if (point.tick > tick)
        {
            if (next == nullptr || next->tick > point.tick)
                next = &point;
        }
    }

    if (previous != nullptr && next != nullptr)
    {
        const auto tickSpan = juce::jmax(1, next->tick - previous->tick);
        const auto alpha = static_cast<double>(tick - previous->tick) / static_cast<double>(tickSpan);
        return juce::jmap(alpha, previous->value, next->value);
    }

    if (previous != nullptr)
        return previous->value;

    return std::nullopt;
}

void appendAutomationPointIfMissing(std::vector<AutomationPoint>& points, int tick, double value)
{
    const auto found = std::find_if(points.begin(),
                                    points.end(),
                                    [tick] (const AutomationPoint& point)
                                    {
                                        return point.tick == tick;
                                    });
    if (found != points.end())
    {
        found->value = value;
        return;
    }

    AutomationPoint point;
    point.tick = tick;
    point.value = value;
    points.push_back(point);
}

double sampleClipVisibleStartRatio(const SampleClip& clip)
{
    if (clip.sourceFileDurationSec <= 1.0e-6)
        return 0.0;

    return juce::jlimit(0.0, 1.0, clip.sourceOffsetSec / clip.sourceFileDurationSec);
}

double sampleClipVisibleEndRatio(const SampleClip& clip)
{
    if (clip.sourceFileDurationSec <= 1.0e-6)
        return 1.0;

    return juce::jlimit(0.0,
                        1.0,
                        (clip.sourceOffsetSec + juce::jmax(0.0, clip.durationSec)) / clip.sourceFileDurationSec);
}

juce::String sampleClipDisplayName(const SampleClip& clip)
{
    return juce::File(clip.path).getFileNameWithoutExtension();
}

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

bool tickRangesMergeableWithGap(int startTick,
                                int endTick,
                                int otherStartTick,
                                int otherEndTick,
                                int allowedGapTicks)
{
    if (tickRangesTouchOrOverlap(startTick, endTick, otherStartTick, otherEndTick))
        return true;

    const auto gapTicks = startTick < otherStartTick
        ? otherStartTick - endTick
        : startTick - otherEndTick;
    return gapTicks <= juce::jmax(0, allowedGapTicks);
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

const AutomationLane* findTrackAutomationLane(const TrackState& track, const juce::String& target)
{
    for (const auto& lane : track.automationLanes)
    {
        if (lane.target.equalsIgnoreCase(target))
            return &lane;
    }

    return nullptr;
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
    if (!juce::isPositiveAndBelow(selectedSampleClipIndex, static_cast<int>(project().sampleClips.size())))
        selectedSampleClipIndex = -1;
    draggedSectionIndices.clear();
    draggedSectionBaseStartTicks.clear();
    draggedSectionBaseTrackIndices.clear();
    draggedSampleClipIndex = -1;
    if (!previewActive)
        updateContentSize();
    repaint();
}

void ArrangementOverviewComponent::setHorizontalZoom(float pixelsPerBarIn)
{
    const auto clamped = juce::jlimit(48.0f, 640.0f, pixelsPerBarIn);
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

void ArrangementOverviewComponent::setZoomChangedCallback(ZoomChangedCallback zoomChangedCallbackIn)
{
    zoomChangedCallback = std::move(zoomChangedCallbackIn);
}

void ArrangementOverviewComponent::setLaneHeightChangedCallback(LaneHeightChangedCallback laneHeightChangedCallbackIn)
{
    laneHeightChangedCallback = std::move(laneHeightChangedCallbackIn);
}

void ArrangementOverviewComponent::setSampleClipStemSeparationCallback(SampleClipStemSeparationCallback callbackIn)
{
    sampleClipStemSeparationCallback = std::move(callbackIn);
}

void ArrangementOverviewComponent::setSampleFileDropCallback(SampleFileDropCallback callbackIn)
{
    sampleFileDropCallback = std::move(callbackIn);
}

bool ArrangementOverviewComponent::selectAllSections()
{
    const auto sectionCount = static_cast<int>(project().midiSections.size());
    if (sectionCount <= 0)
        return false;

    selectedSectionIndices.clear();
    selectedSectionIndices.reserve(sectionCount);
    for (int sectionIndex = 0; sectionIndex < sectionCount; ++sectionIndex)
        selectedSectionIndices.push_back(sectionIndex);

    repaint();
    if (sectionSelectCallback != nullptr)
    {
        if (sectionCount == 1)
            sectionSelectCallback(0, false);
        else
            sectionSelectCallback(-1, false);
    }

    return true;
}

std::vector<int> ArrangementOverviewComponent::getSelectedSectionIndices() const
{
    return selectedSectionIndices;
}

void ArrangementOverviewComponent::paint(juce::Graphics& g)
{
    g.fillAll(kBackground);

    const auto& state = displayedProject();
    const auto headerWidth = laneHeaderWidth();
    const auto topHeight = rulerHeight();
    const auto selectedSectionIndex = selectedSectionGetter != nullptr ? selectedSectionGetter() : -1;
    const auto projectBarTicks = ticksPerBar(state);
    const auto projectBeatTicks = ticksPerTimeSignatureBeat(state);

    g.setColour(kHeader);
    g.fillRect(0.0f, 0.0f, static_cast<float>(getWidth()), topHeight);
    g.fillRect(0.0f, 0.0f, headerWidth, static_cast<float>(getHeight()));

    float currentY = topHeight;
    for (int trackIndex = 0; trackIndex < static_cast<int>(state.tracks.size()); ++trackIndex)
    {
        const auto& track = state.tracks[static_cast<size_t>(trackIndex)];
        const auto baseColour = trackDisplayColour(track, trackIndex);
        auto clipRow = juce::Rectangle<float>(headerWidth,
                                              currentY,
                                              static_cast<float>(getWidth()) - headerWidth,
                                              laneHeight());

        g.setColour((trackIndex % 2) == 0 ? kLaneEven : kLaneOdd);
        g.fillRect(clipRow);
        g.setColour(baseColour.withAlpha(0.10f));
        g.fillRect(clipRow);

        const auto label = track.name.isNotEmpty() ? track.name : "Track " + juce::String(trackIndex + 1);
        g.setColour(juce::Colour::fromRGB(182, 190, 201));
        g.setFont(ui::font());
        g.drawText(label,
                   6,
                   juce::roundToInt(currentY),
                   juce::roundToInt(headerWidth) - 8,
                   juce::roundToInt(laneHeight()),
                   juce::Justification::centredLeft,
                   true);

        g.setColour(juce::Colour::fromRGB(46, 52, 64));
        g.drawHorizontalLine(juce::roundToInt(currentY), 0.0f, static_cast<float>(getWidth()));

        currentY += laneHeight();
        const auto visibleTargets = visibleArrangementAutomationTargets(track);
        for (int laneIndex = 0; laneIndex < visibleTargets.size(); ++laneIndex)
        {
            const auto target = visibleTargets[laneIndex];
            const auto* lane = findTrackAutomationLane(track, target);
            auto laneRow = juce::Rectangle<float>(headerWidth,
                                                  currentY,
                                                  static_cast<float>(getWidth()) - headerWidth,
                                                  laneHeight());

            auto laneFill = ((trackIndex + laneIndex) % 2) == 0
                ? kLaneOdd.brighter(0.03f)
                : kLaneEven.darker(0.03f);
            laneFill = laneFill.interpolatedWith(baseColour.withAlpha(0.16f), 0.35f);
            if (lane != nullptr && !lane->enabled)
                laneFill = laneFill.darker(0.18f);

            g.setColour(laneFill);
            g.fillRect(laneRow);

            auto laneLabel = automationTargetLabel(track, target);
            if (lane != nullptr && !lane->enabled)
                laneLabel << " (Byp)";

            g.setColour(juce::Colour::fromRGB(154, 163, 179));
            g.setFont(ui::tinyFont());
            g.drawText(laneLabel,
                       14,
                       juce::roundToInt(currentY),
                       juce::roundToInt(headerWidth) - 18,
                       juce::roundToInt(laneHeight()),
                       juce::Justification::centredLeft,
                       true);

            g.setColour(juce::Colour::fromRGB(40, 46, 58));
            g.drawHorizontalLine(juce::roundToInt(currentY), 0.0f, static_cast<float>(getWidth()));
            currentY += laneHeight();
        }

        g.setColour(juce::Colour::fromRGB(46, 52, 64));
        g.drawHorizontalLine(juce::roundToInt(currentY), 0.0f, static_cast<float>(getWidth()));
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
        const auto clipIndex = static_cast<int>(&clip - state.sampleClips.data());
        auto rect = clipRect(clip).reduced(1.5f, 4.0f);
        if (!juce::isPositiveAndBelow(clip.trackIndex, static_cast<int>(state.tracks.size())))
            continue;

        const auto baseColour = trackDisplayColour(state.tracks[static_cast<size_t>(clip.trackIndex)], clip.trackIndex);
        auto fill = baseColour.withAlpha(0.66f);
        if (previewActive && clipIndex == draggedSampleClipIndex)
            fill = fill.brighter(0.12f);

        g.setColour(fill);
        g.fillRoundedRectangle(rect, 5.0f);

        const auto isSelected = clipIndex == selectedSampleClipIndex;
        g.setColour(isSelected ? juce::Colours::white : baseColour.darker(0.55f));
        g.drawRoundedRectangle(rect, 5.0f, isSelected ? 2.0f : 1.2f);

        if (!clip.waveformPreview.empty() && rect.getWidth() > 12.0f && rect.getHeight() > 10.0f)
        {
            juce::Path waveform;
            const auto midY = rect.getCentreY();
            const auto amplitude = juce::jmax(4.0f, rect.getHeight() * 0.5f - 6.0f);
            const auto startRatio = sampleClipVisibleStartRatio(clip);
            const auto endRatio = juce::jmax(startRatio, sampleClipVisibleEndRatio(clip));
            const auto previewPointCount = static_cast<int>(clip.waveformPreview.size());
            waveform.startNewSubPath(rect.getX(), midY);
            for (int pointIndex = 0; pointIndex < previewPointCount; ++pointIndex)
            {
                const auto alpha = previewPointCount > 1
                    ? static_cast<double>(pointIndex) / static_cast<double>(previewPointCount - 1)
                    : 0.0;
                const auto previewAlpha = juce::jmap(alpha, startRatio, endRatio);
                const auto previewIndex = juce::jlimit(0,
                                                       previewPointCount - 1,
                                                       juce::roundToInt(previewAlpha * static_cast<double>(previewPointCount - 1)));
                const auto x = rect.getX()
                    + (rect.getWidth() * static_cast<float>(pointIndex))
                        / static_cast<float>(juce::jmax(1, previewPointCount - 1));
                const auto y = midY
                    - (juce::jlimit(0.0f, 1.0f, clip.waveformPreview[static_cast<size_t>(previewIndex)]) * amplitude);
                waveform.lineTo(x, y);
            }

            g.setColour(trackTextColour(baseColour).withAlpha(0.82f));
            g.strokePath(waveform, juce::PathStrokeType(1.2f));
        }

        g.setColour(trackTextColour(baseColour));
        g.setFont(ui::font());
        g.drawText(sampleClipDisplayName(clip),
                   rect.toNearestInt().reduced(6, 0),
                   juce::Justification::centredLeft,
                   true);
    }

    if (sampleFileDragActive && juce::isPositiveAndBelow(sampleFileDragTrackIndex, static_cast<int>(state.tracks.size())))
    {
        auto dropRow = juce::Rectangle<float>(headerWidth,
                                              trackClipTop(sampleFileDragTrackIndex, state),
                                              static_cast<float>(getWidth()) - headerWidth,
                                              laneHeight());
        const auto trackColour = trackDisplayColour(state.tracks[static_cast<size_t>(sampleFileDragTrackIndex)],
                                                    sampleFileDragTrackIndex);
        g.setColour(trackColour.withAlpha(0.18f));
        g.fillRect(dropRow);
        g.setColour(trackColour.brighter(0.2f));
        g.drawRect(dropRow.toNearestInt(), 2);

        const auto dropX = tickToX(sampleFileDragStartTick);
        g.setColour(trackColour.brighter(0.35f));
        g.drawVerticalLine(juce::roundToInt(dropX),
                           dropRow.getY(),
                           dropRow.getBottom());

        auto badge = juce::Rectangle<float>(dropX + 6.0f,
                                            dropRow.getY() + 6.0f,
                                            116.0f,
                                            juce::jmin(22.0f, dropRow.getHeight() - 10.0f));
        g.setColour(kHeader.brighter(0.18f).withAlpha(0.92f));
        g.fillRoundedRectangle(badge, 4.0f);
        g.setColour(trackTextColour(trackColour));
        g.setFont(ui::font());
        g.drawText("Drop Sample",
                   badge.toNearestInt().reduced(8, 0),
                   juce::Justification::centredLeft,
                   true);
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

    for (int trackIndex = 0; trackIndex < static_cast<int>(state.tracks.size()); ++trackIndex)
    {
        const auto& track = state.tracks[static_cast<size_t>(trackIndex)];
        const auto visibleTargets = visibleArrangementAutomationTargets(track);
        if (visibleTargets.isEmpty())
            continue;

        auto laneTop = trackClipTop(trackIndex, state) + laneHeight();
        const auto baseColour = trackDisplayColour(track, trackIndex);
        const auto strokeColour = baseColour.brighter(0.18f);
        const auto defaultLineColour = juce::Colour::fromRGB(100, 116, 139);

        for (const auto& target : visibleTargets)
        {
            const auto* lane = findTrackAutomationLane(track, target);
            if (lane == nullptr || lane->points.empty())
            {
                laneTop += laneHeight();
                continue;
            }

            auto rowRect = juce::Rectangle<float>(headerWidth,
                                                  laneTop,
                                                  static_cast<float>(getWidth()) - headerWidth,
                                                  laneHeight()).reduced(0.0f, 4.0f);
            if (rowRect.getWidth() <= 8.0f || rowRect.getHeight() <= 6.0f)
            {
                laneTop += laneHeight();
                continue;
            }

            const auto [minimum, maximum] = automationTargetBounds(track, target);
            const auto valueSpan = juce::jmax(1.0e-6, maximum - minimum);
            const auto defaultValue = automationTargetDefaultValue(track, target);
            const auto valueToY = [&rowRect, minimum, valueSpan] (double value)
            {
                const auto clamped = juce::jlimit(minimum, minimum + valueSpan, value);
                const auto alpha = static_cast<float>((clamped - minimum) / valueSpan);
                return rowRect.getBottom() - (alpha * rowRect.getHeight());
            };

            const auto defaultY = valueToY(defaultValue);
            juce::Path defaultPath;
            defaultPath.startNewSubPath(rowRect.getX(), defaultY);
            defaultPath.lineTo(rowRect.getRight(), defaultY);
            g.setColour(defaultLineColour.withAlpha(lane->enabled ? 0.55f : 0.32f));
            g.strokePath(defaultPath,
                         juce::PathStrokeType(1.0f, juce::PathStrokeType::curved, juce::PathStrokeType::rounded));

            juce::Path automationPath;
            const auto firstPointX = tickToX(lane->points.front().tick);
            const auto firstPointY = valueToY(lane->points.front().value);
            if (lane->points.front().tick > 0)
            {
                automationPath.startNewSubPath(rowRect.getX(), defaultY);
                automationPath.lineTo(firstPointX, defaultY);
            }
            else
            {
                automationPath.startNewSubPath(firstPointX, firstPointY);
            }

            for (const auto& point : lane->points)
                automationPath.lineTo(tickToX(point.tick), valueToY(point.value));

            automationPath.lineTo(rowRect.getRight(), valueToY(lane->points.back().value));

            g.setColour(strokeColour.withAlpha(lane->enabled ? 0.95f : 0.40f));
            g.strokePath(automationPath,
                         juce::PathStrokeType(1.6f, juce::PathStrokeType::curved, juce::PathStrokeType::rounded));

            for (const auto& point : lane->points)
            {
                auto pointBounds = juce::Rectangle<float>(tickToX(point.tick) - 3.0f,
                                                          valueToY(point.value) - 3.0f,
                                                          6.0f,
                                                          6.0f);
                g.setColour(baseColour.brighter(0.28f).withAlpha(lane->enabled ? 0.92f : 0.45f));
                g.fillEllipse(pointBounds);
                g.setColour(juce::Colour::fromRGB(12, 16, 22).withAlpha(lane->enabled ? 0.70f : 0.30f));
                g.drawEllipse(pointBounds, 1.0f);
            }

            laneTop += laneHeight();
        }
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

    int hitTrackIndex = -1;
    juce::String hitAutomationTarget;
    hitTestTrackLane(event.position.y, hitTrackIndex, hitAutomationTarget);

    juce::String hitEdge;
    const auto hitSampleClipIndex = hitTestSampleClip(event.position, hitEdge);
    const auto hitSectionIndex = hitTestMidiSection(event.position, hitEdge);
    contextMenuTrackIndex = hitTrackIndex >= 0 ? hitTrackIndex : yToTrackIndex(event.position.y);
    contextMenuTick = xToSnapStartTick(event.position.x);
    if (event.mods.isPopupMenu())
    {
        if (hitSectionIndex >= 0 && sectionSelectCallback != nullptr)
            sectionSelectCallback(hitSectionIndex, false);
        if (hitSampleClipIndex >= 0)
            selectedSampleClipIndex = hitSampleClipIndex;
        showSectionContextMenu(hitSectionIndex,
                               hitSampleClipIndex,
                               event.getScreenPosition(),
                               contextMenuTrackIndex,
                               contextMenuTick);
        return;
    }

    previewActive = false;
    previewDirty = false;
    dragMode = DragMode::none;
    draggedSectionIndex = -1;
    draggedSampleClipIndex = -1;
    draggedSectionIndices.clear();
    draggedSectionBaseStartTicks.clear();
    draggedSectionBaseTrackIndices.clear();
    draggedSampleClipBase = {};
    dragOffsetTick = 0;
    dragAnchorStartTick = 0;
    dragAnchorTrackIndex = 0;
    createStartTick = 0;
    resizeAnchorStartTick = 0;
    resizeAnchorLengthTicks = ticksPerBar(project());
    dragCreatesCopy = false;
    marqueeStart = {};
    marqueeRect = {};
    erasedSampleClipIndices.clear();

    if (hitSampleClipIndex >= 0)
    {
        selectedSampleClipIndex = hitSampleClipIndex;
        selectedSectionIndices.clear();
        if (sectionSelectCallback != nullptr)
            sectionSelectCallback(-1, false);

        if (event.getNumberOfClicks() > 1)
            return;

        if (toolMode == EditorToolMode::eraser && event.mods.isLeftButtonDown())
        {
            beforeProject = project();
            previewProject = beforeProject;
            previewActive = true;
            previewDirty = false;
            dragMode = DragMode::eraseSections;
            erasedSectionIndices.clear();
            erasedSampleClipIndices.clear();
            if (markSampleClipForErase(hitSampleClipIndex))
                previewDirty = true;
            repaint();
            return;
        }

        if (toolMode == EditorToolMode::scissors && event.mods.isLeftButtonDown())
        {
            const auto splitTick = xToSnapStartTick(event.position.x);
            if (splitSampleClipAtTick(hitSampleClipIndex, splitTick))
                refreshFromModel();
            return;
        }

        if (toolMode == EditorToolMode::glue && event.mods.isLeftButtonDown())
            return;

        beforeProject = project();
        previewProject = beforeProject;
        previewActive = true;
        previewDirty = false;
        draggedSampleClipIndex = hitSampleClipIndex;
        draggedSampleClipBase = beforeProject.sampleClips[static_cast<size_t>(hitSampleClipIndex)];
        dragOffsetTick = juce::jmax(0,
                                    xToSnapStartTick(event.position.x)
                                        - secondsToTick(beforeProject, draggedSampleClipBase.startSec));
        resizeAnchorStartTick = secondsToTick(beforeProject, draggedSampleClipBase.startSec);
        resizeAnchorLengthTicks = clipLengthTicks(draggedSampleClipBase, beforeProject);

        const bool duplicateDragRequested = event.mods.isLeftButtonDown()
            && hitEdge == "body"
            && (event.mods.isAltDown() || event.mods.isCtrlDown() || event.mods.isCommandDown());
        if (duplicateDragRequested)
        {
            previewProject.sampleClips.push_back(draggedSampleClipBase);
            draggedSampleClipIndex = static_cast<int>(previewProject.sampleClips.size()) - 1;
            selectedSampleClipIndex = draggedSampleClipIndex;
            dragCreatesCopy = true;
        }

        if (hitEdge == "left")
            dragMode = DragMode::resizeSampleLeft;
        else if (hitEdge == "right")
            dragMode = DragMode::resizeSampleRight;
        else
            dragMode = DragMode::moveSampleClip;

        repaint();
        return;
    }

    if (hitSectionIndex >= 0)
    {
        selectedSampleClipIndex = -1;
        const bool preserveGlueSelection = toolMode == EditorToolMode::glue
            && event.mods.isLeftButtonDown()
            && selectedSectionIndices.size() > 1
            && std::find(selectedSectionIndices.begin(), selectedSectionIndices.end(), hitSectionIndex) != selectedSectionIndices.end();
        const bool clickedSelected = std::find(selectedSectionIndices.begin(),
                                               selectedSectionIndices.end(),
                                               hitSectionIndex) != selectedSectionIndices.end();
        const bool preserveDragSelection = hitEdge == "body"
            && event.mods.isLeftButtonDown()
            && clickedSelected
            && selectedSectionIndices.size() > 1;

        if (!preserveGlueSelection && !preserveDragSelection)
        {
            selectedSectionIndices.clear();
            selectedSectionIndices.push_back(hitSectionIndex);
            if (sectionSelectCallback != nullptr)
                sectionSelectCallback(hitSectionIndex, event.getNumberOfClicks() > 1);
        }

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

        if (toolMode == EditorToolMode::scissors && event.mods.isLeftButtonDown())
        {
            const auto splitTick = xToSnapStartTick(event.position.x);
            if (splitSectionAtTick(hitSectionIndex, splitTick))
                refreshFromModel();
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
        draggedSectionIndices = preserveDragSelection ? selectedSectionIndices : std::vector<int>{ hitSectionIndex };
        std::sort(draggedSectionIndices.begin(), draggedSectionIndices.end());
        draggedSectionIndices.erase(std::unique(draggedSectionIndices.begin(), draggedSectionIndices.end()),
                                    draggedSectionIndices.end());
        draggedSectionBaseStartTicks.clear();
        draggedSectionBaseTrackIndices.clear();
        draggedSectionBaseStartTicks.reserve(draggedSectionIndices.size());
        draggedSectionBaseTrackIndices.reserve(draggedSectionIndices.size());
        for (const auto sectionIndex : draggedSectionIndices)
        {
            if (!juce::isPositiveAndBelow(sectionIndex, static_cast<int>(state.midiSections.size())))
                continue;

            const auto& draggedSection = state.midiSections[static_cast<size_t>(sectionIndex)];
            draggedSectionBaseStartTicks.push_back(draggedSection.startTick);
            draggedSectionBaseTrackIndices.push_back(draggedSection.trackIndex);
        }
        dragAnchorStartTick = sourceSection.startTick;
        dragAnchorTrackIndex = sourceSection.trackIndex;
        const bool duplicateDragRequested = event.mods.isLeftButtonDown()
            && hitEdge == "body"
            && (event.mods.isAltDown() || event.mods.isCtrlDown() || event.mods.isCommandDown());

        if (duplicateDragRequested)
        {
            const auto anchorPosition = std::find(draggedSectionIndices.begin(),
                                                  draggedSectionIndices.end(),
                                                  hitSectionIndex);
            const auto anchorOffset = anchorPosition != draggedSectionIndices.end()
                ? static_cast<int>(std::distance(draggedSectionIndices.begin(), anchorPosition))
                : -1;
            std::vector<int> duplicateIndices;
            duplicateIndices.reserve(draggedSectionIndices.size());
            for (const auto sectionIndex : draggedSectionIndices)
            {
                if (!juce::isPositiveAndBelow(sectionIndex, static_cast<int>(state.midiSections.size())))
                    continue;

                auto duplicateSection = state.midiSections[static_cast<size_t>(sectionIndex)];
                previewProject.midiSections.push_back(std::move(duplicateSection));
                duplicateIndices.push_back(static_cast<int>(previewProject.midiSections.size()) - 1);
            }

            if (!duplicateIndices.empty())
            {
                draggedSectionIndices = std::move(duplicateIndices);
                if (anchorOffset >= 0 && anchorOffset < static_cast<int>(draggedSectionIndices.size()))
                    draggedSectionIndex = draggedSectionIndices[static_cast<size_t>(anchorOffset)];
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
        selectedSampleClipIndex = -1;
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
        selectedSampleClipIndex = -1;
        if (sectionSelectCallback != nullptr)
            sectionSelectCallback(-1, false);
        updateCursorForPosition(event.position);
        return;
    }

    if (hitAutomationTarget.isNotEmpty())
    {
        selectedSectionIndices.clear();
        selectedSampleClipIndex = -1;
        if (sectionSelectCallback != nullptr)
            sectionSelectCallback(-1, false);
        updateCursorForPosition(event.position);
        return;
    }

    const auto trackIndex = hitTrackIndex;
    if (!juce::isPositiveAndBelow(trackIndex, static_cast<int>(project().tracks.size())))
        return;

    if (project().tracks[static_cast<size_t>(trackIndex)].trackType.equalsIgnoreCase("sample"))
    {
        selectedSectionIndices.clear();
        selectedSampleClipIndex = -1;
        if (sectionSelectCallback != nullptr)
            sectionSelectCallback(-1, false);
        repaint();
        return;
    }

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
        const auto hitSampleClipIndex = hitTestSampleClip(beforeProject, event.position, hitEdge);
        if (hitSectionIndex >= 0 && markSectionForErase(hitSectionIndex))
        {
            previewDirty = true;
            repaint();
        }
        if (hitSampleClipIndex >= 0 && markSampleClipForErase(hitSampleClipIndex))
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

    if (dragMode == DragMode::moveSampleClip
        || dragMode == DragMode::resizeSampleLeft
        || dragMode == DragMode::resizeSampleRight)
    {
        if (!juce::isPositiveAndBelow(draggedSampleClipIndex, static_cast<int>(previewProject.sampleClips.size())))
            return;

        if (!dragCreatesCopy)
            previewProject = beforeProject;

        auto& clip = previewProject.sampleClips[static_cast<size_t>(draggedSampleClipIndex)];
        const auto minimumSliceTicks = arrangementSnapTickLength(beforeProject);

        if (dragMode == DragMode::moveSampleClip)
        {
            const auto requestedStartTick = juce::jmax(0, xToSnapStartTick(event.position.x) - dragOffsetTick);
            const auto requestedTrackIndex = juce::jlimit(0,
                                                          juce::jmax(0, static_cast<int>(previewProject.tracks.size()) - 1),
                                                          yToTrackIndex(event.position.y));
            clip.startSec = tickToSeconds(beforeProject, requestedStartTick);
            if (juce::isPositiveAndBelow(requestedTrackIndex, static_cast<int>(previewProject.tracks.size()))
                && previewProject.tracks[static_cast<size_t>(requestedTrackIndex)].trackType.equalsIgnoreCase("sample"))
            {
                clip.trackIndex = requestedTrackIndex;
            }
            else
            {
                clip.trackIndex = draggedSampleClipBase.trackIndex;
            }

            previewDirty = true;
            repaint();
            return;
        }

        if (dragMode == DragMode::resizeSampleLeft)
        {
            const auto oldEndTick = resizeAnchorStartTick + resizeAnchorLengthTicks;
            const auto newStartTick = juce::jlimit(0,
                                                   oldEndTick - minimumSliceTicks,
                                                   xToSnapStartTick(event.position.x));
            const auto newStartSec = tickToSeconds(beforeProject, newStartTick);
            const auto newEndSec = tickToSeconds(beforeProject, oldEndTick);
            auto newDurationSec = juce::jmax(0.001, newEndSec - newStartSec);
            auto newSourceOffsetSec = draggedSampleClipBase.sourceOffsetSec
                + juce::jmax(0.0, newStartSec - draggedSampleClipBase.startSec);

            if (draggedSampleClipBase.sourceFileDurationSec > 0.0)
            {
                newSourceOffsetSec = juce::jlimit(0.0,
                                                  draggedSampleClipBase.sourceFileDurationSec,
                                                  newSourceOffsetSec);
                newDurationSec = juce::jmin(newDurationSec,
                                            juce::jmax(0.001,
                                                       draggedSampleClipBase.sourceFileDurationSec - newSourceOffsetSec));
            }

            clip.startSec = newStartSec;
            clip.durationSec = newDurationSec;
            clip.sourceOffsetSec = newSourceOffsetSec;
            previewDirty = true;
            repaint();
            return;
        }

        if (dragMode == DragMode::resizeSampleRight)
        {
            const auto clipStartTick = secondsToTick(beforeProject, draggedSampleClipBase.startSec);
            const auto newEndTick = juce::jmax(clipStartTick + minimumSliceTicks,
                                               xToSnapEndTick(event.position.x));
            const auto newEndSec = tickToSeconds(beforeProject, newEndTick);
            auto newDurationSec = juce::jmax(0.001, newEndSec - draggedSampleClipBase.startSec);

            if (draggedSampleClipBase.sourceFileDurationSec > 0.0)
            {
                const auto maximumDuration = juce::jmax(0.001,
                                                        draggedSampleClipBase.sourceFileDurationSec
                                                            - draggedSampleClipBase.sourceOffsetSec);
                newDurationSec = juce::jmin(newDurationSec, maximumDuration);
            }

            clip.startSec = draggedSampleClipBase.startSec;
            clip.durationSec = newDurationSec;
            clip.sourceOffsetSec = draggedSampleClipBase.sourceOffsetSec;
            previewDirty = true;
            repaint();
            return;
        }
    }

    if (!juce::isPositiveAndBelow(draggedSectionIndex, static_cast<int>(previewProject.midiSections.size())))
        return;

    if (dragMode != DragMode::createSection && !dragCreatesCopy)
        previewProject = beforeProject;

    if (dragMode == DragMode::moveSection)
    {
        const auto requestedAnchorStartTick = juce::jmax(0, xToSnapStartTick(event.position.x) - dragOffsetTick);
        const auto targetTrackIndex = juce::jlimit(0,
                                                   juce::jmax(0, static_cast<int>(previewProject.tracks.size()) - 1),
                                                   yToTrackIndex(event.position.y));
        const auto trackCount = static_cast<int>(previewProject.tracks.size());

        auto minimumBaseStartTick = std::numeric_limits<int>::max();
        auto minimumBaseTrackIndex = std::numeric_limits<int>::max();
        auto maximumBaseTrackIndex = std::numeric_limits<int>::min();
        for (size_t index = 0; index < draggedSectionBaseStartTicks.size(); ++index)
        {
            minimumBaseStartTick = juce::jmin(minimumBaseStartTick, draggedSectionBaseStartTicks[index]);
            minimumBaseTrackIndex = juce::jmin(minimumBaseTrackIndex, draggedSectionBaseTrackIndices[index]);
            maximumBaseTrackIndex = juce::jmax(maximumBaseTrackIndex, draggedSectionBaseTrackIndices[index]);
        }

        auto tickDelta = requestedAnchorStartTick - dragAnchorStartTick;
        if (minimumBaseStartTick != std::numeric_limits<int>::max())
            tickDelta = juce::jmax(tickDelta, -minimumBaseStartTick);

        auto trackDelta = targetTrackIndex - dragAnchorTrackIndex;
        if (minimumBaseTrackIndex != std::numeric_limits<int>::max()
            && maximumBaseTrackIndex != std::numeric_limits<int>::min()
            && trackCount > 0)
        {
            trackDelta = juce::jlimit(-minimumBaseTrackIndex,
                                      (trackCount - 1) - maximumBaseTrackIndex,
                                      trackDelta);
        }

        for (size_t index = 0; index < draggedSectionIndices.size(); ++index)
        {
            const auto sectionIndex = draggedSectionIndices[index];
            if (!juce::isPositiveAndBelow(sectionIndex, static_cast<int>(previewProject.midiSections.size())))
                continue;

            auto& draggedSection = previewProject.midiSections[static_cast<size_t>(sectionIndex)];
            draggedSection.startTick = juce::jmax(0, draggedSectionBaseStartTicks[index] + tickDelta);
            draggedSection.trackIndex = juce::jlimit(0,
                                                     juce::jmax(0, trackCount - 1),
                                                     draggedSectionBaseTrackIndices[index] + trackDelta);
        }
        previewDirty = true;
        repaint();
        return;
    }

    auto& section = previewProject.midiSections[static_cast<size_t>(draggedSectionIndex)];

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
        draggedSectionIndices.clear();
        draggedSectionBaseStartTicks.clear();
        draggedSectionBaseTrackIndices.clear();
        dragOffsetTick = 0;
        dragAnchorStartTick = 0;
        dragAnchorTrackIndex = 0;
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
        else if (dragMode == DragMode::moveSampleClip)
            actionName = dragCreatesCopy ? "Duplicate Sample Clip" : "Move Sample Clip";
        else if (dragMode == DragMode::resizeSampleLeft || dragMode == DragMode::resizeSampleRight)
            actionName = "Resize Sample Clip";
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
            actionName = erasedSampleClipIndices.empty() ? "Erase Pattern Clips"
                                                         : (erasedSectionIndices.empty() ? "Erase Sample Clips"
                                                                                         : "Erase Clips");
        }
        projectWriter(previewProject, undoable, actionName);
        if ((dragMode == DragMode::moveSampleClip
             || dragMode == DragMode::resizeSampleLeft
             || dragMode == DragMode::resizeSampleRight)
            && sectionSelectCallback != nullptr)
        {
            selectedSectionIndices.clear();
            if (juce::isPositiveAndBelow(draggedSampleClipIndex, static_cast<int>(previewProject.sampleClips.size())))
                selectedSampleClipIndex = draggedSampleClipIndex;
            sectionSelectCallback(-1, false);
        }
        else if (dragMode == DragMode::eraseSections && sectionSelectCallback != nullptr)
        {
            if (!erasedSampleClipIndices.empty())
                selectedSampleClipIndex = -1;
            sectionSelectCallback(-1, false);
        }
        else if (sectionSelectCallback != nullptr)
        {
            if (draggedSectionIndices.size() > 1)
            {
                selectedSectionIndices = draggedSectionIndices;
                sectionSelectCallback(-1, false);
            }
            else if (draggedSectionIndex >= 0)
            {
                selectedSectionIndices = { draggedSectionIndex };
                sectionSelectCallback(draggedSectionIndex, false);
            }
        }
    }

    previewActive = false;
    previewDirty = false;
    dragMode = DragMode::none;
    draggedSectionIndex = -1;
    draggedSectionIndices.clear();
    draggedSectionBaseStartTicks.clear();
    draggedSectionBaseTrackIndices.clear();
    draggedSampleClipIndex = -1;
    draggedSampleClipBase = {};
    dragOffsetTick = 0;
    dragAnchorStartTick = 0;
    dragAnchorTrackIndex = 0;
    createStartTick = 0;
    resizeAnchorStartTick = 0;
    resizeAnchorLengthTicks = ticksPerBar(project());
    dragCreatesCopy = false;
    marqueeStart = {};
    marqueeRect = {};
    erasedSectionIndices.clear();
    erasedSampleClipIndices.clear();
    updateCursorForPosition({ -1.0f, -1.0f });
    refreshFromModel();
}

void ArrangementOverviewComponent::mouseExit(const juce::MouseEvent&)
{
    updateCursorForPosition({ -1.0f, -1.0f });
}

void ArrangementOverviewComponent::mouseWheelMove(const juce::MouseEvent& event, const juce::MouseWheelDetails& wheel)
{
    if (isLaneHeightWheelGesture(event, wheel))
    {
        const auto oldLaneHeight = laneHeight();
        const auto newLaneHeight = oldLaneHeight + laneHeightDeltaFromWheel(wheel.deltaY);
        if (std::abs(newLaneHeight - oldLaneHeight) < 0.01f)
            return;

        const auto mouseYInViewport = [this, &event]() -> float
        {
            if (auto* viewport = findParentComponentOfClass<juce::Viewport>())
                return event.position.y - static_cast<float>(viewport->getViewPositionY());
            return event.position.y;
        }();
        const auto focusRow = juce::jmax(0.0f, event.position.y - rulerHeight()) / juce::jmax(1.0f, oldLaneHeight);

        setLaneHeight(newLaneHeight);
        if (laneHeightChangedCallback != nullptr)
            laneHeightChangedCallback(getLaneHeight());

        if (auto* viewport = findParentComponentOfClass<juce::Viewport>())
        {
            const auto newContentY = rulerHeight() + (focusRow * laneHeight());
            viewport->setViewPosition(viewport->getViewPositionX(),
                                      juce::jmax(0, juce::roundToInt(newContentY - mouseYInViewport)));
        }
        return;
    }

    if (!isZoomWheelGesture(event, wheel))
    {
        if (auto* viewport = findParentComponentOfClass<juce::Viewport>())
            viewport->mouseWheelMove(event.getEventRelativeTo(viewport), wheel);
        return;
    }

    const auto oldPixelsPerBar = pixelsPerBar;
    const auto mouseXInViewport = [this, &event]() -> float
    {
        if (auto* viewport = findParentComponentOfClass<juce::Viewport>())
            return event.position.x - static_cast<float>(viewport->getViewPositionX());
        return event.position.x;
    }();
    const auto focusTick = juce::jmax(0.0,
                                      (static_cast<double>(juce::jmax(0.0f, event.position.x - laneHeaderWidth()))
                                       / static_cast<double>(juce::jmax(1.0f, oldPixelsPerBar)))
                                          * static_cast<double>(ticksPerBar(project())));

    setHorizontalZoom(oldPixelsPerBar * zoomScaleFromWheelDelta(wheel.deltaY));
    if (zoomChangedCallback != nullptr)
        zoomChangedCallback(getHorizontalZoom());

    if (auto* viewport = findParentComponentOfClass<juce::Viewport>())
    {
        const auto newContentX = laneHeaderWidth()
            + (static_cast<float>(focusTick) / static_cast<float>(ticksPerBar(project()))) * pixelsPerBar;
        viewport->setViewPosition(juce::jmax(0, juce::roundToInt(newContentX - mouseXInViewport)),
                                  viewport->getViewPositionY());
    }
}

bool ArrangementOverviewComponent::keyPressed(const juce::KeyPress& key)
{
    if (matchesCommandShortcut(key, 'a'))
    {
        selectAllSections();
        return true;
    }

    if (matchesCommandShortcut(key, 'd'))
        return duplicateSelectedSections();

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
    {
        if (!juce::isPositiveAndBelow(selectedSampleClipIndex, static_cast<int>(project().sampleClips.size())))
            return false;

        auto updatedProject = project();
        updatedProject.sampleClips.erase(updatedProject.sampleClips.begin() + selectedSampleClipIndex);
        selectedSampleClipIndex = -1;
        selectedSectionIndices.clear();
        projectWriter(updatedProject, true, "Delete Sample Clip");
        if (sectionSelectCallback != nullptr)
            sectionSelectCallback(-1, false);
        return true;
    }

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

bool ArrangementOverviewComponent::duplicateSelectedSections()
{
    std::vector<int> sourceIndices = selectedSectionIndices;
    if (sourceIndices.empty())
    {
        const auto selectedSectionIndex = selectedSectionGetter != nullptr ? selectedSectionGetter() : -1;
        if (juce::isPositiveAndBelow(selectedSectionIndex, static_cast<int>(project().midiSections.size())))
            sourceIndices.push_back(selectedSectionIndex);
    }

    if (sourceIndices.empty())
        return false;

    std::sort(sourceIndices.begin(), sourceIndices.end());
    sourceIndices.erase(std::unique(sourceIndices.begin(), sourceIndices.end()), sourceIndices.end());

    const auto& state = project();
    auto updatedProject = state;
    auto minimumStartTick = std::numeric_limits<int>::max();
    auto maximumEndTick = std::numeric_limits<int>::min();

    for (const auto sectionIndex : sourceIndices)
    {
        if (!juce::isPositiveAndBelow(sectionIndex, static_cast<int>(state.midiSections.size())))
            continue;

        const auto& section = state.midiSections[static_cast<size_t>(sectionIndex)];
        minimumStartTick = juce::jmin(minimumStartTick, section.startTick);
        maximumEndTick = juce::jmax(maximumEndTick, section.startTick + clipLengthTicks(section, state));
    }

    if (minimumStartTick == std::numeric_limits<int>::max() || maximumEndTick == std::numeric_limits<int>::min())
        return false;

    const auto duplicateOffsetTicks = juce::jmax(arrangementSnapTickLength(state), maximumEndTick - minimumStartTick);
    std::vector<int> duplicatedIndices;
    duplicatedIndices.reserve(sourceIndices.size());

    for (const auto sectionIndex : sourceIndices)
    {
        if (!juce::isPositiveAndBelow(sectionIndex, static_cast<int>(state.midiSections.size())))
            continue;

        auto duplicateSection = state.midiSections[static_cast<size_t>(sectionIndex)];
        duplicateSection.startTick = juce::jmax(0, duplicateSection.startTick + duplicateOffsetTicks);
        updatedProject.midiSections.push_back(std::move(duplicateSection));
        duplicatedIndices.push_back(static_cast<int>(updatedProject.midiSections.size()) - 1);
    }

    if (duplicatedIndices.empty())
        return false;

    selectedSectionIndices = duplicatedIndices;
    projectWriter(updatedProject,
                  true,
                  duplicatedIndices.size() > 1 ? "Duplicate Pattern Clips" : "Duplicate Pattern Clip");
    if (sectionSelectCallback != nullptr)
        sectionSelectCallback(duplicatedIndices.size() == 1 ? duplicatedIndices.front() : -1, false);
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
    const auto width = juce::roundToInt(tickToX(displayedBarCount() * ticksPerBar(displayedProject())));
    auto totalHeight = rulerHeight();
    const auto& state = displayedProject();
    if (state.tracks.empty())
    {
        totalHeight += laneHeight();
    }
    else
    {
        for (int trackIndex = 0; trackIndex < static_cast<int>(state.tracks.size()); ++trackIndex)
            totalHeight += trackTotalHeight(trackIndex, state);
    }

    const auto height = juce::roundToInt(totalHeight);
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

float ArrangementOverviewComponent::trackClipTop(int trackIndex, const ProjectState& state) const
{
    auto top = rulerHeight();
    const auto clampedTrackIndex = juce::jlimit(0, static_cast<int>(state.tracks.size()), trackIndex);
    for (int index = 0; index < clampedTrackIndex; ++index)
        top += trackTotalHeight(index, state);
    return top;
}

float ArrangementOverviewComponent::trackTotalHeight(int trackIndex, const ProjectState& state) const
{
    if (!juce::isPositiveAndBelow(trackIndex, static_cast<int>(state.tracks.size())))
        return laneHeight();

    const auto visibleTargets = visibleArrangementAutomationTargets(state.tracks[static_cast<size_t>(trackIndex)]);
    return laneHeight() * static_cast<float>(1 + visibleTargets.size());
}

bool ArrangementOverviewComponent::hitTestTrackLane(float y,
                                                    int& outTrackIndex,
                                                    juce::String& outAutomationTarget) const
{
    outTrackIndex = -1;
    outAutomationTarget.clear();

    const auto& state = displayedProject();
    if (state.tracks.empty() || y < rulerHeight())
        return false;

    auto currentY = rulerHeight();
    for (int trackIndex = 0; trackIndex < static_cast<int>(state.tracks.size()); ++trackIndex)
    {
        const auto clipBottom = currentY + laneHeight();
        if (y >= currentY && y < clipBottom)
        {
            outTrackIndex = trackIndex;
            return true;
        }

        currentY = clipBottom;
        const auto visibleTargets = visibleArrangementAutomationTargets(state.tracks[static_cast<size_t>(trackIndex)]);
        for (const auto& target : visibleTargets)
        {
            const auto laneBottom = currentY + laneHeight();
            if (y >= currentY && y < laneBottom)
            {
                outTrackIndex = trackIndex;
                outAutomationTarget = target;
                return true;
            }

            currentY = laneBottom;
        }
    }

    return false;
}

bool ArrangementOverviewComponent::isInterestedInFileDrag(const juce::StringArray& files)
{
    if (sampleFileDropCallback == nullptr || files.isEmpty())
        return false;

    for (const auto& path : files)
    {
        if (isSupportedAudioSampleFile(path))
            return true;
    }

    return false;
}

void ArrangementOverviewComponent::fileDragEnter(const juce::StringArray& files, int x, int y)
{
    fileDragMove(files, x, y);
}

void ArrangementOverviewComponent::fileDragMove(const juce::StringArray& files, int x, int y)
{
    if (!isInterestedInFileDrag(files))
        return;

    sampleFileDragActive = true;
    sampleFileDragTrackIndex = yToTrackIndex(static_cast<float>(y));
    sampleFileDragStartTick = xToSnapStartTick(static_cast<float>(x));
    repaint();
}

void ArrangementOverviewComponent::fileDragExit(const juce::StringArray& files)
{
    juce::ignoreUnused(files);
    sampleFileDragActive = false;
    sampleFileDragTrackIndex = -1;
    sampleFileDragStartTick = 0;
    repaint();
}

void ArrangementOverviewComponent::filesDropped(const juce::StringArray& files, int x, int y)
{
    const auto targetTrackIndex = yToTrackIndex(static_cast<float>(y));
    const auto targetStartTick = xToSnapStartTick(static_cast<float>(x));
    fileDragExit(files);

    if (sampleFileDropCallback == nullptr || files.isEmpty() || targetTrackIndex < 0)
        return;

    ignoreUnused(sampleFileDropCallback(files, targetTrackIndex, targetStartTick));
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
    int trackIndex = -1;
    juce::String automationTarget;
    if (hitTestTrackLane(y, trackIndex, automationTarget))
        return trackIndex;

    const auto trackCount = static_cast<int>(displayedProject().tracks.size());
    if (trackCount <= 0)
        return -1;
    if (y < rulerHeight())
        return 0;
    return trackCount - 1;
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

int ArrangementOverviewComponent::clipLengthTicks(const SampleClip& clip, const ProjectState& state) const
{
    const auto startTick = secondsToTick(state, juce::jmax(0.0, clip.startSec));
    const auto endTick = secondsToTick(state, juce::jmax(0.0, clip.startSec + juce::jmax(0.0, clip.durationSec)));
    return juce::jmax(kMinSequenceSnapTicks, endTick - startTick);
}

juce::Rectangle<float> ArrangementOverviewComponent::sectionRect(const MidiSection& section, const ProjectState& state) const
{
    const auto lengthTicks = clipLengthTicks(section, state);
    return { tickToX(section.startTick),
             trackClipTop(juce::jmax(0, section.trackIndex), state),
             juce::jmax(14.0f, tickToX(section.startTick + lengthTicks) - tickToX(section.startTick)),
             laneHeight() };
}

juce::Rectangle<float> ArrangementOverviewComponent::clipRect(const SampleClip& clip) const
{
    const auto& state = displayedProject();
    const auto startTick = secondsToTick(state, clip.startSec);
    const auto endTick = startTick + clipLengthTicks(clip, state);
    return { tickToX(startTick),
             trackClipTop(juce::jmax(0, clip.trackIndex), state),
             juce::jmax(10.0f, tickToX(juce::jmax(startTick + 1, endTick)) - tickToX(startTick)),
             laneHeight() };
}

int ArrangementOverviewComponent::hitTestSampleClip(const ProjectState& state,
                                                    juce::Point<float> position,
                                                    juce::String& edgeOut) const
{
    edgeOut.clear();
    for (int clipIndex = static_cast<int>(state.sampleClips.size()) - 1; clipIndex >= 0; --clipIndex)
    {
        const auto& clip = state.sampleClips[static_cast<size_t>(clipIndex)];
        const auto startTick = secondsToTick(state, clip.startSec);
        const auto endTick = startTick + clipLengthTicks(clip, state);
        auto rect = juce::Rectangle<float>(tickToX(startTick),
                                           trackClipTop(juce::jmax(0, clip.trackIndex), state),
                                           juce::jmax(10.0f,
                                                      tickToX(juce::jmax(startTick + 1, endTick)) - tickToX(startTick)),
                                           laneHeight()).reduced(1.5f, 4.0f);
        if (!rect.contains(position))
            continue;

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
        return clipIndex;
    }

    return -1;
}

int ArrangementOverviewComponent::hitTestSampleClip(juce::Point<float> position, juce::String& edgeOut) const
{
    return hitTestSampleClip(displayedProject(), position, edgeOut);
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
    const auto hitSampleClipIndex = hitTestSampleClip(position, hitEdge);
    if (hitSampleClipIndex >= 0)
    {
        const auto toolMode = toolGetter != nullptr ? toolGetter() : EditorToolMode::pencil;
        if (toolMode == EditorToolMode::eraser)
        {
            setMouseCursor(juce::MouseCursor::CrosshairCursor);
            return;
        }

        if (toolMode == EditorToolMode::scissors)
        {
            setMouseCursor(scissorsToolCursor());
            return;
        }

        if (hitEdge == "left" || hitEdge == "right")
            setMouseCursor(juce::MouseCursor::LeftRightResizeCursor);
        else
            setMouseCursor(juce::MouseCursor::DraggingHandCursor);
        return;
    }

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
            setMouseCursor(glueToolCursor());
            return;
        }

        if (toolMode == EditorToolMode::scissors)
        {
            setMouseCursor(scissorsToolCursor());
            return;
        }

        if (hitEdge == "left" || hitEdge == "right")
            setMouseCursor(juce::MouseCursor::LeftRightResizeCursor);
        else
            setMouseCursor(juce::MouseCursor::DraggingHandCursor);
        return;
    }

    int trackIndex = -1;
    juce::String automationTarget;
    if (hitTestTrackLane(position.y, trackIndex, automationTarget) && automationTarget.isNotEmpty())
    {
        setMouseCursor(juce::MouseCursor::PointingHandCursor);
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

bool ArrangementOverviewComponent::splitSectionAtTick(int sectionIndex, int splitTick)
{
    auto updatedProject = project();
    if (!juce::isPositiveAndBelow(sectionIndex, static_cast<int>(updatedProject.midiSections.size())))
        return false;

    const auto sourceSection = updatedProject.midiSections[static_cast<size_t>(sectionIndex)];
    auto* sourcePattern = findMidiPattern(updatedProject, sourceSection.patternId);
    if (sourcePattern == nullptr)
        return false;

    const auto sourceLengthTicks = patternLengthTicks(*sourcePattern);
    const auto minimumSliceTicks = arrangementSnapTickLength(updatedProject);
    const auto localSplitTick = splitTick - sourceSection.startTick;
    if (localSplitTick < minimumSliceTicks || localSplitTick > sourceLengthTicks - minimumSliceTicks)
        return false;

    MidiPattern leftPattern;
    leftPattern.id = juce::Uuid().toString();
    leftPattern.name = sourcePattern->name.trim().isNotEmpty() ? sourcePattern->name : sourceSection.name;
    leftPattern.lengthTicks = localSplitTick;

    MidiPattern rightPattern;
    rightPattern.id = juce::Uuid().toString();
    rightPattern.name = leftPattern.name;
    rightPattern.lengthTicks = sourceLengthTicks - localSplitTick;

    leftPattern.notes.reserve(sourcePattern->notes.size());
    rightPattern.notes.reserve(sourcePattern->notes.size());

    for (const auto& sourceNote : sourcePattern->notes)
    {
        const auto noteStartTick = sourceNote.startTick;
        const auto noteEndTick = sourceNote.startTick + sourceNote.durationTick;

        if (noteEndTick <= localSplitTick)
        {
            auto leftNote = sourceNote;
            leftNote.selected = false;
            leftPattern.notes.push_back(std::move(leftNote));
            continue;
        }

        if (noteStartTick >= localSplitTick)
        {
            auto rightNote = sourceNote;
            rightNote.startTick = juce::jmax(0, rightNote.startTick - localSplitTick);
            rightNote.selected = false;
            rightPattern.notes.push_back(std::move(rightNote));
            continue;
        }

        auto leftNote = sourceNote;
        leftNote.durationTick = juce::jmax(1, localSplitTick - noteStartTick);
        leftNote.selected = false;
        leftPattern.notes.push_back(std::move(leftNote));

        auto rightNote = sourceNote;
        rightNote.startTick = 0;
        rightNote.durationTick = juce::jmax(1, noteEndTick - localSplitTick);
        rightNote.selected = false;
        rightPattern.notes.push_back(std::move(rightNote));
    }

    for (const auto& sourceLane : sourcePattern->controllerLanes)
    {
        AutomationLane leftLane;
        leftLane.target = sourceLane.target;
        leftLane.enabled = sourceLane.enabled;

        AutomationLane rightLane;
        rightLane.target = sourceLane.target;
        rightLane.enabled = sourceLane.enabled;

        for (const auto& point : sourceLane.points)
        {
            if (point.tick < localSplitTick)
            {
                leftLane.points.push_back(point);
                continue;
            }

            if (point.tick > localSplitTick)
            {
                auto shiftedPoint = point;
                shiftedPoint.tick = juce::jmax(0, shiftedPoint.tick - localSplitTick);
                rightLane.points.push_back(std::move(shiftedPoint));
                continue;
            }
        }

        if (const auto splitValue = interpolatedControllerValueAtTick(sourceLane, localSplitTick))
        {
            appendAutomationPointIfMissing(leftLane.points, localSplitTick, *splitValue);
            appendAutomationPointIfMissing(rightLane.points, 0, *splitValue);
        }

        if (!leftLane.points.empty())
            leftPattern.controllerLanes.push_back(std::move(leftLane));
        if (!rightLane.points.empty())
            rightPattern.controllerLanes.push_back(std::move(rightLane));
    }

    sortPatternNotes(leftPattern.notes);
    sortPatternNotes(rightPattern.notes);
    sanitisePatternControllerLanes(leftPattern);
    sanitisePatternControllerLanes(rightPattern);

    MidiSection leftSection = sourceSection;
    leftSection.patternId = leftPattern.id;
    leftSection.lengthTicks = leftPattern.lengthTicks;

    MidiSection rightSection = sourceSection;
    rightSection.startTick = splitTick;
    rightSection.patternId = rightPattern.id;
    rightSection.lengthTicks = rightPattern.lengthTicks;

    updatedProject.midiPatterns.push_back(std::move(leftPattern));
    updatedProject.midiPatterns.push_back(std::move(rightPattern));
    updatedProject.midiSections.erase(updatedProject.midiSections.begin() + sectionIndex);
    updatedProject.midiSections.insert(updatedProject.midiSections.begin() + sectionIndex, std::move(leftSection));
    updatedProject.midiSections.insert(updatedProject.midiSections.begin() + sectionIndex + 1, std::move(rightSection));
    updatedProject.recalculateTimeFields();

    previewActive = false;
    previewDirty = false;
    dragMode = DragMode::none;
    draggedSectionIndex = -1;
    erasedSectionIndices.clear();
    selectedSectionIndices.clear();
    selectedSectionIndices.push_back(sectionIndex + 1);

    projectWriter(updatedProject, true, "Split Pattern Clip");
    if (sectionSelectCallback != nullptr)
        sectionSelectCallback(sectionIndex + 1, false);
    return true;
}

bool ArrangementOverviewComponent::splitSampleClipAtTick(int clipIndex, int splitTick)
{
    auto updatedProject = project();
    if (!juce::isPositiveAndBelow(clipIndex, static_cast<int>(updatedProject.sampleClips.size())))
        return false;

    const auto sourceClip = updatedProject.sampleClips[static_cast<size_t>(clipIndex)];
    const auto clipStartTick = secondsToTick(updatedProject, sourceClip.startSec);
    const auto sourceLengthTicks = clipLengthTicks(sourceClip, updatedProject);
    const auto minimumSliceTicks = arrangementSnapTickLength(updatedProject);
    const auto localSplitTick = splitTick - clipStartTick;
    if (localSplitTick < minimumSliceTicks || localSplitTick > sourceLengthTicks - minimumSliceTicks)
        return false;

    const auto splitSec = tickToSeconds(updatedProject, splitTick);
    const auto localSplitSec = splitSec - sourceClip.startSec;
    if (localSplitSec <= 0.0 || localSplitSec >= sourceClip.durationSec)
        return false;

    auto leftClip = sourceClip;
    leftClip.durationSec = localSplitSec;

    auto rightClip = sourceClip;
    rightClip.startSec = splitSec;
    rightClip.durationSec = juce::jmax(0.001, sourceClip.durationSec - localSplitSec);
    rightClip.sourceOffsetSec = sourceClip.sourceOffsetSec + localSplitSec;

    updatedProject.sampleClips.erase(updatedProject.sampleClips.begin() + clipIndex);
    updatedProject.sampleClips.insert(updatedProject.sampleClips.begin() + clipIndex, std::move(leftClip));
    updatedProject.sampleClips.insert(updatedProject.sampleClips.begin() + clipIndex + 1, std::move(rightClip));
    updatedProject.recalculateTimeFields();

    previewActive = false;
    previewDirty = false;
    dragMode = DragMode::none;
    draggedSectionIndex = -1;
    draggedSampleClipIndex = -1;
    draggedSampleClipBase = {};
    erasedSectionIndices.clear();
    erasedSampleClipIndices.clear();
    selectedSectionIndices.clear();
    selectedSampleClipIndex = clipIndex + 1;

    projectWriter(updatedProject, true, "Split Sample Clip");
    if (sectionSelectCallback != nullptr)
        sectionSelectCallback(-1, false);
    return true;
}

bool ArrangementOverviewComponent::glueSectionsAtIndex(int sectionIndex)
{
    const auto sourceProject = project();
    if (!juce::isPositiveAndBelow(sectionIndex, static_cast<int>(sourceProject.midiSections.size())))
        return false;

    const auto clickedSection = sourceProject.midiSections[static_cast<size_t>(sectionIndex)];
    std::vector<int> cluster;
    const auto clickedSelected = std::find(selectedSectionIndices.begin(), selectedSectionIndices.end(), sectionIndex)
        != selectedSectionIndices.end();
    if (clickedSelected && selectedSectionIndices.size() > 1)
    {
        for (const auto selectedIndex : selectedSectionIndices)
        {
            if (!juce::isPositiveAndBelow(selectedIndex, static_cast<int>(sourceProject.midiSections.size())))
                continue;

            const auto& selectedSection = sourceProject.midiSections[static_cast<size_t>(selectedIndex)];
            if (selectedSection.trackIndex != clickedSection.trackIndex)
                continue;

            cluster.push_back(selectedIndex);
        }
    }

    const auto findNextRightSectionIndex = [&]() -> int
    {
        int nextIndex = -1;
        int nextStartTick = std::numeric_limits<int>::max();
        int nextEndTick = std::numeric_limits<int>::max();

        for (int index = 0; index < static_cast<int>(sourceProject.midiSections.size()); ++index)
        {
            if (index == sectionIndex)
                continue;

            const auto& section = sourceProject.midiSections[static_cast<size_t>(index)];
            if (section.trackIndex != clickedSection.trackIndex)
                continue;

            if (section.startTick < clickedSection.startTick)
                continue;

            const auto sectionEndTick = section.startTick + clipLengthTicks(section, sourceProject);
            if (section.startTick < nextStartTick
                || (section.startTick == nextStartTick && sectionEndTick < nextEndTick)
                || (section.startTick == nextStartTick && sectionEndTick == nextEndTick
                    && (nextIndex < 0 || index < nextIndex)))
            {
                nextIndex = index;
                nextStartTick = section.startTick;
                nextEndTick = sectionEndTick;
            }
        }

        return nextIndex;
    };

    if (cluster.empty())
        cluster.push_back(sectionIndex);

    if (cluster.size() < 2)
    {
        const auto nextIndex = findNextRightSectionIndex();
        if (nextIndex >= 0
            && std::find(cluster.begin(), cluster.end(), nextIndex) == cluster.end())
        {
            cluster.clear();
            cluster.push_back(sectionIndex);
            cluster.push_back(nextIndex);
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

    const auto clickedPatternName = clickedPattern->name.trim();

    const auto newStartTick = sourceProject.midiSections[static_cast<size_t>(cluster.front())].startTick;
    int newEndTick = newStartTick + clipLengthTicks(sourceProject.midiSections[static_cast<size_t>(cluster.front())],
                                                    sourceProject);

    MidiPattern gluedPattern;
    gluedPattern.id = juce::Uuid().toString();
    gluedPattern.name = clickedPatternName.isNotEmpty() ? clickedPatternName
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
    {
        if (!juce::isPositiveAndBelow(*iter, static_cast<int>(updatedProject.midiSections.size())))
            return false;
        updatedProject.midiSections.erase(updatedProject.midiSections.begin() + *iter);
    }

    updatedProject.midiPatterns.push_back(std::move(gluedPattern));

    const auto boundedInsertIndex = juce::jlimit(0,
                                                 static_cast<int>(updatedProject.midiSections.size()),
                                                 insertionIndex);
    updatedProject.midiSections.insert(updatedProject.midiSections.begin() + boundedInsertIndex,
                                       std::move(gluedSection));
    updatedProject.recalculateTimeFields();

    previewActive = false;
    previewDirty = false;
    dragMode = DragMode::none;
    draggedSectionIndex = -1;
    erasedSectionIndices.clear();
    selectedSectionIndices.clear();
    selectedSectionIndices.push_back(boundedInsertIndex);

    projectWriter(updatedProject, true, "Glue Pattern Clips");
    if (sectionSelectCallback != nullptr)
        sectionSelectCallback(boundedInsertIndex, false);
    return true;
}

void ArrangementOverviewComponent::showSectionContextMenu(int sectionIndex,
                                                          int sampleClipIndex,
                                                          juce::Point<int> screenPosition,
                                                          int targetTrackIndex,
                                                          int targetStartTick)
{
    const bool hasSection = juce::isPositiveAndBelow(sectionIndex, static_cast<int>(project().midiSections.size()));
    const bool hasSampleClip = juce::isPositiveAndBelow(sampleClipIndex, static_cast<int>(project().sampleClips.size()));
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
        menuToolScissors,
        menuToolGlue,
        menuToolEraser,
        menuOpen,
        menuCut,
        menuCopy,
        menuPaste,
        menuDelete,
        menuDuplicate,
        menuSeparateStems
    };

    juce::PopupMenu menu;
    const auto toolMode = toolGetter != nullptr ? toolGetter() : EditorToolMode::pencil;
    juce::PopupMenu audioMenu;

    menu.addSectionHeader("Tools");
    menu.addItem(menuToolPencil, "Pencil", true, toolMode == EditorToolMode::pencil, createToolMenuIcon(EditorToolMode::pencil));
    menu.addItem(menuToolSelect, "Select", true, toolMode == EditorToolMode::selection, createToolMenuIcon(EditorToolMode::selection));
    menu.addItem(menuToolScissors, "Scissors", true, toolMode == EditorToolMode::scissors, createToolMenuIcon(EditorToolMode::scissors));
    menu.addItem(menuToolGlue, "Glue", true, toolMode == EditorToolMode::glue, createToolMenuIcon(EditorToolMode::glue));
    menu.addItem(menuToolEraser, "Eraser", true, toolMode == EditorToolMode::eraser, createToolMenuIcon(EditorToolMode::eraser));
    menu.addSeparator();
    menu.addSectionHeader("Clips");
    menu.addItem(menuOpen, "Open Pattern", hasSection);
    menu.addItem(menuCut, "Cut", hasSection);
    menu.addItem(menuCopy, "Copy", hasSection);
    menu.addItem(menuPaste, "Paste", canPaste);
    menu.addItem(menuDelete, "Delete", hasSection);
    menu.addItem(menuDuplicate, "Duplicate", hasSection);
    if (hasSampleClip)
    {
        menu.addSeparator();
        audioMenu.addItem(menuSeparateStems, "Extract Audio Stems");
        menu.addSubMenu("Audio", audioMenu);
    }

    menu.showMenuAsync(juce::PopupMenu::Options().withTargetScreenArea(juce::Rectangle<int>(screenPosition.x, screenPosition.y, 1, 1)),
                       [safeThis = juce::Component::SafePointer<ArrangementOverviewComponent>(this),
                        sectionIndex,
                        sampleClipIndex,
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

                                case menuToolScissors:
                                    if (safeThis->toolModeChangeCallback != nullptr)
                                        safeThis->toolModeChangeCallback(EditorToolMode::scissors);
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
                           const bool sampleClipAvailable = juce::isPositiveAndBelow(sampleClipIndex,
                                                                                    static_cast<int>(safeThis->project().sampleClips.size()));

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

                           if (result == menuSeparateStems)
                           {
                               if (sampleClipAvailable && safeThis->sampleClipStemSeparationCallback != nullptr)
                                   safeThis->sampleClipStemSeparationCallback(sampleClipIndex);
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

bool ArrangementOverviewComponent::markSampleClipForErase(int clipIndex)
{
    if (!juce::isPositiveAndBelow(clipIndex, static_cast<int>(beforeProject.sampleClips.size())))
        return false;

    if (std::find(erasedSampleClipIndices.begin(), erasedSampleClipIndices.end(), clipIndex) != erasedSampleClipIndices.end())
        return false;

    erasedSampleClipIndices.push_back(clipIndex);
    rebuildErasePreview();
    return true;
}

void ArrangementOverviewComponent::rebuildErasePreview()
{
    previewProject = beforeProject;

    std::sort(erasedSectionIndices.begin(), erasedSectionIndices.end());
    erasedSectionIndices.erase(std::unique(erasedSectionIndices.begin(), erasedSectionIndices.end()), erasedSectionIndices.end());
    std::sort(erasedSampleClipIndices.begin(), erasedSampleClipIndices.end());
    erasedSampleClipIndices.erase(std::unique(erasedSampleClipIndices.begin(), erasedSampleClipIndices.end()),
                                  erasedSampleClipIndices.end());

    for (auto iter = erasedSectionIndices.rbegin(); iter != erasedSectionIndices.rend(); ++iter)
    {
        if (juce::isPositiveAndBelow(*iter, static_cast<int>(previewProject.midiSections.size())))
            previewProject.midiSections.erase(previewProject.midiSections.begin() + *iter);
    }

    for (auto iter = erasedSampleClipIndices.rbegin(); iter != erasedSampleClipIndices.rend(); ++iter)
    {
        if (juce::isPositiveAndBelow(*iter, static_cast<int>(previewProject.sampleClips.size())))
            previewProject.sampleClips.erase(previewProject.sampleClips.begin() + *iter);
    }
}

} // namespace aims
