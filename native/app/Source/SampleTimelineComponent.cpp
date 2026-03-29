#include "SampleTimelineComponent.h"

#include <cmath>

namespace aims
{
namespace
{
const juce::Colour kBackground = juce::Colour::fromRGB(14, 16, 21);
const juce::Colour kLaneEven = juce::Colour::fromRGB(28, 31, 38);
const juce::Colour kLaneOdd = juce::Colour::fromRGB(22, 25, 31);
const juce::Colour kHeader = juce::Colour::fromRGB(18, 21, 27);
const juce::Colour kGrid = juce::Colour::fromRGB(63, 72, 88);
const juce::Colour kClip = juce::Colour::fromRGB(126, 213, 140);
const juce::Colour kClipSelected = juce::Colour::fromRGB(166, 233, 176);
const juce::Colour kPlayhead = juce::Colour::fromRGB(255, 102, 102);
const juce::Colour kLeftLocator = juce::Colour::fromRGB(120, 212, 255);
const juce::Colour kRightLocator = juce::Colour::fromRGB(255, 209, 102);
}

SampleTimelineComponent::SampleTimelineComponent(ProjectGetter projectGetterIn,
                                                 ProjectWriter projectWriterIn)
    : projectGetter(std::move(projectGetterIn)),
      projectWriter(std::move(projectWriterIn))
{
    setWantsKeyboardFocus(true);
    updateContentSize();
}

void SampleTimelineComponent::refreshFromModel()
{
    if (!previewActive)
        updateContentSize();
    clearSelectionIfInvalid();
    repaint();
}

void SampleTimelineComponent::paint(juce::Graphics& g)
{
    g.fillAll(kBackground);

    const auto& state = previewActive ? previewProject : project();
    const auto sampleTracks = sampleTrackIndices(state);
    const auto laneCount = juce::jmax(1, static_cast<int>(sampleTracks.size()));
    const auto headerWidth = laneHeaderWidth();
    const auto topHeight = rulerHeight();

    g.setColour(kHeader);
    g.fillRect(0.0f, 0.0f, static_cast<float>(getWidth()), topHeight);
    g.fillRect(0.0f, 0.0f, headerWidth, static_cast<float>(getHeight()));

    for (int lane = 0; lane < laneCount; ++lane)
    {
        const auto y = topHeight + (static_cast<float>(lane) * laneHeight());
        g.setColour((lane % 2) == 0 ? kLaneEven : kLaneOdd);
        g.fillRect(headerWidth, y, static_cast<float>(getWidth()) - headerWidth, laneHeight());

        auto label = juce::String("Sample Track");
        if (!sampleTracks.empty() && juce::isPositiveAndBelow(lane, static_cast<int>(sampleTracks.size())))
        {
            const auto trackIndex = sampleTracks[static_cast<size_t>(lane)];
            if (juce::isPositiveAndBelow(trackIndex, static_cast<int>(state.tracks.size())))
                label = state.tracks[static_cast<size_t>(trackIndex)].name;
        }
        else if (sampleTracks.empty())
        {
            label = "No Sample Tracks";
        }

        g.setColour(juce::Colour::fromRGB(182, 190, 201));
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

    const auto totalSeconds = juce::jmax(static_cast<double>(minimumSeconds), state.rightLocatorSec + 4.0);
    for (int second = 0; second <= static_cast<int>(std::ceil(totalSeconds)); ++second)
    {
        const auto x = secondsToX(static_cast<double>(second));
        g.setColour((second % 4) == 0 ? kGrid : kGrid.withAlpha(0.45f));
        g.drawVerticalLine(juce::roundToInt(x), 0.0f, static_cast<float>(getHeight()));

        g.setColour(juce::Colour::fromRGB(219, 224, 232));
        g.drawText(juce::String(second) + "s",
                   juce::roundToInt(x) + 3,
                   0,
                   36,
                   juce::roundToInt(topHeight),
                   juce::Justification::centredLeft);
    }

    for (int clipIndex = 0; clipIndex < static_cast<int>(state.sampleClips.size()); ++clipIndex)
    {
        const auto& clip = state.sampleClips[static_cast<size_t>(clipIndex)];
        const auto laneIndex = laneForTrackIndex(clip.trackIndex, state);
        if (laneIndex < 0)
            continue;

        auto rect = clipRect(clip, state).reduced(1.5f, 6.0f);
        const auto fill = clipIndex == selectedClipIndex ? kClipSelected : kClip;
        g.setColour(fill.withAlpha(0.86f));
        g.fillRoundedRectangle(rect, 5.0f);
        g.setColour(fill.darker(0.5f));
        g.drawRoundedRectangle(rect, 5.0f, clipIndex == selectedClipIndex ? 2.0f : 1.2f);

        if (!clip.waveformPreview.empty())
        {
            juce::Path waveform;
            const auto midY = rect.getCentreY();
            const auto amplitude = rect.getHeight() * 0.5f - 10.0f;
            waveform.startNewSubPath(rect.getX(), midY);
            for (int pointIndex = 0; pointIndex < static_cast<int>(clip.waveformPreview.size()); ++pointIndex)
            {
                const auto x = rect.getX()
                    + (rect.getWidth() * static_cast<float>(pointIndex))
                        / static_cast<float>(juce::jmax(1, static_cast<int>(clip.waveformPreview.size()) - 1));
                const auto y = midY - (juce::jlimit(0.0f, 1.0f, clip.waveformPreview[static_cast<size_t>(pointIndex)]) * amplitude);
                waveform.lineTo(x, y);
            }
            g.setColour(juce::Colour::fromRGB(232, 244, 255).withAlpha(0.92f));
            g.strokePath(waveform, juce::PathStrokeType(1.2f));
        }

        g.setColour(juce::Colour::fromRGB(18, 26, 31));
        g.drawText(juce::File(clip.path).getFileName(),
                   rect.toNearestInt().reduced(8, 4),
                   juce::Justification::topLeft,
                   true);
    }

    for (const auto& locator : { std::pair(state.leftLocatorSec, kLeftLocator),
                                 std::pair(state.rightLocatorSec, kRightLocator),
                                 std::pair(state.playheadSec, kPlayhead) })
    {
        const auto x = secondsToX(locator.first);
        g.setColour(locator.second);
        g.drawVerticalLine(juce::roundToInt(x), 0.0f, static_cast<float>(getHeight()));
    }

    if (sampleTracks.empty())
    {
        g.setColour(juce::Colour::fromRGB(206, 214, 225));
        g.drawFittedText("Set a track's type to sample, then place audio here.",
                         juce::Rectangle<int>(juce::roundToInt(headerWidth + 20.0f),
                                              juce::roundToInt(topHeight + 20.0f),
                                              juce::jmax(120, getWidth() - juce::roundToInt(headerWidth + 40.0f)),
                                              32),
                         juce::Justification::centredLeft,
                         2);
    }
}

void SampleTimelineComponent::resized()
{
    updateContentSize();
}

void SampleTimelineComponent::mouseDown(const juce::MouseEvent& event)
{
    grabKeyboardFocus();

    if (event.position.x > laneHeaderWidth() && event.position.y <= rulerHeight())
    {
        auto updatedProject = project();
        updatedProject.playheadTick = juce::jmax(0, secondsToTick(xToSeconds(event.position.x), updatedProject.bpm));
        updatedProject.recalculateTimeFields();
        projectWriter(updatedProject, false, "Move Sample Transport");
        refreshFromModel();
        return;
    }

    previewActive = false;
    previewDirty = false;
    draggedClipIndex = -1;

    const auto& state = project();
    selectedClipIndex = -1;
    for (int clipIndex = static_cast<int>(state.sampleClips.size()) - 1; clipIndex >= 0; --clipIndex)
    {
        const auto& clip = state.sampleClips[static_cast<size_t>(clipIndex)];
        if (!clipRect(clip, state).contains(event.position))
            continue;

        selectedClipIndex = clipIndex;
        previewProject = state;
        previewActive = true;
        draggedClipIndex = clipIndex;
        dragOffsetSeconds = xToSeconds(event.position.x) - clip.startSec;
        repaint();
        return;
    }

    repaint();
}

void SampleTimelineComponent::mouseDrag(const juce::MouseEvent& event)
{
    if (!previewActive || !juce::isPositiveAndBelow(draggedClipIndex, static_cast<int>(previewProject.sampleClips.size())))
        return;

    auto& clip = previewProject.sampleClips[static_cast<size_t>(draggedClipIndex)];
    clip.startSec = juce::jmax(0.0, xToSeconds(event.position.x) - dragOffsetSeconds);
    const auto laneIndex = yToLaneIndex(event.position.y);
    const auto trackIndex = sampleTrackForLane(laneIndex, previewProject);
    if (trackIndex >= 0)
        clip.trackIndex = trackIndex;

    previewDirty = true;
    repaint();
}

void SampleTimelineComponent::mouseUp(const juce::MouseEvent&)
{
    if (!previewActive)
        return;

    if (previewDirty && draggedClipIndex >= 0)
        projectWriter(previewProject, true, "Move Sample Clip");

    previewActive = false;
    previewDirty = false;
    refreshFromModel();
}

void SampleTimelineComponent::mouseWheelMove(const juce::MouseEvent&, const juce::MouseWheelDetails& wheel)
{
    if (wheel.deltaY == 0.0f)
        return;

    pixelsPerSecond = juce::jlimit(32.0f, 320.0f, pixelsPerSecond + (wheel.deltaY > 0.0f ? 8.0f : -8.0f));
    updateContentSize();
    repaint();
}

bool SampleTimelineComponent::keyPressed(const juce::KeyPress& key)
{
    if (key.getKeyCode() != juce::KeyPress::deleteKey
        && key.getKeyCode() != juce::KeyPress::backspaceKey)
        return false;

    auto updatedProject = project();
    if (!juce::isPositiveAndBelow(selectedClipIndex, static_cast<int>(updatedProject.sampleClips.size())))
        return false;

    updatedProject.sampleClips.erase(updatedProject.sampleClips.begin() + selectedClipIndex);
    selectedClipIndex = -1;
    projectWriter(updatedProject, true, "Remove Sample Clip");
    return true;
}

const ProjectState& SampleTimelineComponent::project() const
{
    return projectGetter();
}

std::vector<int> SampleTimelineComponent::sampleTrackIndices(const ProjectState& state) const
{
    std::vector<int> indices;
    for (int trackIndex = 0; trackIndex < static_cast<int>(state.tracks.size()); ++trackIndex)
    {
        if (state.tracks[static_cast<size_t>(trackIndex)].trackType.equalsIgnoreCase("sample"))
            indices.push_back(trackIndex);
    }
    return indices;
}

int SampleTimelineComponent::sampleTrackForLane(int laneIndex, const ProjectState& state) const
{
    const auto indices = sampleTrackIndices(state);
    if (!juce::isPositiveAndBelow(laneIndex, static_cast<int>(indices.size())))
        return -1;
    return indices[static_cast<size_t>(laneIndex)];
}

int SampleTimelineComponent::laneForTrackIndex(int trackIndex, const ProjectState& state) const
{
    const auto indices = sampleTrackIndices(state);
    for (int lane = 0; lane < static_cast<int>(indices.size()); ++lane)
    {
        if (indices[static_cast<size_t>(lane)] == trackIndex)
            return lane;
    }
    return -1;
}

void SampleTimelineComponent::updateContentSize()
{
    const auto& state = project();
    const auto laneCount = juce::jmax(1, static_cast<int>(sampleTrackIndices(state).size()));
    auto totalSeconds = juce::jmax(static_cast<double>(minimumSeconds), state.rightLocatorSec + 6.0);
    for (const auto& clip : state.sampleClips)
        totalSeconds = juce::jmax(totalSeconds, clip.startSec + clip.durationSec + 2.0);

    const auto width = juce::roundToInt(secondsToX(totalSeconds));
    const auto height = juce::roundToInt(rulerHeight() + laneCount * laneHeight());
    setSize(juce::jmax(width, 900), juce::jmax(height, 160));
}

float SampleTimelineComponent::laneHeaderWidth() const
{
    return 132.0f;
}

float SampleTimelineComponent::rulerHeight() const
{
    return 24.0f;
}

float SampleTimelineComponent::laneHeight() const
{
    return 96.0f;
}

float SampleTimelineComponent::secondsToX(double seconds) const
{
    return laneHeaderWidth() + static_cast<float>(juce::jmax(0.0, seconds)) * pixelsPerSecond;
}

double SampleTimelineComponent::xToSeconds(float x) const
{
    return juce::jmax(0.0, static_cast<double>(x - laneHeaderWidth()) / static_cast<double>(pixelsPerSecond));
}

int SampleTimelineComponent::yToLaneIndex(float y) const
{
    return static_cast<int>(juce::jmax(0.0f, y - rulerHeight()) / laneHeight());
}

juce::Rectangle<float> SampleTimelineComponent::clipRect(const SampleClip& clip, const ProjectState& state) const
{
    const auto laneIndex = juce::jmax(0, laneForTrackIndex(clip.trackIndex, state));
    return { secondsToX(clip.startSec),
             rulerHeight() + static_cast<float>(laneIndex) * laneHeight(),
             juce::jmax(10.0f, static_cast<float>(clip.durationSec) * pixelsPerSecond),
             laneHeight() };
}

void SampleTimelineComponent::clearSelectionIfInvalid()
{
    const auto& state = previewActive ? previewProject : project();
    if (!juce::isPositiveAndBelow(selectedClipIndex, static_cast<int>(state.sampleClips.size())))
        selectedClipIndex = -1;
}

} // namespace aims
