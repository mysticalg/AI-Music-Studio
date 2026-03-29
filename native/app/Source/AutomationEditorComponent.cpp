#include "AutomationEditorComponent.h"

#include <algorithm>
#include <cmath>

namespace aims
{
namespace
{
juce::String formatAutomationValue(const TrackState& track, const juce::String& target, double value)
{
    const auto numeric = static_cast<double>(value);

    if (target.equalsIgnoreCase(kAutomationTargetVolume))
        return juce::String(numeric * 100.0, 0) + "%";
    if (target.equalsIgnoreCase(kAutomationTargetPan))
        return juce::String(numeric >= 0.0 ? "+" : "") + juce::String(numeric, 2);
    if (target.equalsIgnoreCase(kAutomationTargetVstOutputGainDb))
        return juce::String(numeric >= 0.0 ? "+" : "") + juce::String(numeric, 1) + " dB";
    if (automationTargetIsVstParameter(target))
        return juce::String(numeric, 1);

    juce::ignoreUnused(track);
    return juce::String(numeric, 2);
}

bool automationPointsEqual(const std::vector<AutomationPoint>& lhs, const std::vector<AutomationPoint>& rhs)
{
    if (lhs.size() != rhs.size())
        return false;

    for (size_t index = 0; index < lhs.size(); ++index)
    {
        if (lhs[index].tick != rhs[index].tick)
            return false;
        if (std::abs(lhs[index].value - rhs[index].value) > 1.0e-6)
            return false;
    }

    return true;
}
} // namespace

class AutomationEditorComponent::LaneListModel final : public juce::ListBoxModel
{
public:
    explicit LaneListModel(AutomationEditorComponent& ownerIn)
        : owner(ownerIn)
    {
    }

    int getNumRows() override
    {
        return static_cast<int>(owner.currentLanes.size());
    }

    void paintListBoxItem(int rowNumber,
                          juce::Graphics& g,
                          int width,
                          int height,
                          bool rowIsSelected) override
    {
        juce::ignoreUnused(height);

        if (!juce::isPositiveAndBelow(rowNumber, static_cast<int>(owner.currentLanes.size())))
            return;

        const auto& lane = owner.currentLanes[static_cast<size_t>(rowNumber)];
        const auto label = automationTargetLabel(owner.currentTrackSnapshot, lane.target);
        const auto pointCount = static_cast<int>(lane.points.size());
        const auto status = lane.enabled ? "Read" : "Bypassed";
        const auto text = label + "  [" + status + "]  " + juce::String(pointCount)
            + " pt" + (pointCount == 1 ? "" : "s");

        g.fillAll(rowIsSelected ? juce::Colour::fromRGB(42, 85, 138)
                                : ((rowNumber % 2) == 0 ? juce::Colour::fromRGB(26, 33, 43)
                                                        : juce::Colour::fromRGB(20, 26, 34)));

        g.setColour(rowIsSelected ? juce::Colours::white : juce::Colour::fromRGB(219, 229, 241));
        g.setFont(juce::FontOptions(13.0f, rowIsSelected ? juce::Font::bold : juce::Font::plain));
        g.drawText(text, 10, 0, width - 20, height, juce::Justification::centredLeft, true);

        g.setColour(juce::Colour::fromRGB(48, 60, 76));
        g.fillRect(0, height - 1, width, 1);
    }

    void selectedRowsChanged(int lastRowSelected) override
    {
        owner.handleLaneSelectionChanged(lastRowSelected);
    }

private:
    AutomationEditorComponent& owner;
};

class AutomationEditorComponent::CurveComponent final : public juce::Component
{
public:
    using CommitCallback = std::function<void(const std::vector<AutomationPoint>&, const juce::String&)>;
    using Formatter = std::function<juce::String(double)>;

    explicit CurveComponent(CommitCallback onCommitIn)
        : onCommit(std::move(onCommitIn))
    {
        setWantsKeyboardFocus(true);
        setMouseClickGrabsKeyboardFocus(true);
    }

    void clearLane(int playheadTickIn, int loopStartTickIn, int loopEndTickIn, int maxTickIn)
    {
        hasLane = false;
        laneLabel = "Automation";
        points.clear();
        selectedPointIndex = -1;
        hoverPointIndex = -1;
        draggingPoint = false;
        minimum = 0.0;
        maximum = 1.0;
        defaultValue = 0.0;
        formatter = {};
        updateTimeline(playheadTickIn, loopStartTickIn, loopEndTickIn, maxTickIn);
    }

    void setLane(const juce::String& laneLabelIn,
                 double minimumIn,
                 double maximumIn,
                 double defaultValueIn,
                 int maxTickIn,
                 int playheadTickIn,
                 int loopStartTickIn,
                 int loopEndTickIn,
                 std::vector<AutomationPoint> pointsIn,
                 Formatter formatterIn)
    {
        hasLane = true;
        laneLabel = laneLabelIn.isNotEmpty() ? laneLabelIn : "Automation";
        minimum = minimumIn;
        maximum = maximumIn > minimumIn ? maximumIn : minimumIn + 1.0;
        defaultValue = juce::jlimit(minimum, maximum, defaultValueIn);
        formatter = std::move(formatterIn);
        points = std::move(pointsIn);
        normalisePoints(points);
        selectedPointIndex = juce::jlimit(-1, static_cast<int>(points.size()) - 1, selectedPointIndex);
        hoverPointIndex = -1;
        draggingPoint = false;
        updateTimeline(playheadTickIn, loopStartTickIn, loopEndTickIn, maxTickIn);
    }

    void updateTimeline(int playheadTickIn, int loopStartTickIn, int loopEndTickIn, int maxTickIn)
    {
        playheadTick = juce::jmax(0, playheadTickIn);
        loopStartTick = juce::jmax(0, loopStartTickIn);
        loopEndTick = juce::jmax(loopStartTick + 1, loopEndTickIn);
        maxTick = juce::jmax(kTicksPerBar, maxTickIn);
        repaint();
    }

    void paint(juce::Graphics& g) override
    {
        g.fillAll(juce::Colour::fromRGB(12, 16, 22));

        auto rect = getContentBounds();
        if (rect.getWidth() < 10.0f || rect.getHeight() < 10.0f)
            return;

        g.setColour(juce::Colour::fromRGB(24, 33, 44));
        g.fillRoundedRectangle(rect, 8.0f);
        g.setColour(juce::Colour::fromRGB(45, 59, 77));
        g.drawRoundedRectangle(rect, 8.0f, 1.0f);

        const auto loopLeft = tickToX(loopStartTick);
        const auto loopRight = tickToX(loopEndTick);
        if (loopRight > loopLeft)
        {
            g.setColour(juce::Colour::fromRGBA(64, 124, 191, 36));
            g.fillRect(juce::Rectangle<float>(loopLeft, rect.getY(), loopRight - loopLeft, rect.getHeight()));
        }

        for (int tick = 0; tick <= maxTick; tick += kTicksPerBeat)
        {
            const auto x = tickToX(tick);
            g.setColour((tick % kTicksPerBar) == 0 ? juce::Colour::fromRGB(49, 65, 84)
                                                   : juce::Colour::fromRGB(33, 45, 60));
            g.drawVerticalLine(juce::roundToInt(x), rect.getY(), rect.getBottom());
        }

        constexpr int valueSteps = 4;
        for (int step = 0; step <= valueSteps; ++step)
        {
            const auto progress = static_cast<float>(step) / static_cast<float>(valueSteps);
            const auto y = rect.getY() + rect.getHeight() * progress;
            g.setColour(juce::Colour::fromRGB(33, 45, 60));
            g.drawHorizontalLine(juce::roundToInt(y), rect.getX(), rect.getRight());

            const auto value = maximum - (static_cast<double>(progress) * (maximum - minimum));
            g.setColour(juce::Colour::fromRGB(154, 172, 194));
            g.drawText(formatValue(value),
                       juce::Rectangle<int>(6, juce::roundToInt(y) - 10, 42, 20),
                       juce::Justification::centredRight,
                       false);
        }

        const auto defaultY = valueToY(defaultValue);
        juce::Path defaultPath;
        defaultPath.startNewSubPath(rect.getX(), defaultY);
        defaultPath.lineTo(rect.getRight(), defaultY);
        g.setColour(juce::Colour::fromRGB(100, 116, 139));
        g.strokePath(defaultPath,
                     juce::PathStrokeType(1.0f, juce::PathStrokeType::beveled, juce::PathStrokeType::rounded));

        juce::Path automationPath;
        const auto sortedPoints = getSortedPoints();
        if (!sortedPoints.empty())
        {
            const auto firstPoint = sortedPoints.front();
            if (firstPoint.tick > 0)
            {
                automationPath.startNewSubPath(rect.getX(), defaultY);
                automationPath.lineTo(tickToX(firstPoint.tick), defaultY);
            }
            else
            {
                automationPath.startNewSubPath(tickToX(firstPoint.tick), valueToY(firstPoint.value));
            }

            for (const auto& point : sortedPoints)
                automationPath.lineTo(tickToX(point.tick), valueToY(point.value));

            automationPath.lineTo(rect.getRight(), valueToY(sortedPoints.back().value));
        }
        else
        {
            automationPath.startNewSubPath(rect.getX(), defaultY);
            automationPath.lineTo(rect.getRight(), defaultY);
        }

        g.setColour(juce::Colour::fromRGB(102, 195, 255));
        g.strokePath(automationPath, juce::PathStrokeType(2.0f));

        for (int index = 0; index < static_cast<int>(points.size()); ++index)
        {
            const auto& point = points[static_cast<size_t>(index)];
            const auto pointBounds = juce::Rectangle<float>(tickToX(point.tick) - 6.0f,
                                                            valueToY(point.value) - 6.0f,
                                                            12.0f,
                                                            12.0f);

            if (index == selectedPointIndex)
            {
                g.setColour(juce::Colours::white);
                g.fillEllipse(pointBounds);
                g.setColour(juce::Colour::fromRGB(102, 195, 255));
                g.drawEllipse(pointBounds, 2.0f);
            }
            else if (index == hoverPointIndex)
            {
                g.setColour(juce::Colour::fromRGB(172, 222, 255));
                g.fillEllipse(pointBounds);
                g.setColour(juce::Colour::fromRGB(216, 240, 255));
                g.drawEllipse(pointBounds, 1.5f);
            }
            else
            {
                g.setColour(juce::Colour::fromRGB(102, 195, 255));
                g.fillEllipse(pointBounds);
                g.setColour(juce::Colour::fromRGB(14, 23, 33));
                g.drawEllipse(pointBounds, 1.2f);
            }
        }

        g.setColour(juce::Colour::fromRGB(244, 211, 94));
        g.drawVerticalLine(juce::roundToInt(tickToX(playheadTick)), rect.getY(), rect.getBottom());

        g.setColour(juce::Colour::fromRGB(220, 228, 239));
        g.setFont(juce::FontOptions(14.0f, juce::Font::bold));
        g.drawText(laneLabel,
                   juce::Rectangle<int>(juce::roundToInt(rect.getX()), 4, juce::roundToInt(rect.getWidth()), 18),
                   juce::Justification::centredLeft,
                   true);

        g.setColour(juce::Colour::fromRGB(152, 176, 198));
        g.setFont(juce::FontOptions(12.0f));
        g.drawText(buildFooterText(),
                   juce::Rectangle<int>(juce::roundToInt(rect.getX()),
                                        juce::roundToInt(rect.getBottom()) + 6,
                                        juce::roundToInt(rect.getWidth()),
                                        18),
                   juce::Justification::centredLeft,
                   true);
    }

    void mouseDown(const juce::MouseEvent& event) override
    {
        grabKeyboardFocus();
        const auto pointIndex = findPointAt(event.position);

        if (event.mods.isLeftButtonDown())
        {
            selectedPointIndex = pointIndex;
            draggingPoint = pointIndex >= 0;
            dragStartPoints = points;
            repaint();
            return;
        }

        if (event.mods.isRightButtonDown() && pointIndex >= 0)
        {
            points.erase(points.begin() + pointIndex);
            selectedPointIndex = juce::jmin(pointIndex, static_cast<int>(points.size()) - 1);
            commit("Remove Automation Point");
        }
    }

    void mouseDrag(const juce::MouseEvent& event) override
    {
        if (!draggingPoint || !juce::isPositiveAndBelow(selectedPointIndex, static_cast<int>(points.size())))
            return;

        points[static_cast<size_t>(selectedPointIndex)].tick = xToTick(event.position.x);
        points[static_cast<size_t>(selectedPointIndex)].value = yToValue(event.position.y);
        normalisePoints(points);

        const auto updatedTick = xToTick(event.position.x);
        selectedPointIndex = -1;
        for (int index = 0; index < static_cast<int>(points.size()); ++index)
        {
            if (points[static_cast<size_t>(index)].tick == updatedTick)
            {
                selectedPointIndex = index;
                break;
            }
        }

        repaint();
    }

    void mouseUp(const juce::MouseEvent& event) override
    {
        juce::ignoreUnused(event);

        if (!draggingPoint)
            return;

        draggingPoint = false;
        if (!automationPointsEqual(points, dragStartPoints))
            commit("Move Automation Point");
        else
            repaint();
    }

    void mouseDoubleClick(const juce::MouseEvent& event) override
    {
        if (!event.mods.isLeftButtonDown() || !hasLane)
            return;

        AutomationPoint point;
        point.tick = xToTick(event.position.x);
        point.value = yToValue(event.position.y);
        points.push_back(point);
        normalisePoints(points);

        selectedPointIndex = -1;
        for (int index = 0; index < static_cast<int>(points.size()); ++index)
        {
            if (points[static_cast<size_t>(index)].tick == point.tick)
            {
                selectedPointIndex = index;
                break;
            }
        }

        commit("Add Automation Point");
    }

    void mouseMove(const juce::MouseEvent& event) override
    {
        const auto pointIndex = findPointAt(event.position);
        if (pointIndex != hoverPointIndex)
        {
            hoverPointIndex = pointIndex;
            repaint();
        }
    }

    void mouseExit(const juce::MouseEvent& event) override
    {
        juce::ignoreUnused(event);
        hoverPointIndex = -1;
        repaint();
    }

    bool keyPressed(const juce::KeyPress& key) override
    {
        if (key.getKeyCode() != juce::KeyPress::deleteKey
            && key.getKeyCode() != juce::KeyPress::backspaceKey)
            return false;

        if (!juce::isPositiveAndBelow(selectedPointIndex, static_cast<int>(points.size())))
            return false;

        points.erase(points.begin() + selectedPointIndex);
        selectedPointIndex = juce::jmin(selectedPointIndex, static_cast<int>(points.size()) - 1);
        commit("Remove Automation Point");
        return true;
    }

private:
    juce::Rectangle<float> getContentBounds() const
    {
        return { 56.0f,
                 20.0f,
                 juce::jmax(1.0f, static_cast<float>(getWidth()) - 74.0f),
                 juce::jmax(1.0f, static_cast<float>(getHeight()) - 48.0f) };
    }

    float tickToX(int tick) const
    {
        const auto rect = getContentBounds();
        const auto normalised = juce::jlimit(0.0f,
                                             1.0f,
                                             static_cast<float>(juce::jmax(0, tick))
                                                 / static_cast<float>(juce::jmax(1, maxTick)));
        return rect.getX() + rect.getWidth() * normalised;
    }

    int xToTick(float x) const
    {
        const auto rect = getContentBounds();
        if (rect.getWidth() <= 1.0f)
            return 0;

        const auto normalised = juce::jlimit(0.0f, 1.0f, (x - rect.getX()) / rect.getWidth());
        return juce::jlimit(0, maxTick, juce::roundToInt(normalised * static_cast<float>(maxTick)));
    }

    float valueToY(double value) const
    {
        const auto rect = getContentBounds();
        const auto span = juce::jmax(1.0e-9, maximum - minimum);
        const auto normalised = static_cast<float>((juce::jlimit(minimum, maximum, value) - minimum) / span);
        return rect.getBottom() - rect.getHeight() * normalised;
    }

    double yToValue(float y) const
    {
        const auto rect = getContentBounds();
        if (rect.getHeight() <= 1.0f)
            return defaultValue;

        const auto normalised = juce::jlimit(0.0f, 1.0f, (rect.getBottom() - y) / rect.getHeight());
        return juce::jlimit(minimum, maximum, minimum + (maximum - minimum) * static_cast<double>(normalised));
    }

    int findPointAt(juce::Point<float> position) const
    {
        constexpr float threshold = 11.0f;
        for (int index = 0; index < static_cast<int>(points.size()); ++index)
        {
            const auto& point = points[static_cast<size_t>(index)];
            const juce::Point<float> pointPosition(tickToX(point.tick), valueToY(point.value));
            if (pointPosition.getDistanceFrom(position) <= threshold)
                return index;
        }

        return -1;
    }

    std::vector<AutomationPoint> getSortedPoints() const
    {
        auto sorted = points;
        std::sort(sorted.begin(), sorted.end(), [] (const AutomationPoint& lhs, const AutomationPoint& rhs)
        {
            if (lhs.tick != rhs.tick)
                return lhs.tick < rhs.tick;
            return lhs.value < rhs.value;
        });
        return sorted;
    }

    void normalisePoints(std::vector<AutomationPoint>& pointsToNormalise) const
    {
        std::sort(pointsToNormalise.begin(), pointsToNormalise.end(), [] (const AutomationPoint& lhs, const AutomationPoint& rhs)
        {
            if (lhs.tick != rhs.tick)
                return lhs.tick < rhs.tick;
            return lhs.value < rhs.value;
        });

        std::vector<AutomationPoint> deduped;
        deduped.reserve(pointsToNormalise.size());
        for (const auto& point : pointsToNormalise)
        {
            AutomationPoint normalised;
            normalised.tick = juce::jlimit(0, maxTick, point.tick);
            normalised.value = juce::jlimit(minimum, maximum, point.value);

            if (!deduped.empty() && deduped.back().tick == normalised.tick)
                deduped.back().value = normalised.value;
            else
                deduped.push_back(normalised);
        }

        pointsToNormalise = std::move(deduped);
    }

    juce::String formatValue(double value) const
    {
        if (formatter)
            return formatter(value);
        return juce::String(value, 2);
    }

    juce::String buildFooterText() const
    {
        if (juce::isPositiveAndBelow(selectedPointIndex, static_cast<int>(points.size())))
        {
            const auto& point = points[static_cast<size_t>(selectedPointIndex)];
            return "Tick " + juce::String(point.tick) + "   Value " + formatValue(point.value);
        }

        if (!hasLane)
            return "Choose a lane to edit.";

        return "Double-click to add a point. Drag to reshape. Right-click or Delete to remove.";
    }

    void commit(const juce::String& actionName)
    {
        normalisePoints(points);
        if (onCommit)
            onCommit(points, actionName);
        repaint();
    }

    CommitCallback onCommit;
    Formatter formatter;
    bool hasLane = false;
    juce::String laneLabel = "Automation";
    double minimum = 0.0;
    double maximum = 1.0;
    double defaultValue = 0.0;
    int maxTick = kTicksPerBar;
    int playheadTick = 0;
    int loopStartTick = 0;
    int loopEndTick = kTicksPerBar;
    std::vector<AutomationPoint> points;
    std::vector<AutomationPoint> dragStartPoints;
    int selectedPointIndex = -1;
    int hoverPointIndex = -1;
    bool draggingPoint = false;
};

AutomationEditorComponent::AutomationEditorComponent(ProjectGetter projectGetterIn,
                                                     SelectedTrackIndexGetter selectedTrackIndexGetterIn,
                                                     TrackWriter trackWriterIn)
    : projectGetter(std::move(projectGetterIn)),
      selectedTrackIndexGetter(std::move(selectedTrackIndexGetterIn)),
      trackWriter(std::move(trackWriterIn))
{
    headerLabel.setText("Automation", juce::dontSendNotification);
    headerLabel.setFont(juce::FontOptions(16.0f, juce::Font::bold));
    headerLabel.setColour(juce::Label::textColourId, juce::Colour::fromRGB(233, 238, 245));
    addAndMakeVisible(headerLabel);
    trackSummaryLabel.setColour(juce::Label::textColourId, juce::Colour::fromRGB(214, 223, 235));
    trackSummaryLabel.setFont(juce::FontOptions(13.0f, juce::Font::bold));
    addAndMakeVisible(trackSummaryLabel);

    laneSummaryLabel.setColour(juce::Label::textColourId, juce::Colour::fromRGB(151, 167, 186));
    laneSummaryLabel.setFont(juce::FontOptions(12.0f));
    addAndMakeVisible(laneSummaryLabel);

    targetLabel.setText("Target", juce::dontSendNotification);
    targetLabel.setColour(juce::Label::textColourId, juce::Colour::fromRGB(181, 194, 209));
    addAndMakeVisible(targetLabel);

    addLaneCombo.setTextWhenNoChoicesAvailable("No lanes left to add");
    addLaneCombo.setTextWhenNothingSelected("Choose automation target");
    addAndMakeVisible(addLaneCombo);

    auto configureButton = [this] (juce::TextButton& button, const juce::String& text)
    {
        button.setButtonText(text);
        button.setColour(juce::TextButton::buttonColourId, juce::Colour::fromRGB(44, 52, 64));
        button.setColour(juce::TextButton::textColourOffId, juce::Colours::white);
        addAndMakeVisible(button);
    };

    configureButton(addLaneButton, "Add Lane");
    configureButton(removeLaneButton, "Remove Lane");
    configureButton(resetLaneButton, "Reset Lane");

    enableLaneToggle.setButtonText("Read Enabled");
    addAndMakeVisible(enableLaneToggle);

    laneListModel = std::make_unique<LaneListModel>(*this);
    laneList.setModel(laneListModel.get());
    laneList.setRowHeight(30);
    laneList.setColour(juce::ListBox::backgroundColourId, juce::Colour::fromRGB(24, 33, 44));
    laneList.setColour(juce::ListBox::outlineColourId, juce::Colour::fromRGB(45, 59, 77));
    laneList.setOutlineThickness(1);
    addAndMakeVisible(laneList);

    curveComponent = std::make_unique<CurveComponent>(
        [this] (const std::vector<AutomationPoint>& points, const juce::String& actionName)
        {
            commitCurrentLanePoints(points, actionName);
        });
    addAndMakeVisible(*curveComponent);

    addLaneButton.onClick = [this] { addSelectedLane(); };
    removeLaneButton.onClick = [this] { removeSelectedLane(); };
    enableLaneToggle.onClick = [this] { toggleSelectedLaneEnabled(); };
    resetLaneButton.onClick = [this] { resetSelectedLane(); };
}

AutomationEditorComponent::~AutomationEditorComponent() = default;

void AutomationEditorComponent::paint(juce::Graphics& g)
{
    g.setColour(juce::Colour::fromRGB(16, 20, 26));
    g.fillRoundedRectangle(getLocalBounds().toFloat(), 10.0f);
    g.setColour(juce::Colour::fromRGB(42, 50, 62));
    g.drawRoundedRectangle(getLocalBounds().toFloat().reduced(0.5f), 10.0f, 1.0f);
}

void AutomationEditorComponent::resized()
{
    auto area = getLocalBounds().reduced(10);

    headerLabel.setBounds(area.removeFromTop(22));
    area.removeFromTop(4);
    trackSummaryLabel.setBounds(area.removeFromTop(20));
    area.removeFromTop(2);
    laneSummaryLabel.setBounds(area.removeFromTop(18));
    area.removeFromTop(8);

    auto controls = area.removeFromTop(28);
    targetLabel.setBounds(controls.removeFromLeft(46));
    addLaneCombo.setBounds(controls.removeFromLeft(240));
    controls.removeFromLeft(8);
    addLaneButton.setBounds(controls.removeFromLeft(88));
    controls.removeFromLeft(6);
    removeLaneButton.setBounds(controls.removeFromLeft(110));
    controls.removeFromLeft(6);
    enableLaneToggle.setBounds(controls.removeFromLeft(112));
    controls.removeFromLeft(6);
    resetLaneButton.setBounds(controls.removeFromLeft(92));

    area.removeFromTop(8);
    auto body = area;
    auto laneArea = body.removeFromLeft(280);
    laneList.setBounds(laneArea);
    body.removeFromLeft(10);
    curveComponent->setBounds(body);
}

void AutomationEditorComponent::refreshFromModel()
{
    updatingUi = true;

    hasTrackSnapshot = false;
    currentLanes.clear();

    const auto* track = getSelectedTrack();
    const auto trackIndex = getSelectedTrackIndex();
    if (track == nullptr || trackIndex < 0)
    {
        trackSummaryLabel.setText("No track selected.", juce::dontSendNotification);
        laneSummaryLabel.setText("Choose a track to create or edit automation lanes.", juce::dontSendNotification);
        comboTargets.clear();
        addLaneCombo.clear(juce::dontSendNotification);
        laneList.updateContent();
        laneList.deselectAllRows();
        curveComponent->clearLane(projectGetter().playheadTick,
                                  projectGetter().leftLocatorTick,
                                  projectGetter().rightLocatorTick,
                                  computeCurveMaxTick());
        updateLaneControls();
        updatingUi = false;
        return;
    }

    currentTrackSnapshot = *track;
    sanitiseAutomationLanes(currentTrackSnapshot);
    currentLanes = currentTrackSnapshot.automationLanes;
    hasTrackSnapshot = true;

    trackSummaryLabel.setText(currentTrackSnapshot.name + "  |  "
                              + currentTrackSnapshot.trackType + " track  |  "
                              + juce::String(static_cast<int>(currentLanes.size())) + " lane"
                              + (currentLanes.size() == 1 ? "" : "s"),
                              juce::dontSendNotification);

    populateTargetCombo();
    laneList.updateContent();

    auto selectedTarget = getSelectedTarget();
    int selectedRow = -1;
    if (selectedTarget.isNotEmpty())
    {
        for (int index = 0; index < static_cast<int>(currentLanes.size()); ++index)
        {
            if (currentLanes[static_cast<size_t>(index)].target.equalsIgnoreCase(selectedTarget))
            {
                selectedRow = index;
                break;
            }
        }
    }

    if (selectedRow < 0 && !currentLanes.empty())
    {
        selectedRow = 0;
        setSelectedTarget(currentLanes.front().target);
    }
    else if (selectedRow < 0)
    {
        setSelectedTarget({});
    }

    if (selectedRow >= 0)
        laneList.selectRow(selectedRow);
    else
        laneList.deselectAllRows();

    updateLaneControls();
    syncCurveFromSelection();
    updatingUi = false;
}

void AutomationEditorComponent::refreshViewState()
{
    curveComponent->updateTimeline(projectGetter().playheadTick,
                                   projectGetter().leftLocatorTick,
                                   projectGetter().rightLocatorTick,
                                   computeCurveMaxTick());
}

const TrackState* AutomationEditorComponent::getSelectedTrack() const
{
    const auto index = getSelectedTrackIndex();
    const auto& project = projectGetter();
    if (!juce::isPositiveAndBelow(index, static_cast<int>(project.tracks.size())))
        return nullptr;

    return &project.tracks[static_cast<size_t>(index)];
}

int AutomationEditorComponent::getSelectedTrackIndex() const
{
    return selectedTrackIndexGetter ? selectedTrackIndexGetter() : -1;
}

const AutomationLane* AutomationEditorComponent::getSelectedLane() const
{
    const auto selectedTarget = getSelectedTarget();
    if (selectedTarget.isEmpty())
        return nullptr;

    for (const auto& lane : currentLanes)
    {
        if (lane.target.equalsIgnoreCase(selectedTarget))
            return &lane;
    }

    return nullptr;
}

juce::String AutomationEditorComponent::getSelectedTarget() const
{
    const auto trackIndex = getSelectedTrackIndex();
    if (trackIndex < 0)
        return {};

    if (const auto iterator = selectedTargetByTrack.find(trackIndex); iterator != selectedTargetByTrack.end())
        return iterator->second;

    return {};
}

void AutomationEditorComponent::setSelectedTarget(const juce::String& target)
{
    const auto trackIndex = getSelectedTrackIndex();
    if (trackIndex < 0)
        return;

    const auto trimmed = target.trim();
    if (trimmed.isEmpty())
        selectedTargetByTrack.erase(trackIndex);
    else
        selectedTargetByTrack[trackIndex] = trimmed;
}

void AutomationEditorComponent::populateTargetCombo()
{
    comboTargets.clear();
    addLaneCombo.clear(juce::dontSendNotification);

    if (!hasTrackSnapshot)
        return;

    juce::StringArray existingTargets;
    for (const auto& lane : currentLanes)
        existingTargets.add(lane.target);

    const auto availableTargets = availableAutomationTargets(currentTrackSnapshot);
    int comboId = 1;
    for (const auto& target : availableTargets)
    {
        if (existingTargets.contains(target, true))
            continue;

        comboTargets.push_back(target);
        addLaneCombo.addItem(automationTargetLabel(currentTrackSnapshot, target), comboId++);
    }

    if (!comboTargets.empty())
        addLaneCombo.setSelectedId(1, juce::dontSendNotification);
}

void AutomationEditorComponent::updateLaneControls()
{
    const auto* lane = getSelectedLane();
    const auto hasTrack = hasTrackSnapshot;
    const auto hasLane = lane != nullptr;

    addLaneCombo.setEnabled(hasTrack && !comboTargets.empty());
    addLaneButton.setEnabled(hasTrack && !comboTargets.empty());
    removeLaneButton.setEnabled(hasLane);
    enableLaneToggle.setEnabled(hasLane);
    resetLaneButton.setEnabled(hasLane);
    enableLaneToggle.setToggleState(hasLane ? lane->enabled : false, juce::dontSendNotification);

    if (!hasTrack)
    {
        laneSummaryLabel.setText("Choose a track to create or edit automation lanes.", juce::dontSendNotification);
        return;
    }

    if (!hasLane)
    {
        laneSummaryLabel.setText("Add a lane to start automating volume, pan, output gain, or VST parameters.",
                                 juce::dontSendNotification);
        return;
    }

    laneSummaryLabel.setText(automationTargetLabel(currentTrackSnapshot, lane->target)
                             + " automation. Double-click to add points, drag to reshape, and use Read Enabled to bypass the lane.",
                             juce::dontSendNotification);
}

void AutomationEditorComponent::syncCurveFromSelection()
{
    const auto* lane = getSelectedLane();
    if (!hasTrackSnapshot || lane == nullptr)
    {
        curveComponent->clearLane(projectGetter().playheadTick,
                                  projectGetter().leftLocatorTick,
                                  projectGetter().rightLocatorTick,
                                  computeCurveMaxTick());
        return;
    }

    const auto [minimum, maximum] = automationTargetBounds(currentTrackSnapshot, lane->target);
    const auto defaultValue = automationTargetDefaultValue(currentTrackSnapshot, lane->target);
    const auto laneTarget = lane->target;
    const auto trackSnapshot = currentTrackSnapshot;

    curveComponent->setLane(automationTargetLabel(currentTrackSnapshot, laneTarget),
                            minimum,
                            maximum,
                            defaultValue,
                            computeCurveMaxTick(),
                            projectGetter().playheadTick,
                            projectGetter().leftLocatorTick,
                            projectGetter().rightLocatorTick,
                            lane->points,
                            [trackSnapshot, laneTarget] (double value)
                            {
                                return formatAutomationValue(trackSnapshot, laneTarget, value);
                            });
}

int AutomationEditorComponent::computeCurveMaxTick() const
{
    const auto& project = projectGetter();
    int maxTick = juce::jmax(kTicksPerBar, project.rightLocatorTick);

    if (hasTrackSnapshot)
    {
        for (const auto& note : currentTrackSnapshot.notes)
            maxTick = juce::jmax(maxTick, note.startTick + note.durationTick + kTicksPerBar);
    }

    if (const auto* lane = getSelectedLane())
    {
        for (const auto& point : lane->points)
            maxTick = juce::jmax(maxTick, point.tick + kTicksPerBar);
    }

    return maxTick;
}

void AutomationEditorComponent::handleLaneSelectionChanged(int row)
{
    if (updatingUi)
        return;

    if (juce::isPositiveAndBelow(row, static_cast<int>(currentLanes.size())))
        setSelectedTarget(currentLanes[static_cast<size_t>(row)].target);
    else
        setSelectedTarget({});

    updateLaneControls();
    syncCurveFromSelection();
}

void AutomationEditorComponent::addSelectedLane()
{
    const auto trackIndex = getSelectedTrackIndex();
    const auto* liveTrack = getSelectedTrack();
    if (trackIndex < 0 || liveTrack == nullptr)
        return;

    const auto selectedId = addLaneCombo.getSelectedId();
    if (!juce::isPositiveAndBelow(selectedId - 1, static_cast<int>(comboTargets.size())))
        return;

    auto updatedTrack = *liveTrack;
    sanitiseAutomationLanes(updatedTrack);

    const auto target = comboTargets[static_cast<size_t>(selectedId - 1)];
    const auto defaultValue = automationTargetDefaultValue(updatedTrack, target);
    const auto newTick = juce::jmax(0, projectGetter().playheadTick);

    AutomationLane lane;
    lane.target = target;
    lane.enabled = true;
    lane.points.push_back({ newTick, defaultValue });
    updatedTrack.automationLanes.push_back(std::move(lane));
    sanitiseAutomationLanes(updatedTrack);

    setSelectedTarget(target);
    trackWriter(trackIndex, updatedTrack, true, "Add Automation Lane");
}

void AutomationEditorComponent::removeSelectedLane()
{
    const auto trackIndex = getSelectedTrackIndex();
    const auto* liveTrack = getSelectedTrack();
    const auto target = getSelectedTarget();
    if (trackIndex < 0 || liveTrack == nullptr || target.isEmpty())
        return;

    auto updatedTrack = *liveTrack;
    sanitiseAutomationLanes(updatedTrack);
    updatedTrack.automationLanes.erase(std::remove_if(updatedTrack.automationLanes.begin(),
                                                      updatedTrack.automationLanes.end(),
                                                      [&target] (const AutomationLane& lane)
                                                      {
                                                          return lane.target.equalsIgnoreCase(target);
                                                      }),
                                       updatedTrack.automationLanes.end());

    setSelectedTarget({});
    trackWriter(trackIndex, updatedTrack, true, "Remove Automation Lane");
}

void AutomationEditorComponent::toggleSelectedLaneEnabled()
{
    if (updatingUi)
        return;

    const auto trackIndex = getSelectedTrackIndex();
    const auto* liveTrack = getSelectedTrack();
    const auto target = getSelectedTarget();
    if (trackIndex < 0 || liveTrack == nullptr || target.isEmpty())
        return;

    auto updatedTrack = *liveTrack;
    sanitiseAutomationLanes(updatedTrack);
    for (auto& lane : updatedTrack.automationLanes)
    {
        if (lane.target.equalsIgnoreCase(target))
        {
            lane.enabled = enableLaneToggle.getToggleState();
            trackWriter(trackIndex, updatedTrack, true, "Toggle Automation Lane");
            return;
        }
    }
}

void AutomationEditorComponent::resetSelectedLane()
{
    const auto trackIndex = getSelectedTrackIndex();
    const auto* liveTrack = getSelectedTrack();
    const auto target = getSelectedTarget();
    if (trackIndex < 0 || liveTrack == nullptr || target.isEmpty())
        return;

    auto updatedTrack = *liveTrack;
    sanitiseAutomationLanes(updatedTrack);
    const auto defaultValue = automationTargetDefaultValue(updatedTrack, target);
    for (auto& lane : updatedTrack.automationLanes)
    {
        if (lane.target.equalsIgnoreCase(target))
        {
            lane.points = { { 0, defaultValue } };
            trackWriter(trackIndex, updatedTrack, true, "Reset Automation Lane");
            return;
        }
    }
}

void AutomationEditorComponent::commitCurrentLanePoints(const std::vector<AutomationPoint>& points,
                                                        const juce::String& actionName)
{
    const auto trackIndex = getSelectedTrackIndex();
    const auto* liveTrack = getSelectedTrack();
    const auto target = getSelectedTarget();
    if (trackIndex < 0 || liveTrack == nullptr || target.isEmpty())
        return;

    auto updatedTrack = *liveTrack;
    sanitiseAutomationLanes(updatedTrack);
    for (auto& lane : updatedTrack.automationLanes)
    {
        if (lane.target.equalsIgnoreCase(target))
        {
            lane.points = points;
            trackWriter(trackIndex, updatedTrack, true, actionName);
            return;
        }
    }
}

} // namespace aims
